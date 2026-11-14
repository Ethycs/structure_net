from copy import deepcopy
import weakref

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_quadratic_qwen import FoldedReadout, QuadraticQwenBlock, QuadraticQwenStack
from experiments.structure_net.jspace_quadratic_evaluate import paired_summary, stream_score


def fixture():
    torch.manual_seed(777)
    config = Qwen2Config(hidden_size=16, intermediate_size=28, num_attention_heads=4, num_key_value_heads=2,
        num_hidden_layers=2, max_position_embeddings=32, vocab_size=23, tie_word_embeddings=True, attention_dropout=0.)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).float().eval().requires_grad_(False)
    cases = [dict(identity=f"case_{i}", family=f"family_{i%2}", answer=i%2,
                  choices=["a", "b"] if i%2 else ["a", "b", "c"], input_ids=tokens)
             for i, tokens in enumerate(([1, 2, 3], [4, 5, 6, 7, 8], [2], [3, 1, 4, 9]))]
    return model, cases, [9, 10, 11]


def assert_records(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        for key in ("identity", "family", "answer", "prediction"):
            assert a[key] == b[key]
        np.testing.assert_allclose(a["logits"], b["logits"], atol=2e-6, rtol=2e-6)


@torch.inference_mode()
def native_reference(model, cases, labels):
    rows = []
    for case in cases:
        hidden = model.model(input_ids=torch.tensor([case["input_ids"]]), use_cache=False).last_hidden_state[:, -1]
        values = torch.nn.functional.linear(hidden, model.lm_head.weight[labels[:len(case["choices"])]])[0]
        rows.append(dict(identity=case["identity"], family=case["family"], answer=case["answer"],
                         prediction=int(values.argmax()), logits=values.tolist()))
    return rows


def test_native_stream_matches_direct_variable_length_causal_batches_and_releases_blocks(tmp_path):
    model, cases, labels = fixture()
    original = deepcopy(cases)
    references, loaded = [], []
    def loader(layer):
        assert all(ref() is None for ref in references)
        block = deepcopy(model.model.layers[layer])
        references.extend(weakref.ref(v) for v in (block, *block.parameters(), *block.buffers()))
        loaded.append(layer)
        return block
    events, guards = [], []
    result = stream_score(cases, input_embedding=model.model.embed_tokens.weight, block_loader=loader,
        final_readout=(model.model.norm, model.lm_head.weight), label_ids=labels, rotary=model.model.rotary_emb,
        layer_count=2, scratch_dir=tmp_path/"banks", microbatch=3,
        progress=lambda **kw: events.append(kw), guard=lambda: guards.append(True))
    assert_records(result["records"], native_reference(model, cases, labels))
    assert cases == original and loaded == [0, 1] and all(ref() is None for ref in references)
    assert not (tmp_path/"banks").exists() and result["metrics"]["scratch_banks_removed"]
    total = sum(len(c["input_ids"]) for c in cases)
    assert result["metrics"]["layers"][0]["output_shape"] == [total, 16]
    assert result["metrics"]["layers"][1]["output_shape"] == [len(cases), 16]
    assert result["metrics"]["logical_write_bytes"] == (2*total+len(cases))*16*4
    assert result["metrics"]["logical_read_bytes"] == (2*total+len(cases))*16*4
    assert len([e for e in events if e["kind"] == "layer"]) == 2 and len(guards) > 4


@pytest.mark.parametrize("mode", ["quadratic", "softmax"])
def test_compact_stream_bridges_and_folded_readout_match_direct_coordinates(tmp_path, mode):
    native, cases, labels = fixture()
    blocks = []
    for layer, original in enumerate(native.model.layers):
        basis = torch.linalg.qr(torch.randn(16, 16)).Q[:, :8]
        affine = AffineQwenBlock(deepcopy(original), basis, torch.randn(1, 16)/10, parameter_dtype=torch.float32,
            head_projection={"rope_pairs": torch.tensor([[0], [1]]), "value_basis": torch.eye(4).repeat(2, 1, 1)[:, :, :2]})
        block = QuadraticQwenBlock(affine, mode=mode, learn_metric=True)
        with torch.no_grad():
            block.metric.log_metric.fill_(.05*(layer+1))
        blocks.append(block.pack(dtype=torch.float32, buffer_dtype=torch.float32))
    stack = QuadraticQwenStack(deepcopy(blocks), pad_to_multiple=0).eval()
    embedding = native.model.embed_tokens.weight@blocks[0].block.basis
    head = FoldedReadout(native.lm_head.weight, blocks[-1].block.basis, blocks[-1].block.anchor,
                         native.model.norm.weight, native.model.norm.variance_epsilon).eval()
    def bridge(layer):
        value = stack.original.bridges[layer]
        return value.weight, value.shift
    result = stream_score(cases, input_embedding=embedding, block_loader=lambda i: deepcopy(blocks[i]),
        final_readout=head, label_ids=labels, rotary=native.model.rotary_emb, bridge_loader=bridge,
        layer_count=2, scratch_dir=tmp_path/"compact", microbatch=3)
    expected = []
    with torch.inference_mode():
        for case in cases:
            z = embedding[torch.tensor([case["input_ids"]])]
            positions = torch.arange(z.shape[1])[None]
            output = stack.forward_coordinates(z, position_ids=positions,
                                               position_embeddings=native.model.rotary_emb(z, positions))
            values = head(output[:, -1])[0, labels[:len(case["choices"])]]
            expected.append(dict(identity=case["identity"], family=case["family"], answer=case["answer"],
                                 prediction=int(values.argmax()), logits=values.tolist()))
    assert_records(result["records"], expected)
    assert [r["bridge"] for r in result["metrics"]["layers"]] == [True, False]
    assert result["metrics"]["layers"][0]["input_shape"][1] == 8
    assert not (tmp_path/"compact").exists()


def test_case_isolation_order_callback_readout_and_microbatch_independence(tmp_path):
    model, cases, labels = fixture()
    changed = deepcopy(cases); changed[1]["input_ids"] = [19, 20]
    def run(items, size, name):
        return stream_score(items, input_embedding=model.model.embed_tokens.weight,
            block_loader=lambda i: deepcopy(model.model.layers[i]), final_readout=lambda h, ids:
                torch.nn.functional.linear(model.model.norm(h), model.lm_head.weight[ids]),
            label_ids=labels, position_embeddings=model.model.rotary_emb, layer_count=2,
            scratch_dir=tmp_path/name, microbatch=size)["records"]
    original, modified = run(cases, 1, "one"), run(changed, 4, "four")
    assert_records([original[i] for i in (0, 2, 3)], [modified[i] for i in (0, 2, 3)])
    assert original[1]["logits"] != modified[1]["logits"]
    assert [r["identity"] for r in modified] == [c["identity"] for c in cases]


def test_scratch_cleanup_on_failure_and_existing_data_is_never_overwritten(tmp_path):
    model, cases, labels = fixture()
    def broken(layer):
        if layer == 1:
            raise RuntimeError("intentional loader failure")
        return deepcopy(model.model.layers[layer])
    kwargs = dict(input_embedding=model.model.embed_tokens.weight, block_loader=broken,
                  final_readout=(model.model.norm, model.lm_head.weight), label_ids=labels,
                  rotary=model.model.rotary_emb, layer_count=2)
    with pytest.raises(RuntimeError, match="intentional"):
        stream_score(cases, scratch_dir=tmp_path/"failed", **kwargs)
    assert not (tmp_path/"failed").exists()
    existing = tmp_path/"existing"; existing.mkdir(); (existing/"keep.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        stream_score(cases, scratch_dir=existing, **kwargs)
    assert (existing/"keep.txt").read_text() == "mine"
    cases[0]["input_ids"] = [-1]
    with pytest.raises(ValueError, match="token IDs"):
        stream_score(cases, scratch_dir=tmp_path/"invalid", **kwargs)
    assert not (tmp_path/"invalid").exists()


def test_paired_summary_has_aggregate_five_point_gate_only():
    native = [dict(identity=str(i), family="one" if i == 0 else "other", answer=0, prediction=0, logits=[1., 0.])
              for i in range(20)]
    compact = deepcopy(native); compact[0]["prediction"] = 1
    result = paired_summary(native, compact)
    assert result["aggregate_quality_pass"] and result["family_quality_is_gate"] is False
    assert result["quality"]["by_family"]["one"]["net_lost_correct"] == 1
    compact[1]["prediction"] = 1
    assert not paired_summary(native, compact)["aggregate_quality_pass"]
