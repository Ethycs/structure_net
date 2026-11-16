"""Actual packed suffix inference, standalone reload and prefix offload semantics."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net import jspace_3b_packed_response_model as study
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack


def parts(tmp_path, cut):
    torch.manual_seed(12781)
    cfg = Qwen2Config(hidden_size=8, intermediate_size=12, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=32,
        max_position_embeddings=32, tie_word_embeddings=True, attention_dropout=0.,
        pad_token_id=0, eos_token_id=31)
    cfg._attn_implementation = "eager"
    native = Qwen2ForCausalLM(cfg).bfloat16().eval().requires_grad_(False)
    path = tmp_path/"native"; native.save_pretrained(path)
    blocks = []
    for index in range(cut, 4):
        basis = torch.linalg.qr(torch.randn(8, 8)).Q[:, :4]
        block = AffineQwenBlock(native.model.layers[index], basis, torch.randn(12, 8)/20,
            trainable=True, parameter_dtype=torch.bfloat16)
        block.compile_polynomial(dict(anchors=[0, 1], mean=torch.zeros(12), scale=torch.ones(12),
            coefficients={"2":torch.randn(12, 6)/100}), [10, 11])
        for p in block.parameters(): p.data = p.data.bfloat16()
        block.qkv_offset.data.add_(.01)
        blocks.append(block)
    stack = CompactQwenStack(blocks, geometry_mode="bf16", pad_to_multiple=16)
    packed = study.source.builder.packed_clone(stack, device="cpu")
    norm = deepcopy(native.model.norm)
    norm.weight.data.mul_(1.125)
    model = study.PackedResponseModel(packed, norm, native.lm_head.weight.detach(),
        study.source.partial.training_rotary(cfg, "cpu"), config=cfg, max_length=12)
    return path, native, model


@pytest.mark.parametrize("cut", [0, 2])
def test_standalone_packed_export_reload_is_exact_without_native_or_recompilation(tmp_path, cut, monkeypatch):
    path, native, model = parts(tmp_path, cut)
    ids = torch.tensor([[3, 5, 7, 11]])
    batch = {"input_ids":ids}
    if cut:
        with study.attach_prefix(model, path) as prefix:
            batch["boundary_hidden"] = model.native_prefix(ids, torch.ones_like(ids, dtype=torch.bool))
    with torch.no_grad():
        expected = model.readout(model.hidden_forward(batch)[:, -1])
    before = study.packed_state_digest(model)
    receipt = study.export_packed_snapshot(model, tmp_path/"export", identities={"source":"fixture"})
    assert receipt["packed"] and receipt["embedding_copies"] == 1
    assert receipt["packed_state_sha256"] == before["packed_state_sha256"]
    assert receipt["serialized_bytes"] > receipt["tensor_bytes"]
    monkeypatch.setattr(study.source, "build_student", lambda *a, **k: pytest.fail("Reload recompiled native weights"))
    monkeypatch.setattr(study.source.builder.native, "_config", lambda *a, **k: pytest.fail("Reload consulted native files"))
    loaded, proof = study.load_packed_snapshot(tmp_path/"export")
    assert loaded is not model and loaded.stack is not model.stack
    assert proof["strict_state_load"] and proof["packed_state_exact"]
    assert loaded.stack.packed and loaded.stack.stripped and loaded.cut == cut
    assert study.packed_state_digest(loaded) == before
    with torch.no_grad():
        actual = loaded.readout(loaded.hidden_forward(batch)[:, -1])
    assert torch.equal(actual, expected)
    assert not any(p.requires_grad for p in loaded.parameters())


def test_next_logits_projects_only_last_row(tmp_path, monkeypatch):
    _, _, model = parts(tmp_path, 0)
    shapes = []
    actual = model.readout
    def spy(hidden):
        shapes.append(tuple(hidden.shape))
        return actual(hidden)
    monkeypatch.setattr(model, "readout", spy)
    logits = model.next_logits(torch.tensor([[1, 3, 7, 2, 5]]))
    assert logits.shape == (32,) and shapes == [(1, 8)]
    assert model.inference_backend == "packed_bf16_torch" and not model.stack.enabled


def test_cpu_offloaded_prefix_matches_actual_native_boundary_and_resident_prefix(tmp_path):
    path, native, model = parts(tmp_path, 2)
    ids = torch.tensor([[3, 7, 11, 4]])
    outputs = []
    for mode in ("resident", "cpu"):
        with study.attach_prefix(model, path, mode=mode) as receipt:
            outputs.append(model.native_prefix(ids, torch.ones_like(ids, dtype=torch.bool)))
            assert receipt["calls"] == 1 and receipt["prefix_seconds"] > 0
            assert receipt["shared_embedding"] and receipt["setup_seconds"] > 0
        assert model.native_prefix is None
    saved = []
    handle = native.model.layers[2].register_forward_pre_hook(lambda _m, args: saved.append(args[0]))
    with torch.no_grad(): native(input_ids=ids, use_cache=False)
    handle.remove()
    assert torch.equal(outputs[0], outputs[1]) and torch.equal(outputs[0], saved[0])


def test_export_tampering_and_unpacked_input_are_rejected(tmp_path):
    _, _, model = parts(tmp_path, 0)
    study.export_packed_snapshot(model, tmp_path/"export", identities={"source":"fixture"})
    path = tmp_path/"export/structure.json"
    description = json.loads(path.read_text()); description["cut"] = 3
    path.write_text(json.dumps(description))
    with pytest.raises(ValueError, match="snapshot changed"):
        study.load_packed_snapshot(tmp_path/"export")
    model.stack.stripped = False
    with pytest.raises(ValueError, match="packed/stripped"):
        study.PackedResponseModel(model.stack, model.norm, model.embedding, model.rotary,
                                 config=model.config, max_length=12)


def test_mc_quality_calls_packed_stack_and_never_unpacked_scorer(tmp_path, monkeypatch):
    _, _, model = parts(tmp_path, 0)
    cases = [dict(identity=str(i), family=f"family{i//64}", split="development",
                  choices=["first", "second"], answer=0) for i in range(256)]
    native = [dict(identity=c["identity"], family=c["family"], answer=0,
                   prediction=0 if i < 195 else 1, logits=[1., 0.] if i < 195 else [0., 1.])
              for i, c in enumerate(cases)]
    ids = torch.tensor([[1, 3, 7]])
    positions = torch.arange(3)[None]
    mask = torch.full((1, 1, 3, 3), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16).triu(1)
    closed = []
    def batch(indices, **kwargs):
        return dict(hidden=torch.nn.functional.embedding(ids, model.embedding),
                    position_ids=positions, attention_mask=mask)
    dataset = SimpleNamespace(items=cases, development=list(range(256)), batch=batch,
                              close=lambda: closed.append(True))
    p = dict(schema=study.source.joint.SCHEMA, native_correct=195, native_records=native, label_ids=[1, 2])
    p["fingerprint"] = study.source.progressive.base.digest(p)
    monkeypatch.setattr(study.source.joint, "dataset", lambda _: dataset)
    monkeypatch.setattr(study.source.joint.training, "score_unpacked",
                        lambda *a, **k: pytest.fail("Quality silently used unpacked execution"))
    calls = []
    handle = model.stack.register_forward_pre_hook(lambda m, args: calls.append(m.packed))
    try:
        result = study.packed_old_mc_evaluate("all36", p, model)
    finally:
        handle.remove()
    assert calls == [True]*256 and closed == [True]
    assert result["packed"] and result["metrics"]["count"] == 256
    assert result["native_count"] == 256 and result["native_correct"] == 195
