"""Packing checks only; recovery training and source artifacts are untouched."""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding

from experiments.structure_net import jspace_joint_recovery as joint
from experiments.structure_net import jspace_recovered_joint_pack as packed
from experiments.structure_net import jspace_semantic_recovery as semantic
from experiments.structure_net.jspace_single_block_distill import stream_score


def fixture(tmp_path):
    torch.manual_seed(819)
    config = Qwen2Config(hidden_size=16, intermediate_size=24, num_attention_heads=4,
        num_key_value_heads=2, num_hidden_layers=2, vocab_size=23,
        max_position_embeddings=32, attention_dropout=0.)
    config._attn_implementation = "eager"
    native, recovered, fits = [], [], {}
    selected = tmp_path/"selected"; selected.mkdir()
    hashes = {}
    for layer in range(2):
        original = Qwen2DecoderLayer(config, layer_idx=layer).bfloat16().float()
        fit = dict(basis=torch.linalg.qr(torch.randn(16, 8)).Q, means=torch.randn(1, 16)/5)
        block = semantic.build_enriched_block(deepcopy(original), fit, parameter_dtype=torch.float32)
        initializer = tmp_path/f"original_selected_{layer:02d}.pt"
        joint.recipe.save(initializer, dict(block=joint._cpu(block.state_dict())))
        with torch.no_grad():
            # Distinct recovered values make silently recompiling native weights observable.
            for i, parameter in enumerate(block.parameters()): parameter.add_((layer+1)*(i+1)*.003)
        fit_path = tmp_path/("block_zero_original_fit.pt" if layer == 0 else "other_all_layer_fit.pt")
        joint.recipe.save(fit_path, fit)
        checkpoint = selected/f"block_{layer:02d}.pt"
        joint.recipe.save(checkpoint, joint._cpu(block.state_dict()))
        hashes[checkpoint.name] = joint.recipe.sha(checkpoint)
        fits[str(layer)] = dict(fit_path=str(fit_path), checkpoint=str(initializer),
            checkpoint_sha256=joint.recipe.sha(initializer))
        native.append(original); recovered.append(block)
    gain = torch.linspace(.7, 1.3, 16)
    joint.recipe.save(selected/"norm_gain.pt", gain)
    hashes["norm_gain.pt"] = joint.recipe.sha(selected/"norm_gain.pt")
    plan = dict(model=dict(path="tiny-native"), blocks=fits, selected_directory=str(selected),
        checkpoint_artifacts=hashes)
    embedding = torch.randn(23, 16).bfloat16().float()
    return plan, native, recovered, gain, embedding, config


def test_actual_packed_two_block_stream_preserves_selected_weights_geometry_and_trained_norm(tmp_path):
    plan, native, recovered, gain, embedding, config = fixture(tmp_path)
    calls = []
    def native_loader(path, layer, **_):
        assert path == "tiny-native"
        calls.append(layer)
        return deepcopy(native[layer])
    loaded = [packed.load_packed_block(plan, layer, device="cpu", loader=native_loader) for layer in range(2)]
    norm = packed.load_norm(plan, Qwen2RMSNorm(16))
    torch.testing.assert_close(norm.weight, gain, atol=0, rtol=0)
    assert not torch.equal(norm.weight, torch.ones_like(gain))
    for layer, block in enumerate(loaded):
        assert not any(p.requires_grad for p in block.parameters())
        torch.testing.assert_close(block.basis, recovered[layer].basis, atol=0, rtol=0)
    assert calls == [0, 1]
    assert not torch.equal(loaded[0].basis, loaded[1].basis)
    cases = [dict(identity="binary", family="two", split="development", input_ids=[1, 4, 2],
                  choices=["A", "B"], answer=1),
             dict(identity="four", family="four", split="development", input_ids=[7, 3, 1, 5, 2],
                  choices=list("ABCD"), answer=2)]
    labels = [19, 20, 21, 22]
    rotary = Qwen2RotaryEmbedding(config=config)
    result = stream_score(cases, input_embedding=embedding, block_loader=lambda i:loaded[i],
        final_readout=(norm, embedding), label_ids=labels, scratch_dir=tmp_path/"scratch",
        rotary=rotary, layer_count=2, device="cpu", microbatch=2)
    with torch.no_grad():
        for case, record in zip(cases, result["records"]):
            hidden = embedding[case["input_ids"]][None]
            positions = torch.arange(hidden.shape[1])[None]
            allowed = torch.ones(1, 1, hidden.shape[1], hidden.shape[1], dtype=torch.bool).tril()
            for block in recovered:
                hidden = block(hidden, attention_mask=allowed, position_ids=positions,
                    position_embeddings=rotary(hidden, positions))
            h = hidden[:, -1]
            normalized = h*torch.rsqrt(h.square().mean(-1, keepdim=True)+norm.variance_epsilon)*gain
            reference = F.linear(normalized, embedding[labels[:len(case["choices"])]]).squeeze(0)
            torch.testing.assert_close(torch.tensor(record["logits"]), reference, atol=5e-6, rtol=5e-6)
            assert record["prediction"] == int(reference.argmax())
    assert not (tmp_path/"scratch").exists()


@pytest.mark.parametrize("damage", ["hash", "geometry", "missing_parameter"])
def test_incomplete_or_changed_selected_state_never_packs(tmp_path, damage):
    plan, native, *_ = fixture(tmp_path)
    checkpoint = tmp_path/"selected/block_00.pt"
    if damage == "hash":
        with checkpoint.open("ab") as stream: stream.write(b"changed")
    else:
        state = joint._cpu(joint.recipe.load(checkpoint))
        if damage == "geometry":
            name = next(name for name in state if name.endswith("basis"))
            state[name] = state[name].clone(); state[name][0, 0] += .01
        else:
            name = next(iter(dict(semantic.build_enriched_block(deepcopy(native[0]),
                joint.recipe.load(plan["blocks"]["0"]["fit_path"]), parameter_dtype=torch.float32).named_parameters())))
            del state[name]
        torch.save(state, checkpoint)
        plan["checkpoint_artifacts"][checkpoint.name] = joint.recipe.sha(checkpoint)
    with pytest.raises((ValueError, RuntimeError)):
        packed.load_packed_block(plan, 0, device="cpu", loader=lambda *_a, **_k:deepcopy(native[0]))


def test_trained_norm_requires_exact_FP32_shape_and_authenticated_bytes(tmp_path):
    plan, *_ = fixture(tmp_path)
    checkpoint = tmp_path/"selected/norm_gain.pt"
    torch.save(torch.ones(16, dtype=torch.bfloat16), checkpoint)
    with pytest.raises(ValueError): packed.load_norm(plan, Qwen2RMSNorm(16))
    plan["checkpoint_artifacts"][checkpoint.name] = joint.recipe.sha(checkpoint)
    with pytest.raises(ValueError): packed.load_norm(plan, Qwen2RMSNorm(16))
    torch.save(torch.ones(15), checkpoint)
    plan["checkpoint_artifacts"][checkpoint.name] = joint.recipe.sha(checkpoint)
    with pytest.raises(ValueError): packed.load_norm(plan, Qwen2RMSNorm(16))


def test_packing_must_match_both_selected_and_fresh_reference_logits():
    cases = [dict(identity="x", family="tiny", answer=0, choices=["A", "B"])]
    records = [dict(identity="x", family="tiny", answer=0, prediction=0, logits=[1., 0.])]
    selected, fresh = dict(records=deepcopy(records)), dict(records=deepcopy(records))
    assert packed.compare(records, selected, fresh, cases)["passed"]
    fresh["records"][0]["logits"][0] = 2.
    assert not packed.compare(records, selected, fresh, cases)["passed"]
    fresh = dict(records=deepcopy(records)); selected["records"][0]["prediction"] = 1
    selected["records"][0]["logits"] = [0., 1.]
    assert not packed.compare(records, selected, fresh, cases)["passed"]


def test_authenticated_quality_miss_still_packs_but_changed_fresh_selection_does_not(tmp_path, monkeypatch):
    selected = tmp_path/"train/selected"; selected.mkdir(parents=True)
    verification = tmp_path/"verification"; verification.mkdir()
    plan = dict(fingerprint="training")
    metrics = dict(correct=46, count=64)
    checkpoint_manifest = dict(schema=joint.SCHEMA, blocks=36,
        artifacts={**{f"block_{i:02d}.pt":"unused" for i in range(36)}, "norm_gain.pt":"unused"},
        metadata=dict(epoch=2, development=metrics))
    joint.recipe.write(selected/"manifest.json", checkpoint_manifest)
    result = dict(selected_epoch=2, selected=metrics, quality_pass=False,
        selected_manifest_sha256=joint.recipe.sha(selected/"manifest.json"))
    fresh = dict(status="completed", passed=True, fingerprint="training", selected_epoch=2,
        selected_manifest_sha256=result["selected_manifest_sha256"], strict_state_load=True,
        fixed_state_unchanged=True, predictions_exact=True, logits_pass=True)
    joint.recipe.write(verification/"result.json", fresh)
    receipt = dict(status="completed", passed=True, fingerprint="training",
        result_sha256=joint.recipe.sha(verification/"result.json"))
    joint.recipe.write(verification/"manifest.json", receipt)
    monkeypatch.setattr(packed.recovered, "verify", lambda _:plan)
    monkeypatch.setattr(packed.recovered, "completed_training", lambda *_:result)
    assert packed.authenticated(tmp_path)[1]["quality_pass"] is False
    fresh["selected_epoch"] = 1
    (verification/"result.json").write_text(__import__("json").dumps(fresh))
    receipt["result_sha256"] = joint.recipe.sha(verification/"result.json")
    (verification/"manifest.json").write_text(__import__("json").dumps(receipt))
    with pytest.raises(ValueError, match="fresh strict verification"):
        packed.authenticated(tmp_path)


def test_denied_lease_never_initializes_CUDA_or_loads_model(tmp_path, monkeypatch):
    from experiments.structure_net import jspace_full_lease, jspace_qwen_parallel_model
    monkeypatch.setattr(packed, "verify", lambda _:dict(fingerprint="packing"))
    def denied(_): raise RuntimeError("reservation denied")
    monkeypatch.setattr(jspace_full_lease, "wait_ready", denied)
    monkeypatch.setattr(packed, "cap_allocator", lambda *_:pytest.fail("Allocator without lease"))
    monkeypatch.setattr(jspace_qwen_parallel_model, "load_readout_norm", lambda *_a, **_k:pytest.fail("Model before lease"))
    monkeypatch.setattr(torch.cuda, "init", lambda:pytest.fail("CUDA before lease"))
    with pytest.raises(RuntimeError, match="reservation denied"): packed.run(tmp_path)
    receipt = joint.recipe.read(tmp_path/"verification/failure.json")
    assert receipt["status"] == "failed" and receipt["lease_id"] is None
    assert receipt["peak_reserved_bytes"] is None
