from copy import deepcopy

import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding

from experiments.structure_net import jspace_recovered_assembly as assembly


def tiny_assembly(tmp_path):
    torch.manual_seed(151)
    config = Qwen2Config(hidden_size=8, intermediate_size=12, num_attention_heads=2,
        num_key_value_heads=1, num_hidden_layers=36, vocab_size=11, attention_dropout=0.)
    config._attn_implementation = "eager"
    native = [Qwen2DecoderLayer(config, i).bfloat16().float().eval().requires_grad_(False) for i in range(36)]
    compact, selections = [], {}
    for i, block in enumerate(native):
        fit = dict(basis=torch.linalg.qr(torch.randn(8, 4)).Q, means=torch.randn(1, 8)/10)
        branch = assembly.single.build_enriched_block(deepcopy(block), fit, parameter_dtype=torch.float32)
        with torch.no_grad():
            # Verify that assembly actually loads recovery, rather than rerunning initialization.
            next(branch.parameters()).reshape(-1)[i % 8] += .04
        fit_path = tmp_path/("first_enrich.pt" if i == 0 else f"all_enrich_{i:02d}.pt")
        checkpoint = tmp_path/f"selected_{i:02d}.pt"
        assembly.recipe.save(fit_path, fit)
        state = dict(block=assembly.joint._cpu(branch.state_dict()), epoch=i % 3)
        if i: state["layer"] = i
        assembly.recipe.save(checkpoint, state)
        selections[str(i)] = dict(fit_path=str(fit_path), checkpoint=str(checkpoint),
            checkpoint_sha256=assembly.recipe.sha(checkpoint), selected_epoch=i % 3,
            rank=4, head_dim=4, intermediate_size=12)
        compact.append(branch.pack(dtype=torch.float32, buffer_dtype=torch.float32))
    plan = dict(model=dict(path="tiny-native"), arms={a:assembly.compact_layers(a) for a in assembly.ARMS}, blocks=selections)
    cases = [dict(identity="a", family="tiny", split="development", input_ids=[1, 2, 3], choices=["a", "b"], answer=0),
             dict(identity="b", family="tiny", split="development", input_ids=[4, 1, 6, 2, 7], choices=["a", "b"], answer=1)]
    embedding = torch.randn(11, 8).bfloat16().float()
    return plan, native, compact, cases, embedding, Qwen2RMSNorm(8), Qwen2RotaryEmbedding(config)


def test_all_four_arms_execute_full_native_residual_path_and_selected_weights(tmp_path):
    plan, native, compact, cases, embedding, norm, rotary = tiny_assembly(tmp_path)
    for arm in assembly.ARMS:
        loaded = []
        def native_loader(path, layer, **_):
            assert path == "tiny-native"
            loaded.append(layer)
            return deepcopy(native[layer])
        storage = []
        def loader(layer):
            block = assembly.load_block(plan, arm, layer, device="cpu", loader=native_loader)
            storage.append(assembly.tensor_storage(block))
            if layer in plan["arms"][arm]:
                torch.testing.assert_close(block.basis, compact[layer].basis, atol=0, rtol=0)
            return block
        result = assembly.single.stream_score(cases, input_embedding=embedding, block_loader=loader,
            final_readout=(norm, embedding), label_ids=[9, 10], scratch_dir=tmp_path/arm,
            rotary=rotary, layer_count=36, device="cpu", microbatch=2)
        assert loaded == list(range(36)) and len(storage) == 36
        with torch.no_grad():
            for record, case in zip(result["records"], cases):
                hidden = embedding[case["input_ids"]][None]
                positions = torch.arange(len(case["input_ids"]))[None]
                allowed = (positions[:, :, None] >= positions[:, None, :])[:, None]
                mask = torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, torch.finfo(torch.float32).min)
                for layer in range(36):
                    model = compact[layer] if layer in plan["arms"][arm] else native[layer]
                    hidden = model(hidden, attention_mask=mask, position_ids=positions,
                        position_embeddings=rotary(hidden, positions))
                    hidden = hidden[0] if isinstance(hidden, tuple) else hidden
                expected = torch.nn.functional.linear(norm(hidden[:, -1]), embedding[[9, 10]])[0]
                torch.testing.assert_close(torch.tensor(record["logits"]), expected, atol=4e-6, rtol=4e-6)
                assert all(record[k] == case[k] for k in ("identity", "family", "answer"))
        assert not (tmp_path/arm).exists()
    assert assembly.compact_layers("hybrid") == [i for i in range(36) if i not in (12, 15)]
    assert plan["blocks"]["0"]["fit_path"].endswith("first_enrich.pt")


def test_selected_checkpoint_bytes_are_checked_before_load(tmp_path):
    plan, native, *_ = tiny_assembly(tmp_path)
    with open(plan["blocks"]["0"]["checkpoint"], "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        assembly.load_block(plan, "accepted_two", 0, device="cpu", loader=lambda *_a, **_k:deepcopy(native[0]))


def test_storage_counts_fixed_geometry_and_deduplicates_embedding_aliases():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(12.).reshape(3, 4))
            self.register_buffer("weight_view", self.weight.detach()[:, :2])
            self.register_buffer("geometry", torch.eye(4), persistent=False)
    block, embedding = Block(), torch.ones(7, 4)
    value = assembly.tensor_storage(block, embedding, embedding)
    assert value["storage_bytes"] == (12+16+28)*4
    assert value["logical_tensor_bytes"] == (12+6+16+28)*4
    assert value["unique_storages"] == 3


def test_control_requires_exact_predictions_finite_logits_and_existing_tolerance():
    cases = [dict(identity="x", family="tiny", answer=0, choices=["a", "b"])]
    records = [dict(identity="x", family="tiny", answer=0, prediction=0, logits=[1., 0.])]
    ref = dict(records=deepcopy(records))
    assert assembly.compare_control(records, ref, cases, 1)["passed"]
    records[0]["logits"][0] += 1e-5
    assert assembly.compare_control(records, ref, cases, 1)["passed"]
    records[0]["logits"][0] = 10.
    changed = assembly.compare_control(records, ref, cases, 1)
    assert changed["predictions_exact"] and not changed["logits_pass"] and not changed["passed"]
    records[0]["logits"][0] = float("nan")
    with pytest.raises(ValueError, match="Finite"):
        assembly.compare_control(records, ref, cases, 1)


def test_verified_isolated_quality_miss_allowed_but_wrong_selected_receipt_rejected(tmp_path, monkeypatch):
    for name in ("train", "verification"):
        (tmp_path/name).mkdir()
    plan = dict(fingerprint="parent")
    result = dict(selected_epoch=2, selected_checkpoint_sha256="selected", quality_pass=False)
    records = [dict(identity="x", family="tiny", answer=0, prediction=0, logits=[1., 0.])]
    assembly.recipe.write(tmp_path/"items.json", dict(items=[dict(identity="x", family="tiny", answer=0,
        choices=["a", "b"], split="development")]))
    assembly.recipe.write(tmp_path/"train/epoch_02_development.json", dict(records=records))
    verified = dict(status="completed", passed=True, strict_state_load=True, fingerprint="parent",
        campaign_fingerprint="parent", checkpoint_sha256="selected", selected_epoch=2, training_quality_pass=False, records=records)
    assembly.recipe.write(tmp_path/"verification/result.json", verified)
    manifest = dict(status="completed", passed=True, fingerprint="parent", checkpoint_sha256="selected",
        result_sha256=assembly.recipe.sha(tmp_path/"verification/result.json"))
    assembly.recipe.write(tmp_path/"verification/manifest.json", manifest)
    monkeypatch.setattr(assembly.independent, "verify", lambda _:plan)
    monkeypatch.setattr(assembly.independent, "verify_pilot", lambda *_:None)
    monkeypatch.setattr(assembly.sequential, "completed_training", lambda *_:result)
    assert assembly.verified_independent(tmp_path)[1]["quality_pass"] is False
    verified["checkpoint_sha256"] = "another"
    (tmp_path/"verification/result.json").write_text(__import__("json").dumps(verified))
    manifest["result_sha256"] = assembly.recipe.sha(tmp_path/"verification/result.json")
    (tmp_path/"verification/manifest.json").write_text(__import__("json").dumps(manifest))
    with pytest.raises(ValueError, match="fresh verification"):
        assembly.verified_independent(tmp_path)


@pytest.mark.parametrize("hybrid_correct,passes", [(47, True), (46, False)])
def test_assembled_primary_gate_and_paired_storage_reporting(tmp_path, monkeypatch, hybrid_correct, passes):
    cases = [dict(identity=str(i), family="tiny", answer=0, choices=["a", "b"]) for i in range(64)]
    plan = dict(fingerprint="frozen", cases=cases)
    monkeypatch.setattr(assembly, "verify", lambda _:plan)
    counts = dict(native=50, accepted_two=49, all_compact=30, hybrid=hybrid_correct)
    sizes = dict(native=1000, accepted_two=950, all_compact=500, hybrid=600)
    for arm in assembly.ARMS:
        path = tmp_path/arm; path.mkdir()
        rows = [dict(identity=c["identity"], family=c["family"], answer=0, prediction=int(i >= counts[arm]),
            logits=[1., 0.] if i < counts[arm] else [0., 1.]) for i,c in enumerate(cases)]
        result = dict(status="completed", arm=arm, fingerprint="frozen", records=rows,
            control=dict(passed=True) if arm in ("native", "accepted_two") else None,
            accuracy=assembly.single.ranking_summary(rows), resident_tensor_state=dict(storage_bytes=sizes[arm]))
        assembly.recipe.write(path/"result.json", result)
        assembly.recipe.write(path/"manifest.json", dict(status="completed", arm=arm, fingerprint="frozen",
            result_sha256=assembly.recipe.sha(path/"result.json")))
    result = assembly.summarize(tmp_path)
    assert result["primary_quality_pass"] is passes and result["primary_arm"] == "hybrid"
    assert result["comparisons"]["hybrid"]["paired"]["overall"]["net_lost_correct"] == 50-hybrid_correct
    assert result["comparisons"]["hybrid"]["tensor_storage_reduction"] == .4
    assert result["storage_comparison"].startswith("Matched FP32")
    assert result["original_native_bf16_tensor_bytes"] is None and not result["original_native_bf16_storage_measured"]
    assert not result["confirmed"] and not result["speed_measured"]
    with pytest.raises(FileExistsError): assembly.summarize(tmp_path)


def test_cpu_prepare_and_candidate_controls_precede_gpu(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(assembly.recipe, "verify", lambda _:pytest.fail("Source access before CPU-only guard"))
    with pytest.raises(ValueError, match="Hide CUDA"):
        assembly.prepare(*([tmp_path]*7))
    monkeypatch.setattr(assembly, "verify", lambda _:dict(fingerprint="frozen", arms={a:[] for a in assembly.ARMS}))
    monkeypatch.setattr(assembly, "completed_arm", lambda *_:dict(control=dict(passed=False)))
    with pytest.raises(ValueError, match="controls must pass"):
        assembly.run(tmp_path, "hybrid")
    assert not (tmp_path/"hybrid").exists()


def test_denied_lease_preserves_failed_receipt_without_cuda(tmp_path, monkeypatch):
    from experiments.structure_net import jspace_full_lease
    monkeypatch.setattr(assembly, "verify", lambda _:dict(fingerprint="frozen", arms={a:[] for a in assembly.ARMS}))
    def denied(_): raise RuntimeError("No reservation")
    monkeypatch.setattr(jspace_full_lease, "wait_ready", denied)
    monkeypatch.setattr(torch.cuda, "init", lambda:pytest.fail("CUDA before reservation"))
    with pytest.raises(RuntimeError, match="reservation"):
        assembly.run(tmp_path, "native")
    receipt = assembly.recipe.read(tmp_path/"native/failure.json")
    assert receipt["status"] == "failed" and receipt["peak_reserved_bytes"] is None and receipt["lease_id"] is None
    with pytest.raises(FileExistsError): assembly.run(tmp_path, "native")
