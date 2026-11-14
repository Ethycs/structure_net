"""CPU contracts for reserved profiling of immutable compact checkpoints."""
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import jspace_compact_performance as perf


def test_balanced_inputs_use_only_development():
    items = [{"family": f, "split": split, "identity": f + split + str(i)}
             for f in perf.DATASETS for split in ("train", "development", "evaluation") for i in range(3)]
    chosen = perf.selected_cases(items)
    assert len(chosen) == 8
    assert all(x["split"] == "development" for x in chosen)
    assert all(sum(x["family"] == f for x in chosen) == 2 for f in perf.DATASETS)
    with pytest.raises(ValueError, match="Insufficient"):
        perf.selected_cases(items[:2])


def test_five_pairs_balance_first_and_second_positions():
    schedule = perf.paired_orders(5)
    assert len(schedule) == 10
    assert [r for r, order in schedule] == [r for r in range(5) for _ in range(2)]
    for variant in ("reference", "candidate"):
        assert sum(order[0] == variant for _, order in schedule) == 5
        assert sum(order[1] == variant for _, order in schedule) == 5
    with pytest.raises(ValueError):
        perf.paired_orders(0)


def test_timing_summaries_keep_host_and_device_measurements_separate():
    rows = [{"variant": v, "execution": e, "wall_seconds": wall, "cuda_event_seconds": device}
            for e in ("eager", "graph") for v, wall, device in (("reference", 2., 3.), ("candidate", 1., 2.))]
    for value in perf.timing_summary(rows).values():
        assert value["wall_speedup"] == 2.
        assert value["cuda_event_speedup"] == 1.5


def test_errors_are_measured_without_requiring_bit_identity():
    result = perf.numerical_error(torch.tensor([3., 4.]), torch.tensor([4., 3.]))
    assert result["max_abs"] == result["rmse"] == 1.
    assert result["relative_rmse"] == pytest.approx(1 / (12.5 ** .5))
    assert result["equal"] is False
    with pytest.raises(ValueError, match="Nonfinite"):
        perf.numerical_error(torch.tensor([1.]), torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="shapes"):
        perf.numerical_error(torch.ones(1), torch.ones(2))


def valid_candidate():
    error = {"equal": False, "max_abs": .01, "rmse": .005, "relative_rmse": .002}
    return {"development_comparison": {"agreement": 1.}, "choice_logit_error": dict(error), "pairs": [
        {"output_error": dict(error),
         "kernel_error": {"max_abs": 0., "rmse": 0., "relative_rmse": 0., "equal": True},
         "reference_graph": {"graph_error": {"equal": True}},
         "candidate_graph": {"graph_error": {"equal": True}}}]}


def test_full_runtime_allows_quality_change_but_requires_finite_outputs_and_graph_parity():
    result = valid_candidate()
    assert perf.runtime_candidate_valid(result)
    result["development_comparison"]["agreement"] = .99
    assert perf.runtime_candidate_valid(result)
    result["pairs"][0]["candidate_graph"]["graph_error"]["equal"] = False
    assert not perf.runtime_candidate_valid(result)
    result["pairs"][0]["candidate_graph"]["graph_error"]["equal"] = True
    result["choice_logit_error"]["max_abs"] = float("nan")
    assert not perf.runtime_candidate_valid(result)


def test_kernel_fidelity_gate_is_separate_from_approximate_geometry():
    result = valid_candidate()
    result["development_comparison"]["agreement"] = .5
    result["pairs"][0]["output_error"]["relative_rmse"] = .2
    assert perf.runtime_candidate_valid(result)
    result["pairs"][0]["kernel_error"]["relative_rmse"] = .0021
    assert not perf.runtime_candidate_valid(result)


def test_candidate_factory_cannot_mutate_reference(monkeypatch):
    from experiments.structure_net import jspace_qwen_affine_fast as fast
    reference = torch.nn.Linear(2, 2, bias=False)
    original = reference.weight.detach().clone()
    received = []

    def create(block, **options):
        received.append(options)
        with torch.no_grad():
            block.weight.zero_()
        return block

    monkeypatch.setattr(fast, "FastAffineQwenBlock", SimpleNamespace(from_packed=create))
    candidate = perf.candidate_factory(reference, "geometry")
    assert torch.equal(reference.weight, original)
    assert not candidate.weight.any()
    assert received == [perf.CANDIDATES["geometry"]]
    perf.candidate_factory(reference, "combined", enabled=False)
    assert received[-1] == {**perf.CANDIDATES["combined"], "enabled": False}


def test_enabled_fusion_cannot_silently_fall_back():
    with pytest.raises(RuntimeError, match="silently bypass"):
        perf.ensure_fusion(SimpleNamespace(_can_fuse=lambda x: False), "fusion", None)
    assert perf.ensure_fusion(SimpleNamespace(_can_fuse=lambda x: True), "combined", None)
    assert not perf.ensure_fusion(None, "geometry", None)


def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(perf, "ROOT", tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    source = tmp_path / "source.py"
    source.write_text("unchanged = True\n")
    names = ["plan.json", "tasks.json", "fit.pt", "relation.pt", "reload_manifest.json"]
    names += [name for arm in perf.ARMS for name in (arm + ".pt", "training_" + arm + ".json", "train_" + arm + "_manifest.json")]
    for name in names:
        (parent / name).write_text(name)
    prior = {"input_sha256": {}, "fingerprint": "parent", "model": {"files": {}}, "cache_root": "cache", "label_ids": [1, 2]}
    monkeypatch.setattr(perf, "sources", lambda: [source])
    monkeypatch.setattr(perf.base, "cpu_only", lambda: None)
    monkeypatch.setattr(perf.base, "verify", lambda *a, **k: prior)
    checked = []
    monkeypatch.setattr(perf.base, "checked", lambda root, name: checked.append((root, name)))
    before = {p.name: p.read_bytes() for p in parent.iterdir()}
    root = tmp_path / "own_run"
    perf.prepare(root, parent)
    assert {p.name: p.read_bytes() for p in parent.iterdir()} == before
    assert checked == [(parent, "train_" + a) for a in perf.ARMS] + [(parent, "reload")]
    return root, parent, source


@pytest.mark.parametrize("changed", ["source", "snapshot", "input", "config"])
def test_freeze_rejects_changed_producer_or_inputs(tmp_path, monkeypatch, changed):
    root, parent, source = prepared(tmp_path, monkeypatch)
    perf.verify(root, inputs=True)
    if changed == "config":
        plan = json.loads((root / "plan.json").read_text())
        plan["config"]["paired_repeats"] += 1
        plan["fingerprint"] = perf.base.digest({k: v for k, v in plan.items() if k != "fingerprint"})
        (root / "plan.json").write_text(json.dumps(plan))
    else:
        target = {"source": source, "snapshot": root / "producer/source.py", "input": parent / "fit.pt"}[changed]
        target.write_text("changed")
    with pytest.raises(ValueError, match="Changed"):
        perf.verify(root, inputs=True)


def test_prepare_is_append_only(tmp_path, monkeypatch):
    root, parent, _ = prepared(tmp_path, monkeypatch)
    with pytest.raises(FileExistsError):
        perf.prepare(root, parent)


def test_reservation_failure_precedes_model_loading_and_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(perf, "verify", lambda *a, **k: {})
    monkeypatch.setattr(perf, "wait_ready", lambda *a: (_ for _ in ()).throw(RuntimeError("No reservation")))
    monkeypatch.setattr(perf.base, "load_small_qwen", lambda *a, **k: pytest.fail("Loaded model without reservation"))
    with pytest.raises(RuntimeError, match="No reservation"):
        perf.Stage(tmp_path, "profile")
    assert not (tmp_path / "profile").exists()


def test_full_runtime_rejects_unvalidated_candidate_before_gpu_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(perf, "checked", lambda *a: None)
    monkeypatch.setattr(perf, "Stage", lambda *a: pytest.fail("GPU stage started before validation"))
    (tmp_path / "bench").mkdir()
    result = valid_candidate()
    result["pairs"][0]["candidate_graph"]["graph_error"]["equal"] = False
    (tmp_path / "bench/result.json").write_text(json.dumps({"r1536_h64_gb/combined": result}))
    with pytest.raises(ValueError, match="graph parity required"):
        perf.run(tmp_path, "full")


def test_stage_artifact_hashes_are_checked(tmp_path, monkeypatch):
    root, _, _ = prepared(tmp_path, monkeypatch)
    plan = perf.verify(root)
    artifact = root / "profile.json"
    artifact.write_text("baseline profile")
    manifest = {"status": "completed", "fingerprint": plan["fingerprint"], "artifact_sha256": {"profile.json": perf.sha(artifact)}}
    (root / "profile_manifest.json").write_text(json.dumps(manifest))
    perf.checked(root, "profile")
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="Changed performance artifact"):
        perf.checked(root, "profile")


@pytest.mark.parametrize("compact", [False, True])
def test_static_decode_matches_final_prefill_token_and_replays_idempotently(compact):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(861)
    config = Qwen2Config(hidden_size=16, intermediate_size=28, num_attention_heads=4,
                        num_key_value_heads=2, num_hidden_layers=1, vocab_size=32,
                        attention_dropout=0., max_position_embeddings=32)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval().requires_grad_(False)
    block = model.model.layers[0]
    if compact:
        projection = {"rope_pairs": torch.zeros(2, 1, dtype=torch.long),
                      "value_basis": torch.eye(4).repeat(2, 1, 1)[:, :, :2]}
        block = perf.AffineQwenBlock(block, torch.eye(16)[:, :8], torch.zeros(1, 16),
                                    trainable=False, head_projection=projection)
        block.pack(dtype=torch.float32, buffer_dtype=torch.float32)
    stage = SimpleNamespace(model=model)
    x = torch.randn(1, 7, 16)
    with torch.inference_mode():
        prefill = perf.block_call(stage, block, x, "prefill")()
        decode = perf.block_call(stage, block, x, "decode")
        first = decode().clone()
        torch.testing.assert_close(first, prefill[:, -1:], atol=2e-6, rtol=2e-6)
        assert torch.equal(first, decode())
