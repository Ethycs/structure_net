import json
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import jspace_fast_native_compare as compare


def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "ROOT", tmp_path)
    source = tmp_path / "source.py"; source.write_text("fixed producer\n")
    parent, cache = tmp_path / "parent", tmp_path / "cache"
    parent.mkdir(); cache.mkdir()
    names = ["plan.json", "tasks.json", "fit.pt", "relation.pt", compare.ARM + ".pt",
             "training_" + compare.ARM + ".json", "train_" + compare.ARM + "_manifest.json", "reload_manifest.json"]
    for name in names:
        (parent / name).write_text(name)
    items = [{"split": "development", "family": family, "identity": family + str(i)}
             for family in compare.perf.DATASETS for i in range(2)]
    (cache / "items.json").write_text(json.dumps({"items": items}))
    prior = {"input_sha256": {}, "fingerprint": "accepted_parent", "model": {"files": {}},
             "cache_root": str(cache), "label_ids": [1, 2]}
    monkeypatch.setattr(compare, "sources", lambda: [source])
    monkeypatch.setattr(compare.perf.base, "cpu_only", lambda: None)
    monkeypatch.setattr(compare.perf.base, "verify", lambda *a, **k: prior)
    monkeypatch.setattr(compare.perf.base, "checked", lambda *a, **k: None)
    before = {p.name: p.read_bytes() for p in parent.iterdir()}
    root = tmp_path / "new_run"
    result = compare.prepare(root, parent)
    assert result["cases"] == 8 and before == {p.name: p.read_bytes() for p in parent.iterdir()}
    return root, parent, source


@pytest.mark.parametrize("changed", ["source", "snapshot", "input", "protocol"])
def test_own_freeze_rejects_changes_without_modifying_parent(tmp_path, monkeypatch, changed):
    root, parent, source = prepared(tmp_path, monkeypatch)
    compare.verify(root, inputs=True)
    if changed == "protocol":
        plan = json.loads((root / "plan.json").read_text())
        plan["config"]["paired_repeats"] = 6
        plan["fingerprint"] = compare.perf.base.digest({k: v for k, v in plan.items() if k != "fingerprint"})
        (root / "plan.json").write_text(json.dumps(plan))
    else:
        {"source": source, "snapshot": root / "producer/source.py", "input": parent / "fit.pt"}[changed].write_text("changed")
    with pytest.raises(ValueError, match="Changed"):
        compare.verify(root, inputs=True)


def test_prepare_is_append_only(tmp_path, monkeypatch):
    root, parent, _ = prepared(tmp_path, monkeypatch)
    with pytest.raises(FileExistsError):
        compare.prepare(root, parent)


def test_reservation_is_checked_before_artifacts_or_model_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "verify", lambda *a, **k: {})
    monkeypatch.setattr(compare, "wait_ready", lambda *a: (_ for _ in ()).throw(RuntimeError("No reservation")))
    monkeypatch.setattr(compare.perf.base, "load_small_qwen", lambda *a, **k: pytest.fail("Model loaded before reservation"))
    with pytest.raises(RuntimeError, match="No reservation"):
        compare.Stage(tmp_path)
    assert not (tmp_path / "native_compare").exists()


def test_pair_summary_and_aggregate_use_native_not_old_compact_denominator():
    rows = [{"variant": variant, "wall_seconds": wall, "cuda_event_seconds": device}
            for variant, wall, device in (("reference", 4., 6.), ("candidate", 2., 2.))]
    summary = compare.pair_summary(rows)
    assert summary["wall_seconds"]["speedup"] == 2.
    assert summary["cuda_event_seconds"]["speedup"] == 3.
    result = compare.aggregate([{"mode": mode, "summary": summary} for mode in compare.CONFIG["modes"]])
    assert result["prefill"]["wall_seconds"]["speedup"] == 2.
    assert result["decode"]["cuda_event_seconds"]["speedup"] == 3.


def test_orchestration_uses_native_and_fast_graphs_equally_without_quality_gate(tmp_path, monkeypatch):
    native, compact, candidate = object(), object(), object()
    cases = [{"split": "development", "family": family, "identity": family + str(i)}
             for family in compare.perf.DATASETS for i in range(2)]
    calls, timing, installed = [], [], []
    stage = SimpleNamespace(path=tmp_path, items=cases, plan={"case_identities": [c["identity"] for c in cases]},
        model=SimpleNamespace(model=SimpleNamespace(layers=[native])),
        x=lambda case: torch.zeros(1, 5, 16), guard=lambda devices: None, event=lambda *a, **k: None,
        finish=lambda result: result)
    def install(arm):
        installed.append(arm); stage.model.model.layers[-1] = compact; return compact
    stage.install_arm = install
    monkeypatch.setattr(compare, "Stage", lambda root: stage)
    monkeypatch.setattr(compare.perf, "candidate_factory", lambda block, name: candidate if block is compact else pytest.fail("Wrong checkpoint"))
    monkeypatch.setattr(compare.perf, "ensure_fusion", lambda *a: True)
    def block_call(owner, block, x, mode):
        assert owner is stage and block in (native, candidate)
        calls.append((block, mode, tuple(x.shape)))
        return lambda: torch.tensor([1. if block is native else 3.])
    monkeypatch.setattr(compare.perf, "block_call", block_call)
    monkeypatch.setattr(compare.perf, "warm_and_graph", lambda call:
        (SimpleNamespace(replay=call), {"graph_error": {"equal": True}}))
    def timed(call):
        output = float(call()[0]); timing.append(output)
        return {"wall_seconds": 2. if output == 1. else 1., "cuda_event_seconds": 2. if output == 1. else 1.}
    monkeypatch.setattr(compare.perf, "timed", timed)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    result = compare.run(tmp_path)
    assert installed == [compare.ARM] and len(result["pairs"]) == 16
    assert len(calls) == 32 and timing.count(1.) == timing.count(3.) == 160
    assert result["quality_selection"] is False and result["final_norm_in_timing"] is False
    assert all(not pair["native_vs_fast_error"]["equal"] for pair in result["pairs"])
    assert all(len(pair["timings"]) == 20 for pair in result["pairs"])
    assert result["summary"]["decode"]["wall_seconds"]["speedup"] == 2.
