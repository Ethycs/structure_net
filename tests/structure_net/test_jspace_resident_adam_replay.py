"""CPU-only optimizer replay admission, timing and diagnostic isolation."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import jspace_resident_adam_replay as study


def metrics(correct=45, ce=1., original_kl=.2):
    return dict(correct=correct, count=64, accuracy=correct/64, ce=ce, original_kl=original_kl)


def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(study.base, "ROOT", tmp_path)
    source_root = tmp_path / "source"
    directory = source_root / "stages/block_24"
    directory.mkdir(parents=True)
    parent = dict(fingerprint="parent", input_sha256={})
    stage = dict(fingerprint="stage", input_sha256={})
    study.base.write(source_root / "plan.json", parent)
    study.base.write(directory / "plan.json", stage)
    producer = tmp_path / "producer.py"
    producer.write_text("# frozen producer\n")
    calls = []
    monkeypatch.setattr(study.source, "verify_stage", lambda root, cut: (parent, stage))
    monkeypatch.setattr(study.source, "local_import_closure", lambda paths: [producer])
    monkeypatch.setattr(study.source, "_pin_phase", lambda *args: None)
    def checked(path, phase, plan):
        calls.append(phase)
        return dict(passed=True, selected=metrics(46), warm_initial=metrics(), selected_epoch=2)
    monkeypatch.setattr(study.source, "checked", checked)
    root = tmp_path / "replay"
    plan = study.prepare(root, source_root, 24)
    return root, plan, producer, calls


def test_prepare_requires_all_source_phases_and_preserves_recipe(tmp_path, monkeypatch):
    root, plan, _, calls = prepared(tmp_path, monkeypatch)
    assert calls == list(study.source.PHASES)
    assert plan["cut"] == 24 and plan["arms"] == ["cpu", "resident"]
    assert plan["config"] == asdict(study.training.CONFIG)
    assert plan["diagnostic_checkpoint_policy"].endswith("never_auto_adopt")
    assert plan["confirmed"] is False
    assert study.verify_plan(root)[0] == plan
    with pytest.raises(FileExistsError):
        study.prepare(root, plan["source_root"], 24)


@pytest.mark.parametrize("change", ["source", "archive", "input"])
def test_verified_replay_rejects_changed_authenticated_dependencies(tmp_path, monkeypatch, change):
    root, plan, producer, _ = prepared(tmp_path, monkeypatch)
    target = (producer if change == "source" else root / "producer/producer.py"
              if change == "archive" else next(iter(plan["input_identity"])))
    from pathlib import Path
    Path(target).write_text("changed dependency\n")
    with pytest.raises(ValueError, match="changed"):
        study.verify_plan(root)


def phase_receipt(root, arm, phase, result, plan, files=None):
    path = root / arm / phase
    path.mkdir(parents=True)
    for name, value in (files or {}).items():
        if isinstance(value, bytes):
            (path / name).write_bytes(value)
        else:
            study.base.write(path / name, value)
    study.base.write(path / "result.json", result)
    study.base.write(path / "manifest.json", dict(status="completed", fingerprint=plan["fingerprint"],
        arm=arm, phase=phase, artifacts={p.name: study.base.sha(p) for p in path.iterdir() if p.is_file()}))
    return path


def training_receipt(root, plan, arm="cpu", mutate=None):
    path = root / arm / "train"
    path.mkdir(parents=True)
    (path / "selected.pt").write_bytes(b"selected protected checkpoint")
    digest = study.base.sha(path / "selected.pt")
    initial = metrics()
    history = []
    records = [dict(identity=str(i), family=("arc", "boolq", "hellaswag", "winogrande")[i//16],
                    answer=0, prediction=0 if i < 45 else 1, logits=[1., 0.] if i < 45 else [0., 1.]) for i in range(64)]
    for epoch in range(-1, 13):
        report = path / f"epoch_{epoch:02d}_development.json"
        study.base.write(report, dict(metrics=initial, records=records))
        updates = [] if epoch <= 0 else [dict(update=(epoch-1)*32+j+1,
            gradients_present_by_block=[True]*12, synchronized_train_window_seconds=2.,
            optimizer={"synchronized_optimizer_seconds": 1.}) for j in range(32)]
        row = dict(epoch=epoch, metrics=initial, training=updates,
            eligibility=study.training.candidate_eligibility(initial, initial),
            development_sha256=study.base.sha(report), checkpoint_updated=epoch == -1)
        if epoch == -1:
            row["checkpoint_sha256"] = digest
        history.append(row)
        study.base.write(path / f"epoch_{epoch:02d}.json", row)
    result = dict(status="completed", config=asdict(study.CONFIG), completed_epochs=12, updates=384,
        history=history, initial=initial, warm_initial=initial, selected=initial, selected_epoch=-1,
        selected_checkpoint_sha256=digest, strict_reload_exact=True, fixed_state_unchanged=True,
        quality_pass=True, elapsed_seconds=1000., layer_indices=list(range(24, 36)))
    if mutate:
        mutate(result)
    study.base.write(path / "result.json", result)
    study.base.write(path / "manifest.json", dict(status="completed", fingerprint=plan["fingerprint"],
        arm=arm, phase="train", artifacts={p.name: study.base.sha(p) for p in path.iterdir()}))
    return result


@pytest.mark.parametrize("mutation", [
    lambda result: result.update(completed_epochs=11),
    lambda result: result["history"][-1]["training"].pop(),
    lambda result: result["history"][-1].update(development_sha256="0"*64),
])
def test_train_receipt_requires_real_budget_and_authenticated_epoch_reports(tmp_path, mutation):
    plan = {"fingerprint": "replay", "cut": 24}
    training_receipt(tmp_path, plan, mutate=mutation)
    with pytest.raises(ValueError):
        study.checked(tmp_path, "cpu", "train", plan)


def test_checked_requires_matching_arm_and_discarded_pilot(tmp_path):
    plan = {"fingerprint": "replay", "cut": 24}
    result = dict(status="completed", passed=True, systems_only=True, discarded=True,
                  warm_replay_exact=True, probe={"passed": True})
    path = phase_receipt(tmp_path, "resident", "pilot", result, plan)
    receipt = study.source.read(path / "manifest.json")
    receipt["arm"] = "cpu"
    (path / "manifest.json").write_text(__import__("json").dumps(receipt))
    with pytest.raises(ValueError):
        study.checked(tmp_path, "resident", "pilot", plan)


def test_fresh_verify_must_reference_actual_selected_training_checkpoint(tmp_path):
    plan = {"fingerprint": "replay", "cut": 24}
    trained = training_receipt(tmp_path, plan)
    assert study.checked(tmp_path, "cpu", "train", plan)["updates"] == 384
    phase_receipt(tmp_path, "cpu", "verify", dict(status="completed", passed=True,
        exact_packed_replay=True, fixed_state_unchanged=True, metrics=trained["selected"],
        selected_checkpoint_sha256="0"*64), plan)
    with pytest.raises(ValueError):
        study.checked(tmp_path, "cpu", "verify", plan)


def test_gpu_training_cannot_start_before_own_pilot_passes(tmp_path, monkeypatch):
    plan = {"fingerprint": "replay", "cut": 24}
    monkeypatch.setattr(study, "verify_plan", lambda root: (plan, {}, {}))
    def rejected(*args):
        raise ValueError("No completed own-arm pilot")
    monkeypatch.setattr(study, "checked", rejected)
    monkeypatch.setattr(study, "allocator_settings", lambda *a, **k: pytest.fail("CUDA allocator reached"))
    with pytest.raises(ValueError, match="own-arm pilot"):
        study.run(tmp_path, "resident", "train")
    assert not (tmp_path / "resident/train").exists()


def test_timing_adapter_preserves_real_cpu_math_and_state():
    a = torch.nn.Parameter(torch.linspace(-1, 1, 17).bfloat16())
    b = torch.nn.Parameter(a.detach().clone())
    first = study.resident.StreamingCPUAdamW([a], lr=2e-5, chunk_bytes=32, pin_memory=False)
    second = study.resident.StreamingCPUAdamW([b], lr=2e-5, chunk_bytes=32, pin_memory=False)
    events = []
    timed = study.TimedOptimizer(second, devices=("cpu",), guard=lambda: events.append("guard"))
    for i in range(4):
        a.grad = (torch.linspace(-.3, .4, 17)*(i+1)).bfloat16()
        b.grad = a.grad.clone()
        expected = first.step(max_grad_norm=1.)
        actual = timed.step(max_grad_norm=1.)
        assert actual["synchronized_optimizer_seconds"] >= 0
        assert actual["clip_coefficient"] == expected["clip_coefficient"]
        assert torch.equal(a, b)
        assert timed.param_groups is second.param_groups
        for key in study.resident.STATE_KEYS:
            assert torch.equal(first.state[a][key], second.state[b][key])
    assert events == ["guard"]*8


@pytest.mark.parametrize("resident_arm", [False, True])
def test_optimizer_timing_includes_same_two_guard_calls(monkeypatch, resident_arm):
    clock = [0.]
    monkeypatch.setattr(study.time, "perf_counter", lambda: clock[0])
    events = []
    def guard():
        events.append("guard"); clock[0] += 7.
    cls = study.resident.ResidentSpillAdamW if resident_arm else study.resident.StreamingCPUAdamW
    optimizer = object.__new__(cls)
    def update():
        if resident_arm: guard()
        events.append("update"); clock[0] += 3.
        if resident_arm: guard()
        return {}
    optimizer.step = update
    result = study.TimedOptimizer(optimizer, guard=guard).step()
    assert events == ["guard", "update", "guard"]
    assert result["synchronized_optimizer_seconds"] == 17.


def test_transfer_accounting_excludes_generic_pilot_fingerprints():
    accounting = study.TRANSFER_ACCOUNTING
    assert accounting["covers_all_process_transfers"] is False
    assert accounting["generic_master_fingerprint_downloads_in_counters"] is False
    assert accounting["generic_master_fingerprint_downloads_in_update_timers"] is False
    assert "Full phase elapsed time includes" in accounting["scope"]


def test_diagnostic_improvements_remain_separate_from_selected_checkpoint(tmp_path):
    selected = tmp_path / "selected.pt"
    selected.write_bytes(b"protected scientific winner")
    raw = {"value": torch.tensor(-1.)}
    live = {"value": torch.tensor(0.)}
    trainer = SimpleNamespace(snapshot=lambda: deepcopy(live))
    diagnostic = study.DiagnosticCheckpoint(tmp_path, trainer, raw, metrics())
    diagnostic.observe(-1, metrics())
    diagnostic.observe(0, metrics(46, ce=2.))  # Better task score, ineligible CE.
    assert diagnostic.best["epoch"] == 0 and not diagnostic.best["eligibility"]["eligible"]
    assert diagnostic.best["diagnostic_only"] and not diagnostic.best["automatically_adopted"]
    assert torch.load(tmp_path / "diagnostic_best_task.pt", weights_only=True)["value"].item() == 0
    assert selected.read_bytes() == b"protected scientific winner"
    diagnostic.observe(1, metrics(46, ce=2.))
    assert diagnostic.best["epoch"] == 0
    assert len(list(tmp_path.glob("diagnostic_improvement_*.json"))) == 2


class FakeCuda:
    def __init__(self, visible):
        self.visible = visible; self.calls = []
        self.memory = SimpleNamespace(get_allocator_backend=lambda: "native")
    def is_initialized(self): self.calls.append("initialized"); return False
    def get_device_properties(self, index):
        self.calls.append(("properties", index))
        return SimpleNamespace(total_memory=(12 if "0370fdb0" in self.visible[index] else 8)*1024**3)
    def set_per_process_memory_fraction(self, value, device): self.calls.append(("cap", device, value))


def test_dual_uuid_caps_are_role_based_not_device_order():
    front, tail = "GPU-0370fdb0-front", "GPU-7b1db3cb-tail"
    visible = [tail, front]
    cuda = FakeCuda(visible)
    result = study.allocator_settings({"allocation": {front: 11264, tail: 7168}},
                                      cuda=cuda, visible=",".join(visible))
    assert result["front"]["device"] == 1 and result["tail"]["device"] == 0
    assert result["front"]["allocator_limit_bytes"] == 11008*1024**2
    assert result["tail"]["allocator_limit_bytes"] == 6912*1024**2
    assert len([x for x in cuda.calls if isinstance(x, tuple) and x[0] == "cap"]) == 2
    forbidden = FakeCuda(visible)
    with pytest.raises(ValueError):
        study.allocator_settings({"allocation": {front: 11264, tail: 1024}},
                                 cuda=forbidden, visible=",".join(visible))
    assert forbidden.calls == []


def test_comparison_requires_both_fresh_verifications_without_adoption(tmp_path, monkeypatch):
    plan = {"fingerprint": "replay"}
    monkeypatch.setattr(study, "verify_plan", lambda root: (plan, {}, {}))
    calls = []
    def checked(root, arm, phase, passed_plan):
        calls.append((arm, phase))
        if phase == "verify" and arm == "resident":
            raise ValueError("No fresh resident verification")
        if phase == "verify": return {"passed": True}
        return dict(selected=metrics(), selected_epoch=-1, quality_pass=True, elapsed_seconds=10.,
            history=[{"training": [dict(synchronized_train_window_seconds=2.,
                optimizer={"synchronized_optimizer_seconds": 1.})]}])
    monkeypatch.setattr(study, "checked", checked)
    path = tmp_path / "cpu/train"; path.mkdir(parents=True)
    study.base.write(path / "diagnostic_best_task.json", {"automatically_adopted": False})
    with pytest.raises(ValueError, match="fresh resident"):
        study.compare(tmp_path)
    assert calls == [("cpu", "train"), ("cpu", "verify"), ("resident", "train"), ("resident", "verify")]
    assert not (tmp_path / "comparison.json").exists()


def test_historical_summary_uses_two_pilots_without_inventing_cpu_training(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    original_path = source_root / "stages/block_24/train"
    original_path.mkdir(parents=True)
    reference = [dict(identity="a", answer=0, prediction=0), dict(identity="b", answer=0, prediction=1)]
    actual = [dict(identity="a", answer=0, prediction=1), dict(identity="b", answer=0, prediction=0)]
    study.base.write(original_path / "epoch_02_development.json", dict(records=reference))
    diagnostic = tmp_path / "resident/train"
    diagnostic.mkdir(parents=True)
    study.base.write(diagnostic / "diagnostic_best_task.json", {"automatically_adopted": False})
    plan = dict(fingerprint="replay", source_root=str(source_root), cut=24)
    monkeypatch.setattr(study, "verify_plan", lambda root: (plan, {}, {}))
    calls = []
    def checked(root, arm, phase, ignored):
        calls.append((arm, phase))
        if phase == "pilot":
            return {"probe": {"update": dict(synchronized_train_window_seconds=2. if arm == "cpu" else 1.,
                optimizer={"synchronized_optimizer_seconds": 1. if arm == "cpu" else .5})}}
        assert arm == "resident", "Unrun CPU training must not be required or fabricated"
        if phase == "train":
            return dict(selected=metrics(46), quality_pass=True, selected_epoch=1)
        return dict(records=actual)
    monkeypatch.setattr(study, "checked", checked)
    monkeypatch.setattr(study.source, "checked", lambda *args: dict(selected=metrics(46), selected_epoch=2))
    report = study.historical_summary(tmp_path)
    assert report["paired"] == dict(wins=1, losses=1)
    assert report["pilot_train_window_ratio"] == 2.
    assert report["matched_full_training_speedup"] is None
    assert report["cpu_full_replay_included"] is False
    assert not report["confirmed"]
    assert calls == [("cpu", "pilot"), ("resident", "pilot"), ("resident", "train"), ("resident", "verify")]
