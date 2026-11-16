"""Outcome-neutral NAL contracts; synthetic fixtures never certify recovery."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from experiments.neural_architecture_lab import distillation_recovery as cli

nal = cli.module("distillation_recovery")


def study(**kwargs):
    values = dict(study_id="TEST-RESPONSE-RECOVERY", teacher_sha256="1"*64, initializer_sha256="2"*64,
        tokenizer_sha256="3"*64, template_sha256="4"*64, data_sha256="5"*64, geometry_sha256="6"*64,
        arms=(nal.RecoveryArm("dense"),), seeds=(7,), intended_tasks=("qa",), precision="float32",
        deployment_hardware="CPU fixture", max_input_tokens=4, max_output_tokens=4, max_context_tokens=8,
        supervised_token_budget=16, max_updates=2)
    values.update(kwargs)
    return nal.RecoveryStudy(**values)


def quality_study(**kwargs):
    return study(profile="exploratory", arms=(nal.RecoveryArm("dense", runtime_json='{"learning_rate":0.001,"optimizer":"AdamW"}'),),
                  gates=(nal.QualityGate("task_success", 5),
                  nal.QualityGate("generation_success", 5, .5)), **kwargs)


def evidence(config, count=10):
    result = dict(systems={k: True for k in nal.SYSTEM_CHECKS}, checkpoint_sha256="a"*64)
    if config.profile == "systems":
        result["quality"] = None
    else:
        result["quality"] = dict(split="development", judging_provenance={"kind": "deterministic test rubric", "version": "v1"},
            records=[dict(example_id=f"case{i}", group_id=f"source{i}", task="qa",
                teacher_output="answer", student_output="answer", teacher=dict(task_success=True, generation_success=True),
                student=dict(task_success=True, generation_success=True)) for i in range(count)])
    return result


def test_study_is_frozen_and_fingerprints_scientific_inputs_not_device_slots():
    config = study()
    assert nal.RecoveryStudy.from_dict(asdict(config)) == config
    first = nal.make_experiments(config)[0]
    first.device_id = 4
    same = nal.make_experiments(config)[0]
    assert nal.experiment_fingerprint(first) == nal.experiment_fingerprint(same)
    for changed in (replace(config, data_sha256="a"*64), replace(config, seeds=(8,)),
                    replace(config, arms=(nal.RecoveryArm("dense", temperature=2.),))):
        assert nal.experiment_fingerprint(nal.make_experiments(changed)[0]) != nal.experiment_fingerprint(same)
    with pytest.raises(ValueError, match="tuples"):
        replace(config, seeds=[7])
    with pytest.raises(ValueError, match="context"):
        replace(config, max_output_tokens=5)
    with pytest.raises(ValueError, match="generation"):
        replace(config, profile="exploratory", gates=(nal.QualityGate("task_success", 5),))


def test_generation_failure_cannot_be_rescued_by_perfect_task_accuracy_or_trainer_flags():
    config = quality_study()
    value = evidence(config)
    value["confirmed"] = True
    value["quality"]["records"][0]["student"]["generation_success"] = False
    verdict = nal.classify(config, value)
    assert verdict["classification"] == "quality_gate_failed" and not verdict["confirmed"]
    assert verdict["gates"]["task_success"]["point_pass"]
    assert verdict["gates"]["generation_success"]["difference_pp"] == pytest.approx(-10)
    assert verdict["gates"]["generation_success"]["losses"] == 1


def test_quality_pass_alone_does_not_claim_improvement_and_toy_never_claims_quality():
    config = quality_study()
    value = evidence(config)
    assert nal.classify(config, value)["classification"] == "exploratory_quality_pass"
    for row in value["quality"]["records"]:
        row["initializer"] = dict(generation_success=False)
    assert nal.classify(config, value)["classification"] == "exploratory_improvement"
    toy = study()
    assert nal.classify(toy, evidence(toy)) == dict(classification="systems_passed", confirmed=False, gates=None)
    with pytest.raises(ValueError, match="shakedown"):
        nal.classify(toy, value)
    bad = evidence(toy); bad["systems"]["save_reload_exact"] = False
    with pytest.raises(ValueError, match="systems"):
        nal.classify(toy, bad)


def test_family_gate_is_not_rescued_by_pooled_accuracy():
    config = quality_study()
    config = replace(config, intended_tasks=("qa", "hard"),
                     gates=(*config.gates, nal.QualityGate("task_success", 5, task="hard")))
    value = evidence(config, 40)
    value["quality"]["records"][-1]["task"] = "hard"
    value["quality"]["records"][-1]["student"]["task_success"] = False
    result = nal.classify(config, value)
    assert result["gates"]["task_success"]["point_pass"]
    assert not result["gates"]["task_success:hard"]["point_pass"]
    assert result["classification"] == "quality_gate_failed"


def confirmation():
    config = replace(quality_study(), profile="confirmatory", confirmation_data_sha256="b"*64,
                     seeds=tuple(range(5)), confirmation_min_cases=1000)
    value = evidence(config, 4000)
    value["quality"].update(split="test", data_sha256="b"*64, previously_exposed=False,
                            opened_at="2026-11-15T01:00:00+00:00")
    value["selection"] = dict(checkpoint_sha256=value["checkpoint_sha256"], study_fingerprint=config.fingerprint,
                             test_data_sha256="b"*64, frozen_at="2026-11-15T00:00:00+00:00")
    return config, value


@pytest.mark.parametrize("change", ("exposed", "checkpoint", "chronology", "groups", "small", "test_identity"))
def test_confirmation_requires_fresh_paired_powered_evidence_not_a_boolean(change):
    config, value = confirmation()
    assert nal.classify(config, value)["confirmed"]
    if change == "exposed": value["quality"]["previously_exposed"] = True
    elif change == "checkpoint": value["selection"]["checkpoint_sha256"] = "c"*64
    elif change == "chronology": value["selection"]["frozen_at"] = value["quality"]["opened_at"]
    elif change == "groups":
        for row in value["quality"]["records"]: row["group_id"] = "one dependent conversation"
    elif change == "small": value["quality"]["records"] = value["quality"]["records"][:64]
    else: value["quality"]["data_sha256"] = "d"*64
    value["confirmed"] = True
    assert not nal.classify(config, value)["confirmed"]


def setup_run(tmp_path, config=None):
    config = study() if config is None else config
    producer = tmp_path / "producer.py"; producer.write_text("# frozen producer\n")
    root = tmp_path / "run"
    nal.prepare(root, config, [producer])
    return root, config, nal.make_experiments(config)[0], producer


def adapter(**kwargs):
    config, out = kwargs["study"], kwargs["output_dir"]
    kwargs["guard"]()
    (out / "checkpoint.bin").write_bytes(b"real test artifact, no quality claim")
    value = evidence(config)
    value.update(identities={name: getattr(config, field) for name, field in (
        ("teacher", "teacher_sha256"), ("student_initializer", "initializer_sha256"),
        ("tokenizer", "tokenizer_sha256"), ("chat_template", "template_sha256"), ("data", "data_sha256"))},
        checkpoint="checkpoint.bin", checkpoint_sha256=nal.sha(out / "checkpoint.bin"),
        required_artifacts=["checkpoint.bin"], model_architecture=[2, 2], model_parameters=6,
        progress=dict(optimizer_updates=1, supervised_tokens=8, response_tokens=8, input_tokens=12,
                      initialization_seconds=.001, training_seconds=.002))
    return value


def test_worker_uses_existing_ledger_readback_and_resume_without_rerunning(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    root, config, experiment, _ = setup_run(tmp_path)
    calls = []
    def run(**kwargs):
        calls.append(1); return adapter(**kwargs)
    worker = nal.RecoveryWorker(root, run)
    first = worker(experiment, -1)
    assert first.status is nal.ExperimentStatus.COMPLETED and first.primary_metric == 0
    assert worker(experiment, -1).metrics == first.metrics and calls == [1]
    result, loaded = nal.checked(root, experiment)
    assert loaded.experiment_id == experiment.id and result["verdict"]["classification"] == "systems_passed"
    assert not nal.aggregate(root)["confirmed"]
    checkpoint = Path(loaded.model_checkpoint)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact"):
        worker(experiment, -1)


def test_resume_refuses_incomplete_inventory_but_uses_archived_source_for_readback(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    root, _, experiment, producer = setup_run(tmp_path)
    nal.RecoveryWorker(root, adapter)(experiment, -1)
    result_path = nal.run_directory(root, experiment) / "result.json"
    original = nal.read(result_path)
    bad = deepcopy(original); bad["artifacts"].pop("checkpoint.bin")
    result_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="inventory"):
        nal.checked(root, experiment)
    result_path.write_text(json.dumps(original))
    producer.write_text("# changed source\n")
    nal.checked(root, experiment)  # Historical readback survives subsequent code development.
    with pytest.raises(ValueError, match="source"):
        nal.verify_plan(root, live=True)
    archived = next((root / "producer").iterdir())
    archived.write_text("# corrupted archive\n")
    with pytest.raises(ValueError, match="source"):
        nal.checked(root, experiment)


def test_failed_admission_precedes_adapter_and_is_preserved(tmp_path, monkeypatch):
    monkeypatch.delenv("IC_LEASE", raising=False)
    root, _, experiment, _ = setup_run(tmp_path)
    worker = nal.RecoveryWorker(root, lambda **kw: pytest.fail("Adapter ran before lease"))
    result = worker(experiment, 0)
    assert result.status is nal.ExperimentStatus.FAILED and "inference-ctl" in result.error
    failure = nal.read(nal.run_directory(root, experiment) / "failure.json")
    assert failure["classification"] == "execution_failure" and not failure["confirmed"]
    assert nal.ExperimentResultLedger(root).load(experiment) is None
    with pytest.raises(FileExistsError, match="preserved"):
        worker(experiment, 0)


def test_active_lease_guard_rechecks_identity_without_cuda_allocation(monkeypatch):
    monkeypatch.setenv("IC_LEASE", "attached-before-python")
    calls = []
    current = dict(id="lease", allocation={"GPU-device": 1000})
    def check(devices):
        calls.append(devices); return deepcopy(current)
    guard = nal.require_device(0, check)
    assert calls == [[0], [0]]
    current["id"] = "other"
    with pytest.raises(RuntimeError, match="Reservation"):
        guard()


def test_aggregate_keeps_pending_cells_and_quality_failures_separate(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    config = replace(quality_study(), seeds=(7, 8))
    root, _, experiment, _ = setup_run(tmp_path, config)
    def fail_quality(**kwargs):
        value = adapter(**kwargs)
        for row in value["quality"]["records"]: row["student"]["generation_success"] = False
        return value
    result = nal.RecoveryWorker(root, fail_quality)(experiment, -1)
    assert result.status is nal.ExperimentStatus.COMPLETED
    summary = nal.aggregate(root)
    assert summary["status"] == "partial" and not summary["confirmed"]
    assert summary["runs"][0]["classification"] == "quality_gate_failed"
    assert summary["runs"][1]["status"] == "prepared"


def test_actual_cpu_update_save_reload_and_ledger_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    result = cli.cpu_shakedown(tmp_path / "actual")
    assert result["status"] == "completed" and not result["confirmed"]
    assert result["runs"][0]["classification"] == "systems_passed"
    root = tmp_path / "actual"
    _, config = nal.verify_plan(root)
    record, ledger = nal.checked(root, nal.make_experiments(config)[0])
    assert ledger.metrics["supervised_tokens"] == 12 and ledger.metrics["optimizer_updates"] == 2
    assert ledger.primary_metric == 0 and record["verdict"]["gates"] is None
    assert cli.cpu_shakedown(root) == result


@pytest.mark.parametrize("symlink", (False, True))
def test_real_cli_relative_and_symlink_roots_resume_and_aggregate(tmp_path, monkeypatch, symlink):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.chdir(tmp_path)
    physical = tmp_path / "physical"
    physical.mkdir()
    if symlink:
        (tmp_path / "data").symlink_to(physical, target_is_directory=True)
        root = Path("data/study")
    else:
        root = Path("physical/study")
    first = cli.main(["cpu-shakedown", "--root", str(root)])
    _, config = nal.verify_plan(root)
    experiment = nal.make_experiments(config)[0]
    directory = nal.run_directory(root, experiment)
    assert directory == physical / "study/runs/dense_reference/seed_17"
    _, loaded = nal.checked(root, experiment)
    assert Path(loaded.model_checkpoint) == directory / "training/final.pt"
    checkpoint_sha = nal.sha(loaded.model_checkpoint)
    progress_sha = nal.sha(directory / "progress.jsonl")
    assert cli.main(["cpu-shakedown", "--root", str(root)]) == first
    assert nal.sha(loaded.model_checkpoint) == checkpoint_sha
    assert nal.sha(directory / "progress.jsonl") == progress_sha
    output = Path("aggregate/readback.json")
    assert cli.main(["aggregate", "--root", str(root), "--output", str(output)]) == first
    assert nal.read(output) == first
