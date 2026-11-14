"""No real processes/GPU jobs: sequencing, failure isolation and provenance."""
from types import SimpleNamespace

import pytest

from experiments.structure_net import jspace_allblocks_finish as finish


def sequence_fixture(tmp_path, monkeypatch):
    state = tmp_path / "supervisor"; state.mkdir(); (state / "logs").mkdir(); (state / "stages").mkdir()
    plan = {"fingerprint": "supervisor", "primary_fingerprint": "campaign",
            "commands": finish.commands(tmp_path / "run", tmp_path / "bundle", tmp_path / "report with spaces.md", tmp_path / "registry.json")}
    monkeypatch.setattr(finish, "verify", lambda root: plan)
    return state, plan


def test_exact_sequence_cpu_environments_and_reserved_validation(tmp_path, monkeypatch):
    state, plan = sequence_fixture(tmp_path, monkeypatch); seen = []; checked = []
    def runner(command, **kwargs):
        seen.append((command, kwargs["env"])); return SimpleNamespace(returncode=0)
    result = finish.execute(state, plan, runner=runner, checker=lambda stage, _: checked.append(stage) or {"proof": "hash"})
    assert checked == ["coordinator", "export", "validate", "analyze", "report", "store", "readback", "closeout"]
    assert result["status"] == "completed" and finish.read(state / "stages/coordinator.json")["returncode"] == 0
    for index, (command, env) in enumerate(seen):
        assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "1"
        assert env["PIXI_CACHE_DIR"] == "/tmp/structure-net-pixi-cache"
        assert env["UV_CACHE_DIR"] == "/tmp/structure-net-uv-cache"
        if index == 2:
            assert "CUDA_VISIBLE_DEVICES" not in env
            assert command[:6] == ["/usr/bin/python3", "/usr/local/bin/inference-ctl", "lease", "run", "--allocation", "GPU-0370fdb0=11264"]
            assert command[command.index("--") + 1:command.index("--") + 5] == ["pixi", "run", "--locked", "--offline"]
            assert finish.ARTIFACT in command and "validate" in command
        else: assert env["CUDA_VISIBLE_DEVICES"] == ""
        if index >= 3:
            assert "-e" not in command and finish.FINALIZER in command
    with pytest.raises(FileExistsError): finish.execute(state, plan, runner=runner, checker=lambda *_: {})
    assert len(seen) == 8


@pytest.mark.parametrize("provenance_failure", [False, True])
def test_execution_or_provenance_failure_stops_before_any_gpu_stage(tmp_path, monkeypatch, provenance_failure):
    state, plan = sequence_fixture(tmp_path, monkeypatch); seen = []
    def runner(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(returncode=0 if provenance_failure or len(seen) == 1 else 9)
    def checker(stage, _):
        if provenance_failure and stage == "export": raise ValueError("wrong campaign")
        return {}
    with pytest.raises(RuntimeError, match="export failed"):
        finish.execute(state, plan, runner=runner, checker=checker)
    assert len(seen) == 2 and not (state / "logs/validate.log").exists()
    assert finish.read(state / "failed.json")["stage"] == "export"
    receipt = finish.read(state / "stages/export.json")
    assert receipt["status"] == "failed" and receipt["returncode"] == (0 if provenance_failure else 9)
    assert not (state / "completed.json").exists()


def test_completed_negative_scientific_result_is_eligible_but_fingerprint_mismatch_is_not(tmp_path):
    root = tmp_path / "campaign"; root.mkdir(); (root / "evaluation").mkdir()
    plan = {"root": str(root), "artifact": str(tmp_path / "bundle"), "primary_fingerprint": "same"}
    result = {"status": "completed", "fingerprint": "same", "confirmed": False, "targets_met": False}
    finish.write(root / "campaign_results.json", result); finish.write(root / "evaluation/result.json", result)
    manifest = {"status": "completed", "fingerprint": "same", "artifacts": {}}
    finish.write(root / "evaluation/manifest.json", {**manifest, "artifacts": {"evaluation/result.json": finish.sha(root / "evaluation/result.json")}})
    for group in range(6):
        directory = root / "groups" / f"group_{group:02d}"
        phases = ["capture", "basis", "recovery"] + [f"block_{layer:02d}/fit_result" for layer in range(30 - group * 6, 36 - group * 6)]
        for phase in phases:
            target = directory / phase; target.mkdir(parents=True); finish.write(target / "manifest.json", manifest)
    assert finish.check_outputs("coordinator", plan)
    with pytest.raises(ValueError, match="fingerprint"):
        finish.check_outputs("coordinator", {**plan, "primary_fingerprint": "different"})


def test_preparation_freezes_supervisor_sources_and_launch_commands(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ""); monkeypatch.delenv("IC_LEASE", raising=False)
    monkeypatch.setattr(finish, "ROOT", tmp_path)
    source = tmp_path / "source.py"; source.write_text("original")
    monkeypatch.setattr(finish, "_sources", lambda primary: [source])
    root = tmp_path / "primary"; root.mkdir()
    primary = {"schema": "jspace.qwen-parallel.v2", "systems_only": False, "source_sha256": {str(source): finish.sha(source)}}
    primary["fingerprint"] = finish.digest(primary); finish.write(root / "plan.json", primary)
    state = tmp_path / "state"
    finish.prepare(root, state, root / "standalone", tmp_path / "report.md", tmp_path / "registry.json")
    plan = finish.verify(state)
    assert plan["primary_fingerprint"] == primary["fingerprint"] and len(plan["commands"]) == 8
    assert (state / "producer/source.py").read_text() == "original"
    source.write_text("changed")
    with pytest.raises(ValueError, match="source changed"): finish.verify(state)
