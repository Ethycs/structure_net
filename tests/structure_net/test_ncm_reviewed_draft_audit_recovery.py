import json
from types import SimpleNamespace

import pytest

from experiments.structure_net import ncm_reviewed_draft_audit_recovery as recovery


def original_failure(tmp_path, monkeypatch):
    old = tmp_path / "old_workflow"
    (old / "stage_receipts").mkdir(parents=True)
    (old / "stage_receipts/development_finalize.json").write_text(json.dumps({
        "stage": "development_finalize", "exit_code": 1, "observer": "subprocess.wait"}))
    (old / "failure.json").write_text(json.dumps({"status": "stopped_failure_preserved"}))
    old.with_suffix(".worker.log").write_text("ModuleNotFoundError: No module named 'tiktoken'\n")
    monkeypatch.setattr(recovery, "OLD", old)
    monkeypatch.setattr(recovery, "NOMINAL_ARCHIVE", tmp_path / "absent_archive")
    return old


def test_admits_exact_preserved_dependency_failure(tmp_path, monkeypatch):
    original_failure(tmp_path, monkeypatch)
    assert recovery.require_original_failure()["exit_code"] == 1


@pytest.mark.parametrize("change", ["successful_exit", "different_error", "existing_archive"])
def test_rejects_other_recovery_conditions(tmp_path, monkeypatch, change):
    old = original_failure(tmp_path, monkeypatch)
    if change == "successful_exit":
        p = old / "stage_receipts/development_finalize.json"
        row = json.loads(p.read_text()); row["exit_code"] = 0; p.write_text(json.dumps(row))
    elif change == "different_error":
        old.with_suffix(".worker.log").write_text("scientific mismatch")
    else:
        recovery.NOMINAL_ARCHIVE.mkdir()
    with pytest.raises(ValueError):
        recovery.require_original_failure()


def test_dependency_directory_initialized_before_import(monkeypatch):
    calls = []
    monkeypatch.setattr(recovery, "dependencies", lambda: calls.append("dependencies"))
    def import_module(name):
        assert calls == ["dependencies"]
        assert name == "tiktoken"
        calls.append("import")
        return SimpleNamespace(__file__="existing/tiktoken/__init__.py", __version__="test")
    monkeypatch.setattr(recovery.importlib, "import_module", import_module)
    assert recovery.dependency_preflight()["tiktoken_version"] == "test"
    assert calls == ["dependencies", "import"]


def test_unchanged_auditor_runs_only_after_preflight(monkeypatch):
    calls = []
    monkeypatch.setattr(recovery, "require_original_failure", lambda: calls.append("failure"))
    monkeypatch.setattr(recovery, "configure", lambda: SimpleNamespace(finalize=lambda: calls.append("audit")))
    monkeypatch.setattr(recovery, "dependency_preflight", lambda: calls.append("dependency"))
    recovery.execute("finalize")
    assert calls == ["failure", "dependency", "audit"]


def test_fresh_archive_is_distinct_from_failed_nominal_destination():
    assert recovery.DEST != recovery.NOMINAL_ARCHIVE
    assert recovery.WORKFLOW != recovery.OLD
