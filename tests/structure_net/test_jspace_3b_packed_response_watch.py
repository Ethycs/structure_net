import json

from experiments.structure_net import jspace_3b_packed_response_watch as watcher


def test_checkpoint_is_not_ready_until_post_save_event(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    (train / "update_0046.pt").write_bytes(b"partly written checkpoint")
    progress = train / "progress.jsonl"
    progress.write_text(json.dumps({"phase": "training", "update": 46}) + "\n")
    assert watcher.completed_saves(tmp_path) == set()
    progress.write_text(progress.read_text() + json.dumps({"phase": "response_development", "update": 46}) + "\n{" )
    assert watcher.completed_saves(tmp_path) == {46}


def test_post_save_event_alone_does_not_manufacture_a_checkpoint(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train/progress.jsonl").write_text(json.dumps({"phase": "response_development", "update": 0}))
    assert watcher.completed_saves(tmp_path) == set()


def test_observer_refuses_existing_artifact_root(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    (tmp_path / "watcher.json").write_text("preserved")
    with pytest.raises(FileExistsError):
        watcher.run(tmp_path, tmp_path / "campaign")
    assert (tmp_path / "watcher.json").read_text() == "preserved"


def test_observer_requires_cuda_hidden(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CUDA hidden"):
        watcher.run(tmp_path, tmp_path / "campaign")


def test_failed_process_cannot_promote_an_existing_result(tmp_path):
    assert watcher.completed_result(tmp_path, 1) is None


def test_completed_result_requires_authentic_run_receipt(tmp_path):
    import pytest
    (tmp_path / "run").mkdir()
    plan = {"fingerprint": "fp", "checkpoint_sha256": "checkpoint", "checkpoint_metadata": {"update": 46}}
    result = {"fingerprint": "fp", "checkpoint_sha256": "checkpoint", "update": 46, "status": "quality_gate_failed"}
    watcher.write(tmp_path / "plan.json", plan)
    watcher.write(tmp_path / "run/result.json", result)
    watcher.write(tmp_path / "run/manifest.json", {"status": "completed", "fingerprint": "fp",
        "artifacts": {"result.json": watcher.digest(tmp_path / "run/result.json")}})
    assert watcher.completed_result(tmp_path, 0) == result
    result["status"] = "exploratory_quality_pass"
    watcher.write(tmp_path / "run/result.json", result)
    with pytest.raises(ValueError, match="artifact changed"):
        watcher.completed_result(tmp_path, 0)
