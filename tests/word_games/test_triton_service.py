"""Release integrity and lifecycle ownership for the standalone Triton service."""
import json
from pathlib import Path
from unittest.mock import Mock, call
from types import SimpleNamespace

import pytest

from experiments.word_games_triton import standalone_service as service


@pytest.fixture
def release(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    directory = root / "deployments/word_games_triton"
    directory.mkdir(parents=True)
    monkeypatch.setattr(service, "ROOT", root)
    monkeypatch.setattr(service, "RELEASE", directory)
    monkeypatch.setattr(service, "STATE", root / ".pixi/service")
    weights = root / "weights.gguf"
    weights.write_bytes(b"original weights")
    lock = {"schema_version": 1, "model": service.MODEL, "profile": "test",
            "repository_files": {}, "weights": {"path": str(weights),
            "sha256": service.sha(weights), "bytes": weights.stat().st_size},
            "images": {"triton": "sha256:fixture"}, "gpu_uuids": ["GPU-fixture"]}
    for key in ("native_runtime", "native_sysroot"):
        base = root / key
        base.mkdir()
        file = base / "library.so.1"
        file.write_bytes(key.encode())
        (base / "library.so").symlink_to(file.name)
        lock[key] = {"path": key, "files": {file.name: service.sha(file)},
                     "symlinks": {"library.so": file.name}}
    (directory / "release.lock.json").write_text(json.dumps(lock))
    config = {"image": lock["images"]["triton"], "container_name": service.CONTAINER,
              "volumes": [{"target": target, "source": str(source), "read_only": True}
                          for target, source in (("/model.gguf", weights), ("/models", directory / "models"),
                                                 ("/llama", root / "native_runtime"),
                                                 ("/native-sysroot", root / "native_sysroot"))],
              "deploy": {"resources": {"reservations": {"devices": [{"device_ids": lock["gpu_uuids"]}]}}},
              "environment": {"CUDA_VISIBLE_DEVICES": "GPU-fixture"}}
    monkeypatch.setattr(service, "run", Mock(return_value=lock["images"]["triton"]))
    monkeypatch.setattr(service, "compose", Mock(return_value=json.dumps({"services": {"model": config}})))
    return root, lock, config


def test_weight_receipt_is_reused_only_for_unchanged_file(release):
    root, _, _ = release
    assert not service.verify()["weight_receipt_reused"]
    assert service.verify()["weight_receipt_reused"]
    # Same length, different bytes must not reuse the previous digest receipt.
    (root / "weights.gguf").write_bytes(b"modified weights")
    with pytest.raises(ValueError, match="Checkpoint changed"):
        service.verify()


@pytest.mark.parametrize("mutation", ["bytes", "symlink"])
def test_runtime_mutations_are_rejected(release, mutation):
    root, _, _ = release
    if mutation == "bytes":
        (root / "native_runtime/library.so.1").write_bytes(b"changed")
    else:
        link = root / "native_runtime/library.so"
        link.unlink()
        link.symlink_to("missing.so")
    with pytest.raises(ValueError, match="Changed or missing"):
        service.verify()


def test_compose_cannot_substitute_an_unverified_mount(release, monkeypatch):
    _, _, config = release
    config["volumes"][0]["source"] = "/different.gguf"
    monkeypatch.setattr(service, "compose", Mock(return_value=json.dumps({"services": {"model": config}})))
    with pytest.raises(ValueError, match="Unexpected asset mount"):
        service.verify()


def test_foreign_container_cannot_be_stopped_as_this_service():
    with pytest.raises(RuntimeError, match="unmanaged"):
        service.require_owned({"Config": {"Labels": {"com.docker.compose.project": "unrelated"}}})
    service.require_owned(None)


def test_up_keeps_existing_owned_ready_service(release, monkeypatch):
    monkeypatch.setattr(service, "verify", Mock(return_value={"verified": True}))
    monkeypatch.setattr(service, "inspect", Mock(return_value={"Config": {"Labels": {
        "com.docker.compose.project": service.PROJECT}}}))
    monkeypatch.setattr(service, "ready", Mock(return_value=True))
    api = Mock()
    monkeypatch.setattr(service, "api", api)
    assert service.up()["model_ready"]
    api.assert_not_called()
    service.run.assert_not_called()
    service.compose.assert_called_once_with("up", "-d", "--wait", "--wait-timeout", "360")


def test_migration_failure_restores_previous_container_and_model(release, monkeypatch):
    monkeypatch.setattr(service, "verify", Mock(return_value={}))
    monkeypatch.setattr(service, "inspect", Mock(return_value={"Config": {"Labels": {}}, "State": {"Running": True}}))
    monkeypatch.setattr(service, "check_legacy", Mock())
    monkeypatch.setattr(service, "ready", Mock(return_value=True))
    monkeypatch.setattr(service, "wait_native_exit", Mock())
    monkeypatch.setattr(service.time, "time_ns", lambda: 123)
    api = Mock()
    monkeypatch.setattr(service, "api", api)
    service.compose.side_effect = [RuntimeError("startup failed"), "stopped"]
    with pytest.raises(RuntimeError, match="startup failed"):
        service.up(migrate=True)
    backup = service.CONTAINER + "-previous-123"
    assert service.run.call_args_list == [
        call("docker", "stop", "--time", "75", service.CONTAINER),
        call("docker", "rename", service.CONTAINER, backup),
        call("docker", "rename", backup, service.CONTAINER),
        call("docker", "start", service.CONTAINER)]
    assert api.call_args_list == [call("unload"), call("load")]


def test_legacy_adoption_rejects_other_image(release):
    with pytest.raises(ValueError, match="different image"):
        service.check_legacy({"Image": "sha256:unrelated"})


@pytest.mark.parametrize("message", ["Error: No such object: word-games-triton", "error: no such object: word-games-triton"])
def test_missing_container_is_normal_stopped_state(monkeypatch, message):
    monkeypatch.setattr(service.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=1, stderr=message)))
    assert service.inspect() is None


def test_inspect_does_not_hide_docker_daemon_failure(monkeypatch):
    monkeypatch.setattr(service.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=1, stderr="Cannot connect to Docker daemon")))
    with pytest.raises(RuntimeError, match="Docker daemon"):
        service.inspect()


def test_unload_waits_until_native_process_exits(monkeypatch):
    command = Mock(side_effect=["PID COMMAND\n42 /native-sysroot/ld-linux /llama/llama-server -m /model.gguf", "PID COMMAND\n1 tritonserver"])
    monkeypatch.setattr(service, "run", command)
    monkeypatch.setattr(service.time, "sleep", Mock())
    service.wait_native_exit()
    assert command.call_count == 2
    service.time.sleep.assert_called_once_with(.25)


def test_pinned_runtime_preserves_selected_optimizations():
    release = Path("deployments/word_games_triton")
    lock = json.loads((release / "release.lock.json").read_text())
    runtime = json.loads((release / f"models/{service.MODEL}/1/runtime.json").read_text())
    args = runtime["native_args"]
    expected = {"--n-cpu-moe": "0", "--flash-attn": "on", "--cache-type-k": "q8_0",
                "--cache-type-v": "q8_0", "--split-mode": "layer", "--ctx-size": "8192",
                "--batch-size": "2048", "--ubatch-size": "1024", "--ctx-checkpoints": "0",
                "--threads": "1", "--threads-batch": "1", "--parallel": "1", "--spec-type": "none"}
    assert all(args[args.index(flag) + 1] == value for flag, value in expected.items())
    assert "--no-cache-prompt" in args and not runtime.get("public_prefixes")
    assert runtime["profile"] == lock["profile"] == "free_whole1024"
    for relative, expected_hash in lock["repository_files"].items():
        assert service.sha(release / relative) == expected_hash
