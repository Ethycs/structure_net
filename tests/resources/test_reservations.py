"""CPU-only contract/lifecycle tests; the HTTP controller is a local test double."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest

from structure_net_resources.client import ControllerClient, ReservationError, require_gpu_lease
from structure_net_resources.launcher import run_reserved


GPU = "GPU-11111111-2222-3333-4444-555555555555"
OTHER_GPU = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
LEASE_ID = "012345abcdef"


def start_ticks(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


@pytest.fixture
def controller(monkeypatch):
    state = {
        "requests": [], "ready_after": 0, "reads": 0,
        "health": {"status": "ok", "mode": "enforce", "version": "1.5.0"},
        "lease": {"id": LEASE_ID, "state": "active", "ready": True,
                  "holder": "structure_net", "allocation": {GPU: 128},
                  "pid": os.getpid(), "pid_start": start_ticks(os.getpid()),
                  "end": time.time() + 60},
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def handle_request(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.loads(body) if body else None
            state["requests"].append((self.command, self.path, body))
            code = 200
            if self.command == "GET" and self.path == "/health":
                payload = state["health"]
            elif self.command == "GET" and self.path == f"/leases/{LEASE_ID}":
                state["reads"] += 1
                payload = dict(state["lease"])
                if state["reads"] <= state["ready_after"]:
                    payload["ready"] = False
                if state.get("revoke_after", float("inf")) < state["reads"]:
                    payload["state"] = "released"
            elif self.command == "POST" and self.path == "/leases":
                assert self.headers["Authorization"] == "Bearer test-secret"
                assert "holder" not in body  # Server derives ownership from credentials.
                state["lease"].update(body, pid_start=start_ticks(body["pid"]))
                code, payload = 201, {"lease": dict(state["lease"])}
            elif self.command == "DELETE" and self.path == f"/leases/{LEASE_ID}":
                assert self.headers["Authorization"] == "Bearer test-secret"
                state["lease"]["state"] = "released"
                payload = {"lease": dict(state["lease"])}
            else:
                code, payload = 404, {"error": "unsupported endpoint"}
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = do_PUT = do_DELETE = handle_request

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = ControllerClient(f"http://127.0.0.1:{server.server_port}", key="test-secret")
    monkeypatch.setenv("IC_LEASE", LEASE_ID)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    monkeypatch.delenv("STRUCTURENET_GPU_LEASE", raising=False)
    yield client, state
    server.shutdown()
    server.server_close()
    thread.join()


def test_official_launcher_environment_is_accepted(controller):
    client, state = controller
    assert require_gpu_lease([0], client=client)["id"] == LEASE_ID
    assert all(method == "GET" for method, _, _ in state["requests"])


@pytest.mark.parametrize("changes,match", [
    ({"ready": False}, "not ready"),
    ({"state": "released"}, "not active"),
    ({"end": 0}, "expired"),
    ({"holder": "another_group"}, "caller group"),
    ({"pid_start": -1}, "PID identity"),
    ({"allocation": {OTHER_GPU: 128}}, "CUDA visibility"),
])
def test_invalid_live_lease_is_refused(controller, changes, match):
    client, state = controller
    state["lease"].update(changes)
    with pytest.raises(ReservationError, match=match):
        require_gpu_lease([0], client=client)


def test_unrelated_process_is_refused(controller, monkeypatch):
    client, _ = controller
    monkeypatch.setattr("structure_net_resources.client._ancestors", lambda: set())
    with pytest.raises(ReservationError, match="supervisor"):
        require_gpu_lease([0], client=client)


@pytest.mark.parametrize("visible,devices", [("0", [0]), (GPU + "," + GPU, [0]), (GPU, [1])])
def test_cuda_ordinals_are_bounded_by_reserved_uuids(controller, monkeypatch, visible, devices):
    client, _ = controller
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(ReservationError):
        require_gpu_lease(devices, client=client)


def test_conflicting_identity_is_refused_without_network(controller, monkeypatch):
    client, state = controller
    monkeypatch.setenv("STRUCTURENET_GPU_LEASE", "abcdef012345")
    with pytest.raises(ReservationError, match="Conflicting"):
        require_gpu_lease([0], client=client)
    assert state["requests"] == []


def test_cpu_needs_no_lease_or_network(controller, monkeypatch):
    client, state = controller
    monkeypatch.delenv("IC_LEASE")
    assert require_gpu_lease([-1], client=client) is None
    assert state["requests"] == []


@pytest.mark.parametrize("mode", ["observe", "unknown", None])
def test_non_enforcing_controller_is_refused(controller, mode):
    client, state = controller
    state["health"]["mode"] = mode
    with pytest.raises(ReservationError, match="enforce mode"):
        require_gpu_lease([0], client=client)


def test_unsupported_swap_never_calls_server(controller):
    client, state = controller
    with pytest.raises(ReservationError, match="no model swap"):
        client.model_action("model", "load")
    assert state["requests"] == []


def run_options(tmp_path, client):
    return dict(allocation={GPU: 128}, duration=30, reason="CPU contract test",
                contact="local test", receipt=tmp_path / "receipt.jsonl", client=client,
                poll_seconds=.05, startup_timeout=3)


def test_lifecycle_waits_for_ready_exports_identity_and_releases(controller, tmp_path, monkeypatch):
    client, state = controller
    state["ready_after"] = 2
    monkeypatch.setenv("IC_KEY", "must-not-reach-child")
    output = tmp_path / "child.json"
    script = (
        "import json,os,pathlib; from structure_net_resources import require_gpu_lease; "
        "lease=require_gpu_lease([0]); "
        "pathlib.Path(__import__('sys').argv[1]).write_text(json.dumps({"
        "'id':os.environ['IC_LEASE'],'owner':lease['pid'],'ppid':os.getppid(),"
        "'visible':os.environ['CUDA_VISIBLE_DEVICES'],'has_key':'IC_KEY' in os.environ}))"
    )
    assert run_reserved([sys.executable, "-c", script, str(output)], **run_options(tmp_path, client)) == 0
    child = json.loads(output.read_text())
    assert child == {"id": LEASE_ID, "owner": os.getpid(), "ppid": os.getpid(),
                     "visible": GPU, "has_key": False}
    assert state["reads"] >= 4
    assert state["lease"]["state"] == "released"
    assert [method for method, _, _ in state["requests"]].count("POST") == 1
    assert state["requests"][-1][:2] == ("DELETE", f"/leases/{LEASE_ID}")
    receipt = (tmp_path / "receipt.jsonl").read_text()
    assert "test-secret" not in receipt and "must-not-reach-child" not in receipt
    assert json.loads(receipt.splitlines()[-1])["event"] == "released"
    assert not any("heartbeat" in path or path == "/swap" for _, path, _ in state["requests"])


def test_active_but_unready_lease_never_executes(controller, tmp_path):
    client, state = controller
    state["ready_after"] = 100
    output = tmp_path / "should-not-exist"
    options = run_options(tmp_path, client)
    options["startup_timeout"] = .1
    with pytest.raises(ReservationError, match="become ready"):
        run_reserved([sys.executable, "-c", "open(__import__('sys').argv[1],'w').close()", str(output)],
                     **options)
    assert not output.exists()
    assert state["lease"]["state"] == "released"


def test_revocation_stops_running_cpu_child_and_releases(controller, tmp_path):
    client, state = controller
    state["revoke_after"] = 2
    with pytest.raises(ReservationError, match="not active"):
        run_reserved([sys.executable, "-c", "import time; time.sleep(30)"],
                     **run_options(tmp_path, client))
    assert state["lease"]["state"] == "released"


def test_child_failure_still_releases(controller, tmp_path):
    client, state = controller
    assert run_reserved([sys.executable, "-c", "raise SystemExit(7)"],
                        **run_options(tmp_path, client)) == 7
    assert state["lease"]["state"] == "released"


def test_surviving_descendant_is_stopped_before_release(controller, tmp_path):
    client, state = controller
    pid_file = tmp_path / "descendant.pid"
    script = (
        "import subprocess,sys,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid))"
    )
    assert run_reserved([sys.executable, "-c", script, str(pid_file)],
                        **run_options(tmp_path, client)) == 0
    pid = int(pid_file.read_text())
    stat = Path(f"/proc/{pid}/stat")
    assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    assert state["requests"][-1][:2] == ("DELETE", f"/leases/{LEASE_ID}")
