"""Run a host process tree only after a controller lease is active and ready."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .client import (
    ControllerClient, ReservationError, allocation_map, timestamp,
    utc_timestamp, validate_lease,
)


def _group_alive(group):
    # Zombies cannot retain CUDA memory; waiting for their reaper is unnecessary.
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group and fields[0] != "Z":
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def stop_group(process, grace=10):
    """Finish the whole experiment tree before letting services reclaim its cards."""
    for sig, timeout in ((signal.SIGTERM, grace), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + timeout
        while _group_alive(process.pid) and time.monotonic() < deadline:
            time.sleep(.05)
        if not _group_alive(process.pid):
            break
    process.wait(timeout=5)
    if _group_alive(process.pid):
        raise ReservationError("Experiment processes remain alive; lease retained for operator cleanup")


def run_reserved(command, *, allocation, duration, reason, contact,
                 receipt, client=None, poll_seconds=15, startup_timeout=180):
    """Gate exec, retain ownership on uncertain cleanup, and never retry POST /leases.

    Commands must stay in this host process group (no daemonization, Docker,
    SSH or detached systemd units). The controller accounts for that host PID.
    """
    client = client or ControllerClient()
    allocation = allocation_map(allocation)
    if not command or duration <= 0 or poll_seconds <= 0 or startup_timeout <= 0:
        raise ReservationError("Command, positive duration, and positive polling/startup intervals required")
    if not reason.strip() or not contact.strip():
        raise ReservationError("Reason and contact are required")
    client.require_actions()
    receipt = Path(receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: never overwrite an earlier experiment's evidence.
    with receipt.open("x", encoding="utf-8") as log:
        def record(event, **fields):
            log.write(json.dumps({"at": utc_timestamp(time.time()), "event": event, **fields}) + "\n")
            log.flush()
            os.fsync(log.fileno())

        record("request", controller=client.url, allocation=allocation, duration_seconds=duration)
        read_fd, write_fd = os.pipe()
        process = None
        lease_id = None
        previous_handlers = {}

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")

        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[sig] = signal.signal(sig, interrupted)
            env = os.environ.copy()
            for key in ("IC_KEY", "IC_KEY_FILE", "IC_LEASE", "STRUCTURENET_GPU_LEASE"):
                env.pop(key, None)
            env["CUDA_VISIBLE_DEVICES"] = ",".join(allocation)
            env["IC_URL"] = client.url
            process = subprocess.Popen(
                [sys.executable, "-m", "structure_net_resources.launcher", str(read_fd), *command],
                pass_fds=(read_fd,), start_new_session=True, env=env,
            )
            os.close(read_fd)
            read_fd = None
            end = utc_timestamp(time.time() + duration)
            record("child_gated", pid=process.pid, end=end)
            # Keep ownership alive until all children are stopped and released.
            # The canonical inference-ctl launcher also leases its supervisor PID.
            owner_pid = os.getpid()
            response = client.acquire(allocation=allocation, end=end, pid=owner_pid,
                                      reason=reason, contact=contact)
            response = response.get("lease", response) if isinstance(response, dict) else {}
            if not isinstance(response, dict) or not isinstance(response.get("id"), str):
                raise ReservationError("Lease response has no id; inspect controller operations before retrying")
            lease_id = response["id"]
            record("lease_created", lease_id=lease_id)
            deadline = time.monotonic() + startup_timeout
            while True:
                lease = client.lease(lease_id)
                if lease.get("state") == "active" and lease.get("ready") is True:
                    validate_lease(lease, allocation=allocation, pid=owner_pid)
                    # The controller must not silently give us a longer-lived lease.
                    if timestamp(lease["end"]) > timestamp(end) + 1:
                        raise ReservationError("Controller extended the requested lease end")
                    break
                if (lease.get("state") not in ("booked", "active")
                        or timestamp(lease.get("end")) <= time.time()
                        or time.monotonic() >= deadline):
                    raise ReservationError("Lease failed to become ready before startup deadline")
                if process.poll() is not None:
                    raise ReservationError("Gated child exited before lease activation")
                time.sleep(.2)
            record("lease_active", lease_id=lease_id, pid=owner_pid, child_pid=process.pid,
                   allocation=allocation, end=lease["end"])
            with os.fdopen(write_fd, "w") as gate:
                write_fd = None
                gate.write(json.dumps({"lease_id": lease_id}) + "\n")
            next_poll = time.monotonic()
            while process.poll() is None:
                if time.time() >= timestamp(lease["end"]):
                    raise ReservationError("Lease window ended; stopping experiment before release")
                if time.monotonic() >= next_poll:
                    lease = client.lease(lease_id)
                    validate_lease(lease, allocation=allocation, pid=owner_pid)
                    next_poll = time.monotonic() + poll_seconds
                time.sleep(.05)
            record("child_exit", returncode=process.returncode)
            return process.returncode if process.returncode >= 0 else 128 - process.returncode
        except BaseException as exc:
            # Exception text can contain user command material; store the type only.
            record("failed", error_type=type(exc).__name__, lease_id=lease_id)
            raise
        finally:
            # Do not interrupt cleanup with a second Ctrl-C/SIGTERM.
            for sig in previous_handlers:
                signal.signal(sig, signal.SIG_IGN)
            try:
                if read_fd is not None:
                    os.close(read_fd)
                if write_fd is not None:
                    os.close(write_fd)
                if process is not None:
                    stop_group(process)
                if lease_id is not None:
                    try:
                        client.release(lease_id)
                    except Exception:
                        record("release_failed", lease_id=lease_id)
                        raise
                    record("released", lease_id=lease_id)
                elif process is not None:
                    record("no_known_lease", pid=process.pid,
                           note="If acquisition timed out, inspect controller operations before retrying")
            finally:
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)


def _child():
    fd = int(sys.argv[1])
    with os.fdopen(fd) as gate:
        line = gate.readline()
    if not line:
        raise SystemExit(125)
    lease_id = json.loads(line)["lease_id"]
    os.environ["IC_LEASE"] = lease_id
    os.environ["STRUCTURENET_GPU_LEASE"] = lease_id
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)


if __name__ == "__main__":
    _child()
