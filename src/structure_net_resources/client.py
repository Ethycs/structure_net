"""Host-side client for the deployed inference-controller 1.5 lease contract.

Lease state and memory readiness are separate. The controller owns admission
and scheduling; this package only gates StructureNet processes on a live lease.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time
from urllib import error, parse, request


DEFAULT_URL = "http://10.158.218.141:8097"
UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")


class ReservationError(RuntimeError):
    """A refused, unavailable, or unverified reservation; never permission to run."""


def allocation_map(value):
    if not isinstance(value, dict) or not value:
        raise ReservationError("An exact, nonempty GPU UUID -> MiB allocation is required")
    for gpu, mib in value.items():
        if not isinstance(gpu, str) or not UUID.fullmatch(gpu):
            raise ReservationError("Use full physical GPU UUIDs, not CUDA ordinals")
        if type(mib) is not int or mib <= 0:
            raise ReservationError("Every allocation must be a positive integer in MiB")
    return dict(value)


def timestamp(value):
    try:
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("timezone required")
            result = parsed.timestamp()
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result = float(value)
        else:
            raise ValueError("timestamp required")
        if not math.isfinite(result):
            raise ValueError("finite time required")
        return result
    except (ValueError, TypeError, OverflowError) as exc:
        raise ReservationError("Expected a UTC/offset ISO timestamp or Unix seconds") from exc


def utc_timestamp(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def segment(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in (".", ".."):
        raise ReservationError("Invalid controller resource identifier")
    return parse.quote(value, safe="")


class _NoRedirect(request.HTTPRedirectHandler):
    # Never forward a caller credential to a redirect destination.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ControllerClient:
    def __init__(self, url=None, key=None, timeout=10.0):
        self.url = (url or os.environ.get("IC_URL", DEFAULT_URL)).rstrip("/")
        parsed = parse.urlsplit(self.url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/")):
            raise ReservationError("IC_URL must be an HTTP(S) origin without credentials")
        self._key = key
        self.timeout = timeout
        self._opener = request.build_opener(_NoRedirect())

    def _credential(self):
        key = self._key or os.environ.get("IC_KEY")
        if not key and os.environ.get("IC_KEY_FILE"):
            try:
                key = Path(os.environ["IC_KEY_FILE"]).read_text().strip()
            except OSError:
                raise ReservationError("Cannot read IC_KEY_FILE") from None
        if not key or not key.strip() or "\n" in key or "\r" in key:
            raise ReservationError("Set IC_KEY or IC_KEY_FILE to the structure_net caller key")
        return key.strip()

    def request(self, method, path, body=None, *, authenticated=False):
        headers = {"Accept": "application/json"}
        if authenticated:
            headers["Authorization"] = "Bearer " + self._credential()
        payload = None if body is None else json.dumps(body, allow_nan=False).encode()
        if payload is not None:
            if len(payload) > 65536:
                raise ReservationError("Controller request exceeds 64 KiB")
            headers["Content-Type"] = "application/json"
        req = request.Request(self.url + path, data=payload, headers=headers, method=method)
        try:
            with self._opener.open(req, timeout=self.timeout) as response:
                data = response.read()
                return json.loads(data) if data else {}
        except error.HTTPError as exc:
            # Do not include response bodies, headers, or credentials in errors.
            raise ReservationError(f"Controller {method} {path}: HTTP {exc.code}") from None
        except (OSError, ValueError) as exc:
            raise ReservationError(f"Controller {method} {path} failed ({type(exc).__name__})") from None

    def health(self):
        health = self.request("GET", "/health")
        if not isinstance(health, dict) or health.get("status") != "ok":
            raise ReservationError("Reservation controller is unhealthy")
        return health

    def require_actions(self):
        health = self.health()
        mode = health.get("mode")
        if mode != "enforce":
            raise ReservationError(
                f"Controller {health.get('version', '?')} is {mode or 'unknown'}: "
                "GPU experiments require enforce mode. No GPU work was authorized."
            )
        return health

    def status(self):
        self.health()
        return self.request("GET", "/status")

    def registry(self, model=None):
        return self.request("GET", "/registry" + ("/services/" + segment(model) if model else ""))

    def validate(self, model):
        return self.request("POST", f"/registry/services/{segment(model)}/validate")

    def register(self, model, reason):
        if not reason.strip():
            raise ReservationError("A registration reason is required")
        return self.request("POST", "/registry/services", {
            "name": segment(model), "reason": reason, "grandfather": False,
        }, authenticated=True)

    def remove(self, model, reason):
        if not reason.strip():
            raise ReservationError("A revocation reason is required")
        return self.request("DELETE", f"/registry/services/{segment(model)}",
                            {"reason": reason}, authenticated=True)

    def acquire(self, *, allocation, end, pid, reason, contact):
        allocation = allocation_map(allocation)
        if timestamp(end) <= time.time() or type(pid) is not int or pid <= 0:
            raise ReservationError("A future end and positive host PID are required")
        if not reason.strip() or not contact.strip():
            raise ReservationError("A reason and contact are required")
        self.require_actions()
        return self.request("POST", "/leases", {
            "contact": contact, "reason": reason,
            "end": end, "pid": pid, "allocation": allocation,
        }, authenticated=True)

    def lease(self, lease_id):
        data = self.request("GET", f"/leases/{segment(lease_id)}")
        if isinstance(data, dict):
            data = data.get("lease", data)
        if not isinstance(data, dict) or str(data.get("id")) != lease_id:
            raise ReservationError("Controller returned an unrecognized lease response")
        return data

    def release(self, lease_id):
        # Cleanup must remain possible after controller configuration changes.
        return self.request("DELETE", f"/leases/{segment(lease_id)}", authenticated=True)

    def model_action(self, model, action, *, wait_timeout=300):
        """Refuse unsupported swaps; never bypass the reservation controller."""
        segment(model)
        if action not in ("load", "unload"):
            raise ReservationError("Model action must be load or unload")
        raise ReservationError(
            "The deployed controller has no model swap endpoint. "
            "Use its supported service scheduling workflow; no direct Triton fallback."
        )


def validate_lease(lease, *, allocation=None, pid=None):
    if lease.get("state") != "active" or timestamp(lease.get("end")) <= time.time():
        raise ReservationError("GPU lease is not active or has expired")
    if lease.get("ready") is not True:
        raise ReservationError("GPU lease is active but its memory is not ready")
    if lease.get("holder") != "structure_net":
        raise ReservationError("GPU lease does not belong to the structure_net caller group")
    granted = allocation_map(lease.get("allocation"))
    if allocation is not None and granted != allocation_map(allocation):
        raise ReservationError("Granted GPU allocation differs from the requested allocation")
    if type(lease.get("pid")) is not int or lease["pid"] <= 0:
        raise ReservationError("Lease has no valid host PID")
    if pid is not None and lease["pid"] != pid:
        raise ReservationError("Lease belongs to a different process")
    try:
        stat = Path(f"/proc/{lease['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        start = int(stat[19])
    except (OSError, ValueError, IndexError):
        raise ReservationError("Lease owner is not a live host process") from None
    if type(lease.get("pid_start")) is not int or lease["pid_start"] != start:
        raise ReservationError("Lease owner PID identity does not match the live host process")
    return granted


def _ancestors():
    pid = os.getpid()
    seen = set()
    while pid > 0 and pid not in seen:
        seen.add(pid)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            pid = int(stat[1])
        except (OSError, ValueError, IndexError):
            break
    return seen


def require_gpu_lease(device_ids=None, *, client=None):
    """Verify the supervisor's lease before NAL touches CUDA or dispatches work.

    No environment flag grants a lease: the controller supplies live state and
    the PID must be this process or an ancestor. CPU-only jobs need no lease.
    """
    if device_ids is not None and all(device < 0 for device in device_ids):
        return None
    lease_id = os.environ.get("IC_LEASE")
    legacy_id = os.environ.get("STRUCTURENET_GPU_LEASE")
    if lease_id and legacy_id and lease_id != legacy_id:
        raise ReservationError("Conflicting IC_LEASE and STRUCTURENET_GPU_LEASE identities")
    lease_id = lease_id or legacy_id
    if not lease_id:
        raise ReservationError(
            "Launch GPU work with `inference-ctl lease run ... -- pixi run ...` "
            "or `pixi run gpu-run ... -- <command>`; no IC_LEASE is attached"
        )
    client = client or ControllerClient()
    client.require_actions()
    lease = client.lease(lease_id)
    allocation = validate_lease(lease)
    if lease["pid"] not in _ancestors():
        raise ReservationError("Lease PID is not this process or its supervisor")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not visible or len(set(visible)) != len(visible) or any(gpu not in allocation for gpu in visible):
        raise ReservationError("CUDA visibility must contain only UUIDs in the active lease")
    if device_ids is not None and any(d < 0 or d >= len(visible) for d in device_ids):
        raise ReservationError("Logical CUDA device is outside the active lease")
    return lease
