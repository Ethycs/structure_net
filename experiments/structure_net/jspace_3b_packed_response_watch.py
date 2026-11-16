"""Serial reservation client for immutable packed-checkpoint evaluation cells.

This CPU process observes existing recovery workers; it never starts/stops them,
changes their checkpoints, or allocates CUDA outside inference-ctl lease run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
MODULE = "experiments.structure_net.jspace_3b_packed_response_evaluate"
PROTOCOL = ROOT / "docs/07 - Status Reports/jspace/2026-11-16_qwen3b-packed-response-evaluation.md"
UPDATES = (0, 46, 90, 152, 228)
ARMS = ("progressive16", "all36")
PIXI = "/home/rabbit/.pixi/bin/pixi"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def completed_saves(campaign_root):
    """The writer opens checkpoints directly: require its post-save event."""
    path = Path(campaign_root) / "train/progress.jsonl"
    completed = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # A concurrently appended final line may be partial.
            if event.get("phase") == "response_development":
                completed.add(event["update"])
    return {update for update in completed if
            (Path(campaign_root) / f"train/update_{update:04d}.pt").is_file()}


def command(phase, root, *, campaign_root=None, checkpoint=None, limit_mib=5376):
    args = [PIXI, "run", "--locked", "--offline", "-e", "ncm", "python", "-m", MODULE,
            phase, "--root", str(root)]
    if phase == "prepare":
        args += ["--campaign-root", str(campaign_root), "--checkpoint", str(checkpoint),
                 "--prefix-mode", "resident"]
    else:
        args += ["--limit-mib", str(limit_mib)]
    return args


def event(root, **values):
    row = dict(utc=datetime.now(timezone.utc).isoformat(), **values)
    with (root / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def completed_result(cell, returncode):
    """Only a successful process with hash-bound completed evidence is usable."""
    cell = Path(cell)
    if returncode:
        return None
    required = [cell / "plan.json", cell / "run/manifest.json", cell / "run/result.json"]
    if not all(path.is_file() for path in required):
        raise ValueError("Successful evaluator is missing its completed evidence")
    plan, manifest, result = [json.loads(path.read_text()) for path in required]
    if (manifest.get("status") != "completed" or manifest.get("fingerprint") != plan["fingerprint"]
            or result.get("fingerprint") != plan["fingerprint"]
            or result.get("checkpoint_sha256") != plan["checkpoint_sha256"]
            or result.get("update") != plan["checkpoint_metadata"]["update"]):
        raise ValueError("Completed packed result has inconsistent provenance")
    for name, expected in manifest["artifacts"].items():
        if Path(name).name != name or digest(cell / "run" / name) != expected:
            raise ValueError("Completed packed evaluation artifact changed")
    if "result.json" not in manifest["artifacts"]:
        raise ValueError("Completed result is not authenticated")
    return result


def select_results(outcomes):
    from .jspace_3b_response_campaign import select_checkpoint
    selection = {}
    for arm in ARMS:
        candidates = []
        for (candidate_arm, _), outcome in outcomes.items():
            if candidate_arm == arm and "cell" in outcome:
                result = completed_result(outcome["cell"], outcome["returncode"])
                if result is not None:
                    candidates.append(result)
        if not any(row["update"] == 0 for row in candidates):
            selection[arm] = dict(status="incomplete_initializer_evidence", confirmed=False,
                                  selected=None, completed_candidates=len(candidates))
        else:
            selection[arm] = select_checkpoint(candidates)
    return selection


def run(root, campaign_base, *, gpu="GPU-8cf440df", allocation_mib=5632,
        limit_mib=5376, minutes=90, deadline_hours=20, poll_seconds=30):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("The observer must start with CUDA hidden")
    if not (0 < limit_mib < allocation_mib and 1 <= poll_seconds <= 60):
        raise ValueError("Require bounded polling and allocator headroom")
    root, campaign_base = Path(root).resolve(), Path(campaign_base).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "watcher.json").exists():
        raise FileExistsError("Observer runs are append-only; use a new root")
    sources = [Path(__file__), PROTOCOL, ROOT / "experiments/structure_net/jspace_3b_packed_response_model.py",
               ROOT / "experiments/structure_net/jspace_3b_packed_response_evaluate.py"]
    plans = {arm: campaign_base / arm / "plan.json" for arm in ARMS}
    pins = {str(path): digest(path) for path in [*sources, *plans.values()]}
    write(root / "watcher.json", dict(schema="jspace.packed-response-observer.v1", pins=pins,
        campaign_base=str(campaign_base), updates=UPDATES, arms=ARMS, gpu=gpu,
        allocation_mib=allocation_mib, limit_mib=limit_mib, lease_minutes=minutes,
        deadline_hours=deadline_hours, deadline_scope="Stop admitting new cells after this deadline; current lease remains bounded",
        prefix_mode="resident", confirmed=False))
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
               PIXI_CACHE_DIR="/tmp/structure-net-pixi-cache", UV_CACHE_DIR="/tmp/structure-net-uv-cache")
    started = time.monotonic()
    outcomes = {}
    try:
        while time.monotonic() - started < deadline_hours * 3600:
            if any(digest(path) != pinned for path, pinned in pins.items()):
                raise ValueError("Pinned observer/evaluator/protocol/parent plan changed")
            ready = {arm: completed_saves(campaign_base / arm) for arm in ARMS}
            pending = [(arm, update) for update in UPDATES for arm in ARMS
                       if (arm, update) not in outcomes and update in ready[arm]]
            if not pending:
                terminal = all((campaign_base / arm / "train/result.json").exists() or
                               (campaign_base / arm / "train/failure.json").exists() for arm in ARMS)
                if len(outcomes) == len(ARMS) * len(UPDATES) or terminal:
                    break
                time.sleep(poll_seconds)
                continue
            arm, update = pending[0]
            cell = root / arm / f"update_{update:04d}_resident"
            cell.parent.mkdir(exist_ok=True)
            checkpoint = campaign_base / arm / f"train/update_{update:04d}.pt"
            event(root, phase="prepare", arm=arm, update=update, checkpoint=str(checkpoint))
            # The evaluator's prepare freezes its entire source closure and inputs.
            prep = subprocess.run(command("prepare", cell, campaign_root=campaign_base / arm,
                checkpoint=checkpoint), cwd=ROOT, env=env, text=True, capture_output=True)
            (cell.parent / f"prepare_{update:04d}.log").write_text(prep.stdout + prep.stderr)
            if prep.returncode:
                outcomes[arm, update] = dict(status="preparation_failed", returncode=prep.returncode)
                event(root, phase="preparation_failed", arm=arm, update=update, returncode=prep.returncode)
                continue
            # No CUDA in this observer; inference-ctl sets the child's visibility.
            launch = ["/usr/bin/python3", "/usr/local/bin/inference-ctl", "lease", "run",
                      "--allocation", f"{gpu}={allocation_mib}", "--minutes", str(minutes),
                      "--reason", f"Packed response DEV {arm} update{update}; training unchanged", "--",
                      *command("run", cell, limit_mib=limit_mib)]
            log_path = cell.parent / f"run_{update:04d}.log"
            event(root, phase="request_reservation", arm=arm, update=update, command=launch)
            # Reservation denial is terminal for this cell. Never bypass admission,
            # overwrite a partial run, or silently retry resource/quality failures.
            with log_path.open("x") as log:
                process = subprocess.run(launch, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            result = completed_result(cell, process.returncode) or {}
            outcomes[arm, update] = dict(status=result.get("status", "execution_failed"),
                returncode=process.returncode, cell=str(cell))
            event(root, phase="cell_finished", arm=arm, update=update, **outcomes[arm, update])
        missing = [dict(arm=arm, update=update) for arm in ARMS for update in UPDATES
                   if (arm, update) not in outcomes]
        write(root / "result.json", dict(status="completed" if not missing else "incomplete",
            cells=[dict(arm=arm, update=update, **value) for (arm, update), value in outcomes.items()],
            missing=missing, selections=select_results(outcomes),
            elapsed_seconds=time.monotonic()-started, confirmed=False))
    except BaseException as error:
        write(root / "failure.json", dict(status="observer_failed", error=repr(error),
            elapsed_seconds=time.monotonic()-started, training_unchanged=True))
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign-base", type=Path, required=True)
    parser.add_argument("--deadline-hours", type=float, default=20)
    args = parser.parse_args()
    run(args.root, args.campaign_base, deadline_hours=args.deadline_hours)


if __name__ == "__main__":
    main()
