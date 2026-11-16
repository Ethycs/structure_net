"""Stop only the superseded progressive worker after its saved/evaluated boundary."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def identity(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if raw[0] == "Z":
            return None
        return dict(start_ticks=raw[19], command=Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode())
    except FileNotFoundError:
        return None


def run(root, campaign, pid, update):
    root, campaign = Path(root).resolve(), Path(campaign).resolve()
    root.mkdir(parents=True, exist_ok=False)
    admitted = identity(pid)
    if (admitted is None or "jspace_3b_response_campaign run" not in admitted["command"]
            or str(campaign) not in admitted["command"] or campaign.name != "progressive16"):
        raise ValueError("Only the explicitly named progressive response worker is admitted")
    def write(name, row):
        with (root/name).open("x") as stream:
            json.dump(row, stream, indent=2, sort_keys=True)
    write("plan.json", dict(pid=pid, process=admitted, campaign=str(campaign), boundary=update,
        reason="User authorized native-first broad-corpus progressive rebuild; all36 remains running",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), signal="SIGINT"))
    started = time.monotonic()
    while time.monotonic()-started < 7200:
        current = identity(pid)
        if current is None:
            write("result.json", dict(status="worker_already_exited", signal_sent=False))
            return
        if current != admitted:
            raise RuntimeError("Target process identity changed; refusing signal")
        events=[]
        for line in (campaign/"train/progress.jsonl").read_text().splitlines():
            try: events.append(json.loads(line))
            except json.JSONDecodeError: pass
        evaluated = any(r.get("phase")=="evaluation_complete" and r.get("update")==update for r in events)
        resumed = any(r.get("phase")=="training" and r.get("update",0)>update for r in events)
        checkpoint = campaign/f"train/update_{update:04d}.pt"
        if evaluated and resumed and checkpoint.is_file():
            with checkpoint.open("rb") as stream:
                digest=hashlib.file_digest(stream,"sha256").hexdigest()
            if identity(pid) != admitted:
                raise RuntimeError("Target identity changed before signal")
            write("signal.json", dict(utc=datetime.now(timezone.utc).isoformat(), pid=pid,
                signal="SIGINT", checkpoint=str(checkpoint), checkpoint_sha256=digest,
                completed_evaluation=str(campaign/f"train/evaluation_{update:04d}/result.json"),
                all36_untouched=True))
            os.kill(pid, signal.SIGINT)
            print(json.dumps(dict(phase="sent_interrupt", pid=pid, preserved_update=update)), flush=True)
            deadline=time.monotonic()+900
            while identity(pid)==admitted and time.monotonic()<deadline:
                time.sleep(2)
            finished=identity(pid) is None
            write("result.json", dict(status="worker_exited" if finished else "cleanup_pending",
                signal_sent=True, checkpoint_preserved=True, all36_untouched=True,
                elapsed_seconds=time.monotonic()-started))
            print(json.dumps(dict(phase="transition_finished", worker_exited=finished)), flush=True)
            return
        time.sleep(20)
    write("result.json", dict(status="boundary_wait_expired", signal_sent=False, all36_untouched=True))


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--update", type=int, default=90)
    args=parser.parse_args()
    run(args.root,args.campaign,args.pid,args.update)
