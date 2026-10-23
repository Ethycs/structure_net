"""Observe the existing label workflow, then prepare expanded TRAIN data once."""
import json
import os
from pathlib import Path
import sys
import time

from experiments.structure_net import ncm_expanded_word_data as data
from experiments.structure_net.ncm_source_hierarchy_supervised import observe_child
from experiments.structure_net.ncm_memory_data import file_hash, write_json

DEST = data.base.ROOT / "20261023_expanded_word_data_workflow_v1"
LABEL_WORKFLOW = data.base.ROOT / "20261023_additional_whole_labels_workflow_v1"


def kernel_identity(pid):
    """PID plus Linux start tick prevents accepting a recycled PID as live."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    if fields[0] == "Z":
        return None
    return {"pid": pid, "start_ticks": int(fields[19])}


def main():
    if DEST.exists() or data.DEST.exists() or data.PILOT.exists():
        raise ValueError("existing observer or data root must not be duplicated")
    upstream = data.labels.prior.q.read(LABEL_WORKFLOW / "producer.json")
    data.base.verify(upstream)
    progress = data.labels.prior.q.read(LABEL_WORKFLOW / "progress.json")
    identity = kernel_identity(progress["observer_pid"])
    if identity is None or (LABEL_WORKFLOW / "failure.json").exists():
        raise ValueError("the existing labeling supervisor must be authoritatively live")
    files = [Path(__file__), Path(data.__file__), data.PLAN,
        Path("experiments/structure_net/ncm_source_hierarchy_supervised.py")]
    producer = data.base.producing(DEST, "observe_existing_labels_then_prepare_expanded_TRAIN_once",
        [LABEL_WORKFLOW / "producer.json"] + files,
        {"label_worker_restarted": False, "upstream_kernel_identity": identity,
            "poll_seconds": 5, "timeout_does_not_restart": True,
            "training": False, "task_accuracy": None})
    for index, path in enumerate(files):
        producer["source_hashes"][str(path)] = file_hash(path)
        (DEST / "producer_source" / f"expanded_workflow_{index}_{path.name}").write_bytes(path.read_bytes())
    producer["fingerprint"] = data.labels.prior.q.digest({k: v for k, v in producer.items() if k != "fingerprint"})
    write_json(DEST / "producer.json", producer)
    journal = {"status": "waiting_for_existing_label_audit", "observer_pid": os.getpid(),
        "upstream_kernel_identity": identity, "producer_fingerprint": producer["fingerprint"]}
    write_json(DEST / "progress.json", journal)
    audit = data.labels.root(False, "audit")
    try:
        journal.update(status="running_target_encoding_pilot")
        write_json(DEST / "progress.json", journal)
        pilot_args = [sys.executable, "-u", "-m", "experiments.structure_net.ncm_expanded_word_data", "pilot"]
        receipt = observe_child(pilot_args, data.PILOT, DEST)
        journal.update(status="waiting_for_existing_label_audit", pilot_exit=receipt["exit_code"])
        write_json(DEST / "progress.json", journal)
        while not (audit / "worker_terminal_receipt.json").exists():
            if (LABEL_WORKFLOW / "failure.json").exists():
                raise RuntimeError("existing label workflow failed; preserve every first candidate")
            if kernel_identity(identity["pid"]) != identity:
                raise RuntimeError("existing label supervisor is terminal without the required audit receipt")
            time.sleep(5)
        data.base.terminal(audit)
        data.base.verify(producer)
        result = data.labels.prior.q.read(audit / "result.json")
        if result["endpoint"]["cases"] != 512 or result["endpoint"]["valid"] != 512:
            raise ValueError("complete512valid final label decisions required")
        journal.update(status="preparing_expanded_TRAIN")
        write_json(DEST / "progress.json", journal)
        args = [sys.executable, "-u", "-m", "experiments.structure_net.ncm_expanded_word_data", "prepare"]
        receipt = observe_child(args, data.DEST, DEST)
        prepared = data.labels.prior.q.read(data.DEST / "result.json")
        if prepared["status"] != "expanded_whole_summary_TRAIN_data_prepared":
            raise ValueError("unexpected expanded data result")
        for name, expected in prepared["files"].items():
            if file_hash(data.DEST / name) != expected:
                raise ValueError("expanded data output hash differs")
        data.base.verify(producer)
        journal.update(status="completed_expanded_TRAIN_preparation_only", preparation_exit=receipt["exit_code"],
            cases=prepared["cases"], task_accuracy=None, student_trained=False,
            final_test_opened=False, speed_tested=False)
        write_json(DEST / "result.json", journal)
        print(json.dumps(journal), flush=True)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(DEST / "failure.json", journal)
        raise


if __name__ == "__main__":
    main()
