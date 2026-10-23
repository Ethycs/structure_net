"""Repeat a terminal-unobserved systems pilot, then supervise fixed NCM training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.structure_net import ncm_source_hierarchy_bootstrap as worker
from experiments.structure_net.ncm_memory_data import file_hash, write_json

ROOT = worker.study.data.ROOT
ORIGINAL = worker.PILOT
PILOT = ROOT / "20261023_source_hierarchy_bootstrap_pilot_terminal_v2"
TRAIN = ROOT / "20261023_source_hierarchy_bootstrap_train_terminal_v2"
WORKFLOW = ROOT / "20261023_source_hierarchy_bootstrap_workflow_v2"
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-hierarchy-terminal-recovery-plan.md")


def configure():
    worker.PILOT, worker.TRAIN = PILOT, TRAIN
    original_producing = worker.producing

    def producing(root, pilot, rows, manifest):
        producer = original_producing(root, pilot, rows, manifest)
        for index, path in enumerate((Path(__file__), PLAN)):
            producer["source_hashes"][str(path)] = file_hash(path)
            (root / "producer_source" / f"terminal_recovery_{index}_{path.name}").write_bytes(path.read_bytes())
        producer["config"].update(terminal_observer="durable_parent_subprocess_wait",
            original_pilot_exit_unobserved=True, scientific_config_changed=False)
        producer["fingerprint"] = worker.study.q.digest({k: v for k, v in producer.items() if k != "fingerprint"})
        write_json(root / "producer.json", producer)
        return producer

    worker.producing = producing


def observe_child(args, destination, observer_root):
    """Never infer an exit from a result file or overwrite an existing worker."""
    if destination.exists():
        raise ValueError("existing worker root must not be restarted: " + str(destination))
    child = subprocess.Popen(args)
    write_json(observer_root / "active_child.json", {
        "pid": child.pid, "observer_pid": os.getpid(), "command": args,
        "destination": str(destination), "status": "running"})
    code = child.wait()
    receipt = {"status": "observed_worker_terminal", "child_pid": child.pid,
        "observer_pid": os.getpid(), "observer": "subprocess.wait", "exit_code": code}
    if destination.exists():
        path = destination / "worker_terminal_receipt.json"
        if path.exists():
            raise ValueError("child terminal receipt already exists")
        write_json(path, receipt)
    write_json(observer_root / "active_child.json", {**receipt, "command": args,
        "destination": str(destination)})
    if code != 0:
        raise RuntimeError(f"worker exited {code}: {destination}")
    worker.study.data.terminal(destination)
    return receipt


def workflow():
    if PILOT.exists() or TRAIN.exists() or WORKFLOW.exists():
        raise ValueError("fresh supervised roots required; inspect existing workers first")
    worker.study.data.verify(worker.study.q.read(ORIGINAL / "producer.json"))
    prior = worker.study.q.read(ORIGINAL / "result.json")
    if (prior["status"] != "systems_source_hierarchy_bootstrap_passed"
            or prior["checkpoint_sha256"] != file_hash(ORIGINAL / "model.pt")):
        raise ValueError("unchanged original pilot artifacts required")
    if (ORIGINAL / "worker_terminal_receipt.json").exists():
        raise ValueError("reconcile existing original receipt instead of repeating a pilot")
    files = [Path(__file__), PLAN, Path(worker.__file__), Path(worker.core.__file__), worker.PLAN]
    inputs = [ORIGINAL / name for name in ("producer.json", "result.json", "trace.jsonl", "source_only_reload_probe.json")]
    producer = worker.study.producing(WORKFLOW, "durable_source_hierarchy_training_workflow",
        inputs + files, {"scientific_config_changed": False,
            "original_pilot_exit_code": None, "original_pilot_repeated_for_terminal_observation": True,
            "pilot_updates": 2, "training_updates": worker.Config().updates,
            "task_accuracy": None, "speed_tested": False})
    for index, path in enumerate(files):
        producer["source_hashes"][str(path)] = file_hash(path)
        (WORKFLOW / "producer_source" / f"supervisor_{index}_{path.name}").write_bytes(path.read_bytes())
    producer["fingerprint"] = worker.study.q.digest({k: v for k, v in producer.items() if k != "fingerprint"})
    write_json(WORKFLOW / "producer.json", producer)
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [],
        "original_pilot_exit_code": None, "producer_fingerprint": producer["fingerprint"]}
    write_json(WORKFLOW / "progress.json", journal)
    try:
        for stage, destination in (("pilot", PILOT), ("train", TRAIN)):
            worker.study.data.verify(producer)
            if stage == "train":
                result = worker.study.q.read(PILOT / "result.json")
                if result["status"] != "systems_source_hierarchy_bootstrap_passed":
                    raise ValueError("repeated systems pilot failed; training not admitted")
                if result["initial_parameter_sha256"] != prior["initial_parameter_sha256"]:
                    raise ValueError("repeated pilot must start from the identical registered checkpoint")
            journal.update(active_stage=stage)
            write_json(WORKFLOW / "progress.json", journal)
            args = [sys.executable, "-u", "-m", "experiments.structure_net.ncm_source_hierarchy_supervised", stage]
            receipt = observe_child(args, destination, WORKFLOW)
            journal["stages"].append({"stage": stage, "root": str(destination), **receipt})
            write_json(WORKFLOW / "progress.json", journal)
        worker.study.data.verify(producer)
        journal.update(status="completed_concept_bootstrap_only", active_stage=None,
            task_accuracy=None, NCM_success_established=False, speed_tested=False)
        write_json(WORKFLOW / "result.json", journal)
        print(json.dumps(journal), flush=True)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["workflow", "pilot", "train"])
    stage = parser.parse_args().stage
    if stage == "workflow":
        workflow()
    else:
        configure()
        worker.train(pilot=stage == "pilot")
