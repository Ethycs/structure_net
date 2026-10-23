"""Observe existing prerequisites, then run fixed generated-history task training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.structure_net import ncm_generated_history_study as study
from experiments.structure_net import ncm_generated_history_eval as evaluation
from experiments.structure_net.ncm_expanded_word_data_workflow import kernel_identity
from experiments.structure_net.ncm_memory_data import file_hash, write_json

DEST = study.ROOT / "20261023_generated_history_workflow_v1"
DATA_WORKFLOW = study.ROOT / "20261023_expanded_word_data_workflow_v1"


def wait_for_existing(root, identity, ready):
    """A live process or a terminal receipt is required; silence never restarts it."""
    while not ready():
        if (root / "failure.json").exists():
            raise RuntimeError("upstream failure preserved: " + str(root))
        if identity is None or kernel_identity(identity["pid"]) != identity:
            raise RuntimeError("upstream terminal without required successful artifacts: " + str(root))
        time.sleep(5)


def run_stage(name, args, destination, journal, producer, *, existing=False, model_worker=True, cpu=False):
    if destination.exists() != existing:
        raise ValueError("stage destination freshness differs: " + name)
    study.data.verify(producer)
    receipt_path = DEST / "stage_receipts" / (name+".json")
    if receipt_path.exists():
        raise ValueError("stage must never be repeated: " + name)
    env = dict(os.environ)
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    child = subprocess.Popen(args, env=env)
    journal.update(status="running", active_stage=name, child_pid=child.pid)
    write_json(DEST / "progress.json", journal)
    code = child.wait()
    receipt = {"status": "observed_worker_terminal", "stage": name, "child_pid": child.pid,
        "observer_pid": os.getpid(), "observer": "subprocess.wait", "exit_code": code,
        "command": args, "destination": str(destination)}
    write_json(receipt_path, receipt)
    if destination.exists():
        filename = "registry_terminal_receipt.json" if name == "bootstrap_register" else "worker_terminal_receipt.json"
        target = destination / filename
        if target.exists():
            raise ValueError("terminal receipt must not be overwritten: " + str(target))
        write_json(target, receipt)
    journal["stages"].append(receipt)
    write_json(DEST / "progress.json", journal)
    if code != 0:
        raise RuntimeError("stage failed with observed exit: " + name)
    if model_worker:
        study.data.terminal(destination)
    study.data.verify(producer)


def main():
    destinations = [study.CACHE_PILOT, study.CACHE, study.PILOT, study.TRAIN,
        evaluation.PILOT, evaluation.OUTPUTS, evaluation.PACKETS, evaluation.GRADE]
    if DEST.exists() or any(path.exists() for path in destinations):
        raise ValueError("fresh workflow and generated-history stage roots required")
    bootstrap_progress = study.q.read(study.bootstrap.WORKFLOW / "progress.json")
    data_progress = study.q.read(DATA_WORKFLOW / "progress.json")
    identities = {"bootstrap": kernel_identity(bootstrap_progress["observer_pid"]),
        "expanded_data": kernel_identity(data_progress["observer_pid"])}
    bootstrap_ready = lambda: (study.bootstrap.TRAIN / "worker_terminal_receipt.json").exists() and (study.bootstrap.WORKFLOW / "result.json").exists()
    data_ready = lambda: (study.expanded.DEST / "worker_terminal_receipt.json").exists()
    if ((not bootstrap_ready() and identities["bootstrap"] is None)
            or (not data_ready() and identities["expanded_data"] is None)):
        raise ValueError("unfinished prerequisite supervisors must be authoritatively live")
    files = [Path(__file__), Path(study.__file__), Path(evaluation.__file__), Path(study.core.__file__),
        study.PLAN, Path(study.archive.__file__), study.archive.PLAN]
    inputs = [study.bootstrap.WORKFLOW / "producer.json", DATA_WORKFLOW / "producer.json"]
    producer = study.data.producing(DEST, "observe_existing_prerequisites_and_supervise_generated_history_candidate",
        inputs+files, {"word_config": study.asdict(study.Config()), "prerequisite_kernel_identities": identities,
            "upstream_workers_restarted": False, "poll_seconds": 5, "task_accuracy": None,
            "development_only": True, "final_test_opened": False, "speed_tested": False})
    producer["hypothesis_id"] = study.HYPOTHESIS
    for index, path in enumerate(files):
        producer["source_hashes"][str(path)] = file_hash(path)
        (DEST / "producer_source" / f"generated_workflow_{index}_{path.name}").write_bytes(path.read_bytes())
    producer["fingerprint"] = study.q.digest({k: v for k, v in producer.items() if k != "fingerprint"})
    write_json(DEST / "producer.json", producer)
    (DEST / "stage_receipts").mkdir()
    journal = {"status": "waiting_for_existing_concept_bootstrap", "observer_pid": os.getpid(),
        "identities": identities, "stages": [], "producer_fingerprint": producer["fingerprint"]}
    write_json(DEST / "progress.json", journal)
    try:
        wait_for_existing(study.bootstrap.WORKFLOW, identities["bootstrap"], bootstrap_ready)
        study.data.terminal(study.bootstrap.TRAIN)
        bootstrap_result = study.q.read(study.bootstrap.WORKFLOW / "result.json")
        if bootstrap_result["status"] != "completed_concept_bootstrap_only":
            raise ValueError("unchanged complete bootstrap supervisor result required")
        if not study.archive.DEST.exists():
            for stage in ("finalize", "register"):
                args = [sys.executable, "-u", "-m", "experiments.neural_architecture_lab.finalize_ncm_source_hierarchy_bootstrap", stage]
                run_stage("bootstrap_"+stage, args, study.archive.DEST, journal, producer,
                    existing=stage == "register", model_worker=False, cpu=True)
        study.require_forecaster()
        journal.update(status="waiting_for_existing_expanded_TRAIN_data", active_stage=None)
        write_json(DEST / "progress.json", journal)
        wait_for_existing(DATA_WORKFLOW, identities["expanded_data"], data_ready)
        study.require_data()
        stages = [("cache_pilot", "experiments.structure_net.ncm_generated_history_study", "cache-pilot", study.CACHE_PILOT),
            ("cache", "experiments.structure_net.ncm_generated_history_study", "cache", study.CACHE),
            ("train_pilot", "experiments.structure_net.ncm_generated_history_study", "pilot", study.PILOT),
            ("train", "experiments.structure_net.ncm_generated_history_study", "train", study.TRAIN),
            ("inference_pilot", "experiments.structure_net.ncm_generated_history_eval", "pilot", evaluation.PILOT),
            ("outputs", "experiments.structure_net.ncm_generated_history_eval", "generate", evaluation.OUTPUTS),
            ("packets", "experiments.structure_net.ncm_generated_history_eval", "packets", evaluation.PACKETS),
            ("grade", "experiments.structure_net.ncm_generated_history_eval", "grade", evaluation.GRADE)]
        for name, module, stage, destination in stages:
            args = [sys.executable, "-u", "-m", module, stage]
            run_stage(name, args, destination, journal, producer, cpu=name in ("packets", "grade"))
        endpoint = study.q.read(evaluation.GRADE / "result.json")
        journal.update(status="completed_generated_history_development_candidate", active_stage=None,
            development_gate=endpoint["development_gate"], development_results=endpoint["results"],
            final_test_opened=False, NCM_success_established=False, speed_tested=False)
        write_json(DEST / "result.json", journal)
        print(json.dumps(journal), flush=True)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(DEST / "failure.json", journal)
        raise


if __name__ == "__main__":
    argparse.ArgumentParser(__doc__).parse_args()
    main()
