"""Supervise reviewed-draft data preparation and the fixed NCM word experiment."""
import argparse
import json
import os
from pathlib import Path

from experiments.structure_net import ncm_reviewed_draft_recovery as recovery
from experiments.structure_net.ncm_expanded_word_data_workflow import kernel_identity

data = recovery.data
study = recovery.study
evaluation = recovery.evaluation
OLD_WORKFLOWS = [study.ROOT / n for n in (
    "20261023_additional_whole_labels_workflow_v1", "20261023_expanded_word_data_workflow_v1",
    "20261023_generated_history_workflow_v1", "20261023_sealed_final_workflow_v1")]


def old_workers_terminal():
    for root in OLD_WORKFLOWS:
        failure = study.q.read(root / "failure.json")
        if failure["status"] != "stopped_failure_preserved" or kernel_identity(failure["observer_pid"]) is not None:
            raise ValueError("previous dependent workflow must have stopped with failure preserved")
    receipt = study.q.read(data.labels.root(False, "revision") / "worker_terminal_receipt.json")
    if receipt["exit_code"] != -15 or kernel_identity(receipt["child_pid"]) is not None:
        raise ValueError("interrupted teacher worker must be authoritatively terminal")


def main():
    recovery.configure()
    old_workers_terminal()
    destination = recovery.WORD_WORKFLOW
    stages = [("target_pilot", "data", "pilot", data.PILOT, True),
        ("target_data", "data", "prepare", data.DEST, True),
        ("cache_pilot", "study", "cache-pilot", study.CACHE_PILOT, False),
        ("cache", "study", "cache", study.CACHE, False),
        ("train_pilot", "study", "pilot", study.PILOT, False),
        ("train", "study", "train", study.TRAIN, False),
        ("inference_pilot", "eval", "pilot", evaluation.PILOT, False),
        ("outputs", "eval", "generate", evaluation.OUTPUTS, False),
        ("packets", "eval", "packets", evaluation.PACKETS, True),
        ("grade", "eval", "grade", evaluation.GRADE, True)]
    if destination.exists() or any(root.exists() for _, _, _, root, _ in stages):
        raise ValueError("fresh reviewed-draft workflow and all stage roots required")
    study.require_forecaster()
    files = [Path(__file__), Path(recovery.__file__), Path(data.__file__), recovery.PLAN,
        Path(study.__file__), Path(evaluation.__file__), Path(study.core.__file__),
        Path(recovery.word_workflow.__file__)]
    inputs = files + [root / "failure.json" for root in OLD_WORKFLOWS]
    inputs += [data.ABORT / "decision_before_signal.json", data.ABORT / "observation.json"]
    inputs += [data.labels.root(False, stage) / "producer.json" for stage in ("draft", "critique")]
    inputs += [study.archive.DEST / "campaign_results.json", study.archive.DEST / (study.archive.worker.HYPOTHESIS+"_registry_readback.json")]
    producer = study.data.producing(destination, "reviewed_draft_recovery_data_and_generated_history_word_workflow", inputs,
        {"word_config": study.asdict(study.Config()), "teacher_calls": 0,
            "old_failures_preserved": True, "label_policy": "all_first_drafts_passing_first_critique_and_capacity",
            "semantic_rules_unchanged": True, "task_accuracy": None, "final_test_opened": False})
    producer["hypothesis_id"] = study.HYPOTHESIS
    producer["config"]["hypothesis_id"] = study.HYPOTHESIS
    producer = data.capture(destination, producer)
    for i, path in enumerate(files):
        producer["source_hashes"][str(path)] = study.file_hash(path)
        (destination / "producer_source" / f"reviewed_word_workflow_{i}_{path.name}").write_bytes(path.read_bytes())
    producer["fingerprint"] = study.q.digest({k: v for k, v in producer.items() if k != "fingerprint"})
    study.write_json(destination / "producer.json", producer)
    (destination / "stage_receipts").mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [],
        "producer_fingerprint": producer["fingerprint"], "final_test_opened": False}
    study.write_json(destination / "progress.json", journal)
    try:
        for name, scope, stage, root, cpu in stages:
            recovery.word_workflow.run_stage(name, recovery.command(scope, stage), root, journal, producer, cpu=cpu)
        endpoint = study.q.read(evaluation.GRADE / "result.json")
        journal.update(status="completed_generated_history_development_candidate", active_stage=None,
            development_gate=endpoint["development_gate"], development_results=endpoint["results"],
            final_test_opened=False, NCM_success_established=False, speed_tested=False)
        study.write_json(destination / "result.json", journal)
        print(json.dumps(journal), flush=True)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        study.write_json(destination / "failure.json", journal)
        raise


if __name__ == "__main__":
    argparse.ArgumentParser(__doc__).parse_args()
    main()
