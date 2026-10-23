"""Archive development outcomes and conditionally execute the sealed final test."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.structure_net import ncm_sealed_final_quality as final
from experiments.structure_net import ncm_generated_history_workflow as word_workflow
from experiments.neural_architecture_lab import finalize_ncm_sealed_final_quality as audit
from experiments.structure_net.ncm_expanded_word_data_workflow import kernel_identity
from experiments.structure_net.ncm_memory_data import file_hash, write_json

DEST = final.ROOT / "20261023_sealed_final_workflow_v1"


def run_stage(name, args, destination, journal, producer, *, existing=False, model_worker=True, cpu=False):
    if destination.exists() != existing:
        raise ValueError("stage destination freshness differs: " + name)
    final.base.verify(producer)
    receipt_path = DEST / "stage_receipts" / (name+".json")
    if receipt_path.exists():
        raise ValueError("an observed stage must never be repeated")
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
        target = destination / ("registry_terminal_receipt.json" if name.endswith("_register") else "worker_terminal_receipt.json")
        if target.exists():
            raise ValueError("existing terminal evidence must not be overwritten")
        write_json(target, receipt)
    journal["stages"].append(receipt)
    write_json(DEST / "progress.json", journal)
    if code != 0:
        raise RuntimeError("stage failed with observed exit: " + name)
    if model_worker:
        final.base.terminal(destination)
    final.base.verify(producer)


def main():
    stage_roots = [final.DATA, final.FEATURE_PILOT, final.FEATURES, final.PILOT,
        final.OUTPUTS, final.PACKETS, final.GRADE, audit.DEST]
    if DEST.exists() or any(root.exists() for root in stage_roots):
        raise ValueError("fresh conditional final workflow and stage roots required")
    word_progress = final.q.read(word_workflow.DEST / "progress.json")
    identity = kernel_identity(word_progress["observer_pid"])
    ready = lambda: (word_workflow.DEST / "result.json").exists()
    if not ready() and identity is None:
        raise ValueError("unfinished original word workflow must be authoritatively live")
    files = [Path(__file__), final.PLAN, Path(final.__file__), Path(audit.__file__),
        Path(final.development.__file__), final.development.PLAN, Path(word_workflow.__file__)]
    p = final._base_producing(DEST, "wait_for_existing_development_then_conditionally_evaluate_sealed_sources",
        [word_workflow.DEST / "producer.json"]+files,
        {"upstream_kernel_identity": identity, "existing_word_worker_restarted": False,
            "sealed_source_files_read_at_observer_creation": False, "conditional_final_evaluation": True,
            "joint_minimum": {k: final.minimum(n) for k, n in final.COUNTS.items()},
            "final_test_opened": False, "task_accuracy": None, "speed_tested": False})
    p["hypothesis_id"] = final.HYPOTHESIS
    for index, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (DEST / "producer_source" / f"sealed_workflow_{index}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = final.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(DEST / "producer.json", p)
    (DEST / "stage_receipts").mkdir()
    journal = {"status": "waiting_for_existing_generated_history_development", "observer_pid": os.getpid(),
        "upstream_kernel_identity": identity, "stages": [], "final_test_opened": False,
        "producer_fingerprint": p["fingerprint"]}
    write_json(DEST / "progress.json", journal)
    try:
        word_workflow.wait_for_existing(word_workflow.DEST, identity, ready)
        upstream = final.q.read(word_workflow.DEST / "result.json")
        if upstream["status"] != "completed_generated_history_development_candidate":
            raise ValueError("complete original word development workflow required")
        archive = final.development.DEST
        receipt_path = archive / (final.study.HYPOTHESIS+"_registry_readback.json")
        if not archive.exists():
            args = [sys.executable, "-u", "-m", "experiments.neural_architecture_lab.finalize_ncm_generated_history_word", "finalize"]
            run_stage("development_finalize", args, archive, journal, p, model_worker=False, cpu=True)
        if not receipt_path.exists():
            if final.q.read(archive / "worker_terminal_receipt.json")["exit_code"] != 0:
                raise ValueError("actual development finalizer zero exit required")
            args = [sys.executable, "-u", "-m", "experiments.neural_architecture_lab.finalize_ncm_generated_history_word", "register"]
            run_stage("development_register", args, archive, journal, p,
                existing=True, model_worker=False, cpu=True)
        receipt = final.q.read(receipt_path)
        if receipt["status"] != "registered_and_read_back" or receipt["record"]["campaign_sha256"] != file_hash(archive / "campaign_results.json"):
            raise ValueError("exact registered development outcome required")
        development = final.q.read(archive / "campaign_results.json")
        if (development["classification"] != "generated_history_development_passed_final_quality_unproven"
                or not development["development"]["development_gate"]):
            journal.update(status="development_not_qualified_final_sources_unopened", active_stage=None,
                final_test_opened=False, goal_target_met=False,
                development_classification=development["classification"])
            write_json(DEST / "result.json", journal)
            print(json.dumps(journal), flush=True)
            return
        final.require_admission()
        journal.update(status="admitted_for_sealed_final_evaluation", final_source_access_attempted=True,
            final_test_opened=None)
        write_json(DEST / "progress.json", journal)
        stages = [("prepare", final.DATA), ("feature-pilot", final.FEATURE_PILOT),
            ("features", final.FEATURES), ("pilot", final.PILOT), ("generate", final.OUTPUTS),
            ("packets", final.PACKETS), ("grade", final.GRADE)]
        for stage, destination in stages:
            args = [sys.executable, "-u", "-m", "experiments.structure_net.ncm_sealed_final_quality", stage]
            run_stage(stage, args, destination, journal, p, cpu=stage in ("prepare", "packets", "grade"))
            if stage == "prepare":
                journal["final_test_opened"] = True
                write_json(DEST / "progress.json", journal)
        grade = final.q.read(final.GRADE / "result.json")
        complete = {**journal, "status": "completed_sealed_final_evaluation", "active_stage": None,
            "final_test_opened": True, "quality_gate_pending_audit": grade["quality_gate"]}
        write_json(DEST / "evaluation_complete.json", complete)
        for stage in ("finalize", "register"):
            args = [sys.executable, "-u", "-m", "experiments.neural_architecture_lab.finalize_ncm_sealed_final_quality", stage]
            run_stage("final_"+stage, args, audit.DEST, journal, p,
                existing=stage == "register", model_worker=False, cpu=True)
        final_receipt = final.q.read(audit.DEST / (final.HYPOTHESIS+"_registry_readback.json"))
        aggregate = final.q.read(audit.DEST / "campaign_results.json")
        if (final_receipt["status"] != "registered_and_read_back"
                or final_receipt["record"]["campaign_sha256"] != file_hash(audit.DEST / "campaign_results.json")
                or final_receipt["record"]["final_test_opened"] is not True
                or final_receipt["record"]["goal_target_met"] != aggregate["goal_target_met"]):
            raise ValueError("final target/metadata registry readback differs")
        journal.update(status="completed_audited_sealed_final_quality", active_stage=None,
            final_test_opened=True, goal_target_met=aggregate["goal_target_met"],
            final_classification=aggregate["classification"], speed_tested=False)
        write_json(DEST / "result.json", journal)
        print(json.dumps(journal), flush=True)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(DEST / "failure.json", journal)
        raise


if __name__ == "__main__":
    argparse.ArgumentParser(__doc__).parse_args()
    main()
