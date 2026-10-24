"""Recover the completed word audit's missing dependency without rerunning a model."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.structure_net import ncm_reviewed_draft_recovery as recovery
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_native_label_quality import dependencies

ROOT = recovery.ROOT
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-reviewed-draft-audit-recovery-plan.md")
OLD = recovery.FINAL_WORKFLOW
NOMINAL_ARCHIVE = ROOT / "20261023_reviewed_draft_history_results_v1"
DEST = ROOT / "20261023_reviewed_draft_history_results_terminal_v2"
WORKFLOW = ROOT / "20261023_reviewed_draft_history_audit_recovery_workflow_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-reviewed-draft-history-word-audit-recovery.md")
MODULE = "experiments.structure_net.ncm_reviewed_draft_audit_recovery"


def require_original_failure():
    receipt = json.loads((OLD / "stage_receipts/development_finalize.json").read_text())
    failed = json.loads((OLD / "failure.json").read_text())
    log = OLD.with_suffix(".worker.log").read_text()
    if (receipt.get("stage") != "development_finalize" or receipt.get("exit_code") != 1
            or receipt.get("observer") != "subprocess.wait"
            or failed.get("status") != "stopped_failure_preserved"
            or "ModuleNotFoundError: No module named 'tiktoken'" not in log
            or NOMINAL_ARCHIVE.exists()):
        raise ValueError("the original missing-tiktoken audit failure and absent archive are required")
    return receipt


def dependency_preflight():
    # The unchanged grader calls this same dependency initializer before tiktoken.
    dependencies()
    module = importlib.import_module("tiktoken")
    return {"tiktoken_file": str(module.__file__),
            "tiktoken_version": getattr(module, "__version__", None)}


def configure():
    recovery.configure()
    audit = recovery.development
    audit.DEST, audit.REPORT, audit.PLAN = DEST, REPORT, PLAN
    return audit


def execute(stage):
    if stage not in ("finalize", "register"):
        raise ValueError("only unchanged finalization and registration are admitted")
    require_original_failure()
    audit = configure()
    dependency_preflight()
    getattr(audit, stage)()


def main_workflow():
    if WORKFLOW.exists() or DEST.exists() or REPORT.exists():
        raise ValueError("fresh recovery workflow, archive and report required")
    original = require_original_failure()
    audit = configure()
    dependency = dependency_preflight()
    study = recovery.study
    sources = [Path(__file__), PLAN, Path(audit.__file__), Path(recovery.__file__),
               Path(dependencies.__code__.co_filename)]
    inputs = sources + [OLD / "failure.json", OLD / "stage_receipts/development_finalize.json",
        OLD.with_suffix(".worker.log"), study.TRAIN / "result.json",
        recovery.evaluation.GRADE / "result.json", recovery.WORD_WORKFLOW / "result.json"]
    producer = study.data.freeze(WORKFLOW, "recover_completed_word_audit_dependency", inputs,
        {"dependency": dependency, "original_failure_receipt": original,
         "audit_destination": str(DEST), "audit_report": str(REPORT),
         "unchanged_auditor": str(audit.__file__), "model_updates": 0,
         "new_model_outputs": 0, "new_judge_requests": 0,
         "scientific_rules_changed": False, "final_test_opened": False})
    for index, path in enumerate(sources):
        producer["source_hashes"][str(path)] = file_hash(path)
        (WORKFLOW / "producer_source" / f"audit_recovery_{index}_{path.name}").write_bytes(path.read_bytes())
    producer["fingerprint"] = study.q.digest({k: v for k, v in producer.items() if k != "fingerprint"})
    write_json(WORKFLOW / "producer.json", producer)
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "stages": [], "final_test_opened": False,
               "producer_fingerprint": producer["fingerprint"]}
    try:
        for stage in ("finalize", "register"):
            study.data.verify(producer)
            command = [sys.executable, "-u", "-m", MODULE, stage]
            child = subprocess.Popen(command)
            journal.update(active_stage=stage, child_pid=child.pid)
            write_json(WORKFLOW / "progress.json", journal)
            code = child.wait()
            receipt = {"status": "observed_worker_terminal", "stage": stage,
                "child_pid": child.pid, "observer_pid": os.getpid(),
                "observer": "subprocess.wait", "exit_code": code,
                "command": command, "destination": str(DEST)}
            write_json(WORKFLOW / "stage_receipts" / (stage + ".json"), receipt)
            if DEST.exists():
                name = "worker_terminal_receipt.json" if stage == "finalize" else "registry_terminal_receipt.json"
                target = DEST / name
                if target.exists():
                    raise ValueError("recovery terminal receipt must never be overwritten")
                write_json(target, receipt)
            journal["stages"].append(receipt)
            write_json(WORKFLOW / "progress.json", journal)
            if code != 0:
                raise RuntimeError("recovery stage failed with observed exit: " + stage)
        study.data.verify(producer)
        campaign = study.q.read(DEST / "campaign_results.json")
        readback = study.q.read(DEST / (study.HYPOTHESIS + "_registry_readback.json"))
        digest = file_hash(DEST / "campaign_results.json")
        if (readback["status"] != "registered_and_read_back"
                or readback["record"]["campaign_sha256"] != digest
                or readback["record"]["confirmation_status"] != campaign["classification"]
                or campaign["final_test_opened"] or campaign["development"]["development_gate"]):
            raise ValueError("exact failed-development registry readback and closed final gate required")
        journal.update(status="audit_recovered_and_registered", active_stage=None,
            campaign_sha256=digest, classification=campaign["classification"],
            goal_target_met=False, final_test_opened=False, NCM_success_established=False)
        write_json(WORKFLOW / "result.json", journal)
        print(json.dumps(journal), flush=True)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["check", "finalize", "register", "workflow"])
    stage = parser.parse_args().stage
    if stage == "check":
        require_original_failure()
        print(json.dumps(dependency_preflight()), flush=True)
    elif stage == "workflow":
        main_workflow()
    else:
        execute(stage)


if __name__ == "__main__":
    main()
