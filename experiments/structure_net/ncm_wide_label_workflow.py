"""Finish the fixed training-only label pipeline; never restart its draft job."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.structure_net import ncm_wide_source_labels as labels
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, freeze, verify

DEST = ROOT / "20261023_wide_labels_main_workflow_v1"


def main():
    draft = labels.root(False, "draft")
    if not (draft / "producer.json").exists():
        raise ValueError("the separately launched original draft producer is required")
    p = freeze(DEST, "orchestrate_fixed_public_training_label_stages", [draft / "producer.json",
        Path(__file__), Path(labels.__file__), labels.PLAN],
        {"existing_draft_restarted": False, "remaining_stages": ["critique", "revision", "audit"],
         "training_sources_only": 112, "new_training_admitted": False,
         "terminal_receipt_source": "predecessor supervisors use subprocess.wait"})
    journal = {"status": "waiting_for_existing_main_draft", "observer_pid": os.getpid(), "stages": [],
               "producer_fingerprint": p["fingerprint"]}
    write_json(DEST / "progress.json", journal)
    try:
        # The original supervisor writes this only after its actual child exits.
        # No timeout, missing stdout or process guess triggers a duplicate draft.
        while not (draft / "worker_terminal_receipt.json").exists():
            time.sleep(5)
        labels.terminal(draft)
        if labels.q.read(draft / "result.json")["status"] != "generated":
            raise ValueError("original draft did not produce all valid first schemas")
        for stage in ("critique", "revision", "audit"):
            destination = labels.root(False, stage)
            if destination.exists():
                raise ValueError("stage already has artifacts; do not duplicate: " + stage)
            verify(p)
            child = subprocess.Popen([sys.executable, "-m", "experiments.structure_net.ncm_wide_source_labels", stage, "--main"])
            journal.update(status="running", active_stage=stage, child_pid=child.pid)
            write_json(DEST / "progress.json", journal)
            code = child.wait()
            record = {"stage": stage, "child_pid": child.pid, "exit_code": code}
            if destination.exists():
                receipt = destination / "worker_terminal_receipt.json"
                if receipt.exists():
                    raise ValueError("terminal receipt already exists; never overwrite")
                write_json(receipt, {"status": "observed_worker_terminal", "child_pid": child.pid,
                    "exit_code": code, "observer_pid": os.getpid(), "observer": "subprocess.wait"})
            journal["stages"].append(record)
            write_json(DEST / "progress.json", journal)
            if code != 0:
                raise RuntimeError("label stage failed: " + stage)
        verify(p)
        result = labels.q.read(labels.root(False, "audit") / "result.json")
        journal.update(status="completed_training_labels_only", active_stage=None,
            accepted_labels=result["accepted_labels"], accepted_by_bin=result["accepted_by_bin"],
            teacher_endpoint=result["endpoint"], student_training_admitted=False,
            NCM_success_established=False, final_test_opened=False, speed_tested=False)
        write_json(DEST / "result.json", journal)
        print(json.dumps(journal), flush=True)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error),
                       student_training_admitted=False, NCM_success_established=False)
        write_json(DEST / "failure.json", journal)
        raise


if __name__ == "__main__":
    main()
