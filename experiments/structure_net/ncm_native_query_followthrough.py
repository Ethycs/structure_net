"""Generate and grade the fixed final NCM checkpoint after observed training exit."""
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.structure_net import train_ncm_native_query_neighborhood as run
from experiments.structure_net.ncm_memory_data import write_json
from experiments.structure_net.ncm_paper_data import ROOT, freeze, verify

DEST = ROOT / "20261023_native_query_neighborhood_followthrough_v1"


def main():
    receipt = run.q.read(run.TRAIN / "worker_terminal_receipt.json")
    result = run.q.read(run.TRAIN / "result.json")
    if (receipt["exit_code"] != 0 or receipt["status"] != "observed_worker_terminal"
            or result["status"] != "trained_pending_task_evaluation" or result["updates"] != 2048):
        raise ValueError("actual successful terminal fixed2048 training required")
    verify(run.q.read(run.TRAIN / "producer.json"))
    if run.OUTPUTS.exists() or run.GRADE.exists():
        raise ValueError("evaluation artifacts already exist; never duplicate")
    p = freeze(DEST, "supervise_fixed_final_checkpoint_generation_and_grading",
        [run.TRAIN / n for n in ("producer.json", "result.json", "model.pt", "worker_terminal_receipt.json")]
        + [Path(__file__), Path(run.__file__)],
        {"fixed_updates": 2048, "cases": 64, "parallel_passes": 9, "output_repair": False,
         "source_disjoint_development": True, "quality_gate_per_judge": 61,
         "final_test_opened": False, "speed_tested": False})
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [], "producer_fingerprint": p["fingerprint"]}
    write_json(DEST / "progress.json", journal)
    try:
        for stage, destination in (("generate", run.OUTPUTS), ("grade", run.GRADE)):
            verify(p)
            environment = os.environ.copy()
            environment.update(LD_LIBRARY_PATH="", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="2", PYTHONPATH="src:.",
                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                CUDA_VISIBLE_DEVICES="0" if stage == "generate" else "")
            child = subprocess.Popen([sys.executable, "-m", "experiments.structure_net.train_ncm_native_query_neighborhood", stage], env=environment)
            journal.update(active_stage=stage, child_pid=child.pid)
            write_json(DEST / "progress.json", journal)
            code = child.wait()
            if destination.exists():
                target = destination / "worker_terminal_receipt.json"
                if target.exists():
                    raise ValueError("terminal receipt already exists")
                write_json(target, {"status": "observed_worker_terminal", "child_pid": child.pid,
                    "exit_code": code, "observer_pid": os.getpid(), "observer": "subprocess.wait"})
            journal["stages"].append({"stage": stage, "child_pid": child.pid, "exit_code": code})
            write_json(DEST / "progress.json", journal)
            if code != 0:
                raise RuntimeError("fixed evaluation stage failed: " + stage)
        verify(p)
        quality = run.q.read(run.GRADE / "result.json")
        journal.update(status="task_evaluated_success_not_assumed", active_stage=None, grading=quality,
            NCM_success_established=False, final_test_opened=False, speed_tested=False)
        write_json(DEST / "result.json", journal)
        print(json.dumps(journal), flush=True)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error),
                       NCM_success_established=False)
        write_json(DEST / "failure.json", journal)
        raise


if __name__ == "__main__":
    main()
