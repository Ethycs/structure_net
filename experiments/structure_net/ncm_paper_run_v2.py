"""Continue v2 automatically through fixed gates; leave the final test sealed."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT


def read(path):
    return json.loads(path.read_text())


def main():
    destination = ROOT / "20261022_driver_v2"
    if destination.exists():
        raise ValueError("fresh driver record required; inspect existing producer first")
    destination.mkdir(parents=True)
    write_json(destination / "producer.json", {"schema": "nal.ncm-stage-driver.v2", "pid": os.getpid(),
        "source": str(Path(__file__)), "source_sha256": file_hash(__file__), "command": sys.argv,
        "purpose": "sequential gate execution, not a quality endpoint", "final_test_opened": False})
    stage = "wait_for_curated_labels"
    try:
        curated = ROOT / "20261022_curated_v3/result.json"
        print(json.dumps({"stage": stage}), flush=True)
        while not curated.exists():
            time.sleep(5)
        if read(curated)["status"] != "passed":
            raise ValueError("curation gate failed")
        stage = "curated_label_audit"
        audit = ROOT / "20261022_curated_audit_v3/result.json"
        if not audit.exists():
            subprocess.run([sys.executable, "-m", "experiments.structure_net.ncm_paper_curate_v3", "audit"], check=True)
        value = read(audit)
        if value["status"] != "completed" or value["cases"] != 32 or value["passed"] < 31:
            raise ValueError("curated label competence gate failed")
        stage = "codec_semantic_oracle"
        oracle = ROOT / "20261022_oracle_grade_secondary_v2/result.json"
        if not oracle.exists():
            subprocess.run([sys.executable, "-m", "experiments.structure_net.ncm_paper_oracle_v2"], check=True)
        value = read(oracle)
        if value["status"] != "completed" or value["cases"] != 32 or value["passed"] < 31:
            raise ValueError("codec semantic competence gate failed")
        stage = "staged_generator_adaptation"
        codec = ROOT / "20261022_codec_training_v2"
        subprocess.run([sys.executable, "-m", "experiments.structure_net.ncm_paper_campaign_v2",
            "--codec", str(codec / "model.pt"), "--codec-control", str(codec / "result.json"),
            "--oracle-control", str(oracle), "--pilot", str(ROOT / "20261022_generator_pilot_v2")], check=True)
        write_json(destination / "result.json", {"status": "completed_development_stages",
            "campaign": str(ROOT / "20261022_campaign_v2/campaign_results.json"),
            "goal_complete": False, "final_test_opened": False})
    except Exception as error:
        write_json(destination / "result.json", {"status": "stopped_at_gate_or_infrastructure_error",
            "stage": stage, "error": type(error).__name__+": "+str(error),
            "goal_complete": False, "final_test_opened": False})
        raise


if __name__ == "__main__":
    main()
