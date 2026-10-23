"""Execute the explicitly exploratory noisy-pair NCM training diagnostic."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_campaign import completed
from experiments.structure_net.ncm_paper_data import ROOT, TEACHER, freeze, verify
from experiments.structure_net.ncm_paper_train_generic import Config, DIAGNOSTIC_NOTE


def main():
    pilot = ROOT / "20261022_noisy_pilot_v1"
    while not (pilot / "result.json").exists():
        time.sleep(5)
    if q.read(pilot / "result.json")["status"] != "passed":
        raise ValueError("diagnostic lifecycle pilot failed")
    codec = ROOT / "20261022_codec_training_v2"
    if q.read(codec / "result.json")["nonpad_reconstruction"] < .98:
        raise ValueError("codec reconstruction floor failed")
    base = ROOT / "20261022_noisy_campaign_v1"
    audit = ROOT / "20261022_teacher_audit_v1/result.json"
    p = freeze(base, "exploratory_noisy_pair_training", [TEACHER / "train.json", TEACHER / "result.json",
        audit, codec / "model.pt", codec / "result.json", pilot / "producer.json", pilot / "result.json",
        ROOT / "20261022_data_v1/development_sources.json", ROOT / "20261022_features_v1/result.json"], asdict(Config()))
    p["hypothesis_id"] = "ncm-paper-noisy-pairs-v1"
    p["unqualified_training_labels"] = True
    p["execution_environment"] = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu": 0, "workers": 1}
    for source in (Path(__file__), Path("experiments/structure_net/ncm_paper_train_generic.py"),
                   Path("experiments/structure_net/ncm_paper_campaign.py"), DIAGNOSTIC_NOTE):
        p["source_hashes"][str(source)] = file_hash(source)
        (base / "producer_source" / ("diagnostic_"+source.name)).write_bytes(source.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(base / "producer.json", p)
    unpaired = ROOT / "20261022_noisy_unpaired_v1"
    common = ["--diagnostic", "--training-data", str(TEACHER), "--label-audit", str(audit),
        "--codec", str(codec / "model.pt"), "--codec-control", str(codec / "result.json"), "--pilot", str(pilot)]
    results = {}
    for stage in ("unpaired", "native", "hybrid"):
        verify(p)
        destination = ROOT / ("20261022_noisy_"+stage+"_v1")
        command = [sys.executable, "-m", "experiments.structure_net.ncm_paper_train_generic", stage,
                   "--output", str(destination), *common]
        if stage != "unpaired":
            command += ["--unpaired", str(unpaired)]
        print(json.dumps({"stage": stage, "output": str(destination)}), flush=True)
        subprocess.run(command, check=True)
        results[stage] = completed(destination, stage, codec / "model.pt")
        write_json(base / "progress.json", {"completed_stages": list(results), "quality_unmeasured": True})
    if results["native"]["sample_schedule_hashes"]["conditional"] != results["hybrid"]["sample_schedule_hashes"]["conditional"]:
        raise ValueError("conditional minibatch schedules differed")
    verify(p)
    artifacts = {}
    for stage in ("unpaired", "native", "hybrid"):
        directory = ROOT / ("20261022_noisy_"+stage+"_v1")
        for path in [*directory.glob("*.json"), *directory.glob("*.pt")]:
            artifacts[str(path)] = file_hash(path)
    result = {"status": "trained_pending_development_grading", "hypothesis_id": "ncm-paper-noisy-pairs-v1",
        "classification": "staged_training_completed_quality_unmeasured", "training": results,
        "known_label_audit": q.read(audit), "unqualified_training_labels": True,
        "task_accuracy": None, "final_test_opened": False, "confirmed": False, "underpowered": True,
        "speed_tested": False, "producer_fingerprint": p["fingerprint"], "artifacts": artifacts}
    write_json(base / "campaign_results.json", result)
    print(json.dumps({"status": result["status"]}), flush=True)


if __name__ == "__main__":
    main()
