"""Hash-checked stage execution after the paper-regime competence controls."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import PREP, ROOT, freeze, verify
from experiments.structure_net.ncm_paper_train_v2 import Config, NOTE, TEACHER, PAIRED_AUDIT, V2_NOTE


def controls(codec, reconstruction, oracle):
    """Bind competence judgments to the exact codec and exact oracle outputs."""
    cp = q.read(reconstruction.parent / "producer.json")
    cr = q.read(reconstruction)
    is_trained_codec = codec == reconstruction.parent / "model.pt" and cr.get("model_sha256") == file_hash(codec)
    if not is_trained_codec and cp["input_hashes"].get(str(codec)) != file_hash(codec):
        raise ValueError("reconstruction control evaluated a different codec")
    verify(cp)
    op = q.read(oracle.parent / "producer.json")
    outputs = reconstruction.parent / "outputs.json"
    if op["input_hashes"].get(str(outputs)) != file_hash(outputs):
        raise ValueError("semantic oracle did not grade these codec outputs")
    verify(op)
    a, b = q.read(reconstruction), q.read(oracle)
    if a["nonpad_reconstruction"] < .98 or b["status"] != "completed" or b["cases"] != 32 or b["passed"] < 31:
        raise ValueError("codec competence failed")
    audit = q.read(PAIRED_AUDIT)
    if audit["status"] != "completed" or audit["passed"] < 31 or audit["cases"] != 32:
        raise ValueError("teacher competence failed")
    audit_producer = q.read(PAIRED_AUDIT.parent / "producer.json")
    if audit_producer["input_hashes"].get(str(TEACHER / "train.json")) != file_hash(TEACHER / "train.json"):
        raise ValueError("teacher audit evaluated a different dataset")
    verify(audit_producer)
    dev = q.read(PREP / "development_sources.json")
    features = q.read(ROOT / "20261022_features_v1/result.json")
    if features["status"] != "passed" or features["cases"] != len(dev):
        raise ValueError("development body features incomplete")
    expected = {str(ROOT / "20261022_features_v1/features" / (r["id"]+".pt")) for r in dev}
    if set(features["output_hashes"]) != expected:
        raise ValueError("feature cohort differs from development cohort")
    for path, digest in features["output_hashes"].items():
        if file_hash(path) != digest:
            raise ValueError("development feature changed")
    return features


def completed(root, stage, codec):
    result = q.read(root / "result.json")
    p = q.read(root / "producer.json")
    verify(p)
    if (result["status"] != "passed" or p["stage"] != stage or result["producer_fingerprint"] != p["fingerprint"]
            or result["codec_sha256"] != file_hash(codec)):
        raise ValueError("stage cannot be reused")
    expected = Config().unpaired_steps if stage == "unpaired" else Config().conditional_steps
    key = "unpaired" if stage == "unpaired" else "conditional"
    if result["stage_results"][key]["updates"] != expected:
        raise ValueError("pilot or incomplete updates cannot be reused")
    return result


def run(args):
    features = controls(args.codec, args.codec_control, args.oracle_control)
    base = ROOT / "20261022_campaign_v2"
    if base.exists():
        raise ValueError("fresh campaign root required")
    inputs = [PREP / n for n in ("train_sources.json", "development_sources.json", "sealed_test_sources.json")]
    inputs += [TEACHER / "train.json", TEACHER / "producer.json", PAIRED_AUDIT, PAIRED_AUDIT.parent / "producer.json",
        args.codec, args.codec_control, args.codec_control.parent / "producer.json", args.oracle_control,
        args.oracle_control.parent / "producer.json", ROOT / "20261022_features_v1/result.json"]
    inputs += [Path(p) for p in features["output_hashes"]]
    p = freeze(base, "staged_training_and_development", inputs, asdict(Config()))
    p["hypothesis_id"] = "ncm-paper-regime-recovery-v2"
    p["execution_environment"] = {"python": sys.version, "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "physical_gpu": 0, "workers": 1,
        "device": torch.cuda.get_device_name(), "device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("transformers", "numpy", "requests", "safetensors")}}
    for source in [Path(__file__), NOTE, V2_NOTE, *sorted(Path("experiments/structure_net").glob("ncm_paper_*.py"))]:
        p["source_hashes"][str(source)] = file_hash(source)
        (base / "producer_source" / ("campaign_"+source.name)).write_bytes(source.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(base / "producer.json", p)
    common = ["--codec", str(args.codec), "--codec-control", str(args.codec_control),
              "--oracle-control", str(args.oracle_control), "--pilot", str(args.pilot)]
    results = {}
    unpaired = ROOT / "20261022_unpaired_v2"
    for stage in ("unpaired", "native", "hybrid"):
        verify(p)
        destination = ROOT / ("20261022_"+stage+"_v2")
        if not destination.exists():
            command = [sys.executable, "-m", "experiments.structure_net.ncm_paper_train_v2", stage,
                       "--output", str(destination), *common]
            if stage != "unpaired":
                command.extend(["--unpaired", str(unpaired)])
            subprocess.run(command, check=True)
        results[stage] = completed(destination, stage, args.codec)
        write_json(base / "progress.json", {"completed_stages": list(results), "test_opened": False})
    if (results["native"]["sample_schedule_hashes"]["conditional"] !=
            results["hybrid"]["sample_schedule_hashes"]["conditional"]):
        raise ValueError("matched conditional minibatch schedules differed")
    # Candidate selection uses development only. The final test stays sealed.
    development = {}
    for stage in ("native", "hybrid"):
        development[stage] = {}
        for step in (512, 1024, 1536, 2048):
            outputs = ROOT / ("20261022_"+stage+"_v2") / f"development_step{step}.json"
            destination = base / (stage+f"_development_step{step}")
            subprocess.run([sys.executable, "-m", "experiments.structure_net.ncm_paper_grade", "outputs",
                "--role", "primary", "--outputs", str(outputs), "--output", str(destination)], check=True)
            development[stage][str(step)] = q.read(destination / "result.json")
    verify(p)
    eligible = [(stage, int(step), row["passed"]) for stage, rows in development.items() for step, row in rows.items()
                if row["status"] == "completed" and row["cases"] == 64 and row["passed"] >= 61]
    artifacts = list(inputs)
    for stage in ("unpaired", "native", "hybrid"):
        directory = ROOT / ("20261022_"+stage+"_v2")
        artifacts.extend(directory.glob("*.json"))
        artifacts.extend(directory.glob("*.pt"))
    artifacts.extend(path for path in base.rglob("*.json") if "producer_source" not in path.parts)
    result = {"schema": "nal.ncm-paper-campaign.v1", "status": "completed_development_stage",
        "classification": "development_target_reached_pending_independent_test" if eligible else "development_accuracy_below_95",
        "training": results, "development": development, "test_admitted_candidates": eligible,
        "final_test_opened": False, "final_test_accuracy": None, "confirmed": False,
        "speed_tested": False, "underpowered": True, "producer_fingerprint": p["fingerprint"],
        "artifacts": {str(path): file_hash(path) for path in artifacts}}
    write_json(base / "campaign_results.json", result)
    print(json.dumps({"classification": result["classification"], "test_admitted_candidates": eligible}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--codec-control", type=Path, required=True)
    parser.add_argument("--oracle-control", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    run(parser.parse_args())
