"""Serial fresh-request evaluation of one completed neighbourhood student."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--native-quality-fixtures", type=Path, required=True)
    parser.add_argument("--native-runtime-fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-training", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new artifact root required")
    args.output.mkdir(parents=True)
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0",
           "OMP_NUM_THREADS": "2", "LD_LIBRARY_PATH": ""}
    producer = {"self_sha256": sha256(__file__), "fixtures_sha256": sha256(args.fixtures),
                "native_quality_fixture_sha256": sha256(args.native_quality_fixtures),
                "native_runtime_fixture_sha256": sha256(args.native_runtime_fixtures),
                "training": str(args.training), "command": sys.argv,
                "physical_gpu": 0, "concurrency": 1}
    write_json(args.output / "producer.json", producer)
    (args.output / "producer_source.py").write_bytes(Path(__file__).read_bytes())
    training_path = args.training / "training.json"
    if args.wait_training:
        while not training_path.exists():
            if (args.training / "failure.json").exists():
                raise RuntimeError("training failed; preserve failure evidence")
            time.sleep(5)
    training = json.loads(training_path.read_text())
    if training["status"] != "completed" or len(training["groups"]) != 14:
        raise ValueError("complete 14-neighbourhood training required")
    jobs = []
    for arm in ("dense", "initial", "distilled"):
        root = args.output / arm
        cmd = [sys.executable, "-m", "experiments.structure_net.llama_neighbourhood_tensorization", "evaluate",
               "--fixtures", str(args.fixtures), "--training", str(args.training),
               "--output", str(root), "--arm", arm, "--split", "development"]
        jobs.append((arm, root, cmd))
    model = "data/models/Llama-3.2-3B-cotenancy/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    for label, fixtures in (("native_quality", args.native_quality_fixtures), ("native_runtime", args.native_runtime_fixtures)):
        root = args.output / label
        cmd = [sys.executable, "-m", "experiments.structure_net.llama_summary_throughput", "run",
               "--model", model, "--fixtures", str(fixtures), "--output", str(root), "--stage", "development"]
        jobs.append((label, root, cmd))
    results = []
    for label, root, cmd in jobs:
        print(json.dumps({"starting": label}), flush=True)
        started = time.time()
        with (args.output / f"{label}.log").open("w") as log:
            process = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        record = {"arm": label, "root": str(root), "command": cmd, "returncode": process.returncode,
                  "process_seconds": time.time() - started}
        if (root / "result.json").exists():
            record["result_sha256"] = sha256(root / "result.json")
        results.append(record)
        write_json(args.output / "execution.json", {"status": "running" if process.returncode == 0 else "failed",
                   "training_sha256": sha256(training_path), "runs": results})
        if process.returncode:
            raise RuntimeError(f"{label} failed; see preserved log")
        print(json.dumps({"completed": label}), flush=True)
    write_json(args.output / "execution.json", {"status": "completed_development_pending_source_review",
               "training_sha256": sha256(training_path), "runs": results,
               "heldout_executed": False, "quality_classification": "pending_explicit_source_review"})


if __name__ == "__main__":
    main()
