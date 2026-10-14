"""Run both fixed continuation arms serially after independent CUDA pilots."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.structure_net.llama_noise_tensor_continuation import FIXTURES, HYPOTHESIS_ID
from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--local-pilot", type=Path, required=True)
    p.add_argument("--global-pilot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("new coordinator root required")
    for pilot in (args.local_pilot, args.global_pilot):
        t = json.loads((pilot / "training.json").read_text())
        if t["status"] != "completed" or t["stage"] != "systems_lifecycle_only_not_quality_evidence":
            raise ValueError("passed independent CUDA pilots required")
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output / "producer_source.py")
    write_json(args.output / "producer.json", {"hypothesis_id": HYPOTHESIS_ID, "self_sha256": sha256(__file__),
        "training_source_sha256": sha256("experiments/structure_net/llama_noise_tensor_continuation.py"),
        "prepared_sha256": sha256(args.prepared / "prepared.json"),
        "local_pilot_sha256": sha256(args.local_pilot / "training.json"),
        "global_pilot_sha256": sha256(args.global_pilot / "training.json"), "command": sys.argv,
        "physical_gpu": 0, "concurrency": 1})
    jobs = []
    for arm in ("clean", "noise"):
        local = args.root / f"20261014_{arm}_local_v2"
        global_ = args.root / f"20261014_{arm}_global_v2"
        for label, command, root, extra in (
            (arm+"_local", "local", local, []),
            (arm+"_global", "global", global_, ["--local", str(local)]),
            (arm+"_development", "evaluate", args.root / f"20261014_{arm}_development_v2", ["--global-root", str(global_)])
        ):
            cmd = [sys.executable, "-m", "experiments.structure_net.llama_noise_tensor_continuation", command,
                   "--prepared", str(args.prepared), "--arm", arm, "--output", str(root)] + extra
            jobs.append((label, root, cmd))
    fixtures = json.loads(FIXTURES.read_text())
    timing = [dict(e, cap=32, stop_eos=False) for e in fixtures["examples"] if e["split"] == "development"][:4]
    native_fixture = args.output / "native_runtime_fixtures.json"
    write_json(native_fixture, {"schema": "nal.llama-summary-throughput.v1", "stage": "development",
        "context": 8192, "ubatch": 512, "drafts": [8], "repeats": 3, "examples": timing})
    native = args.root / "20261014_native_runtime_v2"
    jobs.append(("native_runtime", native, [sys.executable, "-m", "experiments.structure_net.llama_summary_throughput", "run",
        "--model", "data/models/Llama-3.2-3B-cotenancy/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "--fixtures", str(native_fixture), "--stage", "development", "--output", str(native)]))
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0",
           "OMP_NUM_THREADS": "2", "LD_LIBRARY_PATH": ""}
    runs = []
    for label, root, cmd in jobs:
        print(json.dumps({"starting": label}), flush=True)
        started = time.perf_counter()
        with (args.output / (label+".log")).open("w") as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        runs.append({"label": label, "root": str(root), "command": cmd, "returncode": result.returncode,
                     "seconds": time.perf_counter()-started})
        write_json(args.output / "execution.json", {"status": "running" if not result.returncode else "failed", "runs": runs})
        if result.returncode:
            raise RuntimeError(label+" failed; preserve artifacts and inspect its log")
        print(json.dumps({"completed": label}), flush=True)
    write_json(args.output / "execution.json", {"status": "completed_development_pending_source_review", "runs": runs,
                                               "heldout_executed": False})


if __name__ == "__main__":
    main()
