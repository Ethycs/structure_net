"""Conditional final-test execution after both paired development reviews."""
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


def select_arm(counts, kl):
    passing = [a for a in ("clean", "noise") if counts[a] == 8]
    if not passing:
        raise ValueError("heldout requires an 8/8 development candidate")
    return min(passing, key=lambda a: (kl[a], a != "clean"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--reviews", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("new final-test root required")
    review = json.loads(args.reviews.read_text())
    if review["fixtures_sha256"] != sha256(FIXTURES) or len(review["items"]) != 16:
        raise ValueError("paired development review required")
    counts = {a: sum(r["passed"] for r in review["items"] if r["arm"] == a) for a in ("clean", "noise")}
    kl = {a: json.loads((args.root / f"20261014_{a}_global_v2/training.json").read_text())["final_development"]["mean_kl"]
          for a in ("clean", "noise")}
    chosen = select_arm(counts, kl)
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output / "producer_source.py")
    write_json(args.output / "selection.json", {"hypothesis_id": HYPOTHESIS_ID, "selected_arm": chosen,
        "development_pass_counts": counts, "development_kl": kl, "rule": "passing arm with lowest final development KL; tie clean",
        "reviews_sha256": sha256(args.reviews), "self_sha256": sha256(__file__), "command": sys.argv})
    examples = [dict(e, stop_eos=True) for e in json.loads(FIXTURES.read_text())["examples"] if e["split"] == "heldout"]
    native_fixture = args.output / "native_quality_fixtures.json"
    write_json(native_fixture, {"schema": "nal.llama-summary-throughput.v1", "stage": "heldout",
        "context": 8192, "ubatch": 512, "drafts": [8], "repeats": 1, "examples": examples})
    global_root = args.root / f"20261014_{chosen}_global_v2"
    jobs = [
        (chosen, args.output / "candidate", [sys.executable, "-m", "experiments.structure_net.llama_noise_tensor_continuation", "evaluate",
            "--arm", chosen, "--split", "heldout", "--global-root", str(global_root), "--output", str(args.output / "candidate")]),
        ("dense", args.output / "dense", [sys.executable, "-m", "experiments.structure_net.llama_neighbourhood_tensorization", "evaluate",
            "--arm", "dense", "--split", "heldout", "--fixtures", str(FIXTURES), "--output", str(args.output / "dense")]),
        ("q4_lookup8", args.output / "q4_lookup8", [sys.executable, "-m", "experiments.structure_net.llama_summary_throughput", "run",
            "--model", "data/models/Llama-3.2-3B-cotenancy/Llama-3.2-3B-Instruct-Q4_K_M.gguf", "--stage", "heldout",
            "--fixtures", str(native_fixture), "--output", str(args.output / "q4_lookup8")])]
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "2", "LD_LIBRARY_PATH": ""}
    runs = []
    for arm, root, cmd in jobs:
        print(json.dumps({"starting": arm}), flush=True)
        start = time.perf_counter()
        with (args.output / (arm+".log")).open("w") as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        runs.append({"arm": arm, "root": str(root), "command": cmd, "returncode": result.returncode,
                     "seconds": time.perf_counter()-start})
        write_json(args.output / "execution.json", {"status": "failed" if result.returncode else "running", "runs": runs})
        if result.returncode:
            raise RuntimeError("conditional evaluation failed; preserve log")
        print(json.dumps({"completed": arm}), flush=True)
    write_json(args.output / "execution.json", {"status": "completed_pending_source_review", "runs": runs})


if __name__ == "__main__":
    main()
