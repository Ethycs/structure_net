"""Run conditional student/native final comparisons sequentially on GPU0."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json
from experiments.structure_net.llama_tensor_runtime import HYPOTHESIS_ID


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    selection_path = args.root / "20261014_selection_v1.json"
    selection = json.loads(selection_path.read_text())
    chosen = selection["selected_arm"]
    if chosen not in ("eager", "graph", "graph_lookup8") or selection["development_pass_counts"][chosen] != 8:
        raise ValueError("completed development selection required")
    if args.output.exists():
        raise ValueError("fresh coordinator root required")
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output / "producer_source.py")
    write_json(args.output / "producer.json", {"hypothesis_id":HYPOTHESIS_ID,"self_sha256":sha256(__file__),
        "selection":str(selection_path),"selection_sha256":sha256(selection_path),"command":sys.argv})
    student = args.root / "20261014_heldout_v1"
    native = args.root / "20261014_native_v1"
    jobs = [("student", [sys.executable,"-m","experiments.structure_net.llama_tensor_runtime","run",
        "--output",str(student),"--fixtures",str(args.root / "20261014_fixtures_v1.json"),
        "--split","heldout","--arms",f"reference,{chosen}","--repeats","5"]),
        ("native", [sys.executable,"-m","experiments.structure_net.llama_summary_throughput","run",
        "--output",str(native),"--fixtures",str(args.root / "20261014_native_fixtures_v1.json"),
        "--stage","heldout","--model","data/models/Llama-3.2-3B-cotenancy/Llama-3.2-3B-Instruct-Q4_K_M.gguf"])]
    env = {**os.environ,"CUDA_DEVICE_ORDER":"PCI_BUS_ID","CUDA_VISIBLE_DEVICES":"0","OMP_NUM_THREADS":"2","LD_LIBRARY_PATH":""}
    runs = []
    for label, command in jobs:
        print(json.dumps({"starting":label}),flush=True)
        started = time.perf_counter()
        with (args.output / (label+".log")).open("w") as log:
            run = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        runs.append({"label":label,"command":command,"returncode":run.returncode,"seconds":time.perf_counter()-started})
        write_json(args.output / "execution.json", {"status":"failed" if run.returncode else "running","runs":runs})
        if run.returncode:
            raise RuntimeError("preserve failed execution; no automatic retry")
        print(json.dumps({"completed":label}),flush=True)
    write_json(args.output / "execution.json", {"status":"completed_pending_source_review","runs":runs})


if __name__ == "__main__":
    main()
