"""Counterbalanced five-repeat held-out campaign, with immutable run plans."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json

MODELS = {
    "fp16": {"path": "data/experiments/llama_summary_throughput/20261013_f16_conversion_v2/Llama-3.2-3B-Instruct-F16.gguf",
             "sha256": "4924af1a90334aab0922e60d6285c006485b32809a50f032a9187b8012622174"},
    "q4": {"path": "data/models/Llama-3.2-3B-cotenancy/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
           "sha256": "6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new campaign root")
    value = json.loads(args.fixtures.read_text())
    if value["stage"] != "heldout" or value["drafts"] != [0, 8] or value["repeats"] != 5:
        raise ValueError("frozen held-out five-repeat fixture required")
    args.output.mkdir(parents=True)
    fixtures = {}
    for phase, repeats in (("a", 2), ("b", 3)):
        fixtures[phase] = args.output / f"fixtures_{phase}.json"
        write_json(fixtures[phase], {**value, "repeats": repeats,
                   "campaign_parent_sha256": sha256(args.fixtures), "campaign_phase": phase})
    runs = [{"model": name, "phase": phase, "repeats": 2 if phase == "a" else 3,
             "root": str(args.output / f"{name}_{phase}"), "fixtures": str(fixtures[phase])}
            for name, phase in (("fp16", "a"), ("q4", "a"), ("q4", "b"), ("fp16", "b"))]
    sources = [Path(__file__), Path("experiments/neural_architecture_lab/finalize_llama_summary_throughput.py"),
               Path("experiments/neural_architecture_lab/analyze_llama_summary_throughput.py"),
               Path("docs/07 - Status Reports/2026-10-13_llama32-summary-selection.md")]
    snapshot = args.output / "orchestrator_source"
    snapshot.mkdir()
    for path in sources:
        shutil.copyfile(path, snapshot / path.name)
    manifest = {"schema": "nal.llama-summary-primary-plan.v1", "models": MODELS, "runs": runs,
                "fixtures": str(args.fixtures), "fixtures_sha256": sha256(args.fixtures),
                "sources": {str(p): sha256(p) for p in sources},
                "baseline": "fp16_draft0", "candidate": "q4_draft8", "existing_reference": "q4_draft0",
                "selection_split": "development", "cross_request_reuse": False,
                "quality_boundary": "synthetic final-decision proxy; source review required"}
    write_json(args.output / "manifest.json", manifest)
    for run in runs:
        cmd = [sys.executable, "-m", "experiments.structure_net.llama_summary_throughput", "run",
               "--stage", "heldout", "--model", MODELS[run["model"]]["path"],
               "--fixtures", run["fixtures"], "--output", run["root"]]
        print(json.dumps({"starting": run["root"]}), flush=True)
        process = subprocess.run(cmd, env={**os.environ, "OMP_NUM_THREADS": "2"})
        if process.returncode:
            write_json(args.output / "execution.json", {"status": "failed", "failed_run": run, "returncode": process.returncode})
            raise RuntimeError("campaign stopped; preserve partial evidence")
    write_json(args.output / "execution.json", {"status": "completed", "runs": len(runs)})


if __name__ == "__main__":
    main()
