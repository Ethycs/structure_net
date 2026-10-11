"""Execute the fixed Llama co-tenancy cells serially on physical GPU 0.

Subprocess boundaries release allocator state. This launcher never kills or
reconfigures unrelated jobs and preserves each failed cell's logs/artifacts.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import signal
import time

from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json

ROOT = Path("data/experiments/llama_tensorized_cotenancy")
SETTINGS = ("svd2_r215", "svd2_r768", "tt4_r64", "tt4_r256")


def halt_owned_campaign(root):
    """Stop only this launcher's exact run root after explicit user steering."""
    import psutil
    matches = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        command = process.info["cmdline"] or []
        if process.pid == os.getpid() or "experiments.structure_net.campaign_llama_cotenancy" not in command or "--halt" in command:
            continue
        if "--output" not in command or Path(command[command.index("--output")+1]).resolve() != root.resolve():
            continue
        children = process.children(recursive=True)
        matches.append({"pid": process.pid, "command": command, "children": [p.pid for p in children]})
        for child in reversed(children):
            if child.is_running():
                child.send_signal(signal.SIGINT)
        process.send_signal(signal.SIGINT)
    write_json(root/"user_steering_stop.json", {"reason": "User requested focus on lossless tensorization with a residual network", "processes": matches,
                                              "time": time.time(), "completed_evidence_preserved": True})
    print(json.dumps({"stopped_owned_campaigns": matches}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--projection", type=Path, help="reuse an unchanged, completed projection diagnostic")
    parser.add_argument("--halt", action="store_true", help="stop only this exact owned campaign root after user steering")
    parser.add_argument("--lossless", action="store_true", help="execute the fixed exact-residual follow-up")
    parser.add_argument("--lossless-timing", type=Path, help="decoded checkpoint for the counterbalanced timing follow-up")
    parser.add_argument("--exact-deployment", choices=("pilot", "full"), help="unchanged-model CUDA graph deployment comparison")
    parser.add_argument("--exact-speed", choices=("pilot", "full"), help="projection optimizations against CUDA graph replay")
    parser.add_argument("--exact-blas", choices=("pilot", "full"), help="exact BLAS backend preference comparison")
    args = parser.parse_args()
    if args.halt:
        halt_owned_campaign(args.output)
        return
    if args.output.exists():
        raise ValueError("campaign root must be new")
    args.output.mkdir(parents=True)
    os.environ.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0", LD_LIBRARY_PATH="",
                      PYTHONPATH=str(Path.cwd())+":"+str(Path.cwd()/"src"))
    if args.exact_blas:
        command = [sys.executable, "-m", "experiments.structure_net.run_llama_exact_blas", "--output", str(args.output / "blas")]
        if args.exact_blas == "pilot":
            command.append("--pilot")
        with (args.output / "blas.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(json.dumps({"completed": str(args.output / "blas/result.json")}), flush=True)
        return
    if args.exact_speed:
        command = [sys.executable, "-m", "experiments.structure_net.run_llama_exact_speed", "--output", str(args.output / "speed")]
        if args.exact_speed == "pilot":
            command.append("--pilot")
        with (args.output / "speed.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(json.dumps({"completed": str(args.output / "speed/result.json")}), flush=True)
        return
    if args.exact_deployment:
        command = [sys.executable, "-m", "experiments.structure_net.run_llama_exact_deployment", "--output", str(args.output / "deployment")]
        if args.exact_deployment == "pilot":
            command.append("--pilot")
        with (args.output / "deployment.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(json.dumps({"completed": str(args.output / "deployment/result.json")}), flush=True)
        return
    if args.lossless_timing:
        command=[sys.executable,"-m","experiments.structure_net.run_llama_lossless_timing","--checkpoint",str(args.lossless_timing),"--output",str(args.output/"timing")]
        with (args.output/"timing.log").open("w") as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps({"completed":str(args.output/"timing/result.json")}),flush=True)
        return
    if args.lossless:
        from experiments.structure_net.lossless_llama_campaign import run
        run(args.output)
        return
    control = json.loads((args.control/"result.json").read_text())
    if control["status"] != "completed" or not control["control_pass"]:
        raise RuntimeError("full-model HF/CUDA lifecycle control must pass first")
    checkpoint = Path(control["checkpoint"])
    if sha256(checkpoint) != control["checkpoint_sha256"]:
        raise RuntimeError("dense control checkpoint changed")
    index = {"schema": "nal.llama-cotenancy-campaign-index.v1", "control": str(args.control/"result.json"),
             "control_sha256": sha256(args.control/"result.json"), "launcher_sha256": sha256(__file__),
             "cells": [], "status": "running"}
    (args.output/"launcher.py").write_text(Path(__file__).read_text())

    def run(name, stage, *flags, module="run_llama_cotenancy", result_name="result.json"):
        output = args.output/name
        command = [sys.executable, "-m", "experiments.structure_net."+module]
        if stage:
            command.append(stage)
        command += ["--output", str(output), *map(str, flags)]
        print(json.dumps({"starting": name, "command": command}), flush=True)
        with (args.output/(name+".log")).open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        path = output/result_name
        result = json.loads(path.read_text()) if path.exists() else {"status": "failed", "reason": "no terminal JSON"}
        row = {"name": name, "result": str(path), "sha256": sha256(path) if path.exists() else None,
               "returncode": completed.returncode, "status": result["status"]}
        index["cells"].append(row)
        write_json(args.output/"progress.json", index)
        print(json.dumps({"finished": name, **row}), flush=True)
        if completed.returncode or result["status"] == "failed":
            raise RuntimeError("runtime fault; preserve evidence and inspect " + str(output))
        return result, output

    try:
        if args.projection:
            projection = json.loads(args.projection.read_text())
            if projection["status"] != "completed" or [r["rank"] for r in projection["rows"]] != [64, 128, 215, 256, 384]:
                raise ValueError("invalid projection reuse")
            index["cells"].append({"name": "projection_screen", "result": str(args.projection), "sha256": sha256(args.projection),
                                   "returncode": 0, "status": "completed", "reused": True})
        else:
            run("projection_screen", "", module="llama_cotenancy_projection", result_name="campaign_results.json")
        for count in (1, 2):
            pilot, _ = run(f"gguf_pilot_m{count}", "gguf", "--replicas", count, "--pilot")
            if pilot["status"] != "completed":
                raise RuntimeError("representative GGUF memory/concurrency pilot failed")
        baseline, _ = run("gguf_validation_m1", "gguf")
        for count in (2, 4):
            run(f"gguf_benchmark_m{count}", "gguf", "--replicas", count)
        run("dense_fp16_validation", "evaluate", "--checkpoint", checkpoint)
        candidates = {}
        for setting in SETTINGS:
            conversion, _ = run(setting+"_conversion", "convert", "--setting", setting)
            if conversion["status"] != "completed":
                continue
            model = conversion["checkpoint"]
            # Intended-concurrency shakedown remains separate from measured runs.
            run(setting+"_pilot_m2", "benchmark", "--checkpoint", model, "--replicas", 2, "--pilot")
            candidate, _ = run(setting+"_validation", "evaluate", "--checkpoint", model)
            bench = []
            for count in (1, 2, 4):
                r, _ = run(f"{setting}_benchmark_m{count}", "benchmark", "--checkpoint", model, "--replicas", count)
                bench.append(r)
            from experiments.structure_net.run_llama_cotenancy import promotion
            candidate_gate = promotion(candidate["quality"], baseline["quality"]) if candidate["status"] == "completed" else {"quality_pass": False}
            candidate_gate["packing_pass"] = all(p["pass"] for r in bench if r["status"] == "completed"
                                                 for b in r["benchmarks"] for k, p in b["parity"].items() if isinstance(p, dict))
            candidates[setting] = {"gate": candidate_gate, "checkpoint": model}
        promoted = [name for name, r in candidates.items() if r["gate"]["quality_pass"] and r["gate"]["packing_pass"]]
        if promoted:
            run("gguf_test", "gguf", "--split", "test")
            for name in promoted:
                run(name+"_test", "evaluate", "--checkpoint", candidates[name]["checkpoint"], "--split", "test")
        index.update(status="completed", candidates=candidates, promoted=promoted,
                     conditional_test="executed" if promoted else "not_run_no_passing_validation_candidate")
    except Exception as error:
        index.update(status="stopped_runtime_fault", error=str(error))
        raise
    finally:
        write_json(args.output/"campaign_index.json", index)


if __name__ == "__main__":
    main()
