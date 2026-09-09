#!/usr/bin/env python3
"""Frozen-weight nonattention acceleration with unchanged official MoBA."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

import numpy as np
import torch

from experiments.structure_net import qwen_moba_prefix as q
from experiments.structure_net import moba_head_benchmark as b
from experiments.structure_net import run_qwen_moba_feasibility as base
from experiments.structure_net import qwen_nonattention_fast as fast
from experiments.structure_net import qwen_nonattention_pilot as pilot
from experiments.structure_net import qwen_moba_idle_latency as idle

PROTOCOL = Path("docs/07 - Status Reports/2026-09-09_qwen8b-nonattention-acceleration.md")
SETTING = q.MoBASetting(1024, 2, "all")
HYPOTHESIS = "qwen3-8b-prefix6-nonattention-acceleration-v1"


def select(cells):
    passing = []
    for index, execution in enumerate(fast.VARIANTS):
        rows = [r for r in cells if r["variant"] == execution.name and r["length"] == 16384]
        if len(rows) == 2 and {r["source"] for r in rows} == set(base.SOURCES) and all(r["gate"] for r in rows):
            passing.append((statistics.mean(r["speedup_vs_moba"] for r in rows),
                            statistics.mean(r["metrics_vs_moba"]["change_pearson"] for r in rows), -index, execution))
    if not passing:
        return None
    return {"execution": asdict(max(passing, key=lambda r: r[:3])[3]), "length": 16384}


def record_producer(output, inputs, schedules, stage, selection_path, initial):
    import bitsandbytes, triton
    selected = {"path": str(selection_path), "sha256": q.sha256(selection_path),
                "contents": json.loads(selection_path.read_text())} if selection_path else None
    producing = base.producer(output, inputs, schedules, "nonattention-" + stage, 32, selected)
    package = Path(bitsandbytes.__file__).parent
    paths = [Path(__file__), PROTOCOL, Path(fast.__file__), Path(pilot.__file__), Path(idle.__file__),
             package / "nn/modules.py", package / "autograd/_functions.py", package / "functional.py"]
    for index, path in enumerate(paths, start=20):
        producing["sources"][str(path.resolve())] = q.sha256(path)
        shutil.copyfile(path, output / "producer_source" / f"{index:02d}_{path.name}")
    producing.update(schema="nal.qwen-nonattention.v1", hypothesis_id=HYPOTHESIS,
                     variants=[asdict(v) for v in fast.VARIANTS], settings=[asdict(SETTING)], lengths=[8192, 16384],
                     execution={"mlp_chunk_tokens": "whole_context", "moba_head_chunk_size": 32,
                                "compile_mode": "max-autotune-no-cudagraphs", "fullgraph": True, "dynamic": False,
                                "original_dtype": "float16", "quantized_copies": "int8_when_named"},
                     initial_gpu_state=initial, bitsandbytes=bitsandbytes.__version__, triton=triton.__version__,
                     timing_configuration={"samples": 5, "warmups": 1, "setup_excluded": True},
                     scope="reused_final_windows_regression" if stage == "regression" else "validation_optimization_screen")
    producing.pop("fingerprint")
    producing["fingerprint"] = hashlib.sha256(json.dumps(producing, sort_keys=True).encode()).hexdigest()
    b.write_json(output / "producer.json", producing)
    return producing


def evaluate(model, ids, execution, initial):
    candidate = None
    record = {"variant": execution.name, "execution": asdict(execution), "gpu_before": idle.gpu_state(), "gate": False}
    try:
        torch.cuda.empty_cache()
        setup = time.perf_counter()
        candidate = fast.FastPrefix(model, execution).eval()
        torch.cuda.synchronize()
        record["construction_seconds"] = time.perf_counter() - setup
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            full = model(ids, q.FULL)
            baseline = model(ids, SETTING)
            setup = time.perf_counter()
            actual = candidate(ids, SETTING)
            torch.cuda.synchronize()
            record["first_forward_seconds_including_compile"] = time.perf_counter() - setup
            record.update(metrics_vs_moba=q.readout_metrics(actual, baseline), metrics_vs_full=q.readout_metrics(actual, full),
                          original_moba_vs_full=q.readout_metrics(baseline, full))
            arms = {"full": lambda: model(ids, q.FULL), "moba": lambda: model(ids, SETTING), "candidate": lambda: candidate(ids, SETTING)}
            samples = {name: [] for name in arms}
            names = list(arms)
            for index in range(5):
                order = names[index % 3:] + names[:index % 3]
                if index % 2:
                    order = order[::-1]
                for arm in order:
                    samples[arm].append(b.timed(arms[arm], 1))
            medians = {name: statistics.median(values) for name, values in samples.items()}
            record.update(status="completed", samples_ms=samples, medians_ms=medians,
                          speedup_vs_moba=medians["moba"]/medians["candidate"], speedup_vs_full=medians["full"]/medians["candidate"],
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            record["fidelity_pass"] = fast.fidelity_gate(record["metrics_vs_moba"], execution.int8_threshold is not None)
    except Exception as exc:
        record.update(status="failed", error=repr(exc), fidelity_pass=False)
    finally:
        del candidate
        gc.collect()
        torch.cuda.empty_cache()
    record["gpu_after"] = idle.gpu_state()
    record["no_external_contention_detected"] = all(idle.uncontended(initial, record[key]) for key in ("gpu_before", "gpu_after"))
    record["gate"] = (record["status"] == "completed" and record["fidelity_pass"] and
                      record["speedup_vs_moba"] >= 1.05 and record["no_external_contention_detected"])
    return record


def run(output, stage, selection_path=None):
    if output.exists():
        raise RuntimeError("preserve existing root; choose a new root")
    initial = idle.gpu_state()
    if initial["utilization"] > 10 or initial["free_mib"] < 10000:
        raise RuntimeError("idle GPU required")
    sys.path.insert(0, str(b.UPSTREAM.resolve()))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inputs = base.prepare_inputs()
    schedules = base.window_schedules(inputs)
    producing = record_producer(output, inputs, schedules, stage, selection_path, initial)
    variants, sources, lengths, seeds = fast.VARIANTS, list(base.SOURCES), (8192, 16384), (7,)
    if stage == "pilot":
        variants, sources, lengths = fast.VARIANTS[:1], sources[:1], (8192,)
    elif stage == "regression":
        if not selection_path:
            raise ValueError("regression requires a sealed selection")
        selected = producing["selection_artifact"]["contents"]
        if selected["classification"] != "execution_candidate_selected" or not selected["selection"]:
            raise ValueError("no eligible screen selection")
        variants = (fast.Execution(**selected["selection"]["execution"]),)
        lengths, seeds = (selected["selection"]["length"],), base.SEEDS
    model = q.load_prefix()
    before = q.frozen_hash(model)
    tokens = {name: np.load(artifact["path"], mmap_mode="r") for name, artifact in inputs["tokens"].items()}
    first_start = schedules[sources[0]]["validation_start"]
    ids = torch.tensor(tokens[sources[0]][first_start:first_start+128].copy(), device="cuda", dtype=torch.long)[None, :]
    validity = base.hf_parity(model, ids)
    cells = []
    for source in sources:
        for seed in seeds:
            start = schedules[source]["final_starts"][str(seed)] if stage == "regression" else schedules[source]["validation_start"]
            for length in lengths:
                model.mlp_chunk_tokens, model.moba_head_chunk_size = length, 32
                ids = torch.tensor(tokens[source][start:start+length].copy(), device="cuda", dtype=torch.long)[None, :]
                for execution in variants:
                    print(json.dumps({"starting": execution.name, "source": source, "seed": seed, "length": length}), flush=True)
                    record = evaluate(model, ids, execution, initial)
                    record.update(source=source, seed=seed, start=int(start), length=length)
                    cells.append(record)
                    b.write_json(output / "cells" / f"{source}_s{seed}_n{length}_{execution.name}.json", record)
                    print(json.dumps({k: record.get(k) for k in ("variant", "status", "error", "medians_ms", "speedup_vs_moba", "fidelity_pass", "gate", "first_forward_seconds_including_compile")}), flush=True)
                del ids
                torch.cuda.empty_cache()
    after = q.frozen_hash(model)
    if before != after:
        raise RuntimeError("original checkpoint weights changed")
    selection = select(cells) if stage == "screen" else None
    classification = ("systems_lifecycle_only" if stage == "pilot" else
                      ("execution_candidate_selected" if selection else "no_execution_candidate_passed") if stage == "screen" else
                      "execution_regression_supported" if len(cells) == 10 and all(r["gate"] for r in cells) else "execution_regression_not_confirmed")
    result = {"schema": "nal.qwen-nonattention.v1", "hypothesis_id": HYPOTHESIS, "stage": stage,
              "status": "completed", "classification": classification, "selection": selection, "producer_fingerprint": producing["fingerprint"],
              "validity": validity, "frozen_weights_before": before, "frozen_weights_after": after,
              "application_accuracy_tested": False, "reused_regression_population": stage == "regression", "cells": cells,
              "summary": {"requested": len(variants)*len(lengths)*len(sources)*len(seeds),
                          "completed": sum(r["status"] == "completed" for r in cells), "joint_passes": sum(r["gate"] for r in cells)}}
    b.write_json(output / "campaign_results.json", result)
    print(json.dumps({k: result[k] for k in ("classification", "selection", "summary")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "screen", "regression"), required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    run(args.output, args.stage, args.selection)
