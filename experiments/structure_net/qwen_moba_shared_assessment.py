#!/usr/bin/env python3
"""Explicitly exploratory Qwen-prefix MoBA assessment on a shared GPU."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

import numpy as np
import torch

from experiments.structure_net import run_qwen_moba_feasibility as base
from experiments.structure_net import qwen_moba_prefix as q
from experiments.structure_net import moba_head_benchmark as b
from experiments.structure_net.moba_paper_latency_diagnostic import gpu_state

PROTOCOL = Path("docs/07 - Status Reports/2026-09-09_qwen8b-moba-shared-gpu-amendment.md")
SETTINGS = tuple(q.MoBASetting(block, k, layers) for block, k in ((256, 4), (512, 3), (1024, 2)) for layers in ("all", "last"))
LENGTHS = (8192, 16384)


def paired_times(model, ids, setting):
    functions = {"full": lambda: model(ids, q.FULL), "moba": lambda: model(ids, setting)}
    for fn in functions.values():
        fn()
    torch.cuda.synchronize()
    values = {key: [] for key in functions}
    for sample in range(5):
        order = ("full", "moba") if sample % 2 == 0 else ("moba", "full")
        for key in order:
            values[key].append(b.timed(functions[key], 1))
    return {key: {"samples_ms": v, "median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v)} for key, v in values.items()}


def select(cells):
    for length in LENGTHS:
        candidates = []
        for setting in SETTINGS:
            rows = [c for c in cells if c["length"] == length and c["name"] == setting.name]
            if len(rows) == 2 and all(c["gate"] for c in rows):
                candidates.append((statistics.mean(c["prefix_speedup"] for c in rows), statistics.mean(c["metrics"]["change_pearson"] for c in rows), setting))
        candidates.sort(key=lambda row: row[:2], reverse=True)
        if candidates:
            return {"length": length, "setting": asdict(candidates[0][2])}
    return None


def run(output, selection_path=None):
    if output.exists():
        raise RuntimeError("preserve existing results; choose a new root")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    sys.path.insert(0, str(b.UPSTREAM.resolve()))
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inputs = base.prepare_inputs()
    schedules = base.window_schedules(inputs)
    selection_artifact = {"path": str(selection_path), "sha256": q.sha256(selection_path), "contents": json.loads(selection_path.read_text())} if selection_path else None
    stage = "shared-final" if selection_path else "shared-screen"
    producer = base.producer(output, inputs, schedules, stage, 4, selection_artifact, .45)
    # Include this entry point and its distinct interpretation in the identity.
    import experiments.structure_net.moba_paper_latency_diagnostic as diagnostic
    for index, source in enumerate((Path(__file__), PROTOCOL, Path(diagnostic.__file__)), start=20):
        producer["sources"][str(source.resolve())] = q.sha256(source)
        shutil.copyfile(source, output / "producer_source" / f"{index:02d}_{source.name}")
    producer.update(settings=[asdict(s) for s in SETTINGS], lengths=list(LENGTHS), gpu_before=gpu_state(),
                    timing_configuration={"warmups": 1, "samples": 5, "repeats": 1, "shared_gpu": True})
    producer.pop("fingerprint")
    producer["fingerprint"] = hashlib.sha256(json.dumps(producer, sort_keys=True).encode()).hexdigest()
    b.write_json(output / "producer.json", producer)
    settings, lengths, seeds = SETTINGS, LENGTHS, (7,)
    selection = None
    if selection_artifact:
        screen = selection_artifact["contents"]
        if screen["classification"] != "shared_gpu_candidate_selected" or not screen["selection"]:
            raise RuntimeError("no eligible shared-GPU selection")
        screen_producer = json.loads((selection_path.parent / "producer.json").read_text())
        for key in ("sources", "inputs", "schedules", "model", "execution", "settings", "lengths", "timing_configuration"):
            if producer[key] != screen_producer[key]:
                raise RuntimeError(f"selection producer changed {key}")
        selection = screen["selection"]
        settings, lengths, seeds = (q.MoBASetting(**selection["setting"]),), (selection["length"],), base.SEEDS
    model = q.load_prefix()
    model.moba_head_chunk_size = 4
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("weights must remain frozen")
    before = q.frozen_hash(model)
    tokens = {name: np.load(artifact["path"], mmap_mode="r") for name, artifact in inputs["tokens"].items()}
    start = schedules["wikipedia"]["validation_start"]
    ids = torch.tensor(tokens["wikipedia"][start:start + 128].copy(), device="cuda", dtype=torch.long)[None, :]
    validity = base.hf_parity(model, ids)
    b.write_json(output / "hf_prefix_parity.json", validity)
    cells = []
    for source in base.SOURCES:
        for seed in seeds:
            start = schedules[source]["final_starts"][str(seed)] if selection_artifact else schedules[source]["validation_start"]
            for length in lengths:
                ids = torch.tensor(tokens[source][start:start + length].copy(), device="cuda", dtype=torch.long)[None, :]
                with torch.inference_mode():
                    baseline = model(ids, q.FULL)
                for setting in settings:
                    cell = {"source": source, "seed": seed, "start": start, "length": length,
                            "name": setting.name, "setting": asdict(setting), "gpu_before": gpu_state()}
                    try:
                        torch.cuda.reset_peak_memory_stats()
                        with torch.inference_mode():
                            actual = model(ids, setting)
                            metrics = q.readout_metrics(actual, baseline)
                            timing = paired_times(model, ids, setting)
                        speed = timing["full"]["median_ms"] / timing["moba"]["median_ms"]
                        cell.update(status="completed", metrics=metrics, timing=timing, prefix_speedup=speed,
                                    sparse_active=True, gate=q.feasibility_gate(metrics, speed, True),
                                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), application_accuracy=None,
                                    timing_status="shared_gpu_observation")
                    except torch.cuda.OutOfMemoryError as exc:
                        cell.update(status="out_of_memory", error=str(exc), gate=False)
                        torch.cuda.empty_cache()
                    cell["gpu_after"] = gpu_state()
                    cells.append(cell)
                    b.write_json(output / "cells" / f"{source}_seed{seed}_n{length}_{setting.name}.json", cell)
                    print(json.dumps({k: cell.get(k) for k in ("source", "seed", "length", "name", "status", "prefix_speedup", "gate")}), flush=True)
                del ids, baseline
                torch.cuda.empty_cache()
    after = q.frozen_hash(model)
    if before != after:
        raise RuntimeError("frozen weights changed")
    if not selection_artifact:
        selection = select(cells)
        classification = "shared_gpu_candidate_selected" if selection else "no_shared_gpu_candidate_passed"
    else:
        classification = "shared_gpu_prefix_feasibility_supported" if all(c["gate"] for c in cells) else "shared_gpu_prefix_feasibility_not_confirmed"
    result = {"schema": "nal.qwen3-8b-prefix6-moba-shared-feasibility.v1", "hypothesis_id": "qwen3-8b-prefix6-moba-feasibility-v1",
              "status": "completed", "profile": "EXPLORATORY_SHARED_GPU", "stage": stage,
              "classification": classification, "selection": selection, "producer_fingerprint": producer["fingerprint"],
              "validity": validity, "frozen_weights_before": before, "frozen_weights_after": after, "model_weights_unchanged": True,
              "cells": cells, "summary": {"requested": len(settings) * len(lengths) * len(seeds) * len(base.SOURCES),
                  "completed": sum(c["status"] == "completed" for c in cells), "failed": sum(c["status"] != "completed" for c in cells),
                  "joint_passes": sum(c["gate"] for c in cells)},
              "application_accuracy_tested": False, "uncontended_speed_claim_allowed": False,
              "test_evaluated": bool(selection_artifact), "completed_at": time.time()}
    b.write_json(output / "campaign_results.json", result)
    print(json.dumps({"classification": classification, "summary": result["summary"], "selection": selection}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    run(args.output, args.selection)
