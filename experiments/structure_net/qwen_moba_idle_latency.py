#!/usr/bin/env python3
"""Fixed-setting idle-GPU latency and component diagnostic for Qwen MoBA."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import numpy as np
import torch

from experiments.structure_net import qwen_moba_prefix as q
from experiments.structure_net import moba_head_benchmark as b
from experiments.structure_net import run_qwen_moba_feasibility as base

PROTOCOL = Path("docs/07 - Status Reports/2026-09-09_qwen8b-moba-idle-latency.md")
SETTING = q.MoBASetting(1024, 2, "all")
VARIANTS = {"full_mlp2048": (False, 32, 2048), "full_mlpwhole": (False, 32, 0),
            "moba_heads4_mlp2048": (True, 4, 2048), "moba_heads32_mlp2048": (True, 32, 2048),
            "moba_heads32_mlpwhole": (True, 32, 0)}


def gpu_state():
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], text=True)
    physical = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    fields = next(row.split(", ") for row in gpu.splitlines() if int(row.split(",")[0]) == physical)
    external = {row.split(", ")[1]: int(row.split(", ")[2]) for row in processes.splitlines()
                if row.split(", ")[0] == fields[1] and int(row.split(", ")[1]) != os.getpid()}
    return {"gpu": gpu, "processes": processes, "external_memory_mib": external,
            "free_mib": int(fields[2]), "utilization": int(fields[3])}


def uncontended(initial, current):
    return all(pid in initial["external_memory_mib"] and memory <= initial["external_memory_mib"][pid] + 128
               for pid, memory in current["external_memory_mib"].items())


def configure(model, variant, length):
    sparse, heads, chunk = VARIANTS[variant]
    model.moba_head_chunk_size = heads
    model.mlp_chunk_tokens = chunk or length
    return SETTING if sparse else q.FULL


def profile_components(model, ids, setting):
    events = defaultdict(list)
    pending = {}
    handles = []
    def start(module, _inputs):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        pending[module] = event
    def finish(bucket):
        def record(module, _inputs, _output):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            events[bucket].append((pending.pop(module), event))
        return record
    for name, module in model.named_modules():
        bucket = "mlp" if name.endswith(".mlp") else ("projections" if any(name.endswith("." + x) for x in ("q_proj", "k_proj", "v_proj", "o_proj")) else None)
        if bucket:
            handles.extend([module.register_forward_pre_hook(start), module.register_forward_hook(finish(bucket))])
    old_full, old_sparse = q.flash_call, q.moba_attention
    def wrap(fn):
        def measured(*args, **kwargs):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            result = fn(*args, **kwargs)
            end.record()
            events["attention"].append((begin, end))
            return result
        return measured
    q.flash_call, q.moba_attention = wrap(old_full), wrap(old_sparse)
    try:
        total = b.timed(lambda: model(ids, setting), 1)
    finally:
        q.flash_call, q.moba_attention = old_full, old_sparse
        for handle in handles:
            handle.remove()
    result = {key + "_ms": sum(a.elapsed_time(z) for a, z in pairs) for key, pairs in events.items()}
    result.update(total_ms=total, other_ms=total - sum(result.values()), calls={key: len(pairs) for key, pairs in events.items()})
    return result


def run(output):
    if output.exists():
        raise RuntimeError("choose a new immutable output root")
    initial = gpu_state()
    if initial["utilization"] > 10 or initial["free_mib"] < 10000:
        raise RuntimeError("GPU is not sufficiently idle for this diagnostic")
    sys.path.insert(0, str(b.UPSTREAM.resolve()))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inputs = base.prepare_inputs()
    schedules = base.window_schedules(inputs)
    producer = base.producer(output, inputs, schedules, "idle-latency-diagnostic", 32)
    for index, path in enumerate((Path(__file__), PROTOCOL), start=20):
        producer["sources"][str(path.resolve())] = q.sha256(path)
        shutil.copyfile(path, output / "producer_source" / f"{index:02d}_{path.name}")
    producer.update(initial_gpu_state=initial, variants=VARIANTS, lengths=[8192, 16384],
                    timing_configuration={"samples": 5, "warmups": 1}, scope="systems_diagnostic_reused_validation_window")
    producer.pop("fingerprint")
    producer["fingerprint"] = hashlib.sha256(json.dumps(producer, sort_keys=True).encode()).hexdigest()
    b.write_json(output / "producer.json", producer)
    model = q.load_prefix()
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("model must remain frozen")
    before = q.frozen_hash(model)
    tokens = np.load(inputs["tokens"]["wikipedia"]["path"], mmap_mode="r")
    start = schedules["wikipedia"]["validation_start"]
    ids = torch.tensor(tokens[start:start+128].copy(), device="cuda", dtype=torch.long)[None, :]
    validity = base.hf_parity(model, ids)
    cells = []
    profiles = {}
    with torch.inference_mode():
        for length in (8192, 16384):
            ids = torch.tensor(tokens[start:start+length].copy(), device="cuda", dtype=torch.long)[None, :]
            baseline = model(ids, configure(model, "full_mlp2048", length))
            actuals, records = {}, {}
            for variant in VARIANTS:
                torch.cuda.empty_cache()
                state = gpu_state()
                record = {"variant": variant, "length": length, "source": "wikipedia", "start": start, "gpu_before": state}
                try:
                    setting = configure(model, variant, length)
                    torch.cuda.reset_peak_memory_stats()
                    actuals[variant] = model(ids, setting)
                    record.update(status="completed", metrics=q.readout_metrics(actuals[variant], baseline),
                                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), samples_ms=[])
                except torch.cuda.OutOfMemoryError as exc:
                    record.update(status="out_of_memory", error=str(exc))
                    torch.cuda.empty_cache()
                records[variant] = record
            # Above numerical forwards also serve as one warmup per path.
            active = [name for name, row in records.items() if row["status"] == "completed"]
            for sample in range(5):
                order = active[sample % len(active):] + active[:sample % len(active)]
                if sample % 2:
                    order = order[::-1]
                for variant in order:
                    setting = configure(model, variant, length)
                    records[variant]["samples_ms"].append(b.timed(lambda: model(ids, setting), 1))
            reference = records["full_mlp2048"]["samples_ms"]
            for variant, record in records.items():
                record["gpu_after"] = gpu_state()
                record["no_external_contention_detected"] = uncontended(initial, record["gpu_before"]) and uncontended(initial, record["gpu_after"])
                if record["status"] == "completed":
                    samples = record["samples_ms"]
                    record.update(median_ms=statistics.median(samples), min_ms=min(samples), max_ms=max(samples),
                                  speedup_vs_full_mlp2048=statistics.median(reference)/statistics.median(samples))
                    counterpart = "moba_heads4_mlp2048" if variant.startswith("moba") else "full_mlp2048"
                    record["scheduling_parity"] = q.readout_metrics(actuals[variant], actuals[counterpart])
                cells.append(record)
                b.write_json(output / "cells" / f"n{length}_{variant}.json", record)
                print(json.dumps({k: record.get(k) for k in ("variant", "length", "status", "median_ms", "speedup_vs_full_mlp2048", "no_external_contention_detected")}), flush=True)
            if length == 16384:
                for variant in active:
                    profiles[variant] = profile_components(model, ids, configure(model, variant, length))
            del baseline, actuals, ids
            torch.cuda.empty_cache()
    after = q.frozen_hash(model)
    if before != after:
        raise RuntimeError("frozen weights changed")
    result = {"schema": "nal.qwen-moba-idle-latency.v1", "hypothesis_id": "qwen3-8b-prefix6-moba-idle-latency-v1",
              "status": "completed", "classification": "idle_gpu_latency_diagnostic" if all(c["no_external_contention_detected"] for c in cells) else "contention_detected_no_idle_claim",
              "producer_fingerprint": producer["fingerprint"], "validity": validity, "cells": cells, "instrumented_profiles": profiles,
              "frozen_weights_before": before, "frozen_weights_after": after, "application_accuracy_tested": False,
              "summary": {"requested": 10, "completed": sum(c["status"] == "completed" for c in cells)}}
    b.write_json(output / "campaign_results.json", result)
    print(json.dumps({"classification": result["classification"], "profiles": profiles}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
