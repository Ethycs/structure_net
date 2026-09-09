#!/usr/bin/env python3
"""Official MoBA inference cost across block sizes; no task-quality claims."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from experiments.structure_net import moba_head_benchmark as b


def gpu_state():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)


def run(output, allow_shared=False):
    if output.exists():
        raise RuntimeError("choose a new immutable output root")
    state = gpu_state()
    physical = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    fields = next(line.split(", ") for line in state.splitlines() if int(line.split(",")[0]) == physical)
    shared = int(fields[3]) > 10 or int(fields[2]) < 10000
    if shared and not allow_shared:
        raise RuntimeError("GPU is shared; no uncontended benchmark slot")
    torch.cuda.set_per_process_memory_fraction(.4 if shared else .95)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(b.UPSTREAM.resolve()))
    commit = subprocess.check_output(["git", "-C", str(b.UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if commit != b.UPSTREAM_COMMIT:
        raise RuntimeError("official source changed")
    source_paths = [Path(__file__), Path(b.__file__), Path(b.custom.__file__)] + sorted((b.UPSTREAM / "moba").glob("*.py"))
    producer = {"schema": "nal.moba-paper-latency-diagnostic.v1", "command": sys.argv,
                "sources": {str(p.resolve()): b.sha256(p) for p in source_paths},
                "official_commit": commit, "gpu_before": state,
                "shared_gpu": shared, "dtype": "float16", "seed": 1951,
                "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(), "tf32": False,
                "samples": 5, "repeats": 1, "warmups": 5,
                "scope": "synthetic Q/K/V inference timing; GQA expansion included; no model or accuracy"}
    for index, p in enumerate(source_paths):
        target = output / "producer_source" / f"{index:02d}_{p.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    b.write_json(output / "producer.json", producer)
    settings = [(128, 4), (256, 4), (512, 3), (1024, 2), (2048, 2), (4096, 2)]
    cases = [(1, 1, 32768, 512, 3)]
    cases += [(32, 8, length, block, k) for length in (512, 2048, 8192, 16384, 32768)
              for block, k in settings if block * k < length]
    cells = []
    with torch.inference_mode():
        for heads, kvheads, length, block, k in cases:
            torch.manual_seed(1951)
            query = torch.randn(1, length, heads, 128, dtype=torch.float16, device="cuda").transpose(1, 2)
            key, value = [torch.randn(1, length, kvheads, 128, dtype=torch.float16, device="cuda").transpose(1, 2) for _ in range(2)]
            cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
            cell = {"query_heads": heads, "kv_heads": kvheads, "length": length, "block_size": block, "top_k": k}
            try:
                torch.cuda.reset_peak_memory_stats()
                timing = b.paired_times({"full": lambda: b.flash_call(query, key, value),
                                        "moba": lambda: b.official_call(query, key, value, block, k, cu)}, samples=5, repeats=1)
                cell.update(status="completed", timing=timing,
                            attention_speedup=timing["full"]["median_ms"] / timing["moba"]["median_ms"],
                            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            except torch.cuda.OutOfMemoryError as exc:
                cell.update(status="out_of_memory", error=str(exc))
            cells.append(cell)
            b.write_json(output / "cells" / f"h{heads}_n{length}_b{block}_k{k}.json", cell)
            print(json.dumps({k: v for k, v in cell.items() if k not in {"timing", "error"}}), flush=True)
            del query, key, value, cu
            torch.cuda.empty_cache()
    result = {"status": "completed", "classification": "shared_gpu_systems_diagnostic" if shared else "isolated_attention_systems_benchmark",
              "producer": producer, "cells": cells, "gpu_after": gpu_state(),
              "summary": {"requested": len(cases), "completed": sum(c["status"] == "completed" for c in cells)},
              "application_accuracy_tested": False, "uncontended_speed_claim_allowed": not shared, "completed_at": time.time()}
    b.write_json(output / "campaign_results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-shared", action="store_true")
    args = parser.parse_args()
    run(args.output, args.allow_shared)
