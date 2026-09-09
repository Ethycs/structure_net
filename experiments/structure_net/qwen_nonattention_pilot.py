#!/usr/bin/env python3
"""CUDA microbenchmark on real Qwen MLP input; never application accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments.structure_net import qwen_moba_prefix as q
from experiments.structure_net import moba_head_benchmark as b
from experiments.structure_net import run_qwen_moba_feasibility as base
from experiments.structure_net.qwen_moba_idle_latency import gpu_state, uncontended

PROTOCOL = Path("docs/07 - Status Reports/2026-09-09_qwen8b-nonattention-acceleration.md")


class PackedMLP(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.register_buffer("gate_up", torch.cat((original.gate_proj.weight, original.up_proj.weight), 0))
        self.down = original.down_proj

    def forward(self, x):
        gate, up = F.linear(x, self.gate_up).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


def int8_linear(original, threshold):
    import bitsandbytes as bnb
    layer = bnb.nn.Linear8bitLt(original.in_features, original.out_features, bias=False,
                              has_fp16_weights=False, threshold=threshold, device="meta")
    layer.weight = bnb.nn.Int8Params(original.weight.detach().clone(), requires_grad=False,
                                     has_fp16_weights=False).cuda()
    return layer.eval().requires_grad_(False)


class Int8MLP(nn.Module):
    def __init__(self, original, threshold=6):
        super().__init__()
        self.gate_proj = int8_linear(original.gate_proj, threshold)
        self.up_proj = int8_linear(original.up_proj, threshold)
        self.down_proj = int8_linear(original.down_proj, threshold)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def run(output):
    if output.exists():
        raise RuntimeError("choose a new root")
    initial = gpu_state()
    if initial["utilization"] > 10 or initial["free_mib"] < 10000:
        raise RuntimeError("idle GPU required")
    sys.path.insert(0, str(b.UPSTREAM.resolve()))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inputs, schedules = base.prepare_inputs(), None
    schedules = base.window_schedules(inputs)
    producer = base.producer(output, inputs, schedules, "nonattention-mlp-pilot", 32)
    import bitsandbytes, triton
    import experiments.structure_net.qwen_moba_idle_latency as idle
    paths = [Path(__file__), PROTOCOL, Path(idle.__file__), Path(bitsandbytes.__file__).parent / "nn/modules.py"]
    for index, path in enumerate(paths, start=20):
        producer["sources"][str(path.resolve())] = q.sha256(path)
        shutil.copyfile(path, output / "producer_source" / f"{index:02d}_{path.name}")
    producer.update(initial_gpu_state=initial, bitsandbytes=bitsandbytes.__version__, triton=triton.__version__,
                    scope="real_layer2_mlp_input_microbenchmark", length=8192, timing_configuration={"warmups": 1, "samples": 5})
    producer.pop("fingerprint")
    producer["fingerprint"] = hashlib.sha256(json.dumps(producer, sort_keys=True).encode()).hexdigest()
    b.write_json(output / "producer.json", producer)
    model = q.load_prefix()
    model.mlp_chunk_tokens = 8192
    original_hash = q.frozen_hash(model)
    start = schedules["wikipedia"]["validation_start"]
    tokens = np.load(inputs["tokens"]["wikipedia"]["path"], mmap_mode="r")
    ids = torch.tensor(tokens[start:start+8192].copy(), device="cuda", dtype=torch.long)[None, :]
    captured = []
    handle = model.layers[2].mlp.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    with torch.inference_mode():
        model(ids, q.MoBASetting(1024, 2, "all"))
    handle.remove()
    x, original = captured[0], model.layers[2].mlp
    torch.save({"input": x.cpu(), "layer": 2, "source": "wikipedia", "start": start}, output / "mlp_input.pt")
    records = []
    for variant in ("fp16", "cublaslt", "packed_fp16", "compiled_fp16", "int8_threshold6", "int8_threshold0"):
        record = {"variant": variant, "gpu_before": gpu_state()}
        backend = "cublaslt" if variant == "cublaslt" else "cublas"
        torch.backends.cuda.preferred_blas_library("cublas")
        module = None
        try:
            with torch.inference_mode():
                target = original(x)
                setup_start = time.perf_counter()
                if variant.startswith("int8"):
                    module = Int8MLP(original, 6 if variant.endswith("6") else 0)
                elif variant == "packed_fp16":
                    module = PackedMLP(original)
                elif variant == "compiled_fp16":
                    module = torch.compile(original, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")
                else:
                    module = original
                torch.backends.cuda.preferred_blas_library(backend)
                torch.cuda.reset_peak_memory_stats()
                actual = module(x)
                torch.cuda.synchronize()
                record["setup_seconds"] = time.perf_counter() - setup_start
                record.update(relative_l2=float((actual.float()-target.float()).norm()/target.float().norm()),
                              cosine_mean=float(F.cosine_similarity(actual.float(), target.float(), dim=-1).mean()),
                              max_abs_error=float((actual-target).abs().max()))
                samples = {"reference": [], "candidate": []}
                for sample in range(5):
                    order = ("reference", "candidate") if sample % 2 == 0 else ("candidate", "reference")
                    for arm in order:
                        torch.backends.cuda.preferred_blas_library("cublas" if arm == "reference" else backend)
                        samples[arm].append(b.timed(lambda: original(x) if arm == "reference" else module(x), 1))
                record.update(status="completed", samples_ms=samples,
                              reference_ms=statistics.median(samples["reference"]), candidate_ms=statistics.median(samples["candidate"]),
                              speedup=statistics.median(samples["reference"])/statistics.median(samples["candidate"]),
                              peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        except Exception as exc:
            record.update(status="failed", error=repr(exc))
            print(repr(exc), flush=True)
        finally:
            torch.backends.cuda.preferred_blas_library("cublas")
            del module
            torch.cuda.empty_cache()
        record["gpu_after"] = gpu_state()
        record["no_external_contention_detected"] = uncontended(initial, record["gpu_after"])
        records.append(record)
        b.write_json(output / "cells" / f"{variant}.json", record)
        print(json.dumps({k: v for k, v in record.items() if k not in {"samples_ms", "gpu_before", "gpu_after"}}), flush=True)
    final_hash = q.frozen_hash(model)
    if final_hash != original_hash:
        raise RuntimeError("original weights changed")
    result = {"status": "completed", "classification": "systems_pilot_not_prefix_or_application_accuracy", "producer_fingerprint": producer["fingerprint"],
              "original_weights_before": original_hash, "original_weights_after": final_hash, "cells": records}
    b.write_json(output / "campaign_results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
