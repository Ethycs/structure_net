"""Strict artifacts, producer identity and GPU measurements for co-tenancy."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import torch

from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json

HYPOTHESIS_ID = "llama32-3b-tensorized-cotenancy-v1"
SCHEMA_VERSION = "nal.llama-cotenancy.v1"
ROOT = Path("data/experiments/llama_tensorized_cotenancy")
PROTOCOL = Path("docs/07 - Status Reports/2026-10-10_llama32-tensorized-cotenancy-preregistration.md")
SOURCES = [Path("src/structure_net/components/layers/model_major_tensor.py"),
           *[Path("experiments/structure_net") / name for name in (
               "llama_tensorized_model.py", "llama_cotenancy_data.py", "llama_cotenancy_common.py", "prepare_llama_cotenancy.py")], PROTOCOL,
           Path("docs/07 - Status Reports/2026-10-10_llama32-cotenancy-runtime-amendment.md")]


def gpu_state():
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], text=True)
    physical = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    row = next(r.split(", ") for r in gpu.splitlines() if int(r.split(",")[0]) == physical)
    selected = {r.split(", ")[1]: int(r.split(", ")[2]) for r in processes.splitlines() if r.split(", ")[0] == row[1]}
    return {"gpu": gpu, "processes": processes, "used_mib": int(row[3]), "free_mib": int(row[2]), "utilization": int(row[4]),
            "own_mib": selected.get(str(os.getpid()), 0), "external": {k: v for k, v in selected.items() if k != str(os.getpid())}}


def uncontended(before, after):
    return all(k in before["external"] and v <= before["external"][k]+128 for k, v in after["external"].items())


def setup():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent CPU fallback")


def producer(output, stage, extra_paths=(), configuration=None):
    import transformers, numpy, flash_attn
    if output.exists():
        raise ValueError("choose a new immutable run root")
    paths = SOURCES + list(extra_paths)
    external = Path(transformers.__file__).parent / "models/llama/modeling_llama.py"
    paths += [external, Path(transformers.__file__).parent / "modeling_rope_utils.py"]
    snapshot = output / "producer_source"
    snapshot.mkdir(parents=True)
    for i, path in enumerate(paths):
        shutil.copyfile(path, snapshot / f"{i:02d}_{path.name}")
    inputs = Path("data/models/Llama-3.2-3B-cotenancy/manifest.json")
    value = {"hypothesis_id": HYPOTHESIS_ID, "schema": SCHEMA_VERSION, "stage": stage,
             "configuration": configuration, "sources": {str(p): sha256(p) for p in paths},
             "model_manifest_path": str(inputs), "model_manifest_sha256": sha256(inputs), "model_manifest": json.loads(inputs.read_text()),
             "environment": {"torch": str(torch.__version__), "transformers": transformers.__version__, "numpy": numpy.__version__,
                             "flash_attn": flash_attn.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                             "tf32": False, "dtype": "float16", "decomposition_dtype": "float32"},
             "gpu_before": gpu_state(), "command": sys.argv,
             "deployed_q4_digest_confirmed_by_handoff": "6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff"}
    value["fingerprint"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    write_json(output / "producer.json", value)
    return value


def timed(fn):
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    begin.record()
    value = fn()
    end.record()
    torch.cuda.synchronize()
    return {"cuda_ms": begin.elapsed_time(end), "wall_ms": 1000*(time.perf_counter()-wall)}, value


def paired(arms, samples=5):
    for fn in arms.values():
        fn()
    torch.cuda.synchronize()
    records = {name: [] for name in arms}
    names = list(arms)
    for index in range(samples):
        order = names[index % len(names):] + names[:index % len(names)]
        if index % 2:
            order.reverse()
        for name in order:
            row, value = timed(arms[name])
            records[name].append(row)
            del value
    return {name: {"samples": rows, "cuda_median_ms": statistics.median(r["cuda_ms"] for r in rows),
                   "wall_median_ms": statistics.median(r["wall_ms"] for r in rows)} for name, rows in records.items()}


def parity(actual, expected):
    difference = actual.float()-expected.float()
    relative = float(difference.norm()/expected.float().norm().clamp_min(1e-12))
    absolute = float(difference.abs().max())
    agreement = float((actual.argmax(-1)==expected.argmax(-1)).float().mean())
    return {"relative_l2": relative, "max_abs": absolute, "top_token_agreement": agreement,
            "bitwise_equal": bool(torch.equal(actual, expected)), "pass": relative <= .002 and absolute <= .05 and agreement >= .99}
