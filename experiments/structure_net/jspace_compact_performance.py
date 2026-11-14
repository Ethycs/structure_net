"""Reserved, append-only profiling and paired timing of recovered compact blocks."""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import shutil
import statistics
import time

import numpy as np
import torch

from . import jspace_qwen_affine as base
from .jspace_bench_data import DATASETS, sha, write
from .jspace_bench_tensor import comparison, model_bytes
from .jspace_full_lease import wait_ready
from .jspace_qwen_affine_block import AffineQwenBlock

ROOT = base.ROOT
PARENT = base.DEFAULT_ROOT
DEFAULT_ROOT = ROOT / "data/experiments/jspace_compact_performance/20261113_v1"
ARMS = ("r1536_h64", "r1536_h64_gb")
CONFIG = {"seed": 861, "warmup": 5, "iterations": 20, "paired_repeats": 5,
          "profile_iterations": 3, "cache_capacity": 416, "development_per_task": 2,
          "maximum_kernel_relative_rmse": .002,
          "scope": "exploratory performance; cached final-block inputs; no training or test-set selection"}
CANDIDATES = {
    "wrapper_control": {"geometry_mode": "fp32", "pad_to_multiple": 0, "enabled": False},
    "padding": {"geometry_mode": "fp32", "pad_to_multiple": 128, "enabled": False},
    "fusion": {"geometry_mode": "fp32", "pad_to_multiple": 0, "enabled": True},
    "geometry": {"geometry_mode": "bf16", "pad_to_multiple": 0, "enabled": False},
    "combined": {"geometry_mode": "bf16", "pad_to_multiple": 128, "enabled": True},
}


def sources():
    return sorted(set(base.source_paths() + [Path(__file__),
        Path(__file__).with_name("jspace_qwen_affine_fast.py"),
        Path(__file__).with_name("jspace_qwen_affine_stack.py"),
        ROOT / "tests/structure_net/test_jspace_compact_performance.py",
        ROOT / "tests/structure_net/test_jspace_qwen_affine_stack.py",
        ROOT / "tests/structure_net/test_jspace_qwen_affine_fast.py"]))


def prepare(root, parent=PARENT):
    base.cpu_only()
    if root.exists():
        raise FileExistsError(root)
    prior = base.verify(parent, inputs=True)
    for arm in ARMS:
        base.checked(parent, "train_" + arm)
    base.checked(parent, "reload")
    inputs = dict(prior["input_sha256"])
    names = ["plan.json", "tasks.json", "fit.pt", "relation.pt", "reload_manifest.json"]
    names += [name for arm in ARMS for name in (arm + ".pt", "training_" + arm + ".json", "train_" + arm + "_manifest.json")]
    inputs.update({str(parent / name): sha(parent / name) for name in names})
    plan = {"study_id": "JSPACE-COMPACT-PERFORMANCE-01", "config": CONFIG, "candidates": CANDIDATES,
            "parent_root": str(parent), "parent_fingerprint": prior["fingerprint"],
            "model": prior["model"], "cache_root": prior["cache_root"], "label_ids": prior["label_ids"],
            "created_at": datetime.now(timezone.utc).isoformat(), "input_sha256": inputs,
            "source_sha256": {str(p): sha(p) for p in sources()}}
    plan["fingerprint"] = base.digest(plan)
    root.mkdir(parents=True)
    for p in sources():
        target = root / "producer" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    write(root / "plan.json", plan)
    return {"root": str(root), "fingerprint": plan["fingerprint"]}


def verify(root, inputs=False, model=False):
    plan = json.loads((root / "plan.json").read_text())
    if (plan["config"] != CONFIG or plan["candidates"] != CANDIDATES or
            plan["fingerprint"] != base.digest({k: v for k, v in plan.items() if k != "fingerprint"})):
        raise ValueError("Changed performance protocol")
    for name, expected in plan["source_sha256"].items():
        if sha(name) != expected or sha(root / "producer" / Path(name).relative_to(ROOT)) != expected:
            raise ValueError("Changed performance source: " + name)
    if inputs:
        for name, expected in plan["input_sha256"].items():
            if sha(name) != expected:
                raise ValueError("Changed performance input: " + name)
    if model:
        for name, expected in plan["model"]["files"].items():
            if sha(Path(plan["model"]["path"]) / name) != expected:
                raise ValueError("Changed native weights")
    return plan


def checked(root, name):
    plan = verify(root)
    value = json.loads((root / (name + "_manifest.json")).read_text())
    if value["status"] != "completed" or value["fingerprint"] != plan["fingerprint"]:
        raise ValueError("Incomplete performance stage")
    for name, expected in value["artifact_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Changed performance artifact: " + name)
    return value


def selected_cases(items):
    result = [c for f in DATASETS for c in [x for x in items if x["split"] == "development" and x["family"] == f][:CONFIG["development_per_task"]]]
    if len(result) != len(DATASETS) * CONFIG["development_per_task"]:
        raise ValueError("Insufficient balanced development inputs")
    return result


def paired_orders(repeats):
    if repeats < 1:
        raise ValueError("Positive repeat count required")
    return [(repeat, order) for repeat in range(repeats)
            for order in (("reference", "candidate"), ("candidate", "reference"))]


def numerical_error(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError("Output shapes differ")
    a, b = reference.detach().double(), candidate.detach().double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Nonfinite benchmark output")
    delta = b - a
    rmse = delta.square().mean().sqrt()
    return {"max_abs": float(delta.abs().max()), "rmse": float(rmse),
            "relative_rmse": float(rmse / a.square().mean().sqrt().clamp_min(1e-30)),
            "equal": bool(torch.equal(reference, candidate))}


def candidate_factory(block, name, enabled=None):
    from .jspace_qwen_affine_fast import FastAffineQwenBlock
    options = dict(CANDIDATES[name])
    if enabled is not None:
        options["enabled"] = enabled
    return FastAffineQwenBlock.from_packed(copy.deepcopy(block), **options).eval()


def ensure_fusion(block, name, x):
    if CANDIDATES[name]["enabled"] and not block._can_fuse(x):
        raise RuntimeError("Requested fusion would silently bypass Triton")
    return bool(CANDIDATES[name]["enabled"])


class Stage(base.Stage):
    def __init__(self, root, name):
        self.plan = verify(root, inputs=True, model=True)
        self.lease = wait_ready([0]); self.guard = wait_ready
        self.root, self.name, self.path = root, name, root / name
        self.path.mkdir(exist_ok=False); self.started = time.perf_counter(); self.counts = Counter()
        torch.set_num_threads(4); torch.manual_seed(CONFIG["seed"]); torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        self.parent = Path(self.plan["parent_root"])
        self.model, self.tokenizer = base.load_small_qwen(self.plan["model"]["path"], device="cuda:0", dtype=torch.bfloat16)
        self.native = copy.deepcopy(self.model.model.layers[-1]).cpu()
        self.native_norm = {k: v.detach().cpu().clone() for k, v in self.model.model.norm.state_dict().items()}
        self.tasks = json.loads((self.parent / "tasks.json").read_text())["splits"]
        self.label_ids = torch.tensor(self.plan["label_ids"], device="cuda:0")
        cache = Path(self.plan["cache_root"])
        self.items = json.loads((cache / "items.json").read_text())["items"]
        self.input = np.load(cache / "boundary_35.npy", mmap_mode="r")
        self.targets = np.load(cache / "teacher_logits.npy", mmap_mode="r")
        self.indices = {c["identity"]: i for i, c in enumerate(self.items)}
        self.fit = torch.load(self.parent / "fit.pt", map_location="cpu", weights_only=True)
        self.relation = torch.load(self.parent / "relation.pt", map_location="cpu", weights_only=True)
        write(self.path / "lease.json", {k: self.lease[k] for k in ("id", "holder", "allocation", "state", "ready", "end")})

    def install_arm(self, arm):
        self.reset()
        if arm == "native":
            return self.model.model.layers[-1]
        spec = base.ARMS[arm]
        block = AffineQwenBlock(self.model.model.layers[-1], self.fit["basis"][:, :spec["rank"]].cuda(),
            self.fit["means"].cuda(), trainable=False, parameter_dtype=torch.bfloat16,
            head_projection={"rope_pairs": self.fit["rope_pairs"].cuda(), "value_basis": self.fit["value_basis"].cuda()})
        if spec.get("groebner"):
            block.compile_polynomial(self.relation, self.relation["removed"])
        block.pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        state = torch.load(self.parent / (arm + ".pt"), map_location="cpu", weights_only=True)
        block.load_state_dict(state["block"], strict=True); self.model.model.norm.load_state_dict(state["norm"])
        self.model.model.layers[-1] = block; self.model.eval().requires_grad_(False)
        return block

    def finish(self, result):
        self.guard([0]); verify(self.root)
        peak = torch.cuda.max_memory_reserved()
        if peak > sum(self.lease["allocation"].values()) * 1024**2:
            raise RuntimeError("Reservation exceeded")
        write(self.path / "result.json", result)
        value = {"status": "completed", "fingerprint": self.plan["fingerprint"], "stage": self.name,
                 "elapsed_seconds": time.perf_counter() - self.started, "peak_reserved_bytes": peak,
                 "lease_id": self.lease["id"], "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
                 "artifact_sha256": {str(p.relative_to(self.root)): sha(p) for p in self.path.rglob("*") if p.is_file()}}
        write(self.root / (self.name + "_manifest.json"), value)
        return {k: v for k, v in value.items() if k != "artifact_sha256"}


@torch.inference_mode()
def block_call(stage, block, x, mode):
    """One fixed input, with genuine StaticCache context for the final-token step."""
    from transformers import StaticCache
    length = x.shape[1]; positions = torch.arange(length, device=x.device)
    capacity = CONFIG["cache_capacity"]
    if not 1 <= length <= capacity:
        raise ValueError("Input length outside the frozen capacity")
    cache = None
    if mode == "decode":
        cache = StaticCache(stage.model.config, max_batch_size=1, max_cache_len=capacity, device=x.device, dtype=x.dtype)
        if length > 1:
            p = positions[:-1]
            mask = torch.where(torch.arange(capacity, device=x.device)[None] <= p[:, None], 0., torch.finfo(x.dtype).min).to(x.dtype)[None, None]
            block(x[:, :-1], attention_mask=mask, position_ids=p[None], cache_position=p,
                  position_embeddings=stage.model.model.rotary_emb(x[:, :-1], p[None]), past_key_value=cache, use_cache=True)
        x = x[:, -1:].contiguous(); positions = positions[-1:]
        mask = torch.where(torch.arange(capacity, device=x.device) <= positions[0], 0., torch.finfo(x.dtype).min).to(x.dtype)[None, None, None]
    elif mode == "prefill":
        mask = torch.full((length, length), torch.finfo(x.dtype).min, device=x.device, dtype=x.dtype).triu(1)[None, None]
    else:
        raise ValueError("Unknown block mode")
    kwargs = {"attention_mask": mask, "position_ids": positions[None], "cache_position": positions,
              "position_embeddings": stage.model.model.rotary_emb(x, positions[None]), "past_key_value": cache,
              "use_cache": cache is not None}
    return lambda: block(x, **kwargs)[0]


@torch.inference_mode()
def warm_and_graph(call):
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    started = time.perf_counter()
    with torch.cuda.stream(stream):
        for _ in range(CONFIG["warmup"]):
            call()
    torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
    warm_seconds = time.perf_counter() - started
    graph = torch.cuda.CUDAGraph(); started = time.perf_counter()
    with torch.cuda.graph(graph):
        output = call()
    torch.cuda.synchronize()
    capture_seconds = time.perf_counter() - started
    expected = call().clone(); graph.replay(); torch.cuda.synchronize()
    error = numerical_error(expected, output)
    if not error["equal"]:
        raise ValueError("Graph replay differs from its own eager implementation")
    return graph, {"warm_seconds": warm_seconds, "capture_seconds": capture_seconds, "graph_error": error}


@torch.inference_mode()
def timed(call):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); wall = time.perf_counter(); start.record()
    for _ in range(CONFIG["iterations"]):
        call()
    end.record(); end.synchronize()
    return {"wall_seconds": (time.perf_counter() - wall) / CONFIG["iterations"],
            "cuda_event_seconds": start.elapsed_time(end) / (1000 * CONFIG["iterations"])}


def timing_summary(rows):
    result = {}
    for execution in ("eager", "graph"):
        values = {key: {metric: statistics.median(row[metric] for row in rows
                     if row["variant"] == key and row["execution"] == execution)
                     for metric in ("wall_seconds", "cuda_event_seconds")}
                  for key in ("reference", "candidate")}
        result[execution] = {**values, "wall_speedup": values["reference"]["wall_seconds"] / values["candidate"]["wall_seconds"],
                            "cuda_event_speedup": values["reference"]["cuda_event_seconds"] / values["candidate"]["cuda_event_seconds"]}
    return result


@torch.inference_mode()
def profile_call(call, target):
    for _ in range(CONFIG["warmup"]):
        call()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(CONFIG["profile_iterations"]):
            call()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(target))
    kernels = Counter(e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    if not kernels:
        raise RuntimeError("CUDA profiler returned no kernel events")
    operators = [{"name": e.key, "count": e.count, "input_shapes": e.input_shapes,
                  "self_cpu_us": e.self_cpu_time_total, "self_device_us": e.self_device_time_total}
                 for e in prof.key_averages(group_by_input_shape=True)]
    return {"kernel_count": sum(kernels.values()), "kernels": dict(kernels), "iterations": CONFIG["profile_iterations"],
            "top_operators": sorted(operators, key=lambda x: x["self_device_us"], reverse=True)[:40],
            "top_cpu_operators": sorted(operators, key=lambda x: x["self_cpu_us"], reverse=True)[:40]}


def runtime_candidate_valid(result):
    kernel_errors = [p["kernel_error"] for p in result["pairs"] if p["kernel_error"] is not None]
    errors = [result["choice_logit_error"], *(p["output_error"] for p in result["pairs"]), *kernel_errors]
    finite = all(math.isfinite(e[key]) for e in errors for key in ("max_abs", "rmse", "relative_rmse"))
    faithful = all(e["relative_rmse"] <= CONFIG["maximum_kernel_relative_rmse"] for e in kernel_errors)
    return finite and faithful and bool(result["pairs"]) and all(
        pair["reference_graph"]["graph_error"]["equal"] and pair["candidate_graph"]["graph_error"]["equal"]
        for pair in result["pairs"])


@torch.inference_mode()
def run(root, mode="profile", candidate="combined", arm="r1536_h64_gb"):
    if mode not in ("profile", "bench", "full") or candidate not in CANDIDATES or arm not in ARMS:
        raise ValueError("Unknown benchmark configuration")
    if mode == "bench":
        checked(root, "profile")
    if mode == "full":
        checked(root, "bench")
        previous = json.loads((root / "bench/result.json").read_text())[arm + "/" + candidate]
        if not runtime_candidate_valid(previous):
            raise ValueError("Finite outputs, kernel fidelity, and graph parity required before full-model timing")
    stage = Stage(root, mode if mode != "full" else "full_" + arm + "_" + candidate)
    cases, results = selected_cases(stage.items), {}
    if mode == "profile":
        for tag in ("native", *ARMS):
            block = stage.install_arm(tag); rows = []
            for index, case in enumerate(cases):
                x = stage.x(case)
                for operation in ("prefill", "decode"):
                    call = block_call(stage, block, x, operation)
                    value = profile_call(call, stage.path / f"{tag}_{index}_{operation}.json")
                    rows.append({"identity": case["identity"], "tokens": x.shape[1], "mode": operation, **value})
                    del call
            results[tag] = rows; stage.event("profile", arm=tag)
    elif mode == "bench":
        for tag in ARMS:
            reference = stage.install_arm(tag)
            baseline, baseline_rows = base.development(stage, tag + "_baseline")
            for name in CANDIDATES:
                stage.event("candidate_start", arm=tag, candidate=name)
                optimized = candidate_factory(reference, name); pairs = []
                torch_reference = candidate_factory(reference, name, enabled=False) if CANDIDATES[name]["enabled"] else None
                profiles = {}
                for index, case in enumerate(cases):
                    x = stage.x(case)
                    ensure_fusion(optimized, name, x)
                    for operation in ("prefill", "decode"):
                        calls = {"reference": block_call(stage, reference, x, operation), "candidate": block_call(stage, optimized, x, operation)}
                        graphs, setup = {}, {}
                        for key, call in calls.items():
                            graphs[key], setup[key] = warm_and_graph(call)
                        if index == 0:
                            profiles[operation] = profile_call(calls["candidate"], stage.path / f"{tag}_{name}_{operation}_trace.json")
                        error = numerical_error(calls["reference"](), calls["candidate"]()); timings = []
                        kernel_error = None if torch_reference is None else numerical_error(
                            block_call(stage, torch_reference, x, operation)(), calls["candidate"]())
                        for repeat, order in paired_orders(CONFIG["paired_repeats"]):
                            for execution in ("eager", "graph"):
                                for key in order:
                                    timings.append({"repeat": repeat, "order": list(order), "execution": execution, "variant": key,
                                        **timed(calls[key] if execution == "eager" else graphs[key].replay)})
                        pairs.append({"identity": case["identity"], "tokens": x.shape[1], "mode": operation, "output_error": error,
                                      "reference_graph": setup["reference"], "candidate_graph": setup["candidate"],
                                      "kernel_error": kernel_error, "summary": timing_summary(timings), "timings": timings})
                        del calls, graphs; gc.collect(); torch.cuda.empty_cache(); stage.guard([0])
                stage.model.model.layers[-1] = optimized
                dev, rows = base.development(stage, tag + "_" + name)
                logits_error = numerical_error(torch.tensor([v for r in baseline_rows for v in r["logits"]]),
                                               torch.tensor([v for r in rows for v in r["logits"]]))
                value = {"options": CANDIDATES[name], "baseline_development": baseline, "development": dev,
                         "development_comparison": comparison(baseline_rows, rows), "choice_logit_error": logits_error,
                         "tensor_bytes": model_bytes(stage.model), "pairs": pairs, "profiles": profiles,
                         "fusion_path_verified": bool(CANDIDATES[name]["enabled"])}
                value["eligible_for_full_runtime"] = runtime_candidate_valid(value)
                value["needs_recovery"] = value["development_comparison"]["agreement"] < 1.
                value["needs_recovery_definition"] = "Changed development choices; diagnostic only, never an inference-timing gate"
                results[tag + "/" + name] = value
                write(stage.path / (tag + "_" + name + "_result.json"), value)
                stage.model.model.layers[-1] = reference; del optimized, torch_reference; stage.event("candidate", arm=tag, candidate=name)
    else:
        stage.install_arm("native"); results["native"] = base.runtime(stage, "native")
        reference = stage.install_arm(arm); results["reference"] = base.runtime(stage, "reference")
        stage.model.model.layers[-1] = candidate_factory(reference, candidate)
        ensure_fusion(stage.model.model.layers[-1], candidate, stage.x(cases[0]))
        results["candidate"] = base.runtime(stage, "candidate")
        stage.model.model.layers[-1] = reference; results["reference_repeat"] = base.runtime(stage, "reference_repeat")
        stage.install_arm("native"); results["native_repeat"] = base.runtime(stage, "native_repeat")
    return stage.finish(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run")); parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--parent", type=Path, default=PARENT); parser.add_argument("--mode", choices=("profile", "bench", "full"), default="profile")
    parser.add_argument("--candidate", choices=CANDIDATES, default="combined"); parser.add_argument("--arm", choices=ARMS, default="r1536_h64_gb")
    args = parser.parse_args()
    result = prepare(args.root, args.parent) if args.command == "prepare" else run(args.root, args.mode, args.candidate, args.arm)
    print(json.dumps(result, allow_nan=False), flush=True)
