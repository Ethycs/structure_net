"""Greedy native-target projection through the frozen36 update subspaces.

This uses original teacher hidden states at every layer. It is neither a
deployable model nor a task-accuracy upper bound for the learned student.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from . import jspace_recovered_assembly as assembly
from . import jspace_native_dev_cache as cache
from .jspace_qwen_parallel_evaluate import paired_quality

recipe = assembly.recipe
SCHEMA = "jspace.stacked-projection-oracle.v1"
PROTOCOL = recipe.ROOT/"docs/07 - Status Reports/jspace/2026-11-14_qwen-stacked-projection-oracle.md"
SCOPE = "Development-only teacher-hidden-state oracle; not deployable, not learned recovery, not a task-accuracy upper bound"


def project_step(current, native_next, basis):
    if (current.ndim != 2 or native_next.shape != current.shape or basis.ndim != 2
            or basis.shape[0] != current.shape[1] or not 0 < basis.shape[1] <= basis.shape[0]
            or current.dtype not in (torch.float32, torch.float64)
            or any(t.dtype != current.dtype or t.device != current.device for t in (native_next, basis))):
        raise ValueError("Matching full-width states and a same-device floating update basis required")
    if any(not torch.isfinite(t).all() for t in (current, native_next, basis)):
        raise FloatingPointError("Nonfinite oracle input")
    tolerance = 1e-10 if basis.dtype == torch.float64 else 2e-4
    if not torch.allclose(basis.T@basis, torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device),
                          atol=tolerance, rtol=tolerance):
        raise ValueError("Oracle projection requires orthonormal basis columns")
    # Correct the ACTUAL accumulated oracle state, not the native layer delta.
    return current + ((native_next-current)@basis)@basis.T


def validate_layout(cases, rows):
    offset = 0
    if not cases or len({c["identity"] for c in cases}) != len(cases): raise ValueError("Distinct development cases required")
    for c in cases:
        if (c["split"] != "development" or c["offset"] != offset
                or c["stop"]-offset != len(c["input_ids"]) or not 1 <= len(c["input_ids"]) <= 384):
            raise ValueError("Contiguous valid-token development-only layout required")
        offset = c["stop"]
    if offset != rows: raise ValueError("Boundary bank does not match development token count")


def relative_rms(current, target, cases):
    ratio = (current-target).square().sum(-1)/target.square().sum(-1).clamp_min(1e-20)
    per_case = torch.stack([ratio[c["offset"]:c["stop"]].mean() for c in cases])
    answers = torch.tensor([c["stop"]-1 for c in cases], device=ratio.device)
    return dict(valid_token_relative_rms=float(per_case.mean().sqrt()),
        answer_relative_rms=float(ratio[answers].mean().sqrt()))


@torch.inference_mode()
def evaluate(cases, *, boundary_loader, basis_loader, reader, layers=36, device="cpu", guard=None, progress=lambda **_k:None):
    device = torch.device(device)
    if device.type == "cuda" and guard is None: raise ValueError("Live reservation guard required before GPU evaluation")
    if type(layers) is not int or layers < 1: raise ValueError("Positive layer count required")
    guard = guard or (lambda:None)
    def boundary(index):
        guard(); value = boundary_loader(index)
        value = value.to(device) if isinstance(value, torch.Tensor) else torch.from_numpy(np.array(value, copy=True)).to(device)
        if value.ndim != 2 or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("Finite full-width FP32 boundary banks required")
        return value
    current = boundary(0); validate_layout(cases, len(current)); shape = list(current.shape)
    trajectory = [dict(boundary=0, valid_token_relative_rms=0., answer_relative_rms=0.)]
    for layer in range(layers):
        guard(); target = boundary(layer+1); basis = basis_loader(layer).to(device)
        current = project_step(current, target, basis)
        if not torch.isfinite(current).all(): raise FloatingPointError("Nonfinite projected oracle state")
        trajectory.append(dict(boundary=layer+1, **relative_rms(current, target, cases)))
        progress(layer=layer, **trajectory[-1]); del basis
        if layer < layers-1: del target
    def score(states):
        records = []
        for start in range(0, len(cases), 4):
            guard(); selected = cases[start:start+4]
            answer_rows = torch.tensor([c["stop"]-1 for c in selected], device=device)
            logits = reader(states[answer_rows]).float().cpu()
            if not torch.isfinite(logits).all(): raise FloatingPointError("Nonfinite oracle choice logits")
            for c, values in zip(selected, logits):
                values = values[:len(c["choices"])]
                records.append(dict(identity=c["identity"], family=c["family"], answer=c["answer"],
                    prediction=int(values.argmax()), logits=values.tolist()))
        return dict(records=records, **cache.ranking_summary(records))
    native, oracle = score(target), score(current)
    return dict(native=native, oracle=oracle, paired=paired_quality(native["records"], oracle["records"]),
        trajectory=trajectory, shape=shape, layers=layers, confirmed=False, scope=SCOPE,
        error_definition="sqrt(mean_cases(mean_valid_tokens(||oracle-native||^2 / max(||native||^2,1e-20)))); answer uses last valid row")


def prepare(root, assembly_root, native_cache_root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise ValueError("Hide CUDA for CPU preparation")
    root, assembly_root, native_cache_root = map(lambda p:Path(p).resolve(), (root, assembly_root, native_cache_root))
    if root.exists(): raise FileExistsError(root)
    parent = assembly.verify(assembly_root); cached = cache.verify(native_cache_root, completed=True)
    if cached["source_fingerprint"] != parent["source_fingerprint"]: raise ValueError("Native cache and bases must share the original model/source")
    items = recipe.read(native_cache_root/"items.json"); cases = items["items"]
    validate_layout(cases, items["total_tokens"])
    fields = ("identity", "family", "answer", "choices", "input_ids")
    if len(cases) != 64 or any(tuple(a[k] for k in fields) != tuple(b[k] for k in fields)
        for a,b in zip(cases, parent["cases"])) or len(parent["cases"]) != 64:
        raise ValueError("Only the original matched64 development cases are allowed")
    manifest = recipe.read(native_cache_root/"cache/manifest.json")
    inputs = {str(native_cache_root/name):digest for name,digest in manifest["artifact_sha256"].items()}
    fits = []
    for layer in range(36):
        path = Path(parent["blocks"][str(layer)]["fit_path"]); expected = parent["input_sha256"][str(path)]
        assembly._pin(inputs, path, expected); fits.append(str(path))
    for path in (assembly_root/"plan.json", native_cache_root/"plan.json", native_cache_root/"cache/manifest.json"):
        assembly._pin(inputs, path)
    original = recipe.read(Path(parent["source"])/"plan.json")
    for name, expected in parent["model"]["files"].items():
        # Native model bytes are already authenticated by the frozen source;
        # pin their identity here without rereading multi-GB weights during training.
        inputs[str(Path(parent["model"]["path"])/name)] = expected
    sources = dict(parent["source_sha256"]); assembly._merge_hashes(sources, cached["source_sha256"])
    for path in (Path(__file__).resolve(), recipe.ROOT/"tests/structure_net/test_jspace_stacked_projection_oracle.py", PROTOCOL,
                 Path(__file__).with_name("jspace_qwen_parallel_evaluate.py")):
        assembly._merge_hashes(sources, {str(path):recipe.sha(path)})
    plan = dict(schema=SCHEMA, assembly_root=str(assembly_root), assembly_fingerprint=parent["fingerprint"],
        native_cache_root=str(native_cache_root), native_cache_fingerprint=cached["fingerprint"],
        source=parent["source"], source_fingerprint=parent["source_fingerprint"], model=parent["model"],
        basis_paths=fits, cases=cases, label_ids=original["label_ids"], layers=36, shape=[items["total_tokens"],2048],
        native_correct=50, quality_reference_correct=47, scope=SCOPE, confirmed=False,
        recurrence="h_next = h + ((native_next-h) @ U) @ U.T; frozen exact per-layer U; no learned branches",
        input_sha256=inputs, input_identity={p:recipe.file_identity(p) for p in inputs}, source_sha256=sources)
    plan["fingerprint"] = recipe.digest(plan); root.mkdir(parents=True)
    for path, expected in sources.items():
        if recipe.sha(path) != expected: raise ValueError("Changed oracle producer")
        target = root/"producer"/Path(path).relative_to(recipe.ROOT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path,target)
    recipe.write(root/"plan.json", plan)
    return dict(status="prepared", fingerprint=plan["fingerprint"])


def verify(root):
    root = Path(root); plan = recipe.read(root/"plan.json")
    if plan.get("schema") != SCHEMA or plan["fingerprint"] != recipe.digest({k:v for k,v in plan.items() if k != "fingerprint"}):
        raise ValueError("Changed stacked oracle plan")
    for path, digest in plan["source_sha256"].items():
        if recipe.sha(path) != digest or recipe.sha(root/"producer"/Path(path).relative_to(recipe.ROOT)) != digest: raise ValueError("Changed frozen oracle source")
    for path, identity in plan["input_identity"].items():
        if recipe.file_identity(path) != identity: raise ValueError("Changed pinned oracle input")
    return plan


def cap_allocator(lease, *, cuda=None, visible=None):
    from .jspace_single_block_reserved import allocator_settings
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "") if visible is None else visible
    if len(lease["allocation"]) != 1 or not visible.startswith("GPU-8cf440df") or lease["allocation"].get(visible) != 3072:
        raise ValueError("Oracle requires exactly the leased2060 at3072MiB")
    return allocator_settings(lease, dict(headroom_mib=256, allocator_backend="native"), cuda=cuda, visible=visible)


def run(root):
    from .jspace_full_lease import wait_ready
    from .jspace_qwen_parallel_model import load_readout_norm
    root = Path(root).resolve(); plan = verify(root); directory = root/"evaluation"; directory.mkdir(exist_ok=False)
    lease = allocator = None; started = time.monotonic()
    try:
        lease = wait_ready([0]); allocator = cap_allocator(lease)
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.cuda.reset_peak_memory_stats()
        def guard():
            current = wait_ready([0])
            if current["id"] != lease["id"] or current["allocation"] != lease["allocation"]: raise RuntimeError("Oracle lease changed")
            if torch.cuda.max_memory_reserved() > allocator["allocator_limit_bytes"]: raise MemoryError("Oracle exceeded allocator cap")
        embedding, norm, _ = load_readout_norm(plan["model"]["path"], device="cpu", dtype=torch.bfloat16)
        reader = cache.scoring._reader((norm.float(), embedding.float()), plan["label_ids"], "cuda:0")
        del embedding, norm
        native_root = Path(plan["native_cache_root"])
        def bank(index):
            return np.load(native_root/"cache"/f"boundary_{index:02d}.npy", mmap_mode="r", allow_pickle=False)
        result = evaluate(plan["cases"], boundary_loader=bank, basis_loader=lambda i:recipe.load(plan["basis_paths"][i])["basis"],
            reader=reader, device="cuda:0", guard=guard, progress=lambda **kw:recipe.event(root,"oracle_layer",**kw))
        control = assembly.compare_control(result["native"]["records"], recipe.read(native_root/"cache/result.json"), plan["cases"],50)
        result.update(status="completed", fingerprint=plan["fingerprint"], native_control=control,
            reaches_quality_reference=result["oracle"]["correct"] >= 47, quality_reference_is_diagnostic_only=True)
        recipe.write(directory/"result.json",result)
        if not control["passed"] or result["shape"] != plan["shape"]: raise RuntimeError("Native oracle control or bank shape failed")
        guard(); recipe.write(directory/"manifest.json", dict(status="completed", fingerprint=plan["fingerprint"],
            result_sha256=recipe.sha(directory/"result.json"), lease_id=lease["id"], allocation=lease["allocation"], allocator=allocator,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), elapsed_seconds=time.monotonic()-started))
        return dict(status="completed", native_correct=result["native"]["correct"], oracle_correct=result["oracle"]["correct"], fingerprint=plan["fingerprint"])
    except BaseException as error:
        recipe.write(directory/"failure.json", dict(status="failed", fingerprint=plan["fingerprint"], error=repr(error),
            lease_id=None if lease is None else lease["id"], allocation=None if lease is None else lease["allocation"], allocator=allocator))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","run"));parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--assembly-root",type=Path);parser.add_argument("--native-cache-root",type=Path);args=parser.parse_args()
    if args.stage == "prepare" and (args.assembly_root is None or args.native_cache_root is None): parser.error("prepare requires both input roots")
    print(json.dumps(prepare(args.root,args.assembly_root,args.native_cache_root) if args.stage == "prepare" else run(args.root),allow_nan=False),flush=True)
