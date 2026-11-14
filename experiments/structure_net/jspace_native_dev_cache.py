"""Cache the exact FP32 native development prefix for isolated layer tests.

All37 boundaries retain full valid-token states. This cache is exclusively the
existing64 development cases; original BF16 training/teacher banks are untouched.
Execution follows stream_score's batching, mask, precision, and native readout.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from . import jspace_quadratic_recipe as recipe
from . import jspace_quadratic_evaluate as scoring
from .jspace_single_block_verify import validate_records
from .jspace_semantic_recovery import ranking_summary

SCHEMA = "jspace.native-fp32-development-cache.v1"
PROTOCOL = recipe.ROOT/"docs/07 - Status Reports/jspace/2026-11-14_qwen-parallel-single-layer-recovery.md"


def development_items(source_items):
    """Preserve identities/order but replace576-case offsets with dev-only ones."""
    result, offset = [], 0
    for case in source_items:
        if case["split"] != "development":
            continue
        length = len(case["input_ids"])
        if not length or length > 384:
            raise ValueError("Development prompts must contain1..384 tokens")
        result.append(dict(case, source_offset=case["offset"], source_stop=case["stop"],
            offset=offset, stop=offset+length))
        offset += length
    if len({c["identity"] for c in result}) != len(result) or not result:
        raise ValueError("Distinct development identities required")
    return dict(items=result, total_tokens=offset, layout="contiguous development-only valid-token rows")


@torch.inference_mode()
def capture(cases, *, input_embedding, block_loader, final_readout, label_ids,
            directory, rotary, layer_count=36, device="cpu", microbatch=4,
            guard=lambda:None, progress=lambda **_k:None):
    """One native block resident at a time; keep every FP32 boundary on disk."""
    directory, device = Path(directory), torch.device(device)
    if directory.exists():
        raise FileExistsError(directory)
    if type(layer_count) is not int or layer_count < 1 or type(microbatch) is not int or microbatch < 1:
        raise ValueError("Positive layer_count and microbatch required")
    lengths = scoring._validate_cases(cases, input_embedding, label_ids)
    offsets = np.cumsum([0, *lengths], dtype=np.int64)
    shape = (int(offsets[-1]), input_embedding.shape[1])
    directory.mkdir(parents=True)
    source = destination = None
    try:
        source = np.lib.format.open_memmap(directory/"boundary_00.npy", mode="w+", dtype=np.float32, shape=shape)
        for i, case in enumerate(cases):
            guard()
            hidden = input_embedding.index_select(0, torch.tensor(case["input_ids"])).float()
            if not torch.isfinite(hidden).all():
                raise FloatingPointError("Nonfinite native embedding")
            source[offsets[i]:offsets[i+1]] = hidden.cpu().numpy()
        source.flush()
        for layer in range(layer_count):
            guard()
            block = block_loader(layer).eval().requires_grad_(False)
            if any(t.device != device or t.is_floating_point() and t.dtype != torch.float32
                   for t in (*block.parameters(), *block.buffers())):
                raise ValueError("Native blocks must be FP32 on the selected device")
            destination = np.lib.format.open_memmap(directory/f"boundary_{layer+1:02d}.npy",
                mode="w+", dtype=np.float32, shape=shape)
            for start in range(0, len(cases), microbatch):
                guard()
                indices = list(range(start, min(start+microbatch, len(cases))))
                longest = max(lengths[i] for i in indices)
                host = np.zeros((len(indices), longest, shape[1]), dtype=np.float32)
                for row, i in enumerate(indices):
                    host[row, :lengths[i]] = source[offsets[i]:offsets[i+1]]
                hidden = torch.from_numpy(host).to(device)
                positions = torch.arange(longest, device=device)
                position_ids = positions[None].expand(len(indices), -1)
                key_valid = positions[None] < torch.tensor([lengths[i] for i in indices], device=device)[:, None]
                causal = positions[None, :] <= positions[:, None]
                allowed = key_valid[:, None, None, :] & causal[None, None]
                mask = torch.zeros(allowed.shape, device=device, dtype=torch.float32).masked_fill_(
                    ~allowed, torch.finfo(torch.float32).min)
                embeddings = rotary(hidden, position_ids)
                value = block(hidden, attention_mask=mask, position_ids=position_ids,
                    cache_position=positions, position_embeddings=embeddings, use_cache=False)
                value = value[0] if isinstance(value, tuple) else value
                if value.shape != hidden.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise FloatingPointError("Native block returned invalid FP32 states")
                for row, i in enumerate(indices):
                    destination[offsets[i]:offsets[i+1]] = value[row, :lengths[i]].cpu().numpy()
                del value, hidden, embeddings, host
            destination.flush()
            source._mmap.close()
            source, destination = destination, None
            del block
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            progress(layer=layer, boundary=layer+1)
        reader = scoring._reader(final_readout, label_ids, device)
        records = []
        for start in range(0, len(cases), microbatch):
            guard()
            indices = list(range(start, min(start+microbatch, len(cases))))
            hidden = torch.from_numpy(np.array(source[offsets[np.array(indices)+1]-1], copy=True)).to(device)
            logits = reader(hidden).float().cpu()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite native choice logits")
            for row, i in enumerate(indices):
                case = cases[i]
                choices = logits[row, :len(case["choices"])]
                records.append(dict(identity=case["identity"], family=case["family"], answer=case["answer"],
                    prediction=int(choices.argmax()), logits=choices.tolist()))
        return dict(records=records, shape=list(shape), boundaries=layer_count+1,
            dtype="float32", microbatch=microbatch, max_live_decoder_blocks=1)
    finally:
        for bank in (source, destination):
            if bank is not None and not bank._mmap.closed:
                bank._mmap.close()


def prepare(root, source, native_baseline):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide CUDA for CPU preparation")
    root, source, native_baseline = map(lambda p:Path(p).resolve(), (root, source, native_baseline))
    if root.exists():
        raise FileExistsError(root)
    original = recipe.verify(source)
    items = development_items(recipe.read(source/"items.json")["items"])
    cases = items["items"]
    if len(cases) != 64:
        raise ValueError("Only the original64 development cases are allowed")
    baseline = recipe.read(native_baseline)
    baseline_plan = recipe.read(native_baseline.parent/"plan.json")
    baseline_manifest = recipe.read(native_baseline.parent/"manifest.json")
    if (baseline_plan.get("variant") != "full_rank_full_head" or baseline_plan.get("dtype") != "float32"
            or baseline_plan["fingerprint"] != recipe.digest({k:v for k,v in baseline_plan.items() if k != "fingerprint"})
            or baseline_plan.get("source_fingerprint") != original["fingerprint"]
            or baseline_manifest.get("status") != "completed"
            or baseline_manifest["fingerprint"] != baseline_plan["fingerprint"]
            or baseline.get("fingerprint") != baseline_plan["fingerprint"]
            or baseline_manifest["artifacts"]["result.json"] != recipe.sha(native_baseline)):
        raise ValueError("Authenticated FP32 native-equivalent baseline required")
    validate_records(baseline["records"], cases)
    if sum(r["prediction"] == r["answer"] for r in baseline["records"]) != 50:
        raise ValueError("Declared native development control is50/64")
    inputs = {str(p):recipe.sha(p) for p in (source/"plan.json", source/"items.json", native_baseline,
        native_baseline.parent/"plan.json", native_baseline.parent/"manifest.json")}
    sources = dict(original["source_sha256"])
    for path in (Path(__file__), recipe.ROOT/"tests/structure_net/test_jspace_native_dev_cache.py", PROTOCOL,
            Path(__file__).with_name("jspace_qwen_parallel_model.py"),
            Path(__file__).with_name("jspace_single_block_reserved.py"),
            Path(__file__).with_name("jspace_single_block_verify.py"),
            Path(__file__).with_name("jspace_semantic_recovery.py"),
            Path(__file__).with_name("jspace_joint_recovery.py"),
            Path(__file__).with_name("jspace_single_block_distill.py"),
            Path(__file__).with_name("jspace_full_lease.py")):
        sources[str(path)] = recipe.sha(path)
    plan = dict(schema=SCHEMA, source=str(source), source_fingerprint=original["fingerprint"],
        native_baseline=str(native_baseline), native_correct=50, count=64, layer_count=36, width=2048,
        dtype="FP32 arithmetic from original BF16-rounded weights", microbatch=4,
        items_digest=recipe.digest(items), case_identities=[c["identity"] for c in cases],
        total_tokens=items["total_tokens"], confirmation_used=False,
        compatibility_gate="Identical64 predictions to full-rank/full-head native-equivalent control; its logits may differ",
        consumer_gate="Native layer substitution plus suffix must reproduce THESE pure-native FP32 logits/predictions",
        input_sha256=inputs, input_identity={p:recipe.file_identity(p) for p in inputs}, source_sha256=sources)
    plan["fingerprint"] = recipe.digest(plan)
    root.mkdir(parents=True)
    for path, expected in sources.items():
        if recipe.sha(path) != expected:
            raise ValueError("Changed native-cache producer")
        destination = root/"producer"/Path(path).relative_to(recipe.ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    recipe.write(root/"plan.json", plan)
    recipe.write(root/"items.json", items)
    return dict(status="prepared", fingerprint=plan["fingerprint"], shape=[items["total_tokens"], 2048])


def verify(root, completed=False):
    root = Path(root)
    plan = recipe.read(root/"plan.json")
    if plan.get("schema") != SCHEMA or plan["fingerprint"] != recipe.digest({k:v for k,v in plan.items() if k != "fingerprint"}):
        raise ValueError("Changed native development cache plan")
    if recipe.verify(plan["source"])["fingerprint"] != plan["source_fingerprint"]:
        raise ValueError("Changed original native source")
    if recipe.digest(recipe.read(root/"items.json")) != plan["items_digest"]:
        raise ValueError("Changed development-only item layout")
    for path, expected in plan["source_sha256"].items():
        if recipe.sha(path) != expected or recipe.sha(root/"producer"/Path(path).relative_to(recipe.ROOT)) != expected:
            raise ValueError("Changed native cache producer")
    for path, identity in plan["input_identity"].items():
        if recipe.file_identity(path) != identity:
            raise ValueError("Changed native cache input")
    if completed:
        manifest = recipe.read(root/"cache/manifest.json")
        expected_names = {f"cache/boundary_{i:02d}.npy" for i in range(37)} | {"cache/result.json", "items.json"}
        if (manifest.get("status") != "completed" or manifest.get("fingerprint") != plan["fingerprint"]
                or set(manifest.get("artifact_sha256", {})) != expected_names):
            raise ValueError("Complete matching37-boundary native cache required")
        for name, expected in manifest["artifact_sha256"].items():
            if recipe.sha(root/name) != expected:
                raise ValueError("Changed native cache artifact: "+name)
        result = recipe.read(root/"cache/result.json")
        if (result.get("status") != "completed" or result.get("fingerprint") != plan["fingerprint"]
                or result.get("correct") != 50 or result.get("count") != 64
                or result.get("native_baseline_predictions_exact") is not True):
            raise ValueError("Native cache did not reproduce the development control")
    return plan


def run(root):
    from .jspace_full_lease import wait_ready
    from .jspace_qwen_parallel_model import load_native_block, load_readout_norm
    from .jspace_single_block_reserved import allocator_settings
    from .jspace_joint_recovery import check_reserved_peak
    root = Path(root).resolve()
    plan = verify(root)
    if (root/"cache").exists():
        raise FileExistsError(root/"cache")
    lease = None
    started = time.monotonic()
    try:
        lease = wait_ready([0])
        settings = allocator_settings(lease, dict(headroom_mib=256, allocator_backend="native"))
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        def guard():
            current = wait_ready([0])
            if current["id"] != lease["id"] or current["allocation"] != lease["allocation"]:
                raise RuntimeError("Native cache reservation changed")
            check_reserved_peak(lease["allocation"], torch.cuda.max_memory_reserved())
        original = recipe.read(Path(plan["source"])/"plan.json")
        embedding, norm, _ = load_readout_norm(original["model"]["path"], device="cpu", dtype=torch.bfloat16)
        embedding, norm = embedding.float(), norm.float()
        def loader(layer):
            guard()
            return load_native_block(original["model"]["path"], layer, device="cuda:0", dtype=torch.bfloat16).float()
        cases = recipe.read(root/"items.json")["items"]
        value = capture(cases, input_embedding=embedding, block_loader=loader, final_readout=(norm, embedding),
            label_ids=original["label_ids"], directory=root/"cache", rotary=recipe.rotary_module(original, "cuda:0"),
            device="cuda:0", microbatch=4, guard=guard,
            progress=lambda **v:recipe.event(root, "boundary", **v))
        baseline = recipe.read(plan["native_baseline"])["records"]
        validate_records(value["records"], cases)
        same = [a["prediction"] == b["prediction"] for a,b in zip(value["records"], baseline)]
        summary = ranking_summary(value["records"])
        result = dict(**value, **summary, status="completed", fingerprint=plan["fingerprint"],
            native_baseline_prediction_agreement=sum(same)/len(same), native_baseline_predictions_exact=all(same),
            native_baseline_max_logit_difference=max(abs(x-y) for a,b in zip(value["records"], baseline)
                for x,y in zip(a["logits"], b["logits"])), confirmed=False,
            scope="Pure native FP32 development prefixes; no independent-layer or assembled-model quality claim")
        recipe.write(root/"cache/result.json", result)
        if not all(same) or result["correct"] != 50 or result["shape"] != [plan["total_tokens"], 2048]:
            raise RuntimeError("Pure-native FP32 cache did not reproduce the declared50/64 predictions/layout")
        names = [f"cache/boundary_{i:02d}.npy" for i in range(37)] + ["cache/result.json", "items.json"]
        guard()
        manifest = dict(status="completed", fingerprint=plan["fingerprint"],
            artifact_sha256={name:recipe.sha(root/name) for name in names},
            lease_id=lease["id"], allocation=lease["allocation"], allocator=settings,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), elapsed_seconds=time.monotonic()-started)
        recipe.write(root/"cache/manifest.json", manifest)
        return dict(status="completed", fingerprint=plan["fingerprint"], correct=result["correct"], count=64)
    except Exception as error:
        recipe.write(root/"failure.json", dict(status="failed", error_type=type(error).__name__, error=str(error),
            fingerprint=plan["fingerprint"], lease_id=None if lease is None else lease["id"]))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--native-baseline", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare" and (args.source is None or args.native_baseline is None):
        parser.error("prepare requires --source and --native-baseline")
    value = prepare(args.root, args.source, args.native_baseline) if args.stage == "prepare" else run(args.root)
    print(json.dumps(value, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
