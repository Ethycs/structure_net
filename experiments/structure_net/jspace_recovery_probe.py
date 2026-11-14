"""Development-only residual-carry intervention; every GPU run requires a lease.

Retain native embeddings, residual activations, final norm and vocabulary. A
compressed block changes only its projected update; no native dense block is
executed at the same layer. This is an accuracy diagnostic, not a speed claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/experiments/jspace_quadratic_recipe/20261114_v1"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for value in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(value)
    return result.hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate(mode, cut, split, limit):
    if mode not in ("softmax", "quadratic") or not 0 <= cut <= 36:
        raise ValueError("Known attention and a prefix cut in [0,36] required")
    if split != "development":
        raise ValueError("Recovery probes may only use development cases")
    if limit is not None and not 0 < limit <= 64:
        raise ValueError("Limit must select 1..64 development cases")


def run(root, source=SOURCE, *, mode="softmax", cut=36, selected=False,
        split="development", limit=None):
    validate(mode, cut, split, limit)
    root, source = Path(root).resolve(), Path(source).resolve()
    if root.exists():
        raise FileExistsError("Preserve previous probes: " + str(root))
    from . import jspace_quadratic_recipe as recipe
    from .jspace_full_lease import wait_ready
    from .jspace_qwen_parallel_model import load_native_block, load_readout_norm
    from .jspace_quadratic_evaluate import stream_score
    from .jspace_residual_carry import ResidualCarryBlock

    lease = wait_ready([0])
    plan = recipe.verify(source)
    torch.set_num_threads(2)
    torch.manual_seed(27861)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    cases = read(source / "tasks.json")["splits"][split]
    if limit is not None:
        cases = cases[:limit]
    producers = [Path(__file__), Path(__file__).with_name("jspace_residual_carry.py")]
    config = dict(mode=mode, cut=cut, selected=selected, split=split, limit=limit,
        dtype="float32", seed=27861, carry="native residual; dynamic complement RMS energy",
        source_fingerprint=plan["fingerprint"], source=str(source),
        source_sha256={str(p): sha(p) for p in producers},
        case_identities=[case["identity"] for case in cases],
        original_frozen_sources=plan["source_sha256"],
        scope="development-only mechanism intervention; not confirmation or generation-speed evidence")
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    root.mkdir(parents=True)
    write(root / "plan.json", dict(config=config, fingerprint=fingerprint,
        started=datetime.now(timezone.utc).isoformat(), lease_id=lease["id"]))
    used = {}

    def guard():
        wait_ready([0])

    def checked_weight(path, expected):
        observed = sha(path)
        if observed != expected:
            raise ValueError("Changed source artifact: " + str(path))
        used[str(path)] = observed

    fits = read(source / "basis/manifest.json")["artifacts"]

    def loader(layer):
        guard()
        if layer >= cut:
            return load_native_block(plan["model"]["path"], layer,
                device="cuda:0", dtype=torch.bfloat16).float()
        fit = source / "basis" / f"block_{layer:02d}" / "fit.pt"
        checked_weight(fit, fits[str(fit.relative_to(source))])
        if selected:
            path = source / "arms" / mode / f"layer_{layer:02d}" / "selected.pt"
            manifest = read(path.parent / "manifest.json")
            if manifest["fingerprint"] != plan["fingerprint"]:
                raise ValueError("Wrong selected checkpoint provenance")
            checked_weight(path, manifest["artifacts"][str(path.relative_to(source))])
        wrapped = recipe.build_block(source, mode, layer, "cuda:0",
                                     selected=selected, packed=True)
        return ResidualCarryBlock(wrapped).float().eval().requires_grad_(False)

    try:
        embedding, norm, _ = load_readout_norm(plan["model"]["path"], dtype=torch.bfloat16)
        embedding, norm = embedding.float(), norm.float()
        result = stream_score(cases, input_embedding=embedding, block_loader=loader,
            final_readout=(norm, embedding), label_ids=plan["label_ids"],
            scratch_dir=root / "scratch", rotary=recipe.rotary_module(plan, "cuda:0"),
            layer_count=36, device="cuda:0", guard=guard, microbatch=4,
            progress=lambda **kw: print(json.dumps(kw), flush=True))
        correct = sum(row["prediction"] == row["answer"] for row in result["records"])
        result.update(status="completed", fingerprint=fingerprint,
            correct=correct, count=len(cases), accuracy=correct / len(cases),
            scope=config["scope"], consumed_sha256=used,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            finished=datetime.now(timezone.utc).isoformat())
        guard()
        if result["peak_reserved_bytes"] > sum(lease["allocation"].values()) * 1024**2:
            raise RuntimeError("Probe exceeded its GPU reservation")
        write(root / "result.json", result)
        print(json.dumps({k: result[k] for k in ("status", "correct", "count", "accuracy", "peak_reserved_bytes")}))
        return result
    except BaseException as error:
        write(root / "failed.json", dict(status="failed", error=repr(error), fingerprint=fingerprint))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--mode", choices=("softmax", "quadratic"), default="softmax")
    parser.add_argument("--cut", type=int, default=36)
    parser.add_argument("--selected", action="store_true")
    parser.add_argument("--split", default="development")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(**vars(args))
