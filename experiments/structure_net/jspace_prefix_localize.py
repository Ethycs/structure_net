"""Development-only compressed-prefix/original-suffix localization.

The streaming evaluator's input table contains reconstructed hidden states.
Its temporary integer IDs address rows of that table; they are not vocabulary
tokens. Every original prompt length, label and identity is preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from . import jspace_quadratic_recipe as recipe
from .jspace_quadratic_evaluate import stream_score

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "data/experiments/jspace_prefix_localization/20261114_v1"
DEFAULT_CUTS = [1, 2, 6, 12, 18, 24, 30, 35]
SCHEMA = "jspace.prefix-localization.v1"
read, write, sha, digest = recipe.read, recipe.write, recipe.sha, recipe.digest


def validate_cuts(cuts):
    cuts = list(cuts)
    if not cuts or len(set(cuts)) != len(cuts) or any(isinstance(c, bool) or not isinstance(c, int) or not 1 <= c < 36 for c in cuts):
        raise ValueError("Distinct prefix cuts must be integers1..35")
    return cuts


def development_cases(parent):
    tasks, items = read(parent / "tasks.json"), read(parent / "items.json")["items"]
    cases = tasks["splits"]["development"]
    if len(cases) != 64 or Counter(c["family"] for c in cases) != {f: 16 for f in ("arc", "boolq", "hellaswag", "winogrande")}:
        raise ValueError("The original64 balanced development cases are required")
    validate_development_rows(cases, items)
    return cases, items


def validate_development_rows(cases, items):
    dev = {case["identity"]: case for case in items if case["split"] == "development"}
    if (not cases or len(dev) != sum(case["split"] == "development" for case in items) or
            len({case["identity"] for case in cases}) != len(cases) or {case["identity"] for case in cases} != set(dev)):
        raise ValueError("Localization requires exactly the cached development identities")
    for case in cases:
        source = dev[case["identity"]]
        if (case.get("split", "development") != "development" or
                any(case[key] != source[key] for key in ("input_ids", "family", "answer", "choices"))):
            raise ValueError("Development content or split changed")
    return dev


@torch.inference_mode()
def reconstruct_development(cases, items, coordinates_path, basis, anchor, *, device="cpu", guard=None, chunk_tokens=256):
    """Read development rows only and retain one CPU native-width input table."""
    dev = validate_development_rows(cases, items)
    device = torch.device(device)
    if device.type == "cuda" and guard is None:
        raise ValueError("GPU reconstruction requires a reservation guard")
    if chunk_tokens < 1:
        raise ValueError("Positive reconstruction chunk required")
    guard = guard or (lambda: None)
    basis, anchor = basis.detach().to(device=device, dtype=torch.float32), anchor.detach().to(device=device, dtype=torch.float32)
    if basis.ndim != 2 or anchor.shape not in ((basis.shape[0],), (1, basis.shape[0])):
        raise ValueError("Native-width basis and one global affine anchor required")
    bank = np.load(coordinates_path, mmap_mode="r")
    try:
        if bank.ndim != 2 or bank.shape[1] != basis.shape[1] or bank.dtype != np.float32:
            raise ValueError("Expected the existing FP32 coordinate bank")
        total_tokens = sum(len(case["input_ids"]) for case in cases)
        hidden = torch.empty(total_tokens, basis.shape[0], dtype=torch.float32, device="cpu")
        virtual, provenance, offset = [], [], 0
        for case in cases:
            source = dev[case["identity"]]
            start, stop = source["offset"], source["stop"]
            length = len(case["input_ids"])
            if not 0 <= start < stop <= len(bank) or stop - start != length:
                raise ValueError("Development offsets do not match the frozen prefix bank")
            for local in range(0, length, chunk_tokens):
                guard(); end = min(local + chunk_tokens, length)
                z = torch.from_numpy(np.array(bank[start+local:start+end], copy=True)).to(device)
                values = z @ basis.T + anchor
                if not torch.isfinite(values).all():
                    raise FloatingPointError("Nonfinite reconstructed development hidden states")
                hidden[offset+local:offset+end] = values.cpu()
            copied = deepcopy(case)
            copied["input_ids"] = list(range(offset, offset + length))
            virtual.append(copied)
            provenance.append(dict(identity=case["identity"], family=case["family"], original_length=length,
                original_input_ids_digest=digest(case["input_ids"]), prefix_bank_offset=start,
                virtual_offset=offset, virtual_stop=offset+length))
            offset += length
        return hidden, virtual, dict(cases=provenance, total_tokens=total_tokens,
            cpu_hidden_bytes=hidden.numel()*hidden.element_size(),
            logical_prefix_read_bytes=total_tokens*bank.shape[1]*4,
            reconstruction="h_cut = z_cut @ U_cut.T + a_cut; z_cut already includes the prior fixed coordinate bridge",
            virtual_id_scope="Internal hidden-table row addresses, not model vocabulary IDs or retokenized prompts")
    finally:
        bank._mmap.close()


def source_paths():
    return [Path(__file__), ROOT / "tests/structure_net/test_jspace_prefix_localize.py"]


def prepare(root, parent, arm="softmax", cuts=DEFAULT_CUTS):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Prepare with hidden CUDA")
    root, parent = Path(root).resolve(), Path(parent).resolve()
    if root.exists():
        raise FileExistsError(root)
    cuts = validate_cuts(cuts)
    if arm not in recipe.ARMS:
        raise ValueError("Unknown compressed arm")
    original = recipe.verify(parent)
    cases, _ = development_cases(parent)
    inputs = {str(parent / name): sha(parent / name) for name in ("plan.json", "tasks.json", "items.json", "basis/manifest.json")}
    expected = {}
    basis_manifest = recipe.checked(parent, parent / "basis", hashes=False)
    for cut in cuts:
        directory = recipe.train_path(parent, arm, cut-1)
        manifest = recipe.checked(parent, directory, hashes=False)
        inputs[str(directory / "manifest.json")] = sha(directory / "manifest.json")
        for path, source in ((recipe.prefix_path(parent, arm, cut), manifest), (recipe.fit_path(parent, cut), basis_manifest)):
            expected[str(path)] = source["artifacts"][str(path.relative_to(parent))]
    plan = dict(schema=SCHEMA, parent_root=str(parent), parent_fingerprint=original["fingerprint"],
        arm=arm, cuts=cuts, split="development", case_identities=[case["identity"] for case in cases],
        input_sha256=inputs, consumed_artifact_sha256=expected,
        source_sha256={str(path): sha(path) for path in source_paths()},
        created_at=datetime.now(timezone.utc).isoformat(),
        scope="Diagnostic compressed-prefix/original-suffix accuracy; not the deployed compressed model and not fresh confirmation")
    plan["fingerprint"] = digest(plan)
    root.mkdir(parents=True)
    for path in source_paths():
        target = root / "producer" / path.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write(root / "plan.json", plan)
    return dict(status="prepared", fingerprint=plan["fingerprint"], cuts=cuts, count=len(cases))


def verify(root):
    root = Path(root); plan = read(root / "plan.json")
    if plan["schema"] != SCHEMA or plan["fingerprint"] != digest({k: v for k, v in plan.items() if k != "fingerprint"}):
        raise ValueError("Changed localization plan")
    parent = Path(plan["parent_root"])
    if recipe.verify(parent)["fingerprint"] != plan["parent_fingerprint"]:
        raise ValueError("Changed parent recipe")
    for path, expected in plan["input_sha256"].items():
        if sha(path) != expected:
            raise ValueError("Changed localization input manifest")
    for path, expected in plan["source_sha256"].items():
        if sha(path) != expected or sha(root / "producer" / Path(path).relative_to(ROOT)) != expected:
            raise ValueError("Changed localization source")
    cases, _ = development_cases(parent)
    if [case["identity"] for case in cases] != plan["case_identities"]:
        raise ValueError("Development population changed")
    return plan


def run(root, cut):
    from .jspace_full_lease import wait_ready
    from .jspace_qwen_parallel_model import load_native_block, load_readout_norm
    root = Path(root); plan = verify(root)
    validate_cuts([cut])
    if cut not in plan["cuts"]:
        raise ValueError("Cut was not frozen in this plan")
    lease = wait_ready([0]); guard = lambda: wait_ready([0])
    torch.cuda.init(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    parent = Path(plan["parent_root"]); original = read(parent / "plan.json")
    bank_path, fit_path = recipe.prefix_path(parent, plan["arm"], cut), recipe.fit_path(parent, cut)
    for path in (bank_path, fit_path):
        if sha(path) != plan["consumed_artifact_sha256"][str(path)]:
            raise ValueError("Changed selected prefix bank or coordinate basis")
    cases, items = development_cases(parent)
    fit = recipe.load(fit_path); basis, mean = fit["basis"], fit["means"]
    anchor = mean - (mean @ basis) @ basis.T
    hidden, virtual, reconstruction = reconstruct_development(cases, items, bank_path, basis, anchor,
        device="cuda:0", guard=guard)
    del fit, basis, mean, anchor; gc.collect(); torch.cuda.empty_cache()
    directory = root / f"cut_{cut:02d}"; directory.mkdir(exist_ok=False)
    embedding, norm, _ = load_readout_norm(original["model"]["path"], device="cpu", dtype=torch.bfloat16)
    value = stream_score(virtual, input_embedding=hidden,
        block_loader=lambda local: load_native_block(original["model"]["path"], cut+local,
            device="cuda:0", dtype=torch.bfloat16).float().eval().requires_grad_(False),
        final_readout=(norm.float(), embedding.float()), label_ids=original["label_ids"],
        scratch_dir=directory / "ephemeral_hidden_banks", rotary=recipe.rotary_module(original, "cuda:0"),
        layer_count=36-cut, device="cuda:0", guard=guard, microbatch=4)
    rows = value["records"]
    value.update(schema=SCHEMA, fingerprint=plan["fingerprint"], cut=cut, arm=plan["arm"], split="development",
        reconstruction=reconstruction, original_suffix_layers=list(range(cut, 36)),
        accuracy=dict(correct=sum(r["prediction"] == r["answer"] for r in rows), count=len(rows),
            by_family={f:dict(correct=sum(r["prediction"] == r["answer"] for r in rows if r["family"] == f),
                count=sum(r["family"] == f for r in rows)) for f in sorted({r["family"] for r in rows})}),
        scope=plan["scope"], parent_fingerprint=plan["parent_fingerprint"],
        consumed_artifact_sha256={str(path):plan["consumed_artifact_sha256"][str(path)] for path in (bank_path, fit_path)})
    guard(); peak = torch.cuda.max_memory_reserved()
    if peak > sum(lease["allocation"].values())*1024**2:
        raise RuntimeError("Localization exceeded its reservation")
    write(directory / "result.json", value)
    write(directory / "manifest.json", dict(status="completed", fingerprint=plan["fingerprint"],
        lease_id=lease["id"], peak_reserved_bytes=peak, artifacts={"result.json":sha(directory / "result.json")}))
    return dict(status="completed", cut=cut, accuracy=value["accuracy"], result_sha256=sha(directory / "result.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--parent", type=Path, default=recipe.DEFAULT_ROOT)
    parser.add_argument("--arm", choices=recipe.ARMS, default="softmax")
    parser.add_argument("--cuts", type=int, nargs="+", default=DEFAULT_CUTS)
    parser.add_argument("--cut", type=int)
    args = parser.parse_args()
    result = prepare(args.root, args.parent, args.arm, args.cuts) if args.stage == "prepare" else run(args.root, args.cut)
    print(json.dumps(result, allow_nan=False))
