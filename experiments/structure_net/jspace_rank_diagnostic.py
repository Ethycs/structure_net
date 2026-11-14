"""Append-only streamed spectrum diagnosis; never changes a compression gate.

Reproduces the frozen randomized range finder on every existing BF16 VJP row.
No model is loaded, and no rows are discarded, normalized, or reweighted.
"""
from collections import defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from . import jspace_benchmark_groebner as frozen
from .jspace_bench_data import sha, write
from .jspace_qwen_progressive_calibration import StreamedBFloat16JacobianRows

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = dict(svd_rank=64, oversample=16, power_iterations=1, chunk_rows=256,
                anchors=24, old_rows=8192)


def _event(progress, phase, **values):
    value = {"phase": phase, **values}
    if progress is None: print(json.dumps(value, allow_nan=False), flush=True)
    else: progress(value)


def rank_cutoffs(singular, original_shape, sketch_shape, old_rows=8192):
    """Report the current gate and explicit counterfactual numerical thresholds."""
    m, n = original_shape
    factors = {"frozen_original_shape": max(m, n), "old_8192_row_reference": max(old_rows, n),
               "actual_svd_shape": max(sketch_shape)}
    return {name: {"relative_cutoff": size * torch.finfo(singular.dtype).eps,
                   "absolute_cutoff": float(singular[0] * size * torch.finfo(singular.dtype).eps),
                   "rank": int((singular > singular[0] * size * torch.finfo(singular.dtype).eps).sum())}
            for name, size in factors.items()}


def _concentration(energy, raw_energy, case_ranges):
    total = float(energy.sum()); families = defaultdict(lambda: {"energy": 0., "raw_energy": 0., "rows": 0, "zero_rows": 0})
    rows = []; cursor = 0
    for case in case_ranges:
        start, stop = case["start"], case["stop"]
        if start != cursor or not start < stop <= len(energy): raise ValueError("Case ranges must partition every Jacobian row")
        cursor = stop; selected = energy[start:stop]; raw = raw_energy[start:stop]
        row = {**case, "energy": float(selected.sum()), "raw_energy": float(raw.sum()),
               "rows": stop - start, "zero_rows": int((raw == 0).sum()), "maximum_row_energy": float(selected.max())}
        row["energy_fraction"] = row["energy"] / total; rows.append(row)
        family = families[case["family"]]
        for key in ("energy", "raw_energy", "rows", "zero_rows"): family[key] += row[key]
    if cursor != len(energy): raise ValueError("Case ranges do not cover all Jacobian rows")
    for family in families.values(): family["energy_fraction"] = family["energy"] / total
    return {"cases": rows, "families": dict(families), "zero_rows": int((raw_energy == 0).sum()),
            "top_case_energy_fraction": max(row["energy_fraction"] for row in rows),
            "top_row_energy_fraction": float(energy.max()) / total,
            "top10_row_energy_fraction": float(energy.topk(min(10, len(energy))).values.sum()) / total}


@torch.no_grad()
def diagnose(jacobian_rows, scale, *, seed=20891, case_ranges=None, guard=None, progress=None,
             svd_rank=64, oversample=16, power_iterations=1, chunk_rows=256, anchors=24, old_rows=8192):
    """Return (JSON report, CPU tensors) using the exact frozen sketch arithmetic."""
    device = torch.device(jacobian_rows.device)
    if device.type == "cuda" and guard is None: raise ValueError("CUDA diagnostics require the caller's reservation guard")
    if guard is not None: guard()
    m, n = jacobian_rows.shape
    if scale.shape != (n,) or not 0 < anchors <= min(svd_rank, m, n) or chunk_rows < 1 or oversample < 0 or power_iterations < 0:
        raise ValueError("Invalid diagnostic dimensions/settings")
    dtype = torch.float64 if jacobian_rows.dtype == torch.float64 else torch.float32
    rank = min(svd_rank, m, n); width = min(rank + oversample, m, n)
    s = scale.to(device=device, dtype=dtype)
    if not torch.isfinite(s).all() or (s <= 0).any(): raise ValueError("Finite positive training-only scale required")
    generator = torch.Generator(device=device).manual_seed(seed)
    omega = torch.randn(n, width, device=device, dtype=dtype, generator=generator)
    energy = torch.zeros((), device=device, dtype=torch.float64); pass_number = 0

    def rows():
        nonlocal pass_number
        pass_number += 1
        if guard is not None: guard()
        _event(progress, "jacobian_pass_start", pass_number=pass_number, rows=m)
        for start in range(0, m, chunk_rows):
            raw = jacobian_rows[start:start + chunk_rows].to(dtype); block = raw * s
            if not torch.isfinite(block).all(): raise ValueError("Jacobian contains nonfinite values")
            yield start, raw, block
        _event(progress, "jacobian_pass_complete", pass_number=pass_number)

    def multiply_right(right):
        out = torch.empty(m, right.shape[1], device=device, dtype=dtype)
        for start, _, block in rows(): out[start:start + len(block)] = block @ right
        return out

    q = torch.linalg.qr(multiply_right(omega), mode="reduced").Q
    for _ in range(power_iterations):
        z = torch.zeros(n, q.shape[1], device=device, dtype=dtype)
        for start, _, block in rows(): z.addmm_(block.T, q[start:start + len(block)])
        z = torch.linalg.qr(z, mode="reduced").Q
        q = torch.linalg.qr(multiply_right(z), mode="reduced").Q
    small = torch.zeros(q.shape[1], n, device=device, dtype=dtype)
    row_energy, raw_row_energy = torch.empty(m, dtype=torch.float64), torch.empty(m, dtype=torch.float64)
    for start, raw, block in rows():
        small.addmm_(q[start:start + len(block)].T, block)
        energy += block.double().square().sum()
        row_energy[start:start + len(block)] = block.double().square().sum(1).cpu()
        raw_row_energy[start:start + len(block)] = raw.double().square().sum(1).cpu()
    if not torch.isfinite(energy) or energy <= 0: raise ValueError("Jacobian energy must be finite and positive")
    del q, omega
    _event(progress, "small_svd", shape=list(small.shape))
    _, singular, vh = torch.linalg.svd(small, full_matrices=False)
    cutoffs = rank_cutoffs(singular, (m, n), small.shape, old_rows)
    # Diagnostic pivots do not authorize passing the original failed gate.
    remaining = (singular[:rank, None] * vh[:rank]).clone(); selected = []; pivot_norms = []
    for _ in range(anchors):
        norms = remaining.square().sum(0)
        if selected: norms[selected] = -1
        index = int(norms.argmax()); value = norms[index]
        if not torch.isfinite(value) or value <= 0: break
        direction = remaining[:, index] / value.sqrt()
        remaining -= direction[:, None] * (direction @ remaining)[None, :]
        selected.append(index); pivot_norms.append(float(value.sqrt()))
    ranges = case_ranges or [{"identity": "all_rows", "family": "all", "start": 0, "stop": m}]
    report = {"original_shape": [m, n], "small_shape": list(small.shape), "seed": seed,
        "passes": pass_number, "rank_cutoffs": cutoffs, "old_reference_rows": old_rows,
        "requested_anchors": anchors, "frozen_gate_passes": cutoffs["frozen_original_shape"]["rank"] >= anchors,
        "diagnostic_anchor_indices": selected, "diagnostic_pivot_norms": pivot_norms,
        "singular_values": singular.cpu().tolist(),
        "s24_over_s1": float(singular[23] / singular[0]) if len(singular) >= 24 else None,
        "s64_over_s1": float(singular[63] / singular[0]) if len(singular) >= 64 else None,
        "captured_energy_fraction": float(singular[:rank].double().square().sum() / energy),
        "range_projection_energy_fraction": float(singular.double().square().sum() / energy),
        "total_standardized_jacobian_energy": float(energy),
        "row_concentration": _concentration(row_energy, raw_row_energy, ranges),
        "scope": "read-only diagnosis; identical full-row weighting and frozen sketch; no gate or model changes"}
    tensors = {"small": small.cpu(), "singular": singular.cpu(), "vh": vh.cpu(), "scale": s.cpu(),
               "standardized_row_energy": row_energy, "raw_row_energy": raw_row_energy}
    return report, tensors


def _sources():
    return [Path(__file__), ROOT / "tests/structure_net/test_jspace_rank_diagnostic.py",
            Path(frozen.__file__), ROOT / "experiments/structure_net/jspace_qwen_progressive_calibration.py",
            ROOT / "experiments/structure_net/jspace_logit_sketch.py",
            ROOT / "experiments/structure_net/jspace_bench_data.py"]


def _digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prepare(root, block_dir, *, seed=20891):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise ValueError("Prepare with CUDA hidden")
    root, block_dir = Path(root).resolve(), Path(block_dir).resolve()
    if root.exists(): raise FileExistsError(root)
    interface_path = block_dir / "interface.json"; interface = json.loads(interface_path.read_text())
    capture_dir = Path(interface["capture_block"]["boundary"]).parent.parent
    metadata_path, items_path = capture_dir / "metadata.json", capture_dir / "items.json"
    meta, items = json.loads(metadata_path.read_text()), json.loads(items_path.read_text())["items"]
    if meta["status"] != "completed": raise ValueError("Completed fixed-teacher capture required")
    ranges = []
    for index in meta["calibration_indices"]:
        case = items[index]; count = (case["stop"] - case["offset"]) * meta["probe_groups"] * meta["probes_per_group"]
        ranges.append({"identity": case["identity"], "family": case["family"], "start": case["jacobian_row_offset"],
                       "stop": case["jacobian_row_offset"] + count, "case_index": index})
    inputs = dict(interface["input_sha256"])
    for path in (interface_path, metadata_path, items_path):
        actual = sha(path)
        if str(path) in inputs and inputs[str(path)] != actual: raise ValueError("Changed capture metadata")
        inputs[str(path)] = actual
    plan = {"schema": "jspace.rank-diagnostic.v1", "settings": {**SETTINGS, "seed": seed},
        "layer": interface["layer"], "teacher_fingerprint": interface["teacher_fingerprint"],
        "interface": str(interface_path), "jacobian": interface["jacobian"],
        "native_calibration": interface["native_calibration"], "case_ranges": ranges,
        "source_sha256": {str(path): sha(path) for path in _sources()}, "input_sha256": inputs,
        "scope": "read-only layer spectrum diagnostic; no held-out or model execution"}
    plan["fingerprint"] = _digest(plan); root.mkdir(parents=True)
    for path in _sources():
        target = root / "producer" / path.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    write(root / "plan.json", plan)
    return {"status": "prepared", "fingerprint": plan["fingerprint"], "layer": plan["layer"]}


def run(root, *, device="cuda:0", guard=None):
    root = Path(root).resolve(); device = torch.device(device)
    if device.type == "cuda" and guard is None: raise ValueError("CUDA diagnostics require the caller's reservation guard")
    if guard is not None: guard()
    if (root / "result.json").exists() or (root / "spectral_state.pt").exists(): raise FileExistsError("Diagnostic already produced output")
    plan = json.loads((root / "plan.json").read_text())
    if plan["fingerprint"] != _digest({k: v for k, v in plan.items() if k != "fingerprint"}): raise ValueError("Changed diagnostic plan")
    for path, expected in plan["source_sha256"].items():
        if sha(path) != expected or sha(root / "producer" / Path(path).relative_to(ROOT)) != expected: raise ValueError("Changed diagnostic source")
    for path, expected in plan["input_sha256"].items():
        if sha(path) != expected: raise ValueError("Changed diagnostic input: " + path)
    if device.type == "cuda":
        torch.cuda.init()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(4); started = time.perf_counter()
    raw = np.load(plan["native_calibration"], mmap_mode="r", allow_pickle=False)
    if raw.dtype != np.int16: raise ValueError("Native calibration must preserve BF16 bits")
    training = torch.from_numpy(np.array(raw, copy=True)).view(torch.bfloat16).float().to(device); raw._mmap.close()
    mean, scale = frozen._normalization(training, plan["settings"]["chunk_rows"]); del training
    with StreamedBFloat16JacobianRows(plan["jacobian"], device=device, max_rows=plan["settings"]["chunk_rows"]) as rows:
        report, tensors = diagnose(rows, scale, case_ranges=plan["case_ranges"], guard=guard, **plan["settings"])
        report["streamed_jacobian"] = rows.metadata()
    tensors["mean"] = mean.cpu()
    with (root / "spectral_state.pt").open("xb") as stream: torch.save(tensors, stream)
    report.update(status="completed", layer=plan["layer"], fingerprint=plan["fingerprint"],
        elapsed_seconds=time.perf_counter() - started, spectral_state_sha256=sha(root / "spectral_state.pt"),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0)
    if guard is not None: guard()
    write(root / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--root", required=True); parser.add_argument("--block-dir"); parser.add_argument("--seed", type=int, default=20891)
    args = parser.parse_args()
    if args.stage == "prepare":
        if not args.block_dir: parser.error("prepare requires --block-dir")
        print(json.dumps(prepare(args.root, args.block_dir, seed=args.seed)), flush=True)
    else:
        from .jspace_full_lease import wait_ready
        result = run(args.root, guard=lambda: wait_ready([0]))
        print(json.dumps({k: result[k] for k in ("status", "layer", "rank_cutoffs", "s24_over_s1", "elapsed_seconds")}), flush=True)


if __name__ == "__main__": main()
