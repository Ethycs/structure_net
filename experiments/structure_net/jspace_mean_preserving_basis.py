"""Replace one weak Ritz coordinate with the native training mean direction.

This isolated candidate changes no task population, head projection, rank,
attention kernel or strong Jacobian direction. It does not fit activation PCA.
The original top rank-1 columns remain bitwise unchanged; the last column is
replaced only when the training mean is outside the original retained span.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil

import torch


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _guard(device, guard):
    device = torch.device(device)
    if device.type == "cuda" and guard is None:
        raise ValueError("CUDA mean-preserving fitting requires a reservation guard")
    if guard is not None:
        guard()
    return device


@torch.no_grad()
def replace_mean_direction(basis, mean, gram, *, resolved_functional_rank, guard=None):
    """Return a same-width basis and directly measured native-Gram diagnostics.

    ``gram`` is the original, already-validated PSD task-Jacobian Gramian. No
    normalization, row weighting, random draws or additional data are fitted.
    Numerical tolerances scale with arithmetic epsilon and native dimension.
    """
    _guard(basis.device, guard)
    if (basis.ndim != 2 or basis.dtype not in (torch.float32, torch.float64)
            or not 1 < basis.shape[1] <= basis.shape[0]
            or gram.shape != (basis.shape[0], basis.shape[0])
            or gram.dtype != basis.dtype or gram.device != basis.device
            or mean.shape not in ((basis.shape[0],), (1, basis.shape[0]))
            or mean.dtype != basis.dtype or mean.device != basis.device
            or not isinstance(resolved_functional_rank, int) or isinstance(resolved_functional_rank, bool)
            or not 0 <= resolved_functional_rank < basis.shape[1]):
        raise ValueError("Matching native Gram/mean, rank>=2, and at least one weak Ritz column required")
    if any(not torch.isfinite(x).all() for x in (basis, mean, gram)):
        raise ValueError("Basis, mean and Gramian must be finite")
    width, rank = basis.shape
    eps = torch.finfo(basis.dtype).eps
    orth_bound = 64*eps*math.sqrt(rank)
    original_orth = float((basis.T@basis-torch.eye(rank, device=basis.device, dtype=basis.dtype)).abs().max())
    if original_orth > orth_bound:
        raise ValueError("Original basis must be orthonormal")
    scale = float(gram.abs().max())
    if float((gram-gram.T).abs().max()) > 32*eps*scale or float(gram.trace()) < 0:
        raise ValueError("Original Gramian must be symmetric with nonnegative trace")
    vector = mean.reshape(-1)
    norm = float(vector.norm())
    mean_tolerance = 8*eps*math.sqrt(width)*norm
    original_error = vector-basis@(basis.T@vector)
    already = float(original_error.norm()) <= mean_tolerance
    result = basis.clone()
    if not already:
        fixed = basis[:, :-1]
        direction = vector.clone()
        for _ in range(2):
            direction -= fixed@(fixed.T@direction)
        residual_norm = float(direction.norm())
        if residual_norm <= mean_tolerance:
            raise ValueError("Mean is numerically in the fixed span but failed the original-span check")
        direction /= residual_norm
        # Canonical sign, independent of the scale/sign of the stored mean.
        direction *= direction[direction.abs().argmax()].sign()
        result[:, -1] = direction
    new_error = vector-result@(result.T@vector)
    new_orth = float((result.T@result-torch.eye(rank, device=basis.device, dtype=basis.dtype)).abs().max())
    if new_orth > orth_bound or float(new_error.norm()) > max(mean_tolerance, 4*eps*norm):
        raise ValueError("Mean-preserving replacement failed its numerical geometry checks")
    if not torch.equal(result[:, :-1], basis[:, :-1]):
        raise RuntimeError("Strong Ritz columns changed")
    trace = float(gram.trace())
    old_energy = float((basis*(gram@basis)).sum())
    old_last = float(basis[:, -1]@(gram@basis[:, -1]))
    new_last = float(result[:, -1]@(gram@result[:, -1]))
    new_energy = old_energy-old_last+new_last
    measured_new = float((result*(gram@result)).sum())
    energy_tolerance = 64*eps*max(trace, abs(old_energy), 1.)
    if abs(new_energy-measured_new) > energy_tolerance:
        raise ValueError("Single-column Gram-energy accounting failed")
    info = dict(rank=rank, native_width=width, changed=not already,
        unchanged_prefix_columns=rank-1, resolved_functional_rank=resolved_functional_rank,
        mean_already_in_original_span=already, mean_norm=norm,
        original_anchor_norm=float(original_error.norm()), new_anchor_norm=float(new_error.norm()),
        mean_relative_residual=float(new_error.norm())/norm if norm else 0.,
        mean_tolerance=mean_tolerance, orthogonality_bound=orth_bound,
        original_orthogonality_max=original_orth, new_orthogonality_max=new_orth,
        gram_trace=trace, original_captured_gram_energy=old_energy/trace if trace else None,
        new_captured_gram_energy=measured_new/trace if trace else None,
        captured_gram_energy_delta=(measured_new-old_energy)/trace if trace else None,
        old_last_gram_energy=old_last/trace if trace else None,
        new_last_gram_energy=new_last/trace if trace else None,
        construction="retain original top rank-1 Ritz columns; two-pass Gram-Schmidt of original native training global mean",
        scope="single-coordinate geometry repair; not new semantic CAVs or an accuracy guarantee")
    return result, info


def _fit_from_statistics(original, statistics, layer, device="cpu", guard=None):
    """Use only the pinned native mean/Gram, preserving every other fit tensor."""
    device = _guard(device, guard)
    if not torch.equal(original["means"][0], statistics["mean"][layer]):
        raise ValueError("Original fit mean differs from native training statistics")
    value, info = replace_mean_direction(original["basis"].to(device), original["means"].to(device),
        statistics["grams"][layer].to(device),
        resolved_functional_rank=original["metadata"]["retained_functional_directions"], guard=guard)
    info.update(layer=layer, head_projection="unchanged original rope_pairs and value_basis",
                original_basis_metadata=original["metadata"])
    return {**original, "basis": value.cpu(), "metadata": info}


@torch.no_grad()
def fit_all_layers(source_root, output_root, *, layers=None, device="cpu", guard=None, progress=None):
    """Create an exclusive candidate root from frozen native-training statistics."""
    device = _guard(device, guard)
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    parent = _read(source_root/"plan.json")
    if parent["fingerprint"] != _digest({k:v for k,v in parent.items() if k != "fingerprint"}):
        raise ValueError("Changed original recipe plan")
    original_manifest = _read(source_root/"basis/manifest.json")
    if original_manifest["status"] != "completed" or original_manifest["fingerprint"] != parent["fingerprint"]:
        raise ValueError("Original basis is not complete under its source plan")
    statistics_path = Path(parent["teacher_root"])/"capture/statistics.pt"
    statistics_sha = _sha(statistics_path)
    if parent["input_sha256"][str(statistics_path)] != statistics_sha:
        raise ValueError("Changed original native training statistics")
    stats = torch.load(statistics_path, map_location="cpu", mmap=True, weights_only=True)
    chosen = list(range(parent["config"]["layers"])) if layers is None else list(layers)
    if not chosen or len(set(chosen)) != len(chosen) or any(not isinstance(i, int) or isinstance(i, bool)
            or not 0 <= i < len(stats["grams"]) for i in chosen):
        raise ValueError("Distinct valid native layers required")
    source_inputs = {str(source_root/"plan.json"): _sha(source_root/"plan.json"),
                     str(source_root/"basis/manifest.json"): _sha(source_root/"basis/manifest.json"),
                     str(statistics_path): statistics_sha}
    for layer in chosen:
        path = source_root/"basis"/f"block_{layer:02d}"/"fit.pt"
        value = _sha(path)
        if original_manifest["artifacts"][str(path.relative_to(source_root))] != value:
            raise ValueError("Changed original basis: "+str(path))
        source_inputs[str(path)] = value
    producer = Path(__file__).resolve()
    plan = dict(schema="jspace_mean_preserving_basis_v1", source_root=str(source_root),
        source_fingerprint=parent["fingerprint"], layers=chosen,
        rank=1024, head_dim=64, replacement="weakest of1024 Ritz directions only",
        mean_source="original native128 TRAIN calibration cases,16 sampled positions each; no development/evaluation data",
        input_sha256=source_inputs, source_sha256={str(producer): _sha(producer)})
    plan["fingerprint"] = _digest(plan)
    output_root.mkdir(parents=True)
    (output_root/"producer").mkdir()
    shutil.copy2(producer, output_root/"producer"/producer.name)
    _write(output_root/"plan.json", plan)
    artifacts, reports = {}, {}
    for layer in chosen:
        _guard(device, guard)
        path = source_root/"basis"/f"block_{layer:02d}"/"fit.pt"
        old = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if old["basis"].shape != (2048, 1024) or old["head_dim"] != 64:
            raise ValueError("Candidate requires the fixed rank1024/head64 architecture")
        fit = _fit_from_statistics(old, stats, layer, device=device, guard=guard)
        info = fit["metadata"]
        info.update(original_fit_sha256=source_inputs[str(path)], source_fingerprint=parent["fingerprint"])
        directory = output_root/"basis"/f"block_{layer:02d}"
        directory.mkdir(parents=True)
        with (directory/"fit.pt").open("xb") as stream:
            torch.save(fit, stream)
        _write(directory/"fit.json", info)
        for saved in (directory/"fit.pt", directory/"fit.json"):
            artifacts[str(saved.relative_to(output_root))] = _sha(saved)
        reports[str(layer)] = {k:v for k,v in info.items() if k != "original_basis_metadata"}
        message = dict(phase="mean_preserving_basis", layer=layer,
                       anchor_norm=info["new_anchor_norm"], energy_delta=info["captured_gram_energy_delta"])
        if progress is None:
            print(json.dumps(message), flush=True)
        else:
            progress(message)
        del old, fit
    manifest = dict(status="completed", fingerprint=plan["fingerprint"], source_fingerprint=parent["fingerprint"],
                    artifacts=artifacts, layers=reports)
    _write(output_root/"basis/manifest.json", manifest)
    return manifest


def build_block(source_root, fit_root, arm, layer, device="cpu", *, parameter_dtype=torch.float32, guard=None):
    """Build a native-initialized carry candidate; never load a trained checkpoint."""
    device = _guard(device, guard)
    from .jspace_qwen_parallel_model import load_native_block
    from .jspace_qwen_affine_block import AffineQwenBlock
    from .jspace_quadratic_qwen import QuadraticQwenBlock
    from .jspace_residual_carry import ResidualCarryBlock
    source_root, fit_root = Path(source_root).resolve(), Path(fit_root).resolve()
    source, candidate = _read(source_root/"plan.json"), _read(fit_root/"plan.json")
    manifest = _read(fit_root/"basis/manifest.json")
    if (arm not in ("softmax", "quadratic") or layer not in candidate["layers"]
            or source["fingerprint"] != _digest({k:v for k,v in source.items() if k != "fingerprint"})
            or candidate["source_fingerprint"] != source["fingerprint"]
            or candidate["fingerprint"] != _digest({k:v for k,v in candidate.items() if k != "fingerprint"})
            or manifest["fingerprint"] != candidate["fingerprint"] or manifest["status"] != "completed"):
        raise ValueError("Candidate/source plan, arm or layer mismatch")
    path = fit_root/"basis"/f"block_{layer:02d}"/"fit.pt"
    if _sha(path) != manifest["artifacts"][str(path.relative_to(fit_root))]:
        raise ValueError("Changed mean-preserving fit")
    fit = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    native = load_native_block(source["model"]["path"], layer, device=device, dtype=torch.bfloat16)
    affine = AffineQwenBlock(native, fit["basis"].to(device), fit["means"].to(device),
        trainable=True, parameter_dtype=parameter_dtype,
        head_projection={k:fit[k].to(device) for k in ("rope_pairs", "value_basis")})
    wrapped = QuadraticQwenBlock(affine, mode=arm, learn_metric=True)
    return ResidualCarryBlock(wrapped).to(device)
