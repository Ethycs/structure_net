"""Explicit task-margin-Jacobian block Krylov bases for Qwen compression.

Consumes only the original native capture's statistics.pt. Its Gramian comes
from the correct-choice logit minus mean-other-choice logit, differentiated at
16 sampled input positions per calibration example. It is NOT a full-vocabulary
Jacobian, activation PCA, or a set of identified semantic CAVs. V projections
use the separately recorded uncentered activation second moment.
"""
import json
from pathlib import Path

import torch


def _guard(device, guard):
    device = torch.device(device)
    if device.type == "cuda" and guard is None: raise ValueError("CUDA basis fitting requires the caller's reservation guard")
    if guard is not None: guard()
    if device.type == "cuda" and device.index is None: device = torch.device("cuda", torch.cuda.current_device())
    return device


def _positive_integer(value): return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _symmetric_psd_operator(gram):
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not len(gram) or gram.dtype not in (torch.float32, torch.float64):
        raise ValueError("Gramian must be a nonempty square FP32/FP64 matrix")
    if not torch.isfinite(gram).all(): raise ValueError("Nonfinite Gramian")
    eps = torch.finfo(gram.dtype).eps
    magnitude = gram.abs().max()
    if (gram - gram.T).abs().max() > magnitude * (32 * eps): raise ValueError("Gramian is not symmetric")
    gram = (gram + gram.T) * .5; trace = gram.trace()
    if not torch.isfinite(trace) or trace < 0: raise ValueError("Gramian trace must be finite and nonnegative")
    if trace == 0:
        if torch.count_nonzero(gram): raise ValueError("Nonzero zero-trace matrix is not PSD")
        return gram, 0., 0.
    operator = gram / trace
    # PSD validation does not change the operator; the jitter is used only by
    # Cholesky to allow accumulated floating-point roundoff in a singular Gramian.
    tolerance = 64 * eps * float(operator.diagonal().abs().max())
    _, info = torch.linalg.cholesky_ex(operator + torch.eye(len(gram), device=gram.device, dtype=gram.dtype) * tolerance)
    if int(info) != 0: raise ValueError("Gramian is not positive semidefinite within arithmetic tolerance")
    return operator, float(trace), tolerance


@torch.no_grad()
def block_krylov_basis(gram, rank=1024, *, block_width=1024, steps=1, seed=20861, guard=None):
    """Rayleigh–Ritz in explicit span[B, C B, ..., C**steps B], C=G/tr(G).

    Seed width >= rank guarantees the requested storage width even when G is
    rank deficient. Such retained numerical-null directions are reported as
    seed completion, never as recovered functional information. Block Arnoldi
    uses two projection passes and an SVD reveal scaled to the unprojected block.
    """
    _guard(gram.device, guard)
    if (not _positive_integer(rank) or not _positive_integer(block_width) or
            not isinstance(steps, int) or isinstance(steps, bool) or steps < 0 or
            gram.ndim != 2 or not rank <= block_width <= gram.shape[0] or
            not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError("Require 0 < rank <= block_width <= dimension, nonnegative steps, and integer seed")
    operator, trace, psd_tolerance = _symmetric_psd_operator(gram)
    width = len(gram); eps = torch.finfo(gram.dtype).eps
    generator = torch.Generator(device=gram.device).manual_seed(seed)
    seed_block = torch.randn(width, block_width, device=gram.device, dtype=gram.dtype, generator=generator)
    q = torch.linalg.qr(seed_block, mode="reduced").Q; frontier = q
    history = []
    for step in range(1, steps + 1):
        if guard is not None: guard()
        if q.shape[1] == width: break
        raw = operator @ frontier; residual = raw.clone()
        for _ in range(2): residual -= q @ (q.T @ residual)
        left, singular, _ = torch.linalg.svd(residual, full_matrices=False)
        threshold = max(raw.shape) * eps * float(raw.norm())
        retained = min(int((singular > threshold).sum()), width - q.shape[1])
        history.append({"step": step, "input_block_width": frontier.shape[1], "new_directions": retained,
            "rank_threshold": threshold, "unprojected_frobenius_norm": float(raw.norm()),
            "largest_residual_singular": float(singular[0]), "smallest_retained_singular": float(singular[retained - 1]) if retained else None})
        if not retained: break
        frontier = left[:, :retained]
        for _ in range(2): frontier -= q @ (q.T @ frontier)
        frontier = torch.linalg.qr(frontier, mode="reduced").Q
        q = torch.cat((q, frontier), dim=1)
    # Reorthogonalize the accumulated space before solving its projected operator.
    q = torch.linalg.qr(q, mode="reduced").Q
    projected = q.T @ operator @ q; projected = (projected + projected.T) * .5
    eigenvalues, eigenvectors = torch.linalg.eigh(projected)
    eigenvalues = eigenvalues.flip(0); eigenvectors = eigenvectors.flip(1)
    if float(eigenvalues[-1]) < -max(psd_tolerance, 32 * eps): raise ValueError("Projected operator has materially negative Ritz values")
    basis = (q @ eigenvectors[:, :rank]).contiguous()
    maxima = basis.abs().argmax(0); signs = basis[maxima, torch.arange(rank, device=gram.device)].sign()
    basis *= signs
    orthogonality = float((basis.T @ basis - torch.eye(rank, device=gram.device, dtype=gram.dtype)).abs().max())
    if not torch.isfinite(basis).all() or orthogonality > 64 * eps * max(1, rank**.5):
        raise ValueError("Krylov/Ritz basis lost orthonormality")
    functional_threshold = max(projected.shape) * eps * max(0., float(eigenvalues[0]))
    functional = int((eigenvalues > functional_threshold).sum()) if trace else 0
    retained_functional = min(rank, functional)
    residual = operator @ basis - basis * eigenvalues[:rank]
    metadata = {"operator": "C = G / trace(G); zero G stays zero", "gram_trace": trace,
        "seed": seed, "rank": rank, "block_width": block_width, "requested_power_steps": steps,
        "krylov_dimension": q.shape[1], "expansions": history,
        "construction": "explicit block Arnoldi/Krylov span with two-pass orthogonalization, residual SVD rank reveal, Rayleigh-Ritz",
        "rank_reveal_rule": "max(block.shape)*eps*unprojected_block_Frobenius_norm; discard only below numerical reveal threshold",
        "ritz_values": eigenvalues.cpu().tolist(), "functional_rank_threshold": functional_threshold,
        "resolved_functional_directions_in_krylov_space": functional,
        "retained_functional_directions": retained_functional, "retained_numerical_null_completion": rank - retained_functional,
        "captured_normalized_gram_energy": float(eigenvalues[:rank].sum()) if trace else None,
        "krylov_space_normalized_gram_energy": float(eigenvalues.sum()) if trace else None,
        "ritz_residual_frobenius": float(residual.norm()), "maximum_ritz_residual": float(residual.norm(dim=0).max()),
        "orthogonality_max": orthogonality, "psd_validation_tolerance": psd_tolerance,
        "scope": "functional rank is resolved inside the constructed Krylov space; seed completion is not additional Jacobian information"}
    return basis, metadata


@torch.no_grad()
def fit_layer(statistics, layer, *, rank=1024, block_width=1024, steps=1, head_dim=64,
              seed=20861, device="cpu", guard=None):
    """Compile one native-layer fit from the original frozen statistics format."""
    device = _guard(device, guard)
    count = statistics["count"]
    if not _positive_integer(count): raise ValueError("Positive sampled-position count required")
    if not isinstance(layer, int) or isinstance(layer, bool) or not 0 <= layer < len(statistics["grams"]):
        raise ValueError("Layer index outside captured statistics")
    gram = statistics["grams"][layer].to(device)
    basis, metadata = block_krylov_basis(gram, rank, block_width=block_width, steps=steps, seed=seed + layer, guard=guard)
    mean = statistics["mean"][layer].to(device=device, dtype=gram.dtype)
    scores = statistics["pair_energy"][layer].to(device=device, dtype=gram.dtype)
    covariance = statistics["value_cov"][layer].to(device=device, dtype=gram.dtype) / count
    value_mean = statistics["value_sum"][layer].to(device=device, dtype=gram.dtype) / count
    if (mean.shape != (len(gram),) or scores.ndim != 2 or not _positive_integer(head_dim) or head_dim % 2 or
            head_dim > scores.shape[1] * 2 or covariance.shape != (scores.shape[0], scores.shape[1] * 2, scores.shape[1] * 2) or
            value_mean.shape != covariance.shape[:2] or any(not torch.isfinite(x).all() for x in (mean, scores, covariance, value_mean)) or (scores < 0).any()):
        raise ValueError("Invalid head statistics, mean or retained head dimension")
    pairs = scores.topk(head_dim // 2, dim=-1).indices.sort(-1).values
    value_vectors = []; value_info = []
    for group in range(len(scores)):
        moment, total, _ = _symmetric_psd_operator(covariance[group])
        eigenvalues, vectors = torch.linalg.eigh(moment)
        selected = vectors[:, -head_dim:].flip(-1).contiguous(); value_vectors.append(selected)
        projected_mean = selected @ (selected.T @ value_mean[group])
        value_info.append({"group": group,
            "retained_pair_energy_fraction": float(scores[group, pairs[group]].sum() / scores[group].sum()) if scores[group].sum() > 0 else None,
            "retained_uncentered_value_energy_fraction": float(eigenvalues[-head_dim:].sum()) if total else None,
            "value_mean_projection_squared_error": float((projected_mean - value_mean[group]).square().sum())})
    metadata.update(layer=layer, sampled_calibration_positions=count,
        calibration_examples=count // 16 if count % 16 == 0 else None, positions_per_calibration_example=16,
        jacobian_provenance="original native capture: correct-choice final logit minus mean other-choice final logits; Jacobians at16 sampled input positions; supervised task-margin objective, NOT full-vocabulary",
        mean_provenance="single global mean over sampled native calibration states; no development fitting",
        head_pair_provenance="top pair_energy from sampled squared native Q/K activation-times-task-margin-gradient contributions",
        value_provenance="uncentered value_cov/count; no subtraction of value_mean outer product",
        head_projection=value_info, semantic_scope="functional axes, not experimentally identified semantic CAVs")
    return {"basis": basis.cpu(), "means": mean[None].cpu(), "rope_pairs": pairs.cpu(),
            "value_basis": torch.stack(value_vectors).cpu(), "rank": rank, "head_dim": head_dim, "metadata": metadata}


def fit_all_layers(statistics_path, output_dir, *, layers=None, rank=1024, block_width=1024, steps=1,
                   head_dim=64, seed=20861, device="cpu", guard=None, progress=None):
    """One bounded-memory pass over statistics; caller owns hashes/reservations.

    Does not open boundary/activation caches or rehash the captured dataset.
    Output block directories are exclusive and cannot overwrite a prior attempt.
    """
    device = _guard(device, guard); output_dir = Path(output_dir)
    statistics = torch.load(statistics_path, map_location="cpu", weights_only=True, mmap=True)
    layers = list(range(len(statistics["grams"]))) if layers is None else list(layers)
    if not layers or len(set(layers)) != len(layers): raise ValueError("Unique nonempty layer selection required")
    if any((output_dir / f"block_{layer:02d}").exists() for layer in layers): raise FileExistsError("Basis output already exists")
    result = {}
    for layer in layers:
        fit = fit_layer(statistics, layer, rank=rank, block_width=block_width, steps=steps, head_dim=head_dim,
                        seed=seed, device=device, guard=guard)
        directory = output_dir / f"block_{layer:02d}"; directory.mkdir(parents=True, exist_ok=False)
        with (directory / "fit.pt").open("xb") as stream: torch.save(fit, stream)
        with (directory / "fit.json").open("x") as stream:
            json.dump(fit["metadata"], stream, indent=2, allow_nan=False); stream.write("\n")
        result[str(layer)] = {"fit": str(directory / "fit.pt"), "metadata": str(directory / "fit.json")}
        message = {"phase": "krylov_basis_complete", "layer": layer, "rank": rank,
            "krylov_dimension": fit["metadata"]["krylov_dimension"], "captured_energy": fit["metadata"]["captured_normalized_gram_energy"],
            "numerical_null_completion": fit["metadata"]["retained_numerical_null_completion"]}
        if progress is None: print(json.dumps(message, allow_nan=False), flush=True)
        else: progress(message)
    return result
