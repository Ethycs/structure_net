"""Shared task-benchmark polynomial graph fits with a retained native residual.

The exact algebraic certificate concerns the fitted graph ideal. Neural
activations need not lie on that variety; held-out relation and compiled-output
errors are reported separately. A fixed removal budget is not called coverage.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch


COEFFICIENT_DENOMINATOR = 2**20


def polynomial_features(z: torch.Tensor) -> torch.Tensor:
    """[1, z, z_i*z_j for i<=j], in row-major upper-triangle order."""
    pairs = torch.triu_indices(z.shape[-1], z.shape[-1], device=z.device)
    return torch.cat((torch.ones_like(z[..., :1]), z,
                      z[..., pairs[0]] * z[..., pairs[1]]), dim=-1)


def _chunks(value, rows, size):
    count = len(value) if rows is None else len(rows)
    for start in range(0, count, size):
        yield value[start:start + size] if rows is None else value.index_select(0, rows[start:start + size])


def _normalization(values, chunk_rows):
    total = torch.zeros(values.shape[1], device=values.device, dtype=torch.float64)
    squares = torch.zeros_like(total)
    for block in _chunks(values, None, chunk_rows):
        x = block.double()
        if not torch.isfinite(x).all():
            raise ValueError("Training activations contain nonfinite values")
        total += x.sum(0)
        squares += x.square().sum(0)
    mean = total / len(values)
    variance = ((squares - total.square() / len(values)) / (len(values) - 1)).clamp_min(0)
    dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    return mean.to(dtype), variance.sqrt().clamp_min(1e-5).to(dtype)


@torch.no_grad()
def select_nested_anchors(jacobian_rows, scale, max_anchors=32, *, seed=861,
                          svd_rank=64, oversample=16, power_iterations=1,
                          chunk_rows=256):
    """Randomized SVD of J*scale, streaming all rows without an n-by-n Gramian.

    J contains derivatives with respect to raw activations. Multiplication by
    scale expresses derivatives in standardized activation coordinates. Rows
    are not individually normalized. Greedy column pivots of S*Vh produce one
    nested anchor bank, with deterministic lowest-index tie breaking.
    """
    if jacobian_rows.ndim != 2 or scale.shape != (jacobian_rows.shape[1],):
        raise ValueError("Jacobian/scale shapes do not match")
    if chunk_rows < 1 or power_iterations < 0 or oversample < 0 or svd_rank < max_anchors:
        raise ValueError("Invalid sketch settings")
    m, n = jacobian_rows.shape
    rank = min(svd_rank, m, n)
    if not 1 <= max_anchors <= rank:
        raise ValueError("Too few Jacobian rows/coordinates for the requested anchors")
    width = min(rank + oversample, m, n)
    dtype = torch.float64 if jacobian_rows.dtype == torch.float64 else torch.float32
    s = scale.to(device=jacobian_rows.device, dtype=dtype)
    generator = torch.Generator(device=jacobian_rows.device).manual_seed(seed)
    omega = torch.randn(n, width, device=jacobian_rows.device, dtype=dtype, generator=generator)
    energy = torch.zeros((), device=jacobian_rows.device, dtype=torch.float64)

    def rows():
        for start in range(0, m, chunk_rows):
            block = jacobian_rows[start:start + chunk_rows].to(dtype) * s
            if not torch.isfinite(block).all():
                raise ValueError("Jacobian contains nonfinite values")
            yield start, block

    def multiply_right(right):
        out = torch.empty(m, right.shape[1], device=right.device, dtype=dtype)
        for start, block in rows():
            out[start:start + len(block)] = block @ right
        return out

    q = torch.linalg.qr(multiply_right(omega), mode="reduced").Q
    for _ in range(power_iterations):
        z = torch.zeros(n, q.shape[1], device=q.device, dtype=dtype)
        for start, block in rows():
            z.addmm_(block.T, q[start:start + len(block)])
        z = torch.linalg.qr(z, mode="reduced").Q
        q = torch.linalg.qr(multiply_right(z), mode="reduced").Q
    small = torch.zeros(q.shape[1], n, device=q.device, dtype=dtype)
    for start, block in rows():
        small.addmm_(q[start:start + len(block)].T, block)
        energy += block.double().square().sum()
    if not torch.isfinite(energy) or energy <= 0:
        raise ValueError("Jacobian energy must be finite and positive")
    _, singular, vh = torch.linalg.svd(small, full_matrices=False)
    threshold = singular[0] * max(m, n) * torch.finfo(dtype).eps
    observed_rank = int((singular > threshold).sum())
    if observed_rank < max_anchors:
        raise ValueError("Insufficient independent task-Jacobian directions for the anchor bank")
    remaining = (singular[:rank, None] * vh[:rank]).clone()
    selected = []
    pivot_norms = []
    for _ in range(max_anchors):
        norms = remaining.square().sum(0)
        if selected:
            norms[selected] = -1
        index = int(norms.argmax())
        value = norms[index]
        if not torch.isfinite(value) or value <= 0:
            raise ValueError("Anchor pivots lost independence")
        direction = remaining[:, index] / value.sqrt()
        remaining -= direction[:, None] * (direction @ remaining)[None, :]
        selected.append(index)
        pivot_norms.append(float(value.sqrt()))
    metadata = {
        "seed": seed, "jacobian_rows": m, "coordinates": n,
        "requested_svd_rank": svd_rank, "retained_svd_components": rank,
        "sketch_width": width, "power_iterations": power_iterations,
        "observed_rank_in_sketch": observed_rank,
        "captured_energy_fraction": float(singular[:rank].double().square().sum() / energy),
        "range_projection_energy_fraction": float(singular.double().square().sum() / energy),
        "total_standardized_jacobian_energy": float(energy),
        "singular_values": singular[:rank].cpu().tolist(), "anchors": selected,
        "pivot_norms": pivot_norms, "all_rows_used": True,
        "row_normalization": "none", "coordinate_normalization": "J multiplied by pooled training activation scale",
        "algorithm": "randomized range finder, sketch SVD, greedy independent coordinate pivots",
    }
    return selected, metadata


def _fit(values, rows, anchors, mean, scale, ridge, chunk_rows):
    width = 1 + len(anchors) + len(anchors) * (len(anchors) + 1) // 2
    gram = mean.new_zeros(width, width)
    cross = mean.new_zeros(width, len(mean))
    count = 0
    for block in _chunks(values, rows, chunk_rows):
        z = (block.to(mean.dtype) - mean) / scale
        x = polynomial_features(z[:, anchors])
        gram.addmm_(x.T, x)
        cross.addmm_(x.T, z)
        count += len(block)
    penalty = torch.eye(width, device=mean.device, dtype=mean.dtype) * ridge
    penalty[0, 0] = 0
    coefficient = torch.linalg.solve(gram / count + penalty, cross / count).T.contiguous()
    coefficient = (coefficient * COEFFICIENT_DENOMINATOR).round() / COEFFICIENT_DENOMINATOR
    if not torch.isfinite(coefficient).all():
        raise FloatingPointError("Nonfinite polynomial coefficient")
    return coefficient


def _relation_errors(values, rows, coefficient, anchors, mean, scale, chunk_rows):
    errors, targets = torch.zeros_like(mean), torch.zeros_like(mean)
    count = 0
    for block in _chunks(values, rows, chunk_rows):
        z = (block.to(mean.dtype) - mean) / scale
        predicted = polynomial_features(z[:, anchors]) @ coefficient.T
        errors += (predicted - z).square().sum(0)
        targets += z.square().sum(0)
        count += len(block)
    if not torch.isfinite(errors).all() or not torch.isfinite(targets).all():
        raise FloatingPointError("Nonfinite relation diagnostics")
    return (errors / targets.clamp_min(1e-12)).sqrt(), (errors / count).sqrt() * scale


def _compiled_output_error(values, rows, coefficient, removed, anchors, mean, scale,
                           output_weight, chunk_rows):
    removed_weight = output_weight[:, removed]
    selected_coefficient = coefficient[removed]
    error = torch.zeros((), device=mean.device, dtype=torch.float64)
    target = torch.zeros_like(error)
    count = 0
    for block in _chunks(values, rows, chunk_rows):
        h = block.to(mean.dtype)
        z = (h[:, anchors] - mean[anchors]) / scale[anchors]
        predicted = polynomial_features(z) @ selected_coefficient.T
        raw_error = predicted * scale[removed] + mean[removed] - h[:, removed]
        delta = raw_error @ removed_weight.T
        original = h @ output_weight.T
        error += delta.double().square().sum()
        target += original.double().square().sum()
        count += len(h)
    return {"relative_rmse": float((error / target.clamp_min(1e-24)).sqrt()),
            "output_rmse": float((error / (count * output_weight.shape[0])).sqrt()),
            "squared_error": float(error), "target_squared_energy": float(target), "rows": count}


def _summary(values):
    return {"mean": float(values.mean()), "median": float(values.median()),
            "p90": float(torch.quantile(values, .9)), "max": float(values.max())}


def _certificate(coefficient, anchors, removed, coordinates):
    eliminated = removed.cpu().tolist()
    if set(anchors) & set(eliminated) or len(set(eliminated)) != len(eliminated):
        raise ValueError("Graph outputs must be distinct from retained anchors")
    expected = 1 + len(anchors) + len(anchors) * (len(anchors) + 1) // 2
    if coefficient.shape != (coordinates, expected):
        raise ValueError("Polynomial feature layout mismatch")
    return {
        "field": "QQ", "generator_form_standardized": "z_i - p_i(z_anchors)",
        "raw_coordinate_equivalent": "h_i - [mean_i + scale_i*p_i((h_anchors-mean_anchors)/scale_anchors)]",
        "coefficient_denominator_standardized": COEFFICIENT_DENOMINATOR,
        "generator_count": len(eliminated), "leading_raw_hidden_coordinates": eliminated,
        "order": "block lexicographic: eliminated hidden coordinates before retained coordinates",
        "distinct_monic_leading_variables": True, "buchberger_product_criterion": True,
        "reduced_graph_basis": True,
        "reduced_reason": "tails contain only retained anchors, so no tail monomial is divisible by any leading eliminated variable",
        "standard_monomials": "all monomials in retained (non-eliminated) coordinates; infinite dimensional",
        "compiled_feature_subspace": "degree at most two in the selected anchors; finite computational truncation",
        "compiled_feature_count": expected,
        "scope": "exact certificate for the fitted rational graph ideal, not identities of the pretrained neural network",
        "neural_relation_defects": "measured on native and intended-interface development activations",
    }


@torch.no_grad()
def fit_benchmark_groebner(
    train_activations, train_families: Sequence[str], jacobian_rows, output_weight,
    native_dev_activations, dev_families: Sequence[str], *,
    compressed_dev_activations=None, anchor_widths=(16, 24, 32), remove_count=1024,
    seed=861, svd_rank=64, oversample=16, power_iterations=1, ridge=.01,
    eligibility_nrmse=.5, chunk_rows=256,
):
    """Fit pooled task-benchmark graph relations and choose by dev output error.

    All activation inputs are raw post-SwiGLU h, with shape [positions, channels].
    ``output_weight`` is the intended interface's [output_width, channels] map;
    e.g. a reduced block's U.T @ down_proj. The same map evaluates native and
    compressed development rows. Jacobians are with respect to raw h.

    Removal is selected using pooled TRAIN raw reconstruction RMSE times output
    column norm, with a fixed budget for every candidate. Candidate width is
    selected using compiled-output relative error on compressed development if
    supplied, otherwise native development. No evaluation labels are consumed.
    """
    values = [train_activations, jacobian_rows, output_weight, native_dev_activations]
    if compressed_dev_activations is not None:
        values.append(compressed_dev_activations)
    if any(v.ndim != 2 or min(v.shape) == 0 or not v.is_floating_point() for v in values):
        raise ValueError("Inputs must be nonempty floating-point matrices")
    n = train_activations.shape[1]
    if any(v.shape[1] != n or v.device != train_activations.device for v in values):
        raise ValueError("All inputs must share the activation channel count and device")
    if len(train_activations) < 2 or len(train_families) != len(train_activations) or len(dev_families) != len(native_dev_activations):
        raise ValueError("Activation rows and family labels do not match")
    if compressed_dev_activations is not None and compressed_dev_activations.shape != native_dev_activations.shape:
        raise ValueError("Compressed/native development rows must be aligned")
    widths = sorted(set(anchor_widths))
    if not widths or widths[0] < 1 or not 0 < remove_count <= n - widths[-1]:
        raise ValueError("Invalid anchors/removal budget")
    if chunk_rows < 1 or ridge < 0 or eligibility_nrmse <= 0:
        raise ValueError("Invalid fitting settings")
    families = sorted(set(train_families))
    if set(dev_families) != set(families):
        raise ValueError("Training and development must contain the same task families")
    mean, scale = _normalization(train_activations, chunk_rows)
    weight = output_weight.to(mean.dtype)
    if not torch.isfinite(weight).all():
        raise ValueError("Nonfinite output weight")
    anchors, anchor_info = select_nested_anchors(
        jacobian_rows, scale, widths[-1], seed=seed, svd_rank=svd_rank,
        oversample=oversample, power_iterations=power_iterations, chunk_rows=chunk_rows,
    )
    device = train_activations.device
    train_rows = {f: torch.tensor([i for i, x in enumerate(train_families) if x == f], device=device) for f in families}
    dev_rows = {f: torch.tensor([i for i, x in enumerate(dev_families) if x == f], device=device) for f in families}
    interfaces = {"native_dev": native_dev_activations}
    if compressed_dev_activations is not None:
        interfaces["compressed_dev"] = compressed_dev_activations
    choice_interface = "compressed_dev" if compressed_dev_activations is not None else "native_dev"
    impact = weight.square().sum(0).sqrt()
    cpu_mean, cpu_scale = mean.cpu(), scale.cpu()
    fits, candidate_metrics = {}, {}
    for width in widths:
        chosen = anchors[:width]
        coefficient = _fit(train_activations, None, chosen, mean, scale, ridge, chunk_rows)
        train_errors, raw_errors = _relation_errors(train_activations, None, coefficient, chosen, mean, scale, chunk_rows)
        scores = raw_errors * impact
        scores[chosen] = torch.inf
        removed = torch.argsort(scores, stable=True)[:remove_count]
        own = {f: _fit(train_activations, train_rows[f], chosen, mean, scale, ridge, chunk_rows) for f in families}
        metrics = {"anchor_width": width, "feature_count": coefficient.shape[1], "removed_count": remove_count,
                   "training_selected_nrmse": _summary(train_errors[removed]),
                   "removal_selection": "lowest pooled-training raw activation RMSE times output-column L2 norm; stable coordinate-index ties",
                   "normalization": "pooled training only; reused unchanged for every family and interface",
                   "eligibility_nrmse": eligibility_nrmse, "ridge": ridge, "interfaces": {}}
        error_tensors = {}
        for name, activation in interfaces.items():
            common, task_specific, family_metrics = [], [], {}
            for f in families:
                common_error, _ = _relation_errors(activation, dev_rows[f], coefficient, chosen, mean, scale, chunk_rows)
                own_error, _ = _relation_errors(activation, dev_rows[f], own[f], chosen, mean, scale, chunk_rows)
                common.append(common_error); task_specific.append(own_error)
                family_metrics[f] = {
                    "common_selected_nrmse": _summary(common_error[removed]),
                    "task_specific_selected_nrmse": _summary(own_error[removed]),
                    "common_output": _compiled_output_error(activation, dev_rows[f], coefficient, removed, chosen, mean, scale, weight, chunk_rows),
                    "task_specific_output": _compiled_output_error(activation, dev_rows[f], own[f], removed, chosen, mean, scale, weight, chunk_rows),
                }
            common = torch.stack(common); task_specific = torch.stack(task_specific)
            eligible = common <= eligibility_nrmse
            own_eligible = task_specific <= eligibility_nrmse
            eligible[:, chosen] = False; own_eligible[:, chosen] = False
            occurrences = eligible.sum(0)
            task_uses, unique = int(occurrences.sum()), int((occurrences > 0).sum())
            selected_all = (common[:, removed] <= eligibility_nrmse).all(0)
            metrics["interfaces"][name] = {
                "pooled_output": _compiled_output_error(activation, None, coefficient, removed, chosen, mean, scale, weight, chunk_rows),
                "by_task": family_metrics,
                "redundancy": {"family_order": families, "task_uses": task_uses, "unique_elements": unique,
                    "redundant_occurrences": task_uses - unique,
                    "redundant_fraction": (task_uses - unique) / task_uses if task_uses else 0.,
                    "shared_all_tasks": int((occurrences == len(families)).sum()),
                    "eligible_per_task": eligible.sum(1).cpu().tolist(),
                    "task_specific_eligible_per_task": own_eligible.sum(1).cpu().tolist(),
                    "overlap_counts": (eligible[:, None, :] & eligible[None, :, :]).sum(-1).cpu().tolist(),
                    "selected_not_eligible_all_tasks": int((~selected_all).sum()),
                    "all_removed_meet_diagnostic_threshold": bool(selected_all.all()),
                    "scope": "common fitted relations meeting a diagnostic error threshold across tasks; not exact neural identities"},
            }
            error_tensors[name] = {"common": common.cpu(), "task_specific": task_specific.cpu()}
        certificate = _certificate(coefficient, chosen, removed, n)
        metrics["certificate"] = certificate
        fits[str(width)] = {"anchors": chosen, "mean": cpu_mean, "scale": cpu_scale,
            "coefficients": {"2": coefficient.cpu()}, "removed": removed.cpu().long(),
            "metrics": metrics, "per_channel_nrmse": error_tensors}
        candidate_metrics[str(width)] = metrics
        del own, coefficient, common, task_specific
    selected = min(widths, key=lambda w: (candidate_metrics[str(w)]["interfaces"][choice_interface]["pooled_output"]["relative_rmse"], w))
    return {"fits": fits, "selected_anchor_width": selected, "anchor_selection": anchor_info,
        "metrics": {"families": families, "calibration_rows": len(train_activations),
                    "native_development_rows": len(native_dev_activations), "selection_interface": choice_interface,
                    "selection_metric": "compiled local output relative RMSE; smaller anchor width breaks ties",
                    "selected_anchor_width": selected, "candidates": candidate_metrics,
                    "scope": "task-benchmark-derived approximate polynomial graph compilation with native channels retained"}}
