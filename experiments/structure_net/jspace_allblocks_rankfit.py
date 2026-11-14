"""Candidate small-SVD numerical-rank admission, with unchanged graph fitting.

This version changes only which matrix shape sets the numerical-rank tolerance.
The copied benchmark orchestration calls frozen arithmetic/error helpers. The
selector uses the independently tested unchanged streamed sketch diagnostic.
No raw rows or task weights are changed, and no global functions are patched.
"""
from collections import Counter
from collections.abc import Sequence
import json
import math
from pathlib import Path

import torch

from .jspace_benchmark_groebner import (
    _normalization, _fit, _relation_errors, _compiled_output_error, _summary, _certificate,
)
from .jspace_bench_data import sha, write
from .jspace_qwen_parallel_fit import _guard, _event, _save, _read_bits, _ProgressRows
from .jspace_rank_diagnostic import diagnose


class RankAdmissionError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


@torch.no_grad()
def select_nested_anchors(jacobian_rows, scale, max_anchors=32, *, seed=861,
                          svd_rank=64, oversample=16, power_iterations=1,
                          chunk_rows=256, guard=None, on_spectrum=None):
    report, state = diagnose(jacobian_rows, scale, seed=seed, anchors=max_anchors,
        svd_rank=svd_rank, oversample=oversample, power_iterations=power_iterations,
        chunk_rows=chunk_rows, guard=guard, progress=lambda value: None)
    m, n = jacobian_rows.shape; rank = min(svd_rank, m, n)
    gates = report["rank_cutoffs"]
    diagnostics = {"rule_id": "actual_sketch_svd_shape_fp64" if state["singular"].dtype == torch.float64 else "actual_sketch_svd_shape_fp32",
        "admission_matrix": "actual small SVD matrix",
        "admission_rule": "singular > singular_max * max(small.shape) * finfo(compute_dtype).eps",
        "actual_svd_shape": report["small_shape"], "original_shape": [m, n],
        "original_shape_gate": gates["frozen_original_shape"], "small_svd_gate": gates["actual_svd_shape"],
        "requested_anchors": max_anchors, "all_singular_values": report["singular_values"],
        "s24_over_s1": report["s24_over_s1"], "s64_over_s1": report["s64_over_s1"],
        "diagnostic_anchor_indices": report["diagnostic_anchor_indices"],
        "diagnostic_pivot_norms": report["diagnostic_pivot_norms"],
        "admitted": gates["actual_svd_shape"]["rank"] >= max_anchors,
        "row_concentration": report["row_concentration"],
        "scope": "numerical-rank guard only; raw rows, weighting, normalization, sketch and fixed fit budgets unchanged"}
    if on_spectrum is not None: on_spectrum(diagnostics, state)
    if not diagnostics["admitted"]:
        raise RankAdmissionError("Insufficient independent task-Jacobian directions under the actual small-SVD tolerance", diagnostics)
    selected, pivots = report["diagnostic_anchor_indices"], report["diagnostic_pivot_norms"]
    if len(selected) != max_anchors:
        raise RankAdmissionError("Anchor pivots lost independence", diagnostics)
    metadata = {"seed": seed, "jacobian_rows": m, "coordinates": n,
        "requested_svd_rank": svd_rank, "retained_svd_components": rank,
        "sketch_width": report["small_shape"][0], "power_iterations": power_iterations,
        # Preserve the old descriptive field exactly; the admission rank is explicit below.
        "observed_rank_in_sketch": gates["frozen_original_shape"]["rank"],
        "captured_energy_fraction": report["captured_energy_fraction"],
        "range_projection_energy_fraction": report["range_projection_energy_fraction"],
        "total_standardized_jacobian_energy": report["total_standardized_jacobian_energy"],
        "singular_values": report["singular_values"][:rank], "anchors": selected,
        "pivot_norms": pivots, "all_rows_used": True,
        "row_normalization": "none", "coordinate_normalization": "J multiplied by pooled training activation scale",
        "algorithm": "randomized range finder, sketch SVD, greedy independent coordinate pivots",
        "rank_admission": diagnostics}
    return selected, metadata


@torch.no_grad()
def fit_benchmark_groebner(
    train_activations, train_families: Sequence[str], jacobian_rows, output_weight,
    native_dev_activations, dev_families: Sequence[str], *,
    compressed_dev_activations=None, anchor_widths=(16, 24, 32), remove_count=1024,
    seed=861, svd_rank=64, oversample=16, power_iterations=1, ridge=.01,
    eligibility_nrmse=.5, chunk_rows=256, selector=None, selector_kwargs=None,
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
    anchors, anchor_info = (selector or select_nested_anchors)(
        jacobian_rows, scale, widths[-1], seed=seed, svd_rank=svd_rank,
        oversample=oversample, power_iterations=power_iterations, chunk_rows=chunk_rows, **(selector_kwargs or {}),
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


@torch.no_grad()
def fit_relation(block_dir, *, device="cpu", guard=None, seed=861, anchors=24, remove_count=1024,
                 pad_to_multiple=128, svd_rank=64, oversample=16, power_iterations=1,
                 ridge=.01, eligibility_nrmse=.5, chunk_rows=256, progress=None):
    """Same artifact interface as the frozen fitter; only rank admission changes."""
    device = _guard(device, guard); block_dir = Path(block_dir)
    outputs = ("relation.pt", "relations.json", "rank_spectrum.pt", "rank_diagnostic.json")
    if any((block_dir / name).exists() for name in outputs): raise FileExistsError("Relation or rank diagnosis already exists")
    interface = json.loads((block_dir / "interface.json").read_text())
    _event(progress, "relation_verify_inputs", layer=interface["layer"])
    for name, expected in interface["input_sha256"].items():
        if sha(name) != expected: raise ValueError("Changed fitting input: " + name)
    training = _read_bits(interface["native_calibration"], slice(None), device).float()
    native_dev = _read_bits(interface["native_development"], slice(None), device).float()
    compiled_dev = _read_bits(interface["compiled_development"], slice(None), device).float()
    output_weight = torch.load(interface["output_weight"], map_location="cpu", weights_only=True).to(device=device, dtype=torch.float32)
    if not torch.equal(output_weight, output_weight.bfloat16().float()): raise ValueError("Output map lost intended BF16 rounding provenance")

    def preserve_spectrum(diagnostics, state):
        _save(block_dir / "rank_spectrum.pt", state)
        write(block_dir / "rank_diagnostic.json", {"layer": interface["layer"], "seed": seed,
            "teacher_fingerprint": interface["teacher_fingerprint"], "diagnostics": diagnostics,
            "spectral_state_sha256": sha(block_dir / "rank_spectrum.pt"),
            "input_sha256": {**interface["input_sha256"], str(block_dir / "interface.json"): sha(block_dir / "interface.json")}})
        _event(progress, "rank_admission", layer=interface["layer"], admitted=diagnostics["admitted"],
            original_rank=diagnostics["original_shape_gate"]["rank"], small_svd_rank=diagnostics["small_svd_gate"]["rank"])

    _event(progress, "relation_fit_start", layer=interface["layer"], calibration_rows=len(training), development_rows=len(native_dev))
    with _ProgressRows(interface["jacobian"], device=device, max_rows=chunk_rows,
                       progress=progress, layer=interface["layer"], guard=guard) as rows:
        result = fit_benchmark_groebner(training, interface["calibration_families"], rows, output_weight,
            native_dev, interface["development_families"], compressed_dev_activations=compiled_dev,
            anchor_widths=(anchors,), remove_count=remove_count, seed=seed, svd_rank=svd_rank, oversample=oversample,
            power_iterations=power_iterations, ridge=ridge, eligibility_nrmse=eligibility_nrmse, chunk_rows=chunk_rows,
            selector_kwargs={"guard": guard, "on_spectrum": preserve_spectrum})
        streamed = rows.metadata()
    chosen = result["fits"][str(anchors)]
    _save(block_dir / "relation.pt", chosen)
    logical = training.shape[1] - remove_count + 1 + anchors + anchors * (anchors + 1) // 2
    report = {k: v for k, v in result.items() if k != "fits"}
    report.update(layer=interface["layer"], teacher_fingerprint=interface["teacher_fingerprint"],
        training_families=dict(Counter(interface["calibration_families"])), development_families=dict(Counter(interface["development_families"])),
        input_sha256={**interface["input_sha256"], str(block_dir / "interface.json"): sha(block_dir / "interface.json")},
        streamed_jacobian=streamed, fixed_anchor_width=anchors, removed_channels=remove_count,
        runtime_padding={"multiple": pad_to_multiple, "logical_width": logical,
                         "padded_width": math.ceil(logical / pad_to_multiple) * pad_to_multiple if pad_to_multiple else logical},
        source="fixed accepted-teacher grouped full-context BF16 VJPs and native task activations",
        relation_sha256=sha(block_dir / "relation.pt"),
        rank_diagnostic_sha256=sha(block_dir / "rank_diagnostic.json"), rank_spectrum_sha256=sha(block_dir / "rank_spectrum.pt"),
        scope="approximate fitted graph ideal; exact graph certificate is not an identity of the teacher network; no test selection")
    write(block_dir / "relations.json", report); _guard(device, guard)
    _event(progress, "relation_fit_complete", layer=interface["layer"], anchors=anchors, removed=remove_count)
    return report
