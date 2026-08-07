#!/usr/bin/env python3
"""Audit task-tangent versus task-kernel residuals in a frozen continuation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import experiments.structure_net.tinyllm_frozen_writer_capacity as predecessor
import experiments.structure_net.tinyllm_fixed_semantic_gauge_writer as fixed
import experiments.structure_net.tinyllm_cross_seed_causal_carrier_transport as transport
from experiments.structure_net import tinyllm_continuous_readout_calibration as readout
from experiments.structure_net import tinyllm_reynolds_character_coupling as coupling
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-c2-local-continuation-tangent-kernel.v1.1"
HYPOTHESIS_ID = "tinyllm-c2-local-continuation-tangent-kernel-v1"
PREDECESSOR_CAMPAIGN_SHA256 = (
    "7ab6079d56bc8dc78cc5e1f4011c581b8f507bcd8ec20573bbf8b11227df8d1b"
)
PREDECESSOR_IMPLEMENTATION_SHA256 = (
    "d53edaedd49ae553af9f8393d92254664239e5100246ac0fd3a06cb420ca80ed"
)
REGIMES = fixed.REGIMES
HELDOUT_COHORTS = fixed.HELDOUT_COHORTS


@dataclass(frozen=True)
class LocalContinuationConfig:
    predecessor_root: str = (
        "data/experiments/tinyllm_frozen_writer_capacity/"
        "20260807_d6_preregistered_diagnostic"
    )
    seeds: tuple[int, ...] = fixed.PRIMARY_SEEDS
    orbit_count: int = 64
    random_controls: int = 8
    fourier_order: int = 4
    jacobian_rtol: float = 1e-6
    jacobian_atol: float = 1e-10
    finite_difference_step: float = 1e-2
    finite_difference_tolerance: float = 0.05
    decomposition_tolerance: float = 1e-6
    replay_tolerance: float = 1e-6
    kernel_mean_movement_ceiling_bins: float = 0.05
    kernel_p95_movement_ceiling_bins: float = 0.20
    random_advantage_bins: float = 0.125
    kernel_random_advantage_bins: float = 0.05
    metric_frobenius_ceiling: float = 0.35
    metric_top_cosine_floor: float = 0.90
    device: str = "cuda"
    allow_underpowered: bool = False
    post_outcome_corrective_replication: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != fixed.PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("primary diagnostic seeds are fixed to 7,29,53")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.orbit_count != 64 and not self.allow_underpowered:
            raise ValueError("primary diagnostic fixes 64 exact orbits per cell")
        if self.orbit_count < 8:
            raise ValueError("at least eight exact orbits are required")
        if self.random_controls != 8 and not self.allow_underpowered:
            raise ValueError("primary diagnostic fixes eight random controls")
        if self.random_controls < 2:
            raise ValueError("at least two random controls are required")
        if self.fourier_order != 4:
            raise ValueError("the predecessor writer is fixed to Fourier order four")
        if not self.allow_underpowered and not self.post_outcome_corrective_replication:
            raise ValueError(
                "primary execution requires explicit post-outcome corrective replication"
            )
        positive = (
            self.jacobian_rtol,
            self.jacobian_atol,
            self.finite_difference_step,
            self.finite_difference_tolerance,
            self.decomposition_tolerance,
            self.replay_tolerance,
            self.kernel_mean_movement_ceiling_bins,
            self.kernel_p95_movement_ceiling_bins,
            self.random_advantage_bins,
            self.kernel_random_advantage_bins,
            self.metric_frobenius_ceiling,
            self.metric_top_cosine_floor,
        )
        if min(positive) <= 0.0:
            raise ValueError("all numerical thresholds must be positive")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _json_compatible(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(predecessor.__file__),
        Path(fixed.__file__),
        Path(transport.__file__),
        Path(transport.rank.__file__),
        Path(readout.__file__),
        Path(coupling.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evidence_role(config: LocalContinuationConfig) -> str:
    if config.allow_underpowered:
        return "systems_lifecycle_only_not_quality_evidence"
    if config.post_outcome_corrective_replication:
        return "post_outcome_corrective_replication_evidence"
    raise ValueError("non-shakedown evidence must be an explicit corrective replication")


def complex_posterior_moment(posterior: torch.Tensor) -> torch.Tensor:
    """Return real/imaginary coordinates of the categorical circular moment."""
    posterior = posterior.double()
    angle = torch.arange(
        posterior.shape[-1], device=posterior.device, dtype=torch.float64
    )
    angle = angle * (2.0 * math.pi / posterior.shape[-1])
    return torch.stack(
        (posterior @ torch.cos(angle), posterior @ torch.sin(angle)), dim=-1
    )


def tangent_kernel_decomposition(
    jacobian: torch.Tensor,
    residual: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, torch.Tensor]:
    """Split coordinate residuals into row-space and null-space components."""
    jacobian = jacobian.double()
    residual = residual.double()
    if jacobian.ndim != 3 or jacobian.shape[1:] != (2, 3):
        raise ValueError("jacobian must have shape [observations,2,3]")
    if residual.shape != (len(jacobian), 3):
        raise ValueError("residual must have shape [observations,3]")
    pseudoinverse = torch.linalg.pinv(jacobian, rtol=rtol, atol=atol)
    projector = pseudoinverse @ jacobian
    tangent = torch.einsum("nij,nj->ni", projector, residual)
    kernel = residual - tangent
    singular = torch.linalg.svdvals(jacobian)
    threshold = torch.maximum(
        torch.full_like(singular[:, :1], atol),
        singular[:, :1] * rtol,
    )
    rank = (singular > threshold).sum(dim=1)
    j_residual = torch.einsum("nij,nj->ni", jacobian, residual)
    j_tangent = torch.einsum("nij,nj->ni", jacobian, tangent)
    j_kernel = torch.einsum("nij,nj->ni", jacobian, kernel)
    scale = torch.linalg.vector_norm(j_residual, dim=1).clamp_min(1e-12)
    residual_scale = torch.linalg.vector_norm(residual, dim=1).clamp_min(1e-12)
    return {
        "projector": projector,
        "tangent": tangent,
        "kernel": kernel,
        "singular_values": singular,
        "rank": rank,
        "relative_decomposition_error": torch.linalg.vector_norm(
            residual - tangent - kernel, dim=1
        )
        / residual_scale,
        "relative_tangent_mismatch": torch.linalg.vector_norm(
            j_tangent - j_residual, dim=1
        )
        / scale,
        "relative_kernel_leakage": torch.linalg.vector_norm(j_kernel, dim=1)
        / scale,
    }


def norm_matched_random_controls(
    component: torch.Tensor, count: int, seed: int
) -> torch.Tensor:
    """Generate deterministic isotropic vectors with per-row matched norm."""
    component = component.double()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random = torch.randn(
        (count,) + tuple(component.shape), generator=generator, dtype=torch.float64
    )
    random_norm = torch.linalg.vector_norm(random, dim=-1, keepdim=True).clamp_min(
        1e-12
    )
    target_norm = torch.linalg.vector_norm(component.cpu(), dim=-1, keepdim=True)
    return (random / random_norm * target_norm.unsqueeze(0)).to(component.device)


def normalized_task_metric(jacobian: torch.Tensor) -> dict[str, torch.Tensor]:
    metric = torch.einsum("nai,naj->ij", jacobian.double(), jacobian.double())
    metric = metric / max(len(jacobian), 1)
    trace = torch.trace(metric)
    normalized = metric / trace.clamp_min(1e-24)
    eigenvalues, eigenvectors = torch.linalg.eigh(normalized)
    return {
        "metric": normalized,
        "eigenvalues": eigenvalues.flip(0),
        "top_eigenvector": eigenvectors[:, -1],
        "trace": trace,
    }


def metric_stability(
    metrics: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    frobenius: list[float] = []
    cosines: list[float] = []
    for left in range(len(metrics)):
        for right in range(left + 1, len(metrics)):
            frobenius.append(
                float(
                    torch.linalg.matrix_norm(
                        metrics[left]["metric"] - metrics[right]["metric"]
                    )
                )
            )
            cosines.append(
                float(
                    torch.dot(
                        metrics[left]["top_eigenvector"],
                        metrics[right]["top_eigenvector"],
                    ).abs()
                )
            )
    return {
        "pairwise_normalized_frobenius_distances": frobenius,
        "pairwise_top_eigenvector_absolute_cosines": cosines,
        "maximum_normalized_frobenius_distance": max(frobenius, default=0.0),
        "minimum_top_eigenvector_absolute_cosine": min(cosines, default=1.0),
    }


def _continue_tensor_with_grad(
    system: Any,
    cut: str,
    values: torch.Tensor,
    task: CircleTaskConfig,
    batch_size: int,
) -> torch.Tensor:
    device = values.device
    answer = torch.tensor(task.answer_token_ids, device=device)
    output = []
    for start in range(0, len(values), batch_size):
        value = values[start : start + batch_size]
        pieces = cut.split("_")
        block_index = int(pieces[1])
        if cut.endswith("post_attention"):
            block = system.model.transformer["h"][block_index]
            value = value + block.mlp(block.ln_2(value))
        for block in system.model.transformer["h"][block_index + 1 :]:
            value = block(value)
        logits = coupling.io_source._task_logits(system.model, value[:, -1, :], answer)
        output.append(torch.softmax(logits, -1))
    return torch.cat(output)


def continuation_moment_jacobian(
    system: Any,
    cut: str,
    propagated: torch.Tensor,
    coordinates: torch.Tensor,
    basis: torch.Tensor,
    task: CircleTaskConfig,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate each orbit's circular moment with respect to coordinates."""
    posteriors: list[torch.Tensor] = []
    jacobians: list[torch.Tensor] = []
    per_chunk = max(1, batch_size // 2)
    basis_native = basis.to(device=propagated.device, dtype=propagated.dtype)
    for start in range(0, len(coordinates), per_chunk):
        stop = min(start + per_chunk, len(coordinates))
        local = coordinates[start:stop].to(
            device=propagated.device, dtype=propagated.dtype
        )
        local = local.detach().requires_grad_(True)
        defect = (local @ basis_native).reshape_as(propagated[start:stop])
        state = propagated[start:stop] + defect
        repeated = coupling._repeat_patch(state, 2)
        posterior = _continue_tensor_with_grad(
            system, cut, repeated, task, batch_size
        ).reshape(stop - start, 2, -1)[:, 0]
        moment = complex_posterior_moment(posterior)
        real_grad = torch.autograd.grad(
            moment[:, 0].sum(), local, retain_graph=True
        )[0]
        imag_grad = torch.autograd.grad(moment[:, 1].sum(), local)[0]
        posteriors.append(posterior.detach())
        jacobians.append(torch.stack((real_grad, imag_grad), dim=1).double())
    return torch.cat(posteriors), torch.cat(jacobians)


@torch.no_grad()
def _posteriors_for_coordinates(
    system: Any,
    task: CircleTaskConfig,
    target_cut: str,
    propagated: torch.Tensor,
    basis: torch.Tensor,
    coordinate_states: Mapping[str, torch.Tensor],
    batch_size: int,
) -> dict[str, torch.Tensor]:
    basis_native = basis.to(device=propagated.device, dtype=propagated.dtype)
    states = {
        name: propagated
        + (coordinates.to(basis_native.dtype) @ basis_native).reshape_as(propagated)
        for name, coordinates in coordinate_states.items()
    }
    keys = tuple(states)
    repeated = torch.cat(
        [coupling._repeat_patch(states[key], 2) for key in keys], dim=0
    )
    combined = coupling._continue_tensor(
        system, target_cut, repeated, task, batch_size
    )
    per_state = len(propagated) * 2
    return {
        key: value.reshape(len(propagated), 2, -1)[:, 0]
        for key, value in zip(keys, combined.split(per_state))
    }


@torch.no_grad()
def finite_difference_error(
    system: Any,
    task: CircleTaskConfig,
    target_cut: str,
    propagated: torch.Tensor,
    coordinates: torch.Tensor,
    basis: torch.Tensor,
    jacobian: torch.Tensor,
    residual: torch.Tensor,
    step: float,
    batch_size: int,
) -> torch.Tensor:
    direction = residual.double()
    norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    fallback = torch.zeros_like(direction)
    fallback[:, 0] = 1.0
    direction = torch.where(norm > 1e-12, direction / norm.clamp_min(1e-12), fallback)
    posterior = _posteriors_for_coordinates(
        system,
        task,
        target_cut,
        propagated,
        basis,
        {
            "plus": coordinates + step * direction,
            "minus": coordinates - step * direction,
        },
        batch_size,
    )
    estimate = (
        complex_posterior_moment(posterior["plus"])
        - complex_posterior_moment(posterior["minus"])
    ) / (2.0 * step)
    predicted = torch.einsum("nij,nj->ni", jacobian.double(), direction)
    denominator = torch.maximum(
        torch.maximum(
            torch.linalg.vector_norm(estimate, dim=1),
            torch.linalg.vector_norm(predicted, dim=1),
        ),
        torch.full((len(direction),), 1e-8, dtype=torch.float64, device=direction.device),
    )
    return torch.linalg.vector_norm(estimate - predicted, dim=1) / denominator


def _movement_from_reference(
    posterior: torch.Tensor, reference: torch.Tensor
) -> dict[str, float]:
    angle, _ = readout.posterior_moment(posterior)
    reference_angle, _ = readout.posterior_moment(reference)
    width = 2.0 * math.pi / posterior.shape[-1]
    movement = readout._wrapped_difference(angle, reference_angle).abs() / width
    return {
        "mean_bins": float(movement.mean()),
        "p95_bins": float(torch.quantile(movement, 0.95)),
        "maximum_bins": float(movement.max()),
    }


@torch.no_grad()
def evaluate_states(
    system: Any,
    task: CircleTaskConfig,
    config: LocalContinuationConfig,
    cell: Mapping[str, Any],
    basis: torch.Tensor,
    coordinate_states: Mapping[str, torch.Tensor],
    rotation_bins: float,
) -> dict[str, Any]:
    coordinate_posteriors = _posteriors_for_coordinates(
        system,
        task,
        cell["target_cut"],
        cell["propagated"],
        basis,
        coordinate_states,
        transport._rank_config(fixed._transport_config(
            fixed.FixedSemanticGaugeWriterConfig(
                seeds=config.seeds,
                orbit_count=config.orbit_count,
                device=config.device,
                allow_underpowered=config.allow_underpowered,
            )
        )).continuation_batch_size,
    )
    direct = {
        "zero": cell["propagated"],
        "exact": cell["exact"],
    }
    direct_repeated = torch.cat(
        [coupling._repeat_patch(direct[name], 2) for name in direct], dim=0
    )
    direct_combined = coupling._continue_tensor(
        system,
        cell["target_cut"],
        direct_repeated,
        task,
        transport._rank_config(fixed._transport_config(
            fixed.FixedSemanticGaugeWriterConfig(
                seeds=config.seeds,
                orbit_count=config.orbit_count,
                device=config.device,
                allow_underpowered=config.allow_underpowered,
            )
        )).continuation_batch_size,
    )
    per_state = config.orbit_count * 2
    orbit_posteriors = dict(coordinate_posteriors)
    orbit_posteriors.update(
        {
            name: value.reshape(config.orbit_count, 2, -1)[:, 0]
            for name, value in zip(direct, direct_combined.split(per_state))
        }
    )
    exact_repeated = orbit_posteriors["exact"][:, None].expand(-1, 2, -1).reshape(
        per_state, -1
    )
    exact_diagnostics = coupling._diagnostics(exact_repeated, cell["dataset"])
    targets = torch.as_tensor(
        cell["dataset"].target_bins.reshape(config.orbit_count, 2)[:, 0],
        device=basis.device,
    )
    bins = len(task.answer_token_ids)
    readout_config = transport._readout_config(fixed._transport_config(
        fixed.FixedSemanticGaugeWriterConfig(
            seeds=config.seeds,
            orbit_count=config.orbit_count,
            device=config.device,
            allow_underpowered=config.allow_underpowered,
        )
    ))
    records: dict[str, Any] = {}
    for name, posterior in orbit_posteriors.items():
        repeated = posterior[:, None].expand(-1, 2, -1).reshape(per_state, -1)
        diagnostics = coupling._diagnostics(repeated, cell["dataset"])
        angle, _ = readout.posterior_moment(posterior)
        records[name] = {
            "continuous": readout.continuous_metrics(
                posterior,
                orbit_posteriors["exact"],
                diagnostics,
                exact_diagnostics,
                readout_config,
            ),
            "discrete": readout.discrete_readout_metrics(
                angle,
                targets,
                bins,
                rotation_bins,
                cell["baseline"]["exact_bin_accuracy"],
                readout_config.accuracy_loss_ceiling,
            ),
            "fisher_effect": transport.rank.heads._effect_preservation(
                posterior,
                orbit_posteriors["exact"],
                orbit_posteriors["zero"],
                readout_config.task_effect_floor,
            ),
            "diagnostics": diagnostics,
        }
    predicted = orbit_posteriors["predicted"]
    for name, posterior in orbit_posteriors.items():
        records[name]["movement_from_predicted"] = _movement_from_reference(
            posterior, predicted
        )
    return records


def _load_predecessor(
    config: LocalContinuationConfig,
) -> tuple[dict[str, Any], Path, dict[int, tuple[dict[str, Any], Path]]]:
    campaign_path = Path(config.predecessor_root) / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text())
    if (
        _sha256(campaign_path) != PREDECESSOR_CAMPAIGN_SHA256
        or campaign.get("schema_version") != predecessor.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != predecessor.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256")
        != PREDECESSOR_IMPLEMENTATION_SHA256
        or int(campaign.get("summary", {}).get("completed", -1)) != 3
    ):
        raise ValueError(f"invalid predecessor campaign {campaign_path}")
    details: dict[int, tuple[dict[str, Any], Path]] = {}
    for entry in campaign["results"]:
        path = Path(entry["path"])
        detail = json.loads(path.read_text())
        if (
            _sha256(path) != entry["result_sha256"]
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or detail.get("classification") != "small_writer_insufficient"
        ):
            raise ValueError(f"invalid predecessor result {path}")
        details[int(detail["seed"])] = (detail, path)
    if not set(config.seeds).issubset(details):
        raise ValueError("predecessor lacks a requested checkpoint")
    return campaign, campaign_path, details


def _fingerprint(
    config: LocalContinuationConfig,
    seed: int,
    predecessor_result_sha256: str,
    checkpoint_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "cohort_seeds": fixed.COHORT_SEEDS,
        "seed": seed,
        "predecessor_campaign_sha256": PREDECESSOR_CAMPAIGN_SHA256,
        "predecessor_result_sha256": predecessor_result_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def classify_checkpoint(
    *,
    valid: bool,
    tangent_pass: bool,
    kernel_inert: bool,
    tangent_specificity: bool,
    kernel_specificity: bool,
    metric_stable: bool,
    kernel_causally_active: bool,
    full_pass: bool,
) -> str:
    if not valid:
        return "invalid"
    specificity = tangent_specificity and kernel_specificity
    if tangent_pass and kernel_inert and specificity and metric_stable:
        return "stable_task_tangent_sufficient"
    if tangent_pass and kernel_inert and specificity:
        return "checkpoint_local_task_tangent_sufficient"
    if kernel_causally_active:
        return "nominal_kernel_causally_active"
    if full_pass and not tangent_pass and kernel_inert:
        return "nonlinear_tangent_kernel_interaction_required"
    return "mixed_local_continuation_geometry"


def _random_seed(seed: int, cohort: str, regime: str, family: str) -> int:
    payload = f"{HYPOTHESIS_ID}:{seed}:{cohort}:{regime}:{family}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _campaign_is_reusable(
    value: Mapping[str, Any],
    config: LocalContinuationConfig,
    implementation: str,
) -> bool:
    return bool(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("evidence_role") == _evidence_role(config)
        and value.get("implementation_sha256") == implementation
        and value.get("configuration") == _json_compatible(asdict(config))
        and int(value.get("summary", {}).get("completed", -1)) == len(config.seeds)
        and len(value.get("results", [])) == len(config.seeds)
        and all(
            Path(item.get("path", "")).is_file()
            and _sha256(Path(item["path"])) == item.get("result_sha256")
            for item in value.get("results", [])
        )
    )


def _cell_summary(
    jacobian: torch.Tensor,
    decomposition: Mapping[str, torch.Tensor],
    finite_error: torch.Tensor,
    metric: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    residual_norm = torch.linalg.vector_norm(
        decomposition["tangent"] + decomposition["kernel"], dim=1
    ).clamp_min(1e-12)
    return {
        "jacobian_rank_counts": {
            str(rank): int((decomposition["rank"] == rank).sum())
            for rank in sorted(set(int(value) for value in decomposition["rank"]))
        },
        "minimum_jacobian_rank": int(decomposition["rank"].min()),
        "mean_singular_values": decomposition["singular_values"].mean(0).tolist(),
        "minimum_leading_singular_value": float(
            decomposition["singular_values"][:, 0].min()
        ),
        "maximum_finite_difference_relative_error": float(finite_error.max()),
        "mean_finite_difference_relative_error": float(finite_error.mean()),
        "maximum_relative_decomposition_error": float(
            decomposition["relative_decomposition_error"].max()
        ),
        "maximum_relative_tangent_mismatch": float(
            decomposition["relative_tangent_mismatch"].max()
        ),
        "maximum_relative_kernel_leakage": float(
            decomposition["relative_kernel_leakage"].max()
        ),
        "mean_tangent_norm_fraction": float(
            (torch.linalg.vector_norm(decomposition["tangent"], dim=1) / residual_norm).mean()
        ),
        "mean_kernel_norm_fraction": float(
            (torch.linalg.vector_norm(decomposition["kernel"], dim=1) / residual_norm).mean()
        ),
        "normalized_task_metric": metric["metric"].tolist(),
        "task_metric_eigenvalues": metric["eigenvalues"].tolist(),
        "task_metric_top_eigenvector": metric["top_eigenvector"].tolist(),
        "unnormalized_task_metric_trace": float(metric["trace"]),
        "jacobian_finite": bool(torch.isfinite(jacobian).all()),
    }


def _aggregate_state_mean(cells: Sequence[Mapping[str, Any]], state: str) -> float:
    return sum(
        cell["states"][state]["continuous"]["mean_moment_shift_bins"]
        for cell in cells
    ) / len(cells)


def _aggregate_movement(cells: Sequence[Mapping[str, Any]], state: str) -> float:
    return sum(
        cell["states"][state]["movement_from_predicted"]["mean_bins"]
        for cell in cells
    ) / len(cells)


def run_campaign(config: LocalContinuationConfig, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    campaign_started = time.perf_counter()
    implementation = _implementation_digest()
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text())
        if _campaign_is_reusable(existing, config, implementation):
            print("aggregate already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed aggregate {campaign_path}")

    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    predecessor_campaign, predecessor_path, predecessor_details = _load_predecessor(
        config
    )
    base_config = fixed.FixedSemanticGaugeWriterConfig(
        seeds=config.seeds,
        orbit_count=config.orbit_count,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )
    transport_config = fixed._transport_config(base_config)
    rank_config = transport._rank_config(transport_config)
    bridge = transport.rank._bridge_config(rank_config)
    datasets = {
        cohort: {
            regime: transport.rank.deck.generate_exact_orbits(
                task,
                k=2,
                orbit_count=config.orbit_count,
                seed=fixed.COHORT_SEEDS[cohort][regime],
                regime=regime,
            )
            for regime in REGIMES
        }
        for cohort in fixed.COHORT_SEEDS
    }

    results = []
    for seed in config.seeds:
        seed_started = time.perf_counter()
        prior, prior_path = predecessor_details[seed]
        system, provenance = transport.rank.deck.load_source(
            task, bridge, 2, seed, device
        )
        system.model.eval()
        for parameter in system.model.parameters():
            parameter.requires_grad_(False)
        if provenance["checkpoint_sha256"] != prior["provenance"][
            "checkpoint_sha256"
        ]:
            raise ValueError(f"checkpoint mismatch for seed {seed}")
        frozen, frozen_path = transport.rank._load_character_source(
            rank_config, seed, provenance["checkpoint_sha256"]
        )
        basis, basis_summary = transport._fit_seed_basis(
            system,
            task,
            transport_config,
            rank_config,
            bridge,
            frozen,
            seed,
            device,
        )
        readout_predecessor = transport._load_readout_predecessor(
            transport_config, seed, provenance["checkpoint_sha256"]
        )
        cells = {
            cohort: {
                regime: transport._extract_cell(
                    system,
                    task,
                    transport_config,
                    bridge,
                    datasets[cohort][regime],
                    cohort,
                    regime,
                    device,
                )
                for regime in REGIMES
            }
            for cohort in fixed.COHORT_SEEDS
        }
        mapping_record = prior["alignment_fit"]["mappings"]["fourier_m04"]
        mapping = {
            "linear": torch.tensor(
                mapping_record["linear"], dtype=torch.float64, device=device
            ),
            "intercept": torch.tensor(
                mapping_record["intercept"], dtype=torch.float64, device=device
            ),
        }
        prior_by_cell = {
            (cell["cohort"], cell["regime"]): cell
            for cell in prior["heldout_cells"]
        }
        heldout: list[dict[str, Any]] = []
        metric_tensors: list[dict[str, torch.Tensor]] = []
        replay_errors: list[float | None] = []
        array_payload: dict[str, np.ndarray] = {}
        for cohort in HELDOUT_COHORTS:
            for regime in REGIMES:
                cell = cells[cohort][regime]
                dataset = datasets[cohort][regime]
                theta = (
                    dataset.quotient_phase.to(device)
                    .reshape(config.orbit_count, 2)[:, 0]
                    .double()
                )
                features = predecessor.fourier_features(theta, config.fourier_order)
                predicted = transport.apply_affine(features, mapping)
                target = transport._coordinates(cell, basis)
                residual = target - predicted
                _, jacobian = continuation_moment_jacobian(
                    system,
                    cell["target_cut"],
                    cell["propagated"],
                    predicted,
                    basis,
                    task,
                    rank_config.continuation_batch_size,
                )
                decomposition = tangent_kernel_decomposition(
                    jacobian,
                    residual,
                    rtol=config.jacobian_rtol,
                    atol=config.jacobian_atol,
                )
                finite_error = finite_difference_error(
                    system,
                    task,
                    cell["target_cut"],
                    cell["propagated"],
                    predicted,
                    basis,
                    jacobian,
                    residual,
                    config.finite_difference_step,
                    rank_config.continuation_batch_size,
                )
                task_metric = normalized_task_metric(jacobian)
                metric_tensors.append(task_metric)
                tangent_random = norm_matched_random_controls(
                    decomposition["tangent"],
                    config.random_controls,
                    _random_seed(seed, cohort, regime, "tangent"),
                )
                kernel_random = norm_matched_random_controls(
                    decomposition["kernel"],
                    config.random_controls,
                    _random_seed(seed, cohort, regime, "kernel"),
                )
                coordinate_states: dict[str, torch.Tensor] = {
                    "predicted": predicted,
                    "tangent": predicted + decomposition["tangent"],
                    "kernel": predicted + decomposition["kernel"],
                    "full": target,
                }
                for index in range(config.random_controls):
                    coordinate_states[f"tangent_random_{index:02d}"] = (
                        predicted + tangent_random[index]
                    )
                    coordinate_states[f"kernel_random_{index:02d}"] = (
                        predicted + kernel_random[index]
                    )
                states = evaluate_states(
                    system,
                    task,
                    config,
                    cell,
                    basis,
                    coordinate_states,
                    readout_predecessor["rotation_bins"],
                )
                coordinate_record = transport.coordinate_metrics(predicted, target)
                prior_cell = prior_by_cell[(cohort, regime)]
                replay = max(
                    predecessor.predecessor._numeric_max_difference(
                        coordinate_record,
                        prior_cell["coordinate_metrics"]["fourier_m04"],
                    ),
                    predecessor.predecessor._numeric_max_difference(
                        states["predicted"]["continuous"],
                        prior_cell["states"]["fourier_m04"]["continuous"],
                    ),
                )
                replay_value = float(replay) if math.isfinite(replay) else None
                replay_errors.append(replay_value)
                prefix = f"{cohort}__{regime}"
                array_payload.update(
                    {
                        f"{prefix}__jacobian": jacobian.cpu().numpy(),
                        f"{prefix}__predicted_coordinates": predicted.cpu().numpy(),
                        f"{prefix}__target_coordinates": target.cpu().numpy(),
                        f"{prefix}__residual": residual.cpu().numpy(),
                        f"{prefix}__tangent": decomposition["tangent"].cpu().numpy(),
                        f"{prefix}__kernel": decomposition["kernel"].cpu().numpy(),
                    }
                )
                heldout.append(
                    {
                        "cohort": cohort,
                        "regime": regime,
                        "evaluation_seed": fixed.COHORT_SEEDS[cohort][regime],
                        "coordinate_metrics": coordinate_record,
                        "jacobian": _cell_summary(
                            jacobian, decomposition, finite_error, task_metric
                        ),
                        "predecessor_replay_maximum_absolute_error": replay_value,
                        "states": states,
                    }
                )

        stability = metric_stability(metric_tensors)
        predicted_pass = all(
            cell["states"]["predicted"]["continuous"]["continuous_pass"]
            for cell in heldout
        )
        full_pass = all(
            cell["states"]["full"]["continuous"]["continuous_pass"]
            for cell in heldout
        )
        tangent_pass = all(
            cell["states"]["tangent"]["continuous"]["continuous_pass"]
            for cell in heldout
        )
        kernel_inert = all(
            cell["states"]["kernel"]["movement_from_predicted"]["mean_bins"]
            <= config.kernel_mean_movement_ceiling_bins
            and cell["states"]["kernel"]["movement_from_predicted"]["p95_bins"]
            <= config.kernel_p95_movement_ceiling_bins
            for cell in heldout
        )
        tangent_random_means = [
            _aggregate_state_mean(heldout, f"tangent_random_{index:02d}")
            for index in range(config.random_controls)
        ]
        tangent_random_checkpoint_passes = [
            all(
                cell["states"][f"tangent_random_{index:02d}"]["continuous"][
                    "continuous_pass"
                ]
                for cell in heldout
            )
            for index in range(config.random_controls)
        ]
        tangent_mean = _aggregate_state_mean(heldout, "tangent")
        tangent_random_median = float(np.median(tangent_random_means))
        tangent_specificity = bool(
            sum(tangent_random_checkpoint_passes) <= 1
            and tangent_mean + config.random_advantage_bins <= tangent_random_median
        )
        kernel_random_movements = [
            _aggregate_movement(heldout, f"kernel_random_{index:02d}")
            for index in range(config.random_controls)
        ]
        kernel_movement = _aggregate_movement(heldout, "kernel")
        kernel_random_median = float(np.median(kernel_random_movements))
        kernel_specificity = bool(
            kernel_movement + config.kernel_random_advantage_bins
            <= kernel_random_median
        )
        metric_stable = bool(
            stability["maximum_normalized_frobenius_distance"]
            <= config.metric_frobenius_ceiling
            and stability["minimum_top_eigenvector_absolute_cosine"]
            >= config.metric_top_cosine_floor
        )
        predicted_mean = _aggregate_state_mean(heldout, "predicted")
        kernel_mean = _aggregate_state_mean(heldout, "kernel")
        kernel_causally_active = bool(
            not kernel_inert
            and (
                kernel_mean + config.random_advantage_bins <= predicted_mean
                or any(
                    cell["states"]["kernel"]["continuous"]["continuous_pass"]
                    for cell in heldout
                )
            )
        )
        maximum_fd = max(
            cell["jacobian"]["maximum_finite_difference_relative_error"]
            for cell in heldout
        )
        maximum_decomposition = max(
            max(
                cell["jacobian"]["maximum_relative_decomposition_error"],
                cell["jacobian"]["maximum_relative_tangent_mismatch"],
                cell["jacobian"]["maximum_relative_kernel_leakage"],
            )
            for cell in heldout
        )
        jacobian_contract = all(
            cell["jacobian"]["jacobian_finite"]
            and cell["jacobian"]["minimum_jacobian_rank"] >= 1
            for cell in heldout
        )
        maximum_replay = (
            max(value for value in replay_errors if value is not None)
            if replay_errors and all(value is not None for value in replay_errors)
            else None
        )
        replay_contract = bool(
            maximum_replay is not None
            and maximum_replay <= config.replay_tolerance
        )
        target_control_contract = bool(not predicted_pass and full_pass)
        finite_difference_contract = maximum_fd <= config.finite_difference_tolerance
        decomposition_contract = maximum_decomposition <= config.decomposition_tolerance
        valid = bool(
            replay_contract
            and target_control_contract
            and jacobian_contract
            and finite_difference_contract
            and decomposition_contract
        )
        classification = classify_checkpoint(
            valid=valid,
            tangent_pass=tangent_pass,
            kernel_inert=kernel_inert,
            tangent_specificity=tangent_specificity,
            kernel_specificity=kernel_specificity,
            metric_stable=metric_stable,
            kernel_causally_active=kernel_causally_active,
            full_pass=full_pass,
        )
        fingerprint = _fingerprint(
            config,
            seed,
            _sha256(prior_path),
            provenance["checkpoint_sha256"],
        )
        run_dir = output / "runs" / f"seed_{seed}"
        arrays_path = run_dir / "jacobian_arrays.npz"
        _write_npz(arrays_path, array_payload)
        result_path = run_dir / "result.json"
        result = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_id": f"tinyllm-c2-local-continuation-seed{seed}",
            "status": "completed",
            "evidence_role": _evidence_role(config),
            "completed_at": _utc_now(),
            "seed": seed,
            "configuration": asdict(config),
            "scientific_fingerprint": fingerprint,
            "implementation_sha256": implementation,
            "provenance": {
                "predecessor_campaign": str(predecessor_path),
                "predecessor_campaign_sha256": _sha256(predecessor_path),
                "predecessor_result": str(prior_path),
                "predecessor_result_sha256": _sha256(prior_path),
                "checkpoint": provenance["checkpoint"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "character_result": str(frozen_path),
                "character_result_sha256": _sha256(frozen_path),
                "readout_result": readout_predecessor["result"],
                "readout_result_sha256": readout_predecessor["result_sha256"],
            },
            "basis": basis_summary,
            "heldout_cells": heldout,
            "task_metric_stability": stability,
            "random_controls": {
                "tangent_aggregate_mean_shift_bins": tangent_random_means,
                "tangent_checkpoint_passes": tangent_random_checkpoint_passes,
                "tangent_median_mean_shift_bins": tangent_random_median,
                "kernel_aggregate_mean_movement_bins": kernel_random_movements,
                "kernel_median_mean_movement_bins": kernel_random_median,
            },
            "aggregates": {
                "predicted_aggregate_mean_shift_bins": predicted_mean,
                "tangent_aggregate_mean_shift_bins": tangent_mean,
                "kernel_aggregate_mean_shift_bins": kernel_mean,
                "kernel_aggregate_mean_movement_bins": kernel_movement,
                "predicted_checkpoint_pass": predicted_pass,
                "full_checkpoint_pass": full_pass,
                "tangent_checkpoint_pass": tangent_pass,
                "kernel_output_inert": kernel_inert,
                "tangent_random_specificity": tangent_specificity,
                "kernel_random_specificity": kernel_specificity,
                "task_metric_stable": metric_stable,
                "kernel_causally_active": kernel_causally_active,
            },
            "gates": {
                "predecessor_replay_contract": replay_contract,
                "maximum_predecessor_replay_error": maximum_replay,
                "target_control_contract": target_control_contract,
                "jacobian_numerical_contract": jacobian_contract,
                "finite_difference_contract": finite_difference_contract,
                "maximum_finite_difference_relative_error": maximum_fd,
                "decomposition_contract": decomposition_contract,
                "maximum_decomposition_relative_error": maximum_decomposition,
                "stable_local_task_metric_gate": bool(
                    valid
                    and tangent_pass
                    and kernel_inert
                    and tangent_specificity
                    and kernel_specificity
                    and metric_stable
                ),
            },
            "classification": classification,
            "primary_metric": float(
                classification == "stable_task_tangent_sufficient"
            ),
            "analysis_seconds": time.perf_counter() - seed_started,
            "artifacts": {
                "result": str(result_path),
                "jacobian_arrays": str(arrays_path),
                "jacobian_arrays_sha256": _sha256(arrays_path),
            },
            "method_boundaries": [
                "Exact held-out residuals are diagnostic and not deployable inputs.",
                "Jacobians are decoder-conditioned and local to off-manifold predicted states.",
                "The carrier basis and writer were selected in predecessor studies.",
                "Three selected checkpoints do not establish population prevalence.",
            ],
        }
        _write_json(result_path, result)
        results.append(result)
        print(f"seed {seed}: {classification}", flush=True)
        if _implementation_digest() != implementation:
            raise RuntimeError("implementation changed during campaign")

    classifications = [result["classification"] for result in results]
    valid_campaign = all(name != "invalid" for name in classifications)
    common = classifications[0] if len(set(classifications)) == 1 else None
    if config.allow_underpowered:
        conclusion = "systems_lifecycle_only_not_quality_evidence"
    elif valid_campaign and common is not None:
        conclusion = common
    elif valid_campaign:
        conclusion = "checkpoint_stratified_local_continuation_geometry"
    else:
        conclusion = "invalid_campaign"
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "cohort_seeds": fixed.COHORT_SEEDS,
        "implementation_sha256": implementation,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "provenance": {
            "predecessor_campaign": str(predecessor_path),
            "predecessor_campaign_sha256": _sha256(predecessor_path),
            "predecessor_implementation_sha256": predecessor_campaign[
                "implementation_sha256"
            ],
        },
        "summary": {
            "requested": len(config.seeds),
            "completed": len(results),
            "failed": 0,
            "excluded": 0,
            "reused": 0,
            "trained_models": 0,
            "fitted_writers": 0,
            "fitted_predictive_observers": 0,
            "jacobian_cells": len(results) * len(HELDOUT_COHORTS) * len(REGIMES),
        },
        "aggregates": {
            "conclusion": conclusion,
            "classification_counts": {
                name: classifications.count(name) for name in sorted(set(classifications))
            },
            "classification_by_seed": {
                str(result["seed"]): result["classification"] for result in results
            },
            "stable_gate_count": sum(
                result["gates"]["stable_local_task_metric_gate"] for result in results
            ),
            "gate_counts": {
                name: sum(bool(result["gates"][name]) for result in results)
                for name in (
                    "predecessor_replay_contract",
                    "target_control_contract",
                    "jacobian_numerical_contract",
                    "finite_difference_contract",
                    "decomposition_contract",
                )
            },
        },
        "results": [
            {
                "experiment_id": result["experiment_id"],
                "seed": result["seed"],
                "scientific_fingerprint": result["scientific_fingerprint"],
                "classification": result["classification"],
                "gates": result["gates"],
                "aggregates": result["aggregates"],
                "path": result["artifacts"]["result"],
                "result_sha256": _sha256(Path(result["artifacts"]["result"])),
                "jacobian_arrays": result["artifacts"]["jacobian_arrays"],
                "jacobian_arrays_sha256": result["artifacts"][
                    "jacobian_arrays_sha256"
                ],
            }
            for result in results
        ],
        "analysis_seconds": time.perf_counter() - campaign_started,
        "method_boundaries": [
            "This is a post-outcome diagnostic on three selected frozen checkpoints.",
            "Held-out exact residuals are used only to define causal patch components.",
            "First-order nullness need not imply finite-intervention causal nullness.",
            "Shakedown artifacts are systems-only and are never pooled.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "seed_*" / "result.json"),
            "arrays": str(output / "runs" / "seed_*" / "jacobian_arrays.npz"),
        },
    }
    _write_json(campaign_path, campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_local_continuation_tangent_kernel/"
            "20260807_d6_preregistered_diagnostic"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=_ints, default=fixed.PRIMARY_SEEDS)
    parser.add_argument("--orbit-count", type=int, default=64)
    parser.add_argument("--random-controls", type=int, default=8)
    parser.add_argument("--allow-underpowered", action="store_true")
    parser.add_argument("--post-outcome-corrective-replication", action="store_true")
    args = parser.parse_args()
    config = LocalContinuationConfig(
        seeds=args.seeds,
        orbit_count=args.orbit_count,
        random_controls=args.random_controls,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
        post_outcome_corrective_replication=(
            args.post_outcome_corrective_replication
        ),
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
