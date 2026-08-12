#!/usr/bin/env python3
"""Separate TinyLLM's scalar cosine quotient from full-posterior closure."""

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
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

import experiments.structure_net.tinyllm_final_query_task_kernel as posterior
from experiments.structure_net.tinyllm_predictive_circle import _resolve_device
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-final-query-semantic-kernel.v1"
HYPOTHESIS_ID = "tinyllm-final-query-semantic-kernel-v1"
EVIDENCE_ROLE = "preregistered_underpowered_frozen_semantic_posterior_separation"
REGIMES = posterior.REGIMES
DATASET_SEEDS = posterior.DATASET_SEEDS
RANDOM_PROJECTOR_SEEDS = {
    "training_support": 852_011,
    "outside_range": 852_021,
}
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-final-query-semantic-kernel-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "fcd7e6aedb6e7d53de111fb0ffedafda86f8702bb538ad7d06d60e754e3d7e59"
)
POSTERIOR_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_final_query_task_kernel.py"
)
POSTERIOR_RUNNER_SHA256 = (
    "52b59f96e53e76c1794e9d2d251760a76eb2300f8e64e70dbe165090a5899895"
)
POSTERIOR_CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_final_query_task_kernel/"
    "20260810_d8_seed7_preregistered/campaign_results.json"
)
POSTERIOR_CAMPAIGN_SHA256 = (
    "93d9e22d766aa56943f0bd0c41b31ed25dc592e97ae0667863f3963588c38cde"
)
POSTERIOR_DIAGNOSTICS_PATH = Path(
    "data/experiments/tinyllm_final_query_task_kernel/"
    "20260810_d8_seed7_preregistered/diagnostics.npz"
)
POSTERIOR_DIAGNOSTICS_SHA256 = (
    "ad1798b960375503381cb59725dfb84db9d823f488ab8073fea178d0399e0974"
)
CHECKPOINT_PATH = posterior.CHECKPOINT_PATH
CHECKPOINT_SHA256 = posterior.CHECKPOINT_SHA256
FINAL_CUT = posterior.FINAL_CUT


@dataclass(frozen=True)
class SemanticKernelConfig:
    regimes: tuple[str, ...] = REGIMES
    fiber_pairs: int = 512
    batch_size: int = 64
    jacobian_batch_pairs: int = 32
    jacobian_rtol: float = 1e-6
    jacobian_atol: float = 1e-10
    finite_difference_step: float = 1e-3
    finite_difference_pairs: int = 32
    finite_difference_relative_error_ceiling: float = 0.02
    decomposition_relative_error_ceiling: float = 1e-8
    kernel_leakage_ceiling: float = 1e-6
    nesting_leakage_ceiling: float = 1e-8
    replay_tolerance: float = 2e-6
    baseline_accuracy_floor: float = 0.15
    accuracy_loss_ceiling: float = 0.03
    cross_entropy_increase_ceiling: float = 0.05
    posterior_js_ceiling: float = 0.02
    semantic_mean_change_ceiling: float = 0.01
    semantic_p95_change_ceiling: float = 0.03
    task_map_score_loss_ceiling: float = 0.01
    root_mean_squared_error_increase_ceiling: float = 0.01
    kernel_contraction_ratio_ceiling: float = 0.25
    posterior_kernel_ratio_floor: float = 0.75
    material_full_effect_floor: float = 1e-5
    material_row_fraction_floor: float = 0.10
    task_effect_mean_residual_ceiling: float = 0.25
    task_effect_p95_residual_ceiling: float = 0.75
    task_effect_sign_agreement_floor: float = 0.90
    kernel_effect_mean_ratio_ceiling: float = 0.25
    kernel_effect_p95_ratio_ceiling: float = 0.75
    random_semantic_disadvantage_floor: float = 0.01
    device: str = "cuda:1"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.regimes or set(self.regimes).difference(REGIMES):
            raise ValueError(f"regimes must be drawn from {REGIMES}")
        if self.fiber_pairs < 2 or self.fiber_pairs % 2:
            raise ValueError("fiber_pairs must be positive and even")
        if self.batch_size < 4 or self.batch_size % 4:
            raise ValueError("batch_size must be divisible by four")
        if self.jacobian_batch_pairs < 1 or self.finite_difference_pairs < 1:
            raise ValueError("Jacobian batch and finite-difference counts must be positive")
        positive = (
            self.jacobian_rtol,
            self.jacobian_atol,
            self.finite_difference_step,
            self.finite_difference_relative_error_ceiling,
            self.decomposition_relative_error_ceiling,
            self.kernel_leakage_ceiling,
            self.nesting_leakage_ceiling,
            self.replay_tolerance,
            self.semantic_mean_change_ceiling,
            self.semantic_p95_change_ceiling,
            self.kernel_contraction_ratio_ceiling,
            self.posterior_kernel_ratio_floor,
            self.material_full_effect_floor,
            self.material_row_fraction_floor,
            self.random_semantic_disadvantage_floor,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("all numerical thresholds must be positive")
        if not self.allow_underpowered and (
            self.regimes != REGIMES
            or self.fiber_pairs != 512
            or self.batch_size != 64
            or self.jacobian_batch_pairs != 32
            or self.finite_difference_pairs != 32
        ):
            raise ValueError("primary campaign axes are frozen")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return posterior._sha256(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    posterior._write_json(path, value)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    posterior._write_npz(path, arrays)


def _finite(value: Any) -> bool:
    return posterior._finite(value)


def _json_config(config: SemanticKernelConfig) -> dict[str, Any]:
    value = asdict(config)
    value["regimes"] = list(config.regimes)
    return value


def _source_digests() -> dict[str, str]:
    files = {
        "runner": Path(__file__),
        "preregistration": PREREGISTRATION_PATH,
        "posterior_runner": POSTERIOR_RUNNER_PATH,
        "parent_runner": posterior.PARENT_RUNNER_PATH,
        "tinyllm_model": Path(
            "src/structure_net/components/models/tinyllm_model.py"
        ),
        "circle_source": Path(
            "experiments/structure_net/tinyllm_predictive_circle.py"
        ),
    }
    return {name: _sha256(path) for name, path in files.items()}


def _implementation_digest(source_digests: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(source_digests), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _posterior_source() -> dict[str, Any]:
    required = {
        PREREGISTRATION_PATH: PREREGISTRATION_SHA256,
        POSTERIOR_RUNNER_PATH: POSTERIOR_RUNNER_SHA256,
        POSTERIOR_CAMPAIGN_PATH: POSTERIOR_CAMPAIGN_SHA256,
        POSTERIOR_DIAGNOSTICS_PATH: POSTERIOR_DIAGNOSTICS_SHA256,
    }
    for path, expected in required.items():
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"source artifact changed: {path}")
    campaign = json.loads(POSTERIOR_CAMPAIGN_PATH.read_text(encoding="utf-8"))
    if (
        campaign.get("schema_version") != posterior.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != posterior.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("gates")
        != {
            "primary_hypothesis_pass": False,
            "state_unchanged": True,
            "validity": True,
        }
        or campaign.get("classification")
        != "task_rowspace_contains_material_fiber_chord"
        or set(campaign.get("regimes", {})) != set(REGIMES)
        or any(
            item.get("gates", {}).get("kernel_pair_contraction") is not False
            or item.get("gates", {}).get("kernel_task_sufficient") is not True
            for item in campaign["regimes"].values()
        )
    ):
        raise ValueError("posterior-kernel source contract changed")
    return campaign


def _validate_sources(
    config: SemanticKernelConfig,
) -> tuple[posterior.parent.CircleTaskConfig, dict[str, str], dict[str, Any], dict[str, Any]]:
    posterior_campaign = _posterior_source()
    parent_config = posterior.FinalQueryKernelConfig(
        regimes=config.regimes,
        fiber_pairs=config.fiber_pairs,
        batch_size=config.batch_size,
        jacobian_batch_pairs=config.jacobian_batch_pairs,
        finite_difference_pairs=config.finite_difference_pairs,
        device=config.device,
        allow_underpowered=True,
    )
    task, source_hashes, parent_detail = posterior._validate_parent(parent_config)
    hashes = dict(source_hashes)
    hashes.update(
        {
            "posterior_campaign": POSTERIOR_CAMPAIGN_SHA256,
            "posterior_diagnostics": POSTERIOR_DIAGNOSTICS_SHA256,
            "checkpoint_d8_cosine_interval": CHECKPOINT_SHA256,
        }
    )
    return task, hashes, parent_detail, posterior_campaign


def scalar_coordinate_double(
    model: TinyLLMModel,
    query: torch.Tensor,
    answer_ids: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the declared posterior-mean cosine coordinate in float64."""
    logits = posterior.centered_logits_double(model, query, answer_ids)
    return torch.softmax(logits, -1) @ centers.double()


def semantic_coordinate_jacobian(
    model: TinyLLMModel,
    query: torch.Tensor,
    answer_ids: torch.Tensor,
    centers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Ds plus the centered-logit Jacobian and barycenter posterior."""
    centered = posterior.centered_logits_double(model, query, answer_ids)
    probabilities = torch.softmax(centered, -1)
    logit_jacobian = posterior.centered_logit_jacobian(
        model, query, answer_ids
    )
    mean = probabilities @ centers.double()
    coefficients = probabilities * (centers.double()[None, :] - mean[:, None])
    scalar = torch.einsum("nk,nkd->nd", coefficients, logit_jacobian)
    return scalar[:, None, :], logit_jacobian, probabilities


def finite_difference_relative_error(
    model: TinyLLMModel,
    barycenter: torch.Tensor,
    direction: torch.Tensor,
    jacobian: torch.Tensor,
    answer_ids: torch.Tensor,
    centers: torch.Tensor,
    step: float,
) -> torch.Tensor:
    direction = direction.double()
    norms = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    fallback = torch.zeros_like(direction)
    fallback[:, 0] = 1.0
    direction = torch.where(
        norms > 1e-12, direction / norms.clamp_min(1e-12), fallback
    )
    plus = scalar_coordinate_double(
        model, barycenter.double() + step * direction, answer_ids, centers
    )
    minus = scalar_coordinate_double(
        model, barycenter.double() - step * direction, answer_ids, centers
    )
    estimate = (plus - minus) / (2.0 * step)
    predicted = torch.einsum(
        "nkd,nd->nk", jacobian.double(), direction
    )[:, 0]
    denominator = torch.maximum(
        torch.maximum(estimate.abs(), predicted.abs()),
        torch.full_like(estimate, 1e-10),
    )
    return (estimate - predicted).abs() / denominator


def _quantile(value: torch.Tensor, q: float) -> float:
    return float(torch.quantile(value.double(), q))


def scalar_preservation(
    baseline_coordinate: torch.Tensor,
    candidate_coordinate: torch.Tensor,
    baseline_metrics: Mapping[str, float],
    candidate_metrics: Mapping[str, float],
    config: SemanticKernelConfig,
) -> dict[str, Any]:
    change = (candidate_coordinate.double() - baseline_coordinate.double()).abs()
    result = {
        "mean_absolute_coordinate_change": float(change.mean()),
        "p95_absolute_coordinate_change": _quantile(change, 0.95),
        "task_map_score_loss": float(
            baseline_metrics["task_map_score"]
            - candidate_metrics["task_map_score"]
        ),
        "root_mean_squared_error_increase": float(
            candidate_metrics["root_mean_squared_error"]
            - baseline_metrics["root_mean_squared_error"]
        ),
    }
    result["pass"] = bool(
        result["mean_absolute_coordinate_change"]
        <= config.semantic_mean_change_ceiling
        and result["p95_absolute_coordinate_change"]
        <= config.semantic_p95_change_ceiling
        and result["task_map_score_loss"]
        <= config.task_map_score_loss_ceiling
        and result["root_mean_squared_error_increase"]
        <= config.root_mean_squared_error_increase_ceiling
    )
    return result


def scalar_effect_attribution(
    baseline: torch.Tensor,
    full: torch.Tensor,
    semantic_only: torch.Tensor,
    kernel_only: torch.Tensor,
    config: SemanticKernelConfig,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    full_change = full.double() - baseline.double()
    semantic_change = semantic_only.double() - baseline.double()
    kernel_change = kernel_only.double() - baseline.double()
    mask = full_change.abs() > config.material_full_effect_floor
    material_count = int(mask.sum())
    required_count = int(math.ceil(len(full_change) * config.material_row_fraction_floor))
    if material_count:
        denominator = full_change[mask].abs()
        relative_residual = (
            (full_change[mask] - semantic_change[mask]).abs() / denominator
        )
        sign_agreement = (
            torch.sign(full_change[mask]) == torch.sign(semantic_change[mask])
        ).double()
        kernel_ratio = kernel_change[mask].abs() / denominator
        mean_residual = float(relative_residual.mean())
        p95_residual = _quantile(relative_residual, 0.95)
        agreement = float(sign_agreement.mean())
        mean_kernel = float(kernel_ratio.mean())
        p95_kernel = _quantile(kernel_ratio, 0.95)
    else:
        relative_residual = torch.empty(0, dtype=torch.float64)
        sign_agreement = torch.empty(0, dtype=torch.float64)
        kernel_ratio = torch.empty(0, dtype=torch.float64)
        mean_residual = p95_residual = mean_kernel = p95_kernel = math.inf
        agreement = 0.0
    result = {
        "material_row_count": material_count,
        "required_material_row_count": required_count,
        "mean_relative_semantic_residual": mean_residual,
        "p95_relative_semantic_residual": p95_residual,
        "semantic_full_sign_agreement": agreement,
        "mean_kernel_full_effect_ratio": mean_kernel,
        "p95_kernel_full_effect_ratio": p95_kernel,
    }
    result["pass"] = bool(
        material_count >= required_count
        and mean_residual <= config.task_effect_mean_residual_ceiling
        and p95_residual <= config.task_effect_p95_residual_ceiling
        and agreement >= config.task_effect_sign_agreement_floor
        and mean_kernel <= config.kernel_effect_mean_ratio_ceiling
        and p95_kernel <= config.kernel_effect_p95_ratio_ceiling
    )
    return result, {
        "relative_semantic_residual": relative_residual,
        "semantic_full_sign_agreement": sign_agreement,
        "kernel_full_effect_ratio": kernel_ratio,
    }


def _stored_arrays(regime: str) -> dict[str, torch.Tensor]:
    parent_baseline, parent_full = posterior._parent_arrays(regime)
    with np.load(POSTERIOR_DIAGNOSTICS_PATH) as arrays:
        return {
            "parent_baseline": parent_baseline,
            "parent_full": parent_full,
            "posterior_baseline": torch.from_numpy(
                arrays[f"{regime}__baseline_posterior"]
            ).double(),
            "posterior_full": torch.from_numpy(
                arrays[f"{regime}__full_posterior"]
            ).double(),
            "posterior_kernel": torch.from_numpy(
                arrays[f"{regime}__kernel_only_posterior"]
            ).double(),
            "posterior_original_pair_norm": torch.from_numpy(
                arrays[f"{regime}__original_pair_norm"]
            ).double(),
            "posterior_kernel_pair_norm": torch.from_numpy(
                arrays[f"{regime}__kernel_pair_norm"]
            ).double(),
        }


def analyze_regime(
    model: TinyLLMModel,
    task: posterior.parent.CircleTaskConfig,
    regime: str,
    config: SemanticKernelConfig,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cohort = posterior.parent.generate_exact_task_fibers(
        task,
        pair_count=config.fiber_pairs,
        seed=DATASET_SEEDS[regime],
        regime=regime,
    )
    if cohort.contract.get("pass") is not True:
        raise ValueError(f"cohort contract failed for {regime}")
    answer_ids = torch.tensor(
        task.answer_token_ids, dtype=torch.long, device=device
    )
    centers = torch.linspace(
        -1.0, 1.0, task.phase_bins, dtype=torch.float64, device=device
    )
    query, baseline_posterior, maximum_capture_replay_error = (
        posterior._capture_final_queries(
            model, cohort, answer_ids, config.batch_size
        )
    )
    shaped = query.reshape(config.fiber_pairs, 2, -1)
    barycenter_unique = shaped.mean(1).double()
    full_query = barycenter_unique.repeat_interleave(2, dim=0)
    displacement = full_query - query.double()
    pair_direction = shaped[:, 0].double() - shaped[:, 1].double()

    semantic_parts: list[torch.Tensor] = []
    kernel_parts: list[torch.Tensor] = []
    rank_parts: list[torch.Tensor] = []
    posterior_rank_parts: list[torch.Tensor] = []
    reconstruction_parts: list[torch.Tensor] = []
    leakage_parts: list[torch.Tensor] = []
    nesting_parts: list[torch.Tensor] = []
    finite_difference_parts: list[torch.Tensor] = []
    scalar_singular_parts: list[torch.Tensor] = []
    remaining_fd = min(config.finite_difference_pairs, config.fiber_pairs)
    for start in range(0, config.fiber_pairs, config.jacobian_batch_pairs):
        stop = min(config.fiber_pairs, start + config.jacobian_batch_pairs)
        barycenter = barycenter_unique[start:stop]
        scalar_jacobian, logit_jacobian, _ = semantic_coordinate_jacobian(
            model, barycenter, answer_ids, centers
        )
        row_displacement = displacement[2 * start : 2 * stop]
        repeated_scalar = scalar_jacobian.repeat_interleave(2, dim=0)
        parts = posterior.decompose_displacement(
            repeated_scalar,
            row_displacement,
            rtol=config.jacobian_rtol,
            atol=config.jacobian_atol,
        )
        semantic_parts.append(parts["task"])
        kernel_parts.append(parts["kernel"])
        rank_parts.append(parts["rank"].reshape(stop - start, 2)[:, 0])
        scalar_singular_parts.append(
            parts["singular_values"].reshape(stop - start, 2, -1)[:, 0]
        )
        reconstruction_parts.append(parts["reconstruction_error"])
        leakage_parts.append(parts["kernel_leakage"])

        posterior_parts = posterior.decompose_displacement(
            logit_jacobian,
            scalar_jacobian[:, 0],
            rtol=config.jacobian_rtol,
            atol=config.jacobian_atol,
        )
        posterior_rank_parts.append(posterior_parts["rank"])
        nesting_parts.append(
            torch.linalg.vector_norm(posterior_parts["kernel"], dim=1)
            / torch.linalg.vector_norm(scalar_jacobian[:, 0], dim=1).clamp_min(
                1e-12
            )
        )
        if remaining_fd:
            count = min(remaining_fd, stop - start)
            finite_difference_parts.append(
                finite_difference_relative_error(
                    model,
                    barycenter[:count],
                    pair_direction[start : start + count],
                    scalar_jacobian[:count],
                    answer_ids,
                    centers,
                    config.finite_difference_step,
                )
            )
            remaining_fd -= count

    semantic_displacement = torch.cat(semantic_parts)
    kernel_displacement = torch.cat(kernel_parts)
    ranks = torch.cat(rank_parts)
    posterior_ranks = torch.cat(posterior_rank_parts)
    scalar_singular_values = torch.cat(scalar_singular_parts)
    reconstruction_error = torch.cat(reconstruction_parts)
    kernel_leakage = torch.cat(leakage_parts)
    nesting_leakage = torch.cat(nesting_parts)
    finite_difference_error = torch.cat(finite_difference_parts)
    random_basis = posterior.random_rowspace_basis(
        query.shape[-1],
        1,
        RANDOM_PROJECTOR_SEEDS[regime],
        device=device,
    )
    random_kernel = posterior.random_kernel_displacement(
        displacement, random_basis
    )

    arm_queries = {
        "full": full_query.to(query.dtype),
        "semantic_only": (
            query.double() + semantic_displacement
        ).to(query.dtype),
        "semantic_kernel": (
            query.double() + kernel_displacement
        ).to(query.dtype),
        "random_kernel": (query.double() + random_kernel).to(query.dtype),
    }
    baseline_direct, _ = posterior._final_outputs(model, query, answer_ids)
    arm_posteriors: dict[str, torch.Tensor] = {}
    for name, state in arm_queries.items():
        arm_posteriors[name] = posterior._final_outputs(
            model, state, answer_ids
        )[0].detach().cpu()
    baseline_posterior_cpu = baseline_posterior.detach().cpu()
    baseline_direct_cpu = baseline_direct.detach().cpu()
    direct_replay_error = float(
        torch.max(torch.abs(baseline_direct_cpu - baseline_posterior_cpu))
    )

    stored = _stored_arrays(regime)
    parent_replay_required = bool(
        not config.allow_underpowered
        and config.fiber_pairs == 512
        and set(config.regimes) == set(REGIMES)
    )
    replay_errors = {
        "parent_baseline": 0.0,
        "parent_full": 0.0,
        "posterior_baseline": 0.0,
        "posterior_full": 0.0,
    }
    if parent_replay_required:
        replay_errors = {
            "parent_baseline": float(
                torch.max(
                    torch.abs(
                        stored["parent_baseline"] - baseline_posterior_cpu
                    )
                )
            ),
            "parent_full": float(
                torch.max(
                    torch.abs(stored["parent_full"] - arm_posteriors["full"])
                )
            ),
            "posterior_baseline": float(
                torch.max(
                    torch.abs(
                        stored["posterior_baseline"] - baseline_posterior_cpu
                    )
                )
            ),
            "posterior_full": float(
                torch.max(
                    torch.abs(
                        stored["posterior_full"] - arm_posteriors["full"]
                    )
                )
            ),
        }

    targets = cohort.cosine_targets
    baseline_metrics = posterior.parent._posterior_metrics(
        baseline_posterior_cpu,
        targets,
        "cosine_interval",
        task,
        cohort.future_phase,
        cohort.cosine,
    )
    arms = {
        name: posterior.parent._candidate_record(
            value,
            baseline_posterior_cpu,
            targets,
            "cosine_interval",
            task,
            cohort,
            config,
        )
        for name, value in arm_posteriors.items()
    }
    centers_cpu = centers.cpu()
    baseline_coordinate = baseline_posterior_cpu @ centers_cpu
    arm_coordinates = {
        name: value @ centers_cpu for name, value in arm_posteriors.items()
    }
    for name in arms:
        arms[name]["scalar_preservation"] = scalar_preservation(
            baseline_coordinate,
            arm_coordinates[name],
            baseline_metrics,
            arms[name]["metrics"],
            config,
        )

    original_pair_norm = torch.linalg.vector_norm(pair_direction, dim=1)
    kernel_query = arm_queries["semantic_kernel"].reshape(
        config.fiber_pairs, 2, -1
    ).double()
    kernel_pair_norm = torch.linalg.vector_norm(
        kernel_query[:, 0] - kernel_query[:, 1], dim=1
    )
    contraction_ratio = float(
        kernel_pair_norm.mean() / original_pair_norm.mean().clamp_min(1e-12)
    )
    posterior_kernel_ratio = float(
        stored["posterior_kernel_pair_norm"].mean()
        / stored["posterior_original_pair_norm"].mean().clamp_min(1e-12)
    )
    attribution, attribution_arrays = scalar_effect_attribution(
        baseline_coordinate,
        arm_coordinates["full"],
        arm_coordinates["semantic_only"],
        arm_coordinates["semantic_kernel"],
        config,
    )

    numerical_contract = bool(
        bool((ranks == 1).all())
        and bool((posterior_ranks == 15).all())
        and float(reconstruction_error.max())
        <= config.decomposition_relative_error_ceiling
        and float(kernel_leakage.max()) <= config.kernel_leakage_ceiling
        and float(nesting_leakage.max()) <= config.nesting_leakage_ceiling
        and float(finite_difference_error.max())
        <= config.finite_difference_relative_error_ceiling
    )
    replay_pass = bool(
        maximum_capture_replay_error <= config.replay_tolerance
        and direct_replay_error <= config.replay_tolerance
        and (
            not parent_replay_required
            or all(
                value <= config.replay_tolerance
                for value in replay_errors.values()
            )
        )
    )
    baseline_validity = bool(
        baseline_metrics["exact_bin_accuracy"] >= config.baseline_accuracy_floor
    )
    scalar_preserved = bool(
        arms["semantic_kernel"]["scalar_preservation"]["pass"]
    )
    contraction_pass = bool(
        contraction_ratio <= config.kernel_contraction_ratio_ceiling
    )
    posterior_ratio_pass = bool(
        posterior_kernel_ratio >= config.posterior_kernel_ratio_floor
    )
    posterior_separation = bool(
        arms["semantic_kernel"]["gates"]["mean_posterior_js"]
        > config.posterior_js_ceiling
    )
    semantic_only_active = bool(
        not arms["semantic_only"]["scalar_preservation"]["pass"]
    )
    random_specificity = bool(
        not arms["random_kernel"]["scalar_preservation"]["pass"]
        and arms["random_kernel"]["scalar_preservation"][
            "mean_absolute_coordinate_change"
        ]
        >= arms["semantic_kernel"]["scalar_preservation"][
            "mean_absolute_coordinate_change"
        ]
        + config.random_semantic_disadvantage_floor
    )
    finite_contract = bool(
        torch.isfinite(query).all()
        and torch.isfinite(semantic_displacement).all()
        and torch.isfinite(kernel_displacement).all()
        and torch.isfinite(random_kernel).all()
        and all(torch.isfinite(value).all() for value in arm_posteriors.values())
        and all(torch.isfinite(value).all() for value in arm_coordinates.values())
    )
    valid = bool(
        cohort.contract.get("pass") is True
        and numerical_contract
        and replay_pass
        and baseline_validity
        and finite_contract
    )
    primary_regime_gate = bool(
        valid
        and scalar_preserved
        and contraction_pass
        and posterior_ratio_pass
        and posterior_separation
        and attribution["pass"]
        and semantic_only_active
        and random_specificity
    )
    record = {
        "regime": regime,
        "cohort_contract": dict(cohort.contract),
        "baseline": baseline_metrics,
        "arms": arms,
        "geometry": {
            "mean_original_pair_norm": float(original_pair_norm.mean()),
            "mean_semantic_kernel_pair_norm": float(kernel_pair_norm.mean()),
            "semantic_kernel_pair_contraction_ratio": contraction_ratio,
            "posterior_kernel_pair_contraction_ratio": posterior_kernel_ratio,
            "mean_semantic_displacement_norm_fraction": float(
                (
                    torch.linalg.vector_norm(semantic_displacement, dim=1)
                    / torch.linalg.vector_norm(displacement, dim=1).clamp_min(
                        1e-12
                    )
                ).mean()
            ),
            "mean_kernel_displacement_norm_fraction": float(
                (
                    torch.linalg.vector_norm(kernel_displacement, dim=1)
                    / torch.linalg.vector_norm(displacement, dim=1).clamp_min(
                        1e-12
                    )
                ).mean()
            ),
        },
        "attribution": attribution,
        "jacobian": {
            "scalar_rank_counts": {
                str(rank): int((ranks == rank).sum())
                for rank in sorted(set(ranks.detach().cpu().tolist()))
            },
            "minimum_scalar_rank": int(ranks.min()),
            "maximum_scalar_rank": int(ranks.max()),
            "posterior_rank_counts": {
                str(rank): int((posterior_ranks == rank).sum())
                for rank in sorted(
                    set(posterior_ranks.detach().cpu().tolist())
                )
            },
            "minimum_posterior_rank": int(posterior_ranks.min()),
            "maximum_posterior_rank": int(posterior_ranks.max()),
            "random_control_rank": 1,
            "minimum_scalar_singular_value": float(
                scalar_singular_values.min()
            ),
            "maximum_reconstruction_relative_error": float(
                reconstruction_error.max()
            ),
            "maximum_kernel_leakage": float(kernel_leakage.max()),
            "maximum_posterior_rowspace_nesting_leakage": float(
                nesting_leakage.max()
            ),
            "maximum_finite_difference_relative_error": float(
                finite_difference_error.max()
            ),
        },
        "replay": {
            "maximum_capture_replay_error": maximum_capture_replay_error,
            "maximum_direct_replay_error": direct_replay_error,
            "source_replay_required": parent_replay_required,
            "maximum_parent_baseline_error": replay_errors["parent_baseline"],
            "maximum_parent_full_error": replay_errors["parent_full"],
            "maximum_posterior_baseline_error": replay_errors[
                "posterior_baseline"
            ],
            "maximum_posterior_full_error": replay_errors["posterior_full"],
        },
        "gates": {
            "baseline_validity": baseline_validity,
            "cohort_contract": cohort.contract.get("pass") is True,
            "finite": finite_contract,
            "jacobian_numerical_and_nesting_contract": numerical_contract,
            "replay": replay_pass,
            "semantic_kernel_scalar_preservation": scalar_preserved,
            "semantic_kernel_pair_contraction": contraction_pass,
            "posterior_kernel_noncontraction_replay": posterior_ratio_pass,
            "complete_posterior_separation": posterior_separation,
            "semantic_effect_attribution": attribution["pass"],
            "semantic_only_causally_active": semantic_only_active,
            "random_projector_specificity": random_specificity,
            "validity": valid,
            "primary_regime_gate": primary_regime_gate,
        },
    }
    arrays: dict[str, np.ndarray] = {
        "baseline_posterior": baseline_posterior_cpu.numpy(),
        "full_posterior": arm_posteriors["full"].numpy(),
        "semantic_only_posterior": arm_posteriors["semantic_only"].numpy(),
        "semantic_kernel_posterior": arm_posteriors["semantic_kernel"].numpy(),
        "random_kernel_posterior": arm_posteriors["random_kernel"].numpy(),
        "baseline_coordinate": baseline_coordinate.numpy(),
        "full_coordinate": arm_coordinates["full"].numpy(),
        "semantic_only_coordinate": arm_coordinates["semantic_only"].numpy(),
        "semantic_kernel_coordinate": arm_coordinates[
            "semantic_kernel"
        ].numpy(),
        "random_kernel_coordinate": arm_coordinates["random_kernel"].numpy(),
        "scalar_jacobian_rank": ranks.detach().cpu().numpy(),
        "posterior_jacobian_rank": posterior_ranks.detach().cpu().numpy(),
        "scalar_jacobian_singular_value": scalar_singular_values.detach()
        .cpu()
        .numpy(),
        "reconstruction_relative_error": reconstruction_error.detach()
        .cpu()
        .numpy(),
        "kernel_leakage": kernel_leakage.detach().cpu().numpy(),
        "posterior_rowspace_nesting_leakage": nesting_leakage.detach()
        .cpu()
        .numpy(),
        "finite_difference_relative_error": finite_difference_error.detach()
        .cpu()
        .numpy(),
        "original_pair_norm": original_pair_norm.detach().cpu().numpy(),
        "semantic_kernel_pair_norm": kernel_pair_norm.detach().cpu().numpy(),
        "semantic_displacement_norm": torch.linalg.vector_norm(
            semantic_displacement, dim=1
        )
        .detach()
        .cpu()
        .numpy(),
        "kernel_displacement_norm": torch.linalg.vector_norm(
            kernel_displacement, dim=1
        )
        .detach()
        .cpu()
        .numpy(),
    }
    arrays.update(
        {
            name: value.detach().cpu().numpy()
            for name, value in attribution_arrays.items()
        }
    )
    return record, arrays


def _fingerprint(
    config: SemanticKernelConfig,
    implementation: str,
    source_hashes: Mapping[str, str],
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": _json_config(config),
        "implementation_sha256": implementation,
        "source_hashes": dict(source_hashes),
        "dataset_seeds": DATASET_SEEDS,
        "random_projector_seeds": RANDOM_PROJECTOR_SEEDS,
        "final_cut": FINAL_CUT,
        "semantic_centers": np.linspace(-1.0, 1.0, 16).tolist(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _classification(
    config: SemanticKernelConfig,
    valid: bool,
    primary: bool,
    regimes: Mapping[str, Mapping[str, Any]],
) -> str:
    if config.allow_underpowered:
        return "systems_lifecycle_only_not_quality_evidence"
    if not valid:
        return "invalid_final_query_semantic_kernel_campaign"
    if primary:
        return "scalar_semantic_quotient_separated_from_complete_posterior"
    if all(
        item["gates"]["semantic_kernel_scalar_preservation"]
        and not item["gates"]["semantic_kernel_pair_contraction"]
        for item in regimes.values()
    ):
        return "scalar_kernel_preserves_but_does_not_contain_fiber_chord"
    if any(
        item["gates"]["semantic_kernel_pair_contraction"]
        and not item["gates"]["semantic_kernel_scalar_preservation"]
        for item in regimes.values()
    ):
        return "finite_scalar_kernel_curvature_breaks_preservation"
    if any(
        not item["gates"]["complete_posterior_separation"]
        for item in regimes.values()
    ):
        return "scalar_and_complete_posterior_quotients_not_separated"
    if any(
        not item["gates"]["random_projector_specificity"]
        for item in regimes.values()
    ):
        return "scalar_kernel_lacks_random_specificity"
    if any(
        not item["gates"]["semantic_effect_attribution"]
        for item in regimes.values()
    ):
        return "scalar_tangent_does_not_explain_full_effect"
    return "final_query_semantic_kernel_hypothesis_not_supported"


def run_campaign(config: SemanticKernelConfig, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    source_digests = _source_digests()
    if source_digests["preregistration"] != PREREGISTRATION_SHA256:
        raise ValueError("preregistration changed")
    if source_digests["posterior_runner"] != POSTERIOR_RUNNER_SHA256:
        raise ValueError("posterior-kernel runner changed")
    if source_digests["parent_runner"] != posterior.PARENT_RUNNER_SHA256:
        raise ValueError("parent runner changed")
    implementation = _implementation_digest(source_digests)
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "completed"
            and existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("configuration") == _json_config(config)
            and existing.get("implementation_sha256") == implementation
            and existing.get("scientific_fingerprint")
            == _fingerprint(
                config, implementation, existing.get("source_hashes", {})
            )
            and Path(existing["artifacts"]["diagnostics"]).is_file()
            and _sha256(Path(existing["artifacts"]["diagnostics"]))
            == existing["artifacts"]["diagnostics_sha256"]
        ):
            print(
                "campaign already complete; leaving bytes unchanged", flush=True
            )
            return existing
        raise ValueError(f"incompatible completed campaign {campaign_path}")

    task, source_hashes, parent_detail, posterior_campaign = _validate_sources(
        config
    )
    fingerprint = _fingerprint(config, implementation, source_hashes)
    device = _resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model, initial_system_state = posterior.parent._load_model(
        CHECKPOINT_PATH, "d8", "cosine_interval", device
    )
    initial_model_state = posterior.parent._state_digest(model)
    regime_results: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for regime in config.regimes:
        result, regime_arrays = analyze_regime(
            model, task, regime, config, device
        )
        regime_results[regime] = result
        arrays.update(
            {
                f"{regime}__{name}": value
                for name, value in regime_arrays.items()
            }
        )
    final_model_state = posterior.parent._state_digest(model)
    final_system_state = posterior.parent._model_system_state(model)
    state_unchanged = bool(
        initial_model_state == final_model_state
        and initial_system_state == final_system_state
    )
    valid = bool(
        state_unchanged
        and all(item["gates"]["validity"] for item in regime_results.values())
    )
    primary = bool(
        not config.allow_underpowered
        and set(config.regimes) == set(REGIMES)
        and config.fiber_pairs == 512
        and valid
        and all(
            item["gates"]["primary_regime_gate"]
            for item in regime_results.values()
        )
    )
    classification = _classification(config, valid, primary, regime_results)
    diagnostics_path = output / "diagnostics.npz"
    _write_npz(diagnostics_path, arrays)
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "completed_at": _utc_now(),
        "configuration": _json_config(config),
        "scientific_fingerprint": fingerprint,
        "implementation_sha256": implementation,
        "source_digests": source_digests,
        "source_hashes": source_hashes,
        "task_config": asdict(task),
        "parent_summary": {
            "task_relative_classification": (
                "observational_task_geometry_not_causally_sufficient"
            ),
            "task_relative_valid_d8_null": parent_detail["gates"]["validity"],
            "posterior_kernel_classification": posterior_campaign[
                "classification"
            ],
            "posterior_kernel_valid": posterior_campaign["gates"]["validity"],
            "posterior_kernel_primary_pass": posterior_campaign["gates"][
                "primary_hypothesis_pass"
            ],
        },
        "regimes": regime_results,
        "state_record": {
            "initial_model_state_sha256": initial_model_state,
            "final_model_state_sha256": final_model_state,
            "initial_system_state_sha256": initial_system_state,
            "final_system_state_sha256": final_system_state,
            "unchanged": state_unchanged,
        },
        "gates": {
            "state_unchanged": state_unchanged,
            "validity": valid,
            "primary_hypothesis_pass": primary,
        },
        "classification": classification,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu"
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "summary": {
            "requested": 1,
            "scheduled": 1,
            "completed": 1,
            "failed": 0,
            "excluded": 0,
            "retries": 0,
            "reused": 0,
            "trained_models": 0,
            "trained_probes": 0,
            "fitted_carriers": 0,
            "fitted_decoders": 0,
            "jacobian_cells": len(config.regimes) * config.fiber_pairs,
        },
        "artifacts": {
            "campaign": str(campaign_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": _sha256(diagnostics_path),
        },
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "One retained d8 seed-7 cosine checkpoint is evaluated; population prevalence is not tested.",
            "The declared scalar is exactly the frozen posterior mean over fixed cosine answer centers.",
            "Scalar-coordinate preservation and complete-posterior closure are separate preregistered endpoints.",
            "Only the final post-MLP query vector is decomposed; no earlier-layer scan is licensed.",
            "The analytic activation-local Jacobian does not establish a global invariant subspace.",
            "No model parameter, probe, decoder, carrier, metric, or threshold is fit.",
        ],
    }
    if not _finite(campaign):
        raise ValueError("non-finite final-query semantic-kernel campaign")
    _write_json(campaign_path, campaign)
    return campaign


def _strings(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_final_query_semantic_kernel/"
            "20260810_d8_seed7_preregistered"
        ),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--regimes", type=_strings, default=REGIMES)
    parser.add_argument("--fiber-pairs", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--jacobian-batch-pairs", type=int, default=32)
    parser.add_argument("--finite-difference-pairs", type=int, default=32)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args(argv)
    config = SemanticKernelConfig(
        regimes=args.regimes,
        fiber_pairs=args.fiber_pairs,
        batch_size=args.batch_size,
        jacobian_batch_pairs=args.jacobian_batch_pairs,
        finite_difference_pairs=args.finite_difference_pairs,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(
        json.dumps(
            {
                "classification": campaign["classification"],
                "valid": campaign["gates"]["validity"],
                "primary_hypothesis_pass": campaign["gates"][
                    "primary_hypothesis_pass"
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
