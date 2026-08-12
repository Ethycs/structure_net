#!/usr/bin/env python3
"""Decompose a final-query barycenter chord into task tangent and kernel."""

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

import experiments.structure_net.tinyllm_task_relative_activation_barycenter as parent
from experiments.structure_net.tinyllm_predictive_circle import _resolve_device
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-final-query-task-kernel-barycenter.v1"
HYPOTHESIS_ID = "tinyllm-final-query-task-kernel-barycenter-v1"
EVIDENCE_ROLE = "preregistered_underpowered_frozen_final_query_decomposition"
REGIMES = ("training_support", "outside_range")
DATASET_SEEDS = {"training_support": 850_011, "outside_range": 850_021}
RANDOM_PROJECTOR_SEEDS = {
    "training_support": 851_011,
    "outside_range": 851_021,
}
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-final-query-task-kernel-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "6c99f8b4e6feb91fe5ca90a6846cc2b1472825874cd201d3fb5f0f51504347c3"
)
PARENT_CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_task_relative_activation_barycenter/"
    "20260810_seed7_preregistered/campaign_results.json"
)
PARENT_CAMPAIGN_SHA256 = (
    "b5d6b613683326024eeb00944a3d0aba4dd7a251dd22f5fa7e8561b5e9d6aae4"
)
PARENT_D8_RESULT_PATH = Path(
    "data/experiments/tinyllm_task_relative_activation_barycenter/"
    "20260810_seed7_preregistered/runs/d8/result.json"
)
PARENT_D8_RESULT_SHA256 = (
    "79119a4fa6df16a2fe2d65c34ec1aca30072a0b53d3d8ea312c92b29f86d1476"
)
PARENT_D8_DIAGNOSTICS_PATH = Path(
    "data/experiments/tinyllm_task_relative_activation_barycenter/"
    "20260810_seed7_preregistered/runs/d8/diagnostics.npz"
)
PARENT_D8_DIAGNOSTICS_SHA256 = (
    "ef86c31d418c04b7ffd4bec3b66135057c4ca825b3e4da8263a920cbce0de1dd"
)
PARENT_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_task_relative_activation_barycenter.py"
)
PARENT_RUNNER_SHA256 = (
    "67e0854162cfd5ef4ff7dbc18f2cd2b7e9c6e88e10543e30bf0baf6b87b1c8ee"
)
CHECKPOINT_PATH = parent.CHECKPOINTS[("d8", "cosine_interval")]
CHECKPOINT_SHA256 = parent.CHECKPOINT_SHA256[("d8", "cosine_interval")]
FINAL_CUT = "block_8_post_mlp"


@dataclass(frozen=True)
class FinalQueryKernelConfig:
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
    accuracy_loss_ceiling: float = 0.03
    cross_entropy_increase_ceiling: float = 0.05
    posterior_js_ceiling: float = 0.02
    baseline_accuracy_floor: float = 0.15
    replay_tolerance: float = 2e-6
    kernel_contraction_ratio_ceiling: float = 0.25
    task_effect_mean_residual_ceiling: float = 0.25
    task_effect_p95_residual_ceiling: float = 0.75
    task_effect_median_cosine_floor: float = 0.90
    kernel_effect_mean_ratio_ceiling: float = 0.25
    kernel_effect_p95_ratio_ceiling: float = 0.75
    random_js_disadvantage_floor: float = 0.01
    device: str = "cuda:1"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.regimes or set(self.regimes).difference(REGIMES):
            raise ValueError(f"regimes must be drawn from {REGIMES}")
        if self.fiber_pairs < 2 or self.fiber_pairs % 2:
            raise ValueError("fiber_pairs must be positive and even")
        if self.batch_size < 4 or self.batch_size % 4:
            raise ValueError("batch_size must be divisible by four")
        if self.jacobian_batch_pairs < 1:
            raise ValueError("jacobian_batch_pairs must be positive")
        if self.finite_difference_pairs < 1:
            raise ValueError("finite_difference_pairs must be positive")
        positive = (
            self.jacobian_rtol,
            self.jacobian_atol,
            self.finite_difference_step,
            self.finite_difference_relative_error_ceiling,
            self.decomposition_relative_error_ceiling,
            self.kernel_leakage_ceiling,
            self.replay_tolerance,
            self.kernel_contraction_ratio_ceiling,
            self.random_js_disadvantage_floor,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("numerical thresholds must be positive")
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _json_config(config: FinalQueryKernelConfig) -> dict[str, Any]:
    value = asdict(config)
    value["regimes"] = list(config.regimes)
    return value


def _source_digests() -> dict[str, str]:
    files = {
        "runner": Path(__file__),
        "preregistration": PREREGISTRATION_PATH,
        "parent_runner": PARENT_RUNNER_PATH,
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


def _validate_parent(
    config: FinalQueryKernelConfig,
) -> tuple[
    parent.CircleTaskConfig,
    dict[str, str],
    dict[str, Any],
]:
    required = {
        PREREGISTRATION_PATH: PREREGISTRATION_SHA256,
        PARENT_CAMPAIGN_PATH: PARENT_CAMPAIGN_SHA256,
        PARENT_D8_RESULT_PATH: PARENT_D8_RESULT_SHA256,
        PARENT_D8_DIAGNOSTICS_PATH: PARENT_D8_DIAGNOSTICS_SHA256,
        PARENT_RUNNER_PATH: PARENT_RUNNER_SHA256,
        CHECKPOINT_PATH: CHECKPOINT_SHA256,
    }
    for path, expected in required.items():
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"source artifact changed: {path}")
    campaign = json.loads(PARENT_CAMPAIGN_PATH.read_text(encoding="utf-8"))
    detail = json.loads(PARENT_D8_RESULT_PATH.read_text(encoding="utf-8"))
    entry = next(
        (item for item in campaign.get("results", []) if item.get("preset") == "d8"),
        None,
    )
    if (
        campaign.get("schema_version") != parent.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != parent.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("aggregates", {}).get("primary_hypothesis_pass") is not False
        or entry is None
        or entry.get("result_sha256") != PARENT_D8_RESULT_SHA256
        or entry.get("diagnostics_sha256") != PARENT_D8_DIAGNOSTICS_SHA256
        or detail.get("preset") != "d8"
        or detail.get("seed") != 7
        or detail.get("gates", {}).get("validity") is not True
        or detail.get("gates", {}).get("mature_causal_front_exists") is not False
        or detail.get("summary", {}).get("mature_causal_front") is not None
        or detail.get("summary", {}).get("first_isolated_passing_cut") is not None
    ):
        raise ValueError("parent d8 causal-null contract changed")
    parent_config = parent.TaskBarycenterConfig(
        presets=("d8",),
        regimes=config.regimes,
        fiber_pairs=config.fiber_pairs,
        batch_size=config.batch_size,
        device=config.device,
        allow_underpowered=True,
    )
    task, _, source_hashes = parent._validate_sources(parent_config)
    hashes = dict(source_hashes)
    hashes.update(
        {
            "parent_campaign": PARENT_CAMPAIGN_SHA256,
            "parent_d8_result": PARENT_D8_RESULT_SHA256,
            "parent_d8_diagnostics": PARENT_D8_DIAGNOSTICS_SHA256,
            "checkpoint_d8_cosine_interval": CHECKPOINT_SHA256,
        }
    )
    return task, hashes, detail


def centered_logits_double(
    model: TinyLLMModel,
    query: torch.Tensor,
    answer_ids: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the mathematical final layer norm/head map in float64."""
    layer_norm = model.transformer["ln_f"]
    value = query.double()
    centered = value - value.mean(-1, keepdim=True)
    variance = centered.square().mean(-1, keepdim=True)
    normalized = centered * torch.rsqrt(variance + float(layer_norm.eps))
    normalized = (
        normalized * layer_norm.weight.double()
        + layer_norm.bias.double()
    )
    weights = model.lm_head.weight.index_select(0, answer_ids).double()
    logits = normalized @ weights.T
    return logits - logits.mean(-1, keepdim=True)


def centered_logit_jacobian(
    model: TinyLLMModel,
    query: torch.Tensor,
    answer_ids: torch.Tensor,
) -> torch.Tensor:
    """Return the exact float64 Jacobian of centered final answer logits."""
    layer_norm = model.transformer["ln_f"]
    value = query.double()
    centered = value - value.mean(-1, keepdim=True)
    variance = centered.square().mean(-1, keepdim=True)
    inverse_scale = torch.rsqrt(variance + float(layer_norm.eps))
    normalized = centered * inverse_scale
    weights = model.lm_head.weight.index_select(0, answer_ids).double()
    weights = weights - weights.mean(0, keepdim=True)
    weighted = weights * layer_norm.weight.double()[None, :]
    weighted_mean = weighted.mean(-1, keepdim=True)
    weighted_normalized_mean = torch.einsum(
        "kd,nd->nk", weighted, normalized
    ) / value.shape[-1]
    return inverse_scale[:, None, :] * (
        weighted[None, :, :]
        - weighted_mean[None, :, :]
        - normalized[:, None, :] * weighted_normalized_mean[:, :, None]
    )


def decompose_displacement(
    jacobian: torch.Tensor,
    displacement: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, torch.Tensor]:
    """Project displacement into the Jacobian row space and its complement."""
    jacobian = jacobian.double()
    displacement = displacement.double()
    gram = jacobian @ jacobian.transpose(-1, -2)
    gram_pinv = torch.linalg.pinv(
        gram, rtol=rtol, atol=atol, hermitian=True
    )
    visible = torch.einsum("nkd,nd->nk", jacobian, displacement)
    coefficients = torch.einsum("nkl,nl->nk", gram_pinv, visible)
    task = torch.einsum("nkd,nk->nd", jacobian, coefficients)
    kernel = displacement - task
    singular_values = torch.linalg.svdvals(jacobian)
    threshold = torch.maximum(
        torch.full_like(singular_values[:, :1], atol),
        singular_values[:, :1] * rtol,
    )
    rank = (singular_values > threshold).sum(1)
    reconstructed = task + kernel
    denominator = torch.linalg.vector_norm(displacement, dim=1).clamp_min(1e-12)
    reconstruction_error = torch.linalg.vector_norm(
        reconstructed - displacement, dim=1
    ) / denominator
    kernel_visible = torch.einsum("nkd,nd->nk", jacobian, kernel)
    leakage = torch.linalg.vector_norm(kernel_visible, dim=1) / torch.linalg.vector_norm(
        visible, dim=1
    ).clamp_min(1e-10)
    return {
        "task": task,
        "kernel": kernel,
        "rank": rank,
        "singular_values": singular_values,
        "reconstruction_error": reconstruction_error,
        "kernel_leakage": leakage,
    }


def random_rowspace_basis(
    width: int,
    rank: int,
    seed: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if rank < 1 or rank > width:
        raise ValueError("random row-space rank is invalid")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn((width, rank), generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(matrix, mode="reduced")
    return basis.to(device)


def random_kernel_displacement(
    displacement: torch.Tensor, basis: torch.Tensor
) -> torch.Tensor:
    task = (displacement.double() @ basis) @ basis.T
    return displacement.double() - task


@torch.no_grad()
def _final_outputs(
    model: TinyLLMModel,
    query: torch.Tensor,
    answer_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = model.transformer["ln_f"](query)
    logits = model.lm_head(normalized).index_select(-1, answer_ids)
    posterior = torch.softmax(logits, -1).double()
    centered = logits.double() - logits.double().mean(-1, keepdim=True)
    return posterior, centered


def finite_difference_relative_error(
    model: TinyLLMModel,
    barycenter: torch.Tensor,
    direction: torch.Tensor,
    jacobian: torch.Tensor,
    answer_ids: torch.Tensor,
    step: float,
) -> torch.Tensor:
    direction = direction.double()
    norms = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    fallback = torch.zeros_like(direction)
    fallback[:, 0] = 1.0
    direction = torch.where(
        norms > 1e-12, direction / norms.clamp_min(1e-12), fallback
    )
    plus = centered_logits_double(
        model, barycenter.double() + step * direction, answer_ids
    )
    minus = centered_logits_double(
        model, barycenter.double() - step * direction, answer_ids
    )
    estimate = (plus - minus) / (2.0 * step)
    predicted = torch.einsum("nkd,nd->nk", jacobian.double(), direction)
    denominator = torch.maximum(
        torch.maximum(
            torch.linalg.vector_norm(estimate, dim=1),
            torch.linalg.vector_norm(predicted, dim=1),
        ),
        torch.full(
            (len(direction),), 1e-10, dtype=torch.float64, device=direction.device
        ),
    )
    return torch.linalg.vector_norm(estimate - predicted, dim=1) / denominator


def _quantile(value: torch.Tensor, q: float) -> float:
    return float(torch.quantile(value.double(), q))


def _effect_attribution(
    baseline: torch.Tensor,
    full: torch.Tensor,
    task_only: torch.Tensor,
    kernel_only: torch.Tensor,
    config: FinalQueryKernelConfig,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    full_change = full.double() - baseline.double()
    task_change = task_only.double() - baseline.double()
    kernel_change = kernel_only.double() - baseline.double()
    full_norm = torch.linalg.vector_norm(full_change, dim=1)
    mask = full_norm > 1e-8
    if not bool(mask.any()):
        raise ValueError("full barycenter has no material centered-logit effect")
    denominator = full_norm[mask]
    relative_task_residual = torch.linalg.vector_norm(
        full_change[mask] - task_change[mask], dim=1
    ) / denominator
    task_norm = torch.linalg.vector_norm(task_change[mask], dim=1)
    cosine = (
        (full_change[mask] * task_change[mask]).sum(1)
        / (denominator * task_norm).clamp_min(1e-12)
    )
    kernel_effect_ratio = torch.linalg.vector_norm(
        kernel_change[mask], dim=1
    ) / denominator
    result = {
        "material_row_count": int(mask.sum()),
        "mean_relative_task_residual": float(relative_task_residual.mean()),
        "p95_relative_task_residual": _quantile(relative_task_residual, 0.95),
        "median_task_full_cosine": _quantile(cosine, 0.50),
        "mean_kernel_full_effect_ratio": float(kernel_effect_ratio.mean()),
        "p95_kernel_full_effect_ratio": _quantile(kernel_effect_ratio, 0.95),
    }
    result["pass"] = bool(
        result["mean_relative_task_residual"]
        <= config.task_effect_mean_residual_ceiling
        and result["p95_relative_task_residual"]
        <= config.task_effect_p95_residual_ceiling
        and result["median_task_full_cosine"]
        >= config.task_effect_median_cosine_floor
        and result["mean_kernel_full_effect_ratio"]
        <= config.kernel_effect_mean_ratio_ceiling
        and result["p95_kernel_full_effect_ratio"]
        <= config.kernel_effect_p95_ratio_ceiling
    )
    return result, {
        "relative_task_residual": relative_task_residual,
        "task_full_cosine": cosine,
        "kernel_full_effect_ratio": kernel_effect_ratio,
    }


def _capture_final_queries(
    model: TinyLLMModel,
    cohort: parent.ExactTaskFiberCohort,
    answer_ids: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    states: list[torch.Tensor] = []
    baselines: list[torch.Tensor] = []
    maximum_replay_error = 0.0
    for start in range(0, len(cohort.input_ids), batch_size):
        stop = min(len(cohort.input_ids), start + batch_size)
        inputs = cohort.input_ids[start:stop].to(answer_ids.device)
        captured, baseline = parent._capture_batch(model, inputs, answer_ids)
        query = captured[FINAL_CUT][:, -1, :]
        replay, _ = _final_outputs(model, query, answer_ids)
        maximum_replay_error = max(
            maximum_replay_error,
            float(torch.max(torch.abs(replay - baseline))),
        )
        states.append(query.detach())
        baselines.append(baseline.detach())
    return torch.cat(states), torch.cat(baselines), maximum_replay_error


def _parent_arrays(regime: str) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(PARENT_D8_DIAGNOSTICS_PATH) as arrays:
        baseline = torch.from_numpy(
            arrays[f"cosine_interval__{regime}__baseline_posterior"]
        ).double()
        full = torch.from_numpy(
            arrays[
                f"cosine_interval__{regime}__{FINAL_CUT}__correct_posterior"
            ]
        ).double()
    return baseline, full


def analyze_regime(
    model: TinyLLMModel,
    task: parent.CircleTaskConfig,
    regime: str,
    config: FinalQueryKernelConfig,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cohort = parent.generate_exact_task_fibers(
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
    query, baseline_posterior, maximum_replay_error = _capture_final_queries(
        model, cohort, answer_ids, config.batch_size
    )
    shaped = query.reshape(config.fiber_pairs, 2, -1)
    barycenter_unique = shaped.mean(1).double()
    full_query = barycenter_unique.repeat_interleave(2, dim=0)
    displacement = full_query - query.double()
    pair_direction = shaped[:, 0].double() - shaped[:, 1].double()

    task_parts: list[torch.Tensor] = []
    kernel_parts: list[torch.Tensor] = []
    rank_parts: list[torch.Tensor] = []
    singular_parts: list[torch.Tensor] = []
    reconstruction_parts: list[torch.Tensor] = []
    leakage_parts: list[torch.Tensor] = []
    finite_difference_parts: list[torch.Tensor] = []
    remaining_fd = min(config.finite_difference_pairs, config.fiber_pairs)
    for start in range(0, config.fiber_pairs, config.jacobian_batch_pairs):
        stop = min(config.fiber_pairs, start + config.jacobian_batch_pairs)
        barycenter = barycenter_unique[start:stop]
        jacobian = centered_logit_jacobian(model, barycenter, answer_ids)
        row_displacement = displacement[2 * start : 2 * stop]
        repeated_jacobian = jacobian.repeat_interleave(2, dim=0)
        parts = decompose_displacement(
            repeated_jacobian,
            row_displacement,
            rtol=config.jacobian_rtol,
            atol=config.jacobian_atol,
        )
        task_parts.append(parts["task"])
        kernel_parts.append(parts["kernel"])
        rank_parts.append(parts["rank"].reshape(stop - start, 2)[:, 0])
        singular_parts.append(
            parts["singular_values"].reshape(stop - start, 2, -1)[:, 0]
        )
        reconstruction_parts.append(parts["reconstruction_error"])
        leakage_parts.append(parts["kernel_leakage"])
        if remaining_fd:
            count = min(remaining_fd, stop - start)
            finite_difference_parts.append(
                finite_difference_relative_error(
                    model,
                    barycenter[:count],
                    pair_direction[start : start + count],
                    jacobian[:count],
                    answer_ids,
                    config.finite_difference_step,
                )
            )
            remaining_fd -= count
    task_displacement = torch.cat(task_parts)
    kernel_displacement = torch.cat(kernel_parts)
    ranks = torch.cat(rank_parts)
    singular_values = torch.cat(singular_parts)
    reconstruction_error = torch.cat(reconstruction_parts)
    kernel_leakage = torch.cat(leakage_parts)
    finite_difference_error = torch.cat(finite_difference_parts)
    random_rank = int(torch.median(ranks).item())
    random_basis = random_rowspace_basis(
        query.shape[-1],
        random_rank,
        RANDOM_PROJECTOR_SEEDS[regime],
        device=device,
    )
    random_kernel = random_kernel_displacement(displacement, random_basis)

    arm_queries = {
        "full": full_query.to(query.dtype),
        "task_only": (query.double() + task_displacement).to(query.dtype),
        "kernel_only": (query.double() + kernel_displacement).to(query.dtype),
        "random_kernel": (query.double() + random_kernel).to(query.dtype),
    }
    baseline_direct, baseline_logits = _final_outputs(model, query, answer_ids)
    arm_posteriors: dict[str, torch.Tensor] = {}
    arm_logits: dict[str, torch.Tensor] = {}
    for name, state in arm_queries.items():
        posterior, logits = _final_outputs(model, state, answer_ids)
        arm_posteriors[name] = posterior.detach().cpu()
        arm_logits[name] = logits.detach().cpu()
    baseline_posterior_cpu = baseline_posterior.detach().cpu()
    baseline_direct_cpu = baseline_direct.detach().cpu()
    baseline_logits_cpu = baseline_logits.detach().cpu()
    direct_replay_error = float(
        torch.max(torch.abs(baseline_direct_cpu - baseline_posterior_cpu))
    )
    parent_baseline_error = 0.0
    parent_full_error = 0.0
    parent_replay_required = bool(
        not config.allow_underpowered
        and config.fiber_pairs == 512
        and set(config.regimes) == set(REGIMES)
    )
    if parent_replay_required:
        stored_baseline, stored_full = _parent_arrays(regime)
        parent_baseline_error = float(
            torch.max(torch.abs(stored_baseline - baseline_posterior_cpu))
        )
        parent_full_error = float(
            torch.max(torch.abs(stored_full - arm_posteriors["full"]))
        )

    targets = cohort.cosine_targets
    baseline_metrics = parent._posterior_metrics(
        baseline_posterior_cpu,
        targets,
        "cosine_interval",
        task,
        cohort.future_phase,
        cohort.cosine,
    )
    arms = {
        name: parent._candidate_record(
            posterior,
            baseline_posterior_cpu,
            targets,
            "cosine_interval",
            task,
            cohort,
            config,
        )
        for name, posterior in arm_posteriors.items()
    }
    original_pair_norm = torch.linalg.vector_norm(pair_direction, dim=1)
    kernel_query = arm_queries["kernel_only"].reshape(
        config.fiber_pairs, 2, -1
    ).double()
    kernel_pair_norm = torch.linalg.vector_norm(
        kernel_query[:, 0] - kernel_query[:, 1], dim=1
    )
    contraction_ratio = float(
        kernel_pair_norm.mean() / original_pair_norm.mean().clamp_min(1e-12)
    )
    attribution, attribution_arrays = _effect_attribution(
        baseline_logits_cpu,
        arm_logits["full"],
        arm_logits["task_only"],
        arm_logits["kernel_only"],
        config,
    )
    numerical_contract = bool(
        bool((ranks == 15).all())
        and float(reconstruction_error.max())
        <= config.decomposition_relative_error_ceiling
        and float(kernel_leakage.max()) <= config.kernel_leakage_ceiling
        and float(finite_difference_error.max())
        <= config.finite_difference_relative_error_ceiling
    )
    replay_pass = bool(
        maximum_replay_error <= config.replay_tolerance
        and direct_replay_error <= config.replay_tolerance
        and (
            not parent_replay_required
            or (
                parent_baseline_error <= config.replay_tolerance
                and parent_full_error <= config.replay_tolerance
            )
        )
    )
    kernel_task_sufficient = bool(
        arms["kernel_only"]["gates"]["task_sufficient"]
    )
    contraction_pass = bool(
        contraction_ratio <= config.kernel_contraction_ratio_ceiling
    )
    task_only_active = bool(
        not arms["task_only"]["gates"]["task_sufficient"]
    )
    random_specificity = bool(
        not arms["random_kernel"]["gates"]["task_sufficient"]
        and arms["random_kernel"]["gates"]["mean_posterior_js"]
        >= arms["kernel_only"]["gates"]["mean_posterior_js"]
        + config.random_js_disadvantage_floor
    )
    baseline_validity = bool(
        baseline_metrics["exact_bin_accuracy"]
        >= config.baseline_accuracy_floor
    )
    finite_contract = bool(
        torch.isfinite(query).all()
        and torch.isfinite(task_displacement).all()
        and torch.isfinite(kernel_displacement).all()
        and torch.isfinite(random_kernel).all()
        and all(torch.isfinite(value).all() for value in arm_posteriors.values())
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
        and kernel_task_sufficient
        and contraction_pass
        and attribution["pass"]
        and task_only_active
        and random_specificity
    )
    record = {
        "regime": regime,
        "cohort_contract": dict(cohort.contract),
        "baseline": baseline_metrics,
        "arms": arms,
        "geometry": {
            "mean_original_pair_norm": float(original_pair_norm.mean()),
            "mean_kernel_pair_norm": float(kernel_pair_norm.mean()),
            "kernel_pair_contraction_ratio": contraction_ratio,
            "mean_task_displacement_norm_fraction": float(
                (
                    torch.linalg.vector_norm(task_displacement, dim=1)
                    / torch.linalg.vector_norm(displacement, dim=1).clamp_min(1e-12)
                ).mean()
            ),
            "mean_kernel_displacement_norm_fraction": float(
                (
                    torch.linalg.vector_norm(kernel_displacement, dim=1)
                    / torch.linalg.vector_norm(displacement, dim=1).clamp_min(1e-12)
                ).mean()
            ),
        },
        "attribution": attribution,
        "jacobian": {
            "rank_counts": {
                str(rank): int((ranks == rank).sum())
                for rank in sorted(set(ranks.detach().cpu().tolist()))
            },
            "minimum_rank": int(ranks.min()),
            "maximum_rank": int(ranks.max()),
            "random_control_rank": random_rank,
            "maximum_reconstruction_relative_error": float(
                reconstruction_error.max()
            ),
            "maximum_kernel_leakage": float(kernel_leakage.max()),
            "maximum_finite_difference_relative_error": float(
                finite_difference_error.max()
            ),
            "minimum_nonzero_singular_value": float(
                singular_values[:, :15].min()
            ),
        },
        "replay": {
            "maximum_capture_replay_error": maximum_replay_error,
            "maximum_direct_replay_error": direct_replay_error,
            "parent_replay_required": parent_replay_required,
            "maximum_parent_baseline_error": parent_baseline_error,
            "maximum_parent_full_error": parent_full_error,
        },
        "gates": {
            "baseline_validity": baseline_validity,
            "cohort_contract": cohort.contract.get("pass") is True,
            "finite": finite_contract,
            "jacobian_numerical_contract": numerical_contract,
            "replay": replay_pass,
            "kernel_task_sufficient": kernel_task_sufficient,
            "kernel_pair_contraction": contraction_pass,
            "task_effect_attribution": attribution["pass"],
            "task_only_causally_active": task_only_active,
            "random_projector_specificity": random_specificity,
            "validity": valid,
            "primary_regime_gate": primary_regime_gate,
        },
    }
    arrays: dict[str, np.ndarray] = {
        "baseline_posterior": baseline_posterior_cpu.numpy(),
        "full_posterior": arm_posteriors["full"].numpy(),
        "task_only_posterior": arm_posteriors["task_only"].numpy(),
        "kernel_only_posterior": arm_posteriors["kernel_only"].numpy(),
        "random_kernel_posterior": arm_posteriors["random_kernel"].numpy(),
        "jacobian_rank": ranks.detach().cpu().numpy(),
        "jacobian_singular_values": singular_values.detach().cpu().numpy(),
        "reconstruction_relative_error": reconstruction_error.detach().cpu().numpy(),
        "kernel_leakage": kernel_leakage.detach().cpu().numpy(),
        "finite_difference_relative_error": finite_difference_error.detach().cpu().numpy(),
        "original_pair_norm": original_pair_norm.detach().cpu().numpy(),
        "kernel_pair_norm": kernel_pair_norm.detach().cpu().numpy(),
        "task_displacement_norm": torch.linalg.vector_norm(
            task_displacement, dim=1
        ).detach().cpu().numpy(),
        "kernel_displacement_norm": torch.linalg.vector_norm(
            kernel_displacement, dim=1
        ).detach().cpu().numpy(),
    }
    arrays.update(
        {
            name: value.detach().cpu().numpy()
            for name, value in attribution_arrays.items()
        }
    )
    return record, arrays


def _fingerprint(
    config: FinalQueryKernelConfig,
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
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _classification(
    config: FinalQueryKernelConfig,
    valid: bool,
    primary: bool,
    regimes: Mapping[str, Mapping[str, Any]],
) -> str:
    if config.allow_underpowered:
        return "systems_lifecycle_only_not_quality_evidence"
    if not valid:
        return "invalid_final_query_task_kernel_campaign"
    if primary:
        return "final_query_task_kernel_explains_barycenter_failure"
    if all(
        item["gates"]["kernel_pair_contraction"]
        and not item["gates"]["kernel_task_sufficient"]
        for item in regimes.values()
    ):
        return "finite_task_kernel_curvature_breaks_preservation"
    if any(
        not item["gates"]["kernel_pair_contraction"]
        for item in regimes.values()
    ):
        return "task_rowspace_contains_material_fiber_chord"
    if all(
        item["gates"]["kernel_task_sufficient"]
        for item in regimes.values()
    ):
        return "kernel_preserves_but_complete_attribution_gate_fails"
    return "final_query_task_kernel_hypothesis_not_supported"


def run_campaign(
    config: FinalQueryKernelConfig,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    source_digests = _source_digests()
    if source_digests["preregistration"] != PREREGISTRATION_SHA256:
        raise ValueError("preregistration changed")
    if source_digests["parent_runner"] != PARENT_RUNNER_SHA256:
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
            == _fingerprint(config, implementation, existing.get("source_hashes", {}))
            and Path(existing["artifacts"]["diagnostics"]).is_file()
            and _sha256(Path(existing["artifacts"]["diagnostics"]))
            == existing["artifacts"]["diagnostics_sha256"]
        ):
            print("campaign already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed campaign {campaign_path}")

    task, source_hashes, parent_detail = _validate_parent(config)
    fingerprint = _fingerprint(config, implementation, source_hashes)
    device = _resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model, initial_system_state = parent._load_model(
        CHECKPOINT_PATH, "d8", "cosine_interval", device
    )
    initial_model_state = parent._state_digest(model)
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
    final_model_state = parent._state_digest(model)
    final_system_state = parent._model_system_state(model)
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
            "classification": "observational_task_geometry_not_causally_sufficient",
            "valid_d8_null": parent_detail["gates"]["validity"],
            "mature_causal_front": parent_detail["summary"]["mature_causal_front"],
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
            "Only the final post-MLP query vector and centered native answer-logit map are decomposed.",
            "The Jacobian projector is analytic and activation-local; no carrier, probe, decoder, threshold, or model parameter is fit.",
            "A local task kernel is not a globally invariant representation subspace.",
            "No earlier-layer scan or training is licensed by this campaign.",
        ],
    }
    if not _finite(campaign):
        raise ValueError("non-finite final-query task-kernel campaign")
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
            "data/experiments/tinyllm_final_query_task_kernel/"
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
    config = FinalQueryKernelConfig(
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
