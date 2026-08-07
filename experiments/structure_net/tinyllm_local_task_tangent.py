#!/usr/bin/env python3
"""Causally decompose fixed-writer error into local task tangent and kernel."""

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

import torch

import experiments.structure_net.tinyllm_carrier_jacobian_axis_audit as jacobian
import experiments.structure_net.tinyllm_continuous_readout_calibration as readout
import experiments.structure_net.tinyllm_fixed_gauge_error_decomposition as decomposition
import experiments.structure_net.tinyllm_fixed_gauge_writer_capacity as capacity
import experiments.structure_net.tinyllm_fixed_semantic_gauge_writer as fixed
import experiments.structure_net.tinyllm_cross_seed_causal_carrier_transport as transport
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-c2-local-task-tangent.v1"
HYPOTHESIS_ID = "tinyllm-c2-local-task-tangent-v1"
PREDECESSOR_CAMPAIGN_SHA256 = (
    "c5592ee53b96e2d064a992070be6c3e699b24d88dbb658283e8dc2f00c267078"
)
PREDECESSOR_IMPLEMENTATION_SHA256 = (
    "7c284e35b5afc225eea45309262ab83c5f6d276736a557ebf20675ed3ccbfe7b"
)
CONTROL_NAMES = (
    "kernel_only",
    "tangent_flipped",
    "tangent_shuffled",
    "tangent_random",
)


@dataclass(frozen=True)
class LocalTaskTangentConfig:
    source_campaign: str = (
        "data/experiments/tinyllm_fixed_gauge_writer_capacity/"
        "20260806_d6_preregistered_diagnostic"
    )
    seeds: tuple[int, ...] = fixed.PRIMARY_SEEDS
    orbit_count: int = 64
    carrier_rank: int = 3
    writer_order: int = 4
    fine_step_std: float = 0.025
    coarse_step_std: float = 0.05
    coordinate_scale_floor: float = 1e-8
    gradient_denominator_floor: float = 1e-12
    decomposition_tolerance: float = 1e-8
    replay_tolerance: float = 1e-6
    derivative_cosine_floor: float = 0.98
    derivative_relative_l2_ceiling: float = 0.15
    signed_error_r2_floor: float = 0.50
    residual_mae_fraction_ceiling: float = 0.50
    sign_agreement_floor: float = 0.75
    sign_magnitude_floor_bins: float = 0.01
    specificity_margin_bins: float = 0.125
    helpful_improvement_bins: float = 0.125
    activation_batch_size: int = 256
    continuation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != fixed.PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("primary local-tangent seeds are fixed to 7,29,53")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.orbit_count != 64:
            raise ValueError("the frozen predecessor requires 64 exact orbits")
        if self.carrier_rank != 3 or self.writer_order != 4:
            raise ValueError("carrier rank and writer order are fixed to three and four")
        if not 0.0 < self.fine_step_std < self.coarse_step_std:
            raise ValueError("finite-difference steps must be positive and ordered")
        for name in (
            "coordinate_scale_floor",
            "gradient_denominator_floor",
            "decomposition_tolerance",
            "replay_tolerance",
            "specificity_margin_bins",
            "helpful_improvement_bins",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")


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


def _json_compatible(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(capacity.__file__),
        Path(decomposition.__file__),
        Path(fixed.__file__),
        Path(transport.__file__),
        Path(transport.rank.__file__),
        Path(transport.coupling.__file__),
        Path(jacobian.__file__),
        Path(readout.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evidence_role(config: LocalTaskTangentConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_post_outcome_underpowered_mechanistic_diagnostic"
    )


def tangent_kernel_decomposition(
    residual: torch.Tensor,
    gradient: torch.Tensor,
    coordinate_scale: torch.Tensor,
    denominator_floor: float,
) -> dict[str, torch.Tensor]:
    """Project a raw-coordinate residual onto a local standardized gradient."""
    if (
        residual.ndim != 2
        or gradient.shape != residual.shape
        or coordinate_scale.ndim != 1
        or coordinate_scale.shape[0] != residual.shape[1]
    ):
        raise ValueError("residual, gradient, and scale shapes are incompatible")
    residual_std = residual.double() / coordinate_scale.double()
    gradient = gradient.double()
    denominator = gradient.square().sum(1, keepdim=True).clamp_min(
        denominator_floor
    )
    tangent_std = (
        (residual_std * gradient).sum(1, keepdim=True) / denominator
    ) * gradient
    kernel_std = residual_std - tangent_std
    scale = coordinate_scale.double().unsqueeze(0)
    return {
        "residual_std": residual_std,
        "tangent_std": tangent_std,
        "kernel_std": kernel_std,
        "tangent_raw": tangent_std * scale,
        "kernel_raw": kernel_std * scale,
    }


def decomposition_diagnostics(parts: Mapping[str, torch.Tensor]) -> dict[str, float]:
    residual = parts["residual_std"].double()
    tangent = parts["tangent_std"].double()
    kernel = parts["kernel_std"].double()
    reconstruction = torch.linalg.vector_norm(tangent + kernel - residual) / (
        torch.linalg.vector_norm(residual).clamp_min(1e-24)
    )
    tangent_norm = torch.linalg.vector_norm(tangent, dim=1)
    kernel_norm = torch.linalg.vector_norm(kernel, dim=1)
    residual_norm = torch.linalg.vector_norm(residual, dim=1)
    denominator = (tangent_norm * kernel_norm).clamp_min(1e-24)
    cosine = (tangent * kernel).sum(1).abs() / denominator
    nonzero = (tangent_norm * kernel_norm) > 1e-20
    maximum_cosine = float(cosine[nonzero].max()) if bool(nonzero.any()) else 0.0
    return {
        "reconstruction_relative_error": float(reconstruction),
        "maximum_tangent_kernel_absolute_cosine": maximum_cosine,
        "mean_residual_norm_std": float(residual_norm.mean()),
        "mean_tangent_norm_std": float(tangent_norm.mean()),
        "mean_kernel_norm_std": float(kernel_norm.mean()),
        "mean_tangent_norm_fraction": float(
            (tangent_norm / residual_norm.clamp_min(1e-24)).mean()
        ),
        "mean_kernel_norm_fraction": float(
            (kernel_norm / residual_norm.clamp_min(1e-24)).mean()
        ),
    }


def norm_matched_random(
    tangent_std: torch.Tensor, seed: int
) -> torch.Tensor:
    """Return deterministic random standardized directions with matched row norms."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random = torch.randn(
        tangent_std.shape, generator=generator, dtype=torch.float64, device="cpu"
    ).to(tangent_std.device)
    random = random / torch.linalg.vector_norm(random, dim=1, keepdim=True).clamp_min(
        1e-24
    )
    norms = torch.linalg.vector_norm(tangent_std.double(), dim=1, keepdim=True)
    return random * norms


def fixed_permutation(rows: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(rows, generator=generator).to(device)


def _state_from_coordinates(
    cell: Mapping[str, Any], basis: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
    defect = (coordinates.double() @ basis.double()).to(
        dtype=cell["full_defect"].dtype
    )
    return cell["propagated"] + defect.reshape_as(cell["full_defect"])


@torch.no_grad()
def _continue_states(
    system: Any,
    task: CircleTaskConfig,
    config: LocalTaskTangentConfig,
    target_cut: str,
    states: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    names = tuple(states)
    repeated = torch.cat(
        [transport.coupling._repeat_patch(states[name], 2) for name in names]
    )
    combined = transport.coupling._continue_tensor(
        system,
        target_cut,
        repeated,
        task,
        config.continuation_batch_size,
    )
    per_state = config.orbit_count * 2
    raw = dict(zip(names, combined.split(per_state)))
    return {
        name: value.reshape(config.orbit_count, 2, -1)[:, 0]
        for name, value in raw.items()
    }


def _angles(posteriors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: readout.posterior_moment(value)[0] for name, value in posteriors.items()}


@torch.no_grad()
def finite_difference_cell(
    system: Any,
    task: CircleTaskConfig,
    config: LocalTaskTangentConfig,
    cell: Mapping[str, Any],
    basis: torch.Tensor,
    predicted: torch.Tensor,
    target: torch.Tensor,
    coordinate_scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    coordinate_states: dict[str, torch.Tensor] = {
        "order4": predicted,
        "direct_rank3": target,
    }
    for step_name, step in (
        ("fine", config.fine_step_std),
        ("coarse", config.coarse_step_std),
    ):
        for axis in range(config.carrier_rank):
            delta = torch.zeros_like(predicted)
            delta[:, axis] = step * coordinate_scale[axis]
            coordinate_states[f"{step_name}_axis_{axis + 1}_plus"] = predicted + delta
            coordinate_states[f"{step_name}_axis_{axis + 1}_minus"] = predicted - delta
    states = {
        name: _state_from_coordinates(cell, basis, coordinates)
        for name, coordinates in coordinate_states.items()
    }
    posteriors = _continue_states(
        system, task, config, cell["target_cut"], states
    )
    angles = _angles(posteriors)
    bins = len(task.answer_token_ids)
    gradients: dict[str, torch.Tensor] = {}
    for step_name, step in (
        ("fine", config.fine_step_std),
        ("coarse", config.coarse_step_std),
    ):
        values = []
        for axis in range(config.carrier_rank):
            plus = angles[f"{step_name}_axis_{axis + 1}_plus"]
            minus = angles[f"{step_name}_axis_{axis + 1}_minus"]
            values.append(jacobian._wrapped_bins(plus, minus, bins) / (2.0 * step))
        gradients[step_name] = torch.stack(values, dim=1)
    residual_std = (target.double() - predicted.double()) / coordinate_scale.double()
    predicted_delta = (gradients["fine"] * residual_std).sum(1)
    observed_delta = jacobian._wrapped_bins(
        angles["direct_rank3"], angles["order4"], bins
    )
    return {
        "fine_gradient": gradients["fine"],
        "coarse_gradient": gradients["coarse"],
        "predicted_delta": predicted_delta,
        "observed_delta": observed_delta,
    }


@torch.no_grad()
def evaluate_science_states(
    system: Any,
    task: CircleTaskConfig,
    config: LocalTaskTangentConfig,
    cell: Mapping[str, Any],
    basis: torch.Tensor,
    coordinates: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    states = {
        "zero": cell["propagated"],
        "exact": cell["exact"],
        **{
            name: _state_from_coordinates(cell, basis, value)
            for name, value in coordinates.items()
        },
    }
    posteriors = _continue_states(
        system, task, config, cell["target_cut"], states
    )
    exact = posteriors["exact"]
    per_state = config.orbit_count * 2
    exact_repeated = exact[:, None].expand(-1, 2, -1).reshape(per_state, -1)
    exact_diagnostics = transport.coupling._diagnostics(
        exact_repeated, cell["dataset"]
    )
    endpoint_config = readout.ContinuousReadoutCalibrationConfig()
    records = {}
    for name, posterior in posteriors.items():
        repeated = posterior[:, None].expand(-1, 2, -1).reshape(per_state, -1)
        diagnostics = transport.coupling._diagnostics(repeated, cell["dataset"])
        records[name] = {
            "continuous": readout.continuous_metrics(
                posterior,
                exact,
                diagnostics,
                exact_diagnostics,
                endpoint_config,
            ),
            "diagnostics": diagnostics,
        }
    return records


def _load_predecessor(
    config: LocalTaskTangentConfig,
) -> tuple[dict[str, Any], Path, dict[int, tuple[dict[str, Any], Path]]]:
    campaign_path = Path(config.source_campaign) / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text())
    expected_classes = {str(seed): "unresolved_writer_limited" for seed in fixed.PRIMARY_SEEDS}
    if (
        _sha256(campaign_path) != PREDECESSOR_CAMPAIGN_SHA256
        or campaign.get("schema_version") != capacity.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != capacity.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256") != PREDECESSOR_IMPLEMENTATION_SHA256
        or campaign.get("aggregates", {}).get("classification_by_seed")
        != expected_classes
        or int(campaign.get("summary", {}).get("trained_models", -1)) != 0
    ):
        raise ValueError(f"invalid writer-capacity predecessor {campaign_path}")
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if not set(config.seeds).issubset(entries):
        raise ValueError("writer-capacity predecessor lacks a requested checkpoint")
    details = {}
    for seed in config.seeds:
        entry = entries[seed]
        path = Path(entry["path"])
        detail = json.loads(path.read_text())
        candidate = detail.get("gates", {}).get("candidates", {}).get(
            "quotient_order4", {}
        )
        if (
            _sha256(path) != entry.get("result_sha256")
            or detail.get("schema_version") != capacity.SCHEMA_VERSION
            or detail.get("hypothesis_id") != capacity.HYPOTHESIS_ID
            or detail.get("implementation_sha256") != PREDECESSOR_IMPLEMENTATION_SHA256
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or detail.get("classification") != "unresolved_writer_limited"
            or candidate.get("specific_causal_pass") is not False
            or len(detail.get("heldout_cells", [])) != 4
        ):
            raise ValueError(f"invalid writer-capacity result {path}")
        details[seed] = (detail, path)
    return campaign, campaign_path, details


def _fingerprint(
    config: LocalTaskTangentConfig,
    seed: int,
    predecessor_campaign_sha256: str,
    predecessor_result_sha256: str,
    checkpoint_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "cohort_seeds": fixed.COHORT_SEEDS,
        "seed": seed,
        "predecessor_campaign_sha256": predecessor_campaign_sha256,
        "predecessor_result_sha256": predecessor_result_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def state_gate_summary(
    cells: Sequence[Mapping[str, Any]],
    specificity_margin_bins: float,
) -> dict[str, Any]:
    means = {
        name: sum(
            cell["states"][name]["continuous"]["mean_moment_shift_bins"]
            for cell in cells
        )
        / len(cells)
        for name in (
            "order4",
            "tangent_only",
            *CONTROL_NAMES,
        )
    }
    tangent_pass = all(
        cell["states"]["tangent_only"]["continuous"]["continuous_pass"]
        for cell in cells
    )
    controls = {}
    for name in CONTROL_NAMES:
        any_failure = any(
            not cell["states"][name]["continuous"]["continuous_pass"]
            for cell in cells
        )
        margin = means[name] - means["tangent_only"]
        controls[name] = {
            "any_failure": any_failure,
            "aggregate_mean_shift_bins": means[name],
            "margin_over_tangent_bins": margin,
            "specific": bool(any_failure and margin >= specificity_margin_bins),
        }
    return {
        "aggregate_mean_shift_bins": means,
        "tangent_all_cells_pass": tangent_pass,
        "controls": controls,
        "all_controls_specific": all(value["specific"] for value in controls.values()),
    }


def classify_checkpoint(
    *,
    contracts_valid: bool,
    local_adequate: bool,
    state_gates: Mapping[str, Any],
    helpful_improvement_bins: float,
) -> tuple[str, bool]:
    if not contracts_valid:
        return "invalid", False
    tangent_pass = bool(state_gates["tangent_all_cells_pass"])
    specific = bool(state_gates["all_controls_specific"])
    full_gate = bool(local_adequate and tangent_pass and specific)
    if full_gate:
        return "local_task_tangent_sufficient", True
    if not local_adequate:
        return "nonlinear_at_writer_residual_scale", False
    if tangent_pass:
        return "nonunique_or_curved_correction", False
    means = state_gates["aggregate_mean_shift_bins"]
    if means["order4"] - means["tangent_only"] >= helpful_improvement_bins:
        return "tangent_helpful_not_sufficient", False
    return "task_tangent_insufficient", False


def _campaign_decision(
    classifications: Sequence[str], tangent_pass_count: int, allow_underpowered: bool
) -> dict[str, Any]:
    if allow_underpowered:
        conclusion = "systems_lifecycle_only_not_quality_evidence"
        supported = False
    elif len(classifications) == 3 and tangent_pass_count == 3:
        conclusion = "supported_local_task_tangent_sufficient_three_of_three"
        supported = True
    elif len(set(classifications)) == 1:
        conclusion = classifications[0]
        supported = False
    else:
        conclusion = "checkpoint_stratified_local_geometry"
        supported = False
    return {
        "supported": supported,
        "local_task_tangent_pass_count": tangent_pass_count,
        "required_checkpoint_count": 3,
        "conclusion": conclusion,
    }


def _campaign_is_reusable(
    campaign: Mapping[str, Any],
    config: LocalTaskTangentConfig,
    implementation: str,
) -> bool:
    return bool(
        campaign.get("schema_version") == SCHEMA_VERSION
        and campaign.get("hypothesis_id") == HYPOTHESIS_ID
        and campaign.get("status") == "completed"
        and campaign.get("evidence_role") == _evidence_role(config)
        and campaign.get("implementation_sha256") == implementation
        and campaign.get("configuration") == _json_compatible(asdict(config))
        and int(campaign.get("summary", {}).get("completed", -1))
        == len(config.seeds)
        and len(campaign.get("results", [])) == len(config.seeds)
        and all(
            Path(item.get("path", "")).is_file()
            and _sha256(Path(item["path"])) == item.get("result_sha256")
            for item in campaign.get("results", [])
        )
    )


@torch.no_grad()
def run_campaign(
    config: LocalTaskTangentConfig, output: Path
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
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
    predecessor, predecessor_path, predecessor_details = _load_predecessor(config)
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
            for regime in fixed.REGIMES
        }
        for cohort in fixed.COHORT_SEEDS
    }

    results = []
    for seed in config.seeds:
        prior, prior_path = predecessor_details[seed]
        system, provenance = transport.rank.deck.load_source(
            task, bridge, 2, seed, device
        )
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
        if (
            decomposition._numeric_max_difference(basis_summary, prior["basis"])
            > config.replay_tolerance
        ):
            raise ValueError(f"basis summary mismatch for seed {seed}")
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
                for regime in fixed.REGIMES
            }
            for cohort in fixed.COHORT_SEEDS
        }
        fit_coordinates = torch.cat(
            [
                transport._coordinates(cells["alignment_fit"][regime], basis)
                for regime in fixed.REGIMES
            ]
        )
        coordinate_scale = fit_coordinates.std(0, unbiased=False).clamp_min(
            config.coordinate_scale_floor
        )
        writer = capacity._writer_from_summary(
            prior["alignment_fit"]["writers"]["quotient_order4"], device
        )
        prior_cells = {
            (cell["cohort"], cell["regime"]): cell
            for cell in prior["heldout_cells"]
        }
        heldout = []
        derivative_tensors = []
        replay_errors = []
        numerical_passes = []
        for cohort in fixed.HELDOUT_COHORTS:
            for regime in fixed.REGIMES:
                dataset = datasets[cohort][regime]
                cell = cells[cohort][regime]
                carrier, oracle_audit = decomposition.oracle_orbit_carrier(
                    dataset, task, config.orbit_count, device
                )
                features = capacity.fourier_features(carrier, config.writer_order)
                predicted = transport.apply_affine(features, writer)
                target = transport._coordinates(cell, basis)
                derivative = finite_difference_cell(
                    system,
                    task,
                    config,
                    cell,
                    basis,
                    predicted,
                    target,
                    coordinate_scale,
                )
                parts = tangent_kernel_decomposition(
                    target - predicted,
                    derivative["fine_gradient"],
                    coordinate_scale,
                    config.gradient_denominator_floor,
                )
                diagnostics = decomposition_diagnostics(parts)
                control_seed = (
                    91_000_000
                    + 100_003 * seed
                    + int(fixed.COHORT_SEEDS[cohort][regime])
                )
                permutation = fixed_permutation(
                    config.orbit_count, control_seed, device
                )
                random_std = norm_matched_random(
                    parts["tangent_std"], control_seed + 1
                )
                random_raw = random_std * coordinate_scale.double().unsqueeze(0)
                coordinates = {
                    "direct_rank3": target,
                    "order4": predicted,
                    "tangent_only": predicted + parts["tangent_raw"],
                    "kernel_only": predicted + parts["kernel_raw"],
                    "tangent_flipped": predicted - parts["tangent_raw"],
                    "tangent_shuffled": predicted
                    + parts["tangent_raw"][permutation],
                    "tangent_random": predicted + random_raw,
                }
                states = evaluate_science_states(
                    system,
                    task,
                    config,
                    cell,
                    basis,
                    coordinates,
                )
                coordinate_metrics = transport.coordinate_metrics(predicted, target)
                prior_cell = prior_cells[(cohort, regime)]
                replay = max(
                    decomposition._numeric_max_difference(
                        coordinate_metrics,
                        prior_cell["coordinate_metrics"]["quotient_order4"],
                    ),
                    decomposition._numeric_max_difference(
                        states["order4"]["continuous"],
                        prior_cell["states"]["quotient_order4"]["continuous"],
                    ),
                )
                replay_errors.append(replay)
                finite = all(
                    torch.isfinite(value).all()
                    for value in (
                        derivative["fine_gradient"],
                        derivative["coarse_gradient"],
                        parts["tangent_std"],
                        parts["kernel_std"],
                    )
                )
                numerical = bool(
                    finite
                    and diagnostics["reconstruction_relative_error"]
                    <= config.decomposition_tolerance
                    and diagnostics["maximum_tangent_kernel_absolute_cosine"]
                    <= config.decomposition_tolerance
                    and oracle_audit["mean_shift_bins"] <= 1e-8
                    and oracle_audit["p95_shift_bins"] <= 1e-8
                )
                numerical_passes.append(numerical)
                heldout.append(
                    {
                        "cohort": cohort,
                        "regime": regime,
                        "evaluation_seed": fixed.COHORT_SEEDS[cohort][regime],
                        "coordinate_metrics": coordinate_metrics,
                        "oracle_audit": oracle_audit,
                        "decomposition": diagnostics,
                        "gradient": {
                            "fine_rms_bins_per_std": [
                                jacobian._rms(derivative["fine_gradient"][:, axis])
                                for axis in range(config.carrier_rank)
                            ],
                            "coarse_rms_bins_per_std": [
                                jacobian._rms(derivative["coarse_gradient"][:, axis])
                                for axis in range(config.carrier_rank)
                            ],
                            "minimum_fine_norm": float(
                                torch.linalg.vector_norm(
                                    derivative["fine_gradient"], dim=1
                                ).min()
                            ),
                        },
                        "states": states,
                        "predecessor_replay_maximum_absolute_error": replay,
                        "numerical_contract": numerical,
                        "control_permutation_sha256": hashlib.sha256(
                            permutation.detach().cpu().numpy().tobytes()
                        ).hexdigest(),
                    }
                )
                derivative_tensors.append(
                    {name: value.detach().cpu() for name, value in derivative.items()}
                )

        pooled = {
            name: torch.cat([cell[name] for cell in derivative_tensors], dim=0)
            for name in derivative_tensors[0]
        }
        local = jacobian.linearization_metrics(
            pooled["fine_gradient"],
            pooled["coarse_gradient"],
            pooled["predicted_delta"],
            pooled["observed_delta"],
            config,
        )
        target_controls = all(
            not cell["states"]["zero"]["continuous"]["continuous_pass"]
            and cell["states"]["exact"]["continuous"]["continuous_pass"]
            and cell["states"]["direct_rank3"]["continuous"]["continuous_pass"]
            for cell in heldout
        )
        replay_pass = max(replay_errors) <= config.replay_tolerance
        numerical_pass = bool(
            all(numerical_passes)
            and float(coordinate_scale.min()) > config.coordinate_scale_floor
        )
        state_gates = state_gate_summary(
            heldout, config.specificity_margin_bins
        )
        contracts_valid = bool(replay_pass and numerical_pass and target_controls)
        classification, tangent_gate = classify_checkpoint(
            contracts_valid=contracts_valid,
            local_adequate=local["adequate"],
            state_gates=state_gates,
            helpful_improvement_bins=config.helpful_improvement_bins,
        )
        fingerprint = _fingerprint(
            config,
            seed,
            _sha256(predecessor_path),
            _sha256(prior_path),
            provenance["checkpoint_sha256"],
        )
        result_path = output / "runs" / f"seed_{seed}" / "result.json"
        result = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_id": f"tinyllm-c2-local-task-tangent-seed{seed}",
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
                "frontend_checkpoint": provenance["frontend_checkpoint"],
                "frontend_checkpoint_sha256": provenance[
                    "frontend_checkpoint_sha256"
                ],
                "character_result": str(frozen_path),
                "character_result_sha256": _sha256(frozen_path),
            },
            "basis": basis_summary,
            "coordinate_scale": coordinate_scale.detach().cpu().tolist(),
            "heldout_cells": heldout,
            "local_linearization": local,
            "state_gates": state_gates,
            "gates": {
                "predecessor_replay_contract": replay_pass,
                "maximum_predecessor_replay_error": max(replay_errors),
                "numerical_decomposition_contract": numerical_pass,
                "continuous_target_control_contract": target_controls,
                "local_linearization_adequate": local["adequate"],
                "local_task_tangent_gate": tangent_gate,
            },
            "classification": classification,
            "primary_metric": float(tangent_gate),
            "analysis_seconds": time.perf_counter() - started,
            "artifacts": {"result": str(result_path)},
        }
        _write_json(result_path, result)
        results.append(result)
        print(f"seed {seed}: {classification}", flush=True)
        if _implementation_digest() != implementation:
            raise RuntimeError("local-tangent implementation changed during campaign")

    classifications = [result["classification"] for result in results]
    tangent_count = sum(result["gates"]["local_task_tangent_gate"] for result in results)
    decision = _campaign_decision(
        classifications, tangent_count, config.allow_underpowered
    )
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
            "predecessor_implementation_sha256": predecessor[
                "implementation_sha256"
            ],
        },
        "summary": {
            "requested": len(config.seeds),
            "completed": len(results),
            "failed": 0,
            "excluded": 0,
            "retries": 0,
            "trained_models": 0,
            "fitted_predictive_observers": 0,
            "fitted_writers": 0,
            "finite_difference_continuations": len(config.seeds)
            * len(fixed.HELDOUT_COHORTS)
            * len(fixed.REGIMES)
            * config.carrier_rank
            * 4,
            "causal_component_states": len(config.seeds)
            * len(fixed.HELDOUT_COHORTS)
            * len(fixed.REGIMES)
            * 9,
        },
        "aggregates": {
            **decision,
            "classification_counts": {
                name: classifications.count(name) for name in sorted(set(classifications))
            },
            "classification_by_seed": {
                str(result["seed"]): result["classification"] for result in results
            },
            "gate_counts": {
                name: sum(bool(result["gates"][name]) for result in results)
                for name in (
                    "predecessor_replay_contract",
                    "numerical_decomposition_contract",
                    "continuous_target_control_contract",
                    "local_linearization_adequate",
                    "local_task_tangent_gate",
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
                "path": result["artifacts"]["result"],
                "result_sha256": _sha256(Path(result["artifacts"]["result"])),
            }
            for result in results
        ],
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "The tangent is conditioned on the frozen answer-token decoder and circular angle.",
            "The exact residual, carrier basis, writer, and held-out cells are post-outcome selections.",
            "Finite differences and component patches are local off-manifold interventions.",
            "Three selected checkpoints do not establish population prevalence.",
            "A passing local correction would not itself provide a deployable encoder or shared gauge.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "seed_*" / "result.json"),
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
            "data/experiments/tinyllm_local_task_tangent/"
            "20260807_d6_preregistered_diagnostic"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=_ints, default=fixed.PRIMARY_SEEDS)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = LocalTaskTangentConfig(
        seeds=args.seeds,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
