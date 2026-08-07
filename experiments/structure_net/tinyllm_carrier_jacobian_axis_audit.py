#!/usr/bin/env python3
"""Audit which canonical carrier axes cause failed TinyLLM cross-seed transport."""

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

import experiments.structure_net.tinyllm_cross_seed_causal_carrier_transport as transport
from experiments.structure_net import tinyllm_continuous_readout_calibration as readout
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-carrier-jacobian-axis-audit.v1"
HYPOTHESIS_ID = "tinyllm-c2-carrier-jacobian-axis-audit-v1"
PRIMARY_SEEDS = transport.PRIMARY_SEEDS
REGIMES = transport.REGIMES
HELDOUT_COHORTS = transport.HELDOUT_COHORTS


@dataclass(frozen=True)
class CarrierJacobianAxisAuditConfig:
    source_campaign: str = (
        "data/experiments/tinyllm_cross_seed_causal_carrier_transport/"
        "20260806_d6_preregistered"
    )
    seeds: tuple[int, ...] = PRIMARY_SEEDS
    orbit_count: int = 64
    carrier_rank: int = 3
    fine_step_std: float = 0.025
    coarse_step_std: float = 0.05
    canonical_reproduction_tolerance: float = 1e-10
    axis_reconstruction_tolerance: float = 1e-8
    derivative_cosine_floor: float = 0.98
    derivative_relative_l2_ceiling: float = 0.15
    signed_error_r2_floor: float = 0.50
    residual_mae_fraction_ceiling: float = 0.50
    sign_agreement_floor: float = 0.75
    sign_magnitude_floor_bins: float = 0.01
    dominance_fraction_floor: float = 0.60
    retained_error_floor: float = 0.60
    removed_error_ceiling: float = 0.40
    activation_batch_size: int = 256
    continuation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("confirmatory diagnostic seeds are fixed to 7,29,53")
        if self.orbit_count != 64:
            raise ValueError("the frozen predecessor contract requires 64 exact orbits")
        if self.carrier_rank != 3:
            raise ValueError("the frozen predecessor carrier rank is fixed to three")
        if not 0.0 < self.fine_step_std < self.coarse_step_std:
            raise ValueError("finite-difference steps must be positive and ordered")
        for name in (
            "derivative_cosine_floor",
            "signed_error_r2_floor",
            "residual_mae_fraction_ceiling",
            "sign_agreement_floor",
            "dominance_fraction_floor",
            "retained_error_floor",
            "removed_error_ceiling",
        ):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")


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
        Path(transport.__file__),
        Path(transport.rank.__file__),
        Path(readout.__file__),
        Path(transport.coupling.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evidence_role(config: CarrierJacobianAxisAuditConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_post_outcome_underpowered_mechanistic_diagnostic"
    )


def _tensor(value: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def _covariance_factors(
    centered: torch.Tensor, ridge: float
) -> tuple[torch.Tensor, torch.Tensor]:
    covariance = centered.T @ centered / float(centered.shape[0] - 1)
    values, vectors = torch.linalg.eigh(covariance)
    floor = values.max().clamp_min(1e-24) * ridge
    values = values.clamp_min(floor)
    whitening = (vectors * values.rsqrt().unsqueeze(0)) @ vectors.T
    coloring = (vectors * values.sqrt().unsqueeze(0)) @ vectors.T
    return whitening, coloring


def canonical_alignment_frame(
    source: torch.Tensor, target: torch.Tensor, ridge: float
) -> dict[str, torch.Tensor]:
    """Reproduce the predecessor map and expose its target canonical frame."""
    source = source.double()
    target = target.double()
    if source.ndim != 2 or source.shape != target.shape:
        raise ValueError("paired coordinates must have equal 2D shape")
    if source.shape[0] <= source.shape[1]:
        raise ValueError("canonical alignment needs more rows than features")
    source_mean = source.mean(0)
    target_mean = target.mean(0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_whitening, _ = _covariance_factors(source_centered, ridge)
    target_whitening, target_coloring = _covariance_factors(
        target_centered, ridge
    )
    source_white = source_centered @ source_whitening
    target_white = target_centered @ target_whitening
    left, singular, right_h = torch.linalg.svd(
        source_white.T @ target_white, full_matrices=False
    )
    target_vectors = right_h.T
    rotation = left @ right_h
    linear = source_whitening @ rotation @ target_coloring
    intercept = target_mean - source_mean @ linear
    return {
        "linear": linear,
        "intercept": intercept,
        "singular_values": singular / float(source.shape[0] - 1),
        "target_whitening": target_whitening,
        "target_coloring": target_coloring,
        "target_vectors": target_vectors,
        "target_axis_rows": target_vectors.T @ target_coloring,
    }


def canonical_axis_components(
    error: torch.Tensor, frame: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Decompose a raw target-coordinate error into canonical-axis components."""
    error = error.double()
    canonical = error @ frame["target_whitening"] @ frame["target_vectors"]
    components = canonical[:, :, None] * frame["target_axis_rows"][None, :, :]
    reconstructed = components.sum(1)
    relative_error = float(
        torch.linalg.vector_norm(reconstructed - error)
        / torch.linalg.vector_norm(error).clamp_min(1e-24)
    )
    return canonical, components, relative_error


def _maximum_map_error(
    frame: Mapping[str, torch.Tensor], stored: Mapping[str, Any]
) -> dict[str, float]:
    device = frame["linear"].device
    return {
        "linear_max_abs": float(
            (frame["linear"] - _tensor(stored["linear"], device)).abs().max()
        ),
        "intercept_max_abs": float(
            (frame["intercept"] - _tensor(stored["intercept"], device))
            .abs()
            .max()
        ),
        "singular_values_max_abs": float(
            (
                frame["singular_values"]
                - _tensor(stored["whitened_cross_singular_values"], device)
            )
            .abs()
            .max()
        ),
    }


def _wrapped_bins(left: torch.Tensor, right: torch.Tensor, bins: int) -> torch.Tensor:
    width = 2.0 * math.pi / bins
    return readout._wrapped_difference(left, right) / width


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(value.double().square().mean()))


def _mae(value: torch.Tensor) -> float:
    return float(value.double().abs().mean())


def linearization_metrics(
    fine_derivative: torch.Tensor,
    coarse_derivative: torch.Tensor,
    predicted_error: torch.Tensor,
    observed_error: torch.Tensor,
    config: CarrierJacobianAxisAuditConfig,
) -> dict[str, Any]:
    """Compute the fixed pair-level local-linearization endpoints."""
    fine = fine_derivative.double().reshape(-1)
    coarse = coarse_derivative.double().reshape(-1)
    cosine = float(
        (fine @ coarse)
        / (
            torch.linalg.vector_norm(fine)
            * torch.linalg.vector_norm(coarse)
        ).clamp_min(1e-24)
    )
    relative_l2 = float(
        torch.linalg.vector_norm(coarse - fine)
        / torch.linalg.vector_norm(fine).clamp_min(1e-24)
    )
    predicted = predicted_error.double().reshape(-1)
    observed = observed_error.double().reshape(-1)
    residual = predicted - observed
    zero_r2 = float(
        1.0
        - residual.square().sum()
        / observed.square().sum().clamp_min(1e-24)
    )
    observed_mae = _mae(observed)
    residual_fraction = _mae(residual) / max(observed_mae, 1e-24)
    mask = observed.abs() >= config.sign_magnitude_floor_bins
    sign_count = int(mask.sum())
    sign_agreement = (
        float((torch.sign(predicted[mask]) == torch.sign(observed[mask])).double().mean())
        if sign_count
        else 0.0
    )
    gates = {
        "derivative_cosine": cosine >= config.derivative_cosine_floor,
        "derivative_relative_l2": relative_l2
        <= config.derivative_relative_l2_ceiling,
        "signed_error_zero_r2": zero_r2 >= config.signed_error_r2_floor,
        "residual_mae_fraction": residual_fraction
        <= config.residual_mae_fraction_ceiling,
        "sign_agreement": sign_agreement >= config.sign_agreement_floor,
    }
    return {
        "coarse_fine_derivative_cosine": cosine,
        "coarse_fine_derivative_relative_l2": relative_l2,
        "signed_error_zero_referenced_r2": zero_r2,
        "observed_paired_error_mae_bins": observed_mae,
        "prediction_residual_mae_bins": _mae(residual),
        "prediction_residual_mae_fraction": residual_fraction,
        "sign_agreement": sign_agreement,
        "sign_evaluation_count": sign_count,
        "gates": gates,
        "adequate": all(gates.values()),
    }


def classify_axis_dominance(
    adequate: bool,
    shared_predicted: torch.Tensor,
    third_predicted: torch.Tensor,
    paired_error: torch.Tensor,
    shared_only_error: torch.Tensor,
    third_only_error: torch.Tensor,
    config: CarrierJacobianAxisAuditConfig,
) -> dict[str, Any]:
    """Apply the preregistered reciprocal shared/third dominance rule."""
    shared_rms = _rms(shared_predicted)
    third_rms = _rms(third_predicted)
    denominator = max(shared_rms + third_rms, 1e-24)
    shared_fraction = shared_rms / denominator
    third_fraction = third_rms / denominator
    paired_mae = _mae(paired_error)
    shared_retention = _mae(shared_only_error) / max(paired_mae, 1e-24)
    third_retention = _mae(third_only_error) / max(paired_mae, 1e-24)
    third_dominant = bool(
        adequate
        and third_fraction >= config.dominance_fraction_floor
        and third_retention >= config.retained_error_floor
        and shared_retention <= config.removed_error_ceiling
    )
    shared_dominant = bool(
        adequate
        and shared_fraction >= config.dominance_fraction_floor
        and shared_retention >= config.retained_error_floor
        and third_retention <= config.removed_error_ceiling
    )
    if not adequate:
        classification = "nonlinear_or_unresolved"
    elif third_dominant:
        classification = "axis_3_dominant"
    elif shared_dominant:
        classification = "axes_1_2_dominant"
    else:
        classification = "mixed"
    return {
        "classification": classification,
        "shared_predicted_rms_bins": shared_rms,
        "third_predicted_rms_bins": third_rms,
        "shared_predicted_rms_fraction": shared_fraction,
        "third_predicted_rms_fraction": third_fraction,
        "paired_error_mae_bins": paired_mae,
        "shared_only_error_mae_bins": _mae(shared_only_error),
        "third_only_error_mae_bins": _mae(third_only_error),
        "shared_only_error_retention": shared_retention,
        "third_only_error_retention": third_retention,
    }


def _load_predecessor(
    config: CarrierJacobianAxisAuditConfig,
) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
    root = Path(config.source_campaign)
    campaign_path = root / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text())
    expected_role = "preregistered_underpowered_mechanistic_evidence"
    if (
        campaign.get("schema_version") != transport.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != transport.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != expected_role
        or campaign.get("aggregates", {}).get("gate_counts")
        != {
            "paired_causal_transport": 0,
            "paired_coordinate_transport": 6,
            "shuffled_specificity": 6,
            "target_control_contract": 4,
        }
        or int(campaign.get("summary", {}).get("trained_models", -1)) != 0
        or int(campaign.get("summary", {}).get("fitted_coordinate_maps", -1))
        != 18
    ):
        raise ValueError(f"invalid causal-transport predecessor {campaign_path}")
    entries = {
        (int(item["source_seed"]), int(item["target_seed"])): item
        for item in campaign.get("results", [])
    }
    expected_pairs = {
        (source, target)
        for source in PRIMARY_SEEDS
        for target in PRIMARY_SEEDS
        if source != target
    }
    if set(entries) != expected_pairs:
        raise ValueError("causal-transport predecessor does not contain six pairs")
    details: dict[tuple[int, int], dict[str, Any]] = {}
    for pair, entry in entries.items():
        result_path = Path(entry["path"])
        if _sha256(result_path) != entry.get("result_sha256"):
            raise ValueError(f"predecessor result hash mismatch for {pair}")
        detail = json.loads(result_path.read_text())
        if (
            detail.get("schema_version") != transport.SCHEMA_VERSION
            or detail.get("hypothesis_id") != transport.HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != expected_role
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or detail.get("implementation_sha256")
            != campaign.get("implementation_sha256")
        ):
            raise ValueError(f"invalid predecessor pair result {pair}")
        details[pair] = detail
    return campaign, details


def _transport_config(
    config: CarrierJacobianAxisAuditConfig,
    predecessor: Mapping[str, Any],
) -> transport.CrossSeedCarrierTransportConfig:
    values = dict(predecessor["configuration"])
    values["device"] = config.device
    values["activation_batch_size"] = config.activation_batch_size
    values["continuation_batch_size"] = config.continuation_batch_size
    values["allow_underpowered"] = config.allow_underpowered
    values["seeds"] = tuple(config.seeds)
    return transport.CrossSeedCarrierTransportConfig(**values)


def _state_from_coordinates(
    cell: Mapping[str, Any], basis: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
    defect = (coordinates.double() @ basis.double()).to(
        dtype=cell["full_defect"].dtype
    )
    return cell["propagated"] + defect.reshape_as(cell["full_defect"])


@torch.no_grad()
def _evaluate_axis_cell(
    target_system: Any,
    task: CircleTaskConfig,
    config: CarrierJacobianAxisAuditConfig,
    transport_config: transport.CrossSeedCarrierTransportConfig,
    source_cell: Mapping[str, Any],
    target_cell: Mapping[str, Any],
    source_basis: torch.Tensor,
    target_basis: torch.Tensor,
    frame: Mapping[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    source_coordinates = transport._coordinates(source_cell, source_basis)
    target_coordinates = transport._coordinates(target_cell, target_basis)
    paired_coordinates = transport.apply_affine(source_coordinates, frame)
    raw_error = paired_coordinates - target_coordinates
    canonical_error, components, reconstruction_error = canonical_axis_components(
        raw_error, frame
    )
    shared_coordinates = target_coordinates + components[:, :2].sum(1)
    third_coordinates = target_coordinates + components[:, 2]
    coordinate_states: dict[str, torch.Tensor] = {
        "direct": target_coordinates,
        "paired": paired_coordinates,
        "shared_only": shared_coordinates,
        "third_only": third_coordinates,
    }
    axis_rows = frame["target_axis_rows"]
    for step_name, step in (
        ("fine", config.fine_step_std),
        ("coarse", config.coarse_step_std),
    ):
        for axis in range(config.carrier_rank):
            coordinate_states[f"{step_name}_axis_{axis + 1}_plus"] = (
                target_coordinates + step * axis_rows[axis]
            )
            coordinate_states[f"{step_name}_axis_{axis + 1}_minus"] = (
                target_coordinates - step * axis_rows[axis]
            )
    state_names = tuple(coordinate_states)
    states = {
        name: _state_from_coordinates(target_cell, target_basis, coordinates)
        for name, coordinates in coordinate_states.items()
    }
    repeated = torch.cat(
        [transport.coupling._repeat_patch(states[name], 2) for name in state_names]
    )
    combined = transport.coupling._continue_tensor(
        target_system,
        target_cell["target_cut"],
        repeated,
        task,
        config.continuation_batch_size,
    )
    per_state = config.orbit_count * 2
    raw_posteriors = dict(zip(state_names, combined.split(per_state)))
    posteriors = {
        name: value.reshape(config.orbit_count, 2, -1)[:, 0]
        for name, value in raw_posteriors.items()
    }
    angles = {name: readout.posterior_moment(value)[0] for name, value in posteriors.items()}
    bins = len(task.answer_token_ids)
    direct_angle = angles["direct"]
    state_errors = {
        name: _wrapped_bins(angles[name], direct_angle, bins)
        for name in ("paired", "shared_only", "third_only")
    }
    derivatives: dict[str, torch.Tensor] = {}
    for step_name, step in (
        ("fine", config.fine_step_std),
        ("coarse", config.coarse_step_std),
    ):
        values = []
        for axis in range(config.carrier_rank):
            plus = angles[f"{step_name}_axis_{axis + 1}_plus"]
            minus = angles[f"{step_name}_axis_{axis + 1}_minus"]
            values.append(_wrapped_bins(plus, minus, bins) / (2.0 * step))
        derivatives[step_name] = torch.stack(values, dim=1)
    predicted_by_axis = derivatives["fine"] * canonical_error
    predicted_total = predicted_by_axis.sum(1)
    predicted_shared = predicted_by_axis[:, :2].sum(1)
    predicted_third = predicted_by_axis[:, 2]
    summary = {
        "cohort": target_cell["cohort"],
        "regime": target_cell["regime"],
        "evaluation_seed": target_cell["evaluation_seed"],
        "axis_reconstruction_relative_error": reconstruction_error,
        "canonical_coordinate_error_rms": [
            _rms(canonical_error[:, axis]) for axis in range(config.carrier_rank)
        ],
        "fine_derivative_rms_bins_per_std": [
            _rms(derivatives["fine"][:, axis])
            for axis in range(config.carrier_rank)
        ],
        "coarse_derivative_rms_bins_per_std": [
            _rms(derivatives["coarse"][:, axis])
            for axis in range(config.carrier_rank)
        ],
        "predicted_effect_rms_bins": [
            _rms(predicted_by_axis[:, axis])
            for axis in range(config.carrier_rank)
        ],
        "signed_errors": {
            name: {"mae_bins": _mae(value), "rms_bins": _rms(value)}
            for name, value in state_errors.items()
        },
        "predicted_total": {
            "mae_bins": _mae(predicted_total),
            "rms_bins": _rms(predicted_total),
        },
    }
    tensors = {
        "fine_derivative": derivatives["fine"].cpu(),
        "coarse_derivative": derivatives["coarse"].cpu(),
        "predicted_total": predicted_total.cpu(),
        "predicted_shared": predicted_shared.cpu(),
        "predicted_third": predicted_third.cpu(),
        "paired_error": state_errors["paired"].cpu(),
        "shared_only_error": state_errors["shared_only"].cpu(),
        "third_only_error": state_errors["third_only"].cpu(),
    }
    return summary, tensors


def _pair_fingerprint(
    config: CarrierJacobianAxisAuditConfig,
    source_seed: int,
    target_seed: int,
    source_campaign_sha256: str,
    predecessor_result_sha256: str,
    source_checkpoint_sha256: str,
    target_checkpoint_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "source_seed": source_seed,
        "target_seed": target_seed,
        "source_campaign_sha256": source_campaign_sha256,
        "predecessor_result_sha256": predecessor_result_sha256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "target_checkpoint_sha256": target_checkpoint_sha256,
        "cohort_seeds": transport.COHORT_SEEDS,
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode()).hexdigest()


def _campaign_is_reusable(
    campaign: Mapping[str, Any],
    config: CarrierJacobianAxisAuditConfig,
    implementation: str,
) -> bool:
    required = len(config.seeds) * (len(config.seeds) - 1)
    if (
        campaign.get("schema_version") != SCHEMA_VERSION
        or campaign.get("hypothesis_id") != HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != _evidence_role(config)
        or campaign.get("implementation_sha256") != implementation
        or campaign.get("configuration") != _json_compatible(asdict(config))
        or int(campaign.get("summary", {}).get("completed", -1)) != required
        or len(campaign.get("results", [])) != required
    ):
        return False
    pairs: set[tuple[int, int]] = set()
    fingerprints: set[str] = set()
    for entry in campaign["results"]:
        path = Path(entry.get("path", ""))
        if not path.is_file() or _sha256(path) != entry.get("result_sha256"):
            return False
        detail = json.loads(path.read_text())
        pair = (int(entry["source_seed"]), int(entry["target_seed"]))
        fingerprint = entry.get("scientific_fingerprint")
        if (
            pair[0] == pair[1]
            or pair in pairs
            or not fingerprint
            or fingerprint in fingerprints
            or detail.get("schema_version") != SCHEMA_VERSION
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != _evidence_role(config)
            or detail.get("implementation_sha256") != implementation
            or detail.get("scientific_fingerprint") != fingerprint
        ):
            return False
        pairs.add(pair)
        fingerprints.add(fingerprint)
    return pairs == {
        (source, target)
        for source in config.seeds
        for target in config.seeds
        if source != target
    }


def _campaign_decision(classifications: Sequence[str], adequate_count: int) -> dict[str, Any]:
    required = len(classifications)
    third_count = classifications.count("axis_3_dominant")
    shared_count = classifications.count("axes_1_2_dominant")
    universal = bool(required == 6 and adequate_count == 6 and third_count >= 5)
    shared_alternative = bool(
        required == 6 and adequate_count == 6 and shared_count >= 5
    )
    if universal:
        conclusion = "supported_universal_2d_base_plus_local_scalar_correction"
    elif shared_alternative:
        conclusion = "supported_shared_axes_causally_misoriented"
    elif adequate_count < required:
        conclusion = "unresolved_local_linearization_inadequate"
    else:
        conclusion = "mixed_checkpoint_stratified_carrier_atlas"
    return {
        "universal_2d_base_plus_local_scalar_supported": universal,
        "shared_axes_causally_misoriented_supported": shared_alternative,
        "adequate_pair_count": adequate_count,
        "axis_3_dominant_count": third_count,
        "axes_1_2_dominant_count": shared_count,
        "mixed_count": classifications.count("mixed"),
        "nonlinear_or_unresolved_count": classifications.count(
            "nonlinear_or_unresolved"
        ),
        "required_pair_count": required,
        "conclusion": conclusion,
    }


@torch.no_grad()
def run_campaign(
    config: CarrierJacobianAxisAuditConfig, output: Path
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    implementation = _implementation_digest()
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text())
        if _campaign_is_reusable(existing, config, implementation):
            print("aggregate already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(
            f"incompatible completed aggregate {campaign_path}; use a new root"
        )

    predecessor, predecessor_pairs = _load_predecessor(config)
    predecessor_path = Path(config.source_campaign) / "campaign_results.json"
    predecessor_sha256 = _sha256(predecessor_path)
    transport_config = _transport_config(config, predecessor)
    rank_config = transport._rank_config(transport_config)
    bridge = transport.rank._bridge_config(rank_config)
    task = CircleTaskConfig()
    datasets = {
        cohort: {
            regime: transport.rank.deck.generate_exact_orbits(
                task,
                k=2,
                orbit_count=config.orbit_count,
                seed=transport.COHORT_SEEDS[cohort][regime],
                regime=regime,
            )
            for regime in REGIMES
        }
        for cohort in transport.COHORT_SEEDS
    }

    seed_records: dict[int, dict[str, Any]] = {}
    for seed in config.seeds:
        system, provenance = transport.rank.deck.load_source(
            task, bridge, 2, seed, device
        )
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
            for cohort in transport.COHORT_SEEDS
        }
        seed_records[seed] = {
            "system": system,
            "provenance": {
                **provenance,
                "character_result": str(frozen_path),
                "character_result_sha256": _sha256(frozen_path),
            },
            "basis": basis,
            "basis_summary": basis_summary,
            "cells": cells,
        }
        print(f"seed {seed} frozen carrier reconstructed", flush=True)
        if _implementation_digest() != implementation:
            raise RuntimeError("analysis implementation changed during campaign")

    pair_results: list[dict[str, Any]] = []
    reused = 0
    for source_seed in config.seeds:
        for target_seed in config.seeds:
            if source_seed == target_seed:
                continue
            pair = (source_seed, target_seed)
            predecessor_pair = predecessor_pairs[pair]
            predecessor_entry = next(
                item
                for item in predecessor["results"]
                if (int(item["source_seed"]), int(item["target_seed"])) == pair
            )
            source_record = seed_records[source_seed]
            target_record = seed_records[target_seed]
            for side, record in (("source", source_record), ("target", target_record)):
                predecessor_provenance = predecessor_pair["provenance"][side]
                if (
                    record["provenance"]["checkpoint_sha256"]
                    != predecessor_provenance["checkpoint_sha256"]
                    or record["provenance"]["frontend_checkpoint_sha256"]
                    != predecessor_provenance["frontend_checkpoint_sha256"]
                    or record["provenance"]["character_result_sha256"]
                    != predecessor_provenance["character_result_sha256"]
                ):
                    raise ValueError(f"{side} frozen identity mismatch for pair {pair}")
            source_fit = torch.cat(
                [
                    transport._coordinates(
                        source_record["cells"]["alignment_fit"][regime],
                        source_record["basis"],
                    )
                    for regime in REGIMES
                ]
            )
            target_fit = torch.cat(
                [
                    transport._coordinates(
                        target_record["cells"]["alignment_fit"][regime],
                        target_record["basis"],
                    )
                    for regime in REGIMES
                ]
            )
            frame = canonical_alignment_frame(
                source_fit, target_fit, transport_config.whitening_ridge
            )
            stored_map = predecessor_pair["alignment_fit"]["paired_map"]
            reproduction = _maximum_map_error(frame, stored_map)
            if max(reproduction.values()) > config.canonical_reproduction_tolerance:
                raise ValueError(f"canonical map reproduction failed for pair {pair}")
            if float(frame["singular_values"][:2].min()) <= 0.987:
                raise ValueError(f"shared canonical axes contract failed for pair {pair}")
            if float(frame["singular_values"][2]) >= 0.282:
                raise ValueError(f"weak third canonical axis contract failed for pair {pair}")

            cell_summaries: list[dict[str, Any]] = []
            cell_tensors: list[dict[str, torch.Tensor]] = []
            for cohort in HELDOUT_COHORTS:
                for regime in REGIMES:
                    summary, tensors = _evaluate_axis_cell(
                        target_record["system"],
                        task,
                        config,
                        transport_config,
                        source_record["cells"][cohort][regime],
                        target_record["cells"][cohort][regime],
                        source_record["basis"],
                        target_record["basis"],
                        frame,
                    )
                    if (
                        summary["axis_reconstruction_relative_error"]
                        > config.axis_reconstruction_tolerance
                    ):
                        raise ValueError(f"axis reconstruction failed for pair {pair}")
                    cell_summaries.append(summary)
                    cell_tensors.append(tensors)
            pooled = {
                name: torch.cat([cell[name] for cell in cell_tensors], dim=0)
                for name in cell_tensors[0]
            }
            local = linearization_metrics(
                pooled["fine_derivative"],
                pooled["coarse_derivative"],
                pooled["predicted_total"],
                pooled["paired_error"],
                config,
            )
            attribution = classify_axis_dominance(
                local["adequate"],
                pooled["predicted_shared"],
                pooled["predicted_third"],
                pooled["paired_error"],
                pooled["shared_only_error"],
                pooled["third_only_error"],
                config,
            )
            fingerprint = _pair_fingerprint(
                config,
                source_seed,
                target_seed,
                predecessor_sha256,
                predecessor_entry["result_sha256"],
                source_record["provenance"]["checkpoint_sha256"],
                target_record["provenance"]["checkpoint_sha256"],
            )
            result_path = (
                output
                / "runs"
                / f"source_{source_seed}_target_{target_seed}"
                / "result.json"
            )
            result = {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "experiment_id": (
                    f"tinyllm-c2-carrier-jacobian-{source_seed}-to-{target_seed}"
                ),
                "status": "completed",
                "evidence_role": _evidence_role(config),
                "completed_at": _utc_now(),
                "source_seed": source_seed,
                "target_seed": target_seed,
                "configuration": asdict(config),
                "scientific_fingerprint": fingerprint,
                "implementation_sha256": implementation,
                "provenance": {
                    "source_campaign": str(predecessor_path),
                    "source_campaign_sha256": predecessor_sha256,
                    "source_pair_result": predecessor_entry["path"],
                    "source_pair_result_sha256": predecessor_entry["result_sha256"],
                    "source": source_record["provenance"],
                    "target": target_record["provenance"],
                },
                "source_basis": source_record["basis_summary"],
                "target_basis": target_record["basis_summary"],
                "canonical_frame": {
                    "singular_values": frame["singular_values"].cpu().tolist(),
                    "predecessor_reproduction_max_abs": reproduction,
                },
                "heldout_cells": cell_summaries,
                "local_linearization": local,
                "axis_attribution": attribution,
                "primary_metric": float(
                    attribution["classification"] == "axis_3_dominant"
                ),
                "analysis_seconds": time.perf_counter() - started,
                "artifacts": {"result": str(result_path)},
            }
            if result_path.is_file():
                existing = json.loads(result_path.read_text())
                if (
                    existing.get("schema_version") == SCHEMA_VERSION
                    and existing.get("hypothesis_id") == HYPOTHESIS_ID
                    and existing.get("status") == "completed"
                    and existing.get("evidence_role") == _evidence_role(config)
                    and existing.get("implementation_sha256") == implementation
                    and existing.get("scientific_fingerprint") == fingerprint
                    and int(existing.get("source_seed", -1)) == source_seed
                    and int(existing.get("target_seed", -1)) == target_seed
                ):
                    result = existing
                    reused += 1
                    print(f"resuming {existing['experiment_id']}", flush=True)
                else:
                    raise ValueError(
                        f"incompatible completed result {result_path}; use a new root"
                    )
            else:
                _write_json(result_path, result)
            pair_results.append(result)
            print(
                f"source {source_seed} -> target {target_seed}: "
                f"{result['axis_attribution']['classification']}",
                flush=True,
            )
            if _implementation_digest() != implementation:
                raise RuntimeError("analysis implementation changed during campaign")

    classifications = [
        result["axis_attribution"]["classification"] for result in pair_results
    ]
    adequate_count = sum(
        bool(result["local_linearization"]["adequate"]) for result in pair_results
    )
    decision = _campaign_decision(classifications, adequate_count)
    required = len(config.seeds) * (len(config.seeds) - 1)
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "implementation_sha256": implementation,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
        },
        "provenance": {
            "source_campaign": str(predecessor_path),
            "source_campaign_sha256": predecessor_sha256,
            "source_implementation_sha256": predecessor["implementation_sha256"],
        },
        "summary": {
            "requested": required,
            "scheduled": required - reused,
            "completed": len(pair_results),
            "failed": 0,
            "excluded": 0,
            "retries": 0,
            "reused": reused,
            "trained_models": 0,
            "fitted_predictive_observers": 0,
            "fitted_coordinate_maps": 0,
            "reproduced_predecessor_maps": required,
            "finite_difference_continuations": required
            * len(HELDOUT_COHORTS)
            * len(REGIMES)
            * config.carrier_rank
            * 4,
        },
        "aggregates": {
            **decision,
            "per_pair": {
                f"{result['source_seed']}->{result['target_seed']}": {
                    "classification": result["axis_attribution"]["classification"],
                    "adequate": result["local_linearization"]["adequate"],
                    "signed_error_zero_referenced_r2": result[
                        "local_linearization"
                    ]["signed_error_zero_referenced_r2"],
                    "shared_only_error_retention": result["axis_attribution"][
                        "shared_only_error_retention"
                    ],
                    "third_only_error_retention": result["axis_attribution"][
                        "third_only_error_retention"
                    ],
                }
                for result in pair_results
            },
        },
        "results": [
            {
                "experiment_id": result["experiment_id"],
                "source_seed": result["source_seed"],
                "target_seed": result["target_seed"],
                "scientific_fingerprint": result["scientific_fingerprint"],
                "path": result["artifacts"]["result"],
                "result_sha256": _sha256(Path(result["artifacts"]["result"])),
                "primary_metric": result["primary_metric"],
                "classification": result["axis_attribution"]["classification"],
            }
            for result in pair_results
        ],
        "method_boundaries": [
            "The audit reuses the previously inspected six transport failures and held-out cohorts.",
            "The circular-moment Jacobian is conditioned on the frozen answer-token decoder.",
            "Canonical axes may rotate under near-degenerate singular values.",
            "All residual writes are off-manifold interventions into a frozen continuation.",
            "Three selected checkpoints constitute underpowered mechanistic evidence.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "source_*_target_*" / "result.json"),
        },
        "analysis_seconds": time.perf_counter() - started,
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
            "data/experiments/tinyllm_carrier_jacobian_axis_audit/"
            "20260806_d6_preregistered_diagnostic"
        ),
    )
    parser.add_argument(
        "--source-campaign", default=CarrierJacobianAxisAuditConfig.source_campaign
    )
    parser.add_argument("--seeds", default=",".join(map(str, PRIMARY_SEEDS)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = CarrierJacobianAxisAuditConfig(
        source_campaign=args.source_campaign,
        seeds=_ints(args.seeds),
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
