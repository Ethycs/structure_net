#!/usr/bin/env python3
"""Causally split a TinyLLM writer residual into local task tangent and kernel."""

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

import experiments.structure_net.tinyllm_carrier_jacobian_axis_audit as axis_audit
import experiments.structure_net.tinyllm_fixed_semantic_gauge_writer as fixed
import experiments.structure_net.tinyllm_frozen_writer_capacity as predecessor
import experiments.structure_net.tinyllm_cross_seed_causal_carrier_transport as transport
from experiments.structure_net import tinyllm_continuous_readout_calibration as readout
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-c2-local-continuation-tangent.v1"
HYPOTHESIS_ID = "tinyllm-c2-local-continuation-tangent-v1"
PREDECESSOR_CAMPAIGN_SHA256 = (
    "7ab6079d56bc8dc78cc5e1f4011c581b8f507bcd8ec20573bbf8b11227df8d1b"
)
PREDECESSOR_IMPLEMENTATION_SHA256 = (
    "d53edaedd49ae553af9f8393d92254664239e5100246ac0fd3a06cb420ca80ed"
)
WRITER_NAME = "context_m04"


@dataclass(frozen=True)
class LocalContinuationTangentConfig:
    predecessor_root: str = (
        "data/experiments/tinyllm_frozen_writer_capacity/"
        "20260807_d6_preregistered_diagnostic"
    )
    seeds: tuple[int, ...] = fixed.PRIMARY_SEEDS
    orbit_count: int = 64
    carrier_rank: int = 3
    writer_name: str = WRITER_NAME
    fine_step_std: float = 0.025
    coarse_step_std: float = 0.05
    context_scale_floor: float = 1e-8
    coordinate_scale_floor: float = 1e-8
    gradient_norm_floor: float = 1e-8
    replay_tolerance: float = 1e-6
    reconstruction_tolerance: float = 1e-8
    derivative_cosine_floor: float = 0.98
    derivative_relative_l2_ceiling: float = 0.15
    signed_error_r2_floor: float = 0.50
    residual_mae_fraction_ceiling: float = 0.50
    sign_agreement_floor: float = 0.75
    sign_magnitude_floor_bins: float = 0.01
    specificity_margin_bins: float = 0.125
    material_kernel_fraction: float = 0.10
    activation_batch_size: int = 256
    continuation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != fixed.PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("primary diagnostic seeds are fixed to 7,29,53")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.orbit_count != 64:
            raise ValueError("the frozen predecessor contract requires 64 exact orbits")
        if self.carrier_rank != 3:
            raise ValueError("the frozen predecessor carrier rank is fixed to three")
        if self.writer_name != WRITER_NAME:
            raise ValueError("the preregistered writer arm is context_m04")
        if not 0.0 < self.fine_step_std < self.coarse_step_std:
            raise ValueError("finite-difference steps must be positive and ordered")
        if min(
            self.context_scale_floor,
            self.coordinate_scale_floor,
            self.gradient_norm_floor,
            self.replay_tolerance,
            self.reconstruction_tolerance,
            self.specificity_margin_bins,
        ) <= 0.0:
            raise ValueError("numerical thresholds must be positive")
        for name in (
            "derivative_cosine_floor",
            "derivative_relative_l2_ceiling",
            "signed_error_r2_floor",
            "residual_mae_fraction_ceiling",
            "sign_agreement_floor",
            "material_kernel_fraction",
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
        Path(predecessor.__file__),
        Path(fixed.__file__),
        Path(transport.__file__),
        Path(transport.rank.__file__),
        Path(transport.coupling.__file__),
        Path(readout.__file__),
        Path(axis_audit.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evidence_role(config: LocalContinuationTangentConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_post_outcome_underpowered_mechanistic_diagnostic"
    )


def _tensor_digest(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load_predecessor(
    config: LocalContinuationTangentConfig,
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
        or campaign.get("aggregates", {}).get("conclusion")
        != "small_writer_insufficient"
    ):
        raise ValueError(f"invalid predecessor campaign {campaign_path}")
    details: dict[int, tuple[dict[str, Any], Path]] = {}
    for entry in campaign["results"]:
        path = Path(entry["path"])
        detail = json.loads(path.read_text())
        if (
            _sha256(path) != entry.get("result_sha256")
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or detail.get("classification") != "small_writer_insufficient"
            or WRITER_NAME
            not in detail.get("alignment_fit", {}).get("mappings", {})
        ):
            raise ValueError(f"invalid predecessor result {path}")
        details[int(detail["seed"])] = (detail, path)
    if not set(config.seeds).issubset(details):
        raise ValueError("predecessor lacks a requested checkpoint")
    return campaign, campaign_path, details


def _transport_config(
    config: LocalContinuationTangentConfig,
) -> transport.CrossSeedCarrierTransportConfig:
    base = fixed.FixedSemanticGaugeWriterConfig(
        seeds=config.seeds,
        orbit_count=config.orbit_count,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )
    value = fixed._transport_config(base)
    return transport.CrossSeedCarrierTransportConfig(
        **{
            **asdict(value),
            "activation_batch_size": config.activation_batch_size,
            "continuation_batch_size": config.continuation_batch_size,
        }
    )


def _stored_mapping(
    prior: Mapping[str, Any], name: str, device: torch.device
) -> dict[str, torch.Tensor]:
    record = prior["alignment_fit"]["mappings"][name]
    return {
        "linear": torch.as_tensor(record["linear"], dtype=torch.float64, device=device),
        "intercept": torch.as_tensor(
            record["intercept"], dtype=torch.float64, device=device
        ),
    }


def decompose_task_tangent(
    standardized_residual: torch.Tensor,
    gradient: torch.Tensor,
    gradient_norm_floor: float,
) -> dict[str, torch.Tensor | float | bool]:
    """Project a standardized coordinate residual onto a scalar task gradient."""
    residual = standardized_residual.double()
    gradient = gradient.double()
    if residual.ndim != 2 or residual.shape != gradient.shape:
        raise ValueError("residual and gradient must have equal observation-by-axis shape")
    squared_norm = gradient.square().sum(1, keepdim=True)
    nondegenerate = bool(torch.sqrt(squared_norm).min() > gradient_norm_floor)
    coefficient = (residual * gradient).sum(1, keepdim=True) / squared_norm.clamp_min(
        gradient_norm_floor**2
    )
    tangent = coefficient * gradient
    kernel = residual - tangent
    reconstruction_error = float(
        torch.linalg.vector_norm(tangent + kernel - residual)
        / torch.linalg.vector_norm(residual).clamp_min(1e-24)
    )
    orthogonality_error = float(
        (
            (gradient * kernel).sum(1).abs()
            / (
                torch.linalg.vector_norm(gradient, dim=1)
                * torch.linalg.vector_norm(residual, dim=1)
            ).clamp_min(1e-24)
        ).max()
    )
    return {
        "tangent": tangent,
        "kernel": kernel,
        "minimum_gradient_norm": float(torch.sqrt(squared_norm).min()),
        "median_gradient_norm": float(torch.sqrt(squared_norm).median()),
        "nondegenerate": nondegenerate,
        "reconstruction_relative_error": reconstruction_error,
        "kernel_orthogonality_relative_error": orthogonality_error,
    }


def matched_random_directions(
    norms: torch.Tensor,
    axes: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Return deterministic isotropic directions with exact row-wise norms."""
    if norms.ndim != 1 or axes < 1:
        raise ValueError("norms must be one-dimensional and axes positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn((len(norms), axes), generator=generator, dtype=torch.float64)
    raw /= torch.linalg.vector_norm(raw, dim=1, keepdim=True).clamp_min(1e-24)
    return raw.to(device) * norms.double().to(device)[:, None]


def _state_from_coordinates(
    cell: Mapping[str, Any], basis: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
    flat = (coordinates.double() @ basis.double()).to(
        dtype=cell["full_defect"].dtype
    )
    return cell["propagated"] + flat.reshape_as(cell["full_defect"])


def _continue_states(
    system: Any,
    task: CircleTaskConfig,
    config: LocalContinuationTangentConfig,
    target_cut: str,
    states: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    names = tuple(states)
    repeated = torch.cat(
        [transport.coupling._repeat_patch(states[name], 2) for name in names], dim=0
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
        name: posterior.reshape(config.orbit_count, 2, -1)[:, 0]
        for name, posterior in raw.items()
    }


def _wrapped_bins(left: torch.Tensor, right: torch.Tensor, bins: int) -> torch.Tensor:
    return readout._wrapped_difference(left, right) / (2.0 * math.pi / bins)


def _state_records(
    posteriors: Mapping[str, torch.Tensor],
    cell: Mapping[str, Any],
    task: CircleTaskConfig,
    transport_config: transport.CrossSeedCarrierTransportConfig,
) -> dict[str, dict[str, Any]]:
    per_state = transport_config.orbit_count * 2
    exact = posteriors["exact"]
    zero = posteriors["zero"]
    exact_repeated = exact[:, None].expand(-1, 2, -1).reshape(per_state, -1)
    exact_diagnostics = transport.coupling._diagnostics(
        exact_repeated, cell["dataset"]
    )
    readout_config = transport._readout_config(transport_config)
    records = {}
    for name, posterior in posteriors.items():
        repeated = posterior[:, None].expand(-1, 2, -1).reshape(per_state, -1)
        diagnostics = transport.coupling._diagnostics(repeated, cell["dataset"])
        continuous = readout.continuous_metrics(
            posterior,
            exact,
            diagnostics,
            exact_diagnostics,
            readout_config,
        )
        records[name] = {
            "continuous": continuous,
            "diagnostics": diagnostics,
            "fisher_effect": transport.rank.heads._effect_preservation(
                posterior,
                exact,
                zero,
                transport_config.task_effect_floor,
            ),
        }
    return records


def _control_seed(seed: int, cohort: str, regime: str, stream: int) -> int:
    return (
        8_000_000
        + seed * 1009
        + fixed.HELDOUT_COHORTS.index(cohort) * 9176
        + fixed.REGIMES.index(regime) * 65537
        + stream * 104729
    )


@torch.no_grad()
def _evaluate_cell(
    system: Any,
    task: CircleTaskConfig,
    config: LocalContinuationTangentConfig,
    transport_config: transport.CrossSeedCarrierTransportConfig,
    cell: Mapping[str, Any],
    basis: torch.Tensor,
    context_model: Mapping[str, torch.Tensor],
    coordinate_scale: torch.Tensor,
    mapping: Mapping[str, torch.Tensor],
    theta: torch.Tensor,
    seed: int,
    prior_cell: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    target_coordinates = transport._coordinates(cell, basis)
    context = predecessor.apply_invariant_context(cell["propagated"], context_model)
    features = predecessor.context_features(theta, context, order=4)
    predicted_coordinates = transport.apply_affine(features, mapping)
    standardized_residual = (
        target_coordinates - predicted_coordinates
    ) / coordinate_scale

    coordinate_states: dict[str, torch.Tensor] = {
        "zero": cell["propagated"],
        "exact": cell["exact"],
        "direct_rank3": _state_from_coordinates(cell, basis, target_coordinates),
        "predicted": _state_from_coordinates(cell, basis, predicted_coordinates),
    }
    for step_name, step in (
        ("fine", config.fine_step_std),
        ("coarse", config.coarse_step_std),
    ):
        for coordinate in range(config.carrier_rank):
            delta = torch.zeros_like(predicted_coordinates)
            delta[:, coordinate] = step * coordinate_scale[coordinate]
            coordinate_states[f"{step_name}_axis_{coordinate + 1}_plus"] = (
                _state_from_coordinates(cell, basis, predicted_coordinates + delta)
            )
            coordinate_states[f"{step_name}_axis_{coordinate + 1}_minus"] = (
                _state_from_coordinates(cell, basis, predicted_coordinates - delta)
            )

    first_posteriors = _continue_states(
        system, task, config, cell["target_cut"], coordinate_states
    )
    angles = {
        name: readout.posterior_moment(posterior)[0]
        for name, posterior in first_posteriors.items()
    }
    bins = len(task.answer_token_ids)
    derivatives = {}
    for step_name, step in (
        ("fine", config.fine_step_std),
        ("coarse", config.coarse_step_std),
    ):
        derivative = []
        for coordinate in range(config.carrier_rank):
            plus = angles[f"{step_name}_axis_{coordinate + 1}_plus"]
            minus = angles[f"{step_name}_axis_{coordinate + 1}_minus"]
            derivative.append(_wrapped_bins(plus, minus, bins) / (2.0 * step))
        derivatives[step_name] = torch.stack(derivative, dim=1)

    decomposition = decompose_task_tangent(
        standardized_residual,
        derivatives["fine"],
        config.gradient_norm_floor,
    )
    tangent = decomposition["tangent"]
    kernel = decomposition["kernel"]
    assert isinstance(tangent, torch.Tensor)
    assert isinstance(kernel, torch.Tensor)
    tangent_norm = torch.linalg.vector_norm(tangent, dim=1)
    kernel_norm = torch.linalg.vector_norm(kernel, dim=1)
    random_tangent = matched_random_directions(
        tangent_norm,
        config.carrier_rank,
        _control_seed(seed, cell["cohort"], cell["regime"], 1),
        basis.device,
    )
    random_kernel = matched_random_directions(
        kernel_norm,
        config.carrier_rank,
        _control_seed(seed, cell["cohort"], cell["regime"], 2),
        basis.device,
    )
    corrections = {
        "tangent": tangent,
        "kernel": kernel,
        "full_residual": tangent + kernel,
        "random_tangent": random_tangent,
        "random_kernel": random_kernel,
    }
    second_states = {
        name: _state_from_coordinates(
            cell,
            basis,
            predicted_coordinates + standardized * coordinate_scale,
        )
        for name, standardized in corrections.items()
    }
    second_posteriors = _continue_states(
        system, task, config, cell["target_cut"], second_states
    )
    posteriors = {**first_posteriors, **second_posteriors}
    retained = {
        name: posteriors[name]
        for name in (
            "zero",
            "exact",
            "direct_rank3",
            "predicted",
            "tangent",
            "kernel",
            "full_residual",
            "random_tangent",
            "random_kernel",
        )
    }
    state_records = _state_records(retained, cell, task, transport_config)
    retained_angles = {
        name: readout.posterior_moment(posterior)[0]
        for name, posterior in retained.items()
    }

    replay = max(
        predecessor.predecessor._numeric_max_difference(
            transport.coordinate_metrics(predicted_coordinates, target_coordinates),
            prior_cell["coordinate_metrics"][config.writer_name],
        ),
        *(
            predecessor.predecessor._numeric_max_difference(
                state_records[current]["continuous"],
                prior_cell["states"][stored]["continuous"],
            )
            for current, stored in (
                ("zero", "zero"),
                ("exact", "exact"),
                ("direct_rank3", "direct_rank3"),
                ("predicted", config.writer_name),
            )
        ),
    )
    state_reconstruction_error = float(
        torch.linalg.vector_norm(second_states["full_residual"] - coordinate_states["direct_rank3"])
        / torch.linalg.vector_norm(coordinate_states["direct_rank3"]).clamp_min(1e-24)
    )
    full_replay = predecessor.predecessor._numeric_max_difference(
        state_records["full_residual"]["continuous"],
        state_records["direct_rank3"]["continuous"],
    )
    observed_error = _wrapped_bins(
        retained_angles["direct_rank3"], retained_angles["predicted"], bins
    )
    predicted_error = (derivatives["fine"] * standardized_residual).sum(1)
    kernel_change = _wrapped_bins(
        retained_angles["kernel"], retained_angles["predicted"], bins
    )
    writer_gap = observed_error

    summary = {
        "cohort": cell["cohort"],
        "regime": cell["regime"],
        "evaluation_seed": cell["evaluation_seed"],
        "coordinate_metrics": transport.coordinate_metrics(
            predicted_coordinates, target_coordinates
        ),
        "states": state_records,
        "predecessor_replay_maximum_absolute_error": replay,
        "full_residual_replay_maximum_absolute_error": full_replay,
        "full_residual_state_relative_error": state_reconstruction_error,
        "local_chart": {
            "minimum_gradient_norm": decomposition["minimum_gradient_norm"],
            "median_gradient_norm": decomposition["median_gradient_norm"],
            "gradient_nondegenerate": decomposition["nondegenerate"],
            "reconstruction_relative_error": decomposition[
                "reconstruction_relative_error"
            ],
            "kernel_orthogonality_relative_error": decomposition[
                "kernel_orthogonality_relative_error"
            ],
            "mean_standardized_residual_norm": float(
                torch.linalg.vector_norm(standardized_residual, dim=1).mean()
            ),
            "mean_tangent_norm": float(tangent_norm.mean()),
            "mean_kernel_norm": float(kernel_norm.mean()),
        },
        "signed_error": {
            "observed_mae_bins": float(observed_error.abs().mean()),
            "linear_prediction_mae_bins": float(predicted_error.abs().mean()),
            "kernel_change_mae_bins": float(kernel_change.abs().mean()),
        },
    }
    tensors = {
        "fine_derivative": derivatives["fine"].cpu(),
        "coarse_derivative": derivatives["coarse"].cpu(),
        "predicted_error": predicted_error.cpu(),
        "observed_error": observed_error.cpu(),
        "kernel_change_abs": kernel_change.abs().cpu(),
        "writer_gap_abs": writer_gap.abs().cpu(),
    }
    return summary, tensors


def classify_checkpoint(
    *,
    valid: bool,
    local_adequate: bool,
    tangent_checkpoint_pass: bool,
    tangent_specificity: bool,
    kernel_change_fraction: float,
    material_kernel_fraction: float,
) -> str:
    if not valid:
        return "invalid"
    if local_adequate and tangent_checkpoint_pass and tangent_specificity:
        return "local_task_tangent_sufficient"
    if not local_adequate:
        return "local_linearization_inadequate"
    if kernel_change_fraction >= material_kernel_fraction:
        return "nominal_kernel_causally_active"
    return "tangent_kernel_interaction_or_endpoint_curvature"


def _fingerprint(
    config: LocalContinuationTangentConfig,
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


def _campaign_is_reusable(
    value: Mapping[str, Any],
    config: LocalContinuationTangentConfig,
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


@torch.no_grad()
def run_campaign(
    config: LocalContinuationTangentConfig, output: Path
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
    predecessor_campaign, predecessor_path, predecessor_details = _load_predecessor(
        config
    )
    transport_config = _transport_config(config)
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
    reused = 0
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
        coordinate_scale = fit_coordinates.std(0, unbiased=False)
        fit_propagated = torch.cat(
            [cells["alignment_fit"][regime]["propagated"] for regime in fixed.REGIMES]
        )
        context_model = predecessor.fit_invariant_context(
            fit_propagated, config.carrier_rank, config.context_scale_floor
        )
        context_summary = predecessor._context_summary(context_model)
        context_replay = bool(
            context_summary["model_sha256"]
            == prior["invariant_context"]["model_sha256"]
            and predecessor.predecessor._numeric_max_difference(
                context_summary, prior["invariant_context"]
            )
            <= config.replay_tolerance
        )
        coordinate_scale_contract = bool(
            torch.isfinite(coordinate_scale).all()
            and float(coordinate_scale.min()) > config.coordinate_scale_floor
        )
        mapping = _stored_mapping(prior, config.writer_name, device)
        mapping_digest = _tensor_digest(mapping["linear"], mapping["intercept"])
        prior_by_cell = {
            (cell["cohort"], cell["regime"]): cell
            for cell in prior["heldout_cells"]
        }

        heldout = []
        tensors = []
        for cohort in fixed.HELDOUT_COHORTS:
            for regime in fixed.REGIMES:
                dataset = datasets[cohort][regime]
                theta = (
                    dataset.quotient_phase.to(device)
                    .reshape(config.orbit_count, 2)[:, 0]
                    .double()
                )
                summary, values = _evaluate_cell(
                    system,
                    task,
                    config,
                    transport_config,
                    cells[cohort][regime],
                    basis,
                    context_model,
                    coordinate_scale,
                    mapping,
                    theta,
                    seed,
                    prior_by_cell[(cohort, regime)],
                )
                heldout.append(summary)
                tensors.append(values)

        pooled = {
            name: torch.cat([cell[name] for cell in tensors], dim=0)
            for name in tensors[0]
        }
        local = axis_audit.linearization_metrics(
            pooled["fine_derivative"],
            pooled["coarse_derivative"],
            pooled["predicted_error"],
            pooled["observed_error"],
            config,
        )
        kernel_denominator = float(pooled["writer_gap_abs"].sum())
        kernel_change_fraction = float(
            pooled["kernel_change_abs"].sum()
            / pooled["writer_gap_abs"].sum().clamp_min(1e-24)
        )
        maximum_replay = max(
            cell["predecessor_replay_maximum_absolute_error"] for cell in heldout
        )
        maximum_full_replay = max(
            cell["full_residual_replay_maximum_absolute_error"] for cell in heldout
        )
        maximum_state_reconstruction = max(
            cell["full_residual_state_relative_error"] for cell in heldout
        )
        numerical_chart = bool(
            context_replay
            and coordinate_scale_contract
            and kernel_denominator > 1e-8
            and all(
                cell["local_chart"]["gradient_nondegenerate"]
                and cell["local_chart"]["reconstruction_relative_error"]
                <= config.reconstruction_tolerance
                and cell["local_chart"]["kernel_orthogonality_relative_error"]
                <= config.reconstruction_tolerance
                for cell in heldout
            )
        )
        target_controls = all(
            not cell["states"]["zero"]["continuous"]["continuous_pass"]
            and cell["states"]["exact"]["continuous"]["continuous_pass"]
            and cell["states"]["direct_rank3"]["continuous"]["continuous_pass"]
            and cell["states"]["full_residual"]["continuous"]["continuous_pass"]
            for cell in heldout
        )
        replay_contract = bool(
            maximum_replay <= config.replay_tolerance
            and maximum_full_replay <= config.replay_tolerance
            and maximum_state_reconstruction <= config.reconstruction_tolerance
        )
        tangent_checkpoint_pass = all(
            cell["states"]["tangent"]["continuous"]["continuous_pass"]
            for cell in heldout
        )
        random_any_failure = any(
            not cell["states"]["random_tangent"]["continuous"]["continuous_pass"]
            for cell in heldout
        )
        tangent_mean = sum(
            cell["states"]["tangent"]["continuous"]["mean_moment_shift_bins"]
            for cell in heldout
        ) / len(heldout)
        random_tangent_mean = sum(
            cell["states"]["random_tangent"]["continuous"][
                "mean_moment_shift_bins"
            ]
            for cell in heldout
        ) / len(heldout)
        tangent_specificity = bool(
            random_any_failure
            and tangent_mean + config.specificity_margin_bins
            <= random_tangent_mean
        )
        valid = bool(numerical_chart and target_controls and replay_contract)
        classification = classify_checkpoint(
            valid=valid,
            local_adequate=bool(local["adequate"]),
            tangent_checkpoint_pass=tangent_checkpoint_pass,
            tangent_specificity=tangent_specificity,
            kernel_change_fraction=kernel_change_fraction,
            material_kernel_fraction=config.material_kernel_fraction,
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
            "experiment_id": f"tinyllm-c2-local-continuation-tangent-seed{seed}",
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
            "writer": {
                "name": config.writer_name,
                "mapping_sha256": mapping_digest,
                "context": context_summary,
                "coordinate_scale": coordinate_scale.cpu().tolist(),
                "minimum_coordinate_scale": float(coordinate_scale.min()),
            },
            "heldout_cells": heldout,
            "local_linearization": local,
            "kernel_analysis": {
                "kernel_change_fraction": kernel_change_fraction,
                "material_threshold": config.material_kernel_fraction,
                "material_kernel_effect": bool(
                    kernel_change_fraction >= config.material_kernel_fraction
                ),
            },
            "gates": {
                "predecessor_replay_contract": replay_contract,
                "maximum_predecessor_replay_error": maximum_replay,
                "maximum_full_residual_replay_error": maximum_full_replay,
                "maximum_full_residual_state_relative_error": maximum_state_reconstruction,
                "context_replay_contract": context_replay,
                "coordinate_scale_contract": coordinate_scale_contract,
                "numerical_chart_contract": numerical_chart,
                "target_control_contract": target_controls,
                "local_linearization_adequate": bool(local["adequate"]),
                "tangent_checkpoint_pass": tangent_checkpoint_pass,
                "random_tangent_any_failure": random_any_failure,
                "tangent_aggregate_mean_shift_bins": tangent_mean,
                "random_tangent_aggregate_mean_shift_bins": random_tangent_mean,
                "tangent_specificity": tangent_specificity,
                "primary_checkpoint_pass": bool(
                    classification == "local_task_tangent_sufficient"
                ),
            },
            "classification": classification,
            "primary_metric": float(
                classification == "local_task_tangent_sufficient"
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
                and int(existing.get("seed", -1)) == seed
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
        results.append(result)
        print(f"seed {seed}: {classification}", flush=True)
        if _implementation_digest() != implementation:
            raise RuntimeError("analysis implementation changed during campaign")

    classifications = [result["classification"] for result in results]
    primary_count = classifications.count("local_task_tangent_sufficient")
    if any(name == "invalid" for name in classifications):
        conclusion = "invalid_campaign"
    elif len(config.seeds) == 3 and primary_count == 3:
        conclusion = "supported_local_task_tangent_sufficient_three_of_three"
    elif len(set(classifications)) == 1:
        conclusion = classifications[0]
    else:
        conclusion = "checkpoint_stratified_local_continuation_geometry"
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
            "scheduled": len(config.seeds) - reused,
            "completed": len(results),
            "failed": 0,
            "excluded": 0,
            "retries": 0,
            "reused": reused,
            "trained_models": 0,
            "fitted_predictive_observers": 0,
            "fitted_writers": 0,
            "finite_difference_state_families": len(config.seeds)
            * len(fixed.HELDOUT_COHORTS)
            * len(fixed.REGIMES)
            * config.carrier_rank
            * 4,
        },
        "aggregates": {
            "conclusion": conclusion,
            "primary_checkpoint_pass_count": primary_count,
            "required_checkpoint_pass_count": 3,
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
                    "context_replay_contract",
                    "coordinate_scale_contract",
                    "numerical_chart_contract",
                    "target_control_contract",
                    "local_linearization_adequate",
                    "tangent_checkpoint_pass",
                    "tangent_specificity",
                    "primary_checkpoint_pass",
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
            "The study reuses three selected checkpoints and previously inspected held-out cells.",
            "The task tangent is conditioned on the frozen answer-token circular-moment decoder.",
            "The exact quotient phase and context chart are diagnostic and checkpoint local.",
            "Finite differences and residual patches are off-manifold interventions.",
            "A first-order local kernel is not a globally task-null representation subspace.",
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
            "data/experiments/tinyllm_local_continuation_tangent/"
            "20260807_d6_preregistered_diagnostic"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=_ints, default=fixed.PRIMARY_SEEDS)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = LocalContinuationTangentConfig(
        seeds=args.seeds,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
