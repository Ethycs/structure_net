#!/usr/bin/env python3
"""Test observable scalar sensors with a frozen TinyLLM task covector."""

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
import experiments.structure_net.tinyllm_fixed_gauge_error_decomposition as decomposition
import experiments.structure_net.tinyllm_fixed_gauge_writer_capacity as capacity
import experiments.structure_net.tinyllm_fixed_semantic_gauge_writer as fixed
import experiments.structure_net.tinyllm_local_task_tangent as local
import experiments.structure_net.tinyllm_source_task_covector_portability as source
import experiments.structure_net.tinyllm_cross_seed_causal_carrier_transport as transport
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-c2-observable-scalar-residual.v1"
HYPOTHESIS_ID = "tinyllm-c2-observable-scalar-residual-v1"
SUPERSEDED_BY = "tinyllm-c2-task-activation-scalar-sensor-v2"
SUPERSEDED_REASON = (
    "superseded before fresh-E outcome: the source-only CUDA shakedown failed "
    "the observed-carrier and frozen-covector replay contracts"
)
SOURCE_COVECTOR_CAMPAIGN_SHA256 = (
    "fe153abd0aa1749a862095e577e6e9cf8a0f7054bade2ef3c6833bf32d7f4ac5"
)
SOURCE_COVECTOR_IMPLEMENTATION_SHA256 = (
    "6716b909d0c245059a1ed1310f20f4d9e56deb8c49a7d3a031972542fccb3046"
)
SOURCE_COHORTS = ("heldout_a", "heldout_b")
FRESH_COHORT = "heldout_e"
FRESH_COHORT_SEEDS = {"composition": 630007, "extrapolation": 630008}
CANDIDATE_NAMES = ("phase_only", "posterior", "calibration", "activation")
CONTROLLED_CANDIDATES = ("posterior", "calibration", "activation")
CONTROL_SUFFIXES = ("shuffled", "flipped", "random")
EXPECTED_FEATURE_WIDTHS = {
    "phase_only": 9,
    "posterior": 63,
    "calibration": 135,
    "activation": 189,
}


@dataclass(frozen=True)
class ObservableScalarResidualConfig:
    source_campaign: str = (
        "data/experiments/tinyllm_fixed_gauge_writer_capacity/"
        "20260806_d6_preregistered_diagnostic"
    )
    source_covector_campaign: str = (
        "data/experiments/tinyllm_source_task_covector_portability/"
        "20260807_d6_preregistered_fresh_cohort"
    )
    seeds: tuple[int, ...] = fixed.PRIMARY_SEEDS
    orbit_count: int = 64
    carrier_rank: int = 3
    writer_order: int = 4
    activation_context_rank: int = 2
    scalar_ridge: float = 1e-3
    fine_step_std: float = 0.025
    coarse_step_std: float = 0.05
    coordinate_scale_floor: float = 1e-8
    context_scale_floor: float = 1e-8
    gradient_denominator_floor: float = 1e-12
    replay_tolerance: float = 1e-6
    derivative_cosine_floor: float = 0.98
    derivative_relative_l2_ceiling: float = 0.15
    signed_error_r2_floor: float = 0.50
    residual_mae_fraction_ceiling: float = 0.50
    sign_agreement_floor: float = 0.75
    sign_magnitude_floor_bins: float = 0.01
    observed_alignment_floor: float = 0.99
    observed_mean_shift_ceiling_bins: float = 0.125
    observed_p95_shift_ceiling_bins: float = 0.50
    observed_sheet_difference_ceiling: float = 0.01
    covector_replay_relative_l2_ceiling: float = 0.02
    covector_replay_cosine_floor: float = 0.99
    specificity_margin_bins: float = 0.125
    activation_batch_size: int = 256
    continuation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != fixed.PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("primary scalar-sensor seeds are fixed to 7,29,53")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.orbit_count != 64:
            raise ValueError("the frozen predecessor requires 64 exact orbits")
        if self.carrier_rank != 3 or self.writer_order != 4:
            raise ValueError("carrier rank and writer order are fixed to three and four")
        if self.activation_context_rank != 2:
            raise ValueError("activation context rank is fixed to two per cut")
        if not 0.0 < self.fine_step_std < self.coarse_step_std:
            raise ValueError("finite-difference steps must be positive and ordered")
        for name in (
            "scalar_ridge",
            "coordinate_scale_floor",
            "context_scale_floor",
            "gradient_denominator_floor",
            "replay_tolerance",
            "specificity_margin_bins",
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
        Path(source.__file__),
        Path(local.__file__),
        Path(capacity.__file__),
        Path(decomposition.__file__),
        Path(fixed.__file__),
        Path(transport.__file__),
        Path(transport.rank.__file__),
        Path(transport.coupling.__file__),
        Path(jacobian.__file__),
        Path(local.readout.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evidence_role(config: ObservableScalarResidualConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_fresh_cohort_sequential_underpowered_mechanistic_evidence"
    )


def posterior_statistics(posterior: torch.Tensor) -> torch.Tensor:
    """Return six target-free intrinsic statistics of an answer posterior."""
    if posterior.ndim != 2 or posterior.shape[1] < 3:
        raise ValueError("posterior must be a matrix with at least three bins")
    value = posterior.double().clamp_min(1e-24)
    value = value / value.sum(1, keepdim=True)
    entropy = -(value * value.log()).sum(1) / math.log(value.shape[1])
    top_two = value.topk(2, dim=1).values
    angle, first_radius = local.readout.posterior_moment(value)
    bin_angles = torch.arange(
        value.shape[1], dtype=torch.float64, device=value.device
    ) * (2.0 * math.pi / value.shape[1])
    second_real = value @ torch.cos(2.0 * bin_angles)
    second_imag = value @ torch.sin(2.0 * bin_angles)
    second_radius = torch.sqrt(second_real.square() + second_imag.square())
    coordinate = angle / (2.0 * math.pi / value.shape[1])
    center_distance = torch.remainder(coordinate + 0.5, 1.0) - 0.5
    boundary_margin = 0.5 - center_distance.abs()
    return torch.stack(
        (
            entropy,
            top_two[:, 0],
            top_two[:, 0] - top_two[:, 1],
            first_radius,
            second_radius,
            boundary_margin,
        ),
        dim=1,
    )


def fit_pca_context(value: torch.Tensor, rank: int) -> dict[str, torch.Tensor]:
    value = value.double()
    if value.ndim != 2 or len(value) <= rank or rank < 1:
        raise ValueError("PCA context must be a row matrix larger than its rank")
    mean = value.mean(0)
    centered = value - mean
    _, singular, right = torch.linalg.svd(centered, full_matrices=False)
    if int((singular > singular.max().clamp_min(1e-24) * 1e-10).sum()) < rank:
        raise ValueError("PCA context is numerically rank deficient")
    return {"mean": mean, "basis": right[:rank].T, "singular_values": singular[:rank]}


def apply_pca_context(value: torch.Tensor, model: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return (value.double() - model["mean"].double()) @ model["basis"].double()


def fit_standardizer(
    value: torch.Tensor, scale_floor: float
) -> dict[str, torch.Tensor]:
    value = value.double()
    if value.ndim != 2:
        raise ValueError("standardizer input must be a matrix")
    return {
        "mean": value.mean(0),
        "scale": value.std(0, unbiased=False).clamp_min(scale_floor),
    }


def apply_standardizer(
    value: torch.Tensor, model: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    return (value.double() - model["mean"].double()) / model["scale"].double()


def random_signed_scalar(value: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, value.shape, generator=generator, device="cpu")
    signs = signs.to(value.device, dtype=torch.float64) * 2.0 - 1.0
    return value.double().abs() * signs


def _mapping_summary(mapping: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "linear": mapping["linear"].detach().cpu().tolist(),
        "intercept": mapping["intercept"].detach().cpu().tolist(),
    }


def _pca_summary(model: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {name: value.detach().cpu().tolist() for name, value in model.items()}


def _standardizer_summary(model: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {name: value.detach().cpu().tolist() for name, value in model.items()}


@torch.no_grad()
def order4_posterior(
    system: Any,
    task: CircleTaskConfig,
    config: ObservableScalarResidualConfig,
    cell: Mapping[str, Any],
    basis: torch.Tensor,
    predicted: torch.Tensor,
) -> torch.Tensor:
    state = local._state_from_coordinates(cell, basis, predicted)
    return local._continue_states(
        system, task, config, cell["target_cut"], {"order4": state}
    )["order4"]


@torch.no_grad()
def activation_queries(
    system: Any,
    cell: Mapping[str, Any],
    basis: torch.Tensor,
    predicted: torch.Tensor,
) -> dict[str, torch.Tensor]:
    predicted_state = local._state_from_coordinates(cell, basis, predicted)
    block = system.model.transformer["h"][0]
    post_mlp = predicted_state + block.mlp(block.ln_2(predicted_state))
    return {
        "propagated": cell["propagated"][:, -1, :].double(),
        "post_attention": predicted_state[:, -1, :].double(),
        "post_mlp": post_mlp[:, -1, :].double(),
    }


def calibration_context(
    dataset: Any, orbit_count: int, device: torch.device
) -> tuple[torch.Tensor, float]:
    values = dataset.calibration.to(device).reshape(orbit_count, 2, -1).double()
    return values.mean(1), float((values[:, 0] - values[:, 1]).abs().max())


def candidate_contexts(
    posterior: torch.Tensor,
    calibration: torch.Tensor,
    activation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    post = posterior_statistics(posterior)
    if len(post) != len(calibration) or len(post) != len(activation):
        raise ValueError("candidate contexts must be row aligned")
    return {
        "posterior": post,
        "calibration": torch.cat((post, calibration.double()), dim=1),
        "activation": torch.cat(
            (post, calibration.double(), activation.double()), dim=1
        ),
    }


def candidate_features(
    phase_features: torch.Tensor,
    contexts: Mapping[str, torch.Tensor],
    standardizers: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    output = {"phase_only": phase_features.double()}
    for name in CONTROLLED_CANDIDATES:
        output[name] = capacity.conditional_features(
            phase_features,
            apply_standardizer(contexts[name], standardizers[name]),
        )
    for name, width in EXPECTED_FEATURE_WIDTHS.items():
        if output[name].shape[1] != width:
            raise ValueError(f"{name} feature width changed")
    return output


def fit_scalar_maps(
    features: Mapping[str, torch.Tensor],
    signed_error: torch.Tensor,
    ridge: float,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: capacity.fit_writer(value, signed_error.reshape(-1, 1), ridge)
        for name, value in features.items()
    }


def _fresh_cell(
    system: Any,
    task: CircleTaskConfig,
    transport_config: Any,
    bridge: Any,
    dataset: Any,
    regime: str,
    device: torch.device,
) -> dict[str, Any]:
    cell = transport._extract_cell(
        system,
        task,
        transport_config,
        bridge,
        dataset,
        "heldout_b",
        regime,
        device,
    )
    cell["cohort"] = FRESH_COHORT
    cell["evaluation_seed"] = FRESH_COHORT_SEEDS[regime]
    return cell


def observed_carrier_pass(
    audit: Mapping[str, float], config: ObservableScalarResidualConfig
) -> bool:
    return bool(
        audit["circular_alignment"] >= config.observed_alignment_floor
        and audit["mean_shift_bins"] <= config.observed_mean_shift_ceiling_bins
        and audit["p95_shift_bins"] <= config.observed_p95_shift_ceiling_bins
        and audit["maximum_sheet_difference"]
        <= config.observed_sheet_difference_ceiling
    )


def _load_source_covectors(
    config: ObservableScalarResidualConfig,
) -> tuple[dict[str, Any], Path, dict[int, tuple[dict[str, Any], Path]]]:
    path = Path(config.source_covector_campaign) / "campaign_results.json"
    campaign = json.loads(path.read_text())
    expected = {str(seed): "source_covector_portable_scalar_not" for seed in fixed.PRIMARY_SEEDS}
    if (
        _sha256(path) != SOURCE_COVECTOR_CAMPAIGN_SHA256
        or campaign.get("schema_version") != source.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != source.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256")
        != SOURCE_COVECTOR_IMPLEMENTATION_SHA256
        or campaign.get("aggregates", {}).get("classification_by_seed") != expected
        or campaign.get("aggregates", {}).get("gate_counts", {}).get(
            "source_covector_oracle_error_all_fresh_cells_pass"
        )
        != 3
        or campaign.get("aggregates", {}).get("supported") is not False
    ):
        raise ValueError(f"invalid source-covector campaign {path}")
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if not set(config.seeds).issubset(entries):
        raise ValueError("source-covector campaign lacks a requested checkpoint")
    details = {}
    for seed in config.seeds:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        detail = json.loads(detail_path.read_text())
        if (
            _sha256(detail_path) != entry.get("result_sha256")
            or detail.get("schema_version") != source.SCHEMA_VERSION
            or detail.get("hypothesis_id") != source.HYPOTHESIS_ID
            or detail.get("implementation_sha256")
            != SOURCE_COVECTOR_IMPLEMENTATION_SHA256
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or detail.get("classification") != "source_covector_portable_scalar_not"
            or detail.get("gates", {}).get(
                "source_covector_oracle_error_all_fresh_cells_pass"
            )
            is not True
            or len(detail.get("source_maps", {}).get("covector", {}).get("linear", []))
            != 9
        ):
            raise ValueError(f"invalid source-covector result {detail_path}")
        details[seed] = (detail, detail_path)
    return campaign, path, details


def _fingerprint(
    config: ObservableScalarResidualConfig,
    seed: int,
    source_campaign_sha256: str,
    source_result_sha256: str,
    checkpoint_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "source_cohort_seeds": fixed.COHORT_SEEDS,
        "fresh_cohort_seeds": FRESH_COHORT_SEEDS,
        "seed": seed,
        "source_campaign_sha256": source_campaign_sha256,
        "source_result_sha256": source_result_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def arm_gate_summary(
    cells: Sequence[Mapping[str, Any]], specificity_margin_bins: float
) -> dict[str, Any]:
    means = {}
    endpoint = {}
    for name in CANDIDATE_NAMES:
        records = [cell["states"][name]["continuous"] for cell in cells]
        means[name] = sum(item["mean_moment_shift_bins"] for item in records) / len(records)
        endpoint[name] = all(item["continuous_pass"] for item in records)
    controlled = {}
    for name in CONTROLLED_CANDIDATES:
        controls = {}
        for suffix in CONTROL_SUFFIXES:
            key = f"{name}_{suffix}"
            records = [cell["states"][key]["continuous"] for cell in cells]
            mean = sum(item["mean_moment_shift_bins"] for item in records) / len(records)
            any_failure = any(not item["continuous_pass"] for item in records)
            margin = mean - means[name]
            controls[suffix] = {
                "any_failure": any_failure,
                "aggregate_mean_shift_bins": mean,
                "margin_over_candidate_bins": margin,
                "specific": bool(any_failure and margin >= specificity_margin_bins),
            }
        specificity = all(item["specific"] for item in controls.values())
        controlled[name] = {
            "endpoint_all_fresh_cells_pass": endpoint[name],
            "controls": controls,
            "all_controls_specific": specificity,
            "complete_gate": bool(endpoint[name] and specificity),
        }
    return {
        "aggregate_mean_shift_bins": means,
        "endpoint_all_fresh_cells_pass": endpoint,
        "controlled_arms": controlled,
    }


def classify_checkpoint(
    *, valid: bool, oracle_pass: bool, arm_gates: Mapping[str, Any]
) -> tuple[str, bool]:
    if not valid:
        return "invalid", False
    if not oracle_pass:
        return "fresh_local_mechanism_not_replicated", False
    controlled = arm_gates["controlled_arms"]
    if controlled["posterior"]["complete_gate"]:
        return "posterior_scalar_sensor_sufficient", True
    if controlled["posterior"]["endpoint_all_fresh_cells_pass"]:
        return "posterior_sensor_nonspecific", False
    if controlled["calibration"]["complete_gate"]:
        return "calibration_scalar_rescue", False
    if controlled["activation"]["complete_gate"]:
        return "activation_scalar_rescue", False
    if any(
        controlled[name]["endpoint_all_fresh_cells_pass"]
        for name in ("calibration", "activation")
    ):
        return "secondary_sensor_nonspecific", False
    if not any(arm_gates["endpoint_all_fresh_cells_pass"].values()):
        return "no_observable_scalar_sensor", False
    return "mixed_scalar_sensor_geometry", False


def _campaign_decision(
    classifications: Sequence[str], pass_count: int, allow_underpowered: bool
) -> dict[str, Any]:
    if allow_underpowered:
        conclusion = "systems_lifecycle_only_not_quality_evidence"
        supported = False
    elif len(classifications) == 3 and pass_count == 3:
        conclusion = "supported_posterior_scalar_sensor_three_of_three"
        supported = True
    elif len(set(classifications)) == 1:
        conclusion = classifications[0]
        supported = False
    else:
        conclusion = "checkpoint_stratified_scalar_sensor"
        supported = False
    return {
        "supported": supported,
        "posterior_scalar_sensor_pass_count": pass_count,
        "required_checkpoint_count": 3,
        "conclusion": conclusion,
    }


def _campaign_is_reusable(
    campaign: Mapping[str, Any],
    config: ObservableScalarResidualConfig,
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


def _reusable_seed_result(
    result_path: Path,
    config: ObservableScalarResidualConfig,
    implementation: str,
    seed: int,
    fingerprint: str,
) -> dict[str, Any] | None:
    """Return one completed fingerprint-matched seed without rewriting it."""
    if not result_path.is_file():
        return None
    existing = json.loads(result_path.read_text())
    if not (
        existing.get("schema_version") == SCHEMA_VERSION
        and existing.get("hypothesis_id") == HYPOTHESIS_ID
        and existing.get("status") == "completed"
        and existing.get("evidence_role") == _evidence_role(config)
        and existing.get("implementation_sha256") == implementation
        and existing.get("configuration") == _json_compatible(asdict(config))
        and existing.get("scientific_fingerprint") == fingerprint
        and int(existing.get("seed", -1)) == seed
        and existing.get("artifacts", {}).get("result") == str(result_path)
    ):
        raise ValueError(f"incompatible completed result {result_path}; use a new root")
    return existing


@torch.no_grad()
def run_source_only_shakedown(
    config: ObservableScalarResidualConfig, output: Path
) -> dict[str, Any]:
    """Exercise the CUDA lifecycle without constructing or spending cohort E."""
    if not config.allow_underpowered or len(config.seeds) != 1:
        raise ValueError("source-only shakedown requires one underpowered seed")
    output.mkdir(parents=True, exist_ok=True)
    implementation = _implementation_digest()
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text())
        if (
            existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("hypothesis_id") == HYPOTHESIS_ID
            and existing.get("status") == "completed"
            and existing.get("evidence_role") == _evidence_role(config)
            and existing.get("source_only_shakedown") is True
            and existing.get("implementation_sha256") == implementation
            and existing.get("configuration") == _json_compatible(asdict(config))
        ):
            print("source-only shakedown already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed source-only shakedown {campaign_path}")

    started = time.perf_counter()
    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    writer_campaign, writer_path, writer_details = local._load_predecessor(config)
    covector_campaign, covector_path, covector_details = _load_source_covectors(config)
    base_config = fixed.FixedSemanticGaugeWriterConfig(
        seeds=config.seeds,
        orbit_count=config.orbit_count,
        device=config.device,
        allow_underpowered=True,
    )
    transport_config = fixed._transport_config(base_config)
    rank_config = transport._rank_config(transport_config)
    bridge = transport.rank._bridge_config(rank_config)
    seed = config.seeds[0]
    prior, prior_path = writer_details[seed]
    covector_prior, covector_result_path = covector_details[seed]
    if covector_prior["provenance"]["predecessor_result_sha256"] != _sha256(
        prior_path
    ):
        raise ValueError(f"writer/covector predecessor mismatch for seed {seed}")
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
    if max(
        decomposition._numeric_max_difference(basis_summary, prior["basis"]),
        decomposition._numeric_max_difference(basis_summary, covector_prior["basis"]),
    ) > config.replay_tolerance:
        raise ValueError(f"basis summary mismatch for seed {seed}")

    alignment_datasets = {
        regime: transport.rank.deck.generate_exact_orbits(
            task,
            k=2,
            orbit_count=config.orbit_count,
            seed=fixed.COHORT_SEEDS["alignment_fit"][regime],
            regime=regime,
        )
        for regime in fixed.REGIMES
    }
    alignment_cells = {
        regime: transport._extract_cell(
            system,
            task,
            transport_config,
            bridge,
            alignment_datasets[regime],
            "alignment_fit",
            regime,
            device,
        )
        for regime in fixed.REGIMES
    }
    fit_coordinates = torch.cat(
        [
            transport._coordinates(alignment_cells[regime], basis)
            for regime in fixed.REGIMES
        ]
    )
    coordinate_scale = fit_coordinates.std(0, unbiased=False).clamp_min(
        config.coordinate_scale_floor
    )
    dataset = transport.rank.deck.generate_exact_orbits(
        task,
        k=2,
        orbit_count=config.orbit_count,
        seed=fixed.COHORT_SEEDS["heldout_a"]["composition"],
        regime="composition",
    )
    cell = transport._extract_cell(
        system,
        task,
        transport_config,
        bridge,
        dataset,
        "heldout_a",
        "composition",
        device,
    )
    observed_carrier, observed_audit = fixed.semantic_orbit_carrier(
        dataset, task, config.orbit_count, device
    )
    latent_carrier, _ = decomposition.oracle_orbit_carrier(
        dataset, task, config.orbit_count, device
    )
    q_observed = capacity.fourier_features(observed_carrier, config.writer_order)
    q_latent = capacity.fourier_features(latent_carrier, config.writer_order)
    writer = capacity._writer_from_summary(
        prior["alignment_fit"]["writers"]["quotient_order4"], device
    )
    frozen_covector = capacity._writer_from_summary(
        covector_prior["source_maps"]["covector"], device
    )
    predicted = transport.apply_affine(q_observed, writer)
    target = transport._coordinates(cell, basis)
    derivative = local.finite_difference_cell(
        system,
        task,
        config,
        cell,
        basis,
        predicted,
        target,
        coordinate_scale,
    )
    posterior = order4_posterior(system, task, config, cell, basis, predicted)
    calibration, sheet_difference = calibration_context(
        dataset, config.orbit_count, device
    )
    activations = activation_queries(system, cell, basis, predicted)
    pca_models = {
        cut: fit_pca_context(value, config.activation_context_rank)
        for cut, value in activations.items()
    }
    activation_context = torch.cat(
        [
            apply_pca_context(activations[cut], pca_models[cut])
            for cut in ("propagated", "post_attention", "post_mlp")
        ],
        dim=1,
    )
    contexts = candidate_contexts(posterior, calibration, activation_context)
    standardizers = {
        name: fit_standardizer(value, config.context_scale_floor)
        for name, value in contexts.items()
    }
    features = candidate_features(q_observed, contexts, standardizers)
    scalar_maps = fit_scalar_maps(features, derivative["observed_delta"], config.scalar_ridge)
    source_linearization = jacobian.linearization_metrics(
        derivative["fine_gradient"],
        derivative["coarse_gradient"],
        derivative["predicted_delta"],
        derivative["observed_delta"],
        config,
    )
    covector_replay = source.regression_metrics(
        transport.apply_affine(q_observed, frozen_covector),
        transport.apply_affine(q_latent, frozen_covector),
    )
    feature_contract = bool(
        {name: int(value.shape[1]) for name, value in features.items()}
        == EXPECTED_FEATURE_WIDTHS
        and all(
            len(model["singular_values"]) == config.activation_context_rank
            and bool(torch.isfinite(model["basis"]).all())
            for model in pca_models.values()
        )
        and all(
            float(model["scale"].min()) > config.context_scale_floor
            for model in standardizers.values()
        )
    )
    finite = all(
        bool(torch.isfinite(value).all())
        for value in (
            q_observed,
            q_latent,
            posterior,
            calibration,
            activation_context,
            *features.values(),
            *(tensor for mapping in scalar_maps.values() for tensor in mapping.values()),
        )
    )
    contracts = {
        "provenance_contract": provenance["checkpoint_sha256"]
        == prior["provenance"]["checkpoint_sha256"],
        "observed_carrier_contract": observed_carrier_pass(observed_audit, config),
        "calibration_sheet_contract": sheet_difference <= 1e-8,
        "feature_contract": feature_contract,
        "source_linearization_contract": source_linearization["adequate"],
        "covector_replay_contract": bool(
            covector_replay["relative_l2"]
            <= config.covector_replay_relative_l2_ceiling
            and covector_replay["mean_row_cosine"]
            >= config.covector_replay_cosine_floor
        ),
        "finite_contract": finite,
    }
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "source_only_shakedown": True,
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "implementation_sha256": implementation,
        "seed": seed,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        },
        "provenance": {
            "writer_campaign": str(writer_path),
            "writer_campaign_sha256": _sha256(writer_path),
            "writer_result": str(prior_path),
            "writer_result_sha256": _sha256(prior_path),
            "source_covector_campaign": str(covector_path),
            "source_covector_campaign_sha256": _sha256(covector_path),
            "source_covector_result": str(covector_result_path),
            "source_covector_result_sha256": _sha256(covector_result_path),
            "character_result": str(frozen_path),
            "character_result_sha256": _sha256(frozen_path),
        },
        "contracts": contracts,
        "all_contracts_pass": all(contracts.values()),
        "feature_widths": {
            name: int(value.shape[1]) for name, value in features.items()
        },
        "observed_carrier_audit": observed_audit,
        "source_linearization": source_linearization,
        "covector_replay": covector_replay,
        "analysis_seconds": time.perf_counter() - started,
        "artifacts": {"campaign": str(campaign_path)},
    }
    _write_json(campaign_path, campaign)
    return campaign


@torch.no_grad()
def run_campaign(
    config: ObservableScalarResidualConfig, output: Path
) -> dict[str, Any]:
    raise RuntimeError(f"{SUPERSEDED_REASON}; use {SUPERSEDED_BY}")


@torch.no_grad()
def _run_superseded_campaign_implementation(
    config: ObservableScalarResidualConfig, output: Path
) -> dict[str, Any]:
    """Preserve the never-executed fresh-E implementation as an inert record."""
    if config.allow_underpowered:
        raise ValueError(
            "cohort E cannot be used for a one-checkpoint shakedown; run the "
            "locked three-checkpoint campaign after a source-only CUDA check"
        )
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
    writer_campaign, writer_path, writer_details = local._load_predecessor(config)
    covector_campaign, covector_path, covector_details = _load_source_covectors(config)
    base_config = fixed.FixedSemanticGaugeWriterConfig(
        seeds=config.seeds,
        orbit_count=config.orbit_count,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )
    transport_config = fixed._transport_config(base_config)
    rank_config = transport._rank_config(transport_config)
    bridge = transport.rank._bridge_config(rank_config)
    source_datasets = {
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
    fresh_datasets = {
        regime: transport.rank.deck.generate_exact_orbits(
            task,
            k=2,
            orbit_count=config.orbit_count,
            seed=FRESH_COHORT_SEEDS[regime],
            regime=regime,
        )
        for regime in fixed.REGIMES
    }

    results = []
    reused = 0
    for seed in config.seeds:
        seed_started = time.perf_counter()
        prior, prior_path = writer_details[seed]
        covector_prior, covector_result_path = covector_details[seed]
        if (
            covector_prior["provenance"]["predecessor_result_sha256"]
            != _sha256(prior_path)
        ):
            raise ValueError(f"writer/covector predecessor mismatch for seed {seed}")
        fingerprint = _fingerprint(
            config,
            seed,
            _sha256(covector_path),
            _sha256(covector_result_path),
            covector_prior["provenance"]["checkpoint_sha256"],
        )
        result_path = output / "runs" / f"seed_{seed}" / "result.json"
        existing = _reusable_seed_result(
            result_path, config, implementation, seed, fingerprint
        )
        if existing is not None:
            results.append(existing)
            reused += 1
            print(f"resuming {existing['experiment_id']}", flush=True)
            continue

        system, provenance = transport.rank.deck.load_source(
            task, bridge, 2, seed, device
        )
        if provenance["checkpoint_sha256"] != prior["provenance"]["checkpoint_sha256"]:
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
        if max(
            decomposition._numeric_max_difference(basis_summary, prior["basis"]),
            decomposition._numeric_max_difference(basis_summary, covector_prior["basis"]),
        ) > config.replay_tolerance:
            raise ValueError(f"basis summary mismatch for seed {seed}")
        source_cells = {
            cohort: {
                regime: transport._extract_cell(
                    system,
                    task,
                    transport_config,
                    bridge,
                    source_datasets[cohort][regime],
                    cohort,
                    regime,
                    device,
                )
                for regime in fixed.REGIMES
            }
            for cohort in fixed.COHORT_SEEDS
        }
        fresh_cells = {
            regime: _fresh_cell(
                system,
                task,
                transport_config,
                bridge,
                fresh_datasets[regime],
                regime,
                device,
            )
            for regime in fixed.REGIMES
        }
        fit_coordinates = torch.cat(
            [
                transport._coordinates(source_cells["alignment_fit"][regime], basis)
                for regime in fixed.REGIMES
            ]
        )
        coordinate_scale = fit_coordinates.std(0, unbiased=False).clamp_min(
            config.coordinate_scale_floor
        )
        writer = capacity._writer_from_summary(
            prior["alignment_fit"]["writers"]["quotient_order4"], device
        )
        frozen_covector = capacity._writer_from_summary(
            covector_prior["source_maps"]["covector"], device
        )

        source_records = []
        source_values = []
        observed_audits = []
        for cohort in SOURCE_COHORTS:
            for regime in fixed.REGIMES:
                dataset = source_datasets[cohort][regime]
                cell = source_cells[cohort][regime]
                observed_carrier, observed_audit = fixed.semantic_orbit_carrier(
                    dataset, task, config.orbit_count, device
                )
                latent_carrier, _ = decomposition.oracle_orbit_carrier(
                    dataset, task, config.orbit_count, device
                )
                q_observed = capacity.fourier_features(
                    observed_carrier, config.writer_order
                )
                q_latent = capacity.fourier_features(latent_carrier, config.writer_order)
                predicted = transport.apply_affine(q_observed, writer)
                target = transport._coordinates(cell, basis)
                derivative = local.finite_difference_cell(
                    system,
                    task,
                    config,
                    cell,
                    basis,
                    predicted,
                    target,
                    coordinate_scale,
                )
                posterior = order4_posterior(
                    system, task, config, cell, basis, predicted
                )
                calibration, calibration_sheet_difference = calibration_context(
                    dataset, config.orbit_count, device
                )
                activations = activation_queries(system, cell, basis, predicted)
                observed_audits.append(observed_audit)
                source_values.append(
                    {
                        "q_observed": q_observed,
                        "q_latent": q_latent,
                        "posterior": posterior,
                        "calibration": calibration,
                        "activations": activations,
                        **derivative,
                    }
                )
                source_records.append(
                    {
                        "cohort": cohort,
                        "regime": regime,
                        "evaluation_seed": fixed.COHORT_SEEDS[cohort][regime],
                        "observed_carrier_audit": observed_audit,
                        "calibration_sheet_difference": calibration_sheet_difference,
                    }
                )

        pca_models = {}
        for cut in ("propagated", "post_attention", "post_mlp"):
            pooled = torch.cat(
                [value["activations"][cut] for value in source_values], dim=0
            )
            pca_models[cut] = fit_pca_context(pooled, config.activation_context_rank)
        for value in source_values:
            value["activation_context"] = torch.cat(
                [
                    apply_pca_context(value["activations"][cut], pca_models[cut])
                    for cut in ("propagated", "post_attention", "post_mlp")
                ],
                dim=1,
            )
            value["contexts"] = candidate_contexts(
                value["posterior"], value["calibration"], value["activation_context"]
            )
        standardizers = {
            name: fit_standardizer(
                torch.cat([value["contexts"][name] for value in source_values], dim=0),
                config.context_scale_floor,
            )
            for name in CONTROLLED_CANDIDATES
        }
        for value in source_values:
            value["features"] = candidate_features(
                value["q_observed"], value["contexts"], standardizers
            )
        pooled_source = {
            name: torch.cat([value[name] for value in source_values], dim=0)
            for name in (
                "fine_gradient",
                "coarse_gradient",
                "predicted_delta",
                "observed_delta",
                "q_observed",
                "q_latent",
            )
        }
        pooled_features = {
            name: torch.cat([value["features"][name] for value in source_values], dim=0)
            for name in CANDIDATE_NAMES
        }
        scalar_maps = fit_scalar_maps(
            pooled_features,
            pooled_source["observed_delta"],
            config.scalar_ridge,
        )
        source_predictions = {
            name: transport.apply_affine(pooled_features[name], scalar_maps[name]).reshape(-1)
            for name in CANDIDATE_NAMES
        }
        source_fit = {
            name: source.regression_metrics(
                source_predictions[name],
                pooled_source["observed_delta"],
                config.sign_magnitude_floor_bins,
            )
            for name in CANDIDATE_NAMES
        }
        source_linearization = jacobian.linearization_metrics(
            pooled_source["fine_gradient"],
            pooled_source["coarse_gradient"],
            pooled_source["predicted_delta"],
            pooled_source["observed_delta"],
            config,
        )
        source_covector_replay = source.regression_metrics(
            transport.apply_affine(pooled_source["q_observed"], frozen_covector),
            transport.apply_affine(pooled_source["q_latent"], frozen_covector),
        )
        source_numerical_pass = bool(
            all(
                record["calibration_sheet_difference"] <= 1e-8
                and observed_carrier_pass(record["observed_carrier_audit"], config)
                for record in source_records
            )
            and all(
                torch.isfinite(value).all()
                for value in (
                    *pooled_features.values(),
                    *source_predictions.values(),
                    *(
                        tensor
                        for model in pca_models.values()
                        for tensor in model.values()
                    ),
                    *(
                        tensor
                        for model in standardizers.values()
                        for tensor in model.values()
                    ),
                    *(
                        tensor
                        for mapping in scalar_maps.values()
                        for tensor in mapping.values()
                    ),
                )
            )
        )

        fresh_records = []
        fresh_values = []
        numerical_passes = []
        for regime in fixed.REGIMES:
            dataset = fresh_datasets[regime]
            cell = fresh_cells[regime]
            observed_carrier, observed_audit = fixed.semantic_orbit_carrier(
                dataset, task, config.orbit_count, device
            )
            latent_carrier, _ = decomposition.oracle_orbit_carrier(
                dataset, task, config.orbit_count, device
            )
            q_observed = capacity.fourier_features(
                observed_carrier, config.writer_order
            )
            q_latent = capacity.fourier_features(latent_carrier, config.writer_order)
            predicted = transport.apply_affine(q_observed, writer)
            target = transport._coordinates(cell, basis)
            derivative = local.finite_difference_cell(
                system,
                task,
                config,
                cell,
                basis,
                predicted,
                target,
                coordinate_scale,
            )
            posterior = order4_posterior(system, task, config, cell, basis, predicted)
            calibration, calibration_sheet_difference = calibration_context(
                dataset, config.orbit_count, device
            )
            raw_activations = activation_queries(system, cell, basis, predicted)
            activation_context = torch.cat(
                [
                    apply_pca_context(raw_activations[cut], pca_models[cut])
                    for cut in ("propagated", "post_attention", "post_mlp")
                ],
                dim=1,
            )
            contexts = candidate_contexts(posterior, calibration, activation_context)
            features = candidate_features(q_observed, contexts, standardizers)
            predicted_scalars = {
                name: transport.apply_affine(features[name], scalar_maps[name]).reshape(-1)
                for name in CANDIDATE_NAMES
            }
            observed_gradient = transport.apply_affine(q_observed, frozen_covector)
            latent_gradient = transport.apply_affine(q_latent, frozen_covector)
            local_oracle_std = source.task_inverse_correction(
                derivative["fine_gradient"],
                derivative["observed_delta"],
                config.gradient_denominator_floor,
            )
            frozen_oracle_std = source.task_inverse_correction(
                observed_gradient,
                derivative["observed_delta"],
                config.gradient_denominator_floor,
            )
            corrections = {
                name: source.task_inverse_correction(
                    observed_gradient,
                    predicted_scalars[name],
                    config.gradient_denominator_floor,
                )
                for name in CANDIDATE_NAMES
            }
            control_seed = 97_000_000 + 100_003 * seed + FRESH_COHORT_SEEDS[regime]
            permutation = source.fixed_permutation(
                config.orbit_count, control_seed, device
            )
            controls = {}
            random_digests = {}
            for index, name in enumerate(CONTROLLED_CANDIDATES):
                shuffled = predicted_scalars[name][permutation]
                random = random_signed_scalar(
                    predicted_scalars[name], control_seed + 1 + index
                )
                controls[f"{name}_shuffled"] = source.task_inverse_correction(
                    observed_gradient, shuffled, config.gradient_denominator_floor
                )
                controls[f"{name}_flipped"] = source.task_inverse_correction(
                    observed_gradient, -predicted_scalars[name], config.gradient_denominator_floor
                )
                controls[f"{name}_random"] = source.task_inverse_correction(
                    observed_gradient, random, config.gradient_denominator_floor
                )
                random_digests[name] = hashlib.sha256(
                    random.detach().cpu().numpy().tobytes()
                ).hexdigest()
            scale = coordinate_scale.double().unsqueeze(0)
            coordinates = {
                "direct_rank3": target,
                "order4": predicted,
                "local_oracle": predicted + local_oracle_std * scale,
                "frozen_covector_oracle_error": predicted + frozen_oracle_std * scale,
                **{
                    name: predicted + correction * scale
                    for name, correction in corrections.items()
                },
                **{
                    name: predicted + correction * scale
                    for name, correction in controls.items()
                },
            }
            states = local.evaluate_science_states(
                system, task, config, cell, basis, coordinates
            )
            finite = all(
                torch.isfinite(value).all()
                for value in (
                    q_observed,
                    q_latent,
                    posterior,
                    activation_context,
                    derivative["fine_gradient"],
                    derivative["coarse_gradient"],
                    observed_gradient,
                    latent_gradient,
                    *features.values(),
                    *predicted_scalars.values(),
                    *corrections.values(),
                    *controls.values(),
                )
            )
            numerical = bool(
                finite
                and observed_carrier_pass(observed_audit, config)
                and calibration_sheet_difference <= 1e-8
            )
            numerical_passes.append(numerical)
            observed_audits.append(observed_audit)
            fresh_values.append(
                {
                    "fine_gradient": derivative["fine_gradient"].detach(),
                    "coarse_gradient": derivative["coarse_gradient"].detach(),
                    "predicted_delta": derivative["predicted_delta"].detach(),
                    "observed_delta": derivative["observed_delta"].detach(),
                    "observed_gradient": observed_gradient.detach(),
                    "latent_gradient": latent_gradient.detach(),
                    "predicted_scalars": {
                        name: value.detach() for name, value in predicted_scalars.items()
                    },
                }
            )
            fresh_records.append(
                {
                    "cohort": FRESH_COHORT,
                    "regime": regime,
                    "evaluation_seed": FRESH_COHORT_SEEDS[regime],
                    "observed_carrier_audit": observed_audit,
                    "calibration_sheet_difference": calibration_sheet_difference,
                    "feature_widths": {
                        name: int(value.shape[1]) for name, value in features.items()
                    },
                    "scalar_diagnostics": {
                        name: source.regression_metrics(
                            predicted_scalars[name],
                            derivative["observed_delta"],
                            config.sign_magnitude_floor_bins,
                        )
                        for name in CANDIDATE_NAMES
                    },
                    "covector_replay": source.regression_metrics(
                        observed_gradient, latent_gradient
                    ),
                    "states": states,
                    "numerical_contract": numerical,
                    "control_permutation_sha256": hashlib.sha256(
                        permutation.detach().cpu().numpy().tobytes()
                    ).hexdigest(),
                    "random_scalar_sha256": random_digests,
                }
            )

        pooled_fresh = {
            name: torch.cat([value[name] for value in fresh_values], dim=0)
            for name in (
                "fine_gradient",
                "coarse_gradient",
                "predicted_delta",
                "observed_delta",
                "observed_gradient",
                "latent_gradient",
            )
        }
        fresh_predictions = {
            name: torch.cat(
                [value["predicted_scalars"][name] for value in fresh_values], dim=0
            )
            for name in CANDIDATE_NAMES
        }
        fresh_fit = {
            name: source.regression_metrics(
                fresh_predictions[name],
                pooled_fresh["observed_delta"],
                config.sign_magnitude_floor_bins,
            )
            for name in CANDIDATE_NAMES
        }
        fresh_linearization = jacobian.linearization_metrics(
            pooled_fresh["fine_gradient"],
            pooled_fresh["coarse_gradient"],
            pooled_fresh["predicted_delta"],
            pooled_fresh["observed_delta"],
            config,
        )
        fresh_covector_replay = source.regression_metrics(
            pooled_fresh["observed_gradient"], pooled_fresh["latent_gradient"]
        )
        target_controls = all(
            not cell["states"]["zero"]["continuous"]["continuous_pass"]
            and cell["states"]["exact"]["continuous"]["continuous_pass"]
            and cell["states"]["direct_rank3"]["continuous"]["continuous_pass"]
            for cell in fresh_records
        )
        oracle_pass = all(
            cell["states"]["local_oracle"]["continuous"]["continuous_pass"]
            and cell["states"]["frozen_covector_oracle_error"]["continuous"]
            ["continuous_pass"]
            for cell in fresh_records
        )
        covector_contract = all(
            metrics["relative_l2"] <= config.covector_replay_relative_l2_ceiling
            and metrics["mean_row_cosine"] >= config.covector_replay_cosine_floor
            for metrics in (source_covector_replay, fresh_covector_replay)
        )
        feature_contract = bool(
            all(
                cell["feature_widths"] == EXPECTED_FEATURE_WIDTHS
                for cell in fresh_records
            )
            and all(
                len(model["singular_values"]) == config.activation_context_rank
                and bool(torch.isfinite(model["singular_values"]).all())
                for model in pca_models.values()
            )
            and all(
                float(model["scale"].min()) > config.context_scale_floor
                for model in standardizers.values()
            )
        )
        observed_contract = all(
            observed_carrier_pass(audit, config) for audit in observed_audits
        )
        numerical_pass = bool(
            source_numerical_pass
            and all(numerical_passes)
            and bool(torch.isfinite(coordinate_scale).all())
            and float(coordinate_scale.min()) > config.coordinate_scale_floor
        )
        arm_gates = arm_gate_summary(fresh_records, config.specificity_margin_bins)
        valid = bool(
            numerical_pass
            and feature_contract
            and observed_contract
            and covector_contract
            and source_linearization["adequate"]
            and fresh_linearization["adequate"]
            and target_controls
        )
        classification, primary_gate = classify_checkpoint(
            valid=valid, oracle_pass=oracle_pass, arm_gates=arm_gates
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_id": f"tinyllm-c2-observable-scalar-residual-seed{seed}",
            "status": "completed",
            "evidence_role": _evidence_role(config),
            "completed_at": _utc_now(),
            "seed": seed,
            "configuration": asdict(config),
            "scientific_fingerprint": fingerprint,
            "implementation_sha256": implementation,
            "provenance": {
                "writer_campaign": str(writer_path),
                "writer_campaign_sha256": _sha256(writer_path),
                "writer_result": str(prior_path),
                "writer_result_sha256": _sha256(prior_path),
                "source_covector_campaign": str(covector_path),
                "source_covector_campaign_sha256": _sha256(covector_path),
                "source_covector_result": str(covector_result_path),
                "source_covector_result_sha256": _sha256(covector_result_path),
                "checkpoint": provenance["checkpoint"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "frontend_checkpoint": provenance["frontend_checkpoint"],
                "frontend_checkpoint_sha256": provenance["frontend_checkpoint_sha256"],
                "character_result": str(frozen_path),
                "character_result_sha256": _sha256(frozen_path),
            },
            "basis": basis_summary,
            "coordinate_scale": coordinate_scale.detach().cpu().tolist(),
            "source_cells": source_records,
            "activation_pca": {
                name: _pca_summary(model) for name, model in pca_models.items()
            },
            "context_standardizers": {
                name: _standardizer_summary(model)
                for name, model in standardizers.items()
            },
            "scalar_maps": {
                name: _mapping_summary(mapping) for name, mapping in scalar_maps.items()
            },
            "source_fit": source_fit,
            "fresh_fit": fresh_fit,
            "source_linearization": source_linearization,
            "fresh_linearization": fresh_linearization,
            "source_covector_replay": source_covector_replay,
            "fresh_covector_replay": fresh_covector_replay,
            "fresh_cells": fresh_records,
            "arm_gates": arm_gates,
            "gates": {
                "provenance_contract": True,
                "numerical_contract": numerical_pass,
                "feature_and_leakage_contract": feature_contract,
                "observed_carrier_contract": observed_contract,
                "frozen_covector_replay_contract": covector_contract,
                "source_local_linearization_adequate": source_linearization["adequate"],
                "fresh_local_linearization_adequate": fresh_linearization["adequate"],
                "continuous_target_control_contract": target_controls,
                "local_and_frozen_covector_oracle_contract": oracle_pass,
                "posterior_endpoint_all_fresh_cells_pass": arm_gates[
                    "controlled_arms"
                ]["posterior"]["endpoint_all_fresh_cells_pass"],
                "posterior_controls_specific": arm_gates["controlled_arms"]
                ["posterior"]["all_controls_specific"],
                "posterior_scalar_sensor_gate": primary_gate,
                "calibration_secondary_gate": arm_gates["controlled_arms"]
                ["calibration"]["complete_gate"],
                "activation_secondary_gate": arm_gates["controlled_arms"]
                ["activation"]["complete_gate"],
            },
            "classification": classification,
            "primary_metric": float(primary_gate),
            "analysis_seconds": time.perf_counter() - seed_started,
            "artifacts": {"result": str(result_path)},
        }
        _write_json(result_path, result)
        results.append(result)
        print(f"seed {seed}: {classification}", flush=True)
        if _implementation_digest() != implementation:
            raise RuntimeError("observable scalar-sensor implementation changed during campaign")

    classifications = [result["classification"] for result in results]
    pass_count = sum(result["gates"]["posterior_scalar_sensor_gate"] for result in results)
    decision = _campaign_decision(classifications, pass_count, config.allow_underpowered)
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "source_cohort_seeds": fixed.COHORT_SEEDS,
        "fresh_cohort_seeds": FRESH_COHORT_SEEDS,
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
            "writer_campaign": str(writer_path),
            "writer_campaign_sha256": _sha256(writer_path),
            "writer_implementation_sha256": writer_campaign["implementation_sha256"],
            "source_covector_campaign": str(covector_path),
            "source_covector_campaign_sha256": _sha256(covector_path),
            "source_covector_implementation_sha256": covector_campaign[
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
            "refit_covectors": 0,
            "fitted_scalar_maps": len(config.seeds) * len(CANDIDATE_NAMES),
            "fitted_activation_pca_charts": len(config.seeds) * 3,
            "fresh_primary_cells": len(config.seeds) * len(fixed.REGIMES),
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
                    "numerical_contract",
                    "feature_and_leakage_contract",
                    "observed_carrier_contract",
                    "frozen_covector_replay_contract",
                    "fresh_local_linearization_adequate",
                    "continuous_target_control_contract",
                    "local_and_frozen_covector_oracle_contract",
                    "posterior_endpoint_all_fresh_cells_pass",
                    "posterior_controls_specific",
                    "posterior_scalar_sensor_gate",
                    "calibration_secondary_gate",
                    "activation_secondary_gate",
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
            "The frozen task covector is decoder-conditioned and inherited from a selected source campaign.",
            "The scalar features exclude latent phase, target labels, exact coordinates, exact continuation, and fresh derivatives.",
            "The observed phase carrier and orbit barycenter still use the paired exact-C2 diagnostic chart.",
            "The activation rung is wider than the posterior and calibration rungs.",
            "Cohort E changes generator seeds rather than defining a new shift family.",
            "Three selected checkpoints do not establish population prevalence.",
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
            "data/experiments/tinyllm_observable_scalar_residual/"
            "20260807_d6_preregistered_fresh_e"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=_ints, default=fixed.PRIMARY_SEEDS)
    parser.add_argument("--allow-underpowered", action="store_true")
    parser.add_argument("--source-only-shakedown", action="store_true")
    args = parser.parse_args()
    config = ObservableScalarResidualConfig(
        seeds=args.seeds,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    if args.source_only_shakedown:
        campaign = run_source_only_shakedown(config, args.output)
        print(json.dumps(campaign["contracts"], indent=2, sort_keys=True))
    else:
        campaign = run_campaign(config, args.output)
        print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
