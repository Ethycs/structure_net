#!/usr/bin/env python3
"""Test whether a source-fitted TinyLLM task covector repairs a fresh cohort."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
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
import experiments.structure_net.tinyllm_cross_seed_causal_carrier_transport as transport
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-c2-source-task-covector-portability.v1"
HYPOTHESIS_ID = "tinyllm-c2-source-task-covector-portability-v1"
PREDECESSOR_CAMPAIGN_SHA256 = (
    "c5592ee53b96e2d064a992070be6c3e699b24d88dbb658283e8dc2f00c267078"
)
PREDECESSOR_IMPLEMENTATION_SHA256 = (
    "7c284e35b5afc225eea45309262ab83c5f6d276736a557ebf20675ed3ccbfe7b"
)
LOCAL_TANGENT_CAMPAIGN_SHA256 = (
    "824a655b5c6d74f3c77259b9b7cacce3b4b3ea868ba74f48ba63fd5a24395130"
)
SOURCE_COHORTS = ("heldout_a", "heldout_b")
FRESH_COHORT = "heldout_c"
FRESH_COHORT_SEEDS = {"composition": 430007, "extrapolation": 430008}
PRIMARY_NAMES = (
    "local_oracle",
    "source_covector_oracle_error",
    "local_covector_source_error",
    "source_predicted",
)
CONTROL_NAMES = (
    "source_mean_covector",
    "source_shuffled_error",
    "source_flipped",
    "source_random_direction",
)


@dataclass(frozen=True)
class SourceTaskCovectorPortabilityConfig:
    source_campaign: str = (
        "data/experiments/tinyllm_fixed_gauge_writer_capacity/"
        "20260806_d6_preregistered_diagnostic"
    )
    local_tangent_campaign: str = (
        "data/experiments/tinyllm_local_task_tangent/"
        "20260807_d6_preregistered_diagnostic"
    )
    seeds: tuple[int, ...] = fixed.PRIMARY_SEEDS
    orbit_count: int = 64
    carrier_rank: int = 3
    writer_order: int = 4
    map_ridge: float = 1e-6
    fine_step_std: float = 0.025
    coarse_step_std: float = 0.05
    coordinate_scale_floor: float = 1e-8
    gradient_denominator_floor: float = 1e-12
    replay_tolerance: float = 1e-6
    derivative_cosine_floor: float = 0.98
    derivative_relative_l2_ceiling: float = 0.15
    signed_error_r2_floor: float = 0.50
    residual_mae_fraction_ceiling: float = 0.50
    sign_agreement_floor: float = 0.75
    sign_magnitude_floor_bins: float = 0.01
    specificity_margin_bins: float = 0.125
    activation_batch_size: int = 256
    continuation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != fixed.PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("primary portability seeds are fixed to 7,29,53")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.orbit_count != 64:
            raise ValueError("the frozen predecessor requires 64 exact orbits")
        if self.carrier_rank != 3 or self.writer_order != 4:
            raise ValueError("carrier rank and writer order are fixed to three and four")
        if not 0.0 < self.fine_step_std < self.coarse_step_std:
            raise ValueError("finite-difference steps must be positive and ordered")
        for name in (
            "map_ridge",
            "coordinate_scale_floor",
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


def _evidence_role(config: SourceTaskCovectorPortabilityConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_fresh_cohort_post_outcome_underpowered_mechanistic_evidence"
    )


def task_inverse_correction(
    gradient: torch.Tensor,
    signed_error: torch.Tensor,
    denominator_floor: float,
) -> torch.Tensor:
    """Return the minimum-norm standardized correction for a scalar task error."""
    if gradient.ndim != 2 or signed_error.ndim not in (1, 2):
        raise ValueError("gradient must be a matrix and signed error a vector")
    signed_error = signed_error.reshape(-1).double()
    if len(gradient) != len(signed_error):
        raise ValueError("gradient and signed error rows must match")
    gradient = gradient.double()
    denominator = gradient.square().sum(1, keepdim=True).clamp_min(
        denominator_floor
    )
    return gradient * signed_error[:, None] / denominator


def norm_matched_random(value: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random = torch.randn(
        value.shape, generator=generator, dtype=torch.float64, device="cpu"
    ).to(value.device)
    random = random / torch.linalg.vector_norm(
        random, dim=1, keepdim=True
    ).clamp_min(1e-24)
    return random * torch.linalg.vector_norm(
        value.double(), dim=1, keepdim=True
    )


def fixed_permutation(rows: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(rows, generator=generator).to(device)


def fit_source_maps(
    features: torch.Tensor,
    gradient: torch.Tensor,
    signed_error: torch.Tensor,
    ridge: float,
) -> dict[str, dict[str, torch.Tensor]]:
    if len(features) != len(gradient) or len(features) != len(signed_error):
        raise ValueError("source features and targets must be row aligned")
    return {
        "covector": capacity.fit_writer(features, gradient, ridge),
        "signed_error": capacity.fit_writer(
            features, signed_error.reshape(-1, 1), ridge
        ),
    }


def regression_metrics(
    predicted: torch.Tensor, target: torch.Tensor, sign_floor: float = 0.01
) -> dict[str, Any]:
    predicted = predicted.double()
    target = target.double()
    if predicted.shape != target.shape:
        raise ValueError("predicted and target tensors must have equal shape")
    residual = predicted - target
    zero_r2 = float(
        1.0
        - residual.square().sum()
        / target.square().sum().clamp_min(1e-24)
    )
    output: dict[str, Any] = {
        "zero_referenced_r2": zero_r2,
        "mae": float(residual.abs().mean()),
        "rmse": float(torch.sqrt(residual.square().mean())),
        "target_rms": float(torch.sqrt(target.square().mean())),
    }
    if target.ndim == 2 and target.shape[1] > 1:
        denominator = (
            torch.linalg.vector_norm(predicted, dim=1)
            * torch.linalg.vector_norm(target, dim=1)
        ).clamp_min(1e-24)
        cosine = (predicted * target).sum(1) / denominator
        output.update(
            {
                "mean_row_cosine": float(cosine.mean()),
                "median_row_cosine": float(cosine.median()),
                "minimum_row_cosine": float(cosine.min()),
                "relative_l2": float(
                    torch.linalg.vector_norm(residual)
                    / torch.linalg.vector_norm(target).clamp_min(1e-24)
                ),
            }
        )
    else:
        flat_target = target.reshape(-1)
        flat_predicted = predicted.reshape(-1)
        mask = flat_target.abs() >= sign_floor
        output.update(
            {
                "sign_evaluation_count": int(mask.sum()),
                "sign_agreement": (
                    float(
                        (
                            torch.sign(flat_predicted[mask])
                            == torch.sign(flat_target[mask])
                        )
                        .double()
                        .mean()
                    )
                    if bool(mask.any())
                    else 0.0
                ),
            }
        )
    return output


def _mapping_summary(mapping: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "linear": mapping["linear"].detach().cpu().tolist(),
        "intercept": mapping["intercept"].detach().cpu().tolist(),
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
    """Extract a new cell through the frozen helper, replacing label metadata only."""
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


def _load_local_tangent_provenance(
    config: SourceTaskCovectorPortabilityConfig,
) -> tuple[dict[str, Any], Path]:
    path = Path(config.local_tangent_campaign) / "campaign_results.json"
    campaign = json.loads(path.read_text())
    if (
        _sha256(path) != LOCAL_TANGENT_CAMPAIGN_SHA256
        or campaign.get("schema_version") != local.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != local.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or int(campaign.get("summary", {}).get("trained_models", -1)) != 0
    ):
        raise ValueError(f"invalid local-tangent provenance {path}")
    return campaign, path


def _fingerprint(
    config: SourceTaskCovectorPortabilityConfig,
    seed: int,
    predecessor_campaign_sha256: str,
    predecessor_result_sha256: str,
    checkpoint_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "source_cohort_seeds": fixed.COHORT_SEEDS,
        "fresh_cohort_seeds": FRESH_COHORT_SEEDS,
        "seed": seed,
        "predecessor_campaign_sha256": predecessor_campaign_sha256,
        "predecessor_result_sha256": predecessor_result_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def state_gate_summary(
    cells: Sequence[Mapping[str, Any]], specificity_margin_bins: float
) -> dict[str, Any]:
    names = ("order4", *PRIMARY_NAMES, *CONTROL_NAMES)
    means = {
        name: sum(
            cell["states"][name]["continuous"]["mean_moment_shift_bins"]
            for cell in cells
        )
        / len(cells)
        for name in names
    }
    primary = {
        name: all(
            cell["states"][name]["continuous"]["continuous_pass"]
            for cell in cells
        )
        for name in PRIMARY_NAMES
    }
    controls = {}
    for name in CONTROL_NAMES:
        any_failure = any(
            not cell["states"][name]["continuous"]["continuous_pass"]
            for cell in cells
        )
        margin = means[name] - means["source_predicted"]
        controls[name] = {
            "any_failure": any_failure,
            "aggregate_mean_shift_bins": means[name],
            "margin_over_source_predicted_bins": margin,
            "specific": bool(any_failure and margin >= specificity_margin_bins),
        }
    return {
        "aggregate_mean_shift_bins": means,
        "primary_all_fresh_cells_pass": primary,
        "all_primary_pass": all(primary.values()),
        "controls": controls,
        "all_controls_specific": all(value["specific"] for value in controls.values()),
    }


def classify_checkpoint(
    *, contracts_valid: bool, state_gates: Mapping[str, Any]
) -> tuple[str, bool]:
    if not contracts_valid:
        return "invalid", False
    primary = state_gates["primary_all_fresh_cells_pass"]
    if not primary["local_oracle"]:
        return "local_tangent_not_replicated", False
    full_gate = bool(state_gates["all_primary_pass"] and state_gates["all_controls_specific"])
    if full_gate:
        return "portable_source_covector_and_scalar", True
    if primary["source_predicted"]:
        return "nonspecific_source_correction", False
    covector = primary["source_covector_oracle_error"]
    scalar = primary["local_covector_source_error"]
    if covector and scalar:
        return "source_components_portable_joint_not", False
    if covector and not scalar:
        return "source_covector_portable_scalar_not", False
    if scalar and not covector:
        return "source_scalar_portable_covector_not", False
    if not covector and not scalar:
        return "local_oracle_only", False
    return "mixed_source_portability", False


def _campaign_decision(
    classifications: Sequence[str], pass_count: int, allow_underpowered: bool
) -> dict[str, Any]:
    if allow_underpowered:
        conclusion = "systems_lifecycle_only_not_quality_evidence"
        supported = False
    elif len(classifications) == 3 and pass_count == 3:
        conclusion = "supported_portable_source_covector_and_scalar_three_of_three"
        supported = True
    elif len(set(classifications)) == 1:
        conclusion = classifications[0]
        supported = False
    else:
        conclusion = "checkpoint_stratified_source_portability"
        supported = False
    return {
        "supported": supported,
        "source_covector_portability_pass_count": pass_count,
        "required_checkpoint_count": 3,
        "conclusion": conclusion,
    }


def _campaign_is_reusable(
    campaign: Mapping[str, Any],
    config: SourceTaskCovectorPortabilityConfig,
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
    config: SourceTaskCovectorPortabilityConfig, output: Path
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
    predecessor, predecessor_path, predecessor_details = local._load_predecessor(config)
    tangent_campaign, tangent_campaign_path = _load_local_tangent_provenance(config)
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
    for seed in config.seeds:
        prior, prior_path = predecessor_details[seed]
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
        if (
            decomposition._numeric_max_difference(basis_summary, prior["basis"])
            > config.replay_tolerance
        ):
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

        source_records = []
        source_tensors: list[dict[str, torch.Tensor]] = []
        for cohort in SOURCE_COHORTS:
            for regime in fixed.REGIMES:
                dataset = source_datasets[cohort][regime]
                cell = source_cells[cohort][regime]
                carrier, oracle_audit = decomposition.oracle_orbit_carrier(
                    dataset, task, config.orbit_count, device
                )
                features = capacity.fourier_features(carrier, config.writer_order)
                predicted = transport.apply_affine(features, writer)
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
                source_tensors.append(
                    {
                        "features": features.detach(),
                        **{name: value.detach() for name, value in derivative.items()},
                    }
                )
                source_records.append(
                    {
                        "cohort": cohort,
                        "regime": regime,
                        "evaluation_seed": fixed.COHORT_SEEDS[cohort][regime],
                        "oracle_audit": oracle_audit,
                    }
                )

        pooled_source = {
            name: torch.cat([record[name] for record in source_tensors], dim=0)
            for name in source_tensors[0]
        }
        source_linearization = jacobian.linearization_metrics(
            pooled_source["fine_gradient"],
            pooled_source["coarse_gradient"],
            pooled_source["predicted_delta"],
            pooled_source["observed_delta"],
            config,
        )
        maps = fit_source_maps(
            pooled_source["features"],
            pooled_source["fine_gradient"],
            pooled_source["observed_delta"],
            config.map_ridge,
        )
        source_gradient_prediction = transport.apply_affine(
            pooled_source["features"], maps["covector"]
        )
        source_error_prediction = transport.apply_affine(
            pooled_source["features"], maps["signed_error"]
        ).reshape(-1)
        source_fit = {
            "covector": regression_metrics(
                source_gradient_prediction, pooled_source["fine_gradient"]
            ),
            "signed_error": regression_metrics(
                source_error_prediction,
                pooled_source["observed_delta"],
                config.sign_magnitude_floor_bins,
            ),
        }
        source_mean_gradient = pooled_source["fine_gradient"].mean(0)

        fresh_records = []
        fresh_tensors: list[dict[str, torch.Tensor]] = []
        numerical_passes = []
        for regime in fixed.REGIMES:
            dataset = fresh_datasets[regime]
            cell = fresh_cells[regime]
            carrier, oracle_audit = decomposition.oracle_orbit_carrier(
                dataset, task, config.orbit_count, device
            )
            features = capacity.fourier_features(carrier, config.writer_order)
            predicted = transport.apply_affine(features, writer)
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
            predicted_gradient = transport.apply_affine(features, maps["covector"])
            predicted_error = transport.apply_affine(
                features, maps["signed_error"]
            ).reshape(-1)
            local_oracle_std = task_inverse_correction(
                derivative["fine_gradient"],
                derivative["observed_delta"],
                config.gradient_denominator_floor,
            )
            source_covector_oracle_std = task_inverse_correction(
                predicted_gradient,
                derivative["observed_delta"],
                config.gradient_denominator_floor,
            )
            local_covector_source_std = task_inverse_correction(
                derivative["fine_gradient"],
                predicted_error,
                config.gradient_denominator_floor,
            )
            source_predicted_std = task_inverse_correction(
                predicted_gradient,
                predicted_error,
                config.gradient_denominator_floor,
            )
            source_mean = source_mean_gradient.unsqueeze(0).expand_as(predicted_gradient)
            source_mean_std = task_inverse_correction(
                source_mean,
                predicted_error,
                config.gradient_denominator_floor,
            )
            control_seed = (
                95_000_000 + 100_003 * seed + FRESH_COHORT_SEEDS[regime]
            )
            permutation = fixed_permutation(config.orbit_count, control_seed, device)
            shuffled_std = task_inverse_correction(
                predicted_gradient,
                predicted_error[permutation],
                config.gradient_denominator_floor,
            )
            random_std = norm_matched_random(source_predicted_std, control_seed + 1)
            scale = coordinate_scale.double().unsqueeze(0)
            coordinates = {
                "direct_rank3": target,
                "order4": predicted,
                "local_oracle": predicted + local_oracle_std * scale,
                "source_covector_oracle_error": predicted
                + source_covector_oracle_std * scale,
                "local_covector_source_error": predicted
                + local_covector_source_std * scale,
                "source_predicted": predicted + source_predicted_std * scale,
                "source_mean_covector": predicted + source_mean_std * scale,
                "source_shuffled_error": predicted + shuffled_std * scale,
                "source_flipped": predicted - source_predicted_std * scale,
                "source_random_direction": predicted + random_std * scale,
            }
            states = local.evaluate_science_states(
                system, task, config, cell, basis, coordinates
            )
            finite = all(
                torch.isfinite(value).all()
                for value in (
                    features,
                    derivative["fine_gradient"],
                    derivative["coarse_gradient"],
                    predicted_gradient,
                    predicted_error,
                    source_predicted_std,
                    random_std,
                )
            )
            numerical = bool(
                finite
                and oracle_audit["mean_shift_bins"] <= 1e-8
                and oracle_audit["p95_shift_bins"] <= 1e-8
            )
            numerical_passes.append(numerical)
            fresh_tensors.append(
                {
                    "fine_gradient": derivative["fine_gradient"].detach(),
                    "coarse_gradient": derivative["coarse_gradient"].detach(),
                    "predicted_delta": derivative["predicted_delta"].detach(),
                    "observed_delta": derivative["observed_delta"].detach(),
                    "predicted_gradient": predicted_gradient.detach(),
                    "predicted_error": predicted_error.detach(),
                }
            )
            fresh_records.append(
                {
                    "cohort": FRESH_COHORT,
                    "regime": regime,
                    "evaluation_seed": FRESH_COHORT_SEEDS[regime],
                    "oracle_audit": oracle_audit,
                    "coordinate_metrics": transport.coordinate_metrics(predicted, target),
                    "map_diagnostics": {
                        "covector": regression_metrics(
                            predicted_gradient, derivative["fine_gradient"]
                        ),
                        "signed_error": regression_metrics(
                            predicted_error,
                            derivative["observed_delta"],
                            config.sign_magnitude_floor_bins,
                        ),
                        "mean_source_predicted_correction_norm_std": float(
                            torch.linalg.vector_norm(source_predicted_std, dim=1).mean()
                        ),
                        "mean_local_oracle_correction_norm_std": float(
                            torch.linalg.vector_norm(local_oracle_std, dim=1).mean()
                        ),
                    },
                    "states": states,
                    "numerical_contract": numerical,
                    "control_permutation_sha256": hashlib.sha256(
                        permutation.detach().cpu().numpy().tobytes()
                    ).hexdigest(),
                }
            )

        pooled_fresh = {
            name: torch.cat([record[name] for record in fresh_tensors], dim=0)
            for name in fresh_tensors[0]
        }
        fresh_linearization = jacobian.linearization_metrics(
            pooled_fresh["fine_gradient"],
            pooled_fresh["coarse_gradient"],
            pooled_fresh["predicted_delta"],
            pooled_fresh["observed_delta"],
            config,
        )
        fresh_map_diagnostics = {
            "covector": regression_metrics(
                pooled_fresh["predicted_gradient"],
                pooled_fresh["fine_gradient"],
            ),
            "signed_error": regression_metrics(
                pooled_fresh["predicted_error"],
                pooled_fresh["observed_delta"],
                config.sign_magnitude_floor_bins,
            ),
        }
        target_controls = all(
            not cell["states"]["zero"]["continuous"]["continuous_pass"]
            and cell["states"]["exact"]["continuous"]["continuous_pass"]
            and cell["states"]["direct_rank3"]["continuous"]["continuous_pass"]
            for cell in fresh_records
        )
        numerical_pass = bool(
            all(numerical_passes)
            and bool(torch.isfinite(coordinate_scale).all())
            and float(coordinate_scale.min()) > config.coordinate_scale_floor
            and all(
                torch.isfinite(value).all()
                for mapping in maps.values()
                for value in mapping.values()
            )
        )
        state_gates = state_gate_summary(
            fresh_records, config.specificity_margin_bins
        )
        contracts_valid = bool(
            numerical_pass
            and target_controls
            and source_linearization["adequate"]
            and fresh_linearization["adequate"]
        )
        classification, portability_gate = classify_checkpoint(
            contracts_valid=contracts_valid, state_gates=state_gates
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
            "experiment_id": f"tinyllm-c2-source-task-covector-portability-seed{seed}",
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
                "local_tangent_campaign": str(tangent_campaign_path),
                "local_tangent_campaign_sha256": _sha256(tangent_campaign_path),
                "checkpoint": provenance["checkpoint"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "frontend_checkpoint": provenance["frontend_checkpoint"],
                "frontend_checkpoint_sha256": provenance["frontend_checkpoint_sha256"],
                "character_result": str(frozen_path),
                "character_result_sha256": _sha256(frozen_path),
            },
            "basis": basis_summary,
            "coordinate_scale": coordinate_scale.detach().cpu().tolist(),
            "source_map_fit_cells": source_records,
            "source_maps": {
                name: _mapping_summary(mapping) for name, mapping in maps.items()
            },
            "source_fit": source_fit,
            "source_linearization": source_linearization,
            "fresh_cells": fresh_records,
            "fresh_map_diagnostics": fresh_map_diagnostics,
            "fresh_linearization": fresh_linearization,
            "state_gates": state_gates,
            "gates": {
                "provenance_contract": True,
                "numerical_contract": numerical_pass,
                "continuous_target_control_contract": target_controls,
                "source_local_linearization_adequate": source_linearization["adequate"],
                "fresh_local_linearization_adequate": fresh_linearization["adequate"],
                "local_oracle_all_fresh_cells_pass": state_gates[
                    "primary_all_fresh_cells_pass"
                ]["local_oracle"],
                "source_covector_oracle_error_all_fresh_cells_pass": state_gates[
                    "primary_all_fresh_cells_pass"
                ]["source_covector_oracle_error"],
                "local_covector_source_error_all_fresh_cells_pass": state_gates[
                    "primary_all_fresh_cells_pass"
                ]["local_covector_source_error"],
                "source_predicted_all_fresh_cells_pass": state_gates[
                    "primary_all_fresh_cells_pass"
                ]["source_predicted"],
                "all_controls_specific": state_gates["all_controls_specific"],
                "source_covector_portability_gate": portability_gate,
            },
            "classification": classification,
            "primary_metric": float(portability_gate),
            "analysis_seconds": time.perf_counter() - started,
            "artifacts": {"result": str(result_path)},
        }
        _write_json(result_path, result)
        results.append(result)
        print(f"seed {seed}: {classification}", flush=True)
        if _implementation_digest() != implementation:
            raise RuntimeError("source-covector implementation changed during campaign")

    classifications = [result["classification"] for result in results]
    pass_count = sum(
        result["gates"]["source_covector_portability_gate"] for result in results
    )
    decision = _campaign_decision(
        classifications, pass_count, config.allow_underpowered
    )
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
            "predecessor_campaign": str(predecessor_path),
            "predecessor_campaign_sha256": _sha256(predecessor_path),
            "predecessor_implementation_sha256": predecessor["implementation_sha256"],
            "local_tangent_campaign": str(tangent_campaign_path),
            "local_tangent_campaign_sha256": _sha256(tangent_campaign_path),
            "local_tangent_implementation_sha256": tangent_campaign[
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
            "fitted_predictive_observers": len(config.seeds) * 2,
            "fitted_writers": 0,
            "fresh_primary_cells": len(config.seeds) * len(fixed.REGIMES),
            "source_map_fit_orbits": len(config.seeds)
            * len(SOURCE_COHORTS)
            * len(fixed.REGIMES)
            * config.orbit_count,
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
                    "continuous_target_control_contract",
                    "source_local_linearization_adequate",
                    "fresh_local_linearization_adequate",
                    "local_oracle_all_fresh_cells_pass",
                    "source_covector_oracle_error_all_fresh_cells_pass",
                    "local_covector_source_error_all_fresh_cells_pass",
                    "source_predicted_all_fresh_cells_pass",
                    "all_controls_specific",
                    "source_covector_portability_gate",
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
            "The covector is conditioned on the frozen decoder and circular answer angle.",
            "The source maps consume an oracle quotient-phase chart inherited from the predecessor.",
            "Source cohorts and checkpoints were selected after prior outcomes were known.",
            "Fresh C changes generator seeds, not the declared shift families.",
            "Finite differences and rank-three writes are local off-manifold interventions.",
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
            "data/experiments/tinyllm_source_task_covector_portability/"
            "20260807_d6_preregistered_fresh_cohort"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=_ints, default=fixed.PRIMARY_SEEDS)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = SourceTaskCovectorPortabilityConfig(
        seeds=args.seeds,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
