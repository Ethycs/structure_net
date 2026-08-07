#!/usr/bin/env python3
"""Test whether task-cut activations predict TinyLLM's missing scalar correction."""

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

import experiments.structure_net.tinyllm_source_task_covector_portability as source
import experiments.structure_net.tinyllm_fixed_gauge_writer_capacity as capacity
import experiments.structure_net.tinyllm_fixed_semantic_gauge_writer as fixed
import experiments.structure_net.tinyllm_cross_seed_causal_carrier_transport as transport
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-c2-task-activation-scalar-sensor.v1"
HYPOTHESIS_ID = "tinyllm-c2-task-activation-scalar-sensor-v1"
SUPERSEDED_BY = "tinyllm-c2-observable-scalar-residual-v1"
SUPERSEDED_REASON = (
    "superseded before outcome: the observable-residual protocol owns fresh-D "
    "seeds 530007/530008, which must not be spent twice"
)
SOURCE_CAMPAIGN_SHA256 = (
    "fe153abd0aa1749a862095e577e6e9cf8a0f7054bade2ef3c6833bf32d7f4ac5"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "6716b909d0c245059a1ed1310f20f4d9e56deb8c49a7d3a031972542fccb3046"
)
SOURCE_COHORTS = ("heldout_a", "heldout_b")
FRESH_COHORT = "heldout_d"
FRESH_COHORT_SEEDS = {"composition": 530_007, "extrapolation": 530_008}
PCA_NAMES = ("prewrite_activation", "predicted_activation", "post_mlp_activation")
SCALAR_ARMS = (
    "phase_only",
    "calibration",
    "prewrite_activation",
    "predicted_activation",
    "causal_combined",
    "post_mlp_lookahead",
    "output_lookahead",
    "full_lookahead",
)
LOOKAHEAD_ARMS = ("post_mlp_lookahead", "output_lookahead", "full_lookahead")
PRIMARY_ARM = "causal_combined"
CONTROL_NAMES = (
    "causal_shuffled",
    "causal_flipped",
    "causal_random_direction",
)


@dataclass(frozen=True)
class TaskActivationScalarSensorConfig:
    source_root: str = (
        "data/experiments/tinyllm_source_task_covector_portability/"
        "20260807_d6_preregistered_fresh_cohort"
    )
    seeds: tuple[int, ...] = fixed.PRIMARY_SEEDS
    orbit_count: int = 64
    carrier_rank: int = 3
    writer_order: int = 4
    pca_rank: int = 8
    scalar_ridge: float = 1e-3
    fine_step_std: float = 0.025
    coarse_step_std: float = 0.05
    coordinate_scale_floor: float = 1e-8
    gradient_denominator_floor: float = 1e-12
    replay_tolerance: float = 1e-6
    derivative_cosine_floor: float = 0.98
    derivative_relative_l2_ceiling: float = 0.15
    signed_error_r2_floor: float = 0.50
    residual_mae_fraction_ceiling: float = 0.50
    signed_error_relative_l2_ceiling: float = math.sqrt(0.50)
    sign_agreement_floor: float = 0.75
    sign_magnitude_floor_bins: float = 0.01
    pca_orthogonality_tolerance: float = 1e-8
    specificity_margin_bins: float = 0.05
    activation_batch_size: int = 256
    continuation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != fixed.PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("primary activation-sensor seeds are fixed to 7,29,53")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.orbit_count != 64:
            raise ValueError("the activation-sensor cohort fixes 64 exact orbits")
        if self.carrier_rank != 3 or self.writer_order != 4:
            raise ValueError("carrier rank and writer order are fixed to three and four")
        if self.pca_rank != 8:
            raise ValueError("activation summaries fix eight source PCA components")
        if not 0.0 < self.fine_step_std < self.coarse_step_std:
            raise ValueError("finite-difference steps must be positive and ordered")
        for name in (
            "scalar_ridge",
            "coordinate_scale_floor",
            "gradient_denominator_floor",
            "replay_tolerance",
            "pca_orthogonality_tolerance",
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
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
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
        Path(source.__file__),
        Path(source.local.__file__),
        Path(source.jacobian.__file__),
        Path(source.decomposition.__file__),
        Path(capacity.__file__),
        Path(fixed.__file__),
        Path(transport.__file__),
        Path(transport.rank.__file__),
        Path(transport.coupling.__file__),
        Path(source.local.readout.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evidence_role(config: TaskActivationScalarSensorConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_fresh_d_post_outcome_underpowered_mechanistic_evidence"
    )


def _tensor_digest(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def fit_pca(values: torch.Tensor, rank: int) -> dict[str, Any]:
    """Fit a deterministic source-only principal-component summary."""
    values = values.double()
    if values.ndim != 2 or rank <= 0 or rank > min(values.shape):
        raise ValueError("PCA values and rank are incompatible")
    mean = values.mean(0)
    centered = values - mean
    _, singular, right_h = torch.linalg.svd(centered, full_matrices=False)
    components = right_h[:rank]
    total = singular.square().sum().clamp_min(1e-24)
    energy = singular[:rank].square() / total
    identity = torch.eye(rank, dtype=torch.float64, device=values.device)
    orthogonality = float(torch.linalg.matrix_norm(components @ components.T - identity))
    return {
        "mean": mean,
        "components": components,
        "singular_values": singular[:rank],
        "explained_energy": energy,
        "cumulative_energy": float(energy.sum()),
        "orthogonality_error": orthogonality,
    }


def apply_pca(values: torch.Tensor, mapping: Mapping[str, Any]) -> torch.Tensor:
    values = values.double()
    return (values - mapping["mean"]) @ mapping["components"].T


def fit_scalar_map(
    features: torch.Tensor, target: torch.Tensor, ridge: float
) -> dict[str, torch.Tensor]:
    """Fit a source-standardized scalar ridge map with an explicit intercept."""
    features = features.double()
    target = target.reshape(-1, 1).double()
    if features.ndim != 2 or len(features) != len(target):
        raise ValueError("scalar-map features and target must be row aligned")
    mean = features.mean(0)
    scale = features.std(0, unbiased=False).clamp_min(1e-8)
    standardized = (features - mean) / scale
    target_mean = target.mean(0)
    centered_target = target - target_mean
    identity = torch.eye(
        standardized.shape[1], dtype=torch.float64, device=features.device
    )
    linear = torch.linalg.solve(
        standardized.T @ standardized + ridge * identity,
        standardized.T @ centered_target,
    )
    return {
        "mean": mean,
        "scale": scale,
        "linear": linear,
        "intercept": target_mean,
    }


def apply_scalar_map(
    features: torch.Tensor, mapping: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    standardized = (features.double() - mapping["mean"]) / mapping["scale"]
    return (standardized @ mapping["linear"] + mapping["intercept"]).reshape(-1)


def scalar_metrics(
    predicted: torch.Tensor, target: torch.Tensor, sign_floor: float
) -> dict[str, Any]:
    output = source.regression_metrics(predicted, target, sign_floor)
    residual = predicted.reshape(-1).double() - target.reshape(-1).double()
    denominator = torch.linalg.vector_norm(target.reshape(-1).double()).clamp_min(1e-24)
    output["relative_l2"] = float(torch.linalg.vector_norm(residual) / denominator)
    output["prediction_to_target_rms_ratio"] = float(
        torch.sqrt(predicted.double().square().mean())
        / torch.sqrt(target.double().square().mean()).clamp_min(1e-24)
    )
    return output


def output_features(posterior: torch.Tensor) -> torch.Tensor:
    """Return target-free task-posterior and confidence features."""
    posterior = posterior.double()
    probabilities = posterior / posterior.sum(1, keepdim=True).clamp_min(1e-24)
    entropy = -(probabilities * probabilities.clamp_min(1e-24).log()).sum(1)
    top = torch.topk(probabilities, k=2, dim=1).values
    angle, radius = source.local.readout.posterior_moment(probabilities)
    summaries = torch.stack(
        (
            entropy,
            top[:, 0],
            top[:, 0] - top[:, 1],
            torch.cos(angle),
            torch.sin(angle),
            radius,
        ),
        dim=1,
    )
    return torch.cat((probabilities, summaries), dim=1)


@torch.no_grad()
def observable_tensors(
    system: Any,
    task: CircleTaskConfig,
    config: TaskActivationScalarSensorConfig,
    cell: Mapping[str, Any],
    dataset: Any,
    basis: torch.Tensor,
    predicted: torch.Tensor,
    phase_features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Extract only the registered target-free activation and output summaries."""
    predicted_state = source.local._state_from_coordinates(cell, basis, predicted)
    block = system.model.transformer["h"][0]
    post_mlp = predicted_state + block.mlp(block.ln_2(predicted_state))
    posterior = source.local._continue_states(
        system,
        task,
        config,
        cell["target_cut"],
        {"order4": predicted_state},
    )["order4"]
    calibration = (
        dataset.calibration.to(predicted.device)
        .reshape(config.orbit_count, 2, -1)
        .double()
        .mean(1)
    )
    return {
        "phase_only": phase_features.double(),
        "calibration": calibration,
        "prewrite_activation": cell["propagated"][:, -1].double(),
        "predicted_activation": predicted_state[:, -1].double(),
        "post_mlp_activation": post_mlp[:, -1].double(),
        "output": output_features(posterior),
    }


def feature_arms(
    raw: Mapping[str, torch.Tensor],
    pca: Mapping[str, Mapping[str, Any]],
) -> dict[str, torch.Tensor]:
    prewrite = apply_pca(raw["prewrite_activation"], pca["prewrite_activation"])
    predicted = apply_pca(
        raw["predicted_activation"], pca["predicted_activation"]
    )
    post_mlp = apply_pca(
        raw["post_mlp_activation"], pca["post_mlp_activation"]
    )
    causal = torch.cat((raw["calibration"], prewrite, predicted), dim=1)
    return {
        "phase_only": raw["phase_only"],
        "calibration": raw["calibration"],
        "prewrite_activation": prewrite,
        "predicted_activation": predicted,
        "causal_combined": causal,
        "post_mlp_lookahead": torch.cat((causal, post_mlp), dim=1),
        "output_lookahead": torch.cat((causal, raw["output"]), dim=1),
        "full_lookahead": torch.cat((causal, post_mlp, raw["output"]), dim=1),
    }


def _mapping_summary(mapping: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {name: value.detach().cpu().tolist() for name, value in mapping.items()}


def _pca_summary(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rank": int(mapping["components"].shape[0]),
        "features": int(mapping["components"].shape[1]),
        "singular_values": mapping["singular_values"].detach().cpu().tolist(),
        "explained_energy": mapping["explained_energy"].detach().cpu().tolist(),
        "cumulative_energy": mapping["cumulative_energy"],
        "orthogonality_error": mapping["orthogonality_error"],
        "mapping_sha256": _tensor_digest(mapping["mean"], mapping["components"]),
    }


def _load_source_campaign(
    config: TaskActivationScalarSensorConfig,
) -> tuple[dict[str, Any], Path, dict[int, tuple[dict[str, Any], Path]]]:
    path = Path(config.source_root) / "campaign_results.json"
    campaign = json.loads(path.read_text(encoding="utf-8"))
    expected_classes = {str(seed): "source_covector_portable_scalar_not" for seed in fixed.PRIMARY_SEEDS}
    if (
        _sha256(path) != SOURCE_CAMPAIGN_SHA256
        or campaign.get("schema_version") != source.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != source.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256") != SOURCE_IMPLEMENTATION_SHA256
        or campaign.get("aggregates", {}).get("classification_by_seed")
        != expected_classes
        or int(campaign.get("summary", {}).get("trained_models", -1)) != 0
    ):
        raise ValueError(f"invalid source-covector campaign {path}")
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if not set(config.seeds).issubset(entries):
        raise ValueError("source-covector campaign lacks a requested checkpoint")
    details: dict[int, tuple[dict[str, Any], Path]] = {}
    for seed in config.seeds:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        if (
            _sha256(detail_path) != entry.get("result_sha256")
            or detail.get("schema_version") != source.SCHEMA_VERSION
            or detail.get("hypothesis_id") != source.HYPOTHESIS_ID
            or detail.get("implementation_sha256") != SOURCE_IMPLEMENTATION_SHA256
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or detail.get("classification") != "source_covector_portable_scalar_not"
            or detail.get("gates", {}).get(
                "source_covector_oracle_error_all_fresh_cells_pass"
            )
            is not True
            or detail.get("gates", {}).get("source_predicted_all_fresh_cells_pass")
            is not False
        ):
            raise ValueError(f"invalid source-covector result {detail_path}")
        details[seed] = (detail, detail_path)
    return campaign, path, details


def _source_config(config: TaskActivationScalarSensorConfig) -> Any:
    return source.SourceTaskCovectorPortabilityConfig(
        seeds=config.seeds,
        orbit_count=config.orbit_count,
        carrier_rank=config.carrier_rank,
        writer_order=config.writer_order,
        fine_step_std=config.fine_step_std,
        coarse_step_std=config.coarse_step_std,
        coordinate_scale_floor=config.coordinate_scale_floor,
        gradient_denominator_floor=config.gradient_denominator_floor,
        replay_tolerance=config.replay_tolerance,
        derivative_cosine_floor=config.derivative_cosine_floor,
        derivative_relative_l2_ceiling=config.derivative_relative_l2_ceiling,
        sign_agreement_floor=config.sign_agreement_floor,
        sign_magnitude_floor_bins=config.sign_magnitude_floor_bins,
        activation_batch_size=config.activation_batch_size,
        continuation_batch_size=config.continuation_batch_size,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )


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


def _fingerprint(
    config: TaskActivationScalarSensorConfig,
    seed: int,
    source_result_sha256: str,
    checkpoint_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
        "source_cohort_seeds": fixed.COHORT_SEEDS,
        "fresh_cohort_seeds": FRESH_COHORT_SEEDS,
        "seed": seed,
        "source_result_sha256": source_result_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prediction_pass(
    metrics: Mapping[str, Any], config: TaskActivationScalarSensorConfig
) -> bool:
    return bool(
        metrics["zero_referenced_r2"] >= config.signed_error_r2_floor
        and metrics["sign_agreement"] >= config.sign_agreement_floor
        and metrics["relative_l2"] <= config.signed_error_relative_l2_ceiling
    )


def causal_gate_summary(
    cells: Sequence[Mapping[str, Any]], config: TaskActivationScalarSensorConfig
) -> dict[str, Any]:
    names = (
        "order4",
        "local_oracle",
        "source_covector_oracle_error",
        *SCALAR_ARMS,
        *CONTROL_NAMES,
    )
    means = {
        name: sum(
            cell["states"][name]["continuous"]["mean_moment_shift_bins"]
            for cell in cells
        )
        / len(cells)
        for name in names
    }
    all_cells_pass = {
        name: all(cell["states"][name]["continuous"]["continuous_pass"] for cell in cells)
        for name in (
            "local_oracle",
            "source_covector_oracle_error",
            *SCALAR_ARMS,
        )
    }
    controls: dict[str, Any] = {}
    for name in CONTROL_NAMES:
        any_failure = any(
            not cell["states"][name]["continuous"]["continuous_pass"] for cell in cells
        )
        margin = means[name] - means[PRIMARY_ARM]
        controls[name] = {
            "any_failure": any_failure,
            "aggregate_mean_shift_bins": means[name],
            "margin_over_primary_bins": margin,
            "specific": bool(any_failure and margin >= config.specificity_margin_bins),
        }
    primary = bool(
        all_cells_pass["local_oracle"]
        and all_cells_pass["source_covector_oracle_error"]
        and all_cells_pass[PRIMARY_ARM]
        and all(record["specific"] for record in controls.values())
    )
    return {
        "aggregate_mean_shift_bins": means,
        "all_fresh_cells_pass": all_cells_pass,
        "controls": controls,
        "all_controls_specific": all(record["specific"] for record in controls.values()),
        "primary_causal_scalar_gate": primary,
    }


def classify_checkpoint(
    *,
    valid: bool,
    predictive_pass: bool,
    causal_pass: bool,
    lookahead_pass: bool,
) -> tuple[str, bool]:
    if not valid:
        return "invalid", False
    if predictive_pass and causal_pass:
        return "causal_activation_scalar_supported", True
    if predictive_pass:
        return "causal_activation_predictive_not_causal", False
    if causal_pass:
        return "causal_activation_causal_not_predictive", False
    if lookahead_pass:
        return "lookahead_scalar_only", False
    return "observable_scalar_not_identified", False


def _campaign_is_reusable(
    value: Mapping[str, Any],
    config: TaskActivationScalarSensorConfig,
    implementation: str,
) -> bool:
    if not (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("evidence_role") == _evidence_role(config)
        and value.get("implementation_sha256") == implementation
        and value.get("configuration") == _json_compatible(asdict(config))
        and int(value.get("summary", {}).get("completed", -1)) == len(config.seeds)
    ):
        return False
    for entry in value.get("results", []):
        result = Path(entry.get("path", ""))
        arrays = Path(entry.get("arrays", ""))
        if (
            not result.is_file()
            or not arrays.is_file()
            or _sha256(result) != entry.get("result_sha256")
            or _sha256(arrays) != entry.get("arrays_sha256")
        ):
            return False
    return len(value.get("results", [])) == len(config.seeds)


def _reusable_seed_result(
    result_path: Path,
    arrays_path: Path,
    config: TaskActivationScalarSensorConfig,
    implementation: str,
    seed: int,
    fingerprint: str,
) -> dict[str, Any] | None:
    if not result_path.is_file():
        return None
    existing = json.loads(result_path.read_text(encoding="utf-8"))
    valid = bool(
        existing.get("schema_version") == SCHEMA_VERSION
        and existing.get("hypothesis_id") == HYPOTHESIS_ID
        and existing.get("status") == "completed"
        and existing.get("evidence_role") == _evidence_role(config)
        and existing.get("implementation_sha256") == implementation
        and existing.get("scientific_fingerprint") == fingerprint
        and existing.get("configuration") == _json_compatible(asdict(config))
        and int(existing.get("seed", -1)) == seed
        and existing.get("artifacts", {}).get("result") == str(result_path)
        and existing.get("artifacts", {}).get("arrays") == str(arrays_path)
        and arrays_path.is_file()
        and _sha256(arrays_path)
        == existing.get("artifacts", {}).get("arrays_sha256")
    )
    if not valid:
        raise ValueError(f"incompatible completed result {result_path}; use a new root")
    return existing


def run_campaign(
    config: TaskActivationScalarSensorConfig, output: Path
) -> dict[str, Any]:
    raise RuntimeError(f"{SUPERSEDED_REASON}; use {SUPERSEDED_BY}")


@torch.no_grad()
def _run_superseded_campaign_implementation(
    config: TaskActivationScalarSensorConfig, output: Path
) -> dict[str, Any]:
    """Preserve the never-executed implementation as an inert historical record."""
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    implementation = _implementation_digest()
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if _campaign_is_reusable(existing, config, implementation):
            print("aggregate already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed aggregate {campaign_path}")

    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    source_campaign, source_path, source_details = _load_source_campaign(config)
    predecessor_config = _source_config(config)
    _, writer_campaign_path, writer_details = source.local._load_predecessor(
        predecessor_config
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
        for cohort in SOURCE_COHORTS
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

    results: list[dict[str, Any]] = []
    reused = 0
    for seed in config.seeds:
        seed_started = time.perf_counter()
        source_detail, source_result_path = source_details[seed]
        fingerprint = _fingerprint(
            config,
            seed,
            _sha256(source_result_path),
            source_detail["provenance"]["checkpoint_sha256"],
        )
        result_path = output / "runs" / f"seed_{seed}" / "result.json"
        arrays_path = output / "runs" / f"seed_{seed}" / "activation_sensor_arrays.npz"
        existing = _reusable_seed_result(
            result_path,
            arrays_path,
            config,
            implementation,
            seed,
            fingerprint,
        )
        if existing is not None:
            results.append(existing)
            reused += 1
            print(f"resuming {existing['experiment_id']}", flush=True)
            continue

        writer_detail, writer_result_path = writer_details[seed]
        system, provenance = transport.rank.deck.load_source(
            task, bridge, 2, seed, device
        )
        if provenance["checkpoint_sha256"] != source_detail["provenance"]["checkpoint_sha256"]:
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
        if source.decomposition._numeric_max_difference(
            basis_summary, source_detail["basis"]
        ) > config.replay_tolerance:
            raise ValueError(f"basis summary mismatch for seed {seed}")
        coordinate_scale = torch.tensor(
            source_detail["coordinate_scale"], dtype=torch.float64, device=device
        )
        writer = capacity._writer_from_summary(
            writer_detail["alignment_fit"]["writers"]["quotient_order4"], device
        )
        covector_map = capacity._writer_from_summary(
            source_detail["source_maps"]["covector"], device
        )

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
            for cohort in SOURCE_COHORTS
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

        source_rows: list[dict[str, torch.Tensor]] = []
        source_records: list[dict[str, Any]] = []
        for cohort in SOURCE_COHORTS:
            for regime in fixed.REGIMES:
                dataset = source_datasets[cohort][regime]
                cell = source_cells[cohort][regime]
                carrier, oracle_audit = source.decomposition.oracle_orbit_carrier(
                    dataset, task, config.orbit_count, device
                )
                phase_features = capacity.fourier_features(carrier, config.writer_order)
                predicted = transport.apply_affine(phase_features, writer)
                target = transport._coordinates(cell, basis)
                derivative = source.local.finite_difference_cell(
                    system,
                    task,
                    config,
                    cell,
                    basis,
                    predicted,
                    target,
                    coordinate_scale,
                )
                observable = observable_tensors(
                    system,
                    task,
                    config,
                    cell,
                    dataset,
                    basis,
                    predicted,
                    phase_features,
                )
                source_rows.append(
                    {
                        **{name: value.detach() for name, value in observable.items()},
                        "fine_gradient": derivative["fine_gradient"].detach(),
                        "coarse_gradient": derivative["coarse_gradient"].detach(),
                        "predicted_delta": derivative["predicted_delta"].detach(),
                        "observed_delta": derivative["observed_delta"].detach(),
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
            name: torch.cat([row[name] for row in source_rows], dim=0)
            for name in source_rows[0]
        }
        source_linearization = source.jacobian.linearization_metrics(
            pooled_source["fine_gradient"],
            pooled_source["coarse_gradient"],
            pooled_source["predicted_delta"],
            pooled_source["observed_delta"],
            config,
        )
        pca = {
            name: fit_pca(pooled_source[name], config.pca_rank) for name in PCA_NAMES
        }
        source_arm_features = feature_arms(pooled_source, pca)
        scalar_maps = {
            name: fit_scalar_map(
                source_arm_features[name],
                pooled_source["observed_delta"],
                config.scalar_ridge,
            )
            for name in SCALAR_ARMS
        }
        source_predictions = {
            name: apply_scalar_map(source_arm_features[name], scalar_maps[name])
            for name in SCALAR_ARMS
        }
        source_fit = {
            name: scalar_metrics(
                source_predictions[name],
                pooled_source["observed_delta"],
                config.sign_magnitude_floor_bins,
            )
            for name in SCALAR_ARMS
        }

        fresh_records: list[dict[str, Any]] = []
        fresh_tensors: list[dict[str, torch.Tensor]] = []
        array_payload: dict[str, np.ndarray] = {}
        numerical_passes: list[bool] = []
        for regime in fixed.REGIMES:
            dataset = fresh_datasets[regime]
            cell = fresh_cells[regime]
            carrier, oracle_audit = source.decomposition.oracle_orbit_carrier(
                dataset, task, config.orbit_count, device
            )
            phase_features = capacity.fourier_features(carrier, config.writer_order)
            predicted = transport.apply_affine(phase_features, writer)
            target = transport._coordinates(cell, basis)
            derivative = source.local.finite_difference_cell(
                system,
                task,
                config,
                cell,
                basis,
                predicted,
                target,
                coordinate_scale,
            )
            predicted_gradient = transport.apply_affine(phase_features, covector_map)
            observable = observable_tensors(
                system,
                task,
                config,
                cell,
                dataset,
                basis,
                predicted,
                phase_features,
            )
            arms = feature_arms(observable, pca)
            scalar_predictions = {
                name: apply_scalar_map(arms[name], scalar_maps[name])
                for name in SCALAR_ARMS
            }
            local_oracle_std = source.task_inverse_correction(
                derivative["fine_gradient"],
                derivative["observed_delta"],
                config.gradient_denominator_floor,
            )
            covector_oracle_std = source.task_inverse_correction(
                predicted_gradient,
                derivative["observed_delta"],
                config.gradient_denominator_floor,
            )
            arm_corrections = {
                name: source.task_inverse_correction(
                    predicted_gradient,
                    scalar_predictions[name],
                    config.gradient_denominator_floor,
                )
                for name in SCALAR_ARMS
            }
            control_seed = 115_000_000 + 100_003 * seed + FRESH_COHORT_SEEDS[regime]
            permutation = source.fixed_permutation(
                config.orbit_count, control_seed, device
            )
            shuffled_std = source.task_inverse_correction(
                predicted_gradient,
                scalar_predictions[PRIMARY_ARM][permutation],
                config.gradient_denominator_floor,
            )
            primary_std = arm_corrections[PRIMARY_ARM]
            random_std = source.norm_matched_random(primary_std, control_seed + 1)
            scale = coordinate_scale.double().unsqueeze(0)
            coordinates = {
                "direct_rank3": target,
                "order4": predicted,
                "local_oracle": predicted + local_oracle_std * scale,
                "source_covector_oracle_error": predicted + covector_oracle_std * scale,
                **{
                    name: predicted + arm_corrections[name] * scale
                    for name in SCALAR_ARMS
                },
                "causal_shuffled": predicted + shuffled_std * scale,
                "causal_flipped": predicted - primary_std * scale,
                "causal_random_direction": predicted + random_std * scale,
            }
            states = source.local.evaluate_science_states(
                system, task, config, cell, basis, coordinates
            )
            finite = all(
                bool(torch.isfinite(value).all())
                for value in (
                    *observable.values(),
                    *arms.values(),
                    derivative["fine_gradient"],
                    derivative["coarse_gradient"],
                    predicted_gradient,
                    *scalar_predictions.values(),
                    primary_std,
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
                    **{
                        f"prediction__{name}": value.detach()
                        for name, value in scalar_predictions.items()
                    },
                }
            )
            fresh_records.append(
                {
                    "cohort": FRESH_COHORT,
                    "regime": regime,
                    "evaluation_seed": FRESH_COHORT_SEEDS[regime],
                    "oracle_audit": oracle_audit,
                    "coordinate_metrics": transport.coordinate_metrics(predicted, target),
                    "scalar_diagnostics": {
                        name: scalar_metrics(
                            scalar_predictions[name],
                            derivative["observed_delta"],
                            config.sign_magnitude_floor_bins,
                        )
                        for name in SCALAR_ARMS
                    },
                    "covector_diagnostics": source.regression_metrics(
                        predicted_gradient, derivative["fine_gradient"]
                    ),
                    "states": states,
                    "numerical_contract": numerical,
                    "control_permutation_sha256": hashlib.sha256(
                        permutation.detach().cpu().numpy().tobytes()
                    ).hexdigest(),
                }
            )
            array_payload[f"{regime}__signed_error"] = (
                derivative["observed_delta"].detach().cpu().numpy()
            )
            array_payload[f"{regime}__predicted_covector"] = (
                predicted_gradient.detach().cpu().numpy()
            )
            for name, prediction in scalar_predictions.items():
                array_payload[f"{regime}__prediction__{name}"] = (
                    prediction.detach().cpu().numpy()
                )

        pooled_fresh = {
            name: torch.cat([row[name] for row in fresh_tensors], dim=0)
            for name in fresh_tensors[0]
        }
        fresh_linearization = source.jacobian.linearization_metrics(
            pooled_fresh["fine_gradient"],
            pooled_fresh["coarse_gradient"],
            pooled_fresh["predicted_delta"],
            pooled_fresh["observed_delta"],
            config,
        )
        fresh_diagnostics = {
            "covector": source.regression_metrics(
                pooled_fresh["predicted_gradient"], pooled_fresh["fine_gradient"]
            ),
            "scalar_arms": {
                name: scalar_metrics(
                    pooled_fresh[f"prediction__{name}"],
                    pooled_fresh["observed_delta"],
                    config.sign_magnitude_floor_bins,
                )
                for name in SCALAR_ARMS
            },
        }
        target_controls = all(
            not cell["states"]["zero"]["continuous"]["continuous_pass"]
            and cell["states"]["exact"]["continuous"]["continuous_pass"]
            and cell["states"]["direct_rank3"]["continuous"]["continuous_pass"]
            for cell in fresh_records
        )
        pca_contract = all(
            mapping["components"].shape[0] == config.pca_rank
            and mapping["orthogonality_error"] <= config.pca_orthogonality_tolerance
            and bool(torch.isfinite(mapping["components"]).all())
            for mapping in pca.values()
        )
        numerical_pass = bool(
            all(numerical_passes)
            and pca_contract
            and bool(torch.isfinite(coordinate_scale).all())
            and float(coordinate_scale.min()) > config.coordinate_scale_floor
            and all(
                bool(torch.isfinite(value).all())
                for mapping in scalar_maps.values()
                for value in mapping.values()
            )
        )
        causal_gates = causal_gate_summary(fresh_records, config)
        primary_predictive = prediction_pass(
            fresh_diagnostics["scalar_arms"][PRIMARY_ARM], config
        )
        primary_causal = causal_gates["primary_causal_scalar_gate"]
        lookahead = {
            name: {
                "predictive_pass": prediction_pass(
                    fresh_diagnostics["scalar_arms"][name], config
                ),
                "causal_all_cells_pass": causal_gates["all_fresh_cells_pass"][name],
            }
            for name in LOOKAHEAD_ARMS
        }
        lookahead_pass = any(
            value["predictive_pass"] and value["causal_all_cells_pass"]
            for value in lookahead.values()
        )
        valid = bool(
            numerical_pass
            and target_controls
            and source_linearization["adequate"]
            and fresh_linearization["adequate"]
            and causal_gates["all_fresh_cells_pass"]["local_oracle"]
            and causal_gates["all_fresh_cells_pass"][
                "source_covector_oracle_error"
            ]
        )
        classification, primary_gate = classify_checkpoint(
            valid=valid,
            predictive_pass=primary_predictive,
            causal_pass=primary_causal,
            lookahead_pass=lookahead_pass,
        )
        _write_npz(arrays_path, array_payload)
        arrays_sha256 = _sha256(arrays_path)
        result = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_id": f"tinyllm-c2-task-activation-scalar-sensor-seed{seed}",
            "status": "completed",
            "evidence_role": _evidence_role(config),
            "completed_at": _utc_now(),
            "seed": seed,
            "configuration": asdict(config),
            "scientific_fingerprint": fingerprint,
            "implementation_sha256": implementation,
            "provenance": {
                "source_campaign": str(source_path),
                "source_campaign_sha256": _sha256(source_path),
                "source_result": str(source_result_path),
                "source_result_sha256": _sha256(source_result_path),
                "writer_campaign": str(writer_campaign_path),
                "writer_campaign_sha256": _sha256(writer_campaign_path),
                "writer_result": str(writer_result_path),
                "writer_result_sha256": _sha256(writer_result_path),
                "checkpoint": provenance["checkpoint"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "frontend_checkpoint": provenance["frontend_checkpoint"],
                "frontend_checkpoint_sha256": provenance[
                    "frontend_checkpoint_sha256"
                ],
                "character_result": str(frozen_path),
                "character_result_sha256": _sha256(frozen_path),
                "frozen_covector_map_sha256": _tensor_digest(
                    covector_map["linear"], covector_map["intercept"]
                ),
            },
            "basis": basis_summary,
            "coordinate_scale": coordinate_scale.detach().cpu().tolist(),
            "source_fit_cells": source_records,
            "source_linearization": source_linearization,
            "pca": {name: _pca_summary(mapping) for name, mapping in pca.items()},
            "scalar_maps": {
                name: {
                    "summary": _mapping_summary(mapping),
                    "mapping_sha256": _tensor_digest(*mapping.values()),
                    "feature_width": int(source_arm_features[name].shape[1]),
                }
                for name, mapping in scalar_maps.items()
            },
            "source_fit": source_fit,
            "fresh_cells": fresh_records,
            "fresh_linearization": fresh_linearization,
            "fresh_diagnostics": fresh_diagnostics,
            "causal_gates": causal_gates,
            "lookahead_gates": lookahead,
            "gates": {
                "provenance_contract": True,
                "numerical_contract": numerical_pass,
                "pca_contract": pca_contract,
                "continuous_target_control_contract": target_controls,
                "source_local_linearization_adequate": source_linearization[
                    "adequate"
                ],
                "fresh_local_linearization_adequate": fresh_linearization[
                    "adequate"
                ],
                "local_oracle_all_fresh_cells_pass": causal_gates[
                    "all_fresh_cells_pass"
                ]["local_oracle"],
                "source_covector_oracle_error_all_fresh_cells_pass": causal_gates[
                    "all_fresh_cells_pass"
                ]["source_covector_oracle_error"],
                "causal_activation_predictive_gate": primary_predictive,
                "causal_activation_causal_gate": primary_causal,
                "all_controls_specific": causal_gates["all_controls_specific"],
                "lookahead_any_joint_pass": lookahead_pass,
                "task_activation_scalar_sensor_gate": primary_gate,
            },
            "classification": classification,
            "primary_metric": float(primary_gate),
            "analysis_seconds": time.perf_counter() - seed_started,
            "artifacts": {
                "result": str(result_path),
                "arrays": str(arrays_path),
                "arrays_sha256": arrays_sha256,
            },
        }
        _write_json(result_path, result)
        results.append(result)
        print(f"seed {seed}: {classification}", flush=True)
        if _implementation_digest() != implementation:
            raise RuntimeError("activation-sensor implementation changed during campaign")

    classifications = [result["classification"] for result in results]
    pass_count = sum(
        result["gates"]["task_activation_scalar_sensor_gate"] for result in results
    )
    if any(name == "invalid" for name in classifications):
        conclusion = "invalid_campaign"
    elif len(config.seeds) == 3 and pass_count == 3:
        conclusion = "supported_task_activation_scalar_sensor_three_of_three"
    elif len(set(classifications)) == 1:
        conclusion = classifications[0]
    else:
        conclusion = "checkpoint_stratified_activation_scalar_sensor"
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
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        },
        "provenance": {
            "source_campaign": str(source_path),
            "source_campaign_sha256": _sha256(source_path),
            "source_implementation_sha256": source_campaign[
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
            "fitted_writers": 0,
            "fitted_pca_summaries": len(config.seeds) * len(PCA_NAMES),
            "fitted_scalar_observers": len(config.seeds) * len(SCALAR_ARMS),
            "fresh_primary_cells": len(config.seeds) * len(fixed.REGIMES),
            "source_fit_orbits": len(config.seeds)
            * len(SOURCE_COHORTS)
            * len(fixed.REGIMES)
            * config.orbit_count,
        },
        "aggregates": {
            "conclusion": conclusion,
            "primary_checkpoint_pass_count": pass_count,
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
                    "provenance_contract",
                    "numerical_contract",
                    "pca_contract",
                    "continuous_target_control_contract",
                    "source_local_linearization_adequate",
                    "fresh_local_linearization_adequate",
                    "local_oracle_all_fresh_cells_pass",
                    "source_covector_oracle_error_all_fresh_cells_pass",
                    "causal_activation_predictive_gate",
                    "causal_activation_causal_gate",
                    "all_controls_specific",
                    "lookahead_any_joint_pass",
                    "task_activation_scalar_sensor_gate",
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
                "arrays": result["artifacts"]["arrays"],
                "arrays_sha256": _sha256(Path(result["artifacts"]["arrays"])),
            }
            for result in results
        ],
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "The frozen task covector still consumes an oracle quotient-phase chart.",
            "Source scalar labels use exact diagnostic residuals.",
            "PCA and scalar maps are fitted observers, not natural network modules.",
            "Post-MLP and output arms are later than the primary patch cut.",
            "Every carrier correction is local and off manifold.",
            "Three selected checkpoints do not establish population prevalence.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "seed_*" / "result.json"),
            "arrays": str(output / "runs" / "seed_*" / "activation_sensor_arrays.npz"),
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
            "data/experiments/tinyllm_task_activation_scalar_sensor/"
            "20260807_d6_fresh_d"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=_ints, default=fixed.PRIMARY_SEEDS)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = TaskActivationScalarSensorConfig(
        seeds=args.seeds,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
