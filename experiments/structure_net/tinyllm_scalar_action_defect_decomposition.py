#!/usr/bin/env python3
"""Decompose TinyLLM's scalar action defect into residual and covector terms."""

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


SCHEMA_VERSION = "nal.tinyllm-c2-scalar-action-defect-decomposition.v1"
HYPOTHESIS_ID = "tinyllm-c2-scalar-action-defect-decomposition-v1"
SOURCE_SCHEMA = "nal.tinyllm-c2-nuisance-scalar-transformation-law.v1"
SOURCE_HYPOTHESIS_ID = "tinyllm-c2-nuisance-scalar-transformation-law-v1"
SOURCE_CAMPAIGN_SHA256 = (
    "1e1795c931335c739a38550b85def7c4b442be39afcea76af8cd8491b1ff6589"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "e92c3894b3e8820c0242edeba29e1f893f321e7f1dc20d75583c898ce7b1526f"
)
SOURCE_EVIDENCE_ROLE = (
    "preregistered_existing_artifact_post_outcome_underpowered_mechanistic_evidence"
)
SOURCE_CLASSIFICATION = "scalar_action_dependent"
PRIMARY_SEEDS = (7, 29, 53)
REGIMES = ("composition", "extrapolation")
ACTIONS = ("amplitude", "orientation", "offset", "composed")
EXPECTED_SOURCE_GATES = {
    "basis_gauge_replay_contract": True,
    "correspondence_specificity_all_cells_pass": False,
    "input_and_pair_contract": True,
    "input_identity_replay_contract": True,
    "local_linearization_contract": True,
    "nuisance_scalar_transformation_gate": False,
    "provenance_contract": True,
    "scalar_invariance_all_cells_pass": False,
    "target_control_contract": True,
}


@dataclass(frozen=True)
class ScalarActionDefectConfig:
    source_root: str = (
        "data/experiments/tinyllm_nuisance_scalar_transformation_law/"
        "20260807_d6_existing_group_gauge_replay"
    )
    seeds: tuple[int, ...] = PRIMARY_SEEDS
    phase_count: int = 16
    nuisance_replicates: int = 4
    carrier_rank: int = 3
    circular_bins: int = 16
    sign_magnitude_floor_bins: float = 0.01
    target_rms_floor_bins: float = 0.02
    prediction_r2_floor: float = 0.90
    prediction_relative_l2_ceiling: float = math.sqrt(0.10)
    prediction_sign_agreement_floor: float = 0.90
    control_r2_margin: float = 0.10
    algebraic_tolerance: float = 1e-12
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("primary scalar-action seeds are fixed to 7,29,53")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.phase_count != 16 or self.nuisance_replicates != 4:
            raise ValueError("the stored cohort fixes 16 phases x 4 nuisances")
        if self.phase_count * self.nuisance_replicates != 64:
            raise ValueError("the stored cohort fixes 64 aligned orbits")
        if self.carrier_rank != 3 or self.circular_bins != 16:
            raise ValueError("the stored carrier rank and circular bins are fixed")
        if self.target_rms_floor_bins <= 0 or self.control_r2_margin <= 0:
            raise ValueError("diagnostic floors and margins must be positive")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_compatible(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


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


def _implementation_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _evidence_role(config: ScalarActionDefectConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_stored_array_post_outcome_underpowered_mechanistic_evidence"
    )


def wrap_bins(value: np.ndarray, bins: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return (value + bins / 2.0) % bins - bins / 2.0


def phase_shift_indices(phase_count: int, nuisance_replicates: int) -> np.ndarray:
    indices = np.arange(phase_count * nuisance_replicates, dtype=np.int64)
    phase = indices // nuisance_replicates
    replicate = indices % nuisance_replicates
    return ((phase + 1) % phase_count) * nuisance_replicates + replicate


def scalar_prediction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    sign_floor: float,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction.shape != target.shape or not len(target):
        raise ValueError("scalar predictions must be non-empty and row aligned")
    difference = prediction - target
    target_norm = max(float(np.linalg.norm(target)), 1e-24)
    target_square = max(float(np.square(target).sum()), 1e-24)
    mask = np.abs(target) >= sign_floor
    centered_prediction = prediction - prediction.mean()
    centered_target = target - target.mean()
    correlation_denominator = float(
        np.linalg.norm(centered_prediction) * np.linalg.norm(centered_target)
    )
    correlation = (
        float(np.dot(centered_prediction, centered_target) / correlation_denominator)
        if correlation_denominator > 1e-24
        else 0.0
    )
    return {
        "zero_referenced_r2": float(1.0 - np.square(difference).sum() / target_square),
        "relative_l2": float(np.linalg.norm(difference) / target_norm),
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.square(difference).mean())),
        "target_rms": float(np.sqrt(np.square(target).mean())),
        "prediction_rms": float(np.sqrt(np.square(prediction).mean())),
        "sign_agreement": (
            float(np.mean(np.sign(prediction[mask]) == np.sign(target[mask])))
            if bool(mask.any())
            else 1.0
        ),
        "sign_evaluation_count": int(mask.sum()),
        "correlation": correlation,
    }


def prediction_pass(
    metrics: Mapping[str, Any], config: ScalarActionDefectConfig
) -> bool:
    return bool(
        metrics["zero_referenced_r2"] >= config.prediction_r2_floor
        and metrics["relative_l2"] <= config.prediction_relative_l2_ceiling
        and metrics["sign_agreement"] >= config.prediction_sign_agreement_floor
    )


def symmetric_action_decomposition(
    reference_covector: np.ndarray,
    transformed_covector: np.ndarray,
    reference_residual: np.ndarray,
    transformed_residual: np.ndarray,
) -> dict[str, np.ndarray | float]:
    jr = np.asarray(reference_covector, dtype=np.float64)
    jg = np.asarray(transformed_covector, dtype=np.float64)
    zr = np.asarray(reference_residual, dtype=np.float64)
    zg = np.asarray(transformed_residual, dtype=np.float64)
    if jr.shape != jg.shape or zr.shape != zg.shape or jr.shape != zr.shape:
        raise ValueError("covectors and standardized residuals must have equal shape")
    if jr.ndim != 2 or not len(jr):
        raise ValueError("decomposition inputs must be non-empty matrices")
    residual = 0.5 * ((jr + jg) * (zg - zr)).sum(axis=1)
    covector = 0.5 * ((jg - jr) * (zr + zg)).sum(axis=1)
    joint = residual + covector
    direct = (jg * zg).sum(axis=1) - (jr * zr).sum(axis=1)
    denominator = max(float(np.linalg.norm(direct)), 1e-24)
    error = float(np.linalg.norm(joint - direct) / denominator)
    residual_norm = float(np.linalg.norm(residual))
    covector_norm = float(np.linalg.norm(covector))
    component_cosine = (
        float(np.dot(residual, covector) / (residual_norm * covector_norm))
        if residual_norm > 1e-24 and covector_norm > 1e-24
        else 0.0
    )
    return {
        "residual": residual,
        "covector": covector,
        "joint": joint,
        "direct": direct,
        "algebraic_relative_error": error,
        "component_cosine": component_cosine,
    }


def classify_cell(*, joint: bool, residual: bool, covector: bool) -> str:
    if not joint:
        return "nonlinear_or_unresolved"
    if residual and not covector:
        return "residual_coordinate_defect"
    if covector and not residual:
        return "covector_transport_defect"
    if residual and covector:
        return "dual_sufficient"
    return "coupled_action_defect"


def classify_checkpoint(
    *, valid: bool, cell_labels: Sequence[str], joint_specific: bool
) -> str:
    if not valid:
        return "invalid"
    if not joint_specific or not cell_labels:
        return "checkpoint_internal_mixture"
    labels = set(cell_labels)
    if len(labels) == 1 and "nonlinear_or_unresolved" not in labels:
        return next(iter(labels))
    if labels == {"nonlinear_or_unresolved"}:
        return "nonlinear_or_unresolved"
    return "checkpoint_internal_mixture"


def _load_source_campaign(
    config: ScalarActionDefectConfig,
) -> tuple[dict[str, Any], Path, dict[int, tuple[dict[str, Any], Path, Path, Path]]]:
    path = Path(config.source_root) / "campaign_results.json"
    campaign = json.loads(path.read_text(encoding="utf-8"))
    aggregates = campaign.get("aggregates", {})
    summary = campaign.get("summary", {})
    if (
        _sha256(path) != SOURCE_CAMPAIGN_SHA256
        or campaign.get("schema_version") != SOURCE_SCHEMA
        or campaign.get("hypothesis_id") != SOURCE_HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != SOURCE_EVIDENCE_ROLE
        or campaign.get("implementation_sha256") != SOURCE_IMPLEMENTATION_SHA256
        or int(summary.get("completed", -1)) != 3
        or int(summary.get("trained_models", -1)) != 0
        or int(summary.get("fitted_observers", -1)) != 0
        or aggregates.get("conclusion") != SOURCE_CLASSIFICATION
        or int(aggregates.get("primary_checkpoint_pass_count", -1)) != 0
    ):
        raise ValueError(f"invalid scalar transformation source campaign {path}")
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if not set(config.seeds).issubset(entries):
        raise ValueError("source campaign lacks a requested checkpoint")
    details: dict[int, tuple[dict[str, Any], Path, Path, Path]] = {}
    for seed in config.seeds:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        scalar_arrays_path = Path(entry["arrays"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        group_arrays_path = Path(detail["provenance"]["group_arrays"])
        if (
            _sha256(detail_path) != entry.get("result_sha256")
            or _sha256(scalar_arrays_path) != entry.get("arrays_sha256")
            or _sha256(group_arrays_path)
            != detail.get("provenance", {}).get("group_arrays_sha256")
            or detail.get("schema_version") != SOURCE_SCHEMA
            or detail.get("hypothesis_id") != SOURCE_HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != SOURCE_EVIDENCE_ROLE
            or detail.get("implementation_sha256") != SOURCE_IMPLEMENTATION_SHA256
            or int(detail.get("seed", -1)) != seed
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or detail.get("classification") != SOURCE_CLASSIFICATION
            or detail.get("gates") != EXPECTED_SOURCE_GATES
            or len(detail.get("cells", [])) != 8
            or len(detail.get("coordinate_scale", [])) != config.carrier_rank
        ):
            raise ValueError(f"invalid scalar transformation source result {detail_path}")
        details[seed] = (
            detail,
            detail_path,
            scalar_arrays_path,
            group_arrays_path,
        )
    return campaign, path, details


def _fingerprint(
    config: ScalarActionDefectConfig,
    seed: int,
    source_result_sha256: str,
    scalar_arrays_sha256: str,
    group_arrays_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
        "seed": seed,
        "source_result_sha256": source_result_sha256,
        "scalar_arrays_sha256": scalar_arrays_sha256,
        "group_arrays_sha256": group_arrays_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reusable_seed_result(
    result_path: Path,
    arrays_path: Path,
    config: ScalarActionDefectConfig,
    implementation: str,
    seed: int,
    fingerprint: str,
) -> dict[str, Any] | None:
    if not result_path.is_file():
        return None
    value = json.loads(result_path.read_text(encoding="utf-8"))
    if not (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("evidence_role") == _evidence_role(config)
        and value.get("implementation_sha256") == implementation
        and value.get("configuration") == _json_compatible(asdict(config))
        and int(value.get("seed", -1)) == seed
        and value.get("scientific_fingerprint") == fingerprint
        and value.get("artifacts", {}).get("result") == str(result_path)
        and value.get("artifacts", {}).get("arrays") == str(arrays_path)
        and arrays_path.is_file()
        and _sha256(arrays_path) == value.get("artifacts", {}).get("arrays_sha256")
    ):
        raise ValueError(f"incompatible completed result {result_path}; use a new root")
    return value


def _campaign_is_reusable(
    value: Mapping[str, Any],
    config: ScalarActionDefectConfig,
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
        and len(value.get("results", [])) == len(config.seeds)
    ):
        return False
    return all(
        Path(item.get("path", "")).is_file()
        and Path(item.get("arrays", "")).is_file()
        and _sha256(Path(item["path"])) == item.get("result_sha256")
        and _sha256(Path(item["arrays"])) == item.get("arrays_sha256")
        for item in value.get("results", [])
    )


def _array(source: Mapping[str, np.ndarray], key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(source[key], dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"invalid stored array {key}: expected {shape}, got {value.shape}")
    return value


def run_campaign(
    config: ScalarActionDefectConfig, output: Path
) -> dict[str, Any]:
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

    source_campaign, source_path, source_details = _load_source_campaign(config)
    shift = phase_shift_indices(config.phase_count, config.nuisance_replicates)
    results: list[dict[str, Any]] = []
    reused = 0
    for seed in config.seeds:
        seed_started = time.perf_counter()
        detail, detail_path, scalar_path, group_path = source_details[seed]
        fingerprint = _fingerprint(
            config,
            seed,
            _sha256(detail_path),
            _sha256(scalar_path),
            _sha256(group_path),
        )
        result_path = output / "runs" / f"seed_{seed}" / "result.json"
        arrays_path = output / "runs" / f"seed_{seed}" / "decomposition_arrays.npz"
        existing = _reusable_seed_result(
            result_path, arrays_path, config, implementation, seed, fingerprint
        )
        if existing is not None:
            results.append(existing)
            reused += 1
            print(f"resuming {existing['experiment_id']}", flush=True)
            continue

        with np.load(scalar_path, allow_pickle=False) as scalar_archive:
            scalar_arrays = {name: scalar_archive[name] for name in scalar_archive.files}
        with np.load(group_path, allow_pickle=False) as group_archive:
            group_arrays = {name: group_archive[name] for name in group_archive.files}
        scale = np.asarray(detail["coordinate_scale"], dtype=np.float64)
        if (
            scale.shape != (config.carrier_rank,)
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError(f"invalid coordinate scale for seed {seed}")

        cells: list[dict[str, Any]] = []
        array_payload: dict[str, np.ndarray] = {}
        numerical_valid = True
        for regime in REGIMES:
            reference_prefix = f"{regime}__reference"
            reference_scalar = _array(
                scalar_arrays,
                f"{reference_prefix}__signed_scalar",
                (64,),
            )
            reference_covector = _array(
                scalar_arrays,
                f"{reference_prefix}__fine_gradient",
                (64, config.carrier_rank),
            )
            reference_residual = _array(
                group_arrays,
                f"{reference_prefix}__residual",
                (64, config.carrier_rank),
            ) / scale
            for action in ACTIONS:
                prefix = f"{regime}__{action}"
                transformed_scalar = _array(
                    scalar_arrays, f"{prefix}__signed_scalar", (64,)
                )
                transformed_covector = _array(
                    scalar_arrays,
                    f"{prefix}__fine_gradient",
                    (64, config.carrier_rank),
                )
                transformed_residual = _array(
                    group_arrays,
                    f"{prefix}__residual",
                    (64, config.carrier_rank),
                ) / scale
                observed = wrap_bins(
                    transformed_scalar - reference_scalar,
                    config.circular_bins,
                )
                decomposition = symmetric_action_decomposition(
                    reference_covector,
                    transformed_covector,
                    reference_residual,
                    transformed_residual,
                )
                residual_component = np.asarray(decomposition["residual"])
                covector_component = np.asarray(decomposition["covector"])
                joint_component = np.asarray(decomposition["joint"])
                controls = {
                    "semantic_phase_shift": joint_component[shift],
                    "sign_flip": -joint_component,
                }
                metrics = {
                    "joint": scalar_prediction_metrics(
                        joint_component,
                        observed,
                        config.sign_magnitude_floor_bins,
                    ),
                    "residual": scalar_prediction_metrics(
                        residual_component,
                        observed,
                        config.sign_magnitude_floor_bins,
                    ),
                    "covector": scalar_prediction_metrics(
                        covector_component,
                        observed,
                        config.sign_magnitude_floor_bins,
                    ),
                }
                control_metrics = {
                    name: scalar_prediction_metrics(
                        value, observed, config.sign_magnitude_floor_bins
                    )
                    for name, value in controls.items()
                }
                gates = {
                    name: prediction_pass(metrics[name], config)
                    for name in ("joint", "residual", "covector")
                }
                control_specific = all(
                    not prediction_pass(value, config)
                    and metrics["joint"]["zero_referenced_r2"]
                    - value["zero_referenced_r2"]
                    >= config.control_r2_margin
                    for value in control_metrics.values()
                )
                nondegenerate = bool(
                    metrics["joint"]["target_rms"] >= config.target_rms_floor_bins
                )
                algebraic = bool(
                    decomposition["algebraic_relative_error"]
                    <= config.algebraic_tolerance
                )
                numerical_valid = bool(
                    numerical_valid and nondegenerate and algebraic
                )
                label = classify_cell(
                    joint=gates["joint"],
                    residual=gates["residual"],
                    covector=gates["covector"],
                )
                cells.append(
                    {
                        "regime": regime,
                        "action": action,
                        "metrics": metrics,
                        "controls": control_metrics,
                        "gates": {
                            **{f"{name}_prediction": value for name, value in gates.items()},
                            "control_specificity": control_specific,
                            "action_defect_nondegenerate": nondegenerate,
                            "algebraic_identity": algebraic,
                        },
                        "component_cosine": decomposition["component_cosine"],
                        "algebraic_relative_error": decomposition[
                            "algebraic_relative_error"
                        ],
                        "classification": label,
                    }
                )
                array_payload[f"{prefix}__observed_action_defect"] = observed
                array_payload[f"{prefix}__residual_component"] = residual_component
                array_payload[f"{prefix}__covector_component"] = covector_component
                array_payload[f"{prefix}__joint_component"] = joint_component
                array_payload[f"{prefix}__semantic_phase_shift_control"] = controls[
                    "semantic_phase_shift"
                ]

        source_valid = detail.get("gates") == EXPECTED_SOURCE_GATES
        joint_all = all(cell["gates"]["joint_prediction"] for cell in cells)
        residual_all = all(cell["gates"]["residual_prediction"] for cell in cells)
        covector_all = all(cell["gates"]["covector_prediction"] for cell in cells)
        specificity_all = all(cell["gates"]["control_specificity"] for cell in cells)
        valid = bool(source_valid and numerical_valid)
        classification = classify_checkpoint(
            valid=valid,
            cell_labels=[cell["classification"] for cell in cells],
            joint_specific=specificity_all,
        )
        gates = {
            "source_provenance_and_validity_contract": source_valid,
            "stored_array_numerical_contract": numerical_valid,
            "joint_first_order_all_cells_pass": joint_all,
            "residual_component_all_cells_pass": residual_all,
            "covector_component_all_cells_pass": covector_all,
            "control_specificity_all_cells_pass": specificity_all,
            "residual_coordinate_defect_gate": bool(
                valid
                and classification == "residual_coordinate_defect"
                and joint_all
                and residual_all
                and not covector_all
                and specificity_all
            ),
        }
        _write_npz(arrays_path, array_payload)
        result = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_id": f"tinyllm-c2-scalar-action-defect-seed{seed}",
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
                "source_result": str(detail_path),
                "source_result_sha256": _sha256(detail_path),
                "source_scalar_arrays": str(scalar_path),
                "source_scalar_arrays_sha256": _sha256(scalar_path),
                "source_group_arrays": str(group_path),
                "source_group_arrays_sha256": _sha256(group_path),
                "checkpoint": detail["provenance"]["checkpoint"],
                "checkpoint_sha256": detail["provenance"]["checkpoint_sha256"],
            },
            "cells": cells,
            "gates": gates,
            "classification": classification,
            "primary_metric": float(gates["residual_coordinate_defect_gate"]),
            "analysis_seconds": time.perf_counter() - seed_started,
            "artifacts": {
                "result": str(result_path),
                "arrays": str(arrays_path),
                "arrays_sha256": _sha256(arrays_path),
            },
        }
        _write_json(result_path, result)
        results.append(result)
        print(f"seed {seed}: {classification}", flush=True)
        if _implementation_digest() != implementation:
            raise RuntimeError("scalar action-defect implementation changed during campaign")

    classifications = [result["classification"] for result in results]
    primary_count = sum(
        result["gates"]["residual_coordinate_defect_gate"] for result in results
    )
    joint_count = sum(
        result["gates"]["joint_first_order_all_cells_pass"] for result in results
    )
    if any(name == "invalid" for name in classifications):
        conclusion = "invalid_campaign"
    elif len(config.seeds) == 3 and primary_count == 3:
        conclusion = "supported_residual_coordinate_defect_three_of_three"
    elif len(set(classifications)) == 1:
        conclusion = classifications[0]
    else:
        conclusion = "checkpoint_stratified_action_defect"
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
            "numpy": np.__version__,
            "device": "cpu_stored_arrays_only",
            "peak_cuda_memory_bytes": 0,
        },
        "provenance": {
            "source_campaign": str(source_path),
            "source_campaign_sha256": _sha256(source_path),
            "source_implementation_sha256": source_campaign["implementation_sha256"],
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
            "fitted_encoders": 0,
            "fitted_observers": 0,
            "source_action_cells": len(config.seeds) * len(REGIMES) * len(ACTIONS),
            "derived_component_fields": len(config.seeds)
            * len(REGIMES)
            * len(ACTIONS)
            * 3,
        },
        "aggregates": {
            "conclusion": conclusion,
            "primary_checkpoint_pass_count": primary_count,
            "joint_first_order_checkpoint_pass_count": joint_count,
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
                    "source_provenance_and_validity_contract",
                    "stored_array_numerical_contract",
                    "joint_first_order_all_cells_pass",
                    "residual_component_all_cells_pass",
                    "covector_component_all_cells_pass",
                    "control_specificity_all_cells_pass",
                    "residual_coordinate_defect_gate",
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
            "This is a preregistered decomposition of stored action-law arrays, not fresh checkpoint evidence.",
            "The joint component is algebraically exact only for the two local linear models; agreement with the nonlinear signed scalar is measured.",
            "Residual and covector components use a symmetric two-factor decomposition, not a unique causal temporal ordering.",
            "The exact signed correction and rank-three residual are diagnostic quantities unavailable to a deployable sensor.",
            "Three selected checkpoints do not establish checkpoint-population prevalence.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "seed_*" / "result.json"),
            "arrays": str(output / "runs" / "seed_*" / "decomposition_arrays.npz"),
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
            "data/experiments/tinyllm_scalar_action_defect_decomposition/"
            "20260807_d6_stored_arrays_gauge_replay"
        ),
    )
    parser.add_argument("--seeds", type=_ints, default=PRIMARY_SEEDS)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = ScalarActionDefectConfig(
        seeds=args.seeds,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
