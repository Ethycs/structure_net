#!/usr/bin/env python3
"""Decompose trained exact-C3 sensors into affine and nonlinear responses."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Iterable, Mapping, Sequence

import torch

import experiments.structure_net.tinyllm_c3_temporal_quotient_preflight as generator
import experiments.structure_net.tinyllm_c3_temporal_quotient_training as stage0
import experiments.structure_net.tinyllm_c3_temporal_sensor_only as source
import experiments.structure_net.tinyllm_joint_physical_scalar_interface as joint


SCHEMA_VERSION = "nal.tinyllm-c3-temporal-sensor-mechanism.v1"
HYPOTHESIS_ID = "tinyllm-c3-temporal-sensor-mechanism-v1"
EVIDENCE_ROLE = "registered_post_outcome_artifact_only_sensor_mechanism"
SOURCE_CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_sensor_only/"
    "20260811_preregistered/campaign_results.json"
)
SOURCE_CAMPAIGN_SHA256 = (
    "4d023d9f9d77f64b1fe75970acd63461cfd5a64aa1de60c7954a2c671c427012"
)
SOURCE_RESULT_MANIFEST_SHA256 = (
    "a23d19892f112645a7d3b5401d1528a0eb612a5d6db26520fe3514246c4c6d1a"
)
SOURCE_CHECKPOINT_MANIFEST_SHA256 = (
    "e83832872f29d072d710859e022f4e17d1b6da6a9e16b63049f41d4ea2eb01a0"
)
SOURCE_RUNNER_SHA256 = (
    "7f4a5990f2f9a56bcaad0032d7cf9eca20f74b599a0684337c48dcdf9593b3ed"
)
CAPACITY_RESULT_SHA256 = (
    "6a01db25ebc2ed15d202884c39f16db685d5218647b0bb209e2e5a737696a383"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-temporal-sensor-mechanism-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "b32dbd5fce221b7eec50a7845f41d2dd036275397d69837e293db25bfecb72d5"
)
PRIMARY_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_sensor_mechanism/"
    "20260811_preregistered/campaign_results.json"
)
RESPONSE_ARMS = ("full_replay", "affine_only", "nonlinear_residual_only")
SOURCE_ARMS = source.ARMS
SEEDS = source.SEEDS
REGIMES = source.REGIMES
REPLAY_TOLERANCE = 2e-6
RECONSTRUCTION_TOLERANCE = 1e-6
SLOPE_MAGNITUDE_MINIMUM = 1e-6
CARRIER_DOT_MINIMUM = source.CARRIER_DOT_MINIMUM
CARRIER_RMSE_MAXIMUM = source.CARRIER_RMSE_MAXIMUM
ACTION_ERROR_MAXIMUM = source.ACTION_ERROR_MAXIMUM
REQUIRED_SEED_PASSES = 4
CONTROL_SEED_PASSES_MAXIMUM = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(f"{_sha256(path)}  {path}\n".encode("utf-8"))
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def _source_result_paths(root: Path) -> list[Path]:
    return [root / "runs" / f"seed_{seed}" / "result.json" for seed in SEEDS]


def _source_checkpoint_paths(root: Path) -> list[Path]:
    return [
        root / "runs" / f"seed_{seed}" / f"{arm}_{point}.pt"
        for seed in SEEDS
        for arm in SOURCE_ARMS
        for point in ("midpoint", "final")
    ]


def validate_source_campaign() -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    root = SOURCE_CAMPAIGN_PATH.parent
    campaign = json.loads(SOURCE_CAMPAIGN_PATH.read_text(encoding="utf-8"))
    if (
        _sha256(SOURCE_CAMPAIGN_PATH) != SOURCE_CAMPAIGN_SHA256
        or campaign.get("schema_version") != source.SCHEMA_VERSION
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != source.EVIDENCE_ROLE
        or campaign.get("aggregates", {}).get("classification")
        != "task_only_sensor_acquisition_supported"
        or campaign.get("aggregates", {}).get("learned_true_joint_pass_count")
        != 5
        or campaign.get("aggregates", {}).get(
            "learned_target_shuffled_joint_pass_count"
        )
        != 0
        or campaign.get("result_manifest_sha256")
        != SOURCE_RESULT_MANIFEST_SHA256
        or campaign.get("checkpoint_manifest_sha256")
        != SOURCE_CHECKPOINT_MANIFEST_SHA256
        or campaign.get("source_hashes", {}).get("runner")
        != SOURCE_RUNNER_SHA256
        or campaign.get("source_hashes", {}).get("capacity_result")
        != CAPACITY_RESULT_SHA256
    ):
        raise RuntimeError("invalid source sensor-only campaign")
    if _sha256(Path(source.__file__)) != SOURCE_RUNNER_SHA256:
        raise RuntimeError("source sensor-only runner changed")
    results = _source_result_paths(root)
    checkpoints = _source_checkpoint_paths(root)
    if _manifest(results) != SOURCE_RESULT_MANIFEST_SHA256:
        raise RuntimeError("source result manifest changed")
    if _manifest(checkpoints) != SOURCE_CHECKPOINT_MANIFEST_SHA256:
        raise RuntimeError("source checkpoint manifest changed")
    details = {}
    for path in results:
        detail = json.loads(path.read_text(encoding="utf-8"))
        seed = int(detail.get("seed", -1))
        if (
            seed not in SEEDS
            or detail.get("status") != "completed"
            or detail.get("gates", {}).get("validity") is not True
            or detail.get("gates", {}).get("learned_true_joint") is not True
            or detail.get("gates", {}).get("learned_target_shuffled_joint")
            is not False
        ):
            raise RuntimeError(f"invalid source sensor detail {path}")
        for arm in SOURCE_ARMS:
            arm_result = detail["arms"][arm]
            if (
                arm_result["reload"]["pass"] is not True
                or arm_result["exact_resume"]["pass"] is not True
            ):
                raise RuntimeError(f"invalid source replay {arm}/seed {seed}")
            for point in ("midpoint", "final"):
                artifact = arm_result[f"{point}_checkpoint"]
                artifact_path = Path(artifact["path"])
                if _sha256(artifact_path) != artifact["sha256"]:
                    raise RuntimeError(
                        f"invalid source checkpoint {arm}/{point}/seed {seed}"
                    )
        details[seed] = detail
    if set(details) != set(SEEDS):
        raise RuntimeError("source seed set changed")
    return campaign, details


def effective_response(
    encoder: stage0.LearnedC3InvariantEncoder,
    corrected: torch.Tensor,
) -> torch.Tensor:
    features = encoder.shared_map(corrected.unsqueeze(-1))
    mixer = torch.complex(encoder.mixer_real, encoder.mixer_imag)
    return torch.einsum("...k,k->...", features.to(torch.complex64), mixer)


def fit_complex_affine(
    scalar: torch.Tensor,
    response: torch.Tensor,
) -> dict[str, Any]:
    x = scalar.reshape(-1).double()
    y = response.reshape(-1).to(torch.complex128)
    design = torch.stack((x, torch.ones_like(x)), dim=-1)
    real = torch.linalg.lstsq(design, y.real[:, None]).solution[:, 0]
    imaginary = torch.linalg.lstsq(design, y.imag[:, None]).solution[:, 0]
    coefficient = torch.complex(real, imaginary)
    prediction = design.to(torch.complex128) @ coefficient
    residual = y - prediction
    centered = y - y.mean()
    r2 = 1.0 - float(
        residual.abs().square().sum()
        / centered.abs().square().sum().clamp_min(1e-18)
    )
    return {
        "coefficient": coefficient,
        "slope": coefficient[0],
        "intercept": coefficient[1],
        "slope_values": [float(coefficient[0].real), float(coefficient[0].imag)],
        "intercept_values": [
            float(coefficient[1].real),
            float(coefficient[1].imag),
        ],
        "slope_magnitude": float(coefficient[0].abs()),
        "source_complex_r2": r2,
        "source_response_rmse": float(torch.sqrt(residual.abs().square().mean())),
    }


def character_coefficient(
    response: torch.Tensor,
    encoder: stage0.LearnedC3InvariantEncoder,
) -> torch.Tensor:
    weight = torch.complex(encoder.character_real, encoder.character_imag)
    return torch.einsum("btc,c->bt", response.to(torch.complex64), weight)


def carrier_from_coefficient(coefficient: torch.Tensor) -> torch.Tensor:
    normalized = coefficient / coefficient.abs().clamp_min(1e-6)
    return normalized.pow(3)


@torch.no_grad()
def direct_carrier_from_corrected(
    encoder: stage0.LearnedC3InvariantEncoder,
    corrected: torch.Tensor,
    batch_size: int = 256,
) -> torch.Tensor:
    values = []
    for start in range(0, corrected.shape[0], batch_size):
        feature = encoder(corrected[start : start + batch_size])
        values.append(torch.view_as_complex(feature.contiguous()).cpu())
    return torch.cat(values)


def _responses(
    encoder: stage0.LearnedC3InvariantEncoder,
    corrected: torch.Tensor,
    fit: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    full = effective_response(encoder, corrected)
    coefficient = fit["coefficient"].to(full.device)
    affine = coefficient[0] * corrected.to(torch.complex64) + coefficient[1]
    return {
        "full_replay": full,
        "affine_only": affine,
        "nonlinear_residual_only": full - affine,
    }


def _carrier_bundle(
    encoder: stage0.LearnedC3InvariantEncoder,
    dataset: stage0.C3TrainingDataset,
    fit: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    corrected = stage0.corrected_channels(
        dataset.tokens.to(device), dataset.calibration.to(device)
    )
    responses = _responses(encoder, corrected, fit)
    coefficients = {
        name: character_coefficient(value, encoder)
        for name, value in responses.items()
    }
    reconstruction_error = float(
        (
            coefficients["full_replay"]
            - coefficients["affine_only"]
            - coefficients["nonlinear_residual_only"]
        ).abs().max()
    )
    commuted_carriers = {
        name: carrier_from_coefficient(value).cpu()
        for name, value in coefficients.items()
    }
    carriers = dict(commuted_carriers)
    carriers["full_replay"] = direct_carrier_from_corrected(encoder, corrected)
    commutation_error = float(
        (commuted_carriers["full_replay"] - carriers["full_replay"]).abs().max()
    )
    action_errors: dict[str, dict[str, float]] = {
        name: {} for name in RESPONSE_ARMS
    }
    for element in (1, 2):
        transformed_tokens = generator.apply_deck_action(dataset.tokens, element)
        transformed_corrected = stage0.corrected_channels(
            transformed_tokens.to(device), dataset.calibration.to(device)
        )
        transformed_responses = _responses(encoder, transformed_corrected, fit)
        for name, response in transformed_responses.items():
            transformed_coefficient = character_coefficient(response, encoder)
            transformed_carrier = (
                direct_carrier_from_corrected(encoder, transformed_corrected)
                if name == "full_replay"
                else carrier_from_coefficient(transformed_coefficient).cpu()
            )
            action_errors[name][str(element)] = float(
                (transformed_carrier - carriers[name]).abs().max()
            )
    maximum_action_errors = {
        name: max(errors.values()) for name, errors in action_errors.items()
    }
    coefficient_diagnostics = {
        name: {
            "mean_magnitude": float(value.abs().mean()),
            "minimum_magnitude": float(value.abs().min()),
            "fraction_below_1e_6": float((value.abs() < 1e-6).float().mean()),
        }
        for name, value in coefficients.items()
    }
    return {
        "carriers": carriers,
        "coefficient_diagnostics": coefficient_diagnostics,
        "coefficient_reconstruction_maximum_error": reconstruction_error,
        "full_commutation_carrier_maximum_error": commutation_error,
        "deck_action_maximum_errors": action_errors,
        "maximum_deck_action_errors": maximum_action_errors,
    }


def _posterior(carrier: torch.Tensor) -> torch.Tensor:
    temporal = generator.temporal_prediction(carrier.to(torch.complex128))
    return joint.interval_posterior_unclipped(temporal, 16)


def _metric_replay_error(
    measured: Mapping[str, Any],
    stored: Mapping[str, Any],
) -> dict[str, Any]:
    float_errors = {
        name: abs(float(measured[name]) - float(stored[name]))
        for name in (
            "posterior_mean_correlation",
            "posterior_mean_rmse",
            "exact_bin_accuracy",
            "target_cross_entropy",
        )
    }
    coverage_match = (
        int(measured["predicted_bin_coverage"])
        == int(stored["predicted_bin_coverage"])
    )
    maximum = max(float_errors.values())
    return {
        "float_errors": float_errors,
        "maximum_float_error": maximum,
        "coverage_match": coverage_match,
        "pass": maximum <= REPLAY_TOLERANCE and coverage_match,
    }


def _fit_gauges(
    bundles: Mapping[str, Mapping[str, Any]],
    reference: stage0.C3TrainingDataset,
) -> dict[str, dict[str, Any]]:
    analytic = source.analytic_carrier(reference)
    return {
        name: source.fit_orthogonal_gauge(
            bundles["reference"]["carriers"][name], analytic
        )
        for name in RESPONSE_ARMS
    }


def _evaluate_response(
    carrier: torch.Tensor,
    analytic: torch.Tensor,
    target: torch.Tensor,
    gauge: torch.Tensor,
    regime: str,
    action_errors: Mapping[str, float],
) -> dict[str, Any]:
    left = torch.view_as_real(carrier).reshape(-1, 2).double()
    right = torch.view_as_real(analytic).reshape(-1, 2).double()
    aligned = left @ gauge.double()
    dot = float((aligned * right).sum(-1).mean())
    rmse = float(torch.sqrt((aligned - right).square().mean()))
    posterior = _posterior(carrier)
    task = source.capacity._task_metrics(posterior, target)
    carrier_pass = dot >= CARRIER_DOT_MINIMUM and rmse <= CARRIER_RMSE_MAXIMUM
    task_pass = source._task_gate(task, regime)
    maximum_action_error = max(float(value) for value in action_errors.values())
    return {
        "mean_aligned_unit_dot": dot,
        "aligned_coordinate_rmse": rmse,
        "carrier_pass": carrier_pass,
        "deck_action_maximum_errors": dict(action_errors),
        "maximum_deck_action_error": maximum_action_error,
        "action_pass": maximum_action_error <= ACTION_ERROR_MAXIMUM,
        "task": task,
        "task_pass": task_pass,
        "joint_pass": carrier_pass and task_pass,
    }


def analyze_checkpoint(
    *,
    seed: int,
    arm: str,
    detail: Mapping[str, Any],
    config: source.SensorOnlyConfig,
    datasets: Mapping[str, stage0.C3TrainingDataset],
    device: torch.device,
) -> dict[str, Any]:
    task = stage0.C3TaskConfig()
    training, _batches, _permutation, protocol_hashes = source.protocol_material(
        task, config, seed
    )
    if protocol_hashes != detail["protocol_hashes"]:
        raise RuntimeError(f"training protocol hash changed for seed {seed}")
    checkpoint_record = detail["arms"][arm]["final_checkpoint"]
    checkpoint_path = Path(checkpoint_record["path"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        _sha256(checkpoint_path) != checkpoint_record["sha256"]
        or payload.get("schema_version") != source.SCHEMA_VERSION
        or payload.get("arm") != arm
        or payload.get("seed") != seed
        or payload.get("step") != config.training_steps
        or payload.get("scientific_fingerprint")
        != detail["scientific_fingerprint"]
        or payload.get("protocol_hashes") != protocol_hashes
    ):
        raise RuntimeError(f"checkpoint provenance changed for {arm}/seed {seed}")
    encoder = source._initial_encoder(seed, torch.device("cpu"))
    encoder.load_state_dict(payload["encoder_state"], strict=True)
    encoder.eval()
    state_before = stage0._state_digest(encoder)
    corrected = stage0.corrected_channels(training.tokens, training.calibration)
    fit = fit_complex_affine(corrected, effective_response(encoder, corrected))
    encoder.to(device)
    bundles = {
        "reference": _carrier_bundle(
            encoder, datasets["reference"], fit, device
        ),
        **{
            regime: _carrier_bundle(encoder, datasets[regime], fit, device)
            for regime in REGIMES
        },
    }
    gauges = _fit_gauges(bundles, datasets["reference"])
    gauge_records = {}
    for name, gauge in gauges.items():
        matrix = gauge.pop("matrix")
        gauge_records[name] = {**gauge, "matrix_values": matrix.tolist()}
        gauge["matrix"] = matrix
    regimes = {}
    full_replay_pass = True
    direct_encoder_errors = {}
    for regime in REGIMES:
        analytic = source.analytic_carrier(datasets[regime])
        response_results = {}
        for name in RESPONSE_ARMS:
            response_results[name] = _evaluate_response(
                bundles[regime]["carriers"][name],
                analytic,
                datasets[regime].target,
                gauges[name]["matrix"],
                regime,
                bundles[regime]["deck_action_maximum_errors"][name],
            )
        direct = source.extract_carrier(
            encoder, datasets[regime], device
        )
        direct_posterior = _posterior(direct)
        replay_posterior = _posterior(
            bundles[regime]["carriers"]["full_replay"]
        )
        direct_error = float((direct_posterior - replay_posterior).abs().max())
        metric_replay = _metric_replay_error(
            response_results["full_replay"]["task"],
            detail["arms"][arm]["regimes"][regime]["task"],
        )
        direct_encoder_errors[regime] = direct_error
        full_replay_pass = bool(
            full_replay_pass
            and direct_error <= REPLAY_TOLERANCE
            and metric_replay["pass"]
        )
        regimes[regime] = {
            "responses": response_results,
            "coefficient_diagnostics": bundles[regime][
                "coefficient_diagnostics"
            ],
            "coefficient_reconstruction_maximum_error": bundles[regime][
                "coefficient_reconstruction_maximum_error"
            ],
            "full_commutation_carrier_maximum_error": bundles[regime][
                "full_commutation_carrier_maximum_error"
            ],
            "maximum_deck_action_errors": bundles[regime][
                "maximum_deck_action_errors"
            ],
            "direct_encoder_posterior_maximum_error": direct_error,
            "stored_metric_replay": metric_replay,
        }
    state_after = stage0._state_digest(encoder)
    reconstruction_pass = all(
        bundles[name]["coefficient_reconstruction_maximum_error"]
        <= RECONSTRUCTION_TOLERANCE
        for name in ("reference", *REGIMES)
    )
    response_seed_gates = {
        name: all(
            regimes[regime]["responses"][name]["joint_pass"]
            for regime in REGIMES
        )
        for name in RESPONSE_ARMS
    }
    valid = bool(
        fit["slope_magnitude"] >= SLOPE_MAGNITUDE_MINIMUM
        and reconstruction_pass
        and full_replay_pass
        and state_before == state_after
        and _finite(regimes)
    )
    serial_fit = {
        key: value
        for key, value in fit.items()
        if key not in {"coefficient", "slope", "intercept"}
    }
    return {
        "seed": seed,
        "source_arm": arm,
        "checkpoint": checkpoint_record,
        "state_sha256_before": state_before,
        "state_sha256_after": state_after,
        "state_unchanged": state_before == state_after,
        "affine_fit": serial_fit,
        "gauges": gauge_records,
        "regimes": regimes,
        "response_seed_gates": response_seed_gates,
        "full_replay_pass": full_replay_pass,
        "coefficient_reconstruction_pass": reconstruction_pass,
        "valid": valid,
    }


def classify(cells: Sequence[Mapping[str, Any]], valid: bool) -> dict[str, Any]:
    def count(source_arm: str, response: str) -> int:
        return sum(
            bool(cell["response_seed_gates"][response])
            for cell in cells
            if cell["source_arm"] == source_arm
        )

    true_affine = count("learned_true", "affine_only")
    true_residual = count("learned_true", "nonlinear_residual_only")
    shuffled_affine = count("learned_target_shuffled", "affine_only")
    full_replay = sum(cell["full_replay_pass"] for cell in cells)
    if not valid or full_replay != 10:
        classification = "invalid_source_contract"
        primary = False
    elif shuffled_affine > CONTROL_SEED_PASSES_MAXIMUM:
        classification = "affine_mechanism_specificity_failed"
        primary = False
    elif (
        true_affine >= REQUIRED_SEED_PASSES
        and true_residual <= CONTROL_SEED_PASSES_MAXIMUM
    ):
        classification = "affine_identity_character_carries_learned_solution"
        primary = True
    elif true_affine < REQUIRED_SEED_PASSES and true_residual >= REQUIRED_SEED_PASSES:
        classification = "nonlinear_shared_response_required"
        primary = False
    elif true_affine >= REQUIRED_SEED_PASSES and true_residual >= REQUIRED_SEED_PASSES:
        classification = "affine_and_nonlinear_paths_redundant"
        primary = False
    else:
        classification = "mixed_sensor_mechanisms"
        primary = False
    return {
        "classification": classification,
        "primary_hypothesis_pass": primary,
        "valid": valid,
        "full_replay_pass_count": full_replay,
        "true_affine_only_pass_count": true_affine,
        "true_nonlinear_residual_only_pass_count": true_residual,
        "shuffled_affine_only_pass_count": shuffled_affine,
        "required_seed_passes": REQUIRED_SEED_PASSES,
        "control_seed_passes_maximum": CONTROL_SEED_PASSES_MAXIMUM,
    }


def _resolve_device(device_name: str | None) -> torch.device:
    if device_name in {None, "auto"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested analysis device {device}, CUDA unavailable")
    return device


def build_result(device_name: str | None = None) -> dict[str, Any]:
    device = _resolve_device(device_name)
    campaign, details = validate_source_campaign()
    config = source._config_from_mapping(campaign["configuration"])
    task = stage0.C3TaskConfig()
    datasets = source.build_held_out_datasets(task, config)
    if source.held_out_hashes(datasets) != campaign["evaluation_hashes"]:
        raise RuntimeError("held-out sensor cohorts changed")
    cells = [
        analyze_checkpoint(
            seed=seed,
            arm=arm,
            detail=details[seed],
            config=config,
            datasets=datasets,
            device=device,
        )
        for seed in SEEDS
        for arm in SOURCE_ARMS
    ]
    valid = len(cells) == 10 and all(cell["valid"] for cell in cells)
    aggregates = classify(cells, valid)
    sources = {
        "runner": _sha256(Path(__file__)),
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "source_campaign": _sha256(SOURCE_CAMPAIGN_PATH),
        "source_runner": _sha256(Path(source.__file__)),
    }
    record = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed" if valid else "invalid",
        "evidence_role": EVIDENCE_ROLE,
        "completed_at": _utc_now(),
        "source": {
            "campaign": str(SOURCE_CAMPAIGN_PATH),
            "campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "result_manifest_sha256": SOURCE_RESULT_MANIFEST_SHA256,
            "checkpoint_manifest_sha256": SOURCE_CHECKPOINT_MANIFEST_SHA256,
        },
        "source_hashes": sources,
        "implementation_sha256": _json_hash(sources),
        "configuration": {
            "seeds": list(SEEDS),
            "source_arms": list(SOURCE_ARMS),
            "response_arms": list(RESPONSE_ARMS),
            "replay_tolerance": REPLAY_TOLERANCE,
            "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
            "slope_magnitude_minimum": SLOPE_MAGNITUDE_MINIMUM,
            "carrier_dot_minimum": CARRIER_DOT_MINIMUM,
            "carrier_rmse_maximum": CARRIER_RMSE_MAXIMUM,
            "action_error_maximum": ACTION_ERROR_MAXIMUM,
            "required_seed_passes": REQUIRED_SEED_PASSES,
            "control_seed_passes_maximum": CONTROL_SEED_PASSES_MAXIMUM,
            "analysis_device": str(device),
        },
        "cells": cells,
        "aggregates": aggregates,
        "accounting": {
            "checkpoints_loaded": 10,
            "optimizer_steps": 0,
            "parameters_changed": 0,
            "tinyllm_models_instantiated": 0,
            "target_using_fits": 0,
            "target_free_complex_affine_fits": 10,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "method_boundaries": [
            "The mechanism question was selected after the positive sensor-only outcome.",
            "Affine response fits use reconstructed training observations but no targets, phase, carrier, or held-out metrics.",
            "The response patches alter frozen computation without changing checkpoint parameters.",
            "Functional affine sufficiency does not imply parameter equality to the closed-form witness.",
        ],
    }
    if not _finite(record):
        raise RuntimeError("non-finite C3 sensor mechanism result")
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PRIMARY_RESULT_PATH,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="analysis device; auto reuses CUDA when available",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_result(args.device)
    _write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "classification": result["aggregates"]["classification"],
                "true_affine_passes": result["aggregates"][
                    "true_affine_only_pass_count"
                ],
                "true_residual_passes": result["aggregates"][
                    "true_nonlinear_residual_only_pass_count"
                ],
                "shuffled_affine_passes": result["aggregates"][
                    "shuffled_affine_only_pass_count"
                ],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
