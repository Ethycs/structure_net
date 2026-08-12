#!/usr/bin/env python3
"""Preflight the existing exact-C3 sensor against the analytic carrier."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import torch

import experiments.structure_net.tinyllm_c3_temporal_quotient_preflight as preflight
import experiments.structure_net.tinyllm_c3_temporal_quotient_training as stage0
import experiments.structure_net.tinyllm_joint_physical_scalar_interface as joint


SCHEMA_VERSION = "nal.tinyllm-c3-temporal-sensor-function-class.v1"
HYPOTHESIS_ID = "tinyllm-c3-temporal-sensor-function-class-v1"
EVIDENCE_ROLE = "prospective_no_training_c3_sensor_function_class_preflight"
CLASSIFICATION_PASS = (
    "existing_c3_sensor_contains_analytic_carrier_and_task_gradient"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-temporal-sensor-function-class-"
    "preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "83f20896853cb7612618c93d3302a5bfa0b2091b6c957625eecbe3451401701f"
)
PINNED_SOURCE_SHA256 = {
    "sensor_family": "dbc934190b5f725185ff9a99690409ac63fc4f16bd04686f870e7ef21cba03a6",
    "generator": "812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118",
    "interval_likelihood": "b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6",
    "preregistration": PREREGISTRATION_SHA256,
}
REGIMES = ("composition", "extrapolation")
PARAMETER_COUNT = 184
SCALAR_GRID_POINTS = 10_001
SCALAR_RECONSTRUCTION_MAXIMUM = 5e-7
CARRIER_ERROR_MAXIMUM = 2e-6
ACTION_ERROR_MAXIMUM = 2e-6
TEMPORAL_CORRELATION_MINIMUM = 0.99
TEMPORAL_RMSE_MAXIMUM = 0.08
SHUFFLED_ABSOLUTE_CORRELATION_MAXIMUM = 0.10
SHUFFLED_RMSE_MINIMUM = 0.80
GRADIENT_SEED = 57_031
GRADIENT_DATA_SEED = 97_103
GRADIENT_SAMPLE_COUNT = 512
GRADIENT_NORM_MINIMUM = 1e-6
GRADIENT_NONZERO_FRACTION_MINIMUM = 0.90
DECK_LOSS_ERROR_MAXIMUM = 1e-6
DECK_GRADIENT_ERROR_MAXIMUM = 2e-5
DIAGNOSTIC_STEP_RADIUS = 1e-3
DIAGNOSTIC_LOSS_DECREASE_MINIMUM = 1e-4
TASK_GATES = {
    "posterior_mean_correlation_minimum": 0.90,
    "composition_accuracy_minimum": 0.50,
    "extrapolation_accuracy_minimum": 0.35,
    "composition_cross_entropy_maximum": 1.80,
    "extrapolation_cross_entropy_maximum": 2.20,
    "composition_coverage_minimum": 14,
    "extrapolation_coverage_minimum": 12,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _source_hashes() -> dict[str, str]:
    return {
        "sensor_family": _sha256(Path(stage0.__file__)),
        "generator": _sha256(Path(preflight.__file__)),
        "interval_likelihood": _sha256(Path(joint.__file__)),
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "runner": _sha256(Path(__file__)),
    }


def _state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.reshape(-1).double()
    y = right.reshape(-1).double()
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-12))


def construct_analytic_witness(
    encoder: stage0.LearnedC3InvariantEncoder,
) -> stage0.LearnedC3InvariantEncoder:
    """Assign a closed-form GELU-difference witness without fitting."""
    first = encoder.shared_map[0]
    second = encoder.shared_map[2]
    if not (
        isinstance(first, torch.nn.Linear)
        and isinstance(second, torch.nn.Linear)
        and first.in_features == 1
        and first.out_features >= 2
        and second.in_features == first.out_features
        and second.out_features >= 1
    ):
        raise ValueError("learned C3 sensor structure changed")
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.zero_()
        first.weight[0, 0] = 1.0
        first.weight[1, 0] = -1.0
        second.weight[0, 0] = 1.0
        second.weight[0, 1] = -1.0
        encoder.mixer_real[0] = 1.0
    return encoder


def _sensor_prediction(
    encoder: stage0.LearnedC3InvariantEncoder,
    tokens: torch.Tensor,
    calibration: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    corrected = stage0.corrected_channels(tokens, calibration)
    feature = encoder(corrected)
    carrier = torch.complex(feature[..., 0], feature[..., 1])
    temporal = preflight.temporal_prediction(carrier.to(torch.complex128))
    posterior = joint.interval_posterior_unclipped(temporal, 16)
    return carrier, temporal, posterior


def _task_metrics(
    posterior: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, Any]:
    target_posterior = joint.interval_posterior_unclipped(target.double(), 16)
    centers = torch.linspace(-1.0, 1.0, 16, dtype=torch.float64)
    predicted = posterior.double() @ centers
    predicted_bins = posterior.argmax(-1)
    target_bins = target_posterior.argmax(-1)
    return {
        "posterior_mean_correlation": _correlation(predicted, target),
        "posterior_mean_rmse": float(
            torch.sqrt((predicted - target.double()).square().mean())
        ),
        "exact_bin_accuracy": float(
            (predicted_bins == target_bins).double().mean()
        ),
        "target_cross_entropy": float(
            -(
                target_posterior
                * posterior.double().clamp_min(1e-12).log()
            ).sum(-1).mean()
        ),
        "predicted_bin_coverage": int(torch.unique(predicted_bins).numel()),
    }


def _task_gate(metrics: Mapping[str, Any], regime: str) -> bool:
    return (
        float(metrics["posterior_mean_correlation"])
        >= TASK_GATES["posterior_mean_correlation_minimum"]
        and float(metrics["exact_bin_accuracy"])
        >= TASK_GATES[f"{regime}_accuracy_minimum"]
        and float(metrics["target_cross_entropy"])
        <= TASK_GATES[f"{regime}_cross_entropy_maximum"]
        and int(metrics["predicted_bin_coverage"])
        >= TASK_GATES[f"{regime}_coverage_minimum"]
    )


@torch.no_grad()
def analyze_witness() -> dict[str, Any]:
    encoder = construct_analytic_witness(stage0.LearnedC3InvariantEncoder())
    parameter_count = sum(value.numel() for value in encoder.parameters())
    nonzero_parameter_count = sum(
        int(torch.count_nonzero(value)) for value in encoder.parameters()
    )
    grid = torch.linspace(-4.0, 4.0, SCALAR_GRID_POINTS)
    mapped = encoder.shared_map(grid[:, None])[:, 0]
    scalar_error = float((mapped - grid).abs().max())
    regimes = {}
    for regime in REGIMES:
        dataset = preflight.generate_dataset(regime)
        carrier, temporal, posterior = _sensor_prediction(
            encoder, dataset.tokens, dataset.calibration
        )
        analytic, _ = preflight.analytic_carrier(
            dataset.tokens, dataset.calibration
        )
        action_errors = {}
        for element in (1, 2):
            transformed, _, _ = _sensor_prediction(
                encoder,
                preflight.apply_deck_action(dataset.tokens, element),
                dataset.calibration,
            )
            action_errors[str(element)] = float(
                (transformed - carrier).abs().max()
            )
        shuffled_target = torch.roll(dataset.target, shifts=137)
        task_metrics = _task_metrics(posterior, dataset.target)
        temporal_correlation = _correlation(temporal, dataset.target)
        temporal_rmse = float(
            torch.sqrt((temporal - dataset.target).square().mean())
        )
        shuffled_correlation = _correlation(temporal, shuffled_target)
        shuffled_rmse = float(
            torch.sqrt((temporal - shuffled_target).square().mean())
        )
        carrier_error = float(
            (carrier.to(torch.complex128) - analytic).abs().max()
        )
        maximum_action_error = max(action_errors.values())
        passed = (
            dataset.saturation_count == 0
            and carrier_error <= CARRIER_ERROR_MAXIMUM
            and maximum_action_error <= ACTION_ERROR_MAXIMUM
            and temporal_correlation >= TEMPORAL_CORRELATION_MINIMUM
            and temporal_rmse <= TEMPORAL_RMSE_MAXIMUM
            and _task_gate(task_metrics, regime)
            and abs(shuffled_correlation)
            <= SHUFFLED_ABSOLUTE_CORRELATION_MAXIMUM
            and shuffled_rmse >= SHUFFLED_RMSE_MINIMUM
        )
        regimes[regime] = {
            "sample_count": int(len(dataset.target)),
            "quantizer_saturation_count": int(dataset.saturation_count),
            "carrier_maximum_absolute_error": carrier_error,
            "deck_action_maximum_absolute_error": action_errors,
            "maximum_nonidentity_action_error": maximum_action_error,
            "temporal_target_correlation": temporal_correlation,
            "temporal_target_rmse": temporal_rmse,
            "fixed_interval_task": task_metrics,
            "fixed_interval_task_pass": _task_gate(task_metrics, regime),
            "shuffled_target_correlation": shuffled_correlation,
            "shuffled_target_rmse": shuffled_rmse,
            "pass": passed,
        }
    return {
        "parameter_count": parameter_count,
        "nonzero_parameter_count": nonzero_parameter_count,
        "scalar_grid_points": SCALAR_GRID_POINTS,
        "scalar_reconstruction_maximum_absolute_error": scalar_error,
        "regimes": regimes,
        "pass": (
            parameter_count == PARAMETER_COUNT
            and nonzero_parameter_count == 5
            and scalar_error <= SCALAR_RECONSTRUCTION_MAXIMUM
            and all(item["pass"] for item in regimes.values())
        ),
    }


def _task_loss(
    encoder: stage0.LearnedC3InvariantEncoder,
    dataset: stage0.C3TrainingDataset,
    tokens: torch.Tensor,
) -> torch.Tensor:
    _, _, posterior = _sensor_prediction(
        encoder, tokens, dataset.calibration
    )
    target = dataset.target_posteriors.double()
    return -(target * posterior.clamp_min(1e-12).log()).sum(-1).mean()


def _gradients(
    loss: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[torch.Tensor, ...]:
    values = torch.autograd.grad(loss, parameters)
    if len(values) != len(parameters):
        raise AssertionError("gradient route did not return every parameter tensor")
    return values


def analyze_gradient_route() -> dict[str, Any]:
    task = stage0.C3TaskConfig()
    dataset = stage0.generate_training_dataset(
        task,
        sample_count=GRADIENT_SAMPLE_COUNT,
        seed=GRADIENT_DATA_SEED,
    )
    torch.manual_seed(GRADIENT_SEED)
    encoder = stage0.LearnedC3InvariantEncoder()
    parameters = tuple(encoder.parameters())
    names = tuple(name for name, _ in encoder.named_parameters())
    state_before = _state_digest(encoder)
    loss = _task_loss(encoder, dataset, dataset.tokens)
    gradients = _gradients(loss, parameters)
    vector = torch.cat([value.reshape(-1) for value in gradients])
    tensor_norms = {
        name: float(value.norm()) for name, value in zip(names, gradients)
    }
    total_norm = float(vector.norm())
    nonzero_count = int(torch.count_nonzero(vector))
    nonzero_fraction = nonzero_count / float(vector.numel())

    deck_loss = _task_loss(
        encoder,
        dataset,
        preflight.apply_deck_action(dataset.tokens, 1),
    )
    deck_gradients = _gradients(deck_loss, parameters)
    deck_vector = torch.cat([value.reshape(-1) for value in deck_gradients])
    deck_loss_error = float((deck_loss - loss).abs())
    deck_gradient_error = float((deck_vector - vector).abs().max())

    snapshots = [value.detach().clone() for value in parameters]
    with torch.no_grad():
        denominator = vector.norm().clamp_min(1e-12)
        for parameter, gradient in zip(parameters, gradients):
            parameter.add_(gradient, alpha=-DIAGNOSTIC_STEP_RADIUS / float(denominator))
    perturbed_loss = _task_loss(encoder, dataset, dataset.tokens)
    loss_decrease = float(loss.detach() - perturbed_loss.detach())
    with torch.no_grad():
        for parameter, snapshot in zip(parameters, snapshots):
            parameter.copy_(snapshot)
    state_after = _state_digest(encoder)

    finite = (
        bool(torch.isfinite(loss))
        and bool(torch.isfinite(deck_loss))
        and bool(torch.isfinite(perturbed_loss))
        and bool(torch.isfinite(vector).all())
        and bool(torch.isfinite(deck_vector).all())
    )
    passed = (
        finite
        and total_norm >= GRADIENT_NORM_MINIMUM
        and nonzero_fraction >= GRADIENT_NONZERO_FRACTION_MINIMUM
        and deck_loss_error <= DECK_LOSS_ERROR_MAXIMUM
        and deck_gradient_error <= DECK_GRADIENT_ERROR_MAXIMUM
        and loss_decrease >= DIAGNOSTIC_LOSS_DECREASE_MINIMUM
        and state_before == state_after
    )
    return {
        "random_seed": GRADIENT_SEED,
        "dataset_seed": GRADIENT_DATA_SEED,
        "sample_count": GRADIENT_SAMPLE_COUNT,
        "loss": float(loss.detach()),
        "total_gradient_norm": total_norm,
        "gradient_tensor_norms": tensor_norms,
        "nonzero_gradient_parameter_count": nonzero_count,
        "nonzero_gradient_fraction": nonzero_fraction,
        "deck_rolled_loss": float(deck_loss.detach()),
        "deck_loss_absolute_error": deck_loss_error,
        "deck_gradient_maximum_absolute_error": deck_gradient_error,
        "diagnostic_step_radius": DIAGNOSTIC_STEP_RADIUS,
        "diagnostic_perturbed_loss": float(perturbed_loss.detach()),
        "diagnostic_loss_decrease": loss_decrease,
        "state_sha256_before": state_before,
        "state_sha256_after_restore": state_after,
        "finite": finite,
        "pass": passed,
    }


def build_result() -> dict[str, Any]:
    sources = _source_hashes()
    source_contract = all(
        sources[name] == expected
        for name, expected in PINNED_SOURCE_SHA256.items()
    )
    witness = analyze_witness()
    gradient = analyze_gradient_route()
    if not source_contract:
        classification = "invalid_source_contract"
    elif not witness["pass"]:
        classification = "existing_c3_sensor_function_class_insufficient"
    elif not gradient["pass"]:
        classification = "analytic_witness_exists_but_task_gradient_blocked"
    else:
        classification = CLASSIFICATION_PASS
    record = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "evidence_role": EVIDENCE_ROLE,
        "status": "completed",
        "completed_at": _utc_now(),
        "classification": classification,
        "source_hashes": sources,
        "pinned_source_hashes": PINNED_SOURCE_SHA256,
        "source_contract_pass": source_contract,
        "witness": witness,
        "gradient_route": gradient,
        "gates": {
            "source_contract": source_contract,
            "analytic_function_class_witness": witness["pass"],
            "target_gradient_route": gradient["pass"],
            "campaign_licensed": classification == CLASSIFICATION_PASS,
            "zero_optimizer_steps": True,
            "no_tinyllm_instantiated": True,
            "no_checkpoint_loaded": True,
        },
        "accounting": {
            "optimizer_steps": 0,
            "tinyllm_models_instantiated": 0,
            "checkpoints_loaded": 0,
            "trained_parameters": 0,
            "sensor_family_parameters": PARAMETER_COUNT,
            "closed_form_nonzero_witness_parameters": witness[
                "nonzero_parameter_count"
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": "cpu",
        },
        "method_boundaries": [
            "The five-nonzero-parameter state is assigned analytically and is not fitted.",
            "The normalized negative-gradient perturbation is restored and is not an optimizer step.",
            "Passing establishes function-class capacity and a local target-gradient path, not global trainability.",
            "No TinyLLM model or historical checkpoint is instantiated or loaded.",
        ],
    }
    if not _finite(record):
        record["classification"] = "invalid_source_contract"
        record["status"] = "invalid"
        record["gates"]["campaign_licensed"] = False
        raise RuntimeError("non-finite C3 sensor function-class result")
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_c3_temporal_sensor_function_class/"
            "20260811_preregistered/result.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_result()
    _write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "classification": result["classification"],
                "campaign_licensed": result["gates"]["campaign_licensed"],
                "witness_pass": result["witness"]["pass"],
                "gradient_route_pass": result["gradient_route"]["pass"],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
