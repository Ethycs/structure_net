#!/usr/bin/env python3
"""Preflight an exact C3 connection-conditioned relational function class."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import tempfile
from typing import Any, Mapping, Sequence

import torch
from torch import nn

import experiments.structure_net.tinyllm_c3_relational_connection_preflight as rel
import experiments.structure_net.tinyllm_c3_temporal_quotient_training as stage0
import experiments.structure_net.tinyllm_joint_physical_scalar_interface as joint


SCHEMA_VERSION = "nal.tinyllm-c3-relational-connection-function-class.v1"
HYPOTHESIS_ID = "tinyllm-c3-relational-connection-function-class-v1"
EVIDENCE_ROLE = "no_training_exact_function_class_gradient_and_lifecycle_preflight"
CLASSIFICATION_PASS = (
    "connection_invariant_function_class_contains_transport_and_task_gradient"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-relational-connection-function-class-"
    "preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "421573a12a09a6782bd44b16e0f57e5e13a158f5bc740d19b8d4447964c5ae86"
)
RELATIONAL_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_relational_connection_preflight.py"
)
RELATIONAL_RUNNER_SHA256 = (
    "2ab21b7885fa93c49427973f67e5c57913b7f13c154ad2e970ef9636b2b90214"
)
RELATIONAL_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_relational_connection_preflight/"
    "20260811_preregistered/result.json"
)
RELATIONAL_RESULT_SHA256 = (
    "ae0321b3c3d5b7e8c80ebbd57bfaea10c0522c4b2241082daab007a54e55984e"
)
RELATIONAL_REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-relational-connection-preflight.md"
)
RELATIONAL_REPORT_SHA256 = (
    "4631f3ab2f99702e384d8b66c1dac4251cb63f4207ddbf0dca03cbe413a40aff"
)
SENSOR_FAMILY_PATH = Path(
    "experiments/structure_net/tinyllm_c3_temporal_quotient_training.py"
)
SENSOR_FAMILY_SHA256 = (
    "dbc934190b5f725185ff9a99690409ac63fc4f16bd04686f870e7ef21cba03a6"
)
INTERVAL_PATH = Path(
    "experiments/structure_net/tinyllm_joint_physical_scalar_interface.py"
)
INTERVAL_SHA256 = (
    "b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6"
)
PRIMARY_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_relational_connection_function_class/"
    "20260811_preregistered/result.json"
)

ENCODER_PARAMETER_COUNT = 184
HEAD_PARAMETER_COUNT = 3
TOTAL_PARAMETER_COUNT = 187
WITNESS_NONZERO_PARAMETER_COUNT = 6
GRADIENT_PARAMETER_SEED = 62_117
GRADIENT_DATA_SEED = 1_143_107
LOCAL_ACTION_SEED = 1_145_107
CONNECTION_SHUFFLE_SEED = 1_147_107
TARGET_SHUFFLE_SEED = 1_149_107
PERTURBATION_SEED = 63_103
GRADIENT_SAMPLE_COUNT = 512
CHARACTER_ERROR_MAXIMUM = 2e-6
INVARIANCE_ERROR_MAXIMUM = 2e-5
ACTION_LOSS_ERROR_MAXIMUM = 5e-6
ACTION_GRADIENT_ERROR_MAXIMUM = 2e-5
GRADIENT_NORM_MINIMUM = 1e-6
GRADIENT_NONZERO_FRACTION_MINIMUM = 0.90
CONTROL_GRADIENT_COSINE_MAXIMUM = 0.95
CONTROL_RELATIVE_GRADIENT_DIFFERENCE_MINIMUM = 0.20
DIAGNOSTIC_STEP_RADIUS = 1e-3
DIAGNOSTIC_LOSS_DECREASE_MINIMUM = 1e-4
CUDA_CPU_OUTPUT_ERROR_MAXIMUM = 5e-5
POSITIVE_GATES = rel.POSITIVE_GATES


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
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


def _state_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_sources() -> tuple[dict[str, str], dict[str, Any]]:
    paths = {
        "preregistration": PREREGISTRATION_PATH,
        "relational_runner": RELATIONAL_RUNNER_PATH,
        "relational_result": RELATIONAL_RESULT_PATH,
        "relational_report": RELATIONAL_REPORT_PATH,
        "sensor_family": SENSOR_FAMILY_PATH,
        "interval_likelihood": INTERVAL_PATH,
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "relational_runner": RELATIONAL_RUNNER_SHA256,
        "relational_result": RELATIONAL_RESULT_SHA256,
        "relational_report": RELATIONAL_REPORT_SHA256,
        "sensor_family": SENSOR_FAMILY_SHA256,
        "interval_likelihood": INTERVAL_SHA256,
    }
    observed = {name: _sha256(path) for name, path in paths.items()}
    if observed != expected:
        raise RuntimeError(f"connection function-class sources changed: {observed}")
    result = json.loads(RELATIONAL_RESULT_PATH.read_text(encoding="utf-8"))
    aggregates = result.get("aggregates", {})
    if (
        result.get("status") != "completed"
        or aggregates.get("classification")
        != "observed_edge_connection_identifies_nonpointwise_c3_relation"
        or aggregates.get(
            "connection_conditioned_function_class_preflight_licensed"
        )
        is not True
        or aggregates.get("matched_training_directly_licensed") is not False
    ):
        raise RuntimeError("relational predecessor no longer licenses preflight")
    return observed, result


class LearnedC3ChargedEncoder(nn.Module):
    """Shared channel map followed by a normalized charge-one C3 character."""

    def __init__(self, hidden: int = 16, character_channels: int = 8):
        super().__init__()
        self.shared_map = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, character_channels),
        )
        self.mixer_real = nn.Parameter(torch.empty(character_channels))
        self.mixer_imag = nn.Parameter(torch.empty(character_channels))
        nn.init.normal_(
            self.mixer_real, std=1.0 / math.sqrt(character_channels)
        )
        nn.init.normal_(
            self.mixer_imag, std=1.0 / math.sqrt(character_channels)
        )
        angles = (
            2.0 * math.pi * torch.arange(rel.CHANNELS, dtype=torch.float32)
            / rel.CHANNELS
        )
        self.register_buffer("character_real", torch.cos(-angles))
        self.register_buffer("character_imag", torch.sin(-angles))

    def forward(self, corrected: torch.Tensor) -> torch.Tensor:
        features = self.shared_map(corrected.unsqueeze(-1))
        real = torch.einsum(
            "btck,c->btk", features, self.character_real
        )
        imaginary = torch.einsum(
            "btck,c->btk", features, self.character_imag
        )
        mixed_real = torch.einsum(
            "btk,k->bt", real, self.mixer_real
        ) - torch.einsum("btk,k->bt", imaginary, self.mixer_imag)
        mixed_imaginary = torch.einsum(
            "btk,k->bt", real, self.mixer_imag
        ) + torch.einsum("btk,k->bt", imaginary, self.mixer_real)
        magnitude = torch.sqrt(
            mixed_real.square() + mixed_imaginary.square()
        ).clamp_min(1e-6)
        return torch.stack(
            (mixed_real / magnitude, mixed_imaginary / magnitude), dim=-1
        )


class ConnectionInvariantRelationalModule(nn.Module):
    """Exact neutral product of charged endpoints and an observed connection."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = LearnedC3ChargedEncoder()
        self.head = nn.Linear(2, 1)

    def charged(self, tokens: torch.Tensor, calibration: torch.Tensor) -> torch.Tensor:
        return self.encoder(stage0.corrected_channels(tokens, calibration))

    def neutral(
        self,
        tokens: torch.Tensor,
        calibration: torch.Tensor,
        connection: torch.Tensor,
    ) -> torch.Tensor:
        feature = self.charged(tokens, calibration)
        character = torch.complex(feature[..., 0], feature[..., 1])
        total = connection.sum(dim=1) % rel.CHANNELS
        angle = (
            2.0 * math.pi * total.to(feature.dtype) / rel.CHANNELS
        )
        transport = torch.polar(torch.ones_like(angle), angle)
        relative = (
            character[:, -1] * transport * character[:, 0].conj()
        )
        return torch.stack((relative.real, relative.imag), dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        calibration: torch.Tensor,
        connection: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(
            self.neutral(tokens, calibration, connection)
        ).squeeze(-1)


def construct_analytic_witness(
    module: ConnectionInvariantRelationalModule,
) -> ConnectionInvariantRelationalModule:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
        first = module.encoder.shared_map[0]
        second = module.encoder.shared_map[2]
        first.weight[0, 0] = 1.0
        first.weight[1, 0] = -1.0
        second.weight[0, 0] = 1.0
        second.weight[0, 1] = -1.0
        module.encoder.mixer_real[0] = 1.0
        module.head.weight[0, 0] = 1.0
    return module


def module_contract(module: ConnectionInvariantRelationalModule) -> dict[str, Any]:
    encoder_parameters = sum(
        value.numel() for value in module.encoder.parameters()
    )
    head_parameters = sum(value.numel() for value in module.head.parameters())
    total_parameters = sum(value.numel() for value in module.parameters())
    return {
        "encoder_parameters": encoder_parameters,
        "head_parameters": head_parameters,
        "total_parameters": total_parameters,
        "head_input_features": module.head.in_features,
        "head_output_features": module.head.out_features,
        "shared_channel_map": True,
        "only_raw_to_output_path": (
            "shared_map->first_character->connection_neutral_product->head"
        ),
        "all_parameter_invariance_proof": (
            "channel permutation covariance of shared map and first character; "
            "edge coboundary cancellation in endpoint neutral product"
        ),
        "pass": bool(
            encoder_parameters == ENCODER_PARAMETER_COUNT
            and head_parameters == HEAD_PARAMETER_COUNT
            and total_parameters == TOTAL_PARAMETER_COUNT
            and module.head.in_features == 2
            and module.head.out_features == 1
        ),
    }


def _complex_feature(feature: torch.Tensor) -> torch.Tensor:
    return torch.complex(feature[..., 0], feature[..., 1])


def _positive_metrics_pass(
    scalar: Mapping[str, Any], task: Mapping[str, Any]
) -> bool:
    return bool(
        float(scalar["correlation"])
        >= POSITIVE_GATES["scalar_correlation_minimum"]
        and float(scalar["rmse"]) <= POSITIVE_GATES["scalar_rmse_maximum"]
        and float(task["exact_bin_accuracy"])
        >= POSITIVE_GATES["exact_bin_accuracy_minimum"]
        and float(task["target_cross_entropy"])
        <= POSITIVE_GATES["target_cross_entropy_maximum"]
        and int(task["predicted_bin_coverage"])
        == POSITIVE_GATES["predicted_bin_coverage"]
    )


@torch.no_grad()
def analyze_witness(predecessor: Mapping[str, Any]) -> dict[str, Any]:
    module = construct_analytic_witness(ConnectionInvariantRelationalModule())
    contract = module_contract(module)
    nonzero = sum(int(torch.count_nonzero(value)) for value in module.parameters())
    expected_hashes = {
        (int(cell["seed"]), cell["regime"]): cell["dataset_sha256"]
        for cell in predecessor["cells"]
    }
    cells = []
    for seed in rel.SEEDS:
        for regime in rel.REGIMES:
            dataset = rel.generate_dataset(regime, seed)
            observed_hash = rel.dataset_hash(dataset)
            feature = module.charged(dataset.tokens, dataset.calibration)
            character = _complex_feature(feature).to(torch.complex128)
            analytic, _ = rel.charged_character(
                dataset.tokens, dataset.calibration
            )
            character_error = float((character - analytic).abs().max())
            prediction = module(
                dataset.tokens, dataset.calibration, dataset.connection
            )
            scalar = rel._scalar_metrics(prediction, dataset.target)
            task = rel._task_metrics(prediction, dataset.target)
            action = rel.action_stream(regime, seed, rel.SAMPLE_COUNT)
            transformed_tokens = rel.apply_local_action(dataset.tokens, action)
            transformed_connection = rel.transform_connection(
                dataset.connection, action
            )
            transformed = module(
                transformed_tokens,
                dataset.calibration,
                transformed_connection,
            )
            action_error = float((transformed - prediction).abs().max())
            passed = bool(
                observed_hash == expected_hashes[(seed, regime)]
                and character_error <= CHARACTER_ERROR_MAXIMUM
                and _positive_metrics_pass(scalar, task)
                and action_error <= INVARIANCE_ERROR_MAXIMUM
            )
            cells.append(
                {
                    "seed": seed,
                    "regime": regime,
                    "dataset_sha256": observed_hash,
                    "dataset_hash_match": (
                        observed_hash == expected_hashes[(seed, regime)]
                    ),
                    "charged_character_maximum_error": character_error,
                    "scalar": scalar,
                    "task": task,
                    "local_action_prediction_maximum_error": action_error,
                    "pass": passed,
                }
            )
    return {
        "module_contract": contract,
        "nonzero_parameter_count": nonzero,
        "state_sha256": _state_digest(module),
        "cells": cells,
        "pass": bool(
            contract["pass"]
            and nonzero == WITNESS_NONZERO_PARAMETER_COUNT
            and all(cell["pass"] for cell in cells)
        ),
    }


def generate_diagnostic_dataset(
    *, seed: int = GRADIENT_DATA_SEED, count: int = GRADIENT_SAMPLE_COUNT
) -> rel.RelationalDataset:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    theta_0 = 2.0 * math.pi * torch.rand(
        count, dtype=torch.float64, generator=generator
    )
    delta = math.pi * (
        2.0 * torch.rand(count, dtype=torch.float64, generator=generator) - 1.0
    )
    interior = 2.0 * math.pi * torch.rand(
        count,
        rel.TIME_STEPS - 2,
        dtype=torch.float64,
        generator=generator,
    )
    phase = torch.cat(
        (theta_0[:, None], interior, (theta_0 + delta)[:, None]), dim=1
    )
    ranges = rel.source.REGIME_RANGES["composition"]

    def uniform(bounds: tuple[float, float]) -> torch.Tensor:
        return torch.empty(count, dtype=torch.float64).uniform_(
            *bounds, generator=generator
        )

    amplitude = uniform(ranges["amplitude"])
    offset = uniform(ranges["offset"])
    drift = uniform(ranges["drift"])
    gauge = torch.randint(
        0,
        rel.CHANNELS,
        (count, rel.TIME_STEPS),
        generator=generator,
        dtype=torch.int64,
    )
    continuous = rel._continuous_observation(phase, amplitude, offset, drift)
    canonical = rel.source.quantize(continuous)
    calibration = torch.stack((amplitude, offset, drift), dim=-1)
    return rel.RelationalDataset(
        canonical_tokens=canonical,
        tokens=rel.apply_local_action(canonical, gauge),
        calibration=calibration,
        target=torch.cos(delta),
        phase=phase,
        delta=delta,
        gauge=gauge,
        connection=rel.edge_connection(gauge),
        saturation_count=int(
            (continuous.abs() >= rel.source.QUANTIZATION_LIMIT).sum()
        ),
        dataset_seed=seed,
    )


def _target_posterior(target: torch.Tensor) -> torch.Tensor:
    return joint.interval_posterior_unclipped(target.double(), 16)


def _task_loss(
    module: ConnectionInvariantRelationalModule,
    dataset: rel.RelationalDataset,
    *,
    tokens: torch.Tensor | None = None,
    connection: torch.Tensor | None = None,
    target_posterior: torch.Tensor | None = None,
) -> torch.Tensor:
    prediction = module(
        dataset.tokens if tokens is None else tokens,
        dataset.calibration,
        dataset.connection if connection is None else connection,
    )
    posterior = joint.interval_posterior_unclipped(prediction.double(), 16)
    target = (
        _target_posterior(dataset.target)
        if target_posterior is None
        else target_posterior
    )
    return -(target * posterior.clamp_min(1e-12).log()).sum(dim=-1).mean()


def _gradient_vector(
    loss: torch.Tensor, parameters: tuple[nn.Parameter, ...]
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    gradients = torch.autograd.grad(loss, parameters)
    return gradients, torch.cat([value.reshape(-1) for value in gradients])


def _gradient_comparison(
    candidate: torch.Tensor, reference: torch.Tensor
) -> dict[str, float | bool]:
    cosine = float(torch.nn.functional.cosine_similarity(candidate, reference, dim=0))
    relative = float(
        torch.linalg.vector_norm(candidate - reference)
        / torch.linalg.vector_norm(reference).clamp_min(1e-12)
    )
    return {
        "cosine_with_true_gradient": cosine,
        "relative_vector_difference": relative,
        "pass": bool(
            cosine <= CONTROL_GRADIENT_COSINE_MAXIMUM
            and relative >= CONTROL_RELATIVE_GRADIENT_DIFFERENCE_MINIMUM
        ),
    }


def _parameter_perturbation(
    module: ConnectionInvariantRelationalModule,
) -> ConnectionInvariantRelationalModule:
    perturbed = copy.deepcopy(module)
    generator = torch.Generator(device="cpu").manual_seed(PERTURBATION_SEED)
    with torch.no_grad():
        for parameter in perturbed.parameters():
            noise = torch.randn(
                parameter.shape,
                dtype=parameter.dtype,
                device="cpu",
                generator=generator,
            )
            parameter.add_(noise.to(parameter.device), alpha=0.25)
    return perturbed


@torch.no_grad()
def _state_invariance(
    name: str,
    module: ConnectionInvariantRelationalModule,
    dataset: rel.RelationalDataset,
    action: torch.Tensor,
) -> dict[str, Any]:
    transformed_tokens = rel.apply_local_action(dataset.tokens, action)
    transformed_connection = rel.transform_connection(dataset.connection, action)
    base_feature = module.charged(dataset.tokens, dataset.calibration)
    transformed_feature = module.charged(
        transformed_tokens, dataset.calibration
    )
    base_character = _complex_feature(base_feature)
    transformed_character = _complex_feature(transformed_feature)
    expected = base_character * torch.polar(
        torch.ones_like(action, dtype=base_feature.dtype),
        -2.0 * math.pi * action.to(base_feature.dtype) / rel.CHANNELS,
    )
    character_error = float((transformed_character - expected).abs().max())
    base_output = module(dataset.tokens, dataset.calibration, dataset.connection)
    transformed_output = module(
        transformed_tokens, dataset.calibration, transformed_connection
    )
    output_error = float((transformed_output - base_output).abs().max())
    return {
        "state": name,
        "state_sha256": _state_digest(module),
        "charged_covariance_maximum_error": character_error,
        "output_invariance_maximum_error": output_error,
        "finite": bool(
            torch.isfinite(base_feature).all()
            and torch.isfinite(base_output).all()
        ),
        "pass": bool(
            character_error <= INVARIANCE_ERROR_MAXIMUM
            and output_error <= INVARIANCE_ERROR_MAXIMUM
            and torch.isfinite(base_feature).all()
            and torch.isfinite(base_output).all()
        ),
    }


def analyze_gradient_and_invariance() -> tuple[dict[str, Any], rel.RelationalDataset, ConnectionInvariantRelationalModule]:
    dataset = generate_diagnostic_dataset()
    torch.manual_seed(GRADIENT_PARAMETER_SEED)
    module = ConnectionInvariantRelationalModule()
    parameters = tuple(module.parameters())
    names = tuple(name for name, _ in module.named_parameters())
    state_before = _state_digest(module)
    action_generator = torch.Generator(device="cpu").manual_seed(
        LOCAL_ACTION_SEED
    )
    action = torch.randint(
        0,
        rel.CHANNELS,
        (GRADIENT_SAMPLE_COUNT, rel.TIME_STEPS),
        generator=action_generator,
        dtype=torch.int64,
    )
    connection_permutation = rel.sattolo_derangement(
        GRADIENT_SAMPLE_COUNT, CONNECTION_SHUFFLE_SEED
    )
    target_permutation = rel.sattolo_derangement(
        GRADIENT_SAMPLE_COUNT, TARGET_SHUFFLE_SEED
    )
    target_posterior = _target_posterior(dataset.target)

    true_loss = _task_loss(module, dataset, target_posterior=target_posterior)
    true_gradients, true_vector = _gradient_vector(true_loss, parameters)
    action_loss = _task_loss(
        module,
        dataset,
        tokens=rel.apply_local_action(dataset.tokens, action),
        connection=rel.transform_connection(dataset.connection, action),
        target_posterior=target_posterior,
    )
    _, action_vector = _gradient_vector(action_loss, parameters)
    connection_loss = _task_loss(
        module,
        dataset,
        connection=dataset.connection[connection_permutation],
        target_posterior=target_posterior,
    )
    _, connection_vector = _gradient_vector(connection_loss, parameters)
    shuffled_target_loss = _task_loss(
        module,
        dataset,
        target_posterior=target_posterior[target_permutation],
    )
    _, shuffled_target_vector = _gradient_vector(
        shuffled_target_loss, parameters
    )

    vector_norm = float(torch.linalg.vector_norm(true_vector))
    nonzero_count = int(torch.count_nonzero(true_vector))
    nonzero_fraction = nonzero_count / float(true_vector.numel())
    action_loss_error = float(torch.abs(action_loss - true_loss))
    action_gradient_error = float((action_vector - true_vector).abs().max())
    connection_control = _gradient_comparison(
        connection_vector, true_vector
    )
    target_control = _gradient_comparison(
        shuffled_target_vector, true_vector
    )

    snapshots = [parameter.detach().clone() for parameter in parameters]
    with torch.no_grad():
        scale = DIAGNOSTIC_STEP_RADIUS / max(vector_norm, 1e-12)
        for parameter, gradient in zip(parameters, true_gradients):
            parameter.add_(gradient, alpha=-scale)
    perturbed_loss = _task_loss(
        module, dataset, target_posterior=target_posterior
    )
    loss_decrease = float(true_loss.detach() - perturbed_loss.detach())
    with torch.no_grad():
        for parameter, snapshot in zip(parameters, snapshots):
            parameter.copy_(snapshot)
    state_after = _state_digest(module)

    state_invariance = [
        _state_invariance("random", module, dataset, action),
        _state_invariance(
            "deterministic_perturbation",
            _parameter_perturbation(module),
            dataset,
            action,
        ),
        _state_invariance(
            "analytic_witness",
            construct_analytic_witness(ConnectionInvariantRelationalModule()),
            dataset,
            action,
        ),
    ]
    finite = bool(
        torch.isfinite(true_loss)
        and torch.isfinite(action_loss)
        and torch.isfinite(connection_loss)
        and torch.isfinite(shuffled_target_loss)
        and torch.isfinite(true_vector).all()
        and torch.isfinite(action_vector).all()
        and torch.isfinite(connection_vector).all()
        and torch.isfinite(shuffled_target_vector).all()
        and dataset.saturation_count == 0
    )
    gradient_pass = bool(
        finite
        and vector_norm >= GRADIENT_NORM_MINIMUM
        and nonzero_fraction >= GRADIENT_NONZERO_FRACTION_MINIMUM
        and action_loss_error <= ACTION_LOSS_ERROR_MAXIMUM
        and action_gradient_error <= ACTION_GRADIENT_ERROR_MAXIMUM
        and connection_control["pass"]
        and target_control["pass"]
        and loss_decrease >= DIAGNOSTIC_LOSS_DECREASE_MINIMUM
        and state_before == state_after
        and int(
            (connection_permutation == torch.arange(GRADIENT_SAMPLE_COUNT)).sum()
        )
        == 0
        and int(
            (target_permutation == torch.arange(GRADIENT_SAMPLE_COUNT)).sum()
        )
        == 0
    )
    record = {
        "parameter_seed": GRADIENT_PARAMETER_SEED,
        "dataset_seed": GRADIENT_DATA_SEED,
        "sample_count": GRADIENT_SAMPLE_COUNT,
        "dataset_sha256": rel.dataset_hash(dataset),
        "quantizer_saturation_count": dataset.saturation_count,
        "local_action_seed": LOCAL_ACTION_SEED,
        "connection_shuffle_seed": CONNECTION_SHUFFLE_SEED,
        "target_shuffle_seed": TARGET_SHUFFLE_SEED,
        "connection_shuffle_fixed_points": int(
            (connection_permutation == torch.arange(GRADIENT_SAMPLE_COUNT)).sum()
        ),
        "target_shuffle_fixed_points": int(
            (target_permutation == torch.arange(GRADIENT_SAMPLE_COUNT)).sum()
        ),
        "true_loss": float(true_loss.detach()),
        "action_loss": float(action_loss.detach()),
        "action_loss_absolute_error": action_loss_error,
        "connection_shuffled_loss": float(connection_loss.detach()),
        "target_shuffled_loss": float(shuffled_target_loss.detach()),
        "total_gradient_norm": vector_norm,
        "gradient_tensor_norms": {
            name: float(value.norm())
            for name, value in zip(names, true_gradients)
        },
        "nonzero_gradient_parameter_count": nonzero_count,
        "nonzero_gradient_fraction": nonzero_fraction,
        "action_gradient_maximum_error": action_gradient_error,
        "connection_shuffled_gradient": connection_control,
        "target_shuffled_gradient": target_control,
        "diagnostic_step_radius": DIAGNOSTIC_STEP_RADIUS,
        "diagnostic_perturbed_loss": float(perturbed_loss.detach()),
        "diagnostic_loss_decrease": loss_decrease,
        "state_sha256_before": state_before,
        "state_sha256_after_restore": state_after,
        "state_invariance": state_invariance,
        "structural_invariance_pass": all(
            item["pass"] for item in state_invariance
        ),
        "gradient_route_pass": gradient_pass,
        "finite": finite,
        "pass": bool(
            gradient_pass and all(item["pass"] for item in state_invariance)
        ),
    }
    return record, dataset, module


@torch.no_grad()
def checkpoint_device_lifecycle(
    module: ConnectionInvariantRelationalModule,
    dataset: rel.RelationalDataset,
    *,
    require_cuda: bool = True,
) -> dict[str, Any]:
    sample = slice(0, 128)
    tokens = dataset.tokens[sample]
    calibration = dataset.calibration[sample]
    connection = dataset.connection[sample]
    action_generator = torch.Generator(device="cpu").manual_seed(
        LOCAL_ACTION_SEED + 17
    )
    action = torch.randint(
        0,
        rel.CHANNELS,
        (len(tokens), rel.TIME_STEPS),
        generator=action_generator,
        dtype=torch.int64,
    )
    cpu_output = module(tokens, calibration, connection)
    original_digest = _state_digest(module)
    with tempfile.TemporaryDirectory(prefix="c3-connection-function-class-") as root:
        path = Path(root) / "lifecycle.pt"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "parameter_count": TOTAL_PARAMETER_COUNT,
            "state_dict": module.state_dict(),
        }
        torch.save(payload, path)
        checkpoint_sha256 = _sha256(path)
        checkpoint_size = path.stat().st_size
        restored_payload = torch.load(
            path, map_location="cpu", weights_only=True
        )
        restored = ConnectionInvariantRelationalModule()
        restored.load_state_dict(restored_payload["state_dict"], strict=True)
        restored_output = restored(tokens, calibration, connection)
        cpu_replay_exact = bool(torch.equal(restored_output, cpu_output))
        cpu_state_exact = _state_digest(restored) == original_digest
        schema_exact = bool(
            restored_payload.get("schema_version") == SCHEMA_VERSION
            and restored_payload.get("parameter_count") == TOTAL_PARAMETER_COUNT
        )

        cuda_available = torch.cuda.is_available()
        cuda_record: dict[str, Any] = {
            "available": cuda_available,
            "required": require_cuda,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
        }
        if cuda_available:
            device = torch.device("cuda:0")
            cuda_module = ConnectionInvariantRelationalModule().to(device)
            cuda_module.load_state_dict(
                restored_payload["state_dict"], strict=True
            )
            cuda_tokens = tokens.to(device)
            cuda_calibration = calibration.to(device)
            cuda_connection = connection.to(device)
            cuda_output = cuda_module(
                cuda_tokens, cuda_calibration, cuda_connection
            )
            transformed_output = cuda_module(
                rel.apply_local_action(cuda_tokens, action.to(device)),
                cuda_calibration,
                rel.transform_connection(
                    cuda_connection, action.to(device)
                ),
            )
            torch.cuda.synchronize(device)
            cpu_cuda_error = float(
                (cuda_output.cpu() - cpu_output).abs().max()
            )
            cuda_action_error = float(
                (transformed_output - cuda_output).abs().max().cpu()
            )
            cuda_state_exact = _state_digest(cuda_module) == original_digest
            cuda_finite = bool(torch.isfinite(cuda_output).all())
            cuda_pass = bool(
                cuda_finite
                and cpu_cuda_error <= CUDA_CPU_OUTPUT_ERROR_MAXIMUM
                and cuda_action_error <= INVARIANCE_ERROR_MAXIMUM
                and cuda_state_exact
            )
            cuda_record.update(
                {
                    "device": torch.cuda.get_device_name(0),
                    "cpu_cuda_output_maximum_error": cpu_cuda_error,
                    "local_action_output_maximum_error": cuda_action_error,
                    "state_digest_exact": cuda_state_exact,
                    "finite": cuda_finite,
                    "pass": cuda_pass,
                }
            )
        else:
            cuda_record["pass"] = not require_cuda
    cpu_pass = bool(cpu_replay_exact and cpu_state_exact and schema_exact)
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size": checkpoint_size,
        "weights_only_load": True,
        "schema_and_parameter_count_exact": schema_exact,
        "cpu_state_digest_exact": cpu_state_exact,
        "cpu_output_tensor_exact": cpu_replay_exact,
        "cpu_pass": cpu_pass,
        "cuda": cuda_record,
        "pass": bool(cpu_pass and cuda_record["pass"]),
    }


def classify(
    *,
    source_contract: bool,
    valid: bool,
    witness_pass: bool,
    invariance_pass: bool,
    gradient_pass: bool,
    cpu_pass: bool,
    cuda_pass: bool,
) -> dict[str, Any]:
    if not source_contract or not valid or not invariance_pass or not cpu_pass:
        classification = "invalid_connection_function_class_preflight"
    elif not witness_pass:
        classification = "connection_function_class_missing_analytic_transport"
    elif not gradient_pass:
        classification = "connection_function_class_gradient_route_insufficient"
    elif not cuda_pass:
        classification = "connection_function_class_valid_cuda_lifecycle_pending"
    else:
        classification = CLASSIFICATION_PASS
    return {
        "classification": classification,
        "valid": valid,
        "source_contract_pass": source_contract,
        "analytic_witness_pass": witness_pass,
        "structural_invariance_pass": invariance_pass,
        "gradient_route_pass": gradient_pass,
        "cpu_lifecycle_pass": cpu_pass,
        "cuda_lifecycle_pass": cuda_pass,
        "matched_sensor_readout_campaign_licensed": (
            classification == CLASSIFICATION_PASS
        ),
        "unrestricted_tinyllm_training_licensed": False,
    }


def build_result(*, require_cuda: bool = True) -> dict[str, Any]:
    source_hashes, predecessor = validate_sources()
    source_contract = True
    witness = analyze_witness(predecessor)
    gradient, diagnostic_dataset, random_module = analyze_gradient_and_invariance()
    lifecycle = checkpoint_device_lifecycle(
        random_module, diagnostic_dataset, require_cuda=require_cuda
    )
    valid = bool(
        module_contract(random_module)["pass"]
        and diagnostic_dataset.saturation_count == 0
        and gradient["finite"]
        and _finite(witness)
        and _finite(gradient)
        and _finite(lifecycle)
    )
    aggregates = classify(
        source_contract=source_contract,
        valid=valid,
        witness_pass=witness["pass"],
        invariance_pass=gradient["structural_invariance_pass"],
        gradient_pass=gradient["gradient_route_pass"],
        cpu_pass=lifecycle["cpu_pass"],
        cuda_pass=lifecycle["cuda"]["pass"],
    )
    source_hashes["runner"] = _sha256(Path(__file__))
    record = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed" if aggregates["valid"] else "invalid",
        "evidence_role": EVIDENCE_ROLE,
        "completed_at": _utc_now(),
        "source_hashes": source_hashes,
        "configuration": {
            "encoder_parameter_count": ENCODER_PARAMETER_COUNT,
            "head_parameter_count": HEAD_PARAMETER_COUNT,
            "total_parameter_count": TOTAL_PARAMETER_COUNT,
            "witness_nonzero_parameter_count": WITNESS_NONZERO_PARAMETER_COUNT,
            "gradient_parameter_seed": GRADIENT_PARAMETER_SEED,
            "gradient_data_seed": GRADIENT_DATA_SEED,
            "local_action_seed": LOCAL_ACTION_SEED,
            "connection_shuffle_seed": CONNECTION_SHUFFLE_SEED,
            "target_shuffle_seed": TARGET_SHUFFLE_SEED,
            "perturbation_seed": PERTURBATION_SEED,
            "gradient_sample_count": GRADIENT_SAMPLE_COUNT,
            "character_error_maximum": CHARACTER_ERROR_MAXIMUM,
            "invariance_error_maximum": INVARIANCE_ERROR_MAXIMUM,
            "action_loss_error_maximum": ACTION_LOSS_ERROR_MAXIMUM,
            "action_gradient_error_maximum": ACTION_GRADIENT_ERROR_MAXIMUM,
            "gradient_norm_minimum": GRADIENT_NORM_MINIMUM,
            "gradient_nonzero_fraction_minimum": (
                GRADIENT_NONZERO_FRACTION_MINIMUM
            ),
            "control_gradient_cosine_maximum": (
                CONTROL_GRADIENT_COSINE_MAXIMUM
            ),
            "control_relative_gradient_difference_minimum": (
                CONTROL_RELATIVE_GRADIENT_DIFFERENCE_MINIMUM
            ),
            "diagnostic_step_radius": DIAGNOSTIC_STEP_RADIUS,
            "diagnostic_loss_decrease_minimum": (
                DIAGNOSTIC_LOSS_DECREASE_MINIMUM
            ),
            "cuda_cpu_output_error_maximum": CUDA_CPU_OUTPUT_ERROR_MAXIMUM,
            "positive_gates": POSITIVE_GATES,
            "require_cuda": require_cuda,
        },
        "architecture_contract": module_contract(random_module),
        "analytic_witness": witness,
        "gradient_and_invariance": gradient,
        "checkpoint_device_lifecycle": lifecycle,
        "aggregates": aggregates,
        "accounting": {
            "source_examples_reused": len(witness["cells"]) * rel.SAMPLE_COUNT,
            "fresh_diagnostic_examples": GRADIENT_SAMPLE_COUNT,
            "optimizer_steps": 0,
            "trained_parameters": 0,
            "tinyllm_models_instantiated": 0,
            "historical_checkpoints_loaded": 0,
            "temporary_lifecycle_checkpoints_created": 1,
            "temporary_lifecycle_checkpoint_loads": (
                2 if lifecycle["cuda"]["available"] else 1
            ),
            "closed_form_nonzero_witness_parameters": witness[
                "nonzero_parameter_count"
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": lifecycle["cuda"]["available"],
            "cuda_device_count": lifecycle["cuda"]["device_count"],
        },
        "method_boundaries": [
            "The six-nonzero-parameter witness is assigned analytically and is not fitted.",
            "The normalized negative-gradient perturbation is restored and is not an optimizer step.",
            "Three numerical states audit an all-parameter invariance identity established by architecture.",
            "The checkpoint is temporary lifecycle evidence; no historical or trained checkpoint is loaded.",
            "Passing establishes capacity, local gradient access, and lifecycle, not global trainability or utility beyond the fixed analytic solution.",
        ],
    }
    if not _finite(record):
        raise RuntimeError("non-finite connection function-class result")
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PRIMARY_RESULT_PATH)
    parser.add_argument(
        "--allow-no-cuda",
        action="store_true",
        help="Lifecycle-only diagnostic; never use for the registered primary result.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_result(require_cuda=not args.allow_no_cuda)
    _write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "classification": result["aggregates"]["classification"],
                "witness_pass": result["aggregates"]["analytic_witness_pass"],
                "invariance_pass": result["aggregates"][
                    "structural_invariance_pass"
                ],
                "gradient_pass": result["aggregates"]["gradient_route_pass"],
                "cpu_lifecycle_pass": result["aggregates"][
                    "cpu_lifecycle_pass"
                ],
                "cuda_lifecycle_pass": result["aggregates"][
                    "cuda_lifecycle_pass"
                ],
                "campaign_licensed": result["aggregates"][
                    "matched_sensor_readout_campaign_licensed"
                ],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
