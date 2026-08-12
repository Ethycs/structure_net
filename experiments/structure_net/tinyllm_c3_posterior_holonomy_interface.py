#!/usr/bin/env python3
"""Audit the exact soft posterior interface for uncertain C3 holonomy."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import torch

import experiments.structure_net.tinyllm_c3_connection_observation_identifiability as obs
import experiments.structure_net.tinyllm_c3_relational_connection_function_class as fc
import experiments.structure_net.tinyllm_c3_relational_connection_preflight as rel


SCHEMA_VERSION = "nal.tinyllm-c3-posterior-holonomy-interface.v1"
HYPOTHESIS_ID = "tinyllm-c3-posterior-holonomy-interface-v1"
EVIDENCE_ROLE = "preregistered_deterministic_soft_holonomy_interface_audit"
CLASSIFICATION_PASS = "posterior_holonomy_moment_exact_soft_interface"
CLASSIFICATION_FAIL = "posterior_holonomy_interface_claim_failed"
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-posterior-holonomy-interface-"
    "preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "c95670b166b7ba1368b2241e0a17ea73c84bd07ec51d232b71ee09e0a5321e02"
)
PREDECESSOR_RESULT_PATH = obs.PRIMARY_OUTPUT
PREDECESSOR_RESULT_SHA256 = (
    "23a8989e820d73d1b72c8abaf3f5b4fde0664b854fb03a17ff6df3c5e2d24c7c"
)
PINNED_DEPENDENCY_SHA256 = {
    "observation_runner": (
        "a7ecb704dad4f472d57ee94dbda432a632ee9d5a4657dcbfb49307e5beedb819"
    ),
    "function_class_runner": (
        "7010794c3be5fda05a035e5a0b4a178aacd40c934dc1f5f7b36eb2eb03ea96b1"
    ),
    "relational_runner": (
        "2ab21b7885fa93c49427973f67e5c57913b7f13c154ad2e970ef9636b2b90214"
    ),
}
PRIMARY_OUTPUT = Path(
    "data/experiments/tinyllm_c3_posterior_holonomy_interface/"
    "20260811_preregistered/result.json"
)
SOURCE_SEEDS = obs.SOURCE_SEEDS
REGIMES = obs.REGIMES
REGISTERED_POSTERIORS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    (0.6, 0.3, 0.1),
    (0.1, 0.2, 0.7),
)
ALGEBRA_ERROR_MAXIMUM = 1e-12
FROZEN_REPLAY_ERROR_MAXIMUM = 1e-6


@dataclass(frozen=True)
class InterfaceConfig:
    simplex_denominator: int = 63
    phase_points: int = 2_048
    source_seeds: tuple[int, ...] = SOURCE_SEEDS
    registered_posteriors: tuple[tuple[float, float, float], ...] = (
        REGISTERED_POSTERIORS
    )
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if self.simplex_denominator < 3:
            raise ValueError("simplex denominator must be at least three")
        if self.phase_points < 12 or self.phase_points % 3 == 0:
            raise ValueError("phase_points must be >=12 and not divisible by three")
        if not self.source_seeds or len(set(self.source_seeds)) != len(
            self.source_seeds
        ):
            raise ValueError("source seeds must be non-empty and unique")
        for posterior in self.registered_posteriors:
            if (
                len(posterior) != 3
                or any(value < 0.0 for value in posterior)
                or not math.isclose(sum(posterior), 1.0, abs_tol=1e-12)
            ):
                raise ValueError("registered posteriors must be C3 probabilities")
        if not self.allow_underpowered and (
            self.simplex_denominator != 63
            or self.phase_points != 2_048
            or self.source_seeds != SOURCE_SEEDS
            or self.registered_posteriors != REGISTERED_POSTERIORS
        ):
            raise ValueError("primary posterior-holonomy interface protocol is frozen")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
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


def source_hashes() -> dict[str, str]:
    return {
        "runner": _sha256(Path(__file__)),
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "predecessor_result": _sha256(PREDECESSOR_RESULT_PATH),
        "observation_runner": _sha256(Path(obs.__file__)),
        "function_class_runner": _sha256(Path(fc.__file__)),
        "relational_runner": _sha256(Path(rel.__file__)),
    }


def validate_sources() -> tuple[
    dict[str, str], dict[str, Any], dict[int, dict[str, Any]]
]:
    hashes = source_hashes()
    if hashes["preregistration"] != PREREGISTRATION_SHA256:
        raise RuntimeError("posterior-holonomy preregistration changed")
    if hashes["predecessor_result"] != PREDECESSOR_RESULT_SHA256:
        raise RuntimeError("connection-observation predecessor changed")
    for name, expected in PINNED_DEPENDENCY_SHA256.items():
        if hashes[name] != expected:
            raise RuntimeError(f"posterior-holonomy dependency changed: {name}")
    predecessor = json.loads(PREDECESSOR_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        predecessor.get("status") != "completed"
        or predecessor.get("aggregates", {}).get("classification")
        != obs.CLASSIFICATION_PASS
        or predecessor.get("aggregates", {}).get("primary_hypothesis_pass")
        is not True
        or predecessor.get("accounting", {}).get("optimizer_steps") != 0
    ):
        raise RuntimeError("connection-observation predecessor is not licensed")
    _source_hashes, _campaign, records = obs.validate_sources()
    return hashes, predecessor, records


def character_roots(*, inverse: bool = True) -> torch.Tensor:
    sign = -1.0 if inverse else 1.0
    angles = (
        sign
        * 2.0
        * math.pi
        * torch.arange(rel.CHANNELS, dtype=torch.float64)
        / rel.CHANNELS
    )
    return torch.polar(torch.ones_like(angles), angles).to(torch.complex128)


def posterior_simplex(denominator: int) -> torch.Tensor:
    values = [
        (i / denominator, j / denominator, (denominator - i - j) / denominator)
        for i in range(denominator + 1)
        for j in range(denominator - i + 1)
    ]
    return torch.tensor(values, dtype=torch.float64)


def posterior_moment(posterior: torch.Tensor) -> torch.Tensor:
    if posterior.ndim != 2 or posterior.shape[1] != rel.CHANNELS:
        raise ValueError("posterior must have one probability per C3 error")
    return posterior.to(torch.complex128) @ character_roots(inverse=True)


def inverse_posterior_moment(moment: torch.Tensor) -> torch.Tensor:
    positive = character_roots(inverse=False)
    return torch.stack(
        [
            (1.0 + 2.0 * (moment * positive[index]).real) / rel.CHANNELS
            for index in range(rel.CHANNELS)
        ],
        dim=-1,
    )


def _phase_material(points: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phase = 2.0 * math.pi * torch.arange(points, dtype=torch.float64) / points
    neutral = torch.polar(torch.ones_like(phase), phase).to(torch.complex128)
    candidates = (
        neutral[None, :] * character_roots(inverse=True)[:, None]
    ).real
    return phase, neutral, candidates


def simplex_inverse_audit(config: InterfaceConfig) -> dict[str, Any]:
    posterior = posterior_simplex(config.simplex_denominator)
    moment = posterior_moment(posterior)
    reconstructed = inverse_posterior_moment(moment)
    expected_count = (
        (config.simplex_denominator + 1) * (config.simplex_denominator + 2) // 2
    )
    maximum_probability_error = float((reconstructed - posterior).abs().max())
    maximum_sum_error = float((reconstructed.sum(dim=1) - 1.0).abs().max())
    minimum_probability = float(reconstructed.min())
    barycenter_index = int(
        torch.argmin(
            torch.linalg.vector_norm(
                posterior - torch.full_like(posterior, 1.0 / 3.0), dim=1
            )
        )
    )
    return {
        "simplex_point_count": int(posterior.shape[0]),
        "expected_simplex_point_count": expected_count,
        "maximum_probability_reconstruction_error": maximum_probability_error,
        "maximum_probability_sum_error": maximum_sum_error,
        "minimum_reconstructed_probability": minimum_probability,
        "maximum_moment_magnitude": float(moment.abs().max()),
        "barycenter": {
            "posterior": posterior[barycenter_index].tolist(),
            "moment_magnitude": float(moment[barycenter_index].abs()),
        },
        "pass": bool(
            posterior.shape[0] == expected_count
            and maximum_probability_error <= ALGEBRA_ERROR_MAXIMUM
            and maximum_sum_error <= ALGEBRA_ERROR_MAXIMUM
            and minimum_probability >= -ALGEBRA_ERROR_MAXIMUM
        ),
    }


def factorization_and_risk_audit(config: InterfaceConfig) -> dict[str, Any]:
    posterior = posterior_simplex(config.simplex_denominator)
    moment = posterior_moment(posterior)
    _phase, neutral, candidates = _phase_material(config.phase_points)
    direct_mean = posterior @ candidates
    factorized_mean = (moment[:, None] * neutral[None, :]).real
    maximum_mean_error = float((direct_mean - factorized_mean).abs().max())

    closed_soft = (1.0 - moment.abs().square()) / 2.0
    direct_soft = torch.empty_like(closed_soft)
    direct_hard = torch.empty_like(closed_soft)
    hard_index = posterior.argmax(dim=1)
    for start in range(0, len(posterior), 128):
        stop = min(start + 128, len(posterior))
        q = posterior[start:stop]
        mu = direct_mean[start:stop]
        residual = candidates[None, :, :] - mu[:, None, :]
        direct_soft[start:stop] = (
            q[:, :, None] * residual.square()
        ).sum(dim=1).mean(dim=1)
        selected = candidates[hard_index[start:stop]]
        hard_residual = candidates[None, :, :] - selected[:, None, :]
        direct_hard[start:stop] = (
            q[:, :, None] * hard_residual.square()
        ).sum(dim=1).mean(dim=1)

    roots = character_roots(inverse=True)
    selected_roots = roots[hard_index]
    closed_hard = closed_soft + (selected_roots - moment).abs().square() / 2.0
    maximum_soft_risk_error = float((direct_soft - closed_soft).abs().max())
    maximum_hard_risk_error = float((direct_hard - closed_hard).abs().max())
    regret = direct_hard - direct_soft
    vertex = posterior.max(dim=1).values >= 1.0 - 1e-15
    minimum_regret = float(regret.min())
    minimum_nonvertex_regret = float(regret[~vertex].min())
    maximum_vertex_regret = float(regret[vertex].abs().max())
    return {
        "phase_point_count": config.phase_points,
        "simplex_phase_cell_count": len(posterior) * config.phase_points,
        "maximum_conditional_mean_factorization_error": maximum_mean_error,
        "maximum_soft_risk_formula_error": maximum_soft_risk_error,
        "maximum_hard_risk_formula_error": maximum_hard_risk_error,
        "minimum_hard_minus_soft_regret": minimum_regret,
        "minimum_nonvertex_hard_minus_soft_regret": minimum_nonvertex_regret,
        "maximum_vertex_hard_minus_soft_regret": maximum_vertex_regret,
        "pass": bool(
            maximum_mean_error <= ALGEBRA_ERROR_MAXIMUM
            and maximum_soft_risk_error <= ALGEBRA_ERROR_MAXIMUM
            and maximum_hard_risk_error <= ALGEBRA_ERROR_MAXIMUM
            and minimum_regret >= -ALGEBRA_ERROR_MAXIMUM
            and minimum_nonvertex_regret > 0.0
            and maximum_vertex_regret <= ALGEBRA_ERROR_MAXIMUM
        ),
    }


def covariance_audit(config: InterfaceConfig) -> dict[str, Any]:
    posterior = posterior_simplex(config.simplex_denominator)
    moment = posterior_moment(posterior)
    _phase, neutral, _candidates = _phase_material(config.phase_points)
    roots_inverse = character_roots(inverse=True)
    roots_positive = character_roots(inverse=False)
    rows = []
    for shift in range(rel.CHANNELS):
        shifted_posterior = torch.roll(posterior, shifts=shift, dims=1)
        shifted_moment = posterior_moment(shifted_posterior)
        expected_moment = roots_inverse[shift] * moment
        moment_error = float((shifted_moment - expected_moment).abs().max())
        shifted_neutral = roots_positive[shift] * neutral
        product_error = float(
            (
                shifted_moment[:, None] * shifted_neutral[None, :]
                - moment[:, None] * neutral[None, :]
            )
            .abs()
            .max()
        )
        rows.append(
            {
                "coordinate_shift": shift,
                "moment_covariance_maximum_error": moment_error,
                "corrected_product_maximum_error": product_error,
                "pass": bool(
                    moment_error <= ALGEBRA_ERROR_MAXIMUM
                    and product_error <= ALGEBRA_ERROR_MAXIMUM
                ),
            }
        )
    return {"rows": rows, "pass": all(item["pass"] for item in rows)}


@torch.no_grad()
def frozen_replay_audit(
    config: InterfaceConfig,
    predecessor: Mapping[str, Any],
    records: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_hashes = {
        (int(cell["source_seed"]), cell["regime"]): cell["fresh_dataset_sha256"]
        for cell in predecessor["total_holonomy"]["cells"]
    }
    posteriors = torch.tensor(config.registered_posteriors, dtype=torch.float32)
    roots = character_roots(inverse=True).to(torch.complex64)
    moments = posteriors.to(torch.complex64) @ roots
    cells = []
    for seed in config.source_seeds:
        module, checkpoint = obs._load_true_module(seed, records[seed])
        state_before = fc._state_digest(module)
        for regime in REGIMES:
            dataset = obs._fresh_dataset(obs.AuditConfig(), seed, regime)
            observed_hash = obs.acq.dataset_digest(dataset)
            neutral_components = module.neutral(
                dataset.tokens, dataset.calibration, dataset.connection
            )
            neutral = torch.complex(
                neutral_components[:, 0], neutral_components[:, 1]
            )
            hard_outputs = []
            for error in range(rel.CHANNELS):
                candidate = dataset.connection.clone()
                candidate[:, -1] = (candidate[:, -1] - error) % rel.CHANNELS
                hard_outputs.append(
                    module(dataset.tokens, dataset.calibration, candidate)
                )
            hard = torch.stack(hard_outputs, dim=0)
            averaged = posteriors @ hard
            soft_neutral = moments[:, None] * neutral[None, :]
            soft_components = torch.stack(
                (soft_neutral.real, soft_neutral.imag), dim=-1
            )
            injected = module.head(soft_components).squeeze(-1)
            maximum_error = float((averaged - injected).abs().max())
            cells.append(
                {
                    "source_seed": seed,
                    "regime": regime,
                    "fresh_dataset_sha256": observed_hash,
                    "expected_fresh_dataset_sha256": expected_hashes[(seed, regime)],
                    "source_checkpoint": checkpoint,
                    "posterior_count": len(config.registered_posteriors),
                    "posterior_hard_average_to_soft_injection_maximum_error": (
                        maximum_error
                    ),
                    "dataset_hash_match": observed_hash
                    == expected_hashes[(seed, regime)],
                    "pass": bool(
                        observed_hash == expected_hashes[(seed, regime)]
                        and maximum_error <= FROZEN_REPLAY_ERROR_MAXIMUM
                    ),
                }
            )
        state_after = fc._state_digest(module)
        if state_after != state_before:
            raise RuntimeError(f"frozen source checkpoint changed for seed {seed}")
    return {
        "cells": cells,
        "cell_count": len(cells),
        "pass_count": sum(int(item["pass"]) for item in cells),
        "maximum_replay_error": max(
            item["posterior_hard_average_to_soft_injection_maximum_error"]
            for item in cells
        ),
        "all_source_states_unchanged": True,
        "pass": all(item["pass"] for item in cells),
    }


def run_audit(config: InterfaceConfig = InterfaceConfig()) -> dict[str, Any]:
    hashes, predecessor, records = validate_sources()
    simplex = simplex_inverse_audit(config)
    factorization = factorization_and_risk_audit(config)
    covariance = covariance_audit(config)
    frozen = frozen_replay_audit(config, predecessor, records)
    gates = {
        "simplex_moment_invertibility": simplex["pass"],
        "conditional_mean_and_risk_factorization": factorization["pass"],
        "coordinate_covariance": covariance["pass"],
        "frozen_module_soft_interface": frozen["pass"],
        "source_and_dataset_integrity": bool(
            frozen["pass_count"] == len(config.source_seeds) * len(REGIMES)
        ),
        "zero_optimizer_steps": True,
        "zero_probe_fits": True,
        "zero_tinyllm_models": True,
    }
    primary_pass = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed" if gates["source_and_dataset_integrity"] else "invalid",
        "completed_at": _utc_now(),
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "configuration": json.loads(json.dumps(asdict(config))),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "source_hashes": hashes,
        "implementation_sha256": _json_hash(hashes),
        "predecessor": {
            "path": str(PREDECESSOR_RESULT_PATH),
            "sha256": hashes["predecessor_result"],
            "classification": predecessor["aggregates"]["classification"],
        },
        "simplex_moment_inverse": simplex,
        "conditional_mean_and_risk": factorization,
        "coordinate_covariance": covariance,
        "frozen_module_replay": frozen,
        "gates": gates,
        "aggregates": {
            "primary_hypothesis_pass": primary_pass,
            "classification": CLASSIFICATION_PASS if primary_pass else CLASSIFICATION_FAIL,
            "soft_holonomy_interface_supported": primary_pass,
            "compact_posterior_estimator_function_class_design_licensed": primary_pass,
            "posterior_estimator_training_licensed": False,
            "uncertainty_law_preflight_required": True,
            "unrestricted_tinyllm_training_licensed": False,
        },
        "accounting": {
            "simplex_points": simplex["simplex_point_count"],
            "simplex_phase_cells": factorization["simplex_phase_cell_count"],
            "frozen_source_models_loaded": len(config.source_seeds),
            "frozen_replay_cells": frozen["cell_count"],
            "optimizer_steps": 0,
            "learned_probe_fits": 0,
            "tinyllm_models_instantiated": 0,
        },
        "method_boundaries": [
            "The connection-error factorization conditions on a charged neutral character.",
            "Joint phase/connection uncertainty may require the physical relation posterior moment directly.",
            "First-character probability invertibility is specific to C3.",
            "No uncertainty estimator or TinyLLM was trained or evaluated.",
        ],
    }
    if not _finite(result):
        raise RuntimeError("non-finite posterior-holonomy result")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PRIMARY_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_audit()
    _write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "classification": result["aggregates"]["classification"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
