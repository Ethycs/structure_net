#!/usr/bin/env python3
"""Audit C3 connection sufficiency, erasure, and known-noise identifiability."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
from typing import Any, Iterable, Mapping, Sequence

import torch

import experiments.structure_net.tinyllm_c3_relational_connection_acquisition as acq
import experiments.structure_net.tinyllm_c3_relational_connection_function_class as fc
import experiments.structure_net.tinyllm_c3_relational_connection_preflight as rel


SCHEMA_VERSION = "nal.tinyllm-c3-connection-observation-identifiability.v1"
HYPOTHESIS_ID = "tinyllm-c3-connection-observation-identifiability-v1"
EVIDENCE_ROLE = "preregistered_no_training_identifiability_and_artifact_audit"
CLASSIFICATION_PASS = (
    "total_holonomy_minimal_known_noise_analytic_no_training_scope"
)
CLASSIFICATION_FAIL = "connection_observation_identifiability_claim_failed"
SOURCE_SEEDS = acq.SEEDS
REGIMES = acq.REGIMES
FRESH_DATA_SEED_BASES = {
    "composition": 1_173_107,
    "extrapolation": 1_175_107,
}
NOISE_PROBABILITIES = (1e-5, 1e-4, 1e-3, 1e-2, 5e-2)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-connection-observation-identifiability-"
    "preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "e2c3a75b50403bfcdf86d89db8b3fdb144ac6b324a89221de326023ec003b938"
)
SOURCE_CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_c3_relational_connection_acquisition/"
    "20260811_preregistered/campaign_results.json"
)
SOURCE_CAMPAIGN_SHA256 = (
    "b4b6e94392a866f7a94188d18935db6602fbc83b936e14324b1060e56ce61a4a"
)
PINNED_DEPENDENCY_SHA256 = {
    "acquisition_runner": (
        "cf425970a3424a32e410492ea79d7d17fd579a83cea78bacea9b8a58674116f0"
    ),
    "function_class_runner": (
        "7010794c3be5fda05a035e5a0b4a178aacd40c934dc1f5f7b36eb2eb03ea96b1"
    ),
    "relational_runner": (
        "2ab21b7885fa93c49427973f67e5c57913b7f13c154ad2e970ef9636b2b90214"
    ),
}
PRIMARY_OUTPUT = Path(
    "data/experiments/tinyllm_c3_connection_observation_identifiability/"
    "20260811_preregistered/result.json"
)
TOTAL_EQUIVALENCE_MAXIMUM = 1e-6
ERASURE_TARGET_SEPARATION_MINIMUM = 1.49
ENUMERATION_ERROR_MAXIMUM = 1e-12


@dataclass(frozen=True)
class AuditConfig:
    source_seeds: tuple[int, ...] = SOURCE_SEEDS
    evaluation_samples: int = 1_024
    noise_probabilities: tuple[float, ...] = NOISE_PROBABILITIES
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.source_seeds or len(set(self.source_seeds)) != len(
            self.source_seeds
        ):
            raise ValueError("source seeds must be non-empty and unique")
        if self.evaluation_samples < 16:
            raise ValueError("evaluation_samples must be at least 16")
        if any(not 0.0 < value < 2.0 / 3.0 for value in self.noise_probabilities):
            raise ValueError("noise probabilities must lie strictly between 0 and 2/3")
        if not self.allow_underpowered and (
            self.source_seeds != SOURCE_SEEDS
            or self.evaluation_samples != 1_024
            or self.noise_probabilities != NOISE_PROBABILITIES
        ):
            raise ValueError("primary observation-identifiability protocol is frozen")


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


def _manifest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(f"{_sha256(path)}  {path}\n".encode("utf-8"))
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


def source_hashes() -> dict[str, str]:
    return {
        "runner": _sha256(Path(__file__)),
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "source_campaign": _sha256(SOURCE_CAMPAIGN_PATH),
        "acquisition_runner": _sha256(Path(acq.__file__)),
        "function_class_runner": _sha256(Path(fc.__file__)),
        "relational_runner": _sha256(Path(rel.__file__)),
    }


def _source_result_path(seed: int) -> Path:
    return SOURCE_CAMPAIGN_PATH.parent / "runs" / f"seed_{seed}" / "result.json"


def _all_source_checkpoint_paths() -> list[Path]:
    return [
        SOURCE_CAMPAIGN_PATH.parent
        / "runs"
        / f"seed_{seed}"
        / f"{arm}_{point}.pt"
        for seed in SOURCE_SEEDS
        for arm in acq.ARMS
        for point in ("midpoint", "final")
    ]


def validate_sources() -> tuple[dict[str, str], dict[str, Any], dict[int, dict[str, Any]]]:
    hashes = source_hashes()
    if hashes["preregistration"] != PREREGISTRATION_SHA256:
        raise RuntimeError("observation-identifiability preregistration changed")
    if hashes["source_campaign"] != SOURCE_CAMPAIGN_SHA256:
        raise RuntimeError("source acquisition campaign changed")
    for name, expected in PINNED_DEPENDENCY_SHA256.items():
        if hashes[name] != expected:
            raise RuntimeError(f"observation-identifiability dependency changed: {name}")

    campaign = json.loads(SOURCE_CAMPAIGN_PATH.read_text(encoding="utf-8"))
    if (
        campaign.get("status") != "completed"
        or campaign.get("aggregates", {}).get("classification")
        != "exact_function_class_but_population_acquisition_unreliable"
        or campaign.get("aggregates", {}).get("primary_hypothesis_pass") is not False
    ):
        raise RuntimeError("source campaign is not the locked negative acquisition")

    result_paths = [_source_result_path(seed) for seed in SOURCE_SEEDS]
    checkpoint_paths = _all_source_checkpoint_paths()
    if _manifest(result_paths) != campaign.get("result_manifest_sha256"):
        raise RuntimeError("source result manifest changed")
    if _manifest(checkpoint_paths) != campaign.get("checkpoint_manifest_sha256"):
        raise RuntimeError("source checkpoint manifest changed")

    records: dict[int, dict[str, Any]] = {}
    for seed, path in zip(SOURCE_SEEDS, result_paths, strict=True):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("status") != "completed"
            or int(record.get("seed", -1)) != seed
            or record.get("gates", {}).get("validity") is not True
        ):
            raise RuntimeError(f"invalid source record for seed {seed}")
        for arm in acq.ARMS:
            for point in ("midpoint", "final"):
                metadata = record["artifacts"]["checkpoints"][arm][point]
                checkpoint_path = Path(metadata["path"])
                if _sha256(checkpoint_path) != metadata["sha256"]:
                    raise RuntimeError(f"source checkpoint hash changed: {checkpoint_path}")
        records[seed] = record
    return hashes, campaign, records


def canonical_total_connection(connection: torch.Tensor) -> torch.Tensor:
    if connection.ndim != 2 or connection.shape[1] != rel.TIME_STEPS - 1:
        raise ValueError("connection must contain seven C3 edges per example")
    result = torch.zeros_like(connection)
    result[:, -1] = connection.sum(dim=1) % rel.CHANNELS
    return result


def _load_true_module(seed: int, record: Mapping[str, Any]) -> tuple[fc.ConnectionInvariantRelationalModule, dict[str, str]]:
    metadata = record["artifacts"]["checkpoints"]["learned_true"]["final"]
    path = Path(metadata["path"])
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "schema_version": acq.SCHEMA_VERSION,
        "hypothesis_id": acq.HYPOTHESIS_ID,
        "arm": "learned_true",
        "seed": seed,
        "step": 2_400,
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise RuntimeError(f"learned-true checkpoint provenance mismatch: {path}")
    module = fc.ConnectionInvariantRelationalModule()
    module.load_state_dict(payload["module_state"], strict=True)
    module.eval()
    return module, {"path": str(path), "sha256": metadata["sha256"]}


def _fresh_dataset(config: AuditConfig, seed: int, regime: str) -> rel.RelationalDataset:
    return acq.generate_dataset(
        regime,
        dataset_seed=FRESH_DATA_SEED_BASES[regime] + seed,
        sample_count=config.evaluation_samples,
    )


def _target_bin_coverage(target: torch.Tensor) -> int:
    return int(
        torch.unique(
            fc.joint.interval_posterior_unclipped(target.double(), 16).argmax(dim=-1)
        ).numel()
    )


@torch.no_grad()
def total_holonomy_audit(
    config: AuditConfig, records: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    analytic = fc.construct_analytic_witness(fc.ConnectionInvariantRelationalModule())
    cells = []
    for seed in config.source_seeds:
        learned, checkpoint = _load_true_module(seed, records[seed])
        for regime in REGIMES:
            dataset = _fresh_dataset(config, seed, regime)
            total = canonical_total_connection(dataset.connection)
            analytic_full = acq.predict(analytic, dataset, dataset.connection)
            analytic_total = acq.predict(analytic, dataset, total)
            learned_full = acq.predict(learned, dataset, dataset.connection)
            learned_total = acq.predict(learned, dataset, total)
            analytic_metrics = acq.evaluate_module(
                analytic, dataset, dataset.connection, acq.ANALYTIC_GATES
            )
            learned_metrics = acq.evaluate_module(
                learned, dataset, dataset.connection, acq.LEARNED_GATES
            )
            cells.append(
                {
                    "source_seed": seed,
                    "regime": regime,
                    "fresh_dataset_seed": FRESH_DATA_SEED_BASES[regime] + seed,
                    "fresh_dataset_sha256": acq.dataset_digest(dataset),
                    "saturation_count": dataset.saturation_count,
                    "target_bin_coverage": _target_bin_coverage(dataset.target),
                    "source_checkpoint": checkpoint,
                    "analytic_full_to_total_maximum_error": float(
                        (analytic_full - analytic_total).abs().max()
                    ),
                    "learned_full_to_total_maximum_error": float(
                        (learned_full - learned_total).abs().max()
                    ),
                    "analytic": analytic_metrics,
                    "learned": learned_metrics,
                    "integrity_pass": bool(
                        dataset.saturation_count == 0
                        and _target_bin_coverage(dataset.target) == 16
                    ),
                }
            )
    maximum_error = max(
        max(
            cell["analytic_full_to_total_maximum_error"],
            cell["learned_full_to_total_maximum_error"],
        )
        for cell in cells
    )
    return {
        "cells": cells,
        "cell_count": len(cells),
        "analytic_clean_pass_count": sum(
            int(cell["analytic"]["endpoint_pass"]) for cell in cells
        ),
        "fresh_integrity_pass_count": sum(int(cell["integrity_pass"]) for cell in cells),
        "maximum_full_to_total_prediction_error": maximum_error,
        "pass": bool(
            len(cells) == len(config.source_seeds) * len(REGIMES)
            and maximum_error <= TOTAL_EQUIVALENCE_MAXIMUM
            and all(cell["analytic"]["endpoint_pass"] for cell in cells)
            and all(cell["integrity_pass"] for cell in cells)
        ),
    }


def _observed_tokens(
    phase: torch.Tensor,
    gauge: torch.Tensor,
    amplitude: torch.Tensor,
    offset: torch.Tensor,
    drift: torch.Tensor,
) -> torch.Tensor:
    continuous = rel._continuous_observation(phase, amplitude, offset, drift)
    return rel.apply_local_action(rel.source.quantize(continuous), gauge)


def erasure_witness(edge_index: int) -> dict[str, Any]:
    if not 0 <= edge_index < rel.TIME_STEPS - 1:
        raise ValueError("edge_index is outside the seven-edge path")
    phase_left = torch.zeros((1, rel.TIME_STEPS), dtype=torch.float64)
    gauge_left = torch.zeros((1, rel.TIME_STEPS), dtype=torch.int64)
    phase_right = phase_left.clone()
    gauge_right = gauge_left.clone()
    phase_right[:, edge_index + 1 :] += 2.0 * math.pi / rel.CHANNELS
    gauge_right[:, edge_index + 1 :] = 1
    amplitude = torch.ones(1, dtype=torch.float64)
    offset = torch.zeros(1, dtype=torch.float64)
    drift = torch.zeros(1, dtype=torch.float64)
    calibration_left = torch.stack((amplitude, offset, drift), dim=-1)
    calibration_right = calibration_left.clone()
    tokens_left = _observed_tokens(
        phase_left, gauge_left, amplitude, offset, drift
    )
    tokens_right = _observed_tokens(
        phase_right, gauge_right, amplitude, offset, drift
    )
    connection_left = rel.edge_connection(gauge_left)
    connection_right = rel.edge_connection(gauge_right)
    visible = [index for index in range(rel.TIME_STEPS - 1) if index != edge_index]
    target_left = torch.cos(phase_left[:, -1] - phase_left[:, 0])
    target_right = torch.cos(phase_right[:, -1] - phase_right[:, 0])
    changed = torch.nonzero(
        connection_left[0] != connection_right[0], as_tuple=False
    ).flatten()
    target_separation = float((target_left - target_right).abs().item())
    token_maximum_error = int(
        (tokens_left.to(torch.int64) - tokens_right.to(torch.int64)).abs().max()
    )
    calibration_maximum_error = float(
        (calibration_left - calibration_right).abs().max()
    )
    visible_connection_maximum_error = int(
        (
            connection_left[:, visible].to(torch.int64)
            - connection_right[:, visible].to(torch.int64)
        )
        .abs()
        .max()
    )
    passed = bool(
        token_maximum_error == 0
        and calibration_maximum_error == 0.0
        and visible_connection_maximum_error == 0
        and changed.tolist() == [edge_index]
        and target_separation >= ERASURE_TARGET_SEPARATION_MINIMUM
    )
    return {
        "erased_edge_index": edge_index,
        "token_maximum_integer_error": token_maximum_error,
        "calibration_maximum_error": calibration_maximum_error,
        "visible_connection_maximum_integer_error": visible_connection_maximum_error,
        "changed_connection_indices": changed.tolist(),
        "left_target": float(target_left.item()),
        "right_target": float(target_right.item()),
        "absolute_target_separation": target_separation,
        "pass": passed,
    }


def erasure_audit() -> dict[str, Any]:
    witnesses = [erasure_witness(index) for index in range(rel.TIME_STEPS - 1)]
    return {
        "witnesses": witnesses,
        "pass_count": sum(int(item["pass"]) for item in witnesses),
        "pass": all(item["pass"] for item in witnesses),
    }


def error_patterns() -> torch.Tensor:
    return torch.tensor(
        list(itertools.product((0, 1, 2), repeat=rel.TIME_STEPS - 1)),
        dtype=torch.int64,
    )


def noise_attenuation(probability: float) -> float:
    return (1.0 - 1.5 * probability) ** (rel.TIME_STEPS - 1)


def enumerate_noise_law(probability: float) -> dict[str, Any]:
    patterns = error_patterns()
    probabilities = torch.tensor(
        [1.0 - probability, probability / 2.0, probability / 2.0],
        dtype=torch.float64,
    )
    weights = probabilities[patterns].prod(dim=1)
    total = patterns.sum(dim=1) % rel.CHANNELS
    rotations = 2.0 * math.pi * total.to(torch.float64) / rel.CHANNELS
    coefficient_real = float((weights * torch.cos(rotations)).sum())
    coefficient_imaginary = float((weights * torch.sin(rotations)).sum())
    formula = noise_attenuation(probability)
    angles = torch.linspace(-math.pi, math.pi, 513, dtype=torch.float64)
    enumerated = (
        weights[:, None] * torch.cos(angles[None, :] - rotations[:, None])
    ).sum(dim=0)
    closed_form = formula * torch.cos(angles)
    maximum_prediction_error = float((enumerated - closed_form).abs().max())
    coefficient_error = max(
        abs(coefficient_real - formula), abs(coefficient_imaginary)
    )
    bayes_rmse = math.sqrt((1.0 - formula**2) / 2.0)
    return {
        "per_edge_error_probability": probability,
        "pattern_count": int(patterns.shape[0]),
        "probability_mass": float(weights.sum()),
        "enumerated_character_coefficient_real": coefficient_real,
        "enumerated_character_coefficient_imaginary": coefficient_imaginary,
        "closed_form_attenuation": formula,
        "coefficient_maximum_error": coefficient_error,
        "conditional_mean_maximum_error": maximum_prediction_error,
        "ideal_bayes_correlation": abs(formula),
        "ideal_bayes_rmse": bayes_rmse,
        "correlation_gate_pass": abs(formula)
        >= acq.LEARNED_GATES["scalar_correlation_minimum"],
        "rmse_gate_pass": bayes_rmse
        <= acq.LEARNED_GATES["scalar_rmse_maximum"],
        "pass": bool(
            patterns.shape[0] == 3 ** (rel.TIME_STEPS - 1)
            and abs(float(weights.sum()) - 1.0) <= ENUMERATION_ERROR_MAXIMUM
            and coefficient_error <= ENUMERATION_ERROR_MAXIMUM
            and maximum_prediction_error <= ENUMERATION_ERROR_MAXIMUM
        ),
    }


def gate_noise_tolerances() -> dict[str, float]:
    correlation = acq.LEARNED_GATES["scalar_correlation_minimum"]
    rmse = acq.LEARNED_GATES["scalar_rmse_maximum"]
    edges = rel.TIME_STEPS - 1
    correlation_probability = (1.0 - correlation ** (1.0 / edges)) / 1.5
    required_attenuation = math.sqrt(1.0 - 2.0 * rmse**2)
    rmse_probability = (1.0 - required_attenuation ** (1.0 / edges)) / 1.5
    return {
        "correlation_gate_maximum_per_edge_error_probability": correlation_probability,
        "rmse_gate_maximum_per_edge_error_probability": rmse_probability,
        "joint_scalar_gate_maximum_per_edge_error_probability": min(
            correlation_probability, rmse_probability
        ),
    }


def noise_audit(config: AuditConfig) -> dict[str, Any]:
    cells = [enumerate_noise_law(value) for value in config.noise_probabilities]
    consequence_cells = [
        item for item in cells if item["per_edge_error_probability"] >= 1e-4
    ]
    return {
        "cells": cells,
        "gate_noise_tolerances": gate_noise_tolerances(),
        "enumeration_pass": all(item["pass"] for item in cells),
        "current_gate_consequence_pass": bool(
            consequence_cells
            and all(
                not (
                    item["correlation_gate_pass"] and item["rmse_gate_pass"]
                )
                for item in consequence_cells
            )
        ),
    }


def run_audit(config: AuditConfig = AuditConfig()) -> dict[str, Any]:
    hashes, campaign, records = validate_sources()
    total = total_holonomy_audit(config, records)
    erasure = erasure_audit()
    noise = noise_audit(config)
    integrity = bool(
        campaign["result_manifest_sha256"] == _manifest(
            [_source_result_path(seed) for seed in SOURCE_SEEDS]
        )
        and campaign["checkpoint_manifest_sha256"]
        == _manifest(_all_source_checkpoint_paths())
        and total["fresh_integrity_pass_count"] == len(total["cells"])
    )
    gates = {
        "total_holonomy_sufficiency": total["pass"],
        "single_edge_erasure_nonidentifiability": erasure["pass"],
        "known_noise_enumeration": noise["enumeration_pass"],
        "current_gate_noise_consequence": noise["current_gate_consequence_pass"],
        "source_and_data_integrity": integrity,
        "zero_optimizer_steps": True,
        "zero_tinyllm_models": True,
    }
    primary_pass = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed" if integrity else "invalid",
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
        "source_campaign": {
            "path": str(SOURCE_CAMPAIGN_PATH),
            "sha256": hashes["source_campaign"],
            "classification": campaign["aggregates"]["classification"],
            "primary_hypothesis_pass": campaign["aggregates"][
                "primary_hypothesis_pass"
            ],
        },
        "total_holonomy": total,
        "single_edge_erasure": erasure,
        "known_symmetric_noise": noise,
        "gates": gates,
        "aggregates": {
            "primary_hypothesis_pass": primary_pass,
            "classification": CLASSIFICATION_PASS if primary_pass else CLASSIFICATION_FAIL,
            "missing_or_partial_connection_training_licensed": False,
            "known_symmetric_noise_training_licensed": False,
            "unrestricted_tinyllm_training_licensed": False,
            "unknown_observation_dependent_uncertainty_scope_open": primary_pass,
        },
        "accounting": {
            "source_models_loaded": len(config.source_seeds),
            "fresh_evaluation_cells": len(total["cells"]),
            "optimizer_steps": 0,
            "learned_probe_fits": 0,
            "tinyllm_models_instantiated": 0,
        },
        "method_boundaries": [
            "The exact erasure result relies on independent interior phases in the current generator.",
            "The noise ceiling assumes independent symmetric additive C3 edge errors with known probability.",
            "Observation-dependent, correlated, or unknown error laws are not tested.",
            "This audit cannot alter the failed source acquisition primary gate.",
        ],
    }
    if not _finite(result):
        raise RuntimeError("non-finite observation-identifiability result")
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
