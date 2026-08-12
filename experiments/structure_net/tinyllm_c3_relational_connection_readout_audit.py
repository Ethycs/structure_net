#!/usr/bin/env python3
"""Audit frozen readout capacity after the C3 connection acquisition result."""

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

import experiments.structure_net.tinyllm_c3_relational_connection_acquisition as study


SCHEMA_VERSION = "nal.tinyllm-c3-relational-connection-readout-audit.v1"
HYPOTHESIS_ID = "tinyllm-c3-relational-connection-readout-audit-v1"
EVIDENCE_ROLE = "post_outcome_corrective_artifact_only_readout_audit"
REGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-relational-connection-readout-audit-registration.md"
)
REGISTRATION_SHA256 = (
    "17d76393374a7898943fe9044ea8019281b2bceab693a372c2af6f14c458cf9d"
)
PRIMARY_RUNNER_SHA256 = (
    "cf425970a3424a32e410492ea79d7d17fd579a83cea78bacea9b8a58674116f0"
)
CAMPAIGN_PATH = (
    study.PRIMARY_OUTPUT / "campaign_results.json"
)
CAMPAIGN_SHA256 = (
    "b4b6e94392a866f7a94188d18935db6602fbc83b936e14324b1060e56ce61a4a"
)
PRIMARY_RESULT_SHA256 = {
    1453: "aa30cd1e3e657819536f5bcdfe8db6d74788e919eeb837055011f29bec269205",
    1471: "abae3326b3fa43a82971ab2b54eab5a0de9ef0b7b2eb774e1e7c5bcd06ea379d",
    1483: "82d864c151823943ed2632e56938f377d17e8005fe3998d810d420871b832ec8",
    1531: "d4226a86fd99cb04a6fc7ca66e2948ad4c86898e43a71a5295a511af1a903f36",
    1543: "05a1ffd0891543488ea0fd967e3bf47145d76331297092187e20f94c806e9d19",
}
PRIMARY_CLASSIFICATION = "exact_function_class_but_population_acquisition_unreliable"
OUTPUT_PATH = Path(
    "data/experiments/tinyllm_c3_relational_connection_readout_audit/"
    "20260811_artifact_audit/result.json"
)


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


def validate_sources() -> tuple[dict[str, str], dict[str, Any], dict[int, dict[str, Any]]]:
    hashes = {
        "registration": _sha256(REGISTRATION_PATH),
        "primary_runner": _sha256(Path(study.__file__)),
        "campaign": _sha256(CAMPAIGN_PATH),
    }
    expected = {
        "registration": REGISTRATION_SHA256,
        "primary_runner": PRIMARY_RUNNER_SHA256,
        "campaign": CAMPAIGN_SHA256,
    }
    if hashes != expected:
        raise RuntimeError(f"readout audit sources changed: {hashes}")
    campaign = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    aggregates = campaign.get("aggregates", {})
    if (
        campaign.get("status") != "completed"
        or aggregates.get("classification") != PRIMARY_CLASSIFICATION
        or aggregates.get("primary_hypothesis_pass") is not False
        or aggregates.get("unrestricted_tinyllm_training_licensed") is not False
    ):
        raise RuntimeError("primary connection acquisition result changed")
    details = {}
    for seed, expected_hash in PRIMARY_RESULT_SHA256.items():
        path = study._cell_dir(study.PRIMARY_OUTPUT, seed) / "result.json"
        observed = _sha256(path)
        if observed != expected_hash:
            raise RuntimeError(f"primary seed result changed: {seed}/{observed}")
        detail = json.loads(path.read_text(encoding="utf-8"))
        if detail.get("gates", {}).get("validity") is not True:
            raise RuntimeError(f"primary seed is not valid: {seed}")
        details[seed] = detail
        hashes[f"seed_{seed}_result"] = observed
    return hashes, campaign, details


@torch.no_grad()
def neutral_features(
    module: study.fc.ConnectionInvariantRelationalModule,
    dataset: study.rel.RelationalDataset,
    connection: torch.Tensor,
    batch_size: int = 512,
) -> torch.Tensor:
    module.eval()
    values = []
    for start in range(0, len(dataset.target), batch_size):
        stop = start + batch_size
        values.append(
            module.neutral(
                dataset.tokens[start:stop],
                dataset.calibration[start:stop],
                connection[start:stop],
            ).double().cpu()
        )
    return torch.cat(values)


def fit_linear(features: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    target = target.cpu()
    design = torch.cat(
        (features.double(), torch.ones(len(features), 1, dtype=torch.float64)),
        dim=1,
    )
    solution = torch.linalg.lstsq(design, target.double()[:, None]).solution[:, 0]
    singular = torch.linalg.svdvals(design)
    return {
        "coefficients": solution.tolist(),
        "condition_number": float(singular.max() / singular.min().clamp_min(1e-15)),
        "training_rank": int(torch.linalg.matrix_rank(design)),
        "training_prediction": design @ solution,
    }


def apply_linear(features: torch.Tensor, coefficients: Sequence[float]) -> torch.Tensor:
    design = torch.cat(
        (features.double(), torch.ones(len(features), 1, dtype=torch.float64)),
        dim=1,
    )
    return design @ torch.tensor(coefficients, dtype=torch.float64)


def metric_record(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    target = target.cpu()
    scalar = study.rel._scalar_metrics(prediction, target)
    task = study.rel._task_metrics(prediction, target)
    return {
        "scalar": scalar,
        "task": task,
        "endpoint_pass": study._endpoint_pass(
            scalar, task, study.LEARNED_GATES
        ),
    }


def load_module(
    detail: Mapping[str, Any], arm: str, seed: int, device: torch.device
) -> tuple[study.fc.ConnectionInvariantRelationalModule, dict[str, str]]:
    artifact = detail["artifacts"]["checkpoints"][arm]["final"]
    path = Path(artifact["path"])
    observed = _sha256(path)
    if observed != artifact["sha256"]:
        raise RuntimeError(f"checkpoint changed: {arm}/{seed}")
    payload = torch.load(path, map_location=device, weights_only=True)
    if (
        payload.get("arm") != arm
        or payload.get("seed") != seed
        or payload.get("step") != study.AcquisitionConfig().training_steps
        or payload.get("scientific_fingerprint")
        != detail["scientific_fingerprint"]
    ):
        raise RuntimeError(f"checkpoint provenance mismatch: {arm}/{seed}")
    module = study._initial_module(seed, device)
    module.load_state_dict(payload["module_state"], strict=True)
    if study.fc._state_digest(module) != detail["arms"][arm]["final_state_sha256"]:
        raise RuntimeError(f"checkpoint state mismatch: {arm}/{seed}")
    return module, {"path": str(path), "sha256": observed}


def analyze_arm(
    *,
    arm: str,
    seed: int,
    detail: Mapping[str, Any],
    train: study.rel.RelationalDataset,
    connection_permutation: torch.Tensor,
    target_permutation: torch.Tensor,
    evaluations: Mapping[str, study.rel.RelationalDataset],
    evaluation_permutations: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    module, checkpoint = load_module(detail, arm, seed, device)
    train_connection, _registered_training_target = study._arm_training_material(
        arm, train, connection_permutation, target_permutation
    )
    train_scalar = study.predict(module, train, train_connection).double()
    train_neutral = neutral_features(module, train, train_connection)
    affine_fit = fit_linear(train_scalar[:, None], train.target)
    neutral_fit = fit_linear(train_neutral, train.target)
    fits = {
        "scalar_affine": {
            key: value
            for key, value in affine_fit.items()
            if key != "training_prediction"
        },
        "neutral_linear": {
            key: value
            for key, value in neutral_fit.items()
            if key != "training_prediction"
        },
    }
    fits["scalar_affine"]["training"] = metric_record(
        affine_fit["training_prediction"], train.target
    )
    fits["neutral_linear"]["training"] = metric_record(
        neutral_fit["training_prediction"], train.target
    )
    exact_replay = {}
    for regime in study.REGIMES:
        supplied = study._arm_evaluation_connection(
            arm, evaluations[regime], evaluation_permutations[regime]
        )
        scalar = study.predict(module, evaluations[regime], supplied).double()
        observed_hash = study._tensor_digest(scalar.float())
        expected_hash = detail["arms"][arm]["regimes"][regime][
            "prediction_sha256"
        ]
        exact_replay[regime] = {
            "prediction_sha256": observed_hash,
            "expected_sha256": expected_hash,
            "pass": observed_hash == expected_hash,
        }
        neutral = neutral_features(module, evaluations[regime], supplied)
        fits["scalar_affine"][regime] = metric_record(
            apply_linear(scalar[:, None], affine_fit["coefficients"]),
            evaluations[regime].target,
        )
        fits["neutral_linear"][regime] = metric_record(
            apply_linear(neutral, neutral_fit["coefficients"]),
            evaluations[regime].target,
        )
    for fit in fits.values():
        fit["joint_pass"] = all(
            fit[regime]["endpoint_pass"] for regime in study.REGIMES
        )
    replay_pass = all(item["pass"] for item in exact_replay.values())
    return {
        "arm": arm,
        "checkpoint": checkpoint,
        "exact_primary_prediction_replay": exact_replay,
        "replay_pass": replay_pass,
        "primary_joint_pass": detail["arms"][arm]["joint_pass"],
        "winding_diagnostic": detail["arms"][arm]["winding_diagnostic"],
        "fits": fits,
        "pass": replay_pass,
    }


def run(output_path: Path = OUTPUT_PATH) -> dict[str, Any]:
    sources, campaign, details = validate_sources()
    config = study.AcquisitionConfig()
    cells = []
    for seed in config.seeds:
        train, _batches, connection_permutation, target_permutation, protocol_hashes = (
            study.protocol_material(config, seed)
        )
        evaluations, evaluation_permutations, evaluation_hashes = (
            study.build_evaluation_material(config, seed)
        )
        detail = details[seed]
        if (
            protocol_hashes != detail["protocol_hashes"]
            or evaluation_hashes != detail["evaluation_hashes"]
        ):
            raise RuntimeError(f"regenerated material changed: {seed}")
        device = torch.device(detail["device"])
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(f"primary CUDA device unavailable: {seed}/{device}")
        expected_name = detail["device_name"]
        if torch.cuda.get_device_name(device) != expected_name:
            raise RuntimeError(f"primary CUDA device changed: {seed}/{device}")
        train = study._to_device(train, device)
        evaluations = {
            regime: study._to_device(evaluations[regime], device)
            for regime in study.REGIMES
        }
        connection_permutation = connection_permutation.to(device)
        target_permutation = target_permutation.to(device)
        evaluation_permutations = {
            regime: evaluation_permutations[regime].to(device)
            for regime in study.REGIMES
        }
        arms = {
            arm: analyze_arm(
                arm=arm,
                seed=seed,
                detail=detail,
                train=train,
                connection_permutation=connection_permutation,
                target_permutation=target_permutation,
                evaluations=evaluations,
                evaluation_permutations=evaluation_permutations,
                device=device,
            )
            for arm in study.ARMS
        }
        cells.append(
            {
                "seed": seed,
                "protocol_hashes": protocol_hashes,
                "evaluation_hashes": evaluation_hashes,
                "arms": arms,
                "pass": all(item["pass"] for item in arms.values()),
            }
        )
        torch.cuda.empty_cache()
    counts = {
        fit: {
            arm: sum(
                bool(cell["arms"][arm]["fits"][fit]["joint_pass"])
                for cell in cells
            )
            for arm in study.ARMS
        }
        for fit in ("scalar_affine", "neutral_linear")
    }
    specificity = {
        fit: all(
            counts[fit][arm] <= config.control_seed_passes_maximum
            for arm in (
                "learned_no_connection",
                "learned_connection_shuffled",
            )
        )
        for fit in counts
    }
    newly_repaired = {
        fit: sum(
            bool(cell["arms"]["learned_true"]["fits"][fit]["joint_pass"])
            and not bool(cell["arms"]["learned_true"]["primary_joint_pass"])
            for cell in cells
        )
        for fit in counts
    }
    persistent_true_failures = {
        fit: [
            int(cell["seed"])
            for cell in cells
            if not cell["arms"]["learned_true"]["fits"][fit]["joint_pass"]
        ]
        for fit in counts
    }
    if not all(cell["pass"] for cell in cells):
        classification = "invalid_post_outcome_readout_audit"
    elif (
        counts["scalar_affine"]["learned_true"] >= config.required_seed_passes
        and specificity["scalar_affine"]
    ):
        classification = (
            "posthoc_public_scale_readout_reaches_four_of_five_"
            "one_wrong_winding_remains"
        )
    elif (
        counts["neutral_linear"]["learned_true"]
        >= config.required_seed_passes
        and specificity["neutral_linear"]
    ):
        classification = "frozen_neutral_carrier_readable_but_joint_head_direction_failed"
    else:
        classification = "frozen_carrier_acquisition_remains_population_unreliable"
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "completed_at": _utc_now(),
        "evidence_role": EVIDENCE_ROLE,
        "source_hashes": sources,
        "primary_campaign": {
            "path": str(CAMPAIGN_PATH),
            "sha256": CAMPAIGN_SHA256,
            "classification": campaign["aggregates"]["classification"],
            "primary_hypothesis_pass": False,
        },
        "configuration": {
            "fit_cohort": "sealed_primary_training_cohort",
            "evaluation_cohorts": list(study.REGIMES),
            "scalar_affine_parameter_count": 2,
            "neutral_linear_parameter_count": 3,
            "ridge": 0.0,
            "optimizer_steps": 0,
            "tinyllm_models_instantiated": 0,
            "primary_rescue_allowed": False,
        },
        "cells": cells,
        "aggregates": {
            "valid": all(cell["pass"] for cell in cells),
            "joint_pass_counts": counts,
            "newly_repaired_true_seed_counts": newly_repaired,
            "persistent_true_failure_seeds": persistent_true_failures,
            "information_removal_control_specificity": specificity,
            "classification": classification,
            "primary_classification_unchanged": True,
            "unrestricted_tinyllm_training_licensed": False,
            "further_optimizer_tuning_licensed": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
    }
    if not _finite(result):
        raise RuntimeError("non-finite readout audit result")
    _write_json(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "classification": result["aggregates"]["classification"],
                "joint_pass_counts": result["aggregates"]["joint_pass_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
