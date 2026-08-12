#!/usr/bin/env python3
"""Localize TinyLLM architecture-family failures at the frozen scalar interface."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

import experiments.structure_net.tinyllm_calibrated_architecture_replication as architecture
import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
import experiments.structure_net.tinyllm_calibrated_frontend_causal_closure as closure
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-frozen-scalar-interface-decomposition.v1"
HYPOTHESIS_ID = "tinyllm-frozen-scalar-interface-decomposition-v1"
EVIDENCE_ROLE = "registered_post_outcome_frozen_scalar_causal_diagnostic"
SOURCE_SCHEMA_VERSION = architecture.SCHEMA_VERSION
SOURCE_HYPOTHESIS_ID = architecture.HYPOTHESIS_ID
SOURCE_CAMPAIGN_SHA256 = (
    "656d9814a032d1899810e81d398adf935cea3e1116712460e2062da188a0c9e2"
)
SOURCE_RESULT_MANIFEST_SHA256 = (
    "f87740e70062f5fecf5238f00dd00774246e4f3e155dceb87752b099ce4ca80a"
)
SOURCE_ARTIFACT_MANIFEST_SHA256 = (
    "5c08c771d04aae513ad9605d9e4818867ab0a8b0303680337dabbf87dce352e0"
)
SOURCE_RUNNER_SHA256 = (
    "661384de2eac23d95dbc550e0ecf49b14ebfeb01ef47fc1a8164e7e3b2b0ca90"
)
SOURCE_PREREGISTRATION_SHA256 = (
    "36c1a8c35823fda3076b6a73648facd7fc18513c3c969bfa297ca9c0b34c4c77"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-frozen-scalar-interface-decomposition-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "157d5b1c0a41f3d89f79530a4f3a92123b2c1c0da265a9cd87c5945f0882ea40"
)
SOURCE_PREREGISTRATION_PATH = architecture.PREREGISTRATION_PATH
SOURCE_ROOT = Path(
    "data/experiments/tinyllm_calibrated_architecture_replication/"
    "20260810_d6_d10_preregistered"
)
PRESETS = ("d6", "d10")
CONDITIONS = (
    "analytic_calibrated",
    "learned_calibrated_equivariant",
)
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
DATASET_SPECS = {
    "composition": (1_399, "composition"),
    "extrapolation": (2_408, "extrapolation"),
}
EXPECTED_DATASET_HASHES = dict(closure.EXPECTED_DATASET_HASHES)
SOURCE_FAILURES = {
    ("d6", "analytic_calibrated"): (),
    ("d6", "learned_calibrated_equivariant"): (7, 17, 41, 53),
    ("d10", "analytic_calibrated"): (17, 29),
    ("d10", "learned_calibrated_equivariant"): (7, 29, 41, 53),
}


@dataclass(frozen=True)
class ScalarInterfaceConfig:
    source_root: str = str(SOURCE_ROOT)
    presets: tuple[str, ...] = PRESETS
    conditions: tuple[str, ...] = CONDITIONS
    seeds: tuple[int, ...] = SEEDS
    regimes: tuple[str, ...] = REGIMES
    samples_per_regime: int = 1_024
    scalar_grid_minimum: float = -1.0
    scalar_grid_maximum: float = 1.0
    scalar_grid_points: int = 4_097
    batch_size: int = 512
    replay_tolerance: float = 2e-6
    scalar_range_tolerance: float = 1e-6
    negative_control_pass_ceiling: int = 1
    shuffled_control_pass_ceiling: int = 1
    shuffle_shift: int = 137
    device: str = "cuda:0"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.presets or set(self.presets).difference(PRESETS):
            raise ValueError("unknown or empty preset selection")
        if not self.conditions or set(self.conditions).difference(CONDITIONS):
            raise ValueError("unknown or empty condition selection")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if not self.regimes or set(self.regimes).difference(REGIMES):
            raise ValueError("unknown or empty regime selection")
        if self.samples_per_regime < 2 or self.batch_size < 1:
            raise ValueError("sample and batch counts must be positive")
        if self.scalar_grid_points < 3 or self.scalar_grid_points % 2 != 1:
            raise ValueError("scalar grid must contain an odd number of points")
        if self.scalar_grid_minimum >= self.scalar_grid_maximum:
            raise ValueError("scalar grid bounds are reversed")
        if self.shuffle_shift < 1:
            raise ValueError("shuffle shift must be positive")
        if not self.allow_underpowered:
            expected = {
                "source_root": str(SOURCE_ROOT),
                "presets": PRESETS,
                "conditions": CONDITIONS,
                "seeds": SEEDS,
                "regimes": REGIMES,
                "samples_per_regime": 1_024,
                "scalar_grid_minimum": -1.0,
                "scalar_grid_maximum": 1.0,
                "scalar_grid_points": 4_097,
                "batch_size": 512,
                "replay_tolerance": 2e-6,
                "scalar_range_tolerance": 1e-6,
                "negative_control_pass_ceiling": 1,
                "shuffled_control_pass_ceiling": 1,
                "shuffle_shift": 137,
            }
            actual = {
                "source_root": self.source_root,
                "presets": self.presets,
                "conditions": self.conditions,
                "seeds": self.seeds,
                "regimes": self.regimes,
                "samples_per_regime": self.samples_per_regime,
                "scalar_grid_minimum": self.scalar_grid_minimum,
                "scalar_grid_maximum": self.scalar_grid_maximum,
                "scalar_grid_points": self.scalar_grid_points,
                "batch_size": self.batch_size,
                "replay_tolerance": self.replay_tolerance,
                "scalar_range_tolerance": self.scalar_range_tolerance,
                "negative_control_pass_ceiling": self.negative_control_pass_ceiling,
                "shuffled_control_pass_ceiling": self.shuffled_control_pass_ceiling,
                "shuffle_shift": self.shuffle_shift,
            }
            if actual != expected:
                raise ValueError("primary scalar-interface configuration changed")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@lru_cache(maxsize=256)
def _sha256_cached(path: Path, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256_cached(path, stat.st_size, stat.st_mtime_ns)


def _manifest_sha256(paths: Iterable[Path]) -> str:
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


def _json_config(config: ScalarInterfaceConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _implementation_sources() -> dict[str, str]:
    values = {
        "runner": _sha256(Path(__file__)),
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "source_runner": _sha256(Path(architecture.__file__)),
        "source_preregistration": _sha256(SOURCE_PREREGISTRATION_PATH),
        "calibrated_frontend": _sha256(Path(calibrated.__file__)),
        "causal_closure": _sha256(Path(closure.__file__)),
    }
    if values["preregistration"] != PREREGISTRATION_SHA256:
        raise RuntimeError("scalar-interface preregistration changed")
    if values["source_runner"] != SOURCE_RUNNER_SHA256:
        raise RuntimeError("architecture source runner changed")
    if values["source_preregistration"] != SOURCE_PREREGISTRATION_SHA256:
        raise RuntimeError("architecture source preregistration changed")
    return values


def _implementation_digest(sources: Mapping[str, str] | None = None) -> str:
    return _json_hash(dict(sources or _implementation_sources()))


def _finite(value: Any) -> bool:
    return closure._finite(value)


def _source_details(
    config: ScalarInterfaceConfig,
) -> tuple[dict[str, Any], CircleTaskConfig, dict[tuple[str, str, int], dict[str, Any]]]:
    root = Path(config.source_root)
    campaign_path = root / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if (
        _sha256(campaign_path) != SOURCE_CAMPAIGN_SHA256
        or campaign.get("schema_version") != SOURCE_SCHEMA_VERSION
        or campaign.get("hypothesis_id") != SOURCE_HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role")
        != "prospective_architecture_family_replication"
        or campaign.get("aggregates", {}).get("valid") is not True
        or campaign.get("aggregates", {}).get("classification")
        != "structured_closure_not_architecture_stable"
        or campaign.get("summary", {}).get("completed") != 30
        or campaign.get("summary", {}).get("failed") != 0
        or campaign.get("dataset_hashes") != EXPECTED_DATASET_HASHES
    ):
        raise ValueError(f"invalid architecture source campaign {campaign_path}")

    details: dict[tuple[str, str, int], dict[str, Any]] = {}
    result_paths: list[Path] = []
    artifact_paths: list[Path] = []
    for preset in PRESETS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                directory = root / "runs" / preset / condition / f"seed_{seed}"
                result_path = directory / "result.json"
                detail = json.loads(result_path.read_text(encoding="utf-8"))
                artifacts = detail.get("artifacts", {})
                expected_artifacts = {
                    "checkpoint": directory / "model.pt",
                    "frontend_checkpoint": directory / "frontend.pt",
                    "diagnostics": directory / "closure_diagnostics.npz",
                }
                valid_artifacts = True
                for name, artifact_path in expected_artifacts.items():
                    digest_name = {
                        "checkpoint": "checkpoint_sha256",
                        "frontend_checkpoint": "frontend_checkpoint_sha256",
                        "diagnostics": "diagnostics_sha256",
                    }[name]
                    valid_artifacts = bool(
                        valid_artifacts
                        and artifacts.get(name) == str(artifact_path)
                        and artifact_path.is_file()
                        and _sha256(artifact_path) == artifacts.get(digest_name)
                    )
                    artifact_paths.append(artifact_path)
                expected_failed = seed in SOURCE_FAILURES[(preset, condition)]
                if (
                    detail.get("schema_version") != SOURCE_SCHEMA_VERSION
                    or detail.get("hypothesis_id") != SOURCE_HYPOTHESIS_ID
                    or detail.get("status") != "completed"
                    or detail.get("preset") != preset
                    or detail.get("condition") != condition
                    or int(detail.get("seed", -1)) != seed
                    or detail.get("gates", {}).get("validity") is not True
                    or bool(detail.get("gates", {}).get("task_adequacy_pass"))
                    == expected_failed
                    or set(detail.get("gates", {}).get("task_accuracy_floors", {}))
                    != set(REGIMES)
                    or not valid_artifacts
                    or not _finite(detail)
                ):
                    raise ValueError(f"invalid architecture source cell {result_path}")
                detail["_result_path"] = str(result_path)
                detail["_result_sha256"] = _sha256(result_path)
                details[(preset, condition, seed)] = detail
                result_paths.append(result_path)
    if _manifest_sha256(result_paths) != SOURCE_RESULT_MANIFEST_SHA256:
        raise ValueError("structured source-result manifest changed")
    if _manifest_sha256(artifact_paths) != SOURCE_ARTIFACT_MANIFEST_SHA256:
        raise ValueError("structured source-artifact manifest changed")
    return campaign, CircleTaskConfig(**campaign["task_config"]), details


def _datasets(
    task: CircleTaskConfig, config: ScalarInterfaceConfig
) -> dict[str, calibrated.CalibratedDataset]:
    return {
        regime: calibrated.generate_calibrated_dataset(
            task,
            sample_count=config.samples_per_regime,
            seed=DATASET_SPECS[regime][0],
            regime=DATASET_SPECS[regime][1],
            shuffle=True,
        )
        for regime in config.regimes
    }


def _dataset_hashes(
    datasets: Mapping[str, calibrated.CalibratedDataset],
) -> dict[str, str]:
    return {regime: closure._dataset_hash(dataset) for regime, dataset in datasets.items()}


def _load_system(
    detail: Mapping[str, Any],
    task: CircleTaskConfig,
    preset: str,
    condition: str,
    device: torch.device,
) -> calibrated.CalibratedTinyLLM:
    model_path = Path(detail["artifacts"]["checkpoint"])
    frontend_path = Path(detail["artifacts"]["frontend_checkpoint"])
    model = TinyLLMModel.from_checkpoint(model_path, map_location="cpu")
    source_config = calibrated.CalibratedFrontendConfig(
        preset=preset,
        seeds=SEEDS,
        vector_channels=16,
        allow_underpowered=True,
    )
    system = calibrated.CalibratedTinyLLM(model, condition, task, source_config)
    frontend = torch.load(frontend_path, map_location="cpu", weights_only=True)
    if frontend.get("condition") != condition:
        raise ValueError(f"front-end condition changed {frontend_path}")
    if system.scalar_embedding is None:
        raise AssertionError("structured scalar embedding missing")
    system.scalar_embedding.load_state_dict(frontend["scalar_embedding"])
    if condition == "learned_calibrated_equivariant":
        if system.encoder is None or frontend.get("encoder") is None:
            raise AssertionError("learned encoder state missing")
        system.encoder.load_state_dict(frontend["encoder"])
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    system.to(device).eval()
    if (
        calibrated._state_digest(system.model)
        != detail["training"]["final_model_state_sha256"]
        or calibrated._module_digest(system)
        != detail["training"]["final_system_state_sha256"]
    ):
        raise ValueError(f"source state changed for {preset}/{condition}/{detail['seed']}")
    return system


@torch.inference_mode()
def _natural_scalar(
    system: calibrated.CalibratedTinyLLM,
    dataset: calibrated.CalibratedDataset,
    task: CircleTaskConfig,
    config: ScalarInterfaceConfig,
    device: torch.device,
) -> torch.Tensor:
    sensor = calibrated.decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    output = []
    for start in range(0, len(sensor), config.batch_size):
        stop = min(len(sensor), start + config.batch_size)
        output.append(
            system.feature(
                sensor[start:stop].to(device),
                dataset.calibration[start:stop].to(device),
            )
            .reshape(-1)
            .float()
            .cpu()
        )
    return torch.cat(output)


@torch.inference_mode()
def _direct_posteriors(
    system: calibrated.CalibratedTinyLLM,
    dataset: calibrated.CalibratedDataset,
    task: CircleTaskConfig,
    config: ScalarInterfaceConfig,
    device: torch.device,
) -> torch.Tensor:
    sensor = calibrated.decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    output = []
    for start in range(0, len(sensor), config.batch_size):
        stop = min(len(sensor), start + config.batch_size)
        residual = system.forward_cuts(
            dataset.paired.circle.input_ids[start:stop].to(device),
            sensor[start:stop].to(device),
            dataset.calibration[start:stop].to(device),
        )["full"]
        logits = calibrated._task_logits(system.model, residual, answer_ids)
        output.append(torch.softmax(logits, -1).float().cpu())
    return torch.cat(output)


@torch.inference_mode()
def _scalar_posteriors(
    system: calibrated.CalibratedTinyLLM,
    input_ids: torch.Tensor,
    scalar: torch.Tensor,
    task: CircleTaskConfig,
    config: ScalarInterfaceConfig,
    device: torch.device,
) -> torch.Tensor:
    if system.scalar_embedding is None:
        raise AssertionError("structured scalar embedding missing")
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    output = []
    scalar = scalar.reshape(-1, 1)
    for start in range(0, len(input_ids), config.batch_size):
        stop = min(len(input_ids), start + config.batch_size)
        batch_ids = input_ids[start:stop].to(device)
        batch_scalar = scalar[start:stop].to(device)
        bos = system.model.transformer["wte"](batch_ids[:, :1])
        scalar_token = system.scalar_embedding(batch_scalar)[:, None]
        query = system.model.transformer["wte"](batch_ids[:, -1:])
        value = torch.cat((bos, scalar_token, query), dim=1)
        positions = torch.arange(value.shape[1], device=device)
        value = value + system.model.transformer["wpe"](positions)
        for block in system.model.transformer["h"]:
            value = block(value)
        logits = calibrated._task_logits(system.model, value[:, -1], answer_ids)
        output.append(torch.softmax(logits, -1).float().cpu())
    return torch.cat(output)


def task_metrics(
    posterior: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    predicted_bins = posterior.argmax(1)
    target_bins = target.argmax(1)
    delta = (predicted_bins - target_bins).abs()
    circular = torch.minimum(delta, target.shape[1] - delta)
    return {
        "exact_bin_accuracy": float((predicted_bins == target_bins).float().mean()),
        "mean_circular_error_radians": float(
            circular.float().mean() * (2.0 * math.pi / target.shape[1])
        ),
        "mean_target_cross_entropy": float(
            -(target * posterior.clamp_min(1e-12).log()).sum(1).mean()
        ),
    }


def _metric_replay_error(
    measured: Mapping[str, float], expected: Mapping[str, float]
) -> float:
    return max(abs(float(measured[key]) - float(expected[key])) for key in measured)


def task_adequacy_pass(metrics: Mapping[str, float], floor: float) -> bool:
    return float(metrics["exact_bin_accuracy"]) >= float(floor)


def classify_failed_cell(exact_pass: bool, oracle_pass: bool) -> str:
    if exact_pass and oracle_pass:
        return "sensor_scalar_estimation_failure"
    if not exact_pass and oracle_pass:
        return "scalar_coordinate_or_boundary_failure"
    if not exact_pass and not oracle_pass:
        return "continuation_or_answer_row_failure"
    return "invalid_oracle_resolution"


def _safe_distribution_summary(values: torch.Tensor) -> dict[str, float]:
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(unbiased=False)),
    }


@torch.inference_mode()
def _oracle_reachability(
    system: calibrated.CalibratedTinyLLM,
    input_ids: torch.Tensor,
    natural_scalar: torch.Tensor,
    exact_cosine: torch.Tensor,
    target: torch.Tensor,
    task: CircleTaskConfig,
    config: ScalarInterfaceConfig,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    contexts = torch.stack((input_ids[:, 0], input_ids[:, -1]), dim=1)
    unique_contexts, inverse = torch.unique(
        contexts, sorted=True, dim=0, return_inverse=True
    )
    oracle_posterior = torch.empty_like(target)
    selected_scalar = torch.zeros(len(target), dtype=torch.float32)
    reachable = torch.zeros(len(target), dtype=torch.bool)
    nearest_natural = torch.zeros(len(target), dtype=torch.float32)
    nearest_exact = torch.zeros(len(target), dtype=torch.float32)
    distance_natural = torch.zeros(len(target), dtype=torch.float32)
    distance_exact = torch.zeros(len(target), dtype=torch.float32)
    arrays: dict[str, np.ndarray] = {
        "contexts": unique_contexts.numpy(),
        "context_inverse": inverse.numpy(),
    }
    context_summaries = []
    fixed_grid = torch.linspace(
        config.scalar_grid_minimum,
        config.scalar_grid_maximum,
        config.scalar_grid_points,
        dtype=torch.float32,
    )
    for context_index in range(len(unique_contexts)):
        indices = torch.nonzero(inverse == context_index, as_tuple=False).reshape(-1)
        candidates = torch.unique(
            torch.cat(
                (
                    fixed_grid,
                    natural_scalar[indices],
                    exact_cosine[indices],
                )
            ),
            sorted=True,
        )
        representative = input_ids[indices[0] : indices[0] + 1]
        curve_inputs = representative.repeat(len(candidates), 1)
        curve = _scalar_posteriors(
            system, curve_inputs, candidates, task, config, device
        )
        local_target = target[indices]
        cross_entropy = -torch.matmul(
            local_target.double(), curve.clamp_min(1e-12).log().double().T
        )
        best = cross_entropy.argmin(1)
        oracle_posterior[indices] = curve[best]
        selected_scalar[indices] = candidates[best]
        curve_bins = curve.argmax(1)
        local_target_bins = local_target.argmax(1)
        reachable_bins = sorted(set(int(item) for item in curve_bins.tolist()))
        for target_bin in range(target.shape[1]):
            local = torch.nonzero(
                local_target_bins == target_bin, as_tuple=False
            ).reshape(-1)
            if not len(local):
                continue
            matching = candidates[curve_bins == target_bin]
            if not len(matching):
                continue
            global_indices = indices[local]
            reachable[global_indices] = True
            natural_distances = (
                natural_scalar[global_indices, None] - matching[None]
            ).abs()
            exact_distances = (
                exact_cosine[global_indices, None] - matching[None]
            ).abs()
            natural_best = natural_distances.argmin(1)
            exact_best = exact_distances.argmin(1)
            nearest_natural[global_indices] = matching[natural_best]
            nearest_exact[global_indices] = matching[exact_best]
            distance_natural[global_indices] = natural_distances[
                torch.arange(len(global_indices)), natural_best
            ]
            distance_exact[global_indices] = exact_distances[
                torch.arange(len(global_indices)), exact_best
            ]
        arrays[f"context_{context_index}_candidates"] = candidates.numpy()
        arrays[f"context_{context_index}_posterior_curve"] = curve.numpy()
        context_summaries.append(
            {
                "context_index": context_index,
                "bos_token": int(unique_contexts[context_index, 0]),
                "query_token": int(unique_contexts[context_index, 1]),
                "example_count": int(len(indices)),
                "candidate_count": int(len(candidates)),
                "reachable_target_bins": reachable_bins,
            }
        )

    reachable_dist_natural = distance_natural[reachable]
    reachable_dist_exact = distance_exact[reachable]
    metrics = task_metrics(oracle_posterior, target)
    record = {
        "context_count": int(len(unique_contexts)),
        "contexts": context_summaries,
        "exact_bin_reachability": float(reachable.float().mean()),
        "unreachable_example_count": int((~reachable).sum()),
        "minimum_cross_entropy_selection_metrics": metrics,
        "selected_scalar": _safe_distribution_summary(selected_scalar),
        "natural_to_reachable_distance_median": (
            float(reachable_dist_natural.median()) if len(reachable_dist_natural) else None
        ),
        "natural_to_reachable_distance_p90": (
            float(torch.quantile(reachable_dist_natural, 0.90))
            if len(reachable_dist_natural)
            else None
        ),
        "exact_to_reachable_distance_median": (
            float(reachable_dist_exact.median()) if len(reachable_dist_exact) else None
        ),
        "exact_to_reachable_distance_p90": (
            float(torch.quantile(reachable_dist_exact, 0.90))
            if len(reachable_dist_exact)
            else None
        ),
    }
    arrays.update(
        {
            "oracle_posterior": oracle_posterior.numpy(),
            "oracle_selected_scalar": selected_scalar.numpy(),
            "reachable": reachable.numpy(),
            "nearest_reachable_from_natural": nearest_natural.numpy(),
            "nearest_reachable_from_exact": nearest_exact.numpy(),
            "distance_from_natural": distance_natural.numpy(),
            "distance_from_exact": distance_exact.numpy(),
        }
    )
    return record, arrays


def _fingerprint(
    config: ScalarInterfaceConfig,
    preset: str,
    condition: str,
    seed: int,
    implementation: str,
    source_detail: Mapping[str, Any],
    dataset_hashes: Mapping[str, str],
) -> str:
    return _json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "configuration": _json_config(config),
            "preset": preset,
            "condition": condition,
            "seed": seed,
            "implementation_sha256": implementation,
            "source_result_sha256": source_detail["_result_sha256"],
            "dataset_hashes": dict(dataset_hashes),
        }
    )


def _cell_directory(root: Path, preset: str, condition: str, seed: int) -> Path:
    return root / "runs" / preset / condition / f"seed_{seed}"


def _existing_detail(
    output_root: Path,
    config: ScalarInterfaceConfig,
    preset: str,
    condition: str,
    seed: int,
    implementation: str,
    source_detail: Mapping[str, Any],
    dataset_hashes: Mapping[str, str],
) -> dict[str, Any] | None:
    path = _cell_directory(output_root, preset, condition, seed) / "result.json"
    if not path.is_file():
        return None
    try:
        detail = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_fingerprint = _fingerprint(
        config,
        preset,
        condition,
        seed,
        implementation,
        source_detail,
        dataset_hashes,
    )
    diagnostics = Path(detail.get("artifacts", {}).get("diagnostics", ""))
    if (
        detail.get("schema_version") != SCHEMA_VERSION
        or detail.get("status") != "completed"
        or detail.get("scientific_fingerprint") != expected_fingerprint
        or detail.get("implementation_sha256") != implementation
        or detail.get("gates", {}).get("validity") is not True
        or not diagnostics.is_file()
        or _sha256(diagnostics)
        != detail.get("artifacts", {}).get("diagnostics_sha256")
    ):
        return None
    return detail


def run_cell(
    config: ScalarInterfaceConfig,
    task: CircleTaskConfig,
    preset: str,
    condition: str,
    seed: int,
    source_detail: Mapping[str, Any],
    datasets: Mapping[str, calibrated.CalibratedDataset],
    dataset_hashes: Mapping[str, str],
    implementation_sources: Mapping[str, str],
    implementation: str,
    output_root: Path,
) -> dict[str, Any]:
    output_dir = _cell_directory(output_root, preset, condition, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    system = _load_system(source_detail, task, preset, condition, device)
    state_before = {
        "model": calibrated._state_digest(system.model),
        "system": calibrated._module_digest(system),
    }
    regime_records: dict[str, Any] = {}
    diagnostic_arrays: dict[str, np.ndarray] = {}
    for regime in config.regimes:
        dataset = datasets[regime]
        input_ids = dataset.paired.circle.input_ids
        target = dataset.paired.circle.target_posteriors.float()
        natural_scalar = _natural_scalar(system, dataset, task, config, device)
        exact_cosine = dataset.paired.fiber.cosine.reshape(-1).float()
        shuffled_cosine = torch.roll(exact_cosine, shifts=config.shuffle_shift)

        direct = _direct_posteriors(system, dataset, task, config, device)
        injected_natural = _scalar_posteriors(
            system, input_ids, natural_scalar, task, config, device
        )
        exact = _scalar_posteriors(
            system, input_ids, exact_cosine, task, config, device
        )
        negative = _scalar_posteriors(
            system, input_ids, -exact_cosine, task, config, device
        )
        shuffled = _scalar_posteriors(
            system, input_ids, shuffled_cosine, task, config, device
        )
        oracle, oracle_arrays = _oracle_reachability(
            system,
            input_ids,
            natural_scalar,
            exact_cosine,
            target,
            task,
            config,
            device,
        )
        direct_metrics = task_metrics(direct, target)
        natural_metrics = task_metrics(injected_natural, target)
        exact_metrics = task_metrics(exact, target)
        negative_metrics = task_metrics(negative, target)
        shuffled_metrics = task_metrics(shuffled, target)
        source_metrics = source_detail["representation"]["task_metrics"][regime]
        stored_replay_error = (
            _metric_replay_error(direct_metrics, source_metrics)
            if config.samples_per_regime == 1_024
            else None
        )
        floor = float(source_detail["gates"]["task_accuracy_floors"][regime])
        record = {
            "dataset_sha256": dataset_hashes[regime],
            "task_accuracy_floor": floor,
            "natural_scalar": _safe_distribution_summary(natural_scalar),
            "exact_cosine": _safe_distribution_summary(exact_cosine),
            "natural_metrics": natural_metrics,
            "exact_cosine_metrics": exact_metrics,
            "negative_cosine_metrics": negative_metrics,
            "shuffled_cosine_metrics": shuffled_metrics,
            "oracle": oracle,
            "replay": {
                "direct_vs_injected_natural_maximum_absolute_posterior_error": float(
                    (direct - injected_natural).abs().max()
                ),
                "direct_vs_stored_task_metrics_maximum_absolute_error": stored_replay_error,
                "stored_metrics_same_cohort": config.samples_per_regime == 1_024,
            },
            "gates": {
                "natural_source_adequacy": task_adequacy_pass(natural_metrics, floor),
                "exact_cosine_adequacy": task_adequacy_pass(exact_metrics, floor),
                "negative_cosine_adequacy": task_adequacy_pass(negative_metrics, floor),
                "shuffled_cosine_adequacy": task_adequacy_pass(shuffled_metrics, floor),
                "oracle_adequacy": task_adequacy_pass(
                    oracle["minimum_cross_entropy_selection_metrics"], floor
                ),
            },
        }
        regime_records[regime] = record
        prefix = f"{regime}__"
        diagnostic_arrays.update(
            {
                f"{prefix}natural_scalar": natural_scalar.numpy(),
                f"{prefix}exact_cosine": exact_cosine.numpy(),
                f"{prefix}shuffled_cosine": shuffled_cosine.numpy(),
                f"{prefix}target_posterior": target.numpy(),
                f"{prefix}direct_posterior": direct.numpy(),
                f"{prefix}natural_injected_posterior": injected_natural.numpy(),
                f"{prefix}exact_posterior": exact.numpy(),
                f"{prefix}negative_posterior": negative.numpy(),
                f"{prefix}shuffled_posterior": shuffled.numpy(),
                **{f"{prefix}{name}": value for name, value in oracle_arrays.items()},
            }
        )

    source_failed = not bool(source_detail["gates"]["task_adequacy_pass"])
    exact_pass = all(
        regime_records[regime]["gates"]["exact_cosine_adequacy"]
        for regime in config.regimes
    )
    oracle_pass = all(
        regime_records[regime]["gates"]["oracle_adequacy"]
        for regime in config.regimes
    )
    negative_pass = all(
        regime_records[regime]["gates"]["negative_cosine_adequacy"]
        for regime in config.regimes
    )
    shuffled_pass = all(
        regime_records[regime]["gates"]["shuffled_cosine_adequacy"]
        for regime in config.regimes
    )
    replay_pass = all(
        regime_records[regime]["replay"][
            "direct_vs_injected_natural_maximum_absolute_posterior_error"
        ]
        <= config.replay_tolerance
        and (
            regime_records[regime]["replay"][
                "direct_vs_stored_task_metrics_maximum_absolute_error"
            ]
            <= config.replay_tolerance
            if regime_records[regime]["replay"][
                "direct_vs_stored_task_metrics_maximum_absolute_error"
            ]
            is not None
            else True
        )
        for regime in config.regimes
    )
    scalar_range_pass = all(
        regime_records[regime][name]["minimum"]
        >= config.scalar_grid_minimum - config.scalar_range_tolerance
        and regime_records[regime][name]["maximum"]
        <= config.scalar_grid_maximum + config.scalar_range_tolerance
        for regime in config.regimes
        for name in ("natural_scalar", "exact_cosine")
    )
    positive_control_recovered = oracle_pass if not source_failed else True
    oracle_resolution_valid = not (exact_pass and not oracle_pass)
    state_after = {
        "model": calibrated._state_digest(system.model),
        "system": calibrated._module_digest(system),
    }
    state_unchanged = state_before == state_after == {
        "model": source_detail["training"]["final_model_state_sha256"],
        "system": source_detail["training"]["final_system_state_sha256"],
    }
    finite = _finite(regime_records) and all(
        np.isfinite(value).all() for value in diagnostic_arrays.values()
    )
    validity = bool(
        replay_pass
        and scalar_range_pass
        and positive_control_recovered
        and oracle_resolution_valid
        and state_unchanged
        and finite
    )
    classification = (
        classify_failed_cell(exact_pass, oracle_pass)
        if source_failed
        else "source_passing_positive_control"
    )
    diagnostics_path = output_dir / "diagnostics.npz"
    closure._write_npz(diagnostics_path, diagnostic_arrays)
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    result_path = output_dir / "result.json"
    fingerprint = _fingerprint(
        config,
        preset,
        condition,
        seed,
        implementation,
        source_detail,
        dataset_hashes,
    )
    detail = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-scalar-interface-{preset}-{condition}-seed{seed}",
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_scientific_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "completed_at": _utc_now(),
        "preset": preset,
        "condition": condition,
        "seed": seed,
        "device": str(device),
        "configuration": _json_config(config),
        "task_config": asdict(task),
        "implementation_sha256": implementation,
        "implementation_sources": dict(implementation_sources),
        "scientific_fingerprint": fingerprint,
        "source": {
            "campaign": str(Path(config.source_root) / "campaign_results.json"),
            "campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "result": source_detail["_result_path"],
            "result_sha256": source_detail["_result_sha256"],
            "scientific_fingerprint": source_detail["scientific_fingerprint"],
            "source_task_adequacy_pass": not source_failed,
            "source_joint_seed_pass": source_detail["gates"]["joint_seed_pass"],
            "model_checkpoint": source_detail["artifacts"]["checkpoint"],
            "model_checkpoint_sha256": source_detail["artifacts"]["checkpoint_sha256"],
            "frontend_checkpoint": source_detail["artifacts"]["frontend_checkpoint"],
            "frontend_checkpoint_sha256": source_detail["artifacts"][
                "frontend_checkpoint_sha256"
            ],
        },
        "regimes": regime_records,
        "state_record": {"before": state_before, "after": state_after},
        "classification": classification,
        "gates": {
            "source_failed": source_failed,
            "exact_cosine_both_shifts": exact_pass,
            "oracle_both_shifts": oracle_pass,
            "negative_cosine_both_shifts": negative_pass,
            "shuffled_cosine_both_shifts": shuffled_pass,
            "positive_control_recovered": positive_control_recovered,
            "oracle_resolution_valid": oracle_resolution_valid,
            "natural_replay": replay_pass,
            "scalar_range": scalar_range_pass,
            "state_unchanged": state_unchanged,
            "finite": finite,
            "validity": validity,
        },
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_gb": peak,
        "artifacts": {
            "result": str(result_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": _sha256(diagnostics_path),
        },
    }
    _write_json(result_path, detail)
    del system
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return detail


def aggregate_details(
    details: Sequence[Mapping[str, Any]], config: ScalarInterfaceConfig
) -> dict[str, Any]:
    failed = [detail for detail in details if detail["gates"]["source_failed"]]
    passing = [detail for detail in details if not detail["gates"]["source_failed"]]
    classifications: dict[str, int] = {}
    for detail in details:
        label = str(detail["classification"])
        classifications[label] = classifications.get(label, 0) + 1
    strata = {}
    for preset, condition in (
        ("d6", "learned_calibrated_equivariant"),
        ("d10", "analytic_calibrated"),
        ("d10", "learned_calibrated_equivariant"),
    ):
        cells = [
            detail
            for detail in failed
            if detail["preset"] == preset and detail["condition"] == condition
        ]
        strata[f"{preset}/{condition}"] = {
            "source_failed_count": len(cells),
            "exact_cosine_repair_count": sum(
                bool(item["gates"]["exact_cosine_both_shifts"]) for item in cells
            ),
            "oracle_repair_count": sum(
                bool(item["gates"]["oracle_both_shifts"]) for item in cells
            ),
        }
    exact_count = sum(
        bool(detail["gates"]["exact_cosine_both_shifts"]) for detail in failed
    )
    oracle_count = sum(
        bool(detail["gates"]["oracle_both_shifts"]) for detail in failed
    )
    positive_count = sum(
        bool(detail["gates"]["oracle_both_shifts"]) for detail in passing
    )
    negative_count = sum(
        bool(detail["gates"]["negative_cosine_both_shifts"]) for detail in failed
    )
    shuffled_count = sum(
        bool(detail["gates"]["shuffled_cosine_both_shifts"]) for detail in failed
    )
    uniform_upstream = bool(
        len(failed) == 10
        and exact_count >= 8
        and strata["d6/learned_calibrated_equivariant"][
            "exact_cosine_repair_count"
        ]
        >= 3
        and strata["d10/analytic_calibrated"]["exact_cosine_repair_count"] == 2
        and strata["d10/learned_calibrated_equivariant"][
            "exact_cosine_repair_count"
        ]
        >= 3
    )
    one_dimensional_sufficient = bool(
        len(failed) == 10
        and oracle_count == 10
        and len(passing) == 10
        and positive_count == 10
    )
    specificity = bool(
        negative_count <= config.negative_control_pass_ceiling
        and shuffled_count <= config.shuffled_control_pass_ceiling
    )
    valid = bool(
        len(details) == 20
        and len(failed) == 10
        and len(passing) == 10
        and all(detail["gates"]["validity"] for detail in details)
        and specificity
    )
    if not valid:
        classification = "invalid"
    elif uniform_upstream:
        classification = "source_failure_uniformly_upstream_of_scalar_embedding"
    elif one_dimensional_sufficient and exact_count:
        classification = (
            "mixed_sensor_and_scalar_coordinate_failures_with_sufficient_continuation"
        )
    elif one_dimensional_sufficient:
        classification = "scalar_coordinate_failure_with_sufficient_continuation"
    else:
        classification = "continuation_capacity_failure_present"
    return {
        "valid": valid,
        "classification": classification,
        "source_failed_count": len(failed),
        "source_passing_control_count": len(passing),
        "classification_counts": classifications,
        "strata": strata,
        "exact_cosine_repair_count": exact_count,
        "oracle_repair_count": oracle_count,
        "positive_control_oracle_recovery_count": positive_count,
        "negative_cosine_pass_count": negative_count,
        "shuffled_cosine_pass_count": shuffled_count,
        "gates": {
            "uniformly_upstream_of_scalar_embedding": uniform_upstream,
            "one_dimensional_interface_expressively_sufficient": one_dimensional_sufficient,
            "semantic_specificity": specificity,
            "primary_hypothesis_pass": uniform_upstream,
        },
    }


def _campaign_fingerprint(
    config: ScalarInterfaceConfig,
    implementation: str,
    source_manifest: str,
    dataset_hashes: Mapping[str, str],
) -> str:
    return _json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "configuration": _json_config(config),
            "implementation_sha256": implementation,
            "source_manifest_sha256": source_manifest,
            "dataset_hashes": dict(dataset_hashes),
        }
    )


def run_campaign(
    config: ScalarInterfaceConfig, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = _implementation_sources()
    implementation = _implementation_digest(sources)
    source_campaign, task, source_details = _source_details(config)
    datasets = _datasets(task, config)
    dataset_hashes = _dataset_hashes(datasets)
    if (
        not config.allow_underpowered
        and dataset_hashes != EXPECTED_DATASET_HASHES
    ):
        raise RuntimeError("primary scalar-interface cohort hashes changed")
    source_manifest = _json_hash(
        {
            f"{preset}/{condition}/seed_{seed}": detail["_result_sha256"]
            for (preset, condition, seed), detail in source_details.items()
        }
    )
    fingerprint = _campaign_fingerprint(
        config, implementation, source_manifest, dataset_hashes
    )
    selected = [
        (preset, condition, seed)
        for preset in config.presets
        for condition in config.conditions
        for seed in config.seeds
    ]
    campaign_path = output_dir / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "completed"
            and existing.get("campaign_fingerprint") == fingerprint
            and all(
                _existing_detail(
                    output_dir,
                    config,
                    preset,
                    condition,
                    seed,
                    implementation,
                    source_details[(preset, condition, seed)],
                    dataset_hashes,
                )
                is not None
                for preset, condition, seed in selected
            )
        ):
            print("campaign already complete; leaving bytes unchanged", flush=True)
            return existing

    details = []
    for index, (preset, condition, seed) in enumerate(selected, start=1):
        existing = _existing_detail(
            output_dir,
            config,
            preset,
            condition,
            seed,
            implementation,
            source_details[(preset, condition, seed)],
            dataset_hashes,
        )
        if existing is not None:
            detail = existing
            action = "reused"
        else:
            detail = run_cell(
                config,
                task,
                preset,
                condition,
                seed,
                source_details[(preset, condition, seed)],
                datasets,
                dataset_hashes,
                sources,
                implementation,
                output_dir,
            )
            action = "completed"
        details.append(detail)
        print(
            f"[{index}/{len(selected)}] {action} {preset}/{condition}/seed_{seed} "
            f"classification={detail['classification']}",
            flush=True,
        )

    complete = len(details) == len(selected)
    aggregate = (
        aggregate_details(details, config)
        if complete and not config.allow_underpowered
        else {
            "valid": all(detail["gates"]["validity"] for detail in details),
            "classification": "systems_lifecycle_only_not_scientific_evidence",
        }
    )
    result_paths = [
        Path(detail["artifacts"]["result"]) for detail in details
    ]
    diagnostics_paths = [
        Path(detail["artifacts"]["diagnostics"]) for detail in details
    ]
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed" if complete else "partial",
        "completed_at": _utc_now(),
        "evidence_role": (
            "systems_lifecycle_only_not_scientific_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "configuration": _json_config(config),
        "task_config": asdict(task),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "implementation_sha256": implementation,
        "implementation_sources": sources,
        "campaign_fingerprint": fingerprint,
        "source": {
            "campaign": str(Path(config.source_root) / "campaign_results.json"),
            "campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "result_manifest_sha256": SOURCE_RESULT_MANIFEST_SHA256,
            "artifact_manifest_sha256": SOURCE_ARTIFACT_MANIFEST_SHA256,
            "source_campaign_fingerprint": source_campaign["campaign_fingerprint"],
            "selected_source_manifest_sha256": source_manifest,
        },
        "dataset_hashes": dataset_hashes,
        "summary": {
            "requested": len(selected),
            "completed": len(details),
            "failed": len(selected) - len(details),
            "trained_models": 0,
            "trained_frontends": 0,
            "trained_probes": 0,
            "fitted_parameters": 0,
        },
        "aggregates": aggregate,
        "result_manifest_sha256": _manifest_sha256(result_paths),
        "diagnostics_manifest_sha256": _manifest_sha256(diagnostics_paths),
        "results": [
            {
                "experiment_id": detail["experiment_id"],
                "preset": detail["preset"],
                "condition": detail["condition"],
                "seed": detail["seed"],
                "classification": detail["classification"],
                "source_failed": detail["gates"]["source_failed"],
                "exact_cosine_both_shifts": detail["gates"][
                    "exact_cosine_both_shifts"
                ],
                "oracle_both_shifts": detail["gates"]["oracle_both_shifts"],
                "result": detail["artifacts"]["result"],
            }
            for detail in details
        ],
        "method_boundaries": [
            "Exact cosine and oracle-selected scalars use hidden generator or target information.",
            "Oracle reachability is an expressivity diagnostic, not an observable repair.",
            "The d6/d10 presets co-vary depth, width, and head count.",
            "This is registered post-outcome diagnostic evidence from frozen checkpoints.",
        ],
    }
    _write_json(campaign_path, bundle)
    return bundle


def _comma_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--presets", default=",".join(PRESETS))
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--regimes", default=",".join(REGIMES))
    parser.add_argument("--samples", type=int, default=1_024)
    parser.add_argument("--grid-points", type=int, default=4_097)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-underpowered", action="store_true")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_frozen_scalar_interface_decomposition/"
            "20260811_d6_d10_preregistered"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.shakedown:
        args.presets = "d6"
        args.conditions = "analytic_calibrated"
        args.seeds = "7"
        args.samples = 64
        args.grid_points = 129
        args.batch_size = 64
        args.allow_underpowered = True
    config = ScalarInterfaceConfig(
        source_root=args.source_root,
        presets=_comma_strings(args.presets),
        conditions=_comma_strings(args.conditions),
        seeds=_comma_ints(args.seeds),
        regimes=_comma_strings(args.regimes),
        samples_per_regime=args.samples,
        scalar_grid_points=args.grid_points,
        batch_size=args.batch_size,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    result = run_campaign(config, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "classification": result["aggregates"].get("classification"),
                "valid": result["aggregates"].get("valid"),
                "summary": result["summary"],
                "output": str(args.output / "campaign_results.json"),
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
