#!/usr/bin/env python3
"""Decompose C3 temporal continuation and readout with frozen checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
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

import experiments.structure_net.tinyllm_c3_temporal_quotient_analysis as analysis
import experiments.structure_net.tinyllm_c3_temporal_quotient_campaign as campaign
import experiments.structure_net.tinyllm_c3_temporal_quotient_preflight as preflight
import experiments.structure_net.tinyllm_c3_temporal_quotient_training as stage0
import experiments.structure_net.tinyllm_frozen_interval_readout_decomposition as interval
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-c3-temporal-continuation-readout.v1"
HYPOTHESIS_ID = "tinyllm-c3-temporal-continuation-readout-v1"
EVIDENCE_ROLE = (
    "prospective_artifact_only_c3_continuation_readout_decomposition"
)
SOURCE_HYPOTHESIS_ID = campaign.HYPOTHESIS_ID
SOURCE_ROOT = Path(
    "data/experiments/tinyllm_c3_temporal_quotient/"
    "20260811_d6_preregistered"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-temporal-continuation-readout-"
    "preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "f4833ffcc7455fd7b72c23d49da1db92920cad5e3030928cccef6af53dbc20ae"
)
SOURCE_CAMPAIGN_SHA256 = (
    "e46710de358a3c9c5d30ddd4a19e36182b94a50cabc699014f79d58d3d3de7cc"
)
SOURCE_RESULT_MANIFEST_SHA256 = (
    "7dfdcf1ff80fe20a975fe6a7d1311dc92e3ff1a396a6da9550c91835a568a0ff"
)
SOURCE_ARTIFACT_MANIFEST_SHA256 = (
    "a0b90484863346cf2a5e0ef8be65cac3a221cfa50a373f88c9ead07a0cd351a1"
)
SOURCE_RUNNER_SHA256 = (
    "9b2cd0e3ce3752b7eea80d5859c11880a9d3732fb48b58306e34eab4f080d5ec"
)
SOURCE_ANALYSIS_SHA256 = (
    "89dacc60d02707678e689c6ce1e8f9c963889af352565a227bb90ed8e367e6a3"
)
SOURCE_TRAINING_SHA256 = (
    "dbc934190b5f725185ff9a99690409ac63fc4f16bd04686f870e7ef21cba03a6"
)
INTERVAL_IMPLEMENTATION_SHA256 = (
    "bb6a73c203fcf4e654295bf7567f205826a6f054153ad50bbdd36851297de926"
)
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
FIT_ARMS = (
    "output_scalar_recalibration",
    "untyped_final_readout",
    "typed_final_readout",
)


@dataclass(frozen=True)
class C3ReadoutConfig:
    source_root: str = str(SOURCE_ROOT)
    seeds: tuple[int, ...] = SEEDS
    train_samples: int = 4_096
    evaluation_latents: int = 1_024
    batch_size: int = 256
    ridge_lambda: float = 1e-4
    standardization_floor: float = 1e-6
    scalar_correlation_minimum: float = 0.90
    composition_accuracy_minimum: float = 0.50
    extrapolation_accuracy_minimum: float = 0.35
    composition_cross_entropy_maximum: float = 1.80
    extrapolation_cross_entropy_maximum: float = 2.20
    composition_coverage_minimum: int = 14
    extrapolation_coverage_minimum: int = 12
    posterior_correlation_minimum: float = 0.90
    required_seed_passes: int = 4
    shuffled_seed_pass_ceiling: int = 1
    replay_tolerance: float = 2e-6
    shuffle_seed: int = 20_260_811
    device: str = "cuda:0"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and distinct")
        if min(self.train_samples, self.evaluation_latents, self.batch_size) < 1:
            raise ValueError("sample and batch sizes must be positive")
        if self.ridge_lambda <= 0.0 or self.standardization_floor <= 0.0:
            raise ValueError("ridge and standardization floors must be positive")
        if self.required_seed_passes < 1 or self.shuffled_seed_pass_ceiling < 0:
            raise ValueError("population thresholds are invalid")
        if not self.allow_underpowered:
            expected = C3ReadoutConfig(allow_underpowered=True)
            locked = {
                key: value
                for key, value in asdict(expected).items()
                if key != "allow_underpowered"
            }
            actual = {
                key: value
                for key, value in asdict(self).items()
                if key != "allow_underpowered"
            }
            if actual != locked:
                raise ValueError("primary C3 readout configuration changed")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@lru_cache(maxsize=512)
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


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, Iterable):
        return all(_finite(item) for item in value)
    return True


def _json_config(config: C3ReadoutConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _source_paths(root: Path) -> tuple[list[Path], list[Path]]:
    results = [
        root / "runs" / "analytic" / f"seed_{seed}" / "result.json"
        for seed in SEEDS
    ]
    artifacts = [
        root / "runs" / "analytic" / f"seed_{seed}" / name
        for seed in SEEDS
        for name in ("model.pt", "frontend.pt", "diagnostics.pt")
    ]
    return results, artifacts


def _source_campaign(
    root: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    path = root / "campaign_results.json"
    if _sha256(path) != SOURCE_CAMPAIGN_SHA256:
        raise ValueError("source C3 campaign changed")
    record = json.loads(path.read_text(encoding="utf-8"))
    results, artifacts = _source_paths(root)
    if _manifest_sha256(results) != SOURCE_RESULT_MANIFEST_SHA256:
        raise ValueError("source C3 result manifest changed")
    if _manifest_sha256(artifacts) != SOURCE_ARTIFACT_MANIFEST_SHA256:
        raise ValueError("source C3 artifact manifest changed")
    source_hashes = record.get("source_hashes", {})
    analytic = record.get("aggregates", {}).get("arms", {}).get("analytic", {})
    if (
        record.get("schema_version") != campaign.SCHEMA_VERSION
        or record.get("hypothesis_id") != SOURCE_HYPOTHESIS_ID
        or record.get("status") != "completed"
        or record.get("aggregates", {}).get("classification")
        != "c3_positive_control_task_failure"
        or record.get("summary", {}).get("completed") != 5
        or record.get("summary", {}).get("stopped_by_positive_control") != 10
        or analytic.get("valid_count") != 5
        or analytic.get("natural_task_pass_count") != 2
        or analytic.get("representation_pass_count") != 5
        or analytic.get("causal_pass_count") != 5
        or source_hashes.get("runner") != SOURCE_RUNNER_SHA256
        or source_hashes.get("analysis") != SOURCE_ANALYSIS_SHA256
        or source_hashes.get("stage0_runner") != SOURCE_TRAINING_SHA256
    ):
        raise ValueError("invalid source C3 campaign contract")
    details: dict[int, dict[str, Any]] = {}
    for result_path in results:
        detail = json.loads(result_path.read_text(encoding="utf-8"))
        seed = int(detail.get("seed", -1))
        source_artifacts = detail.get("artifacts", {})
        if (
            seed not in SEEDS
            or detail.get("arm") != "analytic"
            or detail.get("status") != "completed"
            or detail.get("gates", {}).get("validity") is not True
            or detail.get("gates", {}).get("representation") is not True
            or detail.get("gates", {}).get("causal_all_cuts") is not True
            or detail.get("gates", {}).get("exact_action") is not True
            or detail.get("gates", {}).get("identity_replay") is not True
            or Path(source_artifacts.get("result", "")) != result_path
            or _sha256(Path(source_artifacts["checkpoint"]))
            != source_artifacts.get("checkpoint_sha256")
            or _sha256(Path(source_artifacts["frontend_checkpoint"]))
            != source_artifacts.get("frontend_checkpoint_sha256")
            or _sha256(Path(source_artifacts["diagnostics"]))
            != source_artifacts.get("diagnostics_sha256")
        ):
            raise ValueError(f"invalid source C3 seed {seed}")
        detail["_result_path"] = str(result_path)
        detail["_result_sha256"] = _sha256(result_path)
        details[seed] = detail
    if set(details) != set(SEEDS):
        raise ValueError("source C3 seed population changed")
    return record, details


def _implementation_sources() -> dict[str, str]:
    values = {
        "runner": _sha256(Path(__file__)),
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "source_campaign_runner": _sha256(Path(campaign.__file__)),
        "source_analysis": _sha256(Path(analysis.__file__)),
        "source_training": _sha256(Path(stage0.__file__)),
        "interval_readout": _sha256(Path(interval.__file__)),
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "source_campaign_runner": SOURCE_RUNNER_SHA256,
        "source_analysis": SOURCE_ANALYSIS_SHA256,
        "source_training": SOURCE_TRAINING_SHA256,
        "interval_readout": INTERVAL_IMPLEMENTATION_SHA256,
    }
    for name, digest in expected.items():
        if values[name] != digest:
            raise RuntimeError(f"C3 readout source changed: {name}")
    return values


def _implementation_digest(sources: Mapping[str, str] | None = None) -> str:
    return _json_hash(dict(sources or _implementation_sources()))


def _load_system(
    detail: Mapping[str, Any], device: torch.device
) -> stage0.C3TemporalTinyLLM:
    model = TinyLLMModel.from_checkpoint(
        Path(detail["artifacts"]["checkpoint"]), map_location=device
    )
    system = stage0.C3TemporalTinyLLM(model, "analytic").to(device)
    frontend = torch.load(
        Path(detail["artifacts"]["frontend_checkpoint"]),
        map_location=device,
        weights_only=True,
    )
    if frontend.get("encoder") is not None:
        raise ValueError("analytic source unexpectedly contains a learned encoder")
    system.sequence_embedding.load_state_dict(frontend["sequence_embedding"])
    return system


@torch.inference_mode()
def _extract_state(
    system: stage0.C3TemporalTinyLLM,
    dataset: stage0.C3TrainingDataset,
    task: stage0.C3TaskConfig,
    config: C3ReadoutConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    system.eval()
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    residuals: list[torch.Tensor] = []
    posteriors: list[torch.Tensor] = []
    for start in range(0, len(dataset), config.batch_size):
        stop = start + config.batch_size
        cuts = system.residual_cuts(
            dataset.tokens[start:stop].to(device),
            dataset.calibration[start:stop].to(device),
        )
        query = cuts["full"][:, -1]
        normalized = system.model.transformer["ln_f"](query)
        logits = system.model.lm_head(normalized).index_select(-1, answer_ids)
        residuals.append(normalized.float().cpu())
        posteriors.append(torch.softmax(logits, -1).float().cpu())
    posterior = torch.cat(posteriors)
    centers = torch.linspace(-1.0, 1.0, task.phase_bins)
    return {
        "normalized_final_residual": torch.cat(residuals),
        "source_posterior": posterior,
        "source_posterior_mean": posterior @ centers,
    }


def _posterior_metrics(
    posterior: torch.Tensor, dataset: stage0.C3TrainingDataset
) -> dict[str, Any]:
    return analysis.task_metrics_from_logits(
        posterior.to(torch.float64).clamp_min(1e-15).log(),
        dataset.target_posteriors,
        dataset.target_bins,
        dataset.target,
    )


def _scalar_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    x = predicted.detach().cpu().reshape(-1).to(torch.float64)
    y = target.detach().cpu().reshape(-1).to(torch.float64)
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    denominator = torch.linalg.vector_norm(centered_x) * torch.linalg.vector_norm(
        centered_y
    )
    correlation = (
        float((centered_x * centered_y).sum() / denominator)
        if float(denominator) > 0.0
        else 0.0
    )
    return {
        "cosine_pearson": correlation,
        "cosine_rmse": float(torch.sqrt((x - y).square().mean())),
        "minimum": float(x.min()),
        "maximum": float(x.max()),
    }


def _task_gate(
    metrics: Mapping[str, Any], regime: str, config: C3ReadoutConfig
) -> bool:
    if regime == "composition":
        accuracy = config.composition_accuracy_minimum
        cross_entropy = config.composition_cross_entropy_maximum
        coverage = config.composition_coverage_minimum
    elif regime == "extrapolation":
        accuracy = config.extrapolation_accuracy_minimum
        cross_entropy = config.extrapolation_cross_entropy_maximum
        coverage = config.extrapolation_coverage_minimum
    else:
        raise ValueError(f"unknown C3 regime {regime!r}")
    return bool(
        float(metrics["posterior_mean_correlation"])
        >= config.posterior_correlation_minimum
        and float(metrics["exact_bin_accuracy"]) >= accuracy
        and float(metrics["target_cross_entropy"]) <= cross_entropy
        and int(metrics["predicted_bin_coverage"]) >= coverage
    )


def deterministic_target_permutation(
    count: int, seed: int, base_seed: int = 20_260_811
) -> torch.Tensor:
    material = f"{base_seed}:c3-temporal:{seed}".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    generator = torch.Generator(device="cpu").manual_seed(derived % (2**63 - 1))
    permutation = torch.randperm(count, generator=generator)
    if torch.equal(permutation, torch.arange(count)):
        permutation = torch.roll(permutation, 1)
    return permutation


def _fit_maps(
    state: Mapping[str, torch.Tensor],
    dataset: stage0.C3TrainingDataset,
    permutation: torch.Tensor,
    config: C3ReadoutConfig,
) -> dict[str, interval.RidgeMap]:
    target = dataset.target.to(torch.float64)
    log_target = dataset.target_posteriors.to(torch.float64).clamp_min(1e-15).log()
    log_target = log_target - log_target.mean(1, keepdim=True)
    residual = state["normalized_final_residual"]
    natural = state["source_posterior_mean"]
    kwargs = {
        "ridge_lambda": config.ridge_lambda,
        "standardization_floor": config.standardization_floor,
    }
    return {
        "recalibration_true": interval.fit_ridge_map(natural, target, **kwargs),
        "recalibration_shuffled": interval.fit_ridge_map(
            natural, target[permutation], **kwargs
        ),
        "typed_true": interval.fit_ridge_map(residual, target, **kwargs),
        "typed_shuffled": interval.fit_ridge_map(
            residual, target[permutation], **kwargs
        ),
        "untyped_true": interval.fit_ridge_map(residual, log_target, **kwargs),
        "untyped_shuffled": interval.fit_ridge_map(
            residual, log_target[permutation], **kwargs
        ),
    }


def _fit_arrays(maps: Mapping[str, interval.RidgeMap]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, fitted in maps.items():
        arrays[f"fit__{name}__mean"] = fitted.mean.numpy()
        arrays[f"fit__{name}__scale"] = fitted.scale.numpy()
        arrays[f"fit__{name}__coefficients"] = fitted.coefficients.numpy()
    return arrays


def _fingerprint(
    config: C3ReadoutConfig,
    implementation: str,
    source_detail: Mapping[str, Any],
    training_hash: str,
    evaluation_hashes: Mapping[str, str],
) -> str:
    return _json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "configuration": _json_config(config),
            "implementation_sha256": implementation,
            "source_result_sha256": source_detail["_result_sha256"],
            "training_data_sha256": training_hash,
            "evaluation_hashes": dict(evaluation_hashes),
        }
    )


def _cell_dir(root: Path, seed: int) -> Path:
    return root / "runs" / "analytic" / f"seed_{seed}"


def _existing_detail(
    root: Path,
    seed: int,
    fingerprint: str,
) -> dict[str, Any] | None:
    path = _cell_dir(root, seed) / "result.json"
    if not path.is_file():
        return None
    try:
        detail = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    diagnostics = Path(detail.get("artifacts", {}).get("diagnostics", ""))
    if (
        detail.get("schema_version") != SCHEMA_VERSION
        or detail.get("status") != "completed"
        or detail.get("scientific_fingerprint") != fingerprint
        or detail.get("gates", {}).get("validity") is not True
        or not diagnostics.is_file()
        or _sha256(diagnostics)
        != detail.get("artifacts", {}).get("diagnostics_sha256")
    ):
        return None
    return detail


def _maximum_stored_metric_error(
    measured: Mapping[str, Any], stored: Mapping[str, Any]
) -> float:
    names = (
        "exact_bin_accuracy",
        "target_cross_entropy",
        "posterior_mean_correlation",
        "posterior_mean_rmse",
        "mean_triple_angle_error_radians",
    )
    numeric = max(abs(float(measured[name]) - float(stored[name])) for name in names)
    coverage = abs(
        int(measured["predicted_bin_coverage"])
        - int(stored["predicted_bin_coverage"])
    )
    return max(numeric, float(coverage))


def run_cell(
    config: C3ReadoutConfig,
    source_record: Mapping[str, Any],
    detail: Mapping[str, Any],
    task: stage0.C3TaskConfig,
    evaluation: Mapping[str, stage0.C3TrainingDataset],
    evaluation_hashes: Mapping[str, str],
    implementation_sources: Mapping[str, str],
    implementation: str,
    output_root: Path,
) -> dict[str, Any]:
    seed = int(detail["seed"])
    source_config = campaign._config_from_mapping(source_record["configuration"])
    lifecycle = campaign._lifecycle_config(source_config, seed)
    training, _batches, training_hash, minibatch_hash = stage0.protocol_material(
        task, lifecycle
    )
    if len(training) != config.train_samples:
        raise RuntimeError("registered C3 training sample count changed")
    if (
        training_hash != detail["protocol"]["training_data_sha256"]
        or minibatch_hash != detail["protocol"]["minibatch_sha256"]
    ):
        raise RuntimeError(f"source C3 training cohort changed for seed {seed}")
    fingerprint = _fingerprint(
        config, implementation, detail, training_hash, evaluation_hashes
    )
    existing = _existing_detail(output_root, seed, fingerprint)
    if existing is not None:
        existing["_reused"] = True
        return existing

    output_dir = _cell_dir(output_root, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    system = _load_system(detail, device)
    state_before = stage0._state_digest(system)
    if state_before != detail["initialization"]["final_system_sha256"]:
        raise RuntimeError(f"source C3 state digest changed for seed {seed}")

    training_state = _extract_state(system, training, task, config, device)
    permutation = deterministic_target_permutation(
        len(training), seed, config.shuffle_seed
    )
    maps = _fit_maps(training_state, training, permutation, config)
    arrays = _fit_arrays(maps)
    records: dict[str, Any] = {}
    maximum_replay_error = 0.0
    maximum_decoder_error = 0.0
    for regime in REGIMES:
        dataset = evaluation[regime]
        state = _extract_state(system, dataset, task, config, device)
        target_posterior = dataset.target_posteriors.to(torch.float64)
        target = dataset.target.to(torch.float64)
        source_posterior = state["source_posterior"].to(torch.float64)
        source_metrics = _posterior_metrics(source_posterior, dataset)
        stored_error = (
            _maximum_stored_metric_error(source_metrics, detail["task_metrics"][regime])
            if config.evaluation_latents == 1_024
            else None
        )
        maximum_replay_error = max(
            maximum_replay_error, stored_error if stored_error is not None else 0.0
        )

        exact_scalar_carrier, _ = preflight.analytic_carrier(
            dataset.tokens, dataset.calibration
        )
        exact_temporal = preflight.temporal_prediction(exact_scalar_carrier)
        exact_posterior = interval.fixed_interval_posterior(
            exact_temporal, task.phase_bins
        )
        decoder_reference = interval.fixed_interval_posterior(target, task.phase_bins)
        decoder_error = float((decoder_reference - target_posterior).abs().max())
        maximum_decoder_error = max(maximum_decoder_error, decoder_error)

        recal_true = interval.apply_ridge_map(
            maps["recalibration_true"], state["source_posterior_mean"]
        ).reshape(-1)
        recal_shuffled = interval.apply_ridge_map(
            maps["recalibration_shuffled"], state["source_posterior_mean"]
        ).reshape(-1)
        typed_true = interval.apply_ridge_map(
            maps["typed_true"], state["normalized_final_residual"]
        ).reshape(-1)
        typed_shuffled = interval.apply_ridge_map(
            maps["typed_shuffled"], state["normalized_final_residual"]
        ).reshape(-1)
        untyped_true = interval.apply_ridge_map(
            maps["untyped_true"], state["normalized_final_residual"]
        )
        untyped_shuffled = interval.apply_ridge_map(
            maps["untyped_shuffled"], state["normalized_final_residual"]
        )

        posterior_by_arm = {
            "output_scalar_recalibration": {
                "true": interval.fixed_interval_posterior(recal_true, task.phase_bins),
                "shuffled": interval.fixed_interval_posterior(
                    recal_shuffled, task.phase_bins
                ),
                "scalar_true": recal_true,
                "scalar_shuffled": recal_shuffled,
            },
            "typed_final_readout": {
                "true": interval.fixed_interval_posterior(typed_true, task.phase_bins),
                "shuffled": interval.fixed_interval_posterior(
                    typed_shuffled, task.phase_bins
                ),
                "scalar_true": typed_true,
                "scalar_shuffled": typed_shuffled,
            },
            "untyped_final_readout": {
                "true": torch.softmax(untyped_true, -1),
                "shuffled": torch.softmax(untyped_shuffled, -1),
            },
        }
        arms: dict[str, Any] = {}
        for arm, values in posterior_by_arm.items():
            true_metrics = _posterior_metrics(values["true"], dataset)
            shuffled_metrics = _posterior_metrics(values["shuffled"], dataset)
            cell: dict[str, Any] = {
                "true": true_metrics,
                "shuffled": shuffled_metrics,
                "task_gate_true": _task_gate(true_metrics, regime, config),
                "task_gate_shuffled": _task_gate(
                    shuffled_metrics, regime, config
                ),
            }
            if "scalar_true" in values:
                cell["scalar_true"] = _scalar_metrics(values["scalar_true"], target)
                cell["scalar_shuffled"] = _scalar_metrics(
                    values["scalar_shuffled"], target
                )
            arms[arm] = cell

        bypass_metrics = _posterior_metrics(exact_posterior, dataset)
        records[regime] = {
            "dataset_sha256": evaluation_hashes[regime],
            "source": {
                "metrics": source_metrics,
                "stored_metric_replay_error": stored_error,
                "stored_metrics_same_cohort": config.evaluation_latents == 1_024,
                "task_gate": _task_gate(source_metrics, regime, config),
            },
            "exact_interval_decoder": {
                "maximum_absolute_posterior_error": decoder_error,
            },
            "exact_temporal_bypass": {
                "metrics": bypass_metrics,
                "scalar_metrics": _scalar_metrics(exact_temporal, target),
                "task_gate": _task_gate(bypass_metrics, regime, config),
            },
            "arms": arms,
        }
        prefix = f"{regime}__"
        arrays.update(
            {
                f"{prefix}target": target.numpy(),
                f"{prefix}target_posterior": target_posterior.numpy(),
                f"{prefix}source_posterior": source_posterior.numpy(),
                f"{prefix}exact_temporal_scalar": exact_temporal.numpy(),
                f"{prefix}exact_temporal_posterior": exact_posterior.numpy(),
                f"{prefix}recalibration_true_scalar": recal_true.numpy(),
                f"{prefix}recalibration_shuffled_scalar": recal_shuffled.numpy(),
                f"{prefix}typed_true_scalar": typed_true.numpy(),
                f"{prefix}typed_shuffled_scalar": typed_shuffled.numpy(),
                f"{prefix}untyped_true_posterior": posterior_by_arm[
                    "untyped_final_readout"
                ]["true"].numpy(),
                f"{prefix}untyped_shuffled_posterior": posterior_by_arm[
                    "untyped_final_readout"
                ]["shuffled"].numpy(),
            }
        )

    arm_gates: dict[str, Any] = {}
    for arm in FIT_ARMS:
        true_task = all(records[regime]["arms"][arm]["task_gate_true"] for regime in REGIMES)
        shuffled_task = all(
            records[regime]["arms"][arm]["task_gate_shuffled"]
            for regime in REGIMES
        )
        if arm in ("output_scalar_recalibration", "typed_final_readout"):
            true_pass = true_task and all(
                records[regime]["arms"][arm]["scalar_true"]["cosine_pearson"]
                >= config.scalar_correlation_minimum
                for regime in REGIMES
            )
            shuffled_pass = shuffled_task and all(
                records[regime]["arms"][arm]["scalar_shuffled"]["cosine_pearson"]
                >= config.scalar_correlation_minimum
                for regime in REGIMES
            )
        else:
            true_pass = true_task
            shuffled_pass = shuffled_task
        arm_gates[arm] = {
            "true_both_shifts": bool(true_pass),
            "shuffled_both_shifts": bool(shuffled_pass),
        }

    bypass_pass = all(
        records[regime]["exact_temporal_bypass"]["task_gate"]
        for regime in REGIMES
    )
    state_after = stage0._state_digest(system)
    state_unchanged = state_before == state_after
    fits = {name: interval.ridge_record(fitted) for name, fitted in maps.items()}
    finite = _finite({"records": records, "fits": fits}) and all(
        np.isfinite(value).all() for value in arrays.values()
    )
    validity = bool(
        state_unchanged
        and maximum_replay_error <= config.replay_tolerance
        and maximum_decoder_error <= config.replay_tolerance
        and finite
    )

    diagnostics_path = output_dir / "diagnostics.npz"
    _write_npz(diagnostics_path, arrays)
    with np.load(diagnostics_path, allow_pickle=False) as reloaded:
        diagnostics_reload = set(reloaded.files) == set(arrays) and all(
            np.array_equal(reloaded[name], value) for name, value in arrays.items()
        )
    validity = validity and diagnostics_reload
    elapsed = time.perf_counter() - started
    peak_cuda = (
        float(torch.cuda.max_memory_allocated(device)) / 1e9
        if device.type == "cuda"
        else 0.0
    )
    result_path = output_dir / "result.json"
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "status": "completed" if validity else "invalid",
        "completed_at": _utc_now(),
        "seed": seed,
        "configuration": _json_config(config),
        "implementation_sources": dict(implementation_sources),
        "implementation_sha256": implementation,
        "scientific_fingerprint": fingerprint,
        "source": {
            "campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "result": detail["_result_path"],
            "result_sha256": detail["_result_sha256"],
            "checkpoint": detail["artifacts"]["checkpoint"],
            "checkpoint_sha256": detail["artifacts"]["checkpoint_sha256"],
            "frontend_checkpoint": detail["artifacts"]["frontend_checkpoint"],
            "frontend_checkpoint_sha256": detail["artifacts"][
                "frontend_checkpoint_sha256"
            ],
            "known_natural_task_pass": detail["gates"]["natural_task"],
        },
        "data": {
            "training_data_sha256": training_hash,
            "minibatch_sha256": minibatch_hash,
            "evaluation_hashes": dict(evaluation_hashes),
            "training_examples": len(training),
            "evaluation_latents": config.evaluation_latents,
        },
        "fits": fits,
        "regimes": records,
        "gates": {
            "validity": validity,
            "source_state_unchanged": state_unchanged,
            "source_replay": maximum_replay_error <= config.replay_tolerance,
            "exact_decoder": maximum_decoder_error <= config.replay_tolerance,
            "diagnostics_reload": diagnostics_reload,
            "exact_temporal_bypass": bypass_pass,
            "arms": arm_gates,
        },
        "measurements": {
            "maximum_source_replay_error": maximum_replay_error,
            "maximum_exact_decoder_error": maximum_decoder_error,
            "optimizer_steps": 0,
            "trained_model_parameters": 0,
            "fitted_readout_parameters": sum(
                fitted.coefficients.numel() for fitted in maps.values()
            ),
            "state_sha256_before": state_before,
            "state_sha256_after": state_after,
            "wall_seconds": elapsed,
            "peak_cuda_allocation_gb": peak_cuda,
        },
        "artifacts": {
            "result": str(result_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": _sha256(diagnostics_path),
        },
        "method_boundaries": [
            "The source failures are outcome-known; fitted readout outcomes are prospective.",
            "Ridge targets use the sealed source training cohort and are supervised interfaces.",
            "Failure establishes only non-accessibility through the registered affine families.",
            "The exact temporal bypass is a positive localization control, not a TinyLLM repair.",
        ],
    }
    _write_json(result_path, result)
    del system
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def aggregate_details(
    details: Sequence[Mapping[str, Any]], config: C3ReadoutConfig
) -> dict[str, Any]:
    valid = len(details) == len(config.seeds) and all(
        detail.get("gates", {}).get("validity") is True for detail in details
    )
    bypass = all(
        detail.get("gates", {}).get("exact_temporal_bypass") is True
        for detail in details
    )
    arms = {}
    for arm in FIT_ARMS:
        true_count = sum(
            bool(detail["gates"]["arms"][arm]["true_both_shifts"])
            for detail in details
        )
        shuffled_count = sum(
            bool(detail["gates"]["arms"][arm]["shuffled_both_shifts"])
            for detail in details
        )
        arms[arm] = {
            "true_pass_count": true_count,
            "shuffled_pass_count": shuffled_count,
            "population_pass": (
                true_count >= config.required_seed_passes
                and shuffled_count <= config.shuffled_seed_pass_ceiling
            ),
        }
    specificity = all(
        record["shuffled_pass_count"] <= config.shuffled_seed_pass_ceiling
        for record in arms.values()
    )
    typed = arms["typed_final_readout"]["population_pass"]
    recalibration = arms["output_scalar_recalibration"]["population_pass"]
    untyped = arms["untyped_final_readout"]["population_pass"]
    if not valid:
        classification = "invalid"
    elif not bypass:
        classification = "invalid_exact_temporal_positive_control"
    elif not specificity:
        classification = "invalid_target_shuffle_specificity"
    elif typed and recalibration:
        classification = "natural_output_calibration_sufficient"
    elif typed:
        classification = (
            "frozen_continuation_typed_state_sufficient_answer_rows_inadequate"
        )
    elif untyped:
        classification = "frozen_continuation_task_state_untyped_only"
    else:
        classification = (
            "analytic_sensor_valid_frozen_continuation_not_affinely_typed"
        )
    return {
        "valid": valid,
        "exact_temporal_bypass_pass": bypass,
        "specificity_pass": specificity,
        "arms": arms,
        "classification": classification,
        "primary_hypothesis_pass": bool(valid and bypass and specificity and typed),
    }


def run_campaign(config: C3ReadoutConfig, output: Path) -> dict[str, Any]:
    source_record, source_details = _source_campaign(Path(config.source_root))
    task = stage0.C3TaskConfig(**source_record["task"])
    evaluation = analysis.build_task_datasets(task, config.evaluation_latents)
    evaluation_hashes = {
        regime: analysis.dataset_hash(dataset)
        for regime, dataset in evaluation.items()
    }
    if not config.allow_underpowered:
        expected = source_record["dataset_contract"]["task_hashes"]
        if evaluation_hashes != expected:
            raise RuntimeError("primary C3 final cohort hashes changed")
    sources = _implementation_sources()
    implementation = _implementation_digest(sources)
    details = []
    reused = 0
    for seed in config.seeds:
        detail = run_cell(
            config,
            source_record,
            source_details[seed],
            task,
            evaluation,
            evaluation_hashes,
            sources,
            implementation,
            output,
        )
        reused += int(detail.pop("_reused", False))
        details.append(detail)
    aggregate = aggregate_details(details, config)
    result_paths = [_cell_dir(output, seed) / "result.json" for seed in config.seeds]
    diagnostic_paths = [
        _cell_dir(output, seed) / "diagnostics.npz" for seed in config.seeds
    ]
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "status": "completed" if aggregate["valid"] else "invalid",
        "completed_at": _utc_now(),
        "configuration": _json_config(config),
        "source": {
            "root": config.source_root,
            "campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "result_manifest_sha256": SOURCE_RESULT_MANIFEST_SHA256,
            "artifact_manifest_sha256": SOURCE_ARTIFACT_MANIFEST_SHA256,
        },
        "task": asdict(task),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": config.device,
        },
        "implementation_sources": sources,
        "implementation_sha256": implementation,
        "campaign_fingerprint": _json_hash(
            {
                "configuration": _json_config(config),
                "implementation": implementation,
                "source_campaign": SOURCE_CAMPAIGN_SHA256,
                "evaluation_hashes": evaluation_hashes,
            }
        ),
        "evaluation_hashes": evaluation_hashes,
        "summary": {
            "requested": len(config.seeds),
            "completed": len(details),
            "failed": sum(detail["status"] != "completed" for detail in details),
            "reused": reused,
            "optimizer_steps": 0,
            "trained_model_parameters": 0,
        },
        "aggregates": aggregate,
        "results": [
            {
                "seed": detail["seed"],
                "status": detail["status"],
                "result": detail["artifacts"]["result"],
                "scientific_fingerprint": detail["scientific_fingerprint"],
                "gates": detail["gates"],
            }
            for detail in details
        ],
        "result_manifest_sha256": _manifest_sha256(result_paths),
        "diagnostics_manifest_sha256": _manifest_sha256(diagnostic_paths),
        "method_boundaries": [
            "No TinyLLM or front-end parameter is trained or changed.",
            "The three small readout families are supervised diagnostic interfaces.",
            "The stopped raw and learned-C3 arms remain untested.",
        ],
    }
    _write_json(output / "campaign_results.json", bundle)
    return bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--shakedown", action="store_true")
    mode.add_argument("--execute-primary", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = C3ReadoutConfig(device=args.device)
    if args.shakedown:
        config = replace(
            config,
            seeds=(7,),
            evaluation_latents=64,
            batch_size=64,
            allow_underpowered=True,
        )
        output = args.output or Path(
            "/tmp/tinyllm-c3-temporal-continuation-readout-shakedown"
        )
    else:
        output = args.output or Path(
            "data/experiments/tinyllm_c3_temporal_continuation_readout/"
            "20260811_d6_preregistered"
        )
    result = run_campaign(config, output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "classification": result["aggregates"]["classification"],
                "summary": result["summary"],
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
