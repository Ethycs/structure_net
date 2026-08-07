#!/usr/bin/env python3
"""Recalibrate only the frozen TinyLLM answer head under orientation noise."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
import experiments.structure_net.tinyllm_calibration_orientation_noise as orientation
import experiments.structure_net.tinyllm_invariant_frontend_causal as invariant
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-noisy-reference-readout-recalibration.v1"
HYPOTHESIS_ID = "tinyllm-noisy-reference-readout-recalibration-v1"
SOURCE_SCHEMA = orientation.SCHEMA_VERSION
SOURCE_HYPOTHESIS = orientation.HYPOTHESIS_ID
SOURCE_CAMPAIGN_SHA256 = (
    "876af062f9d0bdd365e2f1ffbb959d9ff2e5e4e277321b510bbe6a956456877f"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "990975365c7f0c468c3895029c1a87a98fa14d5a28853a1fc445070769218c70"
)
SOURCE_NOISE_SHA256 = (
    "b8d959169f2f249d635037a362c70522e514ace13d3bf656a91bdaea99ef43e7"
)
CALIBRATED_CAMPAIGN_SHA256 = orientation.SOURCE_CAMPAIGN_SHA256
CALIBRATED_IMPLEMENTATION_SHA256 = orientation.SOURCE_IMPLEMENTATION_SHA256
PRIMARY_SEEDS = orientation.PRIMARY_SEEDS
ARMS = orientation.ARMS
REGIMES = orientation.REGIMES
PRIMARY_REGIMES = orientation.PRIMARY_REGIMES
FIT_CONDITIONS = ("clean", "noisy")
EVALUATION_CONDITIONS = ("clean", "noisy")
SHUFFLE_SEED_OFFSET = 8_100_019


@dataclass(frozen=True)
class ReadoutRecalibrationConfig:
    source_orientation_root: str = (
        "data/experiments/tinyllm_calibration_orientation_noise/"
        "20260807_d8_preregistered"
    )
    seeds: tuple[int, ...] = PRIMARY_SEEDS
    noise_sigma: float = 0.035
    analysis_seed: int = 19_083
    train_samples: int = 2_048
    validation_samples: int = 512
    test_samples: int = 1_024
    activation_batch_size: int = 256
    head_ridge_ratio: float = 0.01
    accuracy_loss_ceiling: float = 0.03
    shuffled_accuracy_ceiling: float = 0.20
    required_seed_passes: int = 4
    replay_tolerance: float = 2e-6
    device: str = "cuda:1"
    allow_underpowered: bool = False
    partial_outcome_relaunch: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.noise_sigma <= 0.0:
            raise ValueError("noise sigma must be positive")
        if self.head_ridge_ratio <= 0.0:
            raise ValueError("head ridge ratio must be positive")
        if min(self.train_samples, self.validation_samples, self.test_samples) < 2:
            raise ValueError("analysis cohorts must be nontrivial")
        if self.allow_underpowered and self.partial_outcome_relaunch:
            raise ValueError(
                "systems-only and partial-outcome evidence roles are exclusive"
            )
        if not self.allow_underpowered:
            if self.seeds != PRIMARY_SEEDS:
                raise ValueError("primary readout seeds are fixed")
            if self.noise_sigma != 0.035:
                raise ValueError("primary readout noise is fixed at 0.035 radians")
            if self.analysis_seed != 19_083:
                raise ValueError("primary readout analysis seed is fixed")
            if (
                self.train_samples,
                self.validation_samples,
                self.test_samples,
                self.activation_batch_size,
            ) != (2_048, 512, 1_024, 256):
                raise ValueError("primary readout cohort protocol is fixed")
            if self.head_ridge_ratio != 0.01:
                raise ValueError("primary readout ridge ratio is fixed")
            if self.required_seed_passes != 4:
                raise ValueError("primary readout population gate is four of five")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    temporary.replace(path)


def _source_digests() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "orientation_noise": Path(orientation.__file__),
        "calibrated_frontend": Path(calibrated.__file__),
        "invariant_frontend": Path(invariant.__file__),
        "tinyllm_model": Path(inspect.getfile(TinyLLMModel)),
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _implementation_digest(digests: Mapping[str, str] | None = None) -> str:
    material = dict(digests or _source_digests())
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _evidence_role(config: ReadoutRecalibrationConfig) -> str:
    if config.allow_underpowered:
        return "systems_lifecycle_only_not_quality_evidence"
    if config.partial_outcome_relaunch:
        return "preregistered_mixed_pedigree_partial_analytic_exposure"
    return "preregistered_frozen_system_readout_intervention"


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _tensor_sha256(*values: torch.Tensor) -> str:
    return invariant._tensor_digest(*values)


def _source_orientation_config(
    config: ReadoutRecalibrationConfig,
    calibrated_root: str,
) -> orientation.OrientationNoiseConfig:
    if config.allow_underpowered:
        return orientation.OrientationNoiseConfig(
            source_root=calibrated_root,
            seeds=config.seeds,
            sigmas=(0.0, config.noise_sigma),
            analysis_seed=config.analysis_seed,
            probe_train_samples=config.train_samples,
            probe_validation_samples=config.validation_samples,
            probe_test_samples=config.test_samples,
            activation_batch_size=config.activation_batch_size,
            required_seed_passes=config.required_seed_passes,
            device=config.device,
            allow_underpowered=True,
        )
    return orientation.OrientationNoiseConfig(
        source_root=calibrated_root,
        device=config.device,
    )


def _load_sources(
    config: ReadoutRecalibrationConfig,
) -> tuple[
    dict[str, Any],
    Path,
    dict[tuple[str, int], Mapping[str, Any]],
    CircleTaskConfig,
    calibrated.CalibratedFrontendConfig,
]:
    campaign_path = Path(config.source_orientation_root) / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    noise_path = Path(campaign.get("artifacts", {}).get("noise_arrays", ""))
    if (
        _sha256(campaign_path) != SOURCE_CAMPAIGN_SHA256
        or campaign.get("schema_version") != SOURCE_SCHEMA
        or campaign.get("hypothesis_id") != SOURCE_HYPOTHESIS
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256") != SOURCE_IMPLEMENTATION_SHA256
        or campaign.get("aggregates", {}).get("classification")
        != "reference_precision_critical"
        or campaign.get("aggregates", {}).get("clean_replay_contract") is not True
        or campaign.get("artifacts", {}).get("noise_arrays_sha256")
        != SOURCE_NOISE_SHA256
        or not noise_path.is_file()
        or _sha256(noise_path) != SOURCE_NOISE_SHA256
    ):
        raise ValueError(f"invalid orientation-noise source campaign {campaign_path}")
    source_configuration = campaign["configuration"]
    expected_protocol = {
        "analysis_seed": config.analysis_seed,
        "probe_train_samples": config.train_samples,
        "probe_validation_samples": config.validation_samples,
        "probe_test_samples": config.test_samples,
        "activation_batch_size": config.activation_batch_size,
    }
    if any(source_configuration.get(key) != value for key, value in expected_protocol.items()):
        raise ValueError("orientation source cohort protocol changed")
    if config.noise_sigma not in source_configuration.get("sigmas", []):
        raise ValueError("requested noise sigma absent from orientation source")
    entries = {
        (item["condition"], int(item["seed"])): item
        for item in campaign.get("results", [])
    }
    expected = {(condition, seed) for condition in ARMS for seed in config.seeds}
    if not expected.issubset(entries):
        raise ValueError("orientation source is missing a requested checkpoint")

    orientation_config = _source_orientation_config(
        config, str(source_configuration["source_root"])
    )
    calibrated_campaign, calibrated_path, task, source_config = (
        orientation._load_source_campaign(orientation_config)
    )
    if (
        _sha256(calibrated_path) != CALIBRATED_CAMPAIGN_SHA256
        or calibrated_campaign.get("implementation_sha256")
        != CALIBRATED_IMPLEMENTATION_SHA256
    ):
        raise ValueError("calibrated source identity changed")
    return campaign, campaign_path, entries, task, source_config


def _datasets(
    config: ReadoutRecalibrationConfig,
    task: CircleTaskConfig,
    source_campaign: Mapping[str, Any],
) -> tuple[
    dict[str, calibrated.CalibratedDataset],
    dict[str, calibrated.CalibratedDataset],
    dict[str, str],
    dict[str, Any],
]:
    datasets = {
        name: calibrated.generate_calibrated_dataset(
            task,
            sample_count=(
                config.train_samples
                if name == "train"
                else config.validation_samples
                if name == "validation"
                else config.test_samples
            ),
            seed=seed,
            regime=regime,
            shuffle=True,
        )
        for name, (seed, regime, _) in orientation.SPLIT_SPECS.items()
    }
    hashes = {name: orientation._dataset_hash(value) for name, value in datasets.items()}
    if hashes != source_campaign.get("dataset_hashes"):
        raise ValueError("regenerated readout cohorts do not replay the source")
    base_noise = {
        name: orientation.build_base_noise(dataset, orientation.SPLIT_SPECS[name][0])
        for name, dataset in datasets.items()
    }
    noise_path = Path(source_campaign["artifacts"]["noise_arrays"])
    with np.load(noise_path, allow_pickle=False) as archive:
        for name, dataset in datasets.items():
            if not np.array_equal(
                archive[f"{name}__fiber_id"],
                dataset.paired.fiber.fiber_id.numpy(),
            ) or not np.array_equal(
                archive[f"{name}__base_noise"], base_noise[name].numpy()
            ):
                raise ValueError(f"stored orientation noise mismatch for {name}")
    noisy: dict[str, calibrated.CalibratedDataset] = {}
    audits: dict[str, Any] = {}
    for name, dataset in datasets.items():
        packet, audit = orientation.corrupt_orientation(
            dataset.calibration, base_noise[name], config.noise_sigma
        )
        pair_error = orientation.pair_shared_error(
            config.noise_sigma * base_noise[name], dataset.paired.fiber.fiber_id
        )
        audit["maximum_pair_shared_error"] = pair_error
        noisy[name] = calibrated.CalibratedDataset(
            paired=dataset.paired,
            calibration=packet,
        )
        audits[name] = audit
    contracts = {
        "dataset_hash_replay_contract": True,
        "noise_array_replay_contract": True,
        "pair_shared_noise_contract": all(
            value["maximum_pair_shared_error"] == 0.0 for value in audits.values()
        ),
        "unit_norm_contract": all(
            value["maximum_unit_norm_error"] <= 1e-6 for value in audits.values()
        ),
        "formula_replay_contract": all(
            value["maximum_formula_replay_error"] <= 1e-7
            for value in audits.values()
        ),
        "noise_audits": audits,
    }
    return datasets, noisy, hashes, contracts


@torch.no_grad()
def _normalized_features(
    system: calibrated.CalibratedTinyLLM,
    dataset: calibrated.CalibratedDataset,
    task: CircleTaskConfig,
    source_config: calibrated.CalibratedFrontendConfig,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    analysis_config = orientation._analysis_config(
        orientation.OrientationNoiseConfig(
            source_root="unused",
            seeds=(7,),
            sigmas=(0.0,),
            probe_train_samples=2,
            probe_validation_samples=2,
            probe_test_samples=2,
            activation_batch_size=batch_size,
            required_seed_passes=1,
            device=str(device),
            allow_underpowered=True,
        ),
        source_config,
    )
    raw = calibrated.extract_features(system, dataset, task, analysis_config, device)[
        "full"
    ]
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    normalized: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    for start in range(0, len(raw), batch_size):
        residual = torch.from_numpy(raw[start : start + batch_size]).to(device)
        value = system.model.transformer["ln_f"](residual)
        logits = system.model.lm_head(value).index_select(-1, answer_ids)
        normalized.append(value.detach().cpu().float())
        probabilities.append(torch.softmax(logits, -1).detach().cpu().float())
    return {
        "normalized": torch.cat(normalized),
        "frozen_probabilities": torch.cat(probabilities),
    }


def _centered_log_targets(target: torch.Tensor) -> torch.Tensor:
    value = target.double().clamp_min(1e-12).log()
    return value - value.mean(1, keepdim=True)


def fit_linear_readout(
    features: torch.Tensor,
    target: torch.Tensor,
    frozen_weight: torch.Tensor,
    ridge_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit the declared no-bias answer head by a deterministic ridge solve."""
    x = features.double()
    y = _centered_log_targets(target)
    w0 = frozen_weight.double()
    gram = x.T @ x
    scale = float(torch.trace(gram) / gram.shape[0])
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("degenerate frozen readout design")
    penalty = float(ridge_ratio) * scale
    system = gram + penalty * torch.eye(gram.shape[0], dtype=torch.float64)
    right = x.T @ y + penalty * w0.T
    weight = torch.linalg.solve(system, right).T.contiguous()
    logits = x @ weight.T
    probabilities = torch.softmax(logits, -1)
    loss = float(-(target.double() * probabilities.clamp_min(1e-12).log()).sum(1).mean())
    displacement = torch.linalg.vector_norm(weight - w0)
    baseline_norm = torch.linalg.vector_norm(w0).clamp_min(1e-12)
    return weight.float(), {
        "ridge_ratio": float(ridge_ratio),
        "ridge_penalty": penalty,
        "fit_target_cross_entropy": loss,
        "weight_displacement_frobenius": float(displacement),
        "relative_weight_displacement": float(displacement / baseline_norm),
    }


def fit_scalar_interval(
    frozen_probabilities: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    centers = torch.linspace(-1.0, 1.0, target.shape[1], dtype=torch.float64)
    source = frozen_probabilities.double() @ centers
    destination = target.double() @ centers
    design = torch.stack((source, torch.ones_like(source)), 1)
    solution = torch.linalg.lstsq(design, destination[:, None]).solution[:, 0]
    prediction = design @ solution
    return {
        "slope": float(solution[0]),
        "intercept": float(solution[1]),
        "fit_coordinate_rmse": float(
            torch.sqrt((prediction - destination).square().mean())
        ),
    }


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.double() - left.double().mean()
    b = right.double() - right.double().mean()
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denominator) <= 0.0:
        return 0.0
    return float((a @ b) / denominator)


def evaluate_probabilities(
    probabilities: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    probabilities = probabilities.double()
    target = target.double()
    centers = torch.linspace(-1.0, 1.0, target.shape[1], dtype=torch.float64)
    predicted_bins = probabilities.argmax(1)
    target_bins = target.argmax(1)
    predicted_coordinate = probabilities @ centers
    target_coordinate = target @ centers
    return {
        "exact_bin_accuracy": float((predicted_bins == target_bins).double().mean()),
        "mean_target_cross_entropy": float(
            -(target * probabilities.clamp_min(1e-12).log()).sum(1).mean()
        ),
        "expected_cosine_correlation": _correlation(
            predicted_coordinate, target_coordinate
        ),
        "expected_cosine_rmse": float(
            torch.sqrt((predicted_coordinate - target_coordinate).square().mean())
        ),
    }


def evaluate_linear_readout(
    weight: torch.Tensor,
    features: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    probabilities = torch.softmax(features.double() @ weight.double().T, -1)
    return evaluate_probabilities(probabilities, target)


def evaluate_scalar_interval(
    parameters: Mapping[str, float],
    frozen_probabilities: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    centers = torch.linspace(-1.0, 1.0, target.shape[1], dtype=torch.float64)
    source = frozen_probabilities.double() @ centers
    prediction = float(parameters["slope"]) * source + float(parameters["intercept"])
    nearest = (prediction[:, None] - centers[None, :]).abs().argmin(1)
    target_bins = target.argmax(1)
    target_coordinate = target.double() @ centers
    return {
        "exact_bin_accuracy": float((nearest == target_bins).double().mean()),
        "expected_cosine_correlation": _correlation(prediction, target_coordinate),
        "expected_cosine_rmse": float(
            torch.sqrt((prediction - target_coordinate).square().mean())
        ),
    }


def _accuracy_gate(
    evaluations: Mapping[str, Mapping[str, float]],
    clean_baseline: Mapping[str, float],
    ceiling: float,
) -> bool:
    return all(
        clean_baseline[regime] - evaluations[regime]["exact_bin_accuracy"] <= ceiling
        for regime in PRIMARY_REGIMES
    )


def classify_campaign(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    valid: bool,
    required: int,
) -> str:
    if not valid:
        return "invalid"
    scalar = {
        arm: int(arms[arm]["scalar_primary_pass_count"]) >= required for arm in ARMS
    }
    linear = {
        arm: int(arms[arm]["linear_primary_pass_count"]) >= required for arm in ARMS
    }
    noisy_only = {
        arm: int(arms[arm]["linear_noisy_repair_pass_count"]) >= required
        for arm in ARMS
    }
    if all(scalar.values()) and all(linear.values()):
        return "scalar_interval_calibration_sufficient"
    if all(linear.values()):
        return "linear_readout_recalibration_sufficient"
    if sum(linear.values()) == 1:
        return "arm_stratified_readout_repair"
    if any(noisy_only.values()):
        return "condition_specific_readout_tradeoff"
    return "probe_readout_gap_persists"


def _fingerprint(
    config: ReadoutRecalibrationConfig,
    implementation: str,
    condition: str,
    seed: int,
    provenance: Mapping[str, Any],
    dataset_hashes: Mapping[str, str],
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "condition": condition,
        "seed": seed,
        "implementation_sha256": implementation,
        "source": dict(provenance),
        "dataset_hashes": dict(dataset_hashes),
        "source_noise_sha256": SOURCE_NOISE_SHA256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reusable_result(
    path: Path,
    fingerprint: str,
    implementation: str,
    condition: str,
    seed: int,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    readout_path = Path(value.get("artifacts", {}).get("readouts", ""))
    if not (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("scientific_fingerprint") == fingerprint
        and value.get("implementation_sha256") == implementation
        and value.get("condition") == condition
        and int(value.get("seed", -1)) == seed
        and readout_path.is_file()
        and _sha256(readout_path) == value.get("artifacts", {}).get("readouts_sha256")
    ):
        raise ValueError(f"incompatible completed readout result {path}")
    return value


def _campaign_reusable(value: Mapping[str, Any], implementation: str) -> bool:
    return bool(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("implementation_sha256") == implementation
        and all(
            Path(item["path"]).is_file()
            and _sha256(Path(item["path"])) == item["result_sha256"]
            and Path(item["readouts"]).is_file()
            and _sha256(Path(item["readouts"])) == item["readouts_sha256"]
            for item in value.get("results", [])
        )
    )


def _source_result(
    entry: Mapping[str, Any],
    condition: str,
    seed: int,
) -> tuple[dict[str, Any], Path]:
    path = Path(entry["path"])
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        _sha256(path) != entry.get("result_sha256")
        or value.get("schema_version") != SOURCE_SCHEMA
        or value.get("hypothesis_id") != SOURCE_HYPOTHESIS
        or value.get("condition") != condition
        or int(value.get("seed", -1)) != seed
        or value.get("scientific_fingerprint") != entry.get("scientific_fingerprint")
        or any(not value["levels"][index]["representation_gate"] for index in (0, 1))
    ):
        raise ValueError(f"invalid source orientation result {path}")
    return value, path


def run_campaign(
    config: ReadoutRecalibrationConfig,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    source_digests = _source_digests()
    implementation = _implementation_digest(source_digests)
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if _campaign_reusable(existing, implementation):
            print("campaign already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed campaign {campaign_path}")

    source_campaign, source_path, source_entries, task, source_config = _load_sources(
        config
    )
    clean_datasets, noisy_datasets, dataset_hashes, corruption_contracts = _datasets(
        config, task, source_campaign
    )
    corruption_valid = all(
        value
        for name, value in corruption_contracts.items()
        if name.endswith("_contract")
    )
    dataset_by_condition = {"clean": clean_datasets, "noisy": noisy_datasets}
    results: list[dict[str, Any]] = []
    reused = 0
    calibrated_root = Path(source_campaign["configuration"]["source_root"])

    for condition in ARMS:
        for seed in config.seeds:
            source_detail, source_result_path = _source_result(
                source_entries[(condition, seed)], condition, seed
            )
            system, provenance = orientation._load_system(
                calibrated_root,
                condition,
                seed,
                task,
                source_config,
                device,
            )
            provenance = {
                **provenance,
                "orientation_result": str(source_result_path),
                "orientation_result_sha256": _sha256(source_result_path),
                "orientation_scientific_fingerprint": source_detail[
                    "scientific_fingerprint"
                ],
            }
            fingerprint = _fingerprint(
                config,
                implementation,
                condition,
                seed,
                provenance,
                dataset_hashes,
            )
            result_path = output / "runs" / condition / f"seed_{seed}" / "result.json"
            existing = _reusable_result(
                result_path, fingerprint, implementation, condition, seed
            )
            if existing is not None:
                results.append(existing)
                reused += 1
                print(f"reuse {condition} seed {seed}", flush=True)
                del system
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            seed_started = time.perf_counter()
            state_before = calibrated._module_digest(system)
            feature_sets: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
            for evaluation_condition in EVALUATION_CONDITIONS:
                feature_sets[evaluation_condition] = {
                    name: _normalized_features(
                        system,
                        dataset,
                        task,
                        source_config,
                        config.activation_batch_size,
                        device,
                    )
                    for name, dataset in dataset_by_condition[
                        evaluation_condition
                    ].items()
                }
            answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long)
            frozen_weight = (
                system.model.lm_head.weight.detach().cpu().index_select(0, answer_ids)
            )
            clean_baseline: dict[str, float] = {}
            frozen: dict[str, Any] = {}
            maximum_replay_error = 0.0
            for evaluation_condition in EVALUATION_CONDITIONS:
                frozen[evaluation_condition] = {}
                level_index = 0 if evaluation_condition == "clean" else 1
                for regime in REGIMES:
                    metrics = evaluate_probabilities(
                        feature_sets[evaluation_condition][regime][
                            "frozen_probabilities"
                        ],
                        dataset_by_condition[evaluation_condition][regime]
                        .paired.circle.target_posteriors,
                    )
                    frozen[evaluation_condition][regime] = metrics
                    source_metrics = source_detail["levels"][level_index][
                        "task_metrics"
                    ][regime]
                    maximum_replay_error = max(
                        maximum_replay_error,
                        abs(
                            metrics["exact_bin_accuracy"]
                            - float(source_metrics["exact_bin_accuracy"])
                        ),
                        abs(
                            metrics["mean_target_cross_entropy"]
                            - float(source_metrics["mean_target_cross_entropy"])
                        ),
                    )
                    if evaluation_condition == "clean" and regime in PRIMARY_REGIMES:
                        clean_baseline[regime] = metrics["exact_bin_accuracy"]

            fits: dict[str, Any] = {}
            readout_payload: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "condition": condition,
                "seed": seed,
                "scientific_fingerprint": fingerprint,
                "implementation_sha256": implementation,
                "linear_weights": {},
                "scalar_parameters": {},
            }
            for fit_condition in FIT_CONDITIONS:
                train_features = feature_sets[fit_condition]["train"]
                train_target = dataset_by_condition[fit_condition][
                    "train"
                ].paired.circle.target_posteriors
                weight, fit_metrics = fit_linear_readout(
                    train_features["normalized"],
                    train_target,
                    frozen_weight,
                    config.head_ridge_ratio,
                )
                scalar = fit_scalar_interval(
                    train_features["frozen_probabilities"], train_target
                )
                evaluations: dict[str, Any] = {}
                for evaluation_condition in EVALUATION_CONDITIONS:
                    evaluations[evaluation_condition] = {"linear": {}, "scalar": {}}
                    for regime in REGIMES:
                        target = dataset_by_condition[evaluation_condition][regime]
                        target_posteriors = target.paired.circle.target_posteriors
                        linear_metrics = evaluate_linear_readout(
                            weight,
                            feature_sets[evaluation_condition][regime]["normalized"],
                            target_posteriors,
                        )
                        scalar_metrics = evaluate_scalar_interval(
                            scalar,
                            feature_sets[evaluation_condition][regime][
                                "frozen_probabilities"
                            ],
                            target_posteriors,
                        )
                        for metrics in (linear_metrics, scalar_metrics):
                            if regime in PRIMARY_REGIMES:
                                metrics["accuracy_loss_from_frozen_clean"] = (
                                    clean_baseline[regime]
                                    - metrics["exact_bin_accuracy"]
                                )
                        evaluations[evaluation_condition]["linear"][regime] = (
                            linear_metrics
                        )
                        evaluations[evaluation_condition]["scalar"][regime] = (
                            scalar_metrics
                        )
                fits[fit_condition] = {
                    "linear_fit": fit_metrics,
                    "scalar_fit": scalar,
                    "evaluations": evaluations,
                }
                readout_payload["linear_weights"][fit_condition] = weight
                readout_payload["scalar_parameters"][fit_condition] = scalar

            generator = torch.Generator(device="cpu").manual_seed(
                SHUFFLE_SEED_OFFSET + seed
            )
            permutation = torch.randperm(config.train_samples, generator=generator)
            noisy_train = feature_sets["noisy"]["train"]
            noisy_target = noisy_datasets["train"].paired.circle.target_posteriors
            shuffled_weight, shuffled_fit = fit_linear_readout(
                noisy_train["normalized"],
                noisy_target[permutation],
                frozen_weight,
                config.head_ridge_ratio,
            )
            shuffled_evaluations = {
                regime: evaluate_linear_readout(
                    shuffled_weight,
                    feature_sets["noisy"][regime]["normalized"],
                    noisy_datasets[regime].paired.circle.target_posteriors,
                )
                for regime in REGIMES
            }
            readout_payload["linear_weights"]["target_shuffled_noisy"] = (
                shuffled_weight
            )
            readout_payload["target_shuffle_sha256"] = _tensor_sha256(permutation)

            clean_linear_control = _accuracy_gate(
                fits["clean"]["evaluations"]["clean"]["linear"],
                clean_baseline,
                config.accuracy_loss_ceiling,
            )
            noisy_linear_repair = _accuracy_gate(
                fits["noisy"]["evaluations"]["noisy"]["linear"],
                clean_baseline,
                config.accuracy_loss_ceiling,
            )
            noisy_linear_clean_compatibility = _accuracy_gate(
                fits["noisy"]["evaluations"]["clean"]["linear"],
                clean_baseline,
                config.accuracy_loss_ceiling,
            )
            noisy_scalar_repair = _accuracy_gate(
                fits["noisy"]["evaluations"]["noisy"]["scalar"],
                clean_baseline,
                config.accuracy_loss_ceiling,
            )
            noisy_scalar_clean_compatibility = _accuracy_gate(
                fits["noisy"]["evaluations"]["clean"]["scalar"],
                clean_baseline,
                config.accuracy_loss_ceiling,
            )
            shuffled_negative_control = all(
                shuffled_evaluations[regime]["exact_bin_accuracy"]
                <= config.shuffled_accuracy_ceiling
                for regime in PRIMARY_REGIMES
            )
            inherited_representation = all(
                source_detail["levels"][index]["representation_gate"]
                for index in (0, 1)
            )
            source_replay = maximum_replay_error <= config.replay_tolerance
            state_after = calibrated._module_digest(system)
            system_unchanged = state_after == state_before == provenance[
                "system_state_sha256"
            ]
            finite = _finite(frozen) and _finite(fits) and _finite(
                shuffled_evaluations
            )
            validity = bool(
                corruption_valid
                and source_replay
                and inherited_representation
                and system_unchanged
                and finite
            )
            linear_primary = bool(
                validity
                and noisy_linear_repair
                and noisy_linear_clean_compatibility
            )
            scalar_primary = bool(
                validity and noisy_scalar_repair and noisy_scalar_clean_compatibility
            )
            gates = {
                "source_metric_replay_contract": source_replay,
                "inherited_representation_contract": inherited_representation,
                "system_state_unchanged_contract": system_unchanged,
                "finite_numerical_contract": finite,
                "clean_linear_control": clean_linear_control,
                "noisy_linear_repair": noisy_linear_repair,
                "noisy_linear_clean_compatibility": (
                    noisy_linear_clean_compatibility
                ),
                "linear_primary": linear_primary,
                "noisy_scalar_repair": noisy_scalar_repair,
                "noisy_scalar_clean_compatibility": (
                    noisy_scalar_clean_compatibility
                ),
                "scalar_primary": scalar_primary,
                "target_shuffled_negative_control": shuffled_negative_control,
                "validity_contract": validity,
            }
            readout_path = result_path.parent / "readouts.pt"
            _write_torch(readout_path, readout_payload)
            readout_sha256 = _sha256(readout_path)
            result = {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "experiment_id": (
                    f"tinyllm-noisy-reference-readout-{condition}-seed{seed}"
                ),
                "status": "completed",
                "evidence_role": _evidence_role(config),
                "condition": condition,
                "seed": seed,
                "configuration": asdict(config),
                "scientific_fingerprint": fingerprint,
                "implementation_sha256": implementation,
                "provenance": provenance,
                "dataset_hashes": dataset_hashes,
                "source_noise_sha256": SOURCE_NOISE_SHA256,
                "frozen": frozen,
                "clean_baseline_accuracy": clean_baseline,
                "fits": fits,
                "target_shuffled": {
                    "fit": shuffled_fit,
                    "permutation_sha256": _tensor_sha256(permutation),
                    "evaluations": shuffled_evaluations,
                },
                "maximum_source_metric_replay_error": maximum_replay_error,
                "gates": gates,
                "artifacts": {
                    "readouts": str(readout_path),
                    "readouts_sha256": readout_sha256,
                },
                "analysis_seconds": time.perf_counter() - seed_started,
                "completed_at": _utc_now(),
            }
            if _implementation_digest() != implementation:
                raise RuntimeError("readout implementation changed during campaign")
            _write_json(result_path, result)
            results.append(result)
            print(
                f"{condition} seed {seed}: frozen-replay={source_replay} "
                f"linear={linear_primary} scalar={scalar_primary}",
                flush=True,
            )
            del system, feature_sets
            if device.type == "cuda":
                torch.cuda.empty_cache()

    arms: dict[str, Any] = {}
    for condition in ARMS:
        selected = [item for item in results if item["condition"] == condition]
        arms[condition] = {
            "requested": len(selected),
            "frozen_noisy_complete_pass_count": sum(
                bool(item["provenance"]) and bool(
                    json.loads(Path(item["provenance"]["orientation_result"]).read_text())[
                        "levels"
                    ][1]["complete_gate"]
                )
                for item in selected
            ),
            "clean_linear_control_pass_count": sum(
                item["gates"]["clean_linear_control"] for item in selected
            ),
            "linear_noisy_repair_pass_count": sum(
                item["gates"]["noisy_linear_repair"] for item in selected
            ),
            "linear_clean_compatibility_pass_count": sum(
                item["gates"]["noisy_linear_clean_compatibility"]
                for item in selected
            ),
            "linear_primary_pass_count": sum(
                item["gates"]["linear_primary"] for item in selected
            ),
            "scalar_primary_pass_count": sum(
                item["gates"]["scalar_primary"] for item in selected
            ),
            "shuffled_negative_control_pass_count": sum(
                item["gates"]["target_shuffled_negative_control"]
                for item in selected
            ),
            "linear_pass_by_seed": {
                str(item["seed"]): bool(item["gates"]["linear_primary"])
                for item in selected
            },
            "scalar_pass_by_seed": {
                str(item["seed"]): bool(item["gates"]["scalar_primary"])
                for item in selected
            },
        }
    valid = bool(
        corruption_valid
        and all(item["gates"]["validity_contract"] for item in results)
        and all(
            arms[condition]["clean_linear_control_pass_count"]
            >= config.required_seed_passes
            for condition in ARMS
        )
        and all(
            arms[condition]["shuffled_negative_control_pass_count"] == len(config.seeds)
            for condition in ARMS
        )
    )
    classification = (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else classify_campaign(arms, valid=valid, required=config.required_seed_passes)
    )
    references = []
    for result in results:
        path = output / "runs" / result["condition"] / f"seed_{result['seed']}" / "result.json"
        references.append(
            {
                "experiment_id": result["experiment_id"],
                "condition": result["condition"],
                "seed": result["seed"],
                "scientific_fingerprint": result["scientific_fingerprint"],
                "path": str(path),
                "result_sha256": _sha256(path),
                "readouts": result["artifacts"]["readouts"],
                "readouts_sha256": result["artifacts"]["readouts_sha256"],
                "gates": result["gates"],
            }
        )
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "configuration": asdict(config),
        "implementation_sha256": implementation,
        "source_digests": source_digests,
        "provenance": {
            "orientation_campaign": str(source_path),
            "orientation_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "orientation_implementation_sha256": SOURCE_IMPLEMENTATION_SHA256,
            "orientation_noise_sha256": SOURCE_NOISE_SHA256,
            "calibrated_campaign_sha256": CALIBRATED_CAMPAIGN_SHA256,
            "calibrated_implementation_sha256": (
                CALIBRATED_IMPLEMENTATION_SHA256
            ),
        },
        "dataset_hashes": dataset_hashes,
        "corruption_contracts": corruption_contracts,
        "results": references,
        "aggregates": {
            "valid": valid,
            "classification": classification,
            "required_seed_passes": config.required_seed_passes,
            "arm_evidence_roles": {
                condition: (
                    "post_outcome_corrective_replication_evidence"
                    if config.partial_outcome_relaunch
                    and condition == "analytic_calibrated"
                    else "preregistered_unseen_arm_evidence"
                )
                for condition in ARMS
            },
            "arms": arms,
        },
        "summary": {
            "requested": len(ARMS) * len(config.seeds),
            "completed": len(results),
            "failed": 0,
            "reused": reused,
            "trained_models": 0,
            "trained_frontends": 0,
            "fitted_linear_readouts": len(results) * 3,
            "fitted_scalar_calibrators": len(results) * 2,
            "new_representation_probes": 0,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "method_boundaries": [
            "Only the detached final answer-token readout is refit; TinyLLM, front ends, layer norm, and residual states remain frozen.",
            "The primary error level is the predecessor's first nonzero orientation sigma, not an estimated real-instrument distribution.",
            "The inherited nonlinear representation probes establish decodability but are not part of the refitted task interface.",
            "The selected five checkpoints are replication units, not a random architecture population.",
            "The mixed-pedigree relaunch exposes the analytic arm as corrective replication while preserving the previously unseen learned-arm outcome.",
        ],
        "analysis_seconds": time.perf_counter() - started,
        "completed_at": _utc_now(),
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "*" / "seed_*" / "result.json"),
            "readouts": str(output / "runs" / "*" / "seed_*" / "readouts.pt"),
        },
    }
    if _implementation_digest() != implementation:
        raise RuntimeError("readout implementation changed before campaign write")
    _write_json(campaign_path, campaign)
    return campaign


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_noisy_reference_readout_recalibration/"
            "20260807_d8_preregistered_cuda1_persistent_mixed_pedigree"
        ),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seeds", default="7,17,29,41,53")
    parser.add_argument("--allow-underpowered", action="store_true")
    parser.add_argument("--partial-outcome-relaunch", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    config = ReadoutRecalibrationConfig(
        seeds=seeds,
        required_seed_passes=(len(seeds) if args.allow_underpowered else 4),
        device=args.device,
        allow_underpowered=args.allow_underpowered,
        partial_outcome_relaunch=args.partial_outcome_relaunch,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2), flush=True)
    print(campaign["artifacts"]["campaign"], flush=True)


if __name__ == "__main__":
    main()
