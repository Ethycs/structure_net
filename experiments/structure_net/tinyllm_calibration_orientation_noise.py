#!/usr/bin/env python3
"""Titrate orientation-reference noise on frozen calibrated TinyLLM systems."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
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
import experiments.structure_net.tinyllm_conditional_branch_depth_scan as probes
import experiments.structure_net.tinyllm_invariant_frontend_causal as invariant
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-calibrated-orientation-noise.v1"
HYPOTHESIS_ID = "tinyllm-calibrated-orientation-noise-radius-v1"
SOURCE_SCHEMA = calibrated.SCHEMA_VERSION
SOURCE_HYPOTHESIS = calibrated.HYPOTHESIS_ID
SOURCE_CAMPAIGN_SHA256 = (
    "80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77"
)
PRIMARY_SEEDS = (7, 17, 29, 41, 53)
ARMS = ("analytic_calibrated", "learned_calibrated_equivariant")
CUTS = ("frontend", "full")
REGIMES = ("in_distribution", "composition", "extrapolation")
PRIMARY_REGIMES = ("composition", "extrapolation")
LOCKED_SIGMAS = (0.0, 0.035, 0.087, 0.175, 0.349, 0.524, 0.785)
NOISE_SEED_OFFSET = 6_700_019
SPLIT_SPECS = {
    "train": (19_184, "interpolation", 2_048),
    "validation": (19_294, "interpolation", 512),
    "in_distribution": (19_390, "interpolation", 1_024),
    "composition": (20_399, "composition", 1_024),
    "extrapolation": (21_408, "extrapolation", 1_024),
}


@dataclass(frozen=True)
class OrientationNoiseConfig:
    source_root: str = (
        "data/experiments/tinyllm_calibrated_frontend_causal/"
        "20260806_d8_preregistered"
    )
    seeds: tuple[int, ...] = PRIMARY_SEEDS
    sigmas: tuple[float, ...] = LOCKED_SIGMAS
    analysis_seed: int = 19_083
    probe_train_samples: int = 2_048
    probe_validation_samples: int = 512
    probe_test_samples: int = 1_024
    activation_batch_size: int = 256
    probe_batch_size: int = 256
    probe_steps: int = 240
    probe_width: int = 128
    validation_interval: int = 20
    early_stopping_patience: int = 5
    cosine_floor: float = 0.90
    branch_accuracy_ceiling: float = 0.55
    log_loss_gain_ceiling: float = 0.02
    accuracy_loss_ceiling: float = 0.03
    unit_norm_tolerance: float = 1e-6
    required_seed_passes: int = 4
    device: str = "cuda:1"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if not self.sigmas or tuple(sorted(self.sigmas)) != self.sigmas:
            raise ValueError("orientation-noise levels must be non-empty and ordered")
        if any(value < 0.0 for value in self.sigmas):
            raise ValueError("orientation-noise levels must be nonnegative")
        if not self.allow_underpowered:
            if self.seeds != PRIMARY_SEEDS:
                raise ValueError("primary orientation-noise seeds are fixed")
            if self.sigmas != LOCKED_SIGMAS:
                raise ValueError("primary orientation-noise grid is fixed")
            if self.analysis_seed != 19_083:
                raise ValueError("primary analysis seed is fixed")
            if (
                self.probe_train_samples,
                self.probe_validation_samples,
                self.probe_test_samples,
                self.probe_steps,
                self.probe_width,
            ) != (2_048, 512, 1_024, 240, 128):
                raise ValueError("primary probe protocol is fixed")
            if self.required_seed_passes != 4:
                raise ValueError("primary population gate is fixed at four of five")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _source_digests() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "calibrated_frontend": Path(calibrated.__file__),
        "conditional_probes": Path(probes.__file__),
        "invariant_frontend": Path(invariant.__file__),
        "tinyllm_model": Path(inspect.getfile(TinyLLMModel)),
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _implementation_digest(digests: Mapping[str, str] | None = None) -> str:
    material = dict(digests or _source_digests())
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _evidence_role(config: OrientationNoiseConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_frozen_checkpoint_causal_titration"
    )


def _dataset_hash(dataset: calibrated.CalibratedDataset) -> str:
    paired = dataset.paired
    return invariant._tensor_digest(
        paired.circle.input_ids,
        paired.circle.target_posteriors,
        paired.circle.target_bins,
        paired.fiber.cosine,
        paired.fiber.branch,
        paired.fiber.fiber_id,
        dataset.calibration,
    )


def build_base_noise(
    dataset: calibrated.CalibratedDataset, dataset_seed: int
) -> torch.Tensor:
    """Return one deterministic normal deviate per exact cosine fiber."""
    fiber_ids = dataset.paired.fiber.fiber_id.long()
    unique, inverse = torch.unique(fiber_ids, sorted=True, return_inverse=True)
    generator = np.random.default_rng(NOISE_SEED_OFFSET + dataset_seed)
    values = torch.from_numpy(generator.standard_normal(len(unique))).double()
    return values[inverse]


def corrupt_orientation(
    calibration: torch.Tensor, base_noise: torch.Tensor, sigma: float
) -> tuple[torch.Tensor, dict[str, float]]:
    if len(calibration) != len(base_noise):
        raise ValueError("calibration and orientation noise lengths differ")
    value = calibration.clone()
    if float(sigma) == 0.0:
        norm_error = float(
            (torch.linalg.vector_norm(value[:, :2].double(), dim=1) - 1.0)
            .abs()
            .max()
        )
        return value, {
            "sigma_radians": 0.0,
            "maximum_unit_norm_error": norm_error,
            "maximum_formula_replay_error": 0.0,
            "maximum_absolute_angular_error_radians": 0.0,
            "rms_angular_error_radians": 0.0,
        }
    angle = torch.atan2(calibration[:, 1].double(), calibration[:, 0].double())
    applied = float(sigma) * base_noise.double()
    expected = torch.stack((torch.cos(angle + applied), torch.sin(angle + applied)), 1)
    value[:, :2] = expected.to(value.dtype)
    norm_error = float((torch.linalg.vector_norm(value[:, :2].double(), dim=1) - 1.0).abs().max())
    replay_error = float((value[:, :2].double() - expected).abs().max())
    return value, {
        "sigma_radians": float(sigma),
        "maximum_unit_norm_error": norm_error,
        "maximum_formula_replay_error": replay_error,
        "maximum_absolute_angular_error_radians": float(applied.abs().max()),
        "rms_angular_error_radians": float(torch.sqrt(applied.square().mean())),
    }


def pair_shared_error(base_noise: torch.Tensor, fiber_ids: torch.Tensor) -> float:
    maximum = 0.0
    for fiber in torch.unique(fiber_ids, sorted=True):
        values = base_noise[fiber_ids == fiber]
        maximum = max(maximum, float((values - values[0]).abs().max()))
    return maximum


def representation_pass(metrics: Mapping[str, float], config: OrientationNoiseConfig) -> bool:
    return bool(
        metrics["cosine_pearson"] >= config.cosine_floor
        and metrics["balanced_accuracy"] <= config.branch_accuracy_ceiling
        and metrics["conditional_log_loss_gain_over_cosine_only"]
        <= config.log_loss_gain_ceiling
    )


def curve_summary(
    results: Sequence[Mapping[str, Any]],
    arm: str,
    config: OrientationNoiseConfig,
) -> dict[str, Any]:
    selected = [result for result in results if result["condition"] == arm]
    levels = []
    for index, sigma in enumerate(config.sigmas):
        passing = [bool(result["levels"][index]["complete_gate"]) for result in selected]
        count = sum(passing)
        levels.append(
            {
                "sigma_radians": float(sigma),
                "pass_count": count,
                "pass_by_seed": {
                    str(result["seed"]): passed for result, passed in zip(selected, passing)
                },
                "population_pass": count >= config.required_seed_passes,
            }
        )
    population = [bool(level["population_pass"]) for level in levels]
    nonmonotone = any(
        not population[index] and any(population[index + 1 :])
        for index in range(len(population) - 1)
    )
    radius = None
    if not nonmonotone:
        passed = [float(level["sigma_radians"]) for level in levels if level["population_pass"]]
        radius = max(passed) if passed else None
    return {
        "levels": levels,
        "nonmonotone": nonmonotone,
        "robustness_radius_radians": radius,
    }


def classify_campaign(
    arms: Mapping[str, Mapping[str, Any]], clean_valid: bool
) -> str:
    if not clean_valid:
        return "invalid"
    if any(bool(value["nonmonotone"]) for value in arms.values()):
        return "nonmonotone_no_single_radius"
    analytic = arms["analytic_calibrated"]["robustness_radius_radians"]
    learned = arms["learned_calibrated_equivariant"]["robustness_radius_radians"]
    if analytic is None or learned is None:
        return "invalid"
    first_nonzero = LOCKED_SIGMAS[1]
    if learned == 0.0 and analytic >= first_nonzero:
        return "learned_clean_only_equivariance"
    if learned == 0.0 and analytic == 0.0:
        return "reference_precision_critical"
    if 0.0 < learned < analytic:
        return "learned_brittle_relative_to_analytic"
    return "learned_matches_analytic_radius"


def _load_source_campaign(
    config: OrientationNoiseConfig,
) -> tuple[dict[str, Any], Path, CircleTaskConfig, calibrated.CalibratedFrontendConfig]:
    root = Path(config.source_root)
    path = root / "campaign_results.json"
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if (
        _sha256(path) != SOURCE_CAMPAIGN_SHA256
        or campaign.get("schema_version") != SOURCE_SCHEMA
        or campaign.get("hypothesis_id") != SOURCE_HYPOTHESIS
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256") != SOURCE_IMPLEMENTATION_SHA256
        or int(campaign.get("summary", {}).get("completed", -1)) != 15
        or int(campaign.get("summary", {}).get("failed", -1)) != 0
        or not all(campaign["aggregates"]["arms"][arm]["success"] for arm in ARMS)
    ):
        raise ValueError(f"invalid calibrated source campaign {path}")
    if _sha256(Path(calibrated.__file__)) != SOURCE_IMPLEMENTATION_SHA256:
        raise ValueError("calibrated source implementation no longer matches campaign")
    source_seeds = tuple(int(seed) for seed in campaign["configuration"]["seeds"])
    if not set(config.seeds).issubset(source_seeds):
        raise ValueError("requested checkpoint absent from calibrated source")
    task = CircleTaskConfig(**campaign["task_config"])
    source_config = calibrated._config_from_mapping(campaign["configuration"])
    return campaign, path, task, source_config


def _load_system(
    root: Path,
    condition: str,
    seed: int,
    task: CircleTaskConfig,
    source_config: calibrated.CalibratedFrontendConfig,
    device: torch.device,
) -> tuple[calibrated.CalibratedTinyLLM, dict[str, Any]]:
    directory = root / "runs" / condition / f"seed_{seed}"
    result_path = directory / "result.json"
    model_path = directory / "model.pt"
    frontend_path = directory / "frontend.pt"
    detail = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        detail.get("schema_version") != SOURCE_SCHEMA
        or detail.get("hypothesis_id") != SOURCE_HYPOTHESIS
        or detail.get("condition") != condition
        or int(detail.get("seed", -1)) != seed
        or detail.get("status") != "completed"
        or detail.get("implementation_sha256") != SOURCE_IMPLEMENTATION_SHA256
        or detail.get("training", {}).get("checkpoint") != str(model_path)
        or detail.get("training", {}).get("frontend_checkpoint") != str(frontend_path)
    ):
        raise ValueError(f"invalid calibrated source result {result_path}")
    model = TinyLLMModel.from_checkpoint(model_path, map_location="cpu")
    system = calibrated.CalibratedTinyLLM(model, condition, task, source_config)
    frontend = torch.load(frontend_path, map_location="cpu", weights_only=True)
    if frontend.get("condition") != condition or system.scalar_embedding is None:
        raise ValueError(f"invalid calibrated front-end checkpoint {frontend_path}")
    system.scalar_embedding.load_state_dict(frontend["scalar_embedding"])
    if condition == "learned_calibrated_equivariant":
        if system.encoder is None or frontend.get("encoder") is None:
            raise ValueError(f"missing learned calibrated encoder {frontend_path}")
        system.encoder.load_state_dict(frontend["encoder"])
    if calibrated._state_digest(system.model) != detail["training"]["final_model_state_sha256"]:
        raise ValueError(f"calibrated model state mismatch seed {seed}")
    if calibrated._module_digest(system) != detail["training"]["final_system_state_sha256"]:
        raise ValueError(f"calibrated system state mismatch seed {seed}")
    provenance = {
        "source_result": str(result_path),
        "source_result_sha256": _sha256(result_path),
        "scientific_fingerprint": detail["scientific_fingerprint"],
        "model_checkpoint": str(model_path),
        "model_checkpoint_sha256": _sha256(model_path),
        "frontend_checkpoint": str(frontend_path),
        "frontend_checkpoint_sha256": _sha256(frontend_path),
        "model_state_sha256": detail["training"]["final_model_state_sha256"],
        "system_state_sha256": detail["training"]["final_system_state_sha256"],
    }
    return system.to(device).eval(), provenance


def _analysis_config(
    config: OrientationNoiseConfig,
    source: calibrated.CalibratedFrontendConfig,
) -> calibrated.CalibratedFrontendConfig:
    return replace(
        source,
        seeds=config.seeds,
        analysis_seed=config.analysis_seed,
        probe_train_samples=config.probe_train_samples,
        probe_validation_samples=config.probe_validation_samples,
        probe_test_samples=config.probe_test_samples,
        activation_batch_size=config.activation_batch_size,
        probe_batch_size=config.probe_batch_size,
        probe_steps=config.probe_steps,
        probe_width=config.probe_width,
        validation_interval=config.validation_interval,
        early_stopping_patience=config.early_stopping_patience,
        allow_underpowered=config.allow_underpowered,
    )


@torch.no_grad()
def _task_metrics_from_full(
    system: calibrated.CalibratedTinyLLM,
    full_features: np.ndarray,
    dataset: calibrated.CalibratedDataset,
    task: CircleTaskConfig,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    probabilities = []
    for start in range(0, len(full_features), batch_size):
        value = torch.from_numpy(full_features[start : start + batch_size]).to(device)
        logits = calibrated._task_logits(system.model, value, answer_ids)
        probabilities.append(torch.softmax(logits, -1).cpu())
    predicted = torch.cat(probabilities)
    target = dataset.paired.circle.target_posteriors
    predicted_bins = predicted.argmax(1)
    target_bins = target.argmax(1)
    delta = (predicted_bins - target_bins).abs()
    circular = torch.minimum(delta, target.shape[1] - delta)
    return {
        "exact_bin_accuracy": float((predicted_bins == target_bins).float().mean()),
        "mean_circular_error_radians": float(
            circular.float().mean() * (2.0 * math.pi / target.shape[1])
        ),
        "mean_target_cross_entropy": float(
            -(target * predicted.clamp_min(1e-12).log()).sum(1).mean()
        ),
    }


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _fingerprint(
    config: OrientationNoiseConfig,
    implementation: str,
    condition: str,
    seed: int,
    provenance: Mapping[str, Any],
    dataset_hashes: Mapping[str, str],
    noise_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "condition": condition,
        "seed": seed,
        "implementation_sha256": implementation,
        "source_artifacts": dict(provenance),
        "dataset_hashes": dict(dataset_hashes),
        "noise_sha256": noise_sha256,
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
    if not (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("scientific_fingerprint") == fingerprint
        and value.get("implementation_sha256") == implementation
        and value.get("condition") == condition
        and int(value.get("seed", -1)) == seed
    ):
        raise ValueError(f"incompatible completed orientation-noise result {path}")
    return value


def _ensure_noise_arrays(
    path: Path,
    datasets: Mapping[str, calibrated.CalibratedDataset],
    base_noise: Mapping[str, torch.Tensor],
) -> str:
    arrays: dict[str, np.ndarray] = {}
    for name, dataset in datasets.items():
        arrays[f"{name}__fiber_id"] = dataset.paired.fiber.fiber_id.numpy()
        arrays[f"{name}__base_noise"] = base_noise[name].numpy()
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays) or any(
                not np.array_equal(archive[name], value) for name, value in arrays.items()
            ):
                raise ValueError(f"incompatible orientation-noise arrays {path}")
    else:
        _write_npz(path, arrays)
    return _sha256(path)


def _campaign_reusable(
    value: Mapping[str, Any], implementation: str
) -> bool:
    return bool(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("implementation_sha256") == implementation
        and all(
            Path(item["path"]).is_file()
            and _sha256(Path(item["path"])) == item["result_sha256"]
            for item in value.get("results", [])
        )
        and Path(value.get("artifacts", {}).get("noise_arrays", "")).is_file()
        and _sha256(Path(value["artifacts"]["noise_arrays"]))
        == value["artifacts"]["noise_arrays_sha256"]
    )


def run_campaign(config: OrientationNoiseConfig, output: Path) -> dict[str, Any]:
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

    source_campaign, source_path, task, source_config = _load_source_campaign(config)
    analysis_config = _analysis_config(config, source_config)
    probe_config = calibrated._analysis_config(analysis_config)
    datasets = {
        name: calibrated.generate_calibrated_dataset(
            task,
            sample_count=(
                config.probe_train_samples
                if name == "train"
                else config.probe_validation_samples
                if name == "validation"
                else config.probe_test_samples
            ),
            seed=seed,
            regime=regime,
            shuffle=True,
        )
        for name, (seed, regime, _) in SPLIT_SPECS.items()
    }
    dataset_hashes = {name: _dataset_hash(value) for name, value in datasets.items()}
    base_noise = {
        name: build_base_noise(dataset, SPLIT_SPECS[name][0])
        for name, dataset in datasets.items()
    }
    noise_path = output / "orientation_noise_arrays.npz"
    noise_sha256 = _ensure_noise_arrays(noise_path, datasets, base_noise)

    corrupted: dict[int, dict[str, calibrated.CalibratedDataset]] = {}
    noise_audits: dict[str, Any] = {}
    maximum_unit_norm_error = 0.0
    maximum_formula_replay_error = 0.0
    maximum_pair_error = 0.0
    maximum_clean_error = 0.0
    for level_index, sigma in enumerate(config.sigmas):
        corrupted[level_index] = {}
        noise_audits[str(level_index)] = {"sigma_radians": float(sigma), "splits": {}}
        for name, dataset in datasets.items():
            packet, audit = corrupt_orientation(dataset.calibration, base_noise[name], sigma)
            corrupted[level_index][name] = calibrated.CalibratedDataset(
                paired=dataset.paired, calibration=packet
            )
            pair_error = pair_shared_error(
                float(sigma) * base_noise[name], dataset.paired.fiber.fiber_id
            )
            clean_error = float((packet - dataset.calibration).abs().max()) if sigma == 0 else 0.0
            audit["maximum_pair_shared_error"] = pair_error
            audit["maximum_clean_packet_error"] = clean_error
            noise_audits[str(level_index)]["splits"][name] = audit
            maximum_unit_norm_error = max(maximum_unit_norm_error, audit["maximum_unit_norm_error"])
            maximum_formula_replay_error = max(
                maximum_formula_replay_error, audit["maximum_formula_replay_error"]
            )
            maximum_pair_error = max(maximum_pair_error, pair_error)
            maximum_clean_error = max(maximum_clean_error, clean_error)
    noise_contracts = {
        "unit_norm_contract": maximum_unit_norm_error <= config.unit_norm_tolerance,
        "formula_replay_contract": maximum_formula_replay_error <= 1e-7,
        "pair_shared_noise_contract": maximum_pair_error == 0.0,
        "clean_packet_identity_contract": maximum_clean_error == 0.0,
        "identical_cross_arm_checkpoint_noise_contract": True,
        "maximum_unit_norm_error": maximum_unit_norm_error,
        "maximum_formula_replay_error": maximum_formula_replay_error,
        "maximum_pair_shared_error": maximum_pair_error,
        "maximum_clean_packet_error": maximum_clean_error,
    }
    noise_valid = all(
        value for name, value in noise_contracts.items() if name.endswith("_contract")
    )

    results: list[dict[str, Any]] = []
    reused = 0
    root = Path(config.source_root)
    for condition_index, condition in enumerate(ARMS):
        for seed in config.seeds:
            system, provenance = _load_system(
                root, condition, seed, task, source_config, device
            )
            fingerprint = _fingerprint(
                config,
                implementation,
                condition,
                seed,
                provenance,
                dataset_hashes,
                noise_sha256,
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
            levels: list[dict[str, Any]] = []
            clean_accuracy: dict[str, float] | None = None
            feature_width_contract = True
            all_finite = True
            for level_index, sigma in enumerate(config.sigmas):
                level_datasets = corrupted[level_index]
                features = {
                    name: calibrated.extract_features(
                        system, dataset, task, analysis_config, device
                    )
                    for name, dataset in level_datasets.items()
                }
                feature_width_contract = bool(
                    feature_width_contract
                    and all(value["frontend"].shape[1] == 1 for value in features.values())
                    and all(
                        value["full"].shape[1] == system.model.config.n_embd
                        for value in features.values()
                    )
                )
                evaluation_fibers = {
                    name: level_datasets[name].paired.fiber for name in REGIMES
                }
                probe_seed = (
                    config.analysis_seed
                    + seed * 100_003
                    + condition_index * 10_000_019
                    + level_index * 1_009
                )
                null = probes.fit_cosine_only_null(
                    level_datasets["train"].paired.fiber,
                    level_datasets["validation"].paired.fiber,
                    evaluation_fibers,
                    probe_config,
                    device,
                    kind="nonlinear",
                    seed=probe_seed + 17,
                )
                cuts: dict[str, Any] = {}
                primary_cells = []
                for cut_index, cut in enumerate(CUTS):
                    probe = probes.fit_conditional_probe(
                        features["train"][cut],
                        features["validation"][cut],
                        {name: features[name][cut] for name in REGIMES},
                        level_datasets["train"].paired.fiber,
                        level_datasets["validation"].paired.fiber,
                        evaluation_fibers,
                        probe_config,
                        device,
                        kind="nonlinear",
                        seed=probe_seed + 101 + cut_index * 1_009,
                    )
                    probes._add_log_loss_gains(probe, null)
                    evaluations = probe["evaluations"]
                    cuts[cut] = {
                        "probe": probe,
                        "passes": {
                            name: representation_pass(evaluations[name], config)
                            for name in REGIMES
                        },
                    }
                    primary_cells.extend(
                        cuts[cut]["passes"][name] for name in PRIMARY_REGIMES
                    )
                task_metrics = {
                    name: _task_metrics_from_full(
                        system,
                        features[name]["full"],
                        level_datasets[name],
                        task,
                        config.activation_batch_size,
                        device,
                    )
                    for name in REGIMES
                }
                if clean_accuracy is None:
                    clean_accuracy = {
                        name: task_metrics[name]["exact_bin_accuracy"]
                        for name in PRIMARY_REGIMES
                    }
                accuracy_losses = {
                    name: clean_accuracy[name] - task_metrics[name]["exact_bin_accuracy"]
                    for name in PRIMARY_REGIMES
                }
                task_gate = all(
                    loss <= config.accuracy_loss_ceiling
                    for loss in accuracy_losses.values()
                )
                finite = _finite(cuts) and _finite(task_metrics)
                all_finite = bool(all_finite and finite)
                complete_gate = bool(
                    noise_valid
                    and feature_width_contract
                    and finite
                    and all(primary_cells)
                    and task_gate
                )
                levels.append(
                    {
                        "level_index": level_index,
                        "sigma_radians": float(sigma),
                        "sigma_degrees": float(sigma * 180.0 / math.pi),
                        "cuts": cuts,
                        "task_metrics": task_metrics,
                        "accuracy_loss_from_clean": accuracy_losses,
                        "representation_gate": bool(all(primary_cells)),
                        "task_gate": task_gate,
                        "finite_contract": finite,
                        "complete_gate": complete_gate,
                    }
                )
                print(
                    f"{condition} seed {seed} sigma {sigma:.3f}: "
                    f"{'pass' if complete_gate else 'fail'}",
                    flush=True,
                )
            clean_joint_pass = bool(levels[0]["complete_gate"])
            validity = bool(
                noise_valid and feature_width_contract and all_finite and clean_joint_pass
            )
            pass_pattern = [bool(level["complete_gate"]) for level in levels]
            nonmonotone = any(
                not pass_pattern[index] and any(pass_pattern[index + 1 :])
                for index in range(len(pass_pattern) - 1)
            )
            radius = None
            if not nonmonotone:
                passed = [
                    float(level["sigma_radians"])
                    for level in levels
                    if level["complete_gate"]
                ]
                radius = max(passed) if passed else None
            result = {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "experiment_id": f"tinyllm-calibration-orientation-noise-{condition}-seed{seed}",
                "status": "completed",
                "evidence_role": _evidence_role(config),
                "completed_at": _utc_now(),
                "condition": condition,
                "seed": seed,
                "configuration": asdict(config),
                "scientific_fingerprint": fingerprint,
                "implementation_sha256": implementation,
                "provenance": provenance,
                "dataset_hashes": dataset_hashes,
                "noise_arrays": str(noise_path),
                "noise_arrays_sha256": noise_sha256,
                "levels": levels,
                "gates": {
                    "source_provenance_contract": True,
                    "noise_contract": noise_valid,
                    "feature_width_contract": feature_width_contract,
                    "all_finite_contract": all_finite,
                    "clean_joint_pass": clean_joint_pass,
                    "validity_contract": validity,
                },
                "pass_pattern": pass_pattern,
                "nonmonotone": nonmonotone,
                "robustness_radius_radians": radius,
                "analysis_seconds": time.perf_counter() - seed_started,
            }
            _write_json(result_path, result)
            results.append(result)
            if _implementation_digest() != implementation:
                raise RuntimeError("orientation-noise implementation changed during campaign")
            del system
            if device.type == "cuda":
                torch.cuda.empty_cache()

    required = config.required_seed_passes if not config.allow_underpowered else len(config.seeds)
    aggregate_config = replace(config, required_seed_passes=required)
    arms = {arm: curve_summary(results, arm, aggregate_config) for arm in ARMS}
    clean_counts = {
        arm: arms[arm]["levels"][0]["pass_count"] for arm in ARMS
    }
    clean_valid = bool(
        noise_valid
        and all(clean_counts[arm] >= required for arm in ARMS)
        and all(result["gates"]["source_provenance_contract"] for result in results)
    )
    classification = (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else classify_campaign(arms, clean_valid)
    )
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "implementation_sha256": implementation,
        "source_digests": source_digests,
        "provenance": {
            "source_campaign": str(source_path),
            "source_campaign_sha256": _sha256(source_path),
            "source_implementation_sha256": source_campaign["implementation_sha256"],
        },
        "dataset_hashes": dataset_hashes,
        "noise_audits": noise_audits,
        "noise_contracts": noise_contracts,
        "summary": {
            "requested": len(ARMS) * len(config.seeds),
            "completed": len(results),
            "failed": 0,
            "reused": reused,
            "trained_models": 0,
            "trained_frontends": 0,
            "fitted_diagnostic_probe_sets": (
                len(results) * len(config.sigmas) * (1 + len(CUTS))
            ),
            "noise_levels": len(config.sigmas),
            "primary_representation_cells": (
                len(results) * len(config.sigmas) * len(CUTS) * len(PRIMARY_REGIMES)
            ),
        },
        "aggregates": {
            "required_seed_passes": required,
            "clean_replay_pass_counts": clean_counts,
            "clean_replay_contract": clean_valid,
            "arms": arms,
            "classification": classification,
        },
        "results": [
            {
                "experiment_id": result["experiment_id"],
                "condition": result["condition"],
                "seed": result["seed"],
                "scientific_fingerprint": result["scientific_fingerprint"],
                "gates": result["gates"],
                "pass_pattern": result["pass_pattern"],
                "nonmonotone": result["nonmonotone"],
                "robustness_radius_radians": result["robustness_radius_radians"],
                "path": str(
                    output
                    / "runs"
                    / result["condition"]
                    / f"seed_{result['seed']}"
                    / "result.json"
                ),
                "result_sha256": _sha256(
                    output
                    / "runs"
                    / result["condition"]
                    / f"seed_{result['seed']}"
                    / "result.json"
                ),
            }
            for result in results
        ],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        },
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "Only orientation-reference error is intervened on; all other calibration fields remain exact.",
            "Fresh nonlinear probes are diagnostic and are not part of the frozen mechanism.",
            "The five checkpoints were selected by the preceding successful calibrated campaign.",
            "The result does not estimate real instrument calibration cost or natural-language behavior.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "*" / "seed_*" / "result.json"),
            "noise_arrays": str(noise_path),
            "noise_arrays_sha256": noise_sha256,
        },
    }
    _write_json(campaign_path, campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seeds", default="7,17,29,41,53")
    parser.add_argument("--allow-underpowered", action="store_true")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_calibration_orientation_noise/"
            "20260807_d8_preregistered"
        ),
    )
    args = parser.parse_args()
    if args.shakedown:
        config = OrientationNoiseConfig(
            seeds=(7,),
            sigmas=(0.0, 0.087),
            probe_train_samples=128,
            probe_validation_samples=64,
            probe_test_samples=128,
            probe_steps=20,
            probe_width=32,
            required_seed_passes=1,
            device=args.device,
            allow_underpowered=True,
        )
    else:
        config = OrientationNoiseConfig(
            seeds=_ints(args.seeds),
            device=args.device,
            allow_underpowered=args.allow_underpowered,
        )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    print(args.output / "campaign_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
