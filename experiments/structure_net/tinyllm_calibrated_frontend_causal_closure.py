#!/usr/bin/env python3
"""Causally test orbit-barycenter closure in calibrated TinyLLM front ends."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Mapping

import numpy as np
import torch

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
import experiments.structure_net.tinyllm_invariant_frontend_causal as invariant
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-calibrated-frontend-causal-closure.v1"
HYPOTHESIS_ID = "tinyllm-calibrated-frontend-causal-closure-v1"
SOURCE_SCHEMA_VERSION = "nal.tinyllm-calibrated-frontend-causal.v1"
SOURCE_HYPOTHESIS_ID = "tinyllm-calibrated-reference-stable-cosine-quotient-v1"
SOURCE_CAMPAIGN_SHA256 = (
    "80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77"
)
SOURCE_RESULT_MANIFEST_SHA256 = (
    "34bf25feb896abc9b9e06386b474fc6795c94566a23cd97a06795435fba64d68"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-calibrated-frontend-causal-closure-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "d5ebcf907d4fe99132779b94d942e6ae4f4465f10a4f01c67f806e1dbb0d802f"
)
CONDITIONS = (
    "raw_calibrated",
    "analytic_calibrated",
    "learned_calibrated_equivariant",
)
PRIMARY_CONDITIONS = (
    "analytic_calibrated",
    "learned_calibrated_equivariant",
)
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
CUTS = (
    "pre_block",
    "block0_post_attention",
    "block0_post_mlp",
    "full",
)
TRANSITIONS = ("block0_attention", "block0_mlp")
SPLIT_SPECS = {
    "composition": (1_399, "composition", 1_024),
    "extrapolation": (2_408, "extrapolation", 1_024),
}
EXPECTED_DATASET_HASHES = {
    "composition": "b025623e23f534ca3670f49f1bafc1c3979d1ca834aec2223f9c3166be8f52a6",
    "extrapolation": "f2f1e8c2c06b38fbecf856f0679e9861d9fab3e1c2a41358e55043f4d309a214",
}
SHUFFLE_SEED = 81_027_401


@dataclass(frozen=True)
class CausalClosureConfig:
    source_root: str = (
        "data/experiments/tinyllm_calibrated_frontend_causal/"
        "20260806_d8_preregistered"
    )
    conditions: tuple[str, ...] = CONDITIONS
    seeds: tuple[int, ...] = SEEDS
    accuracy_loss_ceiling: float = 0.03
    circular_error_increase_ceiling: float = math.pi / 16.0
    cross_entropy_increase_ceiling: float = 0.10
    required_seed_passes: int = 4
    replay_tolerance: float = 2e-6
    state_identity_tolerance: float = 1e-7
    shuffle_seed: int = SHUFFLE_SEED
    batch_size: int = 256
    device: str = "cuda:1"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.conditions or set(self.conditions).difference(CONDITIONS):
            raise ValueError("unknown or empty calibrated condition")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("checkpoint seeds must be non-empty and distinct")
        if not 1 <= self.required_seed_passes <= len(self.seeds):
            raise ValueError("required seed passes is outside selected population")
        if self.batch_size < 1:
            raise ValueError("batch size must be positive")
        if min(
            self.accuracy_loss_ceiling,
            self.circular_error_increase_ceiling,
            self.cross_entropy_increase_ceiling,
        ) < 0.0:
            raise ValueError("task-sufficiency ceilings must be nonnegative")
        if not self.allow_underpowered:
            if self.conditions != CONDITIONS or self.seeds != SEEDS:
                raise ValueError("primary conditions and five checkpoints are fixed")
            if self.required_seed_passes != 4:
                raise ValueError("primary population gate is four of five")
            if self.shuffle_seed != SHUFFLE_SEED:
                raise ValueError("primary shuffle seed is fixed")
            if self.batch_size != 256:
                raise ValueError("primary continuation batch size is fixed")


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
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _json_config(config: CausalClosureConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _source_digests() -> dict[str, str]:
    if _sha256(PREREGISTRATION_PATH) != PREREGISTRATION_SHA256:
        raise ValueError("causal-closure preregistration changed")
    paths = {
        "runner": Path(__file__),
        "calibrated_frontend": Path(calibrated.__file__),
        "invariant_frontend": Path(invariant.__file__),
        "preregistration": PREREGISTRATION_PATH,
    }
    values = {name: _sha256(path) for name, path in paths.items()}
    if values["calibrated_frontend"] != SOURCE_IMPLEMENTATION_SHA256:
        raise ValueError("calibrated source implementation changed")
    return values


def _implementation_digest(digests: Mapping[str, str] | None = None) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(digests or _source_digests()),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _evidence_role(config: CausalClosureConfig) -> str:
    if config.allow_underpowered:
        return "systems_lifecycle_only_not_quality_evidence"
    return "preregistered_outcome_directed_frozen_activation_intervention"


def _source_result_manifest(root: Path, seeds: tuple[int, ...]) -> str:
    entries = []
    for condition in CONDITIONS:
        for seed in seeds:
            path = root / "runs" / condition / f"seed_{seed}" / "result.json"
            entries.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "result_sha256": _sha256(path),
                }
            )
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_source_campaign(
    config: CausalClosureConfig,
) -> tuple[
    dict[str, Any],
    Path,
    CircleTaskConfig,
    calibrated.CalibratedFrontendConfig,
    dict[tuple[str, int], dict[str, Any]],
]:
    root = Path(config.source_root)
    path = root / "campaign_results.json"
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if (
        _sha256(path) != SOURCE_CAMPAIGN_SHA256
        or campaign.get("schema_version") != SOURCE_SCHEMA_VERSION
        or campaign.get("hypothesis_id") != SOURCE_HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256") != SOURCE_IMPLEMENTATION_SHA256
        or campaign.get("summary")
        != {
            "requested": 15,
            "scheduled": 15,
            "completed": 15,
            "failed": 0,
            "reused": 0,
        }
        or any(
            campaign.get("aggregates", {})
            .get("arms", {})
            .get(condition, {})
            .get("success")
            is not True
            for condition in PRIMARY_CONDITIONS
        )
    ):
        raise ValueError(f"invalid calibrated source campaign {path}")
    source_seeds = tuple(int(seed) for seed in campaign["configuration"]["seeds"])
    if not set(config.seeds).issubset(source_seeds):
        raise ValueError("requested checkpoint absent from calibrated source")
    if _source_result_manifest(root, source_seeds) != SOURCE_RESULT_MANIFEST_SHA256:
        raise ValueError("calibrated source-result manifest changed")
    task = CircleTaskConfig(**campaign["task_config"])
    source_config = calibrated._config_from_mapping(campaign["configuration"])
    details: dict[tuple[str, int], dict[str, Any]] = {}
    for condition in config.conditions:
        for seed in config.seeds:
            result_path = root / "runs" / condition / f"seed_{seed}" / "result.json"
            detail = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                detail.get("schema_version") != SOURCE_SCHEMA_VERSION
                or detail.get("hypothesis_id") != SOURCE_HYPOTHESIS_ID
                or detail.get("status") != "completed"
                or detail.get("implementation_sha256")
                != SOURCE_IMPLEMENTATION_SHA256
                or detail.get("condition") != condition
                or int(detail.get("seed", -1)) != seed
            ):
                raise ValueError(f"invalid calibrated source result {result_path}")
            detail["_result_path"] = str(result_path)
            detail["_result_sha256"] = _sha256(result_path)
            details[(condition, seed)] = detail
    return campaign, path, task, source_config, details


def _load_system(
    root: Path,
    condition: str,
    seed: int,
    task: CircleTaskConfig,
    source_config: calibrated.CalibratedFrontendConfig,
    detail: Mapping[str, Any],
    device: torch.device,
) -> tuple[calibrated.CalibratedTinyLLM, dict[str, Any]]:
    directory = root / "runs" / condition / f"seed_{seed}"
    model_path = directory / "model.pt"
    frontend_path = directory / "frontend.pt"
    if (
        detail.get("training", {}).get("checkpoint") != str(model_path)
        or detail.get("training", {}).get("frontend_checkpoint")
        != str(frontend_path)
    ):
        raise ValueError(f"source paths changed for {condition} seed {seed}")
    model = TinyLLMModel.from_checkpoint(model_path, map_location="cpu")
    system = calibrated.CalibratedTinyLLM(
        model, condition, task, source_config
    )
    frontend = torch.load(frontend_path, map_location="cpu", weights_only=True)
    if frontend.get("condition") != condition:
        raise ValueError(f"front-end condition changed {frontend_path}")
    if condition == "raw_calibrated":
        if system.calibration_embedding is None:
            raise AssertionError("raw calibration embedding missing")
        system.calibration_embedding.load_state_dict(frontend["calibration_embedding"])
    else:
        if system.scalar_embedding is None:
            raise AssertionError("structured scalar embedding missing")
        system.scalar_embedding.load_state_dict(frontend["scalar_embedding"])
        if condition == "learned_calibrated_equivariant":
            if system.encoder is None or frontend.get("encoder") is None:
                raise AssertionError("learned calibrated encoder missing")
            system.encoder.load_state_dict(frontend["encoder"])
    model_state = calibrated._state_digest(system.model)
    system_state = calibrated._module_digest(system)
    if (
        model_state != detail["training"]["final_model_state_sha256"]
        or system_state != detail["training"]["final_system_state_sha256"]
    ):
        raise ValueError(f"source state changed for {condition} seed {seed}")
    provenance = {
        "source_result": detail["_result_path"],
        "source_result_sha256": detail["_result_sha256"],
        "scientific_fingerprint": detail["scientific_fingerprint"],
        "model_checkpoint": str(model_path),
        "model_checkpoint_sha256": _sha256(model_path),
        "frontend_checkpoint": str(frontend_path),
        "frontend_checkpoint_sha256": _sha256(frontend_path),
        "model_state_sha256": model_state,
        "system_state_sha256": system_state,
    }
    return system.to(device).eval(), provenance


def _preflight_sources(
    config: CausalClosureConfig,
    root: Path,
    task: CircleTaskConfig,
    source_config: calibrated.CalibratedFrontendConfig,
    details: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[dict[tuple[str, int], dict[str, Any]], str]:
    """Validate every selected frozen system before any intervention runs."""
    provenance: dict[tuple[str, int], dict[str, Any]] = {}
    for condition in config.conditions:
        for seed in config.seeds:
            system, item = _load_system(
                root,
                condition,
                seed,
                task,
                source_config,
                details[(condition, seed)],
                torch.device("cpu"),
            )
            provenance[(condition, seed)] = item
            del system
            gc.collect()
    material = {
        f"{condition}/seed_{seed}": item
        for (condition, seed), item in provenance.items()
    }
    digest = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return provenance, digest


def _datasets(task: CircleTaskConfig) -> dict[str, calibrated.CalibratedDataset]:
    return {
        regime: calibrated.generate_calibrated_dataset(
            task,
            sample_count=count,
            seed=seed,
            regime=generator_regime,
            shuffle=True,
        )
        for regime, (seed, generator_regime, count) in SPLIT_SPECS.items()
    }


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


def _initial_sequence(
    system: calibrated.CalibratedTinyLLM,
    input_ids: torch.Tensor,
    sensor: torch.Tensor,
    calibration_packet: torch.Tensor,
) -> torch.Tensor:
    feature = system.feature(sensor, calibration_packet)
    if system.condition == "raw_calibrated":
        if system.calibration_embedding is None:
            raise AssertionError("raw calibration embedding missing")
        prefix = system.model.transformer["wte"](input_ids[:, :-1])
        calibration_token = system.calibration_embedding(calibration_packet)[:, None]
        query = system.model.transformer["wte"](input_ids[:, -1:])
        value = torch.cat((prefix, calibration_token, query), dim=1)
    else:
        if system.scalar_embedding is None:
            raise AssertionError("structured scalar embedding missing")
        bos = system.model.transformer["wte"](input_ids[:, :1])
        scalar = system.scalar_embedding(feature)[:, None]
        query = system.model.transformer["wte"](input_ids[:, -1:])
        value = torch.cat((bos, scalar, query), dim=1)
    positions = torch.arange(value.shape[1], device=value.device)
    return value + system.model.transformer["wpe"](positions)


def _apply_attention(system: calibrated.CalibratedTinyLLM, value: torch.Tensor) -> torch.Tensor:
    block = system.model.transformer["h"][0]
    return value + block.attn(block.ln_1(value))


def _apply_mlp(system: calibrated.CalibratedTinyLLM, value: torch.Tensor) -> torch.Tensor:
    block = system.model.transformer["h"][0]
    return value + block.mlp(block.ln_2(value))


@torch.no_grad()
def _capture_dataset(
    system: calibrated.CalibratedTinyLLM,
    dataset: calibrated.CalibratedDataset,
    task: CircleTaskConfig,
    config: CausalClosureConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    sensor = calibrated.decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    captured = {cut: [] for cut in CUTS}
    posteriors = []
    for start in range(0, len(sensor), config.batch_size):
        stop = min(len(sensor), start + config.batch_size)
        value = _initial_sequence(
            system,
            dataset.paired.circle.input_ids[start:stop].to(device),
            sensor[start:stop].to(device),
            dataset.calibration[start:stop].to(device),
        )
        captured["pre_block"].append(value.cpu())
        value = _apply_attention(system, value)
        captured["block0_post_attention"].append(value.cpu())
        value = _apply_mlp(system, value)
        captured["block0_post_mlp"].append(value.cpu())
        for block in system.model.transformer["h"][1:]:
            value = block(value)
        captured["full"].append(value.cpu())
        logits = calibrated._task_logits(system.model, value[:, -1], answer_ids)
        posteriors.append(torch.softmax(logits, -1).double().cpu())
    return (
        {cut: torch.cat(parts).float() for cut, parts in captured.items()},
        torch.cat(posteriors),
    )


@torch.no_grad()
def _continue_from_cut(
    system: calibrated.CalibratedTinyLLM,
    cut: str,
    patched: torch.Tensor,
    task: CircleTaskConfig,
    config: CausalClosureConfig,
    device: torch.device,
) -> torch.Tensor:
    if cut not in CUTS:
        raise ValueError(f"unknown activation cut {cut}")
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    posteriors = []
    for start in range(0, len(patched), config.batch_size):
        stop = min(len(patched), start + config.batch_size)
        value = patched[start:stop].to(device)
        if cut == "pre_block":
            for block in system.model.transformer["h"]:
                value = block(value)
        elif cut == "block0_post_attention":
            value = _apply_mlp(system, value)
            for block in system.model.transformer["h"][1:]:
                value = block(value)
        elif cut == "block0_post_mlp":
            for block in system.model.transformer["h"][1:]:
                value = block(value)
        logits = calibrated._task_logits(system.model, value[:, -1], answer_ids)
        posteriors.append(torch.softmax(logits, -1).double().cpu())
    return torch.cat(posteriors)


def _fiber_mapping(
    dataset: calibrated.CalibratedDataset,
) -> tuple[torch.Tensor, torch.Tensor]:
    fibers = dataset.paired.fiber.fiber_id.long()
    unique, inverse, counts = torch.unique(
        fibers, sorted=True, return_inverse=True, return_counts=True
    )
    if len(unique) * 2 != len(fibers) or not torch.all(counts == 2):
        raise ValueError("causal closure requires exact two-sheet fibers")
    return unique, inverse


def orbit_average(
    values: torch.Tensor, fiber_inverse: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    fiber_count = int(fiber_inverse.max()) + 1
    flattened = values.reshape(len(values), -1).double()
    sums = torch.zeros(
        (fiber_count, flattened.shape[1]), dtype=torch.double
    )
    sums.index_add_(0, fiber_inverse, flattened)
    unique_average = (sums / 2.0).reshape((fiber_count,) + values.shape[1:]).float()
    return unique_average[fiber_inverse], unique_average


def shuffled_barycenters(
    unique_average: torch.Tensor,
    fiber_inverse: torch.Tensor,
    shift: int,
) -> torch.Tensor:
    if not 1 <= shift < len(unique_average):
        raise ValueError("shuffle shift must be a nonzero fiber rotation")
    return torch.roll(unique_average, shifts=shift, dims=0)[fiber_inverse]


def paired_state_identity(
    values: torch.Tensor, fiber_inverse: torch.Tensor
) -> float:
    maximum = 0.0
    for index in range(int(fiber_inverse.max()) + 1):
        rows = torch.nonzero(fiber_inverse == index, as_tuple=False).flatten()
        maximum = max(maximum, float((values[rows] - values[rows[:1]]).abs().max()))
    return maximum


def paired_state_geometry(
    values: torch.Tensor, fiber_inverse: torch.Tensor
) -> dict[str, float]:
    differences = []
    for index in range(int(fiber_inverse.max()) + 1):
        rows = torch.nonzero(fiber_inverse == index, as_tuple=False).flatten()
        difference = values[rows[0]].double() - values[rows[1]].double()
        differences.append(torch.sqrt(torch.mean(difference.square())))
    difference_rms = torch.stack(differences)
    state_rms = torch.sqrt(torch.mean(values.double().square())).clamp_min(1e-12)
    return {
        "mean_pair_rms": float(difference_rms.mean()),
        "maximum_pair_rms": float(difference_rms.max()),
        "mean_pair_rms_over_state_rms": float(difference_rms.mean() / state_rms),
    }


def posterior_metrics(
    posterior: torch.Tensor, dataset: calibrated.CalibratedDataset
) -> dict[str, float]:
    target = dataset.paired.circle.target_posteriors.double()
    predicted_bins = posterior.argmax(1)
    target_bins = target.argmax(1)
    delta = (predicted_bins - target_bins).abs()
    circular = torch.minimum(delta, target.shape[1] - delta)
    return {
        "exact_bin_accuracy": float((predicted_bins == target_bins).double().mean()),
        "mean_circular_error_radians": float(
            circular.double().mean() * (2.0 * math.pi / target.shape[1])
        ),
        "mean_target_cross_entropy": float(
            -(target * posterior.clamp_min(1e-12).log()).sum(1).mean()
        ),
    }


def task_sufficiency(
    metrics: Mapping[str, float],
    baseline: Mapping[str, float],
    config: CausalClosureConfig,
) -> tuple[bool, dict[str, Any]]:
    accuracy_loss = float(baseline["exact_bin_accuracy"] - metrics["exact_bin_accuracy"])
    circular_increase = float(
        metrics["mean_circular_error_radians"]
        - baseline["mean_circular_error_radians"]
    )
    cross_entropy_increase = float(
        metrics["mean_target_cross_entropy"]
        - baseline["mean_target_cross_entropy"]
    )
    gates = {
        "accuracy_loss": accuracy_loss,
        "accuracy_pass": accuracy_loss <= config.accuracy_loss_ceiling,
        "circular_error_increase": circular_increase,
        "circular_error_pass": circular_increase
        <= config.circular_error_increase_ceiling,
        "cross_entropy_increase": cross_entropy_increase,
        "cross_entropy_pass": cross_entropy_increase
        <= config.cross_entropy_increase_ceiling,
    }
    passed = bool(
        gates["accuracy_pass"]
        and gates["circular_error_pass"]
        and gates["cross_entropy_pass"]
    )
    return passed, gates


def jensen_shannon(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double().clamp_min(1e-12)
    right = right.double().clamp_min(1e-12)
    middle = 0.5 * (left + right)
    value = 0.5 * (
        (left * (left.log() - middle.log())).sum(1)
        + (right * (right.log() - middle.log())).sum(1)
    )
    return float(value.mean())


def _regime_name(propagated: bool, actual: bool) -> str:
    if not propagated and not actual:
        return "cover_required_after_sublayer"
    if not propagated and actual:
        return "invariant_synthesis"
    if propagated and actual:
        return "quotient_already_closed"
    return "quotient_corruption"


@torch.no_grad()
def analyze_regime(
    system: calibrated.CalibratedTinyLLM,
    dataset: calibrated.CalibratedDataset,
    task: CircleTaskConfig,
    config: CausalClosureConfig,
    device: torch.device,
    shuffle_shift: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    captured, baseline_posterior = _capture_dataset(
        system, dataset, task, config, device
    )
    baseline_metrics = posterior_metrics(baseline_posterior, dataset)
    _, inverse = _fiber_mapping(dataset)
    cut_records: dict[str, Any] = {}
    cut_posteriors: dict[str, dict[str, torch.Tensor]] = {}
    arrays: dict[str, np.ndarray] = {
        "fiber_id": dataset.paired.fiber.fiber_id.numpy(),
        "baseline_posterior": baseline_posterior.float().numpy(),
    }
    maximum_replay_error = 0.0
    maximum_state_identity_error = 0.0
    for cut in CUTS:
        replay = _continue_from_cut(system, cut, captured[cut], task, config, device)
        replay_error = float((replay - baseline_posterior).abs().max())
        maximum_replay_error = max(maximum_replay_error, replay_error)
        averaged, unique_average = orbit_average(captured[cut], inverse)
        shuffled = shuffled_barycenters(unique_average, inverse, shuffle_shift)
        identity_error = paired_state_identity(averaged, inverse)
        shuffle_identity_error = paired_state_identity(shuffled, inverse)
        maximum_state_identity_error = max(
            maximum_state_identity_error,
            identity_error,
            shuffle_identity_error,
        )
        orbit_posterior = _continue_from_cut(
            system, cut, averaged, task, config, device
        )
        shuffle_posterior = _continue_from_cut(
            system, cut, shuffled, task, config, device
        )
        orbit_metrics = posterior_metrics(orbit_posterior, dataset)
        shuffle_metrics = posterior_metrics(shuffle_posterior, dataset)
        orbit_pass, orbit_gate = task_sufficiency(
            orbit_metrics, baseline_metrics, config
        )
        shuffle_pass, shuffle_gate = task_sufficiency(
            shuffle_metrics, baseline_metrics, config
        )
        cut_records[cut] = {
            "activation_geometry": paired_state_geometry(captured[cut], inverse),
            "replay_maximum_absolute_posterior_error": replay_error,
            "orbit_average": {
                "task_metrics": orbit_metrics,
                "task_sufficiency": orbit_gate,
                "task_gate": orbit_pass,
                "posterior_js_from_baseline": jensen_shannon(
                    orbit_posterior, baseline_posterior
                ),
                "maximum_paired_state_identity_error": identity_error,
            },
            "fiber_shuffled": {
                "task_metrics": shuffle_metrics,
                "task_sufficiency": shuffle_gate,
                "task_gate": shuffle_pass,
                "posterior_js_from_baseline": jensen_shannon(
                    shuffle_posterior, baseline_posterior
                ),
                "maximum_paired_state_identity_error": shuffle_identity_error,
            },
        }
        cut_posteriors[cut] = {
            "orbit_average": orbit_posterior,
            "fiber_shuffled": shuffle_posterior,
        }
        arrays[f"{cut}__orbit_posterior"] = orbit_posterior.float().numpy()
        arrays[f"{cut}__shuffle_posterior"] = shuffle_posterior.float().numpy()

    average_pre, _ = orbit_average(captured["pre_block"], inverse)
    propagated_attention = _apply_attention(
        system, average_pre.to(device)
    ).cpu()
    actual_attention, _ = orbit_average(captured["block0_post_attention"], inverse)
    average_attention, _ = orbit_average(captured["block0_post_attention"], inverse)
    propagated_mlp = _apply_mlp(system, average_attention.to(device)).cpu()
    actual_mlp, _ = orbit_average(captured["block0_post_mlp"], inverse)
    transitions = {
        "block0_attention": {
            "source_cut": "pre_block",
            "target_cut": "block0_post_attention",
            "propagated_pass": cut_records["pre_block"]["orbit_average"]["task_gate"],
            "actual_pass": cut_records["block0_post_attention"]["orbit_average"]["task_gate"],
            "causal_regime": _regime_name(
                cut_records["pre_block"]["orbit_average"]["task_gate"],
                cut_records["block0_post_attention"]["orbit_average"]["task_gate"],
            ),
            "defect_rms": float(
                torch.sqrt(torch.mean((actual_attention.double() - propagated_attention.double()).square()))
            ),
            "defect_relative_rms": float(
                torch.sqrt(torch.mean((actual_attention.double() - propagated_attention.double()).square()))
                / torch.sqrt(torch.mean(actual_attention.double().square())).clamp_min(1e-12)
            ),
            "posterior_js_actual_vs_propagated": jensen_shannon(
                cut_posteriors["block0_post_attention"]["orbit_average"],
                cut_posteriors["pre_block"]["orbit_average"],
            ),
        },
        "block0_mlp": {
            "source_cut": "block0_post_attention",
            "target_cut": "block0_post_mlp",
            "propagated_pass": cut_records["block0_post_attention"]["orbit_average"]["task_gate"],
            "actual_pass": cut_records["block0_post_mlp"]["orbit_average"]["task_gate"],
            "causal_regime": _regime_name(
                cut_records["block0_post_attention"]["orbit_average"]["task_gate"],
                cut_records["block0_post_mlp"]["orbit_average"]["task_gate"],
            ),
            "defect_rms": float(
                torch.sqrt(torch.mean((actual_mlp.double() - propagated_mlp.double()).square()))
            ),
            "defect_relative_rms": float(
                torch.sqrt(torch.mean((actual_mlp.double() - propagated_mlp.double()).square()))
                / torch.sqrt(torch.mean(actual_mlp.double().square())).clamp_min(1e-12)
            ),
            "posterior_js_actual_vs_propagated": jensen_shannon(
                cut_posteriors["block0_post_mlp"]["orbit_average"],
                cut_posteriors["block0_post_attention"]["orbit_average"],
            ),
        },
    }
    del captured
    return {
        "baseline_task_metrics": baseline_metrics,
        "cuts": cut_records,
        "transitions": transitions,
        "maximum_replay_error": maximum_replay_error,
        "maximum_state_identity_error": maximum_state_identity_error,
    }, arrays


def classify_campaign(
    *,
    valid: bool,
    cut_pass_counts: Mapping[str, Mapping[str, int]],
    config: CausalClosureConfig,
) -> tuple[str, bool]:
    if not valid:
        return "invalid", False
    required = config.required_seed_passes
    analytic = cut_pass_counts.get("analytic_calibrated", {})
    learned = cut_pass_counts.get("learned_calibrated_equivariant", {})
    analytic_all = all(analytic.get(cut, 0) >= required for cut in CUTS)
    learned_all = all(learned.get(cut, 0) >= required for cut in CUTS)
    analytic_pre = analytic.get("pre_block", 0) >= required
    learned_pre = learned.get("pre_block", 0) >= required
    if analytic_all and learned_all and analytic_pre and learned_pre:
        return "frontend_causal_quotient_closed", True
    if analytic_all and not learned_pre:
        return "analytic_only_frontend_closure", False
    if not learned_pre and any(
        learned.get(cut, 0) >= required for cut in CUTS[1:]
    ):
        return "learned_frontend_requires_transformer_synthesis", False
    if any(
        max(cut_pass_counts.get(condition, {}).values(), default=0) < required
        for condition in PRIMARY_CONDITIONS
    ):
        return "structured_frontend_not_causally_sufficient", False
    return "mixed_frontend_causal_closure", False


def _fingerprint(
    config: CausalClosureConfig,
    implementation: str,
    condition: str,
    seed: int,
    provenance: Mapping[str, Any],
    dataset_hashes: Mapping[str, str],
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": _json_config(config),
        "implementation_sha256": implementation,
        "condition": condition,
        "seed": seed,
        "source": dict(provenance),
        "dataset_hashes": dict(dataset_hashes),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reusable_result(
    path: Path,
    fingerprint: str,
    implementation: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    diagnostics = Path(value.get("artifacts", {}).get("diagnostics", ""))
    if (
        value.get("status") != "completed"
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("scientific_fingerprint") != fingerprint
        or value.get("implementation_sha256") != implementation
        or value.get("artifacts", {}).get("result") != str(path)
        or not diagnostics.is_file()
        or _sha256(diagnostics)
        != value.get("artifacts", {}).get("diagnostics_sha256")
    ):
        raise ValueError(f"incompatible completed causal-closure result {path}")
    return value


def _campaign_reusable(
    campaign: Mapping[str, Any],
    config: CausalClosureConfig,
    implementation: str,
) -> bool:
    entries = campaign.get("results", [])
    expected = {
        (condition, seed)
        for condition in config.conditions
        for seed in config.seeds
    }
    observed = {
        (entry.get("condition"), entry.get("seed")) for entry in entries
    }
    return bool(
        campaign.get("status") == "completed"
        and campaign.get("schema_version") == SCHEMA_VERSION
        and campaign.get("configuration") == _json_config(config)
        and campaign.get("implementation_sha256") == implementation
        and len(entries) == len(expected)
        and observed == expected
        and all(
            Path(entry["path"]).is_file()
            and _sha256(Path(entry["path"])) == entry["result_sha256"]
            and Path(entry["diagnostics_path"]).is_file()
            and _sha256(Path(entry["diagnostics_path"]))
            == entry["diagnostics_sha256"]
            for entry in entries
        )
    )


def run_campaign(config: CausalClosureConfig, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    source_digests = _source_digests()
    implementation = _implementation_digest(source_digests)
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if _campaign_reusable(existing, config, implementation):
            print("campaign already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed campaign {campaign_path}")

    source_campaign, source_path, task, source_config, source_details = (
        _load_source_campaign(config)
    )
    datasets = _datasets(task)
    dataset_hashes = {
        regime: _dataset_hash(dataset) for regime, dataset in datasets.items()
    }
    if dataset_hashes != EXPECTED_DATASET_HASHES:
        raise ValueError("calibrated held-out cohort hashes changed")
    source_root = Path(config.source_root)
    preflight_provenance, preflight_manifest = _preflight_sources(
        config,
        source_root,
        task,
        source_config,
        source_details,
    )
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    results: list[dict[str, Any]] = []
    reused = 0
    for condition in config.conditions:
        for seed in config.seeds:
            cell_started = time.perf_counter()
            detail = source_details[(condition, seed)]
            system, provenance = _load_system(
                source_root,
                condition,
                seed,
                task,
                source_config,
                detail,
                device,
            )
            if provenance != preflight_provenance[(condition, seed)]:
                raise ValueError(
                    f"source changed after preflight for {condition} seed {seed}"
                )
            for parameter in system.parameters():
                parameter.requires_grad_(False)
            fingerprint = _fingerprint(
                config,
                implementation,
                condition,
                seed,
                provenance,
                dataset_hashes,
            )
            result_dir = output / "runs" / condition / f"seed_{seed}"
            result_path = result_dir / "result.json"
            existing = _reusable_result(result_path, fingerprint, implementation)
            if existing is not None:
                results.append(existing)
                reused += 1
                print(f"resuming {condition} seed {seed}", flush=True)
                del system
                continue

            regime_results = {}
            diagnostics_arrays: dict[str, np.ndarray] = {}
            for regime_index, regime in enumerate(REGIMES):
                fiber_count = len(torch.unique(datasets[regime].paired.fiber.fiber_id))
                generator = np.random.default_rng(
                    config.shuffle_seed + regime_index * 1_000_003
                )
                shift = int(generator.integers(1, fiber_count))
                regime_result, regime_arrays = analyze_regime(
                    system,
                    datasets[regime],
                    task,
                    config,
                    device,
                    shift,
                )
                source_metrics = detail["analysis"]["task_metrics"][regime]
                source_replay_error = max(
                    abs(
                        float(regime_result["baseline_task_metrics"][metric])
                        - float(source_metrics[metric])
                    )
                    for metric in (
                        "exact_bin_accuracy",
                        "mean_circular_error_radians",
                        "mean_target_cross_entropy",
                    )
                )
                regime_result["source_task_replay"] = {
                    "maximum_absolute_error": source_replay_error,
                    "pass": source_replay_error <= config.replay_tolerance,
                }
                regime_result["shuffle_cyclic_shift"] = shift
                regime_results[regime] = regime_result
                diagnostics_arrays.update(
                    {
                        f"{regime}__{name}": value
                        for name, value in regime_arrays.items()
                    }
                )

            diagnostics_path = result_dir / "closure_diagnostics.npz"
            _write_npz(diagnostics_path, diagnostics_arrays)
            diagnostics_sha256 = _sha256(diagnostics_path)
            state_unchanged = bool(
                calibrated._state_digest(system.model)
                == provenance["model_state_sha256"]
                and calibrated._module_digest(system)
                == provenance["system_state_sha256"]
            )
            finite = _finite(regime_results)
            replay_pass = all(
                regime_results[regime]["maximum_replay_error"]
                <= config.replay_tolerance
                and regime_results[regime]["source_task_replay"]["pass"]
                for regime in REGIMES
            )
            state_identity_pass = all(
                regime_results[regime]["maximum_state_identity_error"]
                <= config.state_identity_tolerance
                for regime in REGIMES
            )
            validity = bool(
                state_unchanged and finite and replay_pass and state_identity_pass
            )
            cut_seed_gates = {
                cut: all(
                    regime_results[regime]["cuts"][cut]["orbit_average"][
                        "task_gate"
                    ]
                    for regime in REGIMES
                )
                for cut in CUTS
            }
            shuffle_seed_gates = {
                cut: all(
                    regime_results[regime]["cuts"][cut]["fiber_shuffled"][
                        "task_gate"
                    ]
                    for regime in REGIMES
                )
                for cut in CUTS
            }
            transition_seed_gates = {
                transition: all(
                    regime_results[regime]["transitions"][transition][
                        "causal_regime"
                    ]
                    == "quotient_already_closed"
                    for regime in REGIMES
                )
                for transition in TRANSITIONS
            }
            result = {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "experiment_id": f"tinyllm-calibrated-closure-{condition}-seed{seed}",
                "status": "completed",
                "evidence_role": _evidence_role(config),
                "completed_at": _utc_now(),
                "condition": condition,
                "seed": seed,
                "configuration": _json_config(config),
                "implementation_sha256": implementation,
                "scientific_fingerprint": fingerprint,
                "provenance": provenance,
                "dataset_hashes": dataset_hashes,
                "regimes": regime_results,
                "cut_seed_gates": cut_seed_gates,
                "shuffle_seed_gates": shuffle_seed_gates,
                "transition_seed_gates": transition_seed_gates,
                "gates": {
                    "source_and_cut_replay": replay_pass,
                    "paired_state_identity": state_identity_pass,
                    "state_unchanged": state_unchanged,
                    "finite": finite,
                    "validity": validity,
                },
                "analysis_seconds": time.perf_counter() - cell_started,
                "artifacts": {
                    "result": str(result_path),
                    "diagnostics": str(diagnostics_path),
                    "diagnostics_sha256": diagnostics_sha256,
                },
            }
            _write_json(result_path, result)
            results.append(result)
            print(
                f"{condition} seed {seed}: "
                f"pre={cut_seed_gates['pre_block']} "
                f"all={all(cut_seed_gates.values())} "
                f"shuffle={shuffle_seed_gates['pre_block']} valid={validity}",
                flush=True,
            )
            del system
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if _implementation_digest() != implementation:
                raise RuntimeError("causal-closure implementation changed during run")

    arms = {}
    cut_pass_counts = {}
    for condition in config.conditions:
        selected = [item for item in results if item["condition"] == condition]
        cut_counts = {
            cut: sum(int(item["cut_seed_gates"][cut]) for item in selected)
            for cut in CUTS
        }
        shuffle_counts = {
            cut: sum(int(item["shuffle_seed_gates"][cut]) for item in selected)
            for cut in CUTS
        }
        transition_counts = {
            transition: sum(
                int(item["transition_seed_gates"][transition])
                for item in selected
            )
            for transition in TRANSITIONS
        }
        cut_pass_counts[condition] = cut_counts
        arms[condition] = {
            "cut_pass_counts": cut_counts,
            "all_cuts_pass_count": sum(
                int(all(item["cut_seed_gates"].values())) for item in selected
            ),
            "shuffle_pass_counts": shuffle_counts,
            "transition_closed_counts": transition_counts,
        }
    controls_pass = all(
        arms[condition]["shuffle_pass_counts"]["pre_block"] <= 1
        for condition in PRIMARY_CONDITIONS
        if condition in arms
    )
    primary_results = [
        item for item in results if item["condition"] in PRIMARY_CONDITIONS
    ]
    valid = bool(
        all(item["gates"]["validity"] for item in primary_results)
        and controls_pass
    )
    if config.allow_underpowered:
        classification, primary = (
            "systems_lifecycle_only_not_quality_evidence",
            False,
        )
    else:
        classification, primary = classify_campaign(
            valid=valid, cut_pass_counts=cut_pass_counts, config=config
        )

    result_entries = [
        {
            "experiment_id": item["experiment_id"],
            "condition": item["condition"],
            "seed": item["seed"],
            "scientific_fingerprint": item["scientific_fingerprint"],
            "path": item["artifacts"]["result"],
            "result_sha256": _sha256(Path(item["artifacts"]["result"])),
            "diagnostics_path": item["artifacts"]["diagnostics"],
            "diagnostics_sha256": item["artifacts"]["diagnostics_sha256"],
        }
        for item in results
    ]
    result_manifest_sha256 = hashlib.sha256(
        json.dumps(result_entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": _json_config(config),
        "implementation_sha256": implementation,
        "source_digests": source_digests,
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
        "provenance": {
            "source_campaign": str(source_path),
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "source_implementation_sha256": SOURCE_IMPLEMENTATION_SHA256,
            "preregistration": str(PREREGISTRATION_PATH),
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "source_preflight_manifest_sha256": preflight_manifest,
            "source_preflight_completed_before_interventions": True,
        },
        "task_config": asdict(task),
        "dataset_hashes": dataset_hashes,
        "summary": {
            "requested": len(config.conditions) * len(config.seeds),
            "scheduled": len(config.conditions) * len(config.seeds) - reused,
            "completed": len(results),
            "failed": 0,
            "excluded": 0,
            "retries": 0,
            "reused": reused,
            "trained_models": 0,
            "trained_frontends": 0,
            "trained_task_heads": 0,
            "fitted_probes": 0,
            "fitted_observers": 0,
            "fitted_parameters": 0,
        },
        "aggregates": {
            "classification": classification,
            "primary_hypothesis_pass": primary,
            "valid": valid,
            "controls_pass": controls_pass,
            "required_seed_passes": config.required_seed_passes,
            "arms": arms,
        },
        "results": result_entries,
        "result_manifest_sha256": result_manifest_sha256,
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "Exact synthetic fiber membership is used only to construct causal activation barycenters.",
            "TinyLLM, front ends, embeddings, answer rows, probes, and observers remain frozen.",
            "Raw calibrated checkpoints are descriptive because their baseline task accuracy is low.",
            "Task sufficiency is conditioned on the frozen answer decoder and declared held-out cohorts.",
            "Five retained checkpoints do not establish architecture-population prevalence.",
        ],
        "artifacts": {"campaign": str(campaign_path)},
    }
    _write_json(campaign_path, campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _strings(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_calibrated_frontend_causal_closure/"
            "20260810_d15_preregistered"
        ),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--conditions", type=_strings, default=CONDITIONS)
    parser.add_argument("--seeds", type=_ints, default=SEEDS)
    parser.add_argument("--required-seed-passes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = CausalClosureConfig(
        conditions=args.conditions,
        seeds=args.seeds,
        required_seed_passes=args.required_seed_passes,
        batch_size=args.batch_size,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
