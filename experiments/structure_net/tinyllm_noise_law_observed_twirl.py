#!/usr/bin/env python3
"""Test observed C2 quotient closure under reflection-asymmetric sensor noise."""

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
import experiments.structure_net.tinyllm_calibrated_frontend_causal_closure as closure
import experiments.structure_net.tinyllm_observed_deck_twirl as observed
from experiments.structure_net.tinyllm_internal_quotient_probe import FiberDataset
from experiments.structure_net.tinyllm_nuisance_support_scaling import (
    PairedCircleDataset,
)
from experiments.structure_net.tinyllm_predictive_circle import (
    CircleDataset,
    CircleTaskConfig,
)


SCHEMA_VERSION = "nal.tinyllm-noise-law-observed-twirl.v1"
HYPOTHESIS_ID = "tinyllm-noise-law-observed-twirl-v1"
EVIDENCE_ROLE = "preregistered_frozen_sensor_noise_law_intervention"
SOURCE_OBSERVED_SCHEMA = "nal.tinyllm-observed-deck-twirl-causal-closure.v1"
SOURCE_OBSERVED_CAMPAIGN_SHA256 = (
    "79c3e27374d8b6f4611552595de5852ace940204bda825e64cf80eff6ab2050d"
)
SOURCE_OBSERVED_RESULT_MANIFEST_SHA256 = (
    "b91af38162fbf45e29348fbdf583cb676660d68cf22e5a795b438fd8cd015db3"
)
SOURCE_OBSERVED_IMPLEMENTATION_SHA256 = (
    "c970fe8801524f5248a9314e821b6783127596d05a2f206325ed85deb42f9629"
)
SOURCE_OBSERVED_RUNNER_SHA256 = (
    "6468a7af23cd1ae11cdc6cae3cf553252c2cb19352ed89e44c053a1ade60d213"
)
SOURCE_CLOSURE_RUNNER_SHA256 = (
    "0169a1653695ebe7e55b7a3f49b12401f73439f5a784c2ddd5584a4b685761c6"
)
SOURCE_CALIBRATED_RUNNER_SHA256 = (
    "73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-noise-law-observed-twirl-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "8ff50bc47cb7c6223dbf234044a3d18fefd91b15c9d67578d674bb33029be26b"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
SEEDS = (7, 17, 29, 41, 53)
REGIMES = closure.REGIMES
LAWS = ("isotropic", "lab_anisotropic", "lab_biased")
CUTS = ("pre_block", "full")
NOISE_SEEDS = {"composition": 861001, "extrapolation": 861002}


@dataclass(frozen=True)
class NoiseLawObservedTwirlConfig:
    source_observed_root: str = (
        "data/experiments/tinyllm_observed_deck_twirl/"
        "20260810_d10_preregistered"
    )
    source_closure_root: str = (
        "data/experiments/tinyllm_calibrated_frontend_causal_closure/"
        "20260810_d15_preregistered"
    )
    calibrated_source_root: str = (
        "data/experiments/tinyllm_calibrated_frontend_causal/"
        "20260806_d8_preregistered"
    )
    conditions: tuple[str, ...] = CONDITIONS
    seeds: tuple[int, ...] = SEEDS
    laws: tuple[str, ...] = LAWS
    noise_sigma: float = 0.05
    noise_energy_relative_tolerance: float = 0.05
    anisotropic_covariance_defect_floor: float = 0.10
    biased_mean_defect_floor: float = 0.05
    natural_accuracy_loss_ceiling: float = 0.05
    natural_circular_error_increase_ceiling: float = math.pi / 16.0
    natural_cross_entropy_increase_ceiling: float = 0.10
    accuracy_loss_ceiling: float = 0.03
    circular_error_increase_ceiling: float = math.pi / 16.0
    cross_entropy_increase_ceiling: float = 0.10
    analytic_feature_tolerance: float = 1e-6
    replay_tolerance: float = 2e-6
    source_replay_tolerance: float = 2e-6
    required_seed_passes: int = 4
    maximum_control_seed_passes: int = 1
    batch_size: int = 256
    sample_limit: int | None = None
    device: str = "cuda:1"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.conditions or set(self.conditions).difference(CONDITIONS):
            raise ValueError("unknown or empty structured condition")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("checkpoint seeds must be non-empty and distinct")
        if not self.laws or set(self.laws).difference(LAWS):
            raise ValueError("unknown or empty noise law")
        if self.noise_sigma <= 0.0:
            raise ValueError("noise sigma must be positive")
        if self.batch_size < 1:
            raise ValueError("batch size must be positive")
        if self.sample_limit is not None and self.sample_limit < 8:
            raise ValueError("sample limit must be at least eight")
        if not 1 <= self.required_seed_passes <= len(self.seeds):
            raise ValueError("required seed count is outside selected population")
        if not 0 <= self.maximum_control_seed_passes <= len(self.seeds):
            raise ValueError("control ceiling is outside selected population")
        if not self.allow_underpowered:
            expected = (
                self.conditions == CONDITIONS
                and self.seeds == SEEDS
                and self.laws == LAWS
                and self.noise_sigma == 0.05
                and self.sample_limit is None
                and self.batch_size == 256
                and self.required_seed_passes == 4
                and self.maximum_control_seed_passes == 1
            )
            if not expected:
                raise ValueError("primary noise-law campaign configuration is fixed")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
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


def _json_config(config: NoiseLawObservedTwirlConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _source_digests() -> dict[str, str]:
    expected = {
        "preregistration": (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        "observed_deck": (Path(observed.__file__), SOURCE_OBSERVED_RUNNER_SHA256),
        "causal_closure": (Path(closure.__file__), SOURCE_CLOSURE_RUNNER_SHA256),
        "calibrated_frontend": (
            Path(calibrated.__file__),
            SOURCE_CALIBRATED_RUNNER_SHA256,
        ),
    }
    for name, (path, digest) in expected.items():
        if _sha256(path) != digest:
            raise ValueError(f"frozen {name} source changed: {path}")
    if observed._implementation_digest() != SOURCE_OBSERVED_IMPLEMENTATION_SHA256:
        raise ValueError("observed-deck composite implementation changed")
    paths = {"runner": Path(__file__)} | {
        name: path for name, (path, _) in expected.items()
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _implementation_digest(digests: Mapping[str, str] | None = None) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(digests or _source_digests()),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _load_predecessor(
    config: NoiseLawObservedTwirlConfig,
) -> tuple[dict[str, Any], Path, dict[tuple[str, int], dict[str, Any]]]:
    campaign_path = Path(config.source_observed_root) / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if (
        _sha256(campaign_path) != SOURCE_OBSERVED_CAMPAIGN_SHA256
        or campaign.get("schema_version") != SOURCE_OBSERVED_SCHEMA
        or campaign.get("hypothesis_id") != observed.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256")
        != SOURCE_OBSERVED_IMPLEMENTATION_SHA256
        or campaign.get("result_manifest_sha256")
        != SOURCE_OBSERVED_RESULT_MANIFEST_SHA256
        or campaign.get("aggregates", {}).get("classification")
        != "observable_twirl_closed_action_invariant"
        or campaign.get("aggregates", {}).get("primary_hypothesis_pass") is not True
        or campaign.get("aggregates", {}).get("valid") is not True
        or campaign.get("summary", {}).get("completed") != 10
        or campaign.get("summary", {}).get("failed") != 0
        or campaign.get("summary", {}).get("fitted_action_parameters") != 0
    ):
        raise ValueError(f"invalid observed-deck predecessor {campaign_path}")
    expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    indexed = {
        (entry.get("condition"), int(entry.get("seed", -1))): entry
        for entry in campaign.get("results", [])
    }
    if set(indexed) != expected:
        raise ValueError("observed-deck predecessor result index changed")
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for key, entry in indexed.items():
        result_path = Path(entry["path"])
        diagnostics_path = Path(entry["diagnostics_path"])
        detail = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            _sha256(result_path) != entry.get("result_sha256")
            or _sha256(diagnostics_path) != entry.get("diagnostics_sha256")
            or detail.get("status") != "completed"
            or detail.get("condition") != key[0]
            or int(detail.get("seed", -1)) != key[1]
            or detail.get("gates", {}).get("validity") is not True
        ):
            raise ValueError(f"invalid observed-deck result {result_path}")
        if key[0] in config.conditions and key[1] in config.seeds:
            selected[key] = {
                "result_path": str(result_path),
                "result_sha256": entry["result_sha256"],
                "diagnostics_path": str(diagnostics_path),
                "diagnostics_sha256": entry["diagnostics_sha256"],
                "scientific_fingerprint": entry["scientific_fingerprint"],
            }
    return campaign, campaign_path, selected


def _source_loader_config(
    config: NoiseLawObservedTwirlConfig,
) -> observed.ObservedDeckTwirlConfig:
    return observed.ObservedDeckTwirlConfig(
        source_closure_root=config.source_closure_root,
        calibrated_source_root=config.calibrated_source_root,
        conditions=config.conditions,
        seeds=config.seeds,
        accuracy_loss_ceiling=config.accuracy_loss_ceiling,
        circular_error_increase_ceiling=config.circular_error_increase_ceiling,
        cross_entropy_increase_ceiling=config.cross_entropy_increase_ceiling,
        required_seed_passes=min(config.required_seed_passes, len(config.seeds)),
        maximum_control_seed_passes=min(
            config.maximum_control_seed_passes, len(config.seeds)
        ),
        replay_tolerance=config.replay_tolerance,
        batch_size=config.batch_size,
        device="cpu",
        allow_underpowered=True,
    )


def _subset_dataset(
    dataset: calibrated.CalibratedDataset, count: int | None
) -> calibrated.CalibratedDataset:
    if count is None or count >= len(dataset.calibration):
        return dataset
    circle = dataset.paired.circle
    fiber = dataset.paired.fiber
    index = slice(0, count)
    return calibrated.CalibratedDataset(
        paired=PairedCircleDataset(
            circle=CircleDataset(
                input_ids=circle.input_ids[index],
                target_posteriors=circle.target_posteriors[index],
                target_bins=circle.target_bins[index],
                phases=circle.phases[index],
                directions=circle.directions[index],
            ),
            fiber=FiberDataset(
                input_ids=fiber.input_ids[index],
                cosine=fiber.cosine[index],
                branch=fiber.branch[index],
                phase=fiber.phase[index],
                fiber_id=fiber.fiber_id[index],
            ),
        ),
        calibration=dataset.calibration[index],
    )


def generate_noise_laws(
    *, sample_count: int, sensor_steps: int, sigma: float, seed: int
) -> dict[str, torch.Tensor]:
    generator = np.random.default_rng(seed)
    standard = generator.normal(0.0, 1.0, (sample_count, sensor_steps, 2))
    arrays = {
        "isotropic": sigma * standard,
        "lab_anisotropic": sigma
        * standard
        * np.array((math.sqrt(1.8), math.sqrt(0.2)))[None, None, :],
        "lab_biased": sigma
        * (
            standard / math.sqrt(2.0)
            + np.array((1.0, 0.0))[None, None, :]
        ),
    }
    return {
        name: torch.from_numpy(value.astype(np.float32))
        for name, value in arrays.items()
    }


def _noise_arrays_digest(arrays: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = arrays[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _reflection_defects(
    calibration_packet: torch.Tensor, law: str, sigma: float
) -> dict[str, float]:
    axes = calibration_packet[:, :2].double()
    axes = axes / axes.norm(dim=1, keepdim=True).clamp_min(1e-12)
    identity = torch.eye(2, dtype=torch.double)[None]
    reflection = 2.0 * axes[:, :, None] * axes[:, None, :] - identity
    if law == "isotropic":
        mean = torch.zeros(2, dtype=torch.double)
        covariance = sigma**2 * torch.eye(2, dtype=torch.double)
    elif law == "lab_anisotropic":
        mean = torch.zeros(2, dtype=torch.double)
        covariance = sigma**2 * torch.diag(
            torch.tensor((1.8, 0.2), dtype=torch.double)
        )
    elif law == "lab_biased":
        mean = sigma * torch.tensor((1.0, 0.0), dtype=torch.double)
        covariance = 0.5 * sigma**2 * torch.eye(2, dtype=torch.double)
    else:
        raise ValueError(f"unknown noise law {law}")
    reflected_mean = torch.einsum("bij,j->bi", reflection, mean)
    reflected_covariance = torch.einsum(
        "bij,jk,bkl->bil", reflection, covariance, reflection.transpose(1, 2)
    )
    mean_defect = (reflected_mean - mean).norm(dim=1) / sigma
    covariance_defect = (
        (reflected_covariance - covariance).flatten(1).norm(dim=1)
        / (2.0 * sigma**2)
    )
    return {
        "median_normalized_mean_reflection_defect": float(mean_defect.median()),
        "maximum_normalized_mean_reflection_defect": float(mean_defect.max()),
        "median_normalized_covariance_reflection_defect": float(
            covariance_defect.median()
        ),
        "maximum_normalized_covariance_reflection_defect": float(
            covariance_defect.max()
        ),
    }


def noise_law_contract(
    datasets: Mapping[str, calibrated.CalibratedDataset],
    arrays: Mapping[str, Mapping[str, torch.Tensor]],
    config: NoiseLawObservedTwirlConfig,
) -> dict[str, Any]:
    expected_rms = math.sqrt(2.0) * config.noise_sigma
    regimes: dict[str, Any] = {}
    for regime in REGIMES:
        law_records = {}
        for law in config.laws:
            noise = arrays[regime][law].double()
            empirical_rms = float(torch.sqrt(noise.square().sum(-1).mean()))
            energy_relative_error = abs(empirical_rms - expected_rms) / expected_rms
            defects = _reflection_defects(
                datasets[regime].calibration, law, config.noise_sigma
            )
            if law == "isotropic":
                structural_pass = bool(
                    defects["maximum_normalized_mean_reflection_defect"] <= 1e-12
                    and defects[
                        "maximum_normalized_covariance_reflection_defect"
                    ]
                    <= 1e-12
                )
            elif law == "lab_anisotropic":
                structural_pass = bool(
                    defects[
                        "median_normalized_covariance_reflection_defect"
                    ]
                    >= config.anisotropic_covariance_defect_floor
                )
            else:
                structural_pass = bool(
                    defects["median_normalized_mean_reflection_defect"]
                    >= config.biased_mean_defect_floor
                )
            law_records[law] = {
                "expected_planar_rms": expected_rms,
                "empirical_planar_rms": empirical_rms,
                "energy_relative_error": energy_relative_error,
                "energy_pass": energy_relative_error
                <= config.noise_energy_relative_tolerance,
                **defects,
                "structural_pass": structural_pass,
                "pass": bool(
                    energy_relative_error <= config.noise_energy_relative_tolerance
                    and structural_pass
                ),
            }
        regimes[regime] = law_records
    return {
        "common_draws_across_laws": True,
        "expected_squared_planar_norm": 2.0 * config.noise_sigma**2,
        "noise_seeds": NOISE_SEEDS,
        "regimes": regimes,
        "pass": all(
            regimes[regime][law]["pass"]
            for regime in REGIMES
            for law in config.laws
        ),
    }


def task_gate(
    metrics: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    accuracy_loss_ceiling: float,
    circular_error_increase_ceiling: float,
    cross_entropy_increase_ceiling: float,
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
        "accuracy_pass": accuracy_loss <= accuracy_loss_ceiling,
        "circular_error_increase": circular_increase,
        "circular_error_pass": circular_increase
        <= circular_error_increase_ceiling,
        "cross_entropy_increase": cross_entropy_increase,
        "cross_entropy_pass": cross_entropy_increase
        <= cross_entropy_increase_ceiling,
    }
    return bool(all(gates[name] for name in gates if name.endswith("_pass"))), gates


@torch.no_grad()
def _capture(
    system: calibrated.CalibratedTinyLLM,
    input_ids: torch.Tensor,
    sensor: torch.Tensor,
    calibration_packet: torch.Tensor,
    task: CircleTaskConfig,
    config: NoiseLawObservedTwirlConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    captured = {cut: [] for cut in CUTS}
    posteriors = []
    features = []
    for start in range(0, len(sensor), config.batch_size):
        stop = min(len(sensor), start + config.batch_size)
        batch_ids = input_ids[start:stop].to(device)
        batch_sensor = sensor[start:stop].to(device)
        batch_calibration = calibration_packet[start:stop].to(device)
        features.append(system.feature(batch_sensor, batch_calibration).cpu())
        value = closure._initial_sequence(
            system, batch_ids, batch_sensor, batch_calibration
        )
        captured["pre_block"].append(value.cpu())
        for block in system.model.transformer["h"]:
            value = block(value)
        captured["full"].append(value.cpu())
        logits = calibrated._task_logits(system.model, value[:, -1], answer_ids)
        posteriors.append(torch.softmax(logits, -1).double().cpu())
    return (
        {cut: torch.cat(parts).float() for cut, parts in captured.items()},
        torch.cat(posteriors),
        torch.cat(features).float(),
    )


def _natural_gate(
    metrics: Mapping[str, float],
    clean_metrics: Mapping[str, float],
    config: NoiseLawObservedTwirlConfig,
) -> tuple[bool, dict[str, Any]]:
    return task_gate(
        metrics,
        clean_metrics,
        accuracy_loss_ceiling=config.natural_accuracy_loss_ceiling,
        circular_error_increase_ceiling=(
            config.natural_circular_error_increase_ceiling
        ),
        cross_entropy_increase_ceiling=(
            config.natural_cross_entropy_increase_ceiling
        ),
    )


def _intervention_gate(
    metrics: Mapping[str, float],
    baseline: Mapping[str, float],
    config: NoiseLawObservedTwirlConfig,
) -> tuple[bool, dict[str, Any]]:
    return task_gate(
        metrics,
        baseline,
        accuracy_loss_ceiling=config.accuracy_loss_ceiling,
        circular_error_increase_ceiling=config.circular_error_increase_ceiling,
        cross_entropy_increase_ceiling=config.cross_entropy_increase_ceiling,
    )


@torch.no_grad()
def analyze_regime(
    system: calibrated.CalibratedTinyLLM,
    dataset: calibrated.CalibratedDataset,
    noise_arrays: Mapping[str, torch.Tensor],
    source_diagnostics: Mapping[str, np.ndarray],
    regime: str,
    task: CircleTaskConfig,
    config: NoiseLawObservedTwirlConfig,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    input_ids = dataset.paired.circle.input_ids
    sensor = calibrated.decode_sensor_tokens(input_ids, task)
    clean, clean_posterior, clean_feature = _capture(
        system, input_ids, sensor, dataset.calibration, task, config, device
    )
    source_posterior = torch.from_numpy(
        source_diagnostics[f"{regime}__baseline_posterior"]
    )[: len(clean_posterior)].double()
    source_replay_error = float((clean_posterior - source_posterior).abs().max())
    clean_metrics = closure.posterior_metrics(clean_posterior, dataset)
    arrays: dict[str, np.ndarray] = {
        "clean_posterior": clean_posterior.float().numpy(),
        "clean_feature": clean_feature.numpy(),
    }
    law_results: dict[str, Any] = {}
    for law in config.laws:
        noisy_sensor = sensor.clone()
        noisy_sensor[..., :2] += noise_arrays[law]
        correct_sensor, correct_calibration = observed.observed_deck_action(
            noisy_sensor, dataset.calibration, task
        )
        control_sensor, control_calibration = observed.observed_deck_action(
            noisy_sensor, dataset.calibration, task, orthogonal_axis=True
        )
        identity, identity_posterior, identity_feature = _capture(
            system,
            input_ids,
            noisy_sensor,
            dataset.calibration,
            task,
            config,
            device,
        )
        correct, correct_posterior, correct_feature = _capture(
            system,
            input_ids,
            correct_sensor,
            correct_calibration,
            task,
            config,
            device,
        )
        control, control_posterior, control_feature = _capture(
            system,
            input_ids,
            control_sensor,
            control_calibration,
            task,
            config,
            device,
        )
        noisy_metrics = closure.posterior_metrics(identity_posterior, dataset)
        natural_pass, natural_detail = _natural_gate(
            noisy_metrics, clean_metrics, config
        )
        law_arrays: dict[str, np.ndarray] = {
            "noise": noise_arrays[law].numpy(),
            "identity_posterior": identity_posterior.float().numpy(),
            "correct_action_posterior": correct_posterior.float().numpy(),
            "orthogonal_action_posterior": control_posterior.float().numpy(),
            "identity_feature": identity_feature.numpy(),
            "correct_action_feature": correct_feature.numpy(),
            "orthogonal_action_feature": control_feature.numpy(),
        }
        cut_results: dict[str, Any] = {}
        maximum_replay_error = 0.0
        for cut in CUTS:
            identity_replay = closure._continue_from_cut(
                system, cut, identity[cut], task, config, device
            )
            correct_replay = closure._continue_from_cut(
                system, cut, correct[cut], task, config, device
            )
            control_replay = closure._continue_from_cut(
                system, cut, control[cut], task, config, device
            )
            replay_error = max(
                float((identity_replay - identity_posterior).abs().max()),
                float((correct_replay - correct_posterior).abs().max()),
                float((control_replay - control_posterior).abs().max()),
            )
            maximum_replay_error = max(maximum_replay_error, replay_error)
            correct_twirl = 0.5 * (identity[cut] + correct[cut])
            control_twirl = 0.5 * (identity[cut] + control[cut])
            correct_twirl_posterior = closure._continue_from_cut(
                system, cut, correct_twirl, task, config, device
            )
            control_twirl_posterior = closure._continue_from_cut(
                system, cut, control_twirl, task, config, device
            )
            variants = {
                "correct_action": correct_replay,
                "correct_twirl": correct_twirl_posterior,
                "orthogonal_action": control_replay,
                "orthogonal_twirl": control_twirl_posterior,
            }
            records: dict[str, Any] = {}
            for name, posterior in variants.items():
                metrics = closure.posterior_metrics(posterior, dataset)
                passed, gate = _intervention_gate(metrics, noisy_metrics, config)
                records[name] = {
                    "task_metrics": metrics,
                    "task_sufficiency": gate,
                    "task_gate": passed,
                    "posterior_js_from_noisy_identity": closure.jensen_shannon(
                        posterior, identity_posterior
                    ),
                }
                law_arrays[f"{cut}__{name}_posterior"] = (
                    posterior.float().numpy()
                )
            cut_results[cut] = {
                "identity_vs_correct_action_geometry": observed._state_geometry(
                    identity[cut], correct[cut]
                ),
                "identity_vs_orthogonal_action_geometry": observed._state_geometry(
                    identity[cut], control[cut]
                ),
                "replay_maximum_absolute_posterior_error": replay_error,
                **records,
            }
        feature_difference = (identity_feature - correct_feature).double().abs()
        law_results[law] = {
            "clean_task_metrics": clean_metrics,
            "noisy_identity_task_metrics": noisy_metrics,
            "natural_utility": natural_detail,
            "natural_utility_pass": natural_pass,
            "correct_action_feature_maximum_absolute_difference": float(
                feature_difference.max()
            ),
            "correct_action_feature_mean_absolute_difference": float(
                feature_difference.mean()
            ),
            "maximum_replay_error": maximum_replay_error,
            "sensor_noise_geometry": {
                "planar_rms": float(
                    torch.sqrt(noise_arrays[law].double().square().sum(-1).mean())
                ),
                "maximum_absolute_error": float(noise_arrays[law].abs().max()),
                "noisy_planar_maximum_absolute_value": float(
                    noisy_sensor[..., :2].abs().max()
                ),
                "transformed_planar_maximum_absolute_value": float(
                    torch.maximum(
                        correct_sensor[..., :2].abs().max(),
                        control_sensor[..., :2].abs().max(),
                    )
                ),
            },
            "cuts": cut_results,
        }
        arrays.update(
            {f"{law}__{name}": value for name, value in law_arrays.items()}
        )
        del identity, correct, control
    return {
        "source_clean_replay_maximum_absolute_error": source_replay_error,
        "clean_task_metrics": clean_metrics,
        "laws": law_results,
    }, arrays


def classify_campaign(
    *,
    integrity_valid: bool,
    isotropic_positive: bool,
    analytic_positive: bool,
    controls_pass: bool,
    learned_law_passes: Mapping[str, bool],
    any_natural_failure: bool,
) -> tuple[str, bool]:
    if not integrity_valid:
        return "invalid_integrity", False
    if not isotropic_positive:
        return "invalid_isotropic_positive_control", False
    if not analytic_positive:
        return "invalid_analytic_positive_control", False
    if not controls_pass:
        return "nonspecific_target_changing_control", False
    if all(learned_law_passes.values()):
        return "observed_quotient_closed_under_asymmetric_noise", True
    if any_natural_failure:
        return "natural_utility_breaks_before_closure", False
    if learned_law_passes.get("isotropic", False):
        return "learned_quotient_support_relative_to_noise_law", False
    return "learned_quotient_not_noise_robust", False


def _fingerprint(
    config: NoiseLawObservedTwirlConfig,
    implementation: str,
    condition: str,
    seed: int,
    provenance: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    dataset_hashes: Mapping[str, str],
    noise_arrays_sha256: str,
    contract_sha256: str,
) -> str:
    material = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": _json_config(config),
        "implementation_sha256": implementation,
        "condition": condition,
        "seed": seed,
        "calibrated_source": dict(provenance),
        "predecessor": dict(predecessor),
        "dataset_hashes": dict(dataset_hashes),
        "noise_arrays_sha256": noise_arrays_sha256,
        "noise_law_contract_sha256": contract_sha256,
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reusable_result(
    path: Path, fingerprint: str, implementation: str
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
        raise ValueError(f"incompatible completed noise-law result {path}")
    return value


def _campaign_reusable(
    campaign: Mapping[str, Any],
    config: NoiseLawObservedTwirlConfig,
    implementation: str,
) -> bool:
    entries = campaign.get("results", [])
    expected = {
        (condition, seed)
        for condition in config.conditions
        for seed in config.seeds
    }
    observed_cells = {
        (entry.get("condition"), entry.get("seed")) for entry in entries
    }
    noise_path = Path(campaign.get("artifacts", {}).get("noise_law_arrays", ""))
    return bool(
        campaign.get("status") == "completed"
        and campaign.get("schema_version") == SCHEMA_VERSION
        and campaign.get("configuration") == _json_config(config)
        and campaign.get("implementation_sha256") == implementation
        and observed_cells == expected
        and len(entries) == len(expected)
        and noise_path.is_file()
        and _sha256(noise_path)
        == campaign.get("artifacts", {}).get("noise_law_arrays_file_sha256")
        and all(
            Path(entry["path"]).is_file()
            and _sha256(Path(entry["path"])) == entry["result_sha256"]
            and Path(entry["diagnostics_path"]).is_file()
            and _sha256(Path(entry["diagnostics_path"]))
            == entry["diagnostics_sha256"]
            for entry in entries
        )
    )


def run_campaign(
    config: NoiseLawObservedTwirlConfig, output: Path
) -> dict[str, Any]:
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

    predecessor_campaign, predecessor_path, predecessor_details = (
        _load_predecessor(config)
    )
    loader_config = _source_loader_config(config)
    (
        source_closure,
        source_closure_path,
        task,
        source_config,
        original_details,
        _,
        load_config,
    ) = observed._load_sources(loader_config)
    datasets = {
        regime: _subset_dataset(dataset, config.sample_limit)
        for regime, dataset in closure._datasets(task).items()
    }
    dataset_hashes = {
        regime: closure._dataset_hash(dataset)
        for regime, dataset in datasets.items()
    }
    if config.sample_limit is None and dataset_hashes != closure.EXPECTED_DATASET_HASHES:
        raise ValueError("primary noise-law cohorts changed")

    noise_arrays = {
        regime: generate_noise_laws(
            sample_count=len(dataset.calibration),
            sensor_steps=task.sensor_steps,
            sigma=config.noise_sigma,
            seed=NOISE_SEEDS[regime],
        )
        for regime, dataset in datasets.items()
    }
    flat_noise_arrays = {
        f"{regime}__{law}": noise_arrays[regime][law].numpy()
        for regime in REGIMES
        for law in config.laws
    }
    noise_arrays_sha256 = _noise_arrays_digest(
        {
            f"{regime}__{law}": noise_arrays[regime][law]
            for regime in REGIMES
            for law in config.laws
        }
    )
    noise_path = output / "noise_law_arrays.npz"
    _write_npz(noise_path, flat_noise_arrays)
    noise_file_sha256 = _sha256(noise_path)
    contract = noise_law_contract(datasets, noise_arrays, config)
    if not contract["pass"]:
        raise ValueError(f"noise-law generator contract failed: {contract}")
    contract_sha256 = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    source_root = Path(config.calibrated_source_root)
    preflight, preflight_manifest = closure._preflight_sources(
        load_config,
        source_root,
        task,
        source_config,
        original_details,
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
            system, provenance = closure._load_system(
                source_root,
                condition,
                seed,
                task,
                source_config,
                original_details[(condition, seed)],
                device,
            )
            if provenance != preflight[(condition, seed)]:
                raise ValueError(
                    f"source changed after preflight for {condition} seed {seed}"
                )
            for parameter in system.parameters():
                parameter.requires_grad_(False)
            predecessor = predecessor_details[(condition, seed)]
            fingerprint = _fingerprint(
                config,
                implementation,
                condition,
                seed,
                provenance,
                predecessor,
                dataset_hashes,
                noise_arrays_sha256,
                contract_sha256,
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

            with np.load(predecessor["diagnostics_path"], allow_pickle=False) as loaded:
                source_diagnostics = {name: loaded[name] for name in loaded.files}
            regime_results: dict[str, Any] = {}
            diagnostic_arrays: dict[str, np.ndarray] = {}
            for regime in REGIMES:
                regime_result, arrays = analyze_regime(
                    system,
                    datasets[regime],
                    noise_arrays[regime],
                    source_diagnostics,
                    regime,
                    task,
                    config,
                    device,
                )
                regime_results[regime] = regime_result
                diagnostic_arrays.update(
                    {f"{regime}__{name}": value for name, value in arrays.items()}
                )

            diagnostics_path = result_dir / "noise_law_diagnostics.npz"
            _write_npz(diagnostics_path, diagnostic_arrays)
            diagnostics_sha256 = _sha256(diagnostics_path)
            state_unchanged = bool(
                calibrated._state_digest(system.model)
                == provenance["model_state_sha256"]
                and calibrated._module_digest(system)
                == provenance["system_state_sha256"]
            )
            source_replay_pass = all(
                regime_results[regime][
                    "source_clean_replay_maximum_absolute_error"
                ]
                <= config.source_replay_tolerance
                for regime in REGIMES
            )
            cut_replay_pass = all(
                regime_results[regime]["laws"][law]["maximum_replay_error"]
                <= config.replay_tolerance
                for regime in REGIMES
                for law in config.laws
            )
            finite = _finite(regime_results)
            law_seed_gates = {
                law: all(
                    regime_results[regime]["laws"][law]["natural_utility_pass"]
                    and all(
                        regime_results[regime]["laws"][law]["cuts"][cut][
                            variant
                        ]["task_gate"]
                        for cut in CUTS
                        for variant in ("correct_action", "correct_twirl")
                    )
                    for regime in REGIMES
                )
                for law in config.laws
            }
            natural_seed_gates = {
                law: all(
                    regime_results[regime]["laws"][law]["natural_utility_pass"]
                    for regime in REGIMES
                )
                for law in config.laws
            }
            control_seed_gates = {
                law: any(
                    all(
                        regime_results[regime]["laws"][law]["cuts"][cut][
                            variant
                        ]["task_gate"]
                        for regime in REGIMES
                    )
                    for cut in CUTS
                    for variant in ("orthogonal_action", "orthogonal_twirl")
                )
                for law in config.laws
            }
            action_seed_gates = {
                law: {
                    cut: all(
                        regime_results[regime]["laws"][law]["cuts"][cut][
                            "correct_action"
                        ]["task_gate"]
                        for regime in REGIMES
                    )
                    for cut in CUTS
                }
                for law in config.laws
            }
            twirl_seed_gates = {
                law: {
                    cut: all(
                        regime_results[regime]["laws"][law]["cuts"][cut][
                            "correct_twirl"
                        ]["task_gate"]
                        for regime in REGIMES
                    )
                    for cut in CUTS
                }
                for law in config.laws
            }
            analytic_feature_pass = bool(
                condition != "analytic_calibrated"
                or all(
                    regime_results[regime]["laws"][law][
                        "correct_action_feature_maximum_absolute_difference"
                    ]
                    <= config.analytic_feature_tolerance
                    for regime in REGIMES
                    for law in config.laws
                )
            )
            validity = bool(
                state_unchanged
                and source_replay_pass
                and cut_replay_pass
                and analytic_feature_pass
                and finite
            )
            result = {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "experiment_id": f"tinyllm-noise-law-{condition}-seed{seed}",
                "status": "completed",
                "evidence_role": EVIDENCE_ROLE,
                "completed_at": _utc_now(),
                "condition": condition,
                "seed": seed,
                "configuration": _json_config(config),
                "implementation_sha256": implementation,
                "scientific_fingerprint": fingerprint,
                "provenance": {
                    **provenance,
                    "observed_predecessor_result": predecessor["result_path"],
                    "observed_predecessor_result_sha256": predecessor[
                        "result_sha256"
                    ],
                    "observed_predecessor_diagnostics": predecessor[
                        "diagnostics_path"
                    ],
                    "observed_predecessor_diagnostics_sha256": predecessor[
                        "diagnostics_sha256"
                    ],
                },
                "dataset_hashes": dataset_hashes,
                "noise_arrays_sha256": noise_arrays_sha256,
                "noise_law_contract_sha256": contract_sha256,
                "regimes": regime_results,
                "law_seed_gates": law_seed_gates,
                "natural_seed_gates": natural_seed_gates,
                "control_seed_gates": control_seed_gates,
                "action_seed_gates": action_seed_gates,
                "twirl_seed_gates": twirl_seed_gates,
                "gates": {
                    "source_clean_replay": source_replay_pass,
                    "cut_replay": cut_replay_pass,
                    "analytic_feature_invariance": analytic_feature_pass,
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
                f"{condition} seed {seed}: laws={law_seed_gates} "
                f"controls={control_seed_gates} valid={validity}",
                flush=True,
            )
            del system
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if _implementation_digest() != implementation:
                raise RuntimeError("noise-law implementation changed during run")

    arms: dict[str, Any] = {}
    for condition in config.conditions:
        selected = [item for item in results if item["condition"] == condition]
        arms[condition] = {
            "laws": {
                law: {
                    "joint_pass_count": sum(
                        int(item["law_seed_gates"][law]) for item in selected
                    ),
                    "natural_utility_pass_count": sum(
                        int(item["natural_seed_gates"][law]) for item in selected
                    ),
                    "control_pass_count": sum(
                        int(item["control_seed_gates"][law]) for item in selected
                    ),
                    "action_pass_counts": {
                        cut: sum(
                            int(item["action_seed_gates"][law][cut])
                            for item in selected
                        )
                        for cut in CUTS
                    },
                    "twirl_pass_counts": {
                        cut: sum(
                            int(item["twirl_seed_gates"][law][cut])
                            for item in selected
                        )
                        for cut in CUTS
                    },
                }
                for law in config.laws
            }
        }

    integrity_valid = bool(
        contract["pass"] and all(item["gates"]["validity"] for item in results)
    )
    isotropic_positive = all(
        arms[condition]["laws"]["isotropic"]["joint_pass_count"]
        >= config.required_seed_passes
        for condition in config.conditions
    )
    analytic_positive = bool(
        "analytic_calibrated" not in config.conditions
        or all(
            arms["analytic_calibrated"]["laws"][law]["joint_pass_count"]
            >= config.required_seed_passes
            for law in config.laws
        )
    )
    controls_pass = all(
        arms[condition]["laws"][law]["control_pass_count"]
        <= config.maximum_control_seed_passes
        for condition in config.conditions
        for law in config.laws
    )
    learned_law_passes = {
        law: bool(
            "learned_calibrated_equivariant" in config.conditions
            and arms["learned_calibrated_equivariant"]["laws"][law][
                "joint_pass_count"
            ]
            >= config.required_seed_passes
        )
        for law in config.laws
    }
    any_natural_failure = any(
        arms[condition]["laws"][law]["natural_utility_pass_count"]
        < config.required_seed_passes
        for condition in config.conditions
        for law in config.laws
    )
    if config.allow_underpowered:
        classification, primary_pass = (
            "systems_lifecycle_only_not_quality_evidence",
            False,
        )
        valid = integrity_valid
    else:
        classification, primary_pass = classify_campaign(
            integrity_valid=integrity_valid,
            isotropic_positive=isotropic_positive,
            analytic_positive=analytic_positive,
            controls_pass=controls_pass,
            learned_law_passes=learned_law_passes,
            any_natural_failure=any_natural_failure,
        )
        valid = bool(
            integrity_valid
            and isotropic_positive
            and analytic_positive
            and controls_pass
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
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
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
            "source_observed_campaign": str(predecessor_path),
            "source_observed_campaign_sha256": (
                SOURCE_OBSERVED_CAMPAIGN_SHA256
            ),
            "source_observed_implementation_sha256": (
                SOURCE_OBSERVED_IMPLEMENTATION_SHA256
            ),
            "source_observed_result_manifest_sha256": (
                SOURCE_OBSERVED_RESULT_MANIFEST_SHA256
            ),
            "source_closure_campaign": str(source_closure_path),
            "source_closure_campaign_sha256": observed.SOURCE_CLOSURE_CAMPAIGN_SHA256,
            "source_closure_result_manifest_sha256": (
                observed.SOURCE_CLOSURE_RESULT_MANIFEST_SHA256
            ),
            "source_preflight_manifest_sha256": preflight_manifest,
            "source_preflight_completed_before_interventions": True,
            "preregistration": str(PREREGISTRATION_PATH),
            "preregistration_sha256": PREREGISTRATION_SHA256,
        },
        "task_config": asdict(task),
        "dataset_hashes": dataset_hashes,
        "noise_law_contract": contract,
        "noise_law_contract_sha256": contract_sha256,
        "noise_arrays_sha256": noise_arrays_sha256,
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
            "fitted_noise_models": 0,
            "fitted_action_parameters": 0,
        },
        "aggregates": {
            "classification": classification,
            "primary_hypothesis_pass": primary_pass,
            "valid": valid,
            "integrity_valid": integrity_valid,
            "isotropic_positive_control": isotropic_positive,
            "analytic_positive_control": analytic_positive,
            "controls_pass": controls_pass,
            "any_natural_failure": any_natural_failure,
            "required_seed_passes": config.required_seed_passes,
            "maximum_control_seed_passes": config.maximum_control_seed_passes,
            "arms": arms,
        },
        "results": result_entries,
        "result_manifest_sha256": result_manifest_sha256,
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "The intervention adds one fixed-scale error to decoded planar values after tokenization and before the structured front end.",
            "The existing composition and extrapolation cohorts are reused to isolate the noise law; new deterministic error draws are shared across all systems.",
            "The correct action reads decoded planar history, observed calibration, and the fixed time grid only.",
            "The asymmetric transformed observation may be off the natural support of its law; this tests frozen functional closure, not distributional group symmetry.",
            "Temporal correlation, calibration-packet error, requantization, other scales, other groups, and architecture populations are not tested.",
            "TinyLLM, front ends, embeddings, answer rows, actions, probes, and observers remain frozen.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "noise_law_arrays": str(noise_path),
            "noise_law_arrays_file_sha256": noise_file_sha256,
        },
    }
    del predecessor_campaign, source_closure
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
            "data/experiments/tinyllm_noise_law_observed_twirl/"
            "20260810_d10_preregistered"
        ),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--conditions", type=_strings, default=CONDITIONS)
    parser.add_argument("--seeds", type=_ints, default=SEEDS)
    parser.add_argument("--laws", type=_strings, default=LAWS)
    parser.add_argument("--noise-sigma", type=float, default=0.05)
    parser.add_argument("--required-seed-passes", type=int, default=4)
    parser.add_argument("--maximum-control-seed-passes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = NoiseLawObservedTwirlConfig(
        conditions=args.conditions,
        seeds=args.seeds,
        laws=args.laws,
        noise_sigma=args.noise_sigma,
        required_seed_passes=args.required_seed_passes,
        maximum_control_seed_passes=args.maximum_control_seed_passes,
        batch_size=args.batch_size,
        sample_limit=args.sample_limit,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
