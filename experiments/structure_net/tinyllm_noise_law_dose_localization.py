#!/usr/bin/env python3
"""Localize a utility-valid sensor-noise dose before comparing noise laws."""

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

import experiments.structure_net.tinyllm_noise_law_observed_twirl as source


SCHEMA_VERSION = "nal.tinyllm-noise-law-dose-localization.v1"
HYPOTHESIS_ID = "tinyllm-noise-law-dose-localization-v1"
EVIDENCE_ROLE = (
    "preregistered_post_outcome_corrective_frozen_dose_localization"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-noise-law-dose-localization-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "79913c913c7f6f41714400fd4337224f039be0466541ec0a7f26736c599b7a4a"
)
SOURCE_RUNNER_SHA256 = (
    "7bed49c064e8a2148268d2a4ab3a42ec70847a15d83c7297cff3d9dccc7970d2"
)
SOURCE_CAMPAIGN_SHA256 = (
    "868ad0ffee546f157e701790c34a83f20bfb3116e78b2f8c5bc34dd7bfe660d7"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "d4a7e172b0cb9ed5da9a4508c812211882075fcb75db540a17ac6912a8330d6a"
)
SOURCE_RESULT_MANIFEST_SHA256 = (
    "7246968593214d5a91b9283e856472cf351b2e921d6712402f9fc128bb457d4d"
)
SOURCE_NOISE_FILE_SHA256 = (
    "d3771eac8e29f7940df7feaedebe74a5a78fb273cda2e70928c9be9e37ff3ba6"
)
SOURCE_NOISE_CONTENT_SHA256 = (
    "93df61bc76ed073ea241c9450e7ec3523e7a98b5ac06e58d7e920a5df07d70aa"
)
SOURCE_DVC_ROOT = "19f1fbbe86b6b9235eb211a88bb32aa2.dir"
SOURCE_LAKEFS_COMMIT = (
    "f3c895cdf8d5f25e8ae6a87b3f694d0bbacb24cdd14d4736d0c7dfa41399c130"
)
CONDITIONS = source.CONDITIONS
SEEDS = source.SEEDS
REGIMES = source.REGIMES
LAWS = source.LAWS
CUTS = source.CUTS
DOSE_MULTIPLIERS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0)


@dataclass(frozen=True)
class NoiseLawDoseLocalizationConfig:
    source_noise_root: str = (
        "data/experiments/tinyllm_noise_law_observed_twirl/"
        "20260810_d10_preregistered"
    )
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
    dose_multipliers: tuple[float, ...] = DOSE_MULTIPLIERS
    source_noise_sigma: float = 0.05
    natural_accuracy_loss_ceiling: float = 0.05
    natural_circular_error_increase_ceiling: float = math.pi / 16.0
    natural_cross_entropy_increase_ceiling: float = 0.10
    accuracy_loss_ceiling: float = 0.03
    circular_error_increase_ceiling: float = math.pi / 16.0
    cross_entropy_increase_ceiling: float = 0.10
    analytic_feature_tolerance: float = 1e-6
    replay_tolerance: float = 2e-6
    source_replay_tolerance: float = 2e-6
    zero_replay_tolerance: float = 2e-6
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
        if not self.laws or set(self.laws) != set(LAWS):
            raise ValueError("all registered noise laws are required")
        if self.source_noise_sigma != 0.05:
            raise ValueError("the source noise sigma is frozen at 0.05")
        if self.batch_size < 1:
            raise ValueError("batch size must be positive")
        if self.sample_limit is not None and self.sample_limit < 8:
            raise ValueError("sample limit must be at least eight")
        if not 1 <= self.required_seed_passes <= len(self.seeds):
            raise ValueError("required seed count is outside selected population")
        if not 0 <= self.maximum_control_seed_passes <= len(self.seeds):
            raise ValueError("control ceiling is outside selected population")
        if (
            tuple(sorted(set(self.dose_multipliers)))
            != self.dose_multipliers
            or self.dose_multipliers != DOSE_MULTIPLIERS
        ):
            raise ValueError("the registered nested dose ladder is fixed")
        if not self.allow_underpowered:
            expected = (
                self.conditions == CONDITIONS
                and self.seeds == SEEDS
                and self.laws == LAWS
                and self.sample_limit is None
                and self.batch_size == 256
                and self.required_seed_passes == 4
                and self.maximum_control_seed_passes == 1
            )
            if not expected:
                raise ValueError("primary dose-localization configuration is fixed")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
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


def _json_config(config: NoiseLawDoseLocalizationConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _dose_key(multiplier: float) -> str:
    return f"{multiplier:.3f}"


def _source_digests() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "preregistration": PREREGISTRATION_PATH,
        "source_noise_runner": Path(source.__file__),
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "source_noise_runner": SOURCE_RUNNER_SHA256,
    }
    for name, digest in expected.items():
        if _sha256(paths[name]) != digest:
            raise ValueError(f"frozen {name} changed: {paths[name]}")
    return {name: _sha256(path) for name, path in paths.items()}


def _implementation_digest(digests: Mapping[str, str] | None = None) -> str:
    return _json_hash(dict(digests or _source_digests()))


def _source_config(
    config: NoiseLawDoseLocalizationConfig,
    *,
    noise_sigma: float | None = None,
) -> source.NoiseLawObservedTwirlConfig:
    return source.NoiseLawObservedTwirlConfig(
        source_observed_root=config.source_observed_root,
        source_closure_root=config.source_closure_root,
        calibrated_source_root=config.calibrated_source_root,
        conditions=config.conditions,
        seeds=config.seeds,
        laws=config.laws,
        noise_sigma=(
            config.source_noise_sigma if noise_sigma is None else noise_sigma
        ),
        natural_accuracy_loss_ceiling=config.natural_accuracy_loss_ceiling,
        natural_circular_error_increase_ceiling=(
            config.natural_circular_error_increase_ceiling
        ),
        natural_cross_entropy_increase_ceiling=(
            config.natural_cross_entropy_increase_ceiling
        ),
        accuracy_loss_ceiling=config.accuracy_loss_ceiling,
        circular_error_increase_ceiling=config.circular_error_increase_ceiling,
        cross_entropy_increase_ceiling=config.cross_entropy_increase_ceiling,
        analytic_feature_tolerance=config.analytic_feature_tolerance,
        replay_tolerance=config.replay_tolerance,
        source_replay_tolerance=config.source_replay_tolerance,
        required_seed_passes=min(config.required_seed_passes, len(config.seeds)),
        maximum_control_seed_passes=min(
            config.maximum_control_seed_passes, len(config.seeds)
        ),
        batch_size=config.batch_size,
        sample_limit=config.sample_limit,
        device=config.device,
        allow_underpowered=True,
    )


def _load_source_campaign(
    config: NoiseLawDoseLocalizationConfig,
) -> tuple[dict[str, Any], Path, dict[str, dict[str, torch.Tensor]]]:
    root = Path(config.source_noise_root)
    campaign_path = root / "campaign_results.json"
    noise_path = root / "noise_law_arrays.npz"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if (
        _sha256(campaign_path) != SOURCE_CAMPAIGN_SHA256
        or campaign.get("schema_version") != source.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != source.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256")
        != SOURCE_IMPLEMENTATION_SHA256
        or campaign.get("result_manifest_sha256")
        != SOURCE_RESULT_MANIFEST_SHA256
        or campaign.get("aggregates", {}).get("classification")
        != "invalid_isotropic_positive_control"
        or campaign.get("aggregates", {}).get("integrity_valid") is not True
        or campaign.get("aggregates", {}).get("primary_hypothesis_pass")
        is not False
        or campaign.get("summary", {}).get("completed") != 10
        or campaign.get("summary", {}).get("failed") != 0
        or campaign.get("summary", {}).get("trained_models") != 0
        or campaign.get("summary", {}).get("fitted_noise_models") != 0
        or _sha256(noise_path) != SOURCE_NOISE_FILE_SHA256
        or campaign.get("artifacts", {}).get("noise_law_arrays_file_sha256")
        != SOURCE_NOISE_FILE_SHA256
    ):
        raise ValueError(f"invalid frozen noise-law source {campaign_path}")
    expected_keys = {
        f"{regime}__{law}" for regime in REGIMES for law in LAWS
    }
    with np.load(noise_path, allow_pickle=False) as loaded:
        if set(loaded.files) != expected_keys:
            raise ValueError("frozen error-array key set changed")
        arrays = {
            regime: {
                law: torch.from_numpy(loaded[f"{regime}__{law}"].copy())
                for law in LAWS
            }
            for regime in REGIMES
        }
    content = source._noise_arrays_digest(
        {
            f"{regime}__{law}": arrays[regime][law]
            for regime in REGIMES
            for law in LAWS
        }
    )
    if content != SOURCE_NOISE_CONTENT_SHA256:
        raise ValueError("frozen error-array content changed")
    return campaign, campaign_path, arrays


def scaled_noise_arrays(
    base: Mapping[str, Mapping[str, torch.Tensor]], multiplier: float
) -> dict[str, dict[str, torch.Tensor]]:
    if multiplier < 0.0 or multiplier > 1.0:
        raise ValueError("dose multiplier must lie in [0, 1]")
    return {
        regime: {
            law: base[regime][law] * float(multiplier) for law in LAWS
        }
        for regime in REGIMES
    }


def _scaled_array_digest(
    arrays: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    laws: tuple[str, ...] = LAWS,
) -> str:
    return source._noise_arrays_digest(
        {
            f"{regime}__{law}": arrays[regime][law]
            for regime in REGIMES
            for law in laws
        }
    )


def _load_runtime_sources(config: NoiseLawDoseLocalizationConfig) -> dict[str, Any]:
    source_config = _source_config(config)
    predecessor_campaign, predecessor_path, predecessor = source._load_predecessor(
        source_config
    )
    loader_config = source._source_loader_config(source_config)
    (
        source_closure,
        source_closure_path,
        task,
        source_task_config,
        original_details,
        _,
        load_config,
    ) = source.observed._load_sources(loader_config)
    datasets = {
        regime: source._subset_dataset(dataset, config.sample_limit)
        for regime, dataset in source.closure._datasets(task).items()
    }
    dataset_hashes = {
        regime: source.closure._dataset_hash(dataset)
        for regime, dataset in datasets.items()
    }
    if (
        config.sample_limit is None
        and dataset_hashes != source.closure.EXPECTED_DATASET_HASHES
    ):
        raise ValueError("primary dose-localization cohorts changed")
    source_root = Path(config.calibrated_source_root)
    preflight, preflight_manifest = source.closure._preflight_sources(
        load_config,
        source_root,
        task,
        source_task_config,
        original_details,
    )
    return {
        "predecessor_campaign": predecessor_campaign,
        "predecessor_path": predecessor_path,
        "predecessor": predecessor,
        "source_closure": source_closure,
        "source_closure_path": source_closure_path,
        "task": task,
        "source_task_config": source_task_config,
        "original_details": original_details,
        "load_config": load_config,
        "datasets": datasets,
        "dataset_hashes": dataset_hashes,
        "source_root": source_root,
        "preflight": preflight,
        "preflight_manifest": preflight_manifest,
    }


def _load_system(
    runtime: Mapping[str, Any],
    condition: str,
    seed: int,
    device: torch.device,
) -> tuple[source.calibrated.CalibratedTinyLLM, dict[str, Any]]:
    system, provenance = source.closure._load_system(
        runtime["source_root"],
        condition,
        seed,
        runtime["task"],
        runtime["source_task_config"],
        runtime["original_details"][(condition, seed)],
        device,
    )
    if provenance != runtime["preflight"][(condition, seed)]:
        raise ValueError(f"source changed after preflight for {condition} seed {seed}")
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    return system, provenance


def _fingerprint(
    *,
    stage: str,
    config: NoiseLawDoseLocalizationConfig,
    implementation: str,
    condition: str,
    seed: int,
    provenance: Mapping[str, Any],
    dataset_hashes: Mapping[str, str],
    selected_multiplier: float | None = None,
    selected_arrays_sha256: str | None = None,
) -> str:
    return _json_hash(
        {
            "stage": stage,
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "configuration": _json_config(config),
            "implementation_sha256": implementation,
            "condition": condition,
            "seed": seed,
            "provenance": provenance,
            "dataset_hashes": dict(dataset_hashes),
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "source_noise_file_sha256": SOURCE_NOISE_FILE_SHA256,
            "source_noise_content_sha256": SOURCE_NOISE_CONTENT_SHA256,
            "selected_multiplier": selected_multiplier,
            "selected_arrays_sha256": selected_arrays_sha256,
        }
    )


def _reusable_result(
    path: Path,
    *,
    stage: str,
    fingerprint: str,
    implementation: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    diagnostics_path = Path(value.get("artifacts", {}).get("diagnostics", ""))
    if (
        value.get("status") != "completed"
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("stage") != stage
        or value.get("scientific_fingerprint") != fingerprint
        or value.get("implementation_sha256") != implementation
        or not diagnostics_path.is_file()
        or _sha256(diagnostics_path)
        != value.get("artifacts", {}).get("diagnostics_sha256")
    ):
        raise ValueError(f"incompatible completed {stage} result {path}")
    return value


@torch.no_grad()
def _run_stage1_cell(
    *,
    system: source.calibrated.CalibratedTinyLLM,
    condition: str,
    seed: int,
    provenance: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    runtime: Mapping[str, Any],
    base_arrays: Mapping[str, Mapping[str, torch.Tensor]],
    config: NoiseLawDoseLocalizationConfig,
    implementation: str,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    fingerprint = _fingerprint(
        stage="isotropic_localization",
        config=config,
        implementation=implementation,
        condition=condition,
        seed=seed,
        provenance=provenance,
        dataset_hashes=runtime["dataset_hashes"],
    )
    result_dir = output / "runs" / condition / f"seed_{seed}"
    result_path = result_dir / "stage1_result.json"
    reusable = _reusable_result(
        result_path,
        stage="isotropic_localization",
        fingerprint=fingerprint,
        implementation=implementation,
    )
    if reusable is not None:
        print(f"resuming stage1 {condition} seed {seed}", flush=True)
        return reusable

    started = time.perf_counter()
    source_config = _source_config(config)
    with np.load(predecessor["diagnostics_path"], allow_pickle=False) as loaded:
        source_diagnostics = {name: loaded[name] for name in loaded.files}
    regimes: dict[str, Any] = {}
    diagnostics: dict[str, np.ndarray] = {}
    maximum_source_replay = 0.0
    maximum_zero_replay = 0.0
    for regime in REGIMES:
        dataset = runtime["datasets"][regime]
        input_ids = dataset.paired.circle.input_ids
        sensor = source.calibrated.decode_sensor_tokens(
            input_ids, runtime["task"]
        )
        _, clean_posterior, _ = source._capture(
            system,
            input_ids,
            sensor,
            dataset.calibration,
            runtime["task"],
            source_config,
            device,
        )
        source_posterior = torch.from_numpy(
            source_diagnostics[f"{regime}__baseline_posterior"]
        )[: len(clean_posterior)].double()
        source_replay = float((clean_posterior - source_posterior).abs().max())
        maximum_source_replay = max(maximum_source_replay, source_replay)
        clean_metrics = source.closure.posterior_metrics(clean_posterior, dataset)
        dose_records: dict[str, Any] = {}
        diagnostics[f"{regime}__clean_posterior"] = (
            clean_posterior.float().numpy()
        )
        for multiplier in config.dose_multipliers:
            key = _dose_key(multiplier)
            noisy_sensor = sensor.clone()
            noisy_sensor[..., :2] += (
                base_arrays[regime]["isotropic"] * multiplier
            )
            _, posterior, _ = source._capture(
                system,
                input_ids,
                noisy_sensor,
                dataset.calibration,
                runtime["task"],
                source_config,
                device,
            )
            metrics = source.closure.posterior_metrics(posterior, dataset)
            natural_pass, natural_gate = source._natural_gate(
                metrics, clean_metrics, source_config
            )
            zero_replay = (
                float((posterior - clean_posterior).abs().max())
                if multiplier == 0.0
                else None
            )
            if zero_replay is not None:
                maximum_zero_replay = max(maximum_zero_replay, zero_replay)
            dose_records[key] = {
                "multiplier": multiplier,
                "noise_sigma": config.source_noise_sigma * multiplier,
                "task_metrics": metrics,
                "natural_utility": natural_gate,
                "natural_utility_pass": natural_pass,
                "posterior_sha256": hashlib.sha256(
                    posterior.float().contiguous().numpy().tobytes()
                ).hexdigest(),
                "zero_replay_maximum_absolute_posterior_error": zero_replay,
            }
            diagnostics[f"{regime}__dose_{key}__posterior"] = (
                posterior.float().numpy()
            )
        regimes[regime] = {
            "clean_task_metrics": clean_metrics,
            "source_clean_replay_maximum_absolute_error": source_replay,
            "doses": dose_records,
        }

    natural_seed_gates = {
        _dose_key(multiplier): all(
            regimes[regime]["doses"][_dose_key(multiplier)][
                "natural_utility_pass"
            ]
            for regime in REGIMES
        )
        for multiplier in config.dose_multipliers
    }
    state_unchanged = bool(
        source.calibrated._state_digest(system.model)
        == provenance["model_state_sha256"]
        and source.calibrated._module_digest(system)
        == provenance["system_state_sha256"]
    )
    finite = source._finite(regimes)
    zero_replay_pass = maximum_zero_replay <= config.zero_replay_tolerance
    source_replay_pass = (
        maximum_source_replay <= config.source_replay_tolerance
    )
    validity = bool(
        state_unchanged and finite and source_replay_pass and zero_replay_pass
    )
    diagnostics_path = result_dir / "stage1_diagnostics.npz"
    _write_npz(diagnostics_path, diagnostics)
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-noise-dose-stage1-{condition}-seed{seed}",
        "stage": "isotropic_localization",
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
            "observed_predecessor_result_sha256": predecessor["result_sha256"],
            "observed_predecessor_diagnostics": predecessor["diagnostics_path"],
            "observed_predecessor_diagnostics_sha256": predecessor[
                "diagnostics_sha256"
            ],
        },
        "dataset_hashes": runtime["dataset_hashes"],
        "source_noise_content_sha256": SOURCE_NOISE_CONTENT_SHA256,
        "isotropic_scaled_array_sha256": {
            _dose_key(multiplier): _scaled_array_digest(
                scaled_noise_arrays(base_arrays, multiplier),
                laws=("isotropic",),
            )
            for multiplier in config.dose_multipliers
        },
        "regimes": regimes,
        "natural_seed_gates": natural_seed_gates,
        "gates": {
            "source_clean_replay": source_replay_pass,
            "zero_dose_replay": zero_replay_pass,
            "state_unchanged": state_unchanged,
            "finite": finite,
            "validity": validity,
        },
        "analysis_seconds": time.perf_counter() - started,
        "artifacts": {
            "result": str(result_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": _sha256(diagnostics_path),
        },
    }
    _write_json(result_path, result)
    print(
        f"stage1 {condition} seed {seed}: doses={natural_seed_gates} "
        f"valid={validity}",
        flush=True,
    )
    return result


def select_prefix_valid_dose(
    stage1_results: list[Mapping[str, Any]],
    config: NoiseLawDoseLocalizationConfig,
) -> dict[str, Any]:
    expected = {
        (condition, seed)
        for condition in config.conditions
        for seed in config.seeds
    }
    indexed = {
        (str(result["condition"]), int(result["seed"])): result
        for result in stage1_results
    }
    if set(indexed) != expected:
        raise ValueError("stage1 population index changed")
    doses: dict[str, Any] = {}
    prefix_valid = True
    selected: float | None = None
    for multiplier in config.dose_multipliers:
        key = _dose_key(multiplier)
        arms = {
            condition: {
                "natural_utility_pass_count": sum(
                    int(indexed[(condition, seed)]["natural_seed_gates"][key])
                    for seed in config.seeds
                )
            }
            for condition in config.conditions
        }
        population_pass = all(
            arms[condition]["natural_utility_pass_count"]
            >= config.required_seed_passes
            for condition in config.conditions
        )
        if multiplier > 0.0:
            prefix_valid = bool(prefix_valid and population_pass)
            if prefix_valid:
                selected = multiplier
        doses[key] = {
            "multiplier": multiplier,
            "noise_sigma": config.source_noise_sigma * multiplier,
            "arms": arms,
            "joint_population_pass": population_pass,
            "prefix_valid": bool(multiplier > 0.0 and prefix_valid),
        }
    zero_key = _dose_key(0.0)
    zero_control_pass = bool(
        all(
            indexed[(condition, seed)]["natural_seed_gates"][zero_key]
            and indexed[(condition, seed)]["gates"]["zero_dose_replay"]
            for condition in config.conditions
            for seed in config.seeds
        )
    )
    integrity_valid = all(
        result["gates"]["validity"] for result in indexed.values()
    )
    return {
        "doses": doses,
        "selected_multiplier": selected,
        "selected_noise_sigma": (
            None if selected is None else config.source_noise_sigma * selected
        ),
        "zero_dose_control_pass": zero_control_pass,
        "integrity_valid": integrity_valid,
        "selection_uses_asymmetric_outcomes": False,
        "selection_rule": "largest_joint_population_prefix_valid_multiplier",
    }


def _stage2_seed_gates(
    regime_results: Mapping[str, Any],
    condition: str,
    config: NoiseLawDoseLocalizationConfig,
) -> dict[str, Any]:
    law_seed_gates = {
        law: all(
            regime_results[regime]["laws"][law]["natural_utility_pass"]
            and all(
                regime_results[regime]["laws"][law]["cuts"][cut][variant][
                    "task_gate"
                ]
                for cut in CUTS
                for variant in ("correct_action", "correct_twirl")
            )
            for regime in REGIMES
        )
        for law in LAWS
    }
    natural_seed_gates = {
        law: all(
            regime_results[regime]["laws"][law]["natural_utility_pass"]
            for regime in REGIMES
        )
        for law in LAWS
    }
    control_seed_gates = {
        law: any(
            all(
                regime_results[regime]["laws"][law]["cuts"][cut][variant][
                    "task_gate"
                ]
                for regime in REGIMES
            )
            for cut in CUTS
            for variant in ("orthogonal_action", "orthogonal_twirl")
        )
        for law in LAWS
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
        for law in LAWS
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
        for law in LAWS
    }
    analytic_feature_pass = bool(
        condition != "analytic_calibrated"
        or all(
            regime_results[regime]["laws"][law][
                "correct_action_feature_maximum_absolute_difference"
            ]
            <= config.analytic_feature_tolerance
            for regime in REGIMES
            for law in LAWS
        )
    )
    return {
        "law_seed_gates": law_seed_gates,
        "natural_seed_gates": natural_seed_gates,
        "control_seed_gates": control_seed_gates,
        "action_seed_gates": action_seed_gates,
        "twirl_seed_gates": twirl_seed_gates,
        "analytic_feature_pass": analytic_feature_pass,
    }


@torch.no_grad()
def _run_stage2_cell(
    *,
    system: source.calibrated.CalibratedTinyLLM,
    condition: str,
    seed: int,
    provenance: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    runtime: Mapping[str, Any],
    selected_arrays: Mapping[str, Mapping[str, torch.Tensor]],
    selected_multiplier: float,
    selected_arrays_sha256: str,
    selected_contract_sha256: str,
    config: NoiseLawDoseLocalizationConfig,
    implementation: str,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    fingerprint = _fingerprint(
        stage="selected_law_comparison",
        config=config,
        implementation=implementation,
        condition=condition,
        seed=seed,
        provenance=provenance,
        dataset_hashes=runtime["dataset_hashes"],
        selected_multiplier=selected_multiplier,
        selected_arrays_sha256=selected_arrays_sha256,
    )
    result_dir = output / "runs" / condition / f"seed_{seed}"
    result_path = result_dir / "stage2_result.json"
    reusable = _reusable_result(
        result_path,
        stage="selected_law_comparison",
        fingerprint=fingerprint,
        implementation=implementation,
    )
    if reusable is not None:
        print(f"resuming stage2 {condition} seed {seed}", flush=True)
        return reusable

    started = time.perf_counter()
    evaluation_config = _source_config(
        config, noise_sigma=config.source_noise_sigma * selected_multiplier
    )
    with np.load(predecessor["diagnostics_path"], allow_pickle=False) as loaded:
        source_diagnostics = {name: loaded[name] for name in loaded.files}
    regime_results: dict[str, Any] = {}
    diagnostics: dict[str, np.ndarray] = {}
    for regime in REGIMES:
        regime_result, arrays = source.analyze_regime(
            system,
            runtime["datasets"][regime],
            selected_arrays[regime],
            source_diagnostics,
            regime,
            runtime["task"],
            evaluation_config,
            device,
        )
        regime_results[regime] = regime_result
        diagnostics.update(
            {f"{regime}__{name}": value for name, value in arrays.items()}
        )
    seed_gates = _stage2_seed_gates(regime_results, condition, config)
    source_replay_pass = all(
        regime_results[regime]["source_clean_replay_maximum_absolute_error"]
        <= config.source_replay_tolerance
        for regime in REGIMES
    )
    cut_replay_pass = all(
        regime_results[regime]["laws"][law]["maximum_replay_error"]
        <= config.replay_tolerance
        for regime in REGIMES
        for law in LAWS
    )
    state_unchanged = bool(
        source.calibrated._state_digest(system.model)
        == provenance["model_state_sha256"]
        and source.calibrated._module_digest(system)
        == provenance["system_state_sha256"]
    )
    finite = source._finite(regime_results)
    validity = bool(
        source_replay_pass
        and cut_replay_pass
        and seed_gates["analytic_feature_pass"]
        and state_unchanged
        and finite
    )
    diagnostics_path = result_dir / "stage2_diagnostics.npz"
    _write_npz(diagnostics_path, diagnostics)
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-noise-dose-stage2-{condition}-seed{seed}",
        "stage": "selected_law_comparison",
        "status": "completed",
        "evidence_role": EVIDENCE_ROLE,
        "completed_at": _utc_now(),
        "condition": condition,
        "seed": seed,
        "configuration": _json_config(config),
        "implementation_sha256": implementation,
        "scientific_fingerprint": fingerprint,
        "selected_multiplier": selected_multiplier,
        "selected_noise_sigma": config.source_noise_sigma * selected_multiplier,
        "selected_arrays_sha256": selected_arrays_sha256,
        "selected_noise_law_contract_sha256": selected_contract_sha256,
        "provenance": {
            **provenance,
            "observed_predecessor_result": predecessor["result_path"],
            "observed_predecessor_result_sha256": predecessor["result_sha256"],
            "observed_predecessor_diagnostics": predecessor["diagnostics_path"],
            "observed_predecessor_diagnostics_sha256": predecessor[
                "diagnostics_sha256"
            ],
        },
        "dataset_hashes": runtime["dataset_hashes"],
        "regimes": regime_results,
        **seed_gates,
        "gates": {
            "source_clean_replay": source_replay_pass,
            "cut_replay": cut_replay_pass,
            "analytic_feature_invariance": seed_gates[
                "analytic_feature_pass"
            ],
            "state_unchanged": state_unchanged,
            "finite": finite,
            "validity": validity,
        },
        "analysis_seconds": time.perf_counter() - started,
        "artifacts": {
            "result": str(result_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": _sha256(diagnostics_path),
        },
    }
    _write_json(result_path, result)
    print(
        f"stage2 {condition} seed {seed}: laws={seed_gates['law_seed_gates']} "
        f"controls={seed_gates['control_seed_gates']} valid={validity}",
        flush=True,
    )
    return result


def aggregate_stage2(
    results: list[Mapping[str, Any]],
    config: NoiseLawDoseLocalizationConfig,
) -> dict[str, Any]:
    expected = {
        (condition, seed)
        for condition in config.conditions
        for seed in config.seeds
    }
    indexed = {
        (str(result["condition"]), int(result["seed"])): result
        for result in results
    }
    if set(indexed) != expected:
        raise ValueError("stage2 population index changed")
    arms = {
        condition: {
            "laws": {
                law: {
                    "joint_pass_count": sum(
                        int(indexed[(condition, seed)]["law_seed_gates"][law])
                        for seed in config.seeds
                    ),
                    "natural_utility_pass_count": sum(
                        int(
                            indexed[(condition, seed)]["natural_seed_gates"][
                                law
                            ]
                        )
                        for seed in config.seeds
                    ),
                    "control_pass_count": sum(
                        int(
                            indexed[(condition, seed)]["control_seed_gates"][law]
                        )
                        for seed in config.seeds
                    ),
                    "action_pass_counts": {
                        cut: sum(
                            int(
                                indexed[(condition, seed)]["action_seed_gates"][
                                    law
                                ][cut]
                            )
                            for seed in config.seeds
                        )
                        for cut in CUTS
                    },
                    "twirl_pass_counts": {
                        cut: sum(
                            int(
                                indexed[(condition, seed)]["twirl_seed_gates"][
                                    law
                                ][cut]
                            )
                            for seed in config.seeds
                        )
                        for cut in CUTS
                    },
                }
                for law in LAWS
            }
        }
        for condition in config.conditions
    }
    controls_pass = all(
        arms[condition]["laws"][law]["control_pass_count"]
        <= config.maximum_control_seed_passes
        for condition in config.conditions
        for law in LAWS
    )
    isotropic_pass = all(
        arms[condition]["laws"]["isotropic"]["joint_pass_count"]
        >= config.required_seed_passes
        for condition in config.conditions
    )
    all_laws_pass = all(
        arms[condition]["laws"][law]["joint_pass_count"]
        >= config.required_seed_passes
        for condition in config.conditions
        for law in LAWS
    )
    return {
        "arms": arms,
        "controls_pass": controls_pass,
        "isotropic_joint_population_pass": isotropic_pass,
        "all_laws_joint_population_pass": all_laws_pass,
        "integrity_valid": all(
            result["gates"]["validity"] for result in indexed.values()
        ),
    }


def classify_campaign(
    *,
    integrity_valid: bool,
    zero_dose_control_pass: bool,
    selected_multiplier: float | None,
    controls_pass: bool | None,
    isotropic_joint_population_pass: bool | None,
    all_laws_joint_population_pass: bool | None,
) -> tuple[str, bool, bool]:
    if not integrity_valid:
        return "invalid_integrity", False, False
    if not zero_dose_control_pass:
        return "invalid_zero_dose_control", False, False
    if selected_multiplier is None:
        return "no_common_nonzero_utility_window", False, False
    if controls_pass is not True:
        return "nonspecific_target_changing_control", False, False
    if isotropic_joint_population_pass is not True:
        return "isotropic_closure_fails_at_selected_dose", False, True
    if all_laws_joint_population_pass is not True:
        return "asymmetric_law_breaks_within_isotropic_window", False, True
    return "additive_noise_closed_at_selected_dose", True, True


def _result_entry(result: Mapping[str, Any]) -> dict[str, Any]:
    result_path = Path(result["artifacts"]["result"])
    return {
        "stage": result["stage"],
        "condition": result["condition"],
        "seed": result["seed"],
        "path": str(result_path),
        "result_sha256": _sha256(result_path),
        "diagnostics_path": result["artifacts"]["diagnostics"],
        "diagnostics_sha256": result["artifacts"]["diagnostics_sha256"],
        "scientific_fingerprint": result["scientific_fingerprint"],
        "validity": result["gates"]["validity"],
    }


def _campaign_reusable(
    campaign: Mapping[str, Any],
    config: NoiseLawDoseLocalizationConfig,
    implementation: str,
) -> bool:
    entries = list(campaign.get("stage1_results", [])) + list(
        campaign.get("stage2_results", [])
    )
    return bool(
        campaign.get("status") == "completed"
        and campaign.get("schema_version") == SCHEMA_VERSION
        and campaign.get("hypothesis_id") == HYPOTHESIS_ID
        and campaign.get("configuration") == _json_config(config)
        and campaign.get("implementation_sha256") == implementation
        and campaign.get("source_campaign_sha256") == SOURCE_CAMPAIGN_SHA256
        and campaign.get("source_noise_file_sha256") == SOURCE_NOISE_FILE_SHA256
        and campaign.get("source_noise_content_sha256")
        == SOURCE_NOISE_CONTENT_SHA256
        and all(
            Path(entry["path"]).is_file()
            and _sha256(Path(entry["path"])) == entry["result_sha256"]
            and Path(entry["diagnostics_path"]).is_file()
            and _sha256(Path(entry["diagnostics_path"]))
            == entry["diagnostics_sha256"]
            for entry in entries
        )
        and campaign.get("result_manifest_sha256") == _json_hash(entries)
    )


def run_campaign(
    config: NoiseLawDoseLocalizationConfig, output: Path
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

    source_campaign, source_campaign_path, base_arrays = _load_source_campaign(
        config
    )
    runtime = _load_runtime_sources(config)
    base_arrays = {
        regime: {
            law: base_arrays[regime][law][
                : len(runtime["datasets"][regime].calibration)
            ]
            for law in LAWS
        }
        for regime in REGIMES
    }
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    stage1_results: list[dict[str, Any]] = []
    stage1_reused = 0
    for condition in config.conditions:
        for seed in config.seeds:
            system, provenance = _load_system(runtime, condition, seed, device)
            result_path = (
                output / "runs" / condition / f"seed_{seed}" / "stage1_result.json"
            )
            was_present = result_path.is_file()
            result = _run_stage1_cell(
                system=system,
                condition=condition,
                seed=seed,
                provenance=provenance,
                predecessor=runtime["predecessor"][(condition, seed)],
                runtime=runtime,
                base_arrays=base_arrays,
                config=config,
                implementation=implementation,
                output=output,
                device=device,
            )
            stage1_results.append(result)
            stage1_reused += int(was_present)
            del system
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if _implementation_digest() != implementation:
                raise RuntimeError("dose-localization implementation changed")

    stage1 = select_prefix_valid_dose(stage1_results, config)
    selected_multiplier = stage1["selected_multiplier"]
    selected_arrays: dict[str, dict[str, torch.Tensor]] | None = None
    selected_arrays_sha256: str | None = None
    selected_contract: dict[str, Any] | None = None
    selected_contract_sha256: str | None = None
    stage2_results: list[dict[str, Any]] = []
    stage2_reused = 0
    stage2: dict[str, Any] | None = None
    can_run_stage2 = bool(
        stage1["integrity_valid"]
        and stage1["zero_dose_control_pass"]
        and selected_multiplier is not None
    )
    if can_run_stage2:
        assert selected_multiplier is not None
        selected_arrays = scaled_noise_arrays(base_arrays, selected_multiplier)
        selected_arrays_sha256 = _scaled_array_digest(selected_arrays)
        evaluation_config = _source_config(
            config,
            noise_sigma=config.source_noise_sigma * selected_multiplier,
        )
        selected_contract = source.noise_law_contract(
            runtime["datasets"], selected_arrays, evaluation_config
        )
        selected_contract_sha256 = _json_hash(selected_contract)
        if not selected_contract["pass"]:
            raise ValueError(
                f"selected noise-law contract failed: {selected_contract}"
            )
        for condition in config.conditions:
            for seed in config.seeds:
                system, provenance = _load_system(runtime, condition, seed, device)
                result_path = (
                    output
                    / "runs"
                    / condition
                    / f"seed_{seed}"
                    / "stage2_result.json"
                )
                was_present = result_path.is_file()
                result = _run_stage2_cell(
                    system=system,
                    condition=condition,
                    seed=seed,
                    provenance=provenance,
                    predecessor=runtime["predecessor"][(condition, seed)],
                    runtime=runtime,
                    selected_arrays=selected_arrays,
                    selected_multiplier=selected_multiplier,
                    selected_arrays_sha256=selected_arrays_sha256,
                    selected_contract_sha256=selected_contract_sha256,
                    config=config,
                    implementation=implementation,
                    output=output,
                    device=device,
                )
                stage2_results.append(result)
                stage2_reused += int(was_present)
                del system
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if _implementation_digest() != implementation:
                    raise RuntimeError("dose-localization implementation changed")
        stage2 = aggregate_stage2(stage2_results, config)

    integrity_valid = bool(
        stage1["integrity_valid"]
        and (
            stage2 is None
            or (
                selected_contract is not None
                and selected_contract["pass"]
                and stage2["integrity_valid"]
            )
        )
    )
    classification, primary_pass, primary_evaluable = classify_campaign(
        integrity_valid=integrity_valid,
        zero_dose_control_pass=stage1["zero_dose_control_pass"],
        selected_multiplier=selected_multiplier,
        controls_pass=None if stage2 is None else stage2["controls_pass"],
        isotropic_joint_population_pass=(
            None if stage2 is None else stage2["isotropic_joint_population_pass"]
        ),
        all_laws_joint_population_pass=(
            None if stage2 is None else stage2["all_laws_joint_population_pass"]
        ),
    )
    stage1_entries = [_result_entry(result) for result in stage1_results]
    stage2_entries = [_result_entry(result) for result in stage2_results]
    result_manifest = stage1_entries + stage2_entries
    peak_cuda = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": EVIDENCE_ROLE,
        "completed_at": _utc_now(),
        "configuration": _json_config(config),
        "implementation_sha256": implementation,
        "source_digests": source_digests,
        "source_campaign": str(source_campaign_path),
        "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
        "source_campaign_result_manifest_sha256": (
            SOURCE_RESULT_MANIFEST_SHA256
        ),
        "source_noise_file": str(
            Path(config.source_noise_root) / "noise_law_arrays.npz"
        ),
        "source_noise_file_sha256": SOURCE_NOISE_FILE_SHA256,
        "source_noise_content_sha256": SOURCE_NOISE_CONTENT_SHA256,
        "source_dvc_root": SOURCE_DVC_ROOT,
        "source_lakefs_commit": SOURCE_LAKEFS_COMMIT,
        "dataset_hashes": runtime["dataset_hashes"],
        "source_preflight_manifest": runtime["preflight_manifest"],
        "source_predecessor_campaign": str(runtime["predecessor_path"]),
        "stage1": stage1,
        "selected_arrays_sha256": selected_arrays_sha256,
        "selected_noise_law_contract": selected_contract,
        "selected_noise_law_contract_sha256": selected_contract_sha256,
        "stage2": stage2,
        "aggregates": {
            "classification": classification,
            "primary_hypothesis_pass": primary_pass,
            "primary_evaluable": primary_evaluable,
            "valid": bool(integrity_valid and stage1["zero_dose_control_pass"]),
            "integrity_valid": integrity_valid,
            "selected_multiplier": selected_multiplier,
            "selected_noise_sigma": stage1["selected_noise_sigma"],
            "zero_dose_control_pass": stage1["zero_dose_control_pass"],
            "required_seed_passes": config.required_seed_passes,
            "maximum_control_seed_passes": config.maximum_control_seed_passes,
        },
        "stage1_results": stage1_entries,
        "stage2_results": stage2_entries,
        "result_manifest_sha256": _json_hash(result_manifest),
        "summary": {
            "requested_stage1": len(config.conditions) * len(config.seeds),
            "completed_stage1": len(stage1_results),
            "reused_stage1": stage1_reused,
            "requested_stage2": (
                len(config.conditions) * len(config.seeds)
                if can_run_stage2
                else 0
            ),
            "completed_stage2": len(stage2_results),
            "reused_stage2": stage2_reused,
            "failed": 0,
            "excluded": 0,
            "trained_models": 0,
            "trained_frontends": 0,
            "trained_task_heads": 0,
            "fitted_noise_models": 0,
            "fitted_actions": 0,
            "fitted_observers": 0,
            "fitted_probes": 0,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "peak_cuda_memory_allocated_bytes": peak_cuda,
            "deterministic_algorithms": True,
        },
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "The sigma=0.05 source outcome is known and remains invalid for its original asymmetric-law comparison.",
            "Dose selection uses only isotropic natural utility and never asymmetric-law outcomes.",
            "The largest common prefix-valid registered dose is selected without inserting or tuning a dose.",
            "Stage 2 compares all laws once at the locked dose and is corrective evidence, not an independent replication.",
            "Correct action/twirl closure relative to noisy identity is distinct from natural utility relative to clean input.",
            "No model, front end, head, action, observer, probe, or noise model is trained or fitted.",
        ],
        "artifacts": {"campaign": str(campaign_path)},
    }
    del source_campaign
    _write_json(campaign_path, campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_noise_law_dose_localization/"
            "20260810_d10_preregistered"
        ),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--conditions", type=_strings, default=CONDITIONS)
    parser.add_argument("--seeds", type=_ints, default=SEEDS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--required-seed-passes", type=int, default=4)
    parser.add_argument("--maximum-control-seed-passes", type=int, default=1)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = NoiseLawDoseLocalizationConfig(
        conditions=args.conditions,
        seeds=args.seeds,
        batch_size=args.batch_size,
        sample_limit=args.sample_limit,
        required_seed_passes=args.required_seed_passes,
        maximum_control_seed_passes=args.maximum_control_seed_passes,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
