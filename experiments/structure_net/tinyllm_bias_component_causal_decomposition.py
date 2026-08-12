#!/usr/bin/env python3
"""Decompose the selected TinyLLM biased-noise failure into exact components."""

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

import experiments.structure_net.tinyllm_noise_law_dose_localization as dose


SCHEMA_VERSION = "nal.tinyllm-bias-component-causal-decomposition.v1"
HYPOTHESIS_ID = "tinyllm-bias-component-causal-decomposition-v1"
EVIDENCE_ROLE = "preregistered_frozen_bias_component_intervention"
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-bias-component-causal-decomposition-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "a3052f55181f72fb9b53d4bc8ad7a42fe28d5762acd6ee4ffbb6a0d31e81d85e"
)
SOURCE_RUNNER_SHA256 = (
    "39a72dd535f96f13bae644c74096b298b85fb8587d980211dc489ed463aeb725"
)
SOURCE_CAMPAIGN_SHA256 = (
    "9b05823ebdb88bd828f27699da596dc5e7dcf0c4af5e13f1664fa70e5111f9bd"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "bab495e0f3985c8358d90344fc3cf02986b6e138adaeb9fa01c1d38c482187c2"
)
SOURCE_RESULT_MANIFEST_SHA256 = (
    "976545c812e428ea4b020ca46a88643cb741a6ad5c7797389e9a5e6ca81f7562"
)
SOURCE_SELECTED_ARRAYS_SHA256 = (
    "740c5c30f01c482fa799db1865a11c069ad3b59f474879a59f1906b94f4130f3"
)
SOURCE_NOISE_FILE_SHA256 = (
    "d3771eac8e29f7940df7feaedebe74a5a78fb273cda2e70928c9be9e37ff3ba6"
)
SOURCE_NOISE_CONTENT_SHA256 = (
    "93df61bc76ed073ea241c9450e7ec3523e7a98b5ac06e58d7e920a5df07d70aa"
)
SOURCE_DVC_ROOT = "c07286d2b9710cd68228cd21f487e425.dir"
SOURCE_LAKEFS_COMMIT = (
    "d4fb92ef41e39d0cc672d672e55c9192ea0e9dcf01597b1a549efcf973577061"
)
CONDITIONS = dose.CONDITIONS
SEEDS = dose.SEEDS
REGIMES = dose.REGIMES
VARIANTS = ("centered", "mean_plus", "full_plus", "full_minus")
NEW_VARIANTS = ("centered", "mean_plus", "full_minus")


@dataclass(frozen=True)
class BiasComponentCausalConfig:
    source_dose_root: str = (
        "data/experiments/tinyllm_noise_law_dose_localization/"
        "20260810_d10_preregistered"
    )
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
    selected_multiplier: float = 0.625
    source_noise_sigma: float = 0.05
    selected_noise_sigma: float = 0.03125
    component_reconstruction_tolerance: float = 2e-7
    sign_energy_relative_tolerance: float = 0.02
    source_posterior_replay_tolerance: float = 2e-6
    source_metric_replay_tolerance: float = 2e-6
    natural_accuracy_loss_ceiling: float = 0.05
    natural_circular_error_increase_ceiling: float = math.pi / 16.0
    natural_cross_entropy_increase_ceiling: float = 0.10
    required_seed_passes: int = 4
    batch_size: int = 256
    sample_limit: int | None = None
    device: str = "cuda:2"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.conditions or set(self.conditions).difference(CONDITIONS):
            raise ValueError("unknown or empty structured condition")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("checkpoint seeds must be non-empty and distinct")
        if self.selected_multiplier != 0.625:
            raise ValueError("the selected multiplier is frozen at 0.625")
        if self.source_noise_sigma != 0.05 or self.selected_noise_sigma != 0.03125:
            raise ValueError("the source and selected noise scales are frozen")
        if self.batch_size < 1:
            raise ValueError("batch size must be positive")
        if self.sample_limit is not None and self.sample_limit < 8:
            raise ValueError("sample limit must be at least eight")
        if not 1 <= self.required_seed_passes <= len(self.seeds):
            raise ValueError("required seed count is outside selected population")
        if not self.allow_underpowered:
            expected = (
                self.conditions == CONDITIONS
                and self.seeds == SEEDS
                and self.batch_size == 256
                and self.sample_limit is None
                and self.required_seed_passes == 4
            )
            if not expected:
                raise ValueError("primary bias-component configuration is fixed")


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


def _json_config(config: BiasComponentCausalConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _source_digests() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "preregistration": PREREGISTRATION_PATH,
        "source_dose_runner": Path(dose.__file__),
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "source_dose_runner": SOURCE_RUNNER_SHA256,
    }
    for name, digest in expected.items():
        if _sha256(paths[name]) != digest:
            raise ValueError(f"frozen {name} changed: {paths[name]}")
    return {name: _sha256(path) for name, path in paths.items()}


def _implementation_digest(digests: Mapping[str, str] | None = None) -> str:
    return _json_hash(dict(digests or _source_digests()))


def _dose_config(config: BiasComponentCausalConfig) -> dose.NoiseLawDoseLocalizationConfig:
    return dose.NoiseLawDoseLocalizationConfig(
        source_noise_root=config.source_noise_root,
        source_observed_root=config.source_observed_root,
        source_closure_root=config.source_closure_root,
        calibrated_source_root=config.calibrated_source_root,
        conditions=config.conditions,
        seeds=config.seeds,
        required_seed_passes=min(config.required_seed_passes, len(config.seeds)),
        batch_size=config.batch_size,
        sample_limit=config.sample_limit,
        device=config.device,
        allow_underpowered=True,
    )


def _source_capture_config(
    config: BiasComponentCausalConfig,
) -> dose.source.NoiseLawObservedTwirlConfig:
    return dose._source_config(
        _dose_config(config), noise_sigma=config.selected_noise_sigma
    )


def _load_source_campaign(
    config: BiasComponentCausalConfig,
) -> tuple[
    dict[str, Any],
    Path,
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int], Path],
]:
    root = Path(config.source_dose_root)
    campaign_path = root / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    aggregates = campaign.get("aggregates", {})
    stage2 = campaign.get("stage2", {})
    if (
        _sha256(campaign_path) != SOURCE_CAMPAIGN_SHA256
        or campaign.get("schema_version") != dose.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != dose.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256")
        != SOURCE_IMPLEMENTATION_SHA256
        or campaign.get("result_manifest_sha256")
        != SOURCE_RESULT_MANIFEST_SHA256
        or campaign.get("selected_arrays_sha256")
        != SOURCE_SELECTED_ARRAYS_SHA256
        or aggregates.get("classification")
        != "asymmetric_law_breaks_within_isotropic_window"
        or aggregates.get("primary_evaluable") is not True
        or aggregates.get("integrity_valid") is not True
        or aggregates.get("selected_multiplier") != 0.625
        or aggregates.get("selected_noise_sigma") != 0.03125
        or stage2.get("controls_pass") is not True
        or stage2.get("integrity_valid") is not True
        or len(campaign.get("stage2_results", [])) != 10
    ):
        raise ValueError(f"invalid selected-dose source {campaign_path}")
    expected_counts = {
        "analytic_calibrated": 1,
        "learned_calibrated_equivariant": 3,
    }
    for condition, expected in expected_counts.items():
        biased = stage2["arms"][condition]["laws"]["lab_biased"]
        if (
            biased.get("joint_pass_count") != expected
            or biased.get("natural_utility_pass_count") != expected
            or biased.get("control_pass_count") != 0
            or biased.get("action_pass_counts") != {cut: 5 for cut in dose.CUTS}
            or biased.get("twirl_pass_counts") != {cut: 5 for cut in dose.CUTS}
        ):
            raise ValueError("selected-dose biased source counts changed")
    expected_cells = {
        (condition, seed) for condition in CONDITIONS for seed in SEEDS
    }
    details: dict[tuple[str, int], dict[str, Any]] = {}
    diagnostics: dict[tuple[str, int], Path] = {}
    for entry in campaign["stage2_results"]:
        result_path = Path(entry["path"])
        diagnostics_path = Path(entry["diagnostics_path"])
        detail = json.loads(result_path.read_text(encoding="utf-8"))
        cell = (str(detail.get("condition")), int(detail.get("seed", -1)))
        if (
            _sha256(result_path) != entry.get("result_sha256")
            or _sha256(diagnostics_path) != entry.get("diagnostics_sha256")
            or detail.get("status") != "completed"
            or detail.get("stage") != "selected_law_comparison"
            or detail.get("implementation_sha256")
            != SOURCE_IMPLEMENTATION_SHA256
            or detail.get("selected_multiplier") != 0.625
            or detail.get("selected_noise_sigma") != 0.03125
            or detail.get("selected_arrays_sha256")
            != SOURCE_SELECTED_ARRAYS_SHA256
            or detail.get("gates", {}).get("validity") is not True
            or any(detail.get("control_seed_gates", {}).values())
        ):
            raise ValueError(f"invalid selected-dose result {result_path}")
        details[cell] = detail
        diagnostics[cell] = diagnostics_path
    if set(details) != expected_cells:
        raise ValueError("selected-dose source population changed")
    return campaign, campaign_path, details, diagnostics


def _load_base_noise(
    config: BiasComponentCausalConfig,
) -> dict[str, dict[str, torch.Tensor]]:
    noise_path = Path(config.source_noise_root) / "noise_law_arrays.npz"
    if _sha256(noise_path) != SOURCE_NOISE_FILE_SHA256:
        raise ValueError("frozen source noise file changed")
    with np.load(noise_path, allow_pickle=False) as loaded:
        arrays = {
            regime: {
                law: torch.from_numpy(loaded[f"{regime}__{law}"].copy())
                for law in dose.LAWS
            }
            for regime in REGIMES
        }
    content = dose.source._noise_arrays_digest(
        {
            f"{regime}__{law}": arrays[regime][law]
            for regime in REGIMES
            for law in dose.LAWS
        }
    )
    if content != SOURCE_NOISE_CONTENT_SHA256:
        raise ValueError("frozen source noise content changed")
    return arrays


def construct_components(
    base_arrays: Mapping[str, Mapping[str, torch.Tensor]],
    config: BiasComponentCausalConfig,
) -> dict[str, dict[str, torch.Tensor]]:
    output: dict[str, dict[str, torch.Tensor]] = {}
    for regime in REGIMES:
        isotropic = base_arrays[regime]["isotropic"] * config.selected_multiplier
        centered = isotropic / math.sqrt(2.0)
        mean_plus = torch.zeros_like(centered)
        mean_plus[..., 0] = config.selected_noise_sigma
        output[regime] = {
            "centered": centered,
            "mean_plus": mean_plus,
            "full_plus": centered + mean_plus,
            "full_minus": centered - mean_plus,
        }
    return output


def component_contract(
    components: Mapping[str, Mapping[str, torch.Tensor]],
    base_arrays: Mapping[str, Mapping[str, torch.Tensor]],
    config: BiasComponentCausalConfig,
) -> dict[str, Any]:
    regimes: dict[str, Any] = {}
    for regime in REGIMES:
        values = components[regime]
        expected_full = (
            base_arrays[regime]["lab_biased"] * config.selected_multiplier
        )
        expected_centered = (
            base_arrays[regime]["isotropic"]
            * config.selected_multiplier
            / math.sqrt(2.0)
        )
        expected_mean = torch.zeros_like(values["mean_plus"])
        expected_mean[..., 0] = config.selected_noise_sigma
        full_error = float((values["full_plus"] - expected_full).abs().max())
        centered_error = float(
            (values["centered"] - expected_centered).abs().max()
        )
        mean_error = float((values["mean_plus"] - expected_mean).abs().max())
        plus_energy = float(values["full_plus"].double().square().sum(-1).mean())
        minus_energy = float(values["full_minus"].double().square().sum(-1).mean())
        energy_relative_difference = abs(plus_energy - minus_energy) / max(
            plus_energy, minus_energy, 1e-12
        )
        record = {
            "full_plus_reconstruction_maximum_absolute_error": full_error,
            "centered_reconstruction_maximum_absolute_error": centered_error,
            "mean_plus_reconstruction_maximum_absolute_error": mean_error,
            "full_plus_empirical_squared_planar_norm": plus_energy,
            "full_minus_empirical_squared_planar_norm": minus_energy,
            "sign_energy_relative_difference": energy_relative_difference,
            "pass": bool(
                full_error <= config.component_reconstruction_tolerance
                and centered_error <= config.component_reconstruction_tolerance
                and mean_error <= config.component_reconstruction_tolerance
                and energy_relative_difference
                <= config.sign_energy_relative_tolerance
            ),
        }
        regimes[regime] = record
    return {
        "selected_multiplier": config.selected_multiplier,
        "selected_noise_sigma": config.selected_noise_sigma,
        "no_new_random_draws": True,
        "regimes": regimes,
        "pass": all(record["pass"] for record in regimes.values()),
    }


def _fingerprint(
    config: BiasComponentCausalConfig,
    implementation: str,
    condition: str,
    seed: int,
    provenance: Mapping[str, Any],
    dataset_hashes: Mapping[str, str],
    component_contract_sha256: str,
    source_detail_sha256: str,
    source_diagnostics_sha256: str,
) -> str:
    return _json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "configuration": _json_config(config),
            "implementation_sha256": implementation,
            "condition": condition,
            "seed": seed,
            "provenance": provenance,
            "dataset_hashes": dict(dataset_hashes),
            "component_contract_sha256": component_contract_sha256,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "source_selected_arrays_sha256": SOURCE_SELECTED_ARRAYS_SHA256,
            "source_detail_sha256": source_detail_sha256,
            "source_diagnostics_sha256": source_diagnostics_sha256,
        }
    )


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
        or not diagnostics.is_file()
        or _sha256(diagnostics)
        != value.get("artifacts", {}).get("diagnostics_sha256")
    ):
        raise ValueError(f"incompatible completed bias-component result {path}")
    return value


def _metric_error(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return max(abs(float(left[key]) - float(right[key])) for key in left)


@torch.no_grad()
def analyze_cell(
    *,
    system: dose.source.calibrated.CalibratedTinyLLM,
    condition: str,
    seed: int,
    provenance: Mapping[str, Any],
    source_detail: Mapping[str, Any],
    source_detail_path: Path,
    source_diagnostics_path: Path,
    runtime: Mapping[str, Any],
    components: Mapping[str, Mapping[str, torch.Tensor]],
    contract_sha256: str,
    config: BiasComponentCausalConfig,
    implementation: str,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    source_detail_sha256 = _sha256(source_detail_path)
    source_diagnostics_sha256 = _sha256(source_diagnostics_path)
    fingerprint = _fingerprint(
        config,
        implementation,
        condition,
        seed,
        provenance,
        runtime["dataset_hashes"],
        contract_sha256,
        source_detail_sha256,
        source_diagnostics_sha256,
    )
    result_dir = output / "runs" / condition / f"seed_{seed}"
    result_path = result_dir / "result.json"
    reusable = _reusable_result(result_path, fingerprint, implementation)
    if reusable is not None:
        print(f"resuming {condition} seed {seed}", flush=True)
        return reusable

    started = time.perf_counter()
    capture_config = _source_capture_config(config)
    with np.load(source_diagnostics_path, allow_pickle=False) as loaded:
        stored = {name: loaded[name] for name in loaded.files}
    regime_results: dict[str, Any] = {}
    diagnostics: dict[str, np.ndarray] = {}
    maximum_clean_replay = 0.0
    maximum_metric_replay = 0.0
    for regime in REGIMES:
        dataset = runtime["datasets"][regime]
        count = len(dataset.calibration)
        input_ids = dataset.paired.circle.input_ids
        sensor = dose.source.calibrated.decode_sensor_tokens(
            input_ids, runtime["task"]
        )
        _, clean_posterior, _ = dose.source._capture(
            system,
            input_ids,
            sensor,
            dataset.calibration,
            runtime["task"],
            capture_config,
            device,
        )
        stored_clean = torch.from_numpy(
            stored[f"{regime}__clean_posterior"][:count]
        ).double()
        stored_full_plus = torch.from_numpy(
            stored[f"{regime}__lab_biased__identity_posterior"][:count]
        ).double()
        clean_replay = float((clean_posterior - stored_clean).abs().max())
        maximum_clean_replay = max(maximum_clean_replay, clean_replay)
        clean_metrics = dose.source.closure.posterior_metrics(
            clean_posterior, dataset
        )
        stored_clean_metrics = dose.source.closure.posterior_metrics(
            stored_clean, dataset
        )
        full_plus_metrics = dose.source.closure.posterior_metrics(
            stored_full_plus, dataset
        )
        metric_replay = _metric_error(clean_metrics, stored_clean_metrics)
        if config.sample_limit is None:
            metric_replay = max(
                metric_replay,
                _metric_error(
                    full_plus_metrics,
                    source_detail["regimes"][regime]["laws"]["lab_biased"][
                        "noisy_identity_task_metrics"
                    ],
                ),
            )
        maximum_metric_replay = max(maximum_metric_replay, metric_replay)
        full_plus_pass, full_plus_gate = dose.source._natural_gate(
            full_plus_metrics, clean_metrics, capture_config
        )
        variant_records: dict[str, Any] = {
            "full_plus": {
                "source_reused": True,
                "task_metrics": full_plus_metrics,
                "natural_utility": full_plus_gate,
                "natural_utility_pass": full_plus_pass,
                "posterior_js_from_clean": dose.source.closure.jensen_shannon(
                    stored_full_plus, clean_posterior
                ),
            }
        }
        variant_posteriors: dict[str, torch.Tensor] = {
            "full_plus": stored_full_plus
        }
        for variant in NEW_VARIANTS:
            noisy_sensor = sensor.clone()
            noisy_sensor[..., :2] += components[regime][variant][:count]
            _, posterior, _ = dose.source._capture(
                system,
                input_ids,
                noisy_sensor,
                dataset.calibration,
                runtime["task"],
                capture_config,
                device,
            )
            metrics = dose.source.closure.posterior_metrics(posterior, dataset)
            passed, gate = dose.source._natural_gate(
                metrics, clean_metrics, capture_config
            )
            variant_records[variant] = {
                "source_reused": False,
                "task_metrics": metrics,
                "natural_utility": gate,
                "natural_utility_pass": passed,
                "posterior_js_from_clean": dose.source.closure.jensen_shannon(
                    posterior, clean_posterior
                ),
            }
            variant_posteriors[variant] = posterior
            diagnostics[f"{regime}__{variant}__posterior"] = (
                posterior.float().numpy()
            )
        plus_bins = stored_full_plus.argmax(1)
        minus_bins = variant_posteriors["full_minus"].argmax(1)
        sign_disagreement = float((plus_bins != minus_bins).double().mean())
        regime_results[regime] = {
            "clean_task_metrics": clean_metrics,
            "clean_posterior_replay_maximum_absolute_error": clean_replay,
            "source_metric_replay_maximum_absolute_error": metric_replay,
            "variants": variant_records,
            "full_plus_vs_full_minus_predicted_bin_disagreement_fraction": (
                sign_disagreement
            ),
        }
        diagnostics[f"{regime}__clean_posterior"] = clean_posterior.float().numpy()
        diagnostics[f"{regime}__source_full_plus_posterior"] = (
            stored_full_plus.float().numpy()
        )

    variant_seed_gates = {
        variant: all(
            regime_results[regime]["variants"][variant]["natural_utility_pass"]
            for regime in REGIMES
        )
        for variant in VARIANTS
    }
    source_seed_gate_match = bool(
        variant_seed_gates["full_plus"]
        == source_detail["law_seed_gates"]["lab_biased"]
    )
    state_unchanged = bool(
        dose.source.calibrated._state_digest(system.model)
        == provenance["model_state_sha256"]
        and dose.source.calibrated._module_digest(system)
        == provenance["system_state_sha256"]
    )
    finite = dose.source._finite(regime_results)
    clean_replay_pass = (
        maximum_clean_replay <= config.source_posterior_replay_tolerance
    )
    metric_replay_pass = (
        maximum_metric_replay <= config.source_metric_replay_tolerance
    )
    validity = bool(
        clean_replay_pass
        and metric_replay_pass
        and source_seed_gate_match
        and state_unchanged
        and finite
    )
    diagnostics_path = result_dir / "diagnostics.npz"
    _write_npz(diagnostics_path, diagnostics)
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-bias-component-{condition}-seed{seed}",
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
            "source_dose_result": str(source_detail_path),
            "source_dose_result_sha256": source_detail_sha256,
            "source_dose_diagnostics": str(source_diagnostics_path),
            "source_dose_diagnostics_sha256": source_diagnostics_sha256,
        },
        "dataset_hashes": runtime["dataset_hashes"],
        "component_contract_sha256": contract_sha256,
        "regimes": regime_results,
        "variant_seed_gates": variant_seed_gates,
        "gates": {
            "clean_posterior_replay": clean_replay_pass,
            "source_metric_replay": metric_replay_pass,
            "source_full_plus_seed_gate_match": source_seed_gate_match,
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
        f"{condition} seed {seed}: variants={variant_seed_gates} valid={validity}",
        flush=True,
    )
    return result


def aggregate_results(
    results: list[Mapping[str, Any]], config: BiasComponentCausalConfig
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
        raise ValueError("bias-component result population changed")
    arms = {
        condition: {
            "variants": {
                variant: {
                    "natural_utility_pass_count": sum(
                        int(
                            indexed[(condition, seed)]["variant_seed_gates"][
                                variant
                            ]
                        )
                        for seed in config.seeds
                    )
                }
                for variant in VARIANTS
            }
        }
        for condition in config.conditions
    }
    population_passes = {
        variant: {
            condition: arms[condition]["variants"][variant][
                "natural_utility_pass_count"
            ]
            >= config.required_seed_passes
            for condition in config.conditions
        }
        for variant in VARIANTS
    }
    centered_both = all(population_passes["centered"].values())
    mean_plus_fails_both = all(
        not passed for passed in population_passes["mean_plus"].values()
    )
    full_plus_fails_both = all(
        not passed for passed in population_passes["full_plus"].values()
    )
    full_minus_passes = population_passes["full_minus"]
    if all(full_minus_passes.values()):
        sign_classification = "positive_direction_specific"
    elif all(not passed for passed in full_minus_passes.values()):
        sign_classification = "bidirectional_mean_magnitude"
    else:
        sign_classification = "arm_specific_directional"
    return {
        "arms": arms,
        "population_passes": population_passes,
        "centered_passes_both_arms": centered_both,
        "mean_plus_fails_both_arms": mean_plus_fails_both,
        "full_plus_fails_both_arms": full_plus_fails_both,
        "sign_classification": sign_classification,
        "integrity_valid": all(
            result["gates"]["validity"] for result in indexed.values()
        ),
    }


def classify_campaign(
    *,
    integrity_valid: bool,
    component_contract_pass: bool,
    centered_passes_both_arms: bool,
    mean_plus_fails_both_arms: bool,
    full_plus_fails_both_arms: bool,
    mean_plus_passes_both_arms: bool,
) -> tuple[str, bool]:
    if not integrity_valid:
        return "invalid_integrity", False
    if not component_contract_pass:
        return "invalid_component_reconstruction", False
    if not centered_passes_both_arms:
        return "centered_stochastic_breaks_utility", False
    if mean_plus_fails_both_arms:
        return "deterministic_mean_sufficient", True
    if mean_plus_passes_both_arms and full_plus_fails_both_arms:
        return "mean_noise_interaction", False
    return "arm_specific_or_underdetermined", False


def _result_entry(result: Mapping[str, Any]) -> dict[str, Any]:
    result_path = Path(result["artifacts"]["result"])
    return {
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
    config: BiasComponentCausalConfig,
    implementation: str,
) -> bool:
    entries = campaign.get("results", [])
    return bool(
        campaign.get("status") == "completed"
        and campaign.get("schema_version") == SCHEMA_VERSION
        and campaign.get("hypothesis_id") == HYPOTHESIS_ID
        and campaign.get("configuration") == _json_config(config)
        and campaign.get("implementation_sha256") == implementation
        and campaign.get("source_campaign_sha256") == SOURCE_CAMPAIGN_SHA256
        and campaign.get("source_selected_arrays_sha256")
        == SOURCE_SELECTED_ARRAYS_SHA256
        and campaign.get("result_manifest_sha256") == _json_hash(entries)
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
    config: BiasComponentCausalConfig, output: Path
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

    source_campaign, source_campaign_path, source_details, source_diagnostics = (
        _load_source_campaign(config)
    )
    base_arrays = _load_base_noise(config)
    full_components = construct_components(base_arrays, config)
    contract = component_contract(full_components, base_arrays, config)
    contract_sha256 = _json_hash(contract)
    if not contract["pass"]:
        raise ValueError(f"component construction contract failed: {contract}")
    runtime = dose._load_runtime_sources(_dose_config(config))
    components = {
        regime: {
            variant: full_components[regime][variant][
                : len(runtime["datasets"][regime].calibration)
            ]
            for variant in VARIANTS
        }
        for regime in REGIMES
    }

    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    results: list[dict[str, Any]] = []
    reused = 0
    for condition in config.conditions:
        for seed in config.seeds:
            system, provenance = dose._load_system(runtime, condition, seed, device)
            source_detail = source_details[(condition, seed)]
            source_detail_path = Path(source_detail["artifacts"]["result"])
            result_path = output / "runs" / condition / f"seed_{seed}" / "result.json"
            was_present = result_path.is_file()
            result = analyze_cell(
                system=system,
                condition=condition,
                seed=seed,
                provenance=provenance,
                source_detail=source_detail,
                source_detail_path=source_detail_path,
                source_diagnostics_path=source_diagnostics[(condition, seed)],
                runtime=runtime,
                components=components,
                contract_sha256=contract_sha256,
                config=config,
                implementation=implementation,
                output=output,
                device=device,
            )
            results.append(result)
            reused += int(was_present)
            del system
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if _implementation_digest() != implementation:
                raise RuntimeError("bias-component implementation changed")

    population = aggregate_results(results, config)
    mean_plus_passes_both = all(
        population["population_passes"]["mean_plus"].values()
    )
    classification, primary_pass = classify_campaign(
        integrity_valid=population["integrity_valid"],
        component_contract_pass=contract["pass"],
        centered_passes_both_arms=population["centered_passes_both_arms"],
        mean_plus_fails_both_arms=population["mean_plus_fails_both_arms"],
        full_plus_fails_both_arms=population["full_plus_fails_both_arms"],
        mean_plus_passes_both_arms=mean_plus_passes_both,
    )
    entries = [_result_entry(result) for result in results]
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
        "source_selected_arrays_sha256": SOURCE_SELECTED_ARRAYS_SHA256,
        "source_noise_file_sha256": SOURCE_NOISE_FILE_SHA256,
        "source_noise_content_sha256": SOURCE_NOISE_CONTENT_SHA256,
        "source_dvc_root": SOURCE_DVC_ROOT,
        "source_lakefs_commit": SOURCE_LAKEFS_COMMIT,
        "dataset_hashes": runtime["dataset_hashes"],
        "source_preflight_manifest": runtime["preflight_manifest"],
        "component_contract": contract,
        "component_contract_sha256": contract_sha256,
        "population": population,
        "aggregates": {
            "classification": classification,
            "primary_hypothesis_pass": primary_pass,
            "primary_evaluable": True,
            "valid": bool(population["integrity_valid"] and contract["pass"]),
            "integrity_valid": population["integrity_valid"],
            "component_contract_pass": contract["pass"],
            "sign_classification": population["sign_classification"],
            "required_seed_passes": config.required_seed_passes,
        },
        "results": entries,
        "result_manifest_sha256": _json_hash(entries),
        "summary": {
            "requested": len(config.conditions) * len(config.seeds),
            "completed": len(results),
            "failed": 0,
            "excluded": 0,
            "reused": reused,
            "trained_models": 0,
            "trained_frontends": 0,
            "trained_task_heads": 0,
            "fitted_noise_models": 0,
            "fitted_denoisers": 0,
            "fitted_actions": 0,
            "fitted_observers": 0,
            "fitted_probes": 0,
            "new_random_draws": 0,
            "new_forward_variants_per_system_shift": len(NEW_VARIANTS),
            "reused_source_variants_per_system_shift": 2,
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
            "The component arrays are exact algebraic transformations of the frozen selected-dose draw; no randomness is added.",
            "Clean and full-positive-bias posteriors are reused from the sealed source campaign; only centered, mean-only, and sign-reversed full variants receive new forwards.",
            "Mean-only has lower expected energy than the full biased law, so its failure establishes sufficiency but not an equal-energy effect size.",
            "The sign-reversed variant is secondary and cannot rescue the primary mean-sufficiency gate.",
            "The inherited action/twirl mechanics are pinned controls and are not rerun.",
            "No model, front end, head, denoiser, action, observer, probe, or noise process is trained or fitted.",
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
            "data/experiments/tinyllm_bias_component_causal_decomposition/"
            "20260810_d10_preregistered"
        ),
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--conditions", type=_strings, default=CONDITIONS)
    parser.add_argument("--seeds", type=_ints, default=SEEDS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--required-seed-passes", type=int, default=4)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = BiasComponentCausalConfig(
        conditions=args.conditions,
        seeds=args.seeds,
        batch_size=args.batch_size,
        sample_limit=args.sample_limit,
        required_seed_passes=args.required_seed_passes,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
