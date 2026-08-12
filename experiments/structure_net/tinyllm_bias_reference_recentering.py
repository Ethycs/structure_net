#!/usr/bin/env python3
"""Test an exact observed shared-bias pilot at the frozen TinyLLM sensor front."""

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

import experiments.structure_net.tinyllm_bias_component_causal_decomposition as bias


SCHEMA_VERSION = "nal.tinyllm-bias-reference-recentering.v2"
HYPOTHESIS_ID = "tinyllm-bias-reference-recentering-v2"
EVIDENCE_ROLE = (
    "preregistered_pre_model_numerical_corrective_bias_reference_intervention"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-bias-reference-recentering-v2-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "e2d04b2852bffaac0ce190a4245f54a0e587a6d8323468afa91b922ef7c2c86b"
)
SOURCE_RUNNER_SHA256 = (
    "eba5182082d8604fba47d65fc0f64706b00ac9f4fde6dbf45c63fca56ed44bb5"
)
SOURCE_CAMPAIGN_SHA256 = (
    "9f7fdf98e83a320d5d49d9191e6a0f0cd6f872f32f406381c5a290f517dbed4b"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "c1b340bf1e29e485d2c254902ac6aaab87abdd594e2f46875a5e828fff415c98"
)
SOURCE_RESULT_MANIFEST_SHA256 = (
    "17d614cadfeca5e019258578ad9abe8dc269f899f7144e712e8154f7988ce07b"
)
SOURCE_COMPONENT_CONTRACT_SHA256 = (
    "26b8ad368fe8d1af811f2ff62d4874545c6d90b3aa5d376a9a59002092342b2f"
)
SOURCE_DVC_ROOT = "e3bfc6a9401916ffc7f942678044fb0a.dir"
SOURCE_LAKEFS_COMMIT = (
    "a0f6b67d7aad58dc96de58406abf7064728613e73134ba4959e18dd46c0cc92a"
)
CONDITIONS = bias.CONDITIONS
SEEDS = bias.SEEDS
REGIMES = bias.REGIMES
VARIANTS = (
    "source_full_plus",
    "recenter_correct",
    "recenter_wrong_sign",
    "recenter_target_changing",
)
NEW_VARIANTS = (
    "recenter_correct",
    "recenter_wrong_sign",
    "recenter_target_changing",
)


@dataclass(frozen=True)
class BiasReferenceRecenteringConfig:
    source_component_root: str = (
        "data/experiments/tinyllm_bias_component_causal_decomposition/"
        "20260810_d10_preregistered"
    )
    conditions: tuple[str, ...] = CONDITIONS
    seeds: tuple[int, ...] = SEEDS
    selected_noise_sigma: float = 0.03125
    construction_tolerance: float = 2e-7
    feature_equivalence_tolerance: float = 1e-6
    posterior_replay_tolerance: float = 2e-6
    metric_replay_tolerance: float = 2e-6
    action_involution_tolerance: float = 2e-6
    minimum_target_changing_feature_rms: float = 0.50
    natural_accuracy_loss_ceiling: float = 0.05
    natural_circular_error_increase_ceiling: float = math.pi / 16.0
    natural_cross_entropy_increase_ceiling: float = 0.10
    required_seed_passes: int = 4
    maximum_control_seed_passes: int = 1
    batch_size: int = 256
    sample_limit: int | None = None
    device: str = "cuda:2"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.conditions or set(self.conditions).difference(CONDITIONS):
            raise ValueError("unknown or empty structured condition")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("checkpoint seeds must be non-empty and distinct")
        if self.selected_noise_sigma != 0.03125:
            raise ValueError("the selected noise scale is frozen")
        if self.batch_size < 1:
            raise ValueError("batch size must be positive")
        if self.sample_limit is not None and self.sample_limit < 8:
            raise ValueError("sample limit must be at least eight")
        if not 1 <= self.required_seed_passes <= len(self.seeds):
            raise ValueError("required seed count is outside selected population")
        if not 0 <= self.maximum_control_seed_passes <= len(self.seeds):
            raise ValueError("control seed ceiling is outside selected population")
        if not self.allow_underpowered:
            expected = (
                self.conditions == CONDITIONS
                and self.seeds == SEEDS
                and self.batch_size == 256
                and self.sample_limit is None
                and self.required_seed_passes == 4
                and self.maximum_control_seed_passes == 1
            )
            if not expected:
                raise ValueError("primary bias-reference configuration is fixed")


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


def _json_config(config: BiasReferenceRecenteringConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _source_digests() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "preregistration": PREREGISTRATION_PATH,
        "source_component_runner": Path(bias.__file__),
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "source_component_runner": SOURCE_RUNNER_SHA256,
    }
    for name, digest in expected.items():
        if _sha256(paths[name]) != digest:
            raise ValueError(f"frozen {name} changed: {paths[name]}")
    return {name: _sha256(path) for name, path in paths.items()}


def _implementation_digest(digests: Mapping[str, str] | None = None) -> str:
    return _json_hash(dict(digests or _source_digests()))


def _base_config(
    config: BiasReferenceRecenteringConfig,
) -> bias.BiasComponentCausalConfig:
    return bias.BiasComponentCausalConfig(
        conditions=config.conditions,
        seeds=config.seeds,
        batch_size=config.batch_size,
        sample_limit=config.sample_limit,
        required_seed_passes=min(config.required_seed_passes, len(config.seeds)),
        device=config.device,
        allow_underpowered=True,
    )


def _capture_config(
    config: BiasReferenceRecenteringConfig,
) -> bias.dose.source.NoiseLawObservedTwirlConfig:
    return bias._source_capture_config(_base_config(config))


def _load_source_campaign(
    config: BiasReferenceRecenteringConfig,
) -> tuple[
    dict[str, Any],
    Path,
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int], Path],
]:
    root = Path(config.source_component_root)
    campaign_path = root / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if (
        _sha256(campaign_path) != SOURCE_CAMPAIGN_SHA256
        or campaign.get("schema_version") != bias.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != bias.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256") != SOURCE_IMPLEMENTATION_SHA256
        or campaign.get("result_manifest_sha256")
        != SOURCE_RESULT_MANIFEST_SHA256
        or campaign.get("component_contract_sha256")
        != SOURCE_COMPONENT_CONTRACT_SHA256
        or campaign.get("component_contract", {}).get("pass") is not True
        or campaign.get("aggregates", {}).get("classification")
        != "deterministic_mean_sufficient"
        or campaign.get("aggregates", {}).get("sign_classification")
        != "positive_direction_specific"
        or campaign.get("aggregates", {}).get("primary_hypothesis_pass") is not True
        or campaign.get("aggregates", {}).get("integrity_valid") is not True
        or campaign.get("population", {}).get("population_passes")
        != {
            "centered": {condition: True for condition in CONDITIONS},
            "full_minus": {condition: True for condition in CONDITIONS},
            "full_plus": {condition: False for condition in CONDITIONS},
            "mean_plus": {condition: False for condition in CONDITIONS},
        }
        or len(campaign.get("results", [])) != 10
    ):
        raise ValueError(f"invalid bias-component source {campaign_path}")
    expected_cells = {
        (condition, seed) for condition in CONDITIONS for seed in SEEDS
    }
    details: dict[tuple[str, int], dict[str, Any]] = {}
    diagnostics: dict[tuple[str, int], Path] = {}
    for entry in campaign["results"]:
        result_path = Path(entry["path"])
        diagnostics_path = Path(entry["diagnostics_path"])
        detail = json.loads(result_path.read_text(encoding="utf-8"))
        cell = (str(detail.get("condition")), int(detail.get("seed", -1)))
        if (
            _sha256(result_path) != entry.get("result_sha256")
            or _sha256(diagnostics_path) != entry.get("diagnostics_sha256")
            or entry.get("validity") is not True
            or detail.get("status") != "completed"
            or detail.get("schema_version") != bias.SCHEMA_VERSION
            or detail.get("implementation_sha256") != SOURCE_IMPLEMENTATION_SHA256
            or detail.get("component_contract_sha256")
            != SOURCE_COMPONENT_CONTRACT_SHA256
            or detail.get("gates", {}).get("validity") is not True
            or detail.get("gates", {}).get("source_full_plus_seed_gate_match")
            is not True
        ):
            raise ValueError(f"invalid bias-component result {result_path}")
        details[cell] = detail
        diagnostics[cell] = diagnostics_path
    if set(details) != expected_cells:
        raise ValueError("bias-component source population changed")
    return campaign, campaign_path, details, diagnostics


def construct_interventions(
    sensor: torch.Tensor,
    calibration: torch.Tensor,
    components: Mapping[str, torch.Tensor],
    task: Any,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    count = len(sensor)
    centered = components["centered"][:count]
    mean = components["mean_plus"][:count]
    full_plus = components["full_plus"][:count]
    pilot = mean[:, 0, :]
    noisy = sensor.clone()
    noisy[..., :2] += full_plus
    repaired_calibration = calibration.clone()
    repaired_calibration[:, 4:6] += pilot
    wrong_calibration = calibration.clone()
    wrong_calibration[:, 4:6] -= pilot
    target_sensor, target_calibration = (
        bias.dose.source.observed.observed_deck_action(
            noisy,
            repaired_calibration,
            task,
            orthogonal_axis=True,
        )
    )
    centered_sensor = sensor.clone()
    centered_sensor[..., :2] += centered
    doubled_sensor = sensor.clone()
    doubled_sensor[..., :2] += centered + 2.0 * mean
    return {
        "source_centered": (centered_sensor, calibration),
        "source_full_plus": (noisy, calibration),
        "recenter_correct": (noisy, repaired_calibration),
        "recenter_wrong_sign": (noisy, wrong_calibration),
        "recenter_wrong_sign_expected": (doubled_sensor, calibration),
        "recenter_target_changing": (target_sensor, target_calibration),
    }


def intervention_contract(
    runtime: Mapping[str, Any],
    components: Mapping[str, Mapping[str, torch.Tensor]],
    config: BiasReferenceRecenteringConfig,
) -> dict[str, Any]:
    analytic = bias.dose.source.calibrated.AnalyticCalibratedCanonicalizer(
        runtime["task"]
    )
    regimes: dict[str, Any] = {}
    for regime in REGIMES:
        dataset = runtime["datasets"][regime]
        sensor = bias.dose.source.calibrated.decode_sensor_tokens(
            dataset.paired.circle.input_ids, runtime["task"]
        )
        values = construct_interventions(
            sensor,
            dataset.calibration,
            components[regime],
            runtime["task"],
        )
        pilot_rows = components[regime]["mean_plus"][: len(sensor)]
        pilot = pilot_rows[:, 0, :]
        time_constancy = float(
            (pilot_rows - pilot[:, None, :]).abs().max()
        )
        expected_pilot = torch.zeros_like(pilot)
        expected_pilot[:, 0] = config.selected_noise_sigma
        pilot_error = float((pilot - expected_pilot).abs().max())
        centered_corrected = bias.dose.source.observed._corrected_planar(
            values["source_centered"][0].double(),
            values["source_centered"][1].double(),
            runtime["task"],
        )
        repaired_corrected = bias.dose.source.observed._corrected_planar(
            values["recenter_correct"][0].double(),
            values["recenter_correct"][1].double(),
            runtime["task"],
        )
        wrong_corrected = bias.dose.source.observed._corrected_planar(
            values["recenter_wrong_sign"][0].double(),
            values["recenter_wrong_sign"][1].double(),
            runtime["task"],
        )
        wrong_expected = bias.dose.source.observed._corrected_planar(
            values["recenter_wrong_sign_expected"][0].double(),
            values["recenter_wrong_sign_expected"][1].double(),
            runtime["task"],
        )
        target_sensor, target_calibration = values["recenter_target_changing"]
        restored_sensor, restored_calibration = (
            bias.dose.source.observed.observed_deck_action(
                target_sensor,
                target_calibration,
                runtime["task"],
                orthogonal_axis=True,
            )
        )
        repaired_sensor, repaired_calibration = values["recenter_correct"]
        repaired_feature = analytic(repaired_sensor, repaired_calibration)
        target_feature = analytic(target_sensor, target_calibration)
        target_rms = float(
            torch.sqrt(torch.mean((repaired_feature - target_feature).double().square()))
        )
        record = {
            "pilot_time_constancy_maximum_absolute_error": time_constancy,
            "pilot_value_maximum_absolute_error": pilot_error,
            "pilot_planar_norm": float(pilot.double().norm(dim=-1).mean()),
            "repaired_vs_centered_corrected_maximum_absolute_error": float(
                (repaired_corrected - centered_corrected).abs().max()
            ),
            "wrong_sign_vs_centered_plus_two_mean_maximum_absolute_error": float(
                (wrong_corrected - wrong_expected).abs().max()
            ),
            "target_action_sensor_involution_maximum_absolute_error": float(
                (restored_sensor - repaired_sensor).abs().max()
            ),
            "target_action_calibration_involution_maximum_absolute_error": float(
                (restored_calibration - repaired_calibration).abs().max()
            ),
            "target_changing_analytic_feature_rms": target_rms,
        }
        record["pass"] = bool(
            time_constancy <= config.construction_tolerance
            and pilot_error <= config.construction_tolerance
            and record["repaired_vs_centered_corrected_maximum_absolute_error"]
            <= config.construction_tolerance
            and record[
                "wrong_sign_vs_centered_plus_two_mean_maximum_absolute_error"
            ]
            <= config.construction_tolerance
            and record[
                "target_action_sensor_involution_maximum_absolute_error"
            ]
            <= config.action_involution_tolerance
            and record[
                "target_action_calibration_involution_maximum_absolute_error"
            ]
            <= config.action_involution_tolerance
            and target_rms >= config.minimum_target_changing_feature_rms
        )
        regimes[regime] = record
    return {
        "pilot_source": "known_zero_signal_shared_bias_reference",
        "pilot_forbidden_inputs": ["latent_phase", "task_target", "answer_label"],
        "no_new_random_draws": True,
        "regimes": regimes,
        "pass": all(record["pass"] for record in regimes.values()),
    }


def _fingerprint(
    config: BiasReferenceRecenteringConfig,
    implementation: str,
    condition: str,
    seed: int,
    provenance: Mapping[str, Any],
    dataset_hashes: Mapping[str, str],
    intervention_contract_sha256: str,
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
            "intervention_contract_sha256": intervention_contract_sha256,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
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
        raise ValueError(f"incompatible completed bias-reference result {path}")
    return value


def _metric_error(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return max(abs(float(left[key]) - float(right[key])) for key in left)


@torch.no_grad()
def _features(
    system: Any,
    sensor: torch.Tensor,
    calibration: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    output = []
    for start in range(0, len(sensor), batch_size):
        stop = min(len(sensor), start + batch_size)
        output.append(
            system.feature(
                sensor[start:stop].to(device),
                calibration[start:stop].to(device),
            ).cpu()
        )
    return torch.cat(output).float()


@torch.no_grad()
def analyze_cell(
    *,
    system: Any,
    condition: str,
    seed: int,
    provenance: Mapping[str, Any],
    source_detail: Mapping[str, Any],
    source_detail_path: Path,
    source_diagnostics_path: Path,
    runtime: Mapping[str, Any],
    components: Mapping[str, Mapping[str, torch.Tensor]],
    contract_sha256: str,
    config: BiasReferenceRecenteringConfig,
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
    capture_config = _capture_config(config)
    with np.load(source_diagnostics_path, allow_pickle=False) as loaded:
        stored = {name: loaded[name] for name in loaded.files}
    regime_results: dict[str, Any] = {}
    diagnostics: dict[str, np.ndarray] = {}
    maximum_clean_replay = 0.0
    maximum_source_metric_replay = 0.0
    maximum_feature_equivalence = 0.0
    maximum_posterior_equivalence = 0.0
    for regime in REGIMES:
        dataset = runtime["datasets"][regime]
        count = len(dataset.calibration)
        input_ids = dataset.paired.circle.input_ids
        sensor = bias.dose.source.calibrated.decode_sensor_tokens(
            input_ids, runtime["task"]
        )
        interventions = construct_interventions(
            sensor,
            dataset.calibration,
            components[regime],
            runtime["task"],
        )
        _, clean_posterior, _ = bias.dose.source._capture(
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
        stored_centered = torch.from_numpy(
            stored[f"{regime}__centered__posterior"][:count]
        ).double()
        stored_full_plus = torch.from_numpy(
            stored[f"{regime}__source_full_plus_posterior"][:count]
        ).double()
        clean_replay = float((clean_posterior - stored_clean).abs().max())
        maximum_clean_replay = max(maximum_clean_replay, clean_replay)
        clean_metrics = bias.dose.source.closure.posterior_metrics(
            clean_posterior, dataset
        )
        source_centered_metrics = bias.dose.source.closure.posterior_metrics(
            stored_centered, dataset
        )
        source_full_plus_metrics = bias.dose.source.closure.posterior_metrics(
            stored_full_plus, dataset
        )
        metric_replay = 0.0
        if config.sample_limit is None:
            metric_replay = max(
                _metric_error(
                    source_centered_metrics,
                    source_detail["regimes"][regime]["variants"]["centered"][
                        "task_metrics"
                    ],
                ),
                _metric_error(
                    source_full_plus_metrics,
                    source_detail["regimes"][regime]["variants"]["full_plus"][
                        "task_metrics"
                    ],
                ),
            )
        maximum_source_metric_replay = max(
            maximum_source_metric_replay, metric_replay
        )
        centered_feature = _features(
            system,
            *interventions["source_centered"],
            config.batch_size,
            device,
        )
        variant_records: dict[str, Any] = {}
        variant_posteriors: dict[str, torch.Tensor] = {}
        variant_features: dict[str, torch.Tensor] = {}
        for variant in NEW_VARIANTS:
            _, posterior, feature = bias.dose.source._capture(
                system,
                input_ids,
                *interventions[variant],
                runtime["task"],
                capture_config,
                device,
            )
            metrics = bias.dose.source.closure.posterior_metrics(posterior, dataset)
            passed, gate = bias.dose.source._natural_gate(
                metrics, clean_metrics, capture_config
            )
            variant_records[variant] = {
                "source_reused": False,
                "task_metrics": metrics,
                "natural_utility": gate,
                "natural_utility_pass": passed,
                "posterior_js_from_clean": bias.dose.source.closure.jensen_shannon(
                    posterior, clean_posterior
                ),
            }
            variant_posteriors[variant] = posterior
            variant_features[variant] = feature
            diagnostics[f"{regime}__{variant}__posterior"] = (
                posterior.float().numpy()
            )
        source_full_pass, source_full_gate = bias.dose.source._natural_gate(
            source_full_plus_metrics, clean_metrics, capture_config
        )
        variant_records["source_full_plus"] = {
            "source_reused": True,
            "task_metrics": source_full_plus_metrics,
            "natural_utility": source_full_gate,
            "natural_utility_pass": source_full_pass,
            "posterior_js_from_clean": bias.dose.source.closure.jensen_shannon(
                stored_full_plus, clean_posterior
            ),
        }
        feature_equivalence = float(
            (
                variant_features["recenter_correct"] - centered_feature
            ).abs().max()
        )
        posterior_equivalence = float(
            (
                variant_posteriors["recenter_correct"] - stored_centered
            ).abs().max()
        )
        maximum_feature_equivalence = max(
            maximum_feature_equivalence, feature_equivalence
        )
        maximum_posterior_equivalence = max(
            maximum_posterior_equivalence, posterior_equivalence
        )
        regime_results[regime] = {
            "clean_task_metrics": clean_metrics,
            "clean_posterior_replay_maximum_absolute_error": clean_replay,
            "source_metric_replay_maximum_absolute_error": metric_replay,
            "repaired_vs_centered_feature_maximum_absolute_error": (
                feature_equivalence
            ),
            "repaired_vs_centered_posterior_maximum_absolute_error": (
                posterior_equivalence
            ),
            "variants": variant_records,
        }
        diagnostics[f"{regime}__clean_posterior"] = clean_posterior.float().numpy()
        diagnostics[f"{regime}__source_centered_posterior"] = (
            stored_centered.float().numpy()
        )
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
    source_gate_match = bool(
        variant_seed_gates["source_full_plus"]
        == source_detail["variant_seed_gates"]["full_plus"]
    )
    repaired_source_gate_match = bool(
        variant_seed_gates["recenter_correct"]
        == source_detail["variant_seed_gates"]["centered"]
    )
    state_unchanged = bool(
        bias.dose.source.calibrated._state_digest(system.model)
        == provenance["model_state_sha256"]
        and bias.dose.source.calibrated._module_digest(system)
        == provenance["system_state_sha256"]
    )
    finite = bias.dose.source._finite(regime_results)
    gates = {
        "clean_posterior_replay": (
            maximum_clean_replay <= config.posterior_replay_tolerance
        ),
        "source_metric_replay": (
            maximum_source_metric_replay <= config.metric_replay_tolerance
        ),
        "feature_equivalence": (
            maximum_feature_equivalence <= config.feature_equivalence_tolerance
        ),
        "posterior_equivalence": (
            maximum_posterior_equivalence <= config.posterior_replay_tolerance
        ),
        "source_full_plus_seed_gate_match": source_gate_match,
        "repaired_centered_seed_gate_match": repaired_source_gate_match,
        "state_unchanged": state_unchanged,
        "finite": finite,
    }
    gates["validity"] = bool(all(gates.values()))
    diagnostics_path = result_dir / "diagnostics.npz"
    _write_npz(diagnostics_path, diagnostics)
    value = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-bias-reference-{condition}-seed{seed}",
        "status": "completed",
        "evidence_role": EVIDENCE_ROLE,
        "completed_at": _utc_now(),
        "condition": condition,
        "seed": seed,
        "configuration": _json_config(config),
        "implementation_sha256": implementation,
        "scientific_fingerprint": fingerprint,
        "dataset_hashes": runtime["dataset_hashes"],
        "intervention_contract_sha256": contract_sha256,
        "provenance": {
            **dict(provenance),
            "source_component_result": str(source_detail_path),
            "source_component_result_sha256": source_detail_sha256,
            "source_component_diagnostics": str(source_diagnostics_path),
            "source_component_diagnostics_sha256": source_diagnostics_sha256,
        },
        "regimes": regime_results,
        "variant_seed_gates": variant_seed_gates,
        "gates": gates,
        "analysis_seconds": time.perf_counter() - started,
        "artifacts": {
            "result": str(result_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": _sha256(diagnostics_path),
        },
    }
    _write_json(result_path, value)
    print(
        f"{condition} seed {seed}: variants={variant_seed_gates} "
        f"valid={gates['validity']}",
        flush=True,
    )
    return value


def aggregate_results(
    results: list[Mapping[str, Any]],
    config: BiasReferenceRecenteringConfig,
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for condition in config.conditions:
        selected = [item for item in results if item["condition"] == condition]
        arms[condition] = {
            "variants": {
                variant: {
                    "natural_utility_pass_count": sum(
                        bool(item["variant_seed_gates"][variant]) for item in selected
                    )
                }
                for variant in VARIANTS
            }
        }
    population_passes = {
        variant: {
            condition: (
                arms[condition]["variants"][variant][
                    "natural_utility_pass_count"
                ]
                >= config.required_seed_passes
            )
            for condition in config.conditions
        }
        for variant in VARIANTS
    }
    control_specific = {
        variant: {
            condition: (
                arms[condition]["variants"][variant][
                    "natural_utility_pass_count"
                ]
                <= config.maximum_control_seed_passes
            )
            for condition in config.conditions
        }
        for variant in ("recenter_wrong_sign", "recenter_target_changing")
    }
    return {
        "arms": arms,
        "population_passes": population_passes,
        "control_specific": control_specific,
        "source_full_plus_fails_both_arms": all(
            not value for value in population_passes["source_full_plus"].values()
        ),
        "recenter_correct_passes_both_arms": all(
            population_passes["recenter_correct"].values()
        ),
        "wrong_sign_specific_both_arms": all(
            control_specific["recenter_wrong_sign"].values()
        ),
        "target_changing_specific_both_arms": all(
            control_specific["recenter_target_changing"].values()
        ),
        "integrity_valid": all(item["gates"]["validity"] for item in results),
    }


def classify_campaign(
    *,
    integrity_valid: bool,
    contract_pass: bool,
    source_fails: bool,
    repaired_passes: bool,
    wrong_sign_specific: bool,
    target_specific: bool,
) -> tuple[str, bool]:
    if not integrity_valid or not contract_pass or not source_fails:
        return "invalid", False
    if repaired_passes and wrong_sign_specific and target_specific:
        return "observed_bias_reference_repair_specific", True
    if repaired_passes:
        return "algebraic_repair_without_specificity", False
    return "observed_bias_reference_insufficient", False


def _result_entry(result: Mapping[str, Any]) -> dict[str, Any]:
    result_path = Path(result["artifacts"]["result"])
    diagnostics_path = Path(result["artifacts"]["diagnostics"])
    return {
        "condition": result["condition"],
        "seed": result["seed"],
        "path": str(result_path),
        "result_sha256": _sha256(result_path),
        "diagnostics_path": str(diagnostics_path),
        "diagnostics_sha256": _sha256(diagnostics_path),
        "scientific_fingerprint": result["scientific_fingerprint"],
        "validity": result["gates"]["validity"],
    }


def _campaign_reusable(
    campaign: Mapping[str, Any],
    config: BiasReferenceRecenteringConfig,
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
    config: BiasReferenceRecenteringConfig, output: Path
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
    base_config = _base_config(config)
    base_arrays = bias._load_base_noise(base_config)
    full_components = bias.construct_components(base_arrays, base_config)
    source_contract = bias.component_contract(
        full_components, base_arrays, base_config
    )
    if (
        not source_contract["pass"]
        or bias._json_hash(source_contract) != SOURCE_COMPONENT_CONTRACT_SHA256
    ):
        raise ValueError("source component reconstruction changed")
    runtime = bias.dose._load_runtime_sources(bias._dose_config(base_config))
    full_contract = intervention_contract(runtime, full_components, config)
    contract_sha256 = _json_hash(full_contract)
    if not full_contract["pass"]:
        raise ValueError(f"bias-reference intervention contract failed: {full_contract}")
    components = {
        regime: {
            variant: full_components[regime][variant][
                : len(runtime["datasets"][regime].calibration)
            ]
            for variant in bias.VARIANTS
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
            system, provenance = bias.dose._load_system(
                runtime, condition, seed, device
            )
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
                raise RuntimeError("bias-reference implementation changed")

    population = aggregate_results(results, config)
    classification, primary_pass = classify_campaign(
        integrity_valid=population["integrity_valid"],
        contract_pass=full_contract["pass"],
        source_fails=population["source_full_plus_fails_both_arms"],
        repaired_passes=population["recenter_correct_passes_both_arms"],
        wrong_sign_specific=population["wrong_sign_specific_both_arms"],
        target_specific=population["target_changing_specific_both_arms"],
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
        "source_component_contract_sha256": (
            SOURCE_COMPONENT_CONTRACT_SHA256
        ),
        "source_dvc_root": SOURCE_DVC_ROOT,
        "source_lakefs_commit": SOURCE_LAKEFS_COMMIT,
        "dataset_hashes": runtime["dataset_hashes"],
        "source_preflight_manifest": runtime["preflight_manifest"],
        "intervention_contract": full_contract,
        "intervention_contract_sha256": contract_sha256,
        "population": population,
        "aggregates": {
            "classification": classification,
            "primary_hypothesis_pass": primary_pass,
            "primary_evaluable": True,
            "valid": bool(population["integrity_valid"] and full_contract["pass"]),
            "integrity_valid": population["integrity_valid"],
            "intervention_contract_pass": full_contract["pass"],
            "required_seed_passes": config.required_seed_passes,
            "maximum_control_seed_passes": config.maximum_control_seed_passes,
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
            "fitted_bias_estimators": 0,
            "fitted_denoisers": 0,
            "fitted_actions": 0,
            "fitted_observers": 0,
            "fitted_probes": 0,
            "new_random_draws": 0,
            "new_forward_variants_per_system_shift": len(NEW_VARIANTS),
            "reused_source_variants_per_system_shift": 3,
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
            "The shared-bias pilot is an exact zero-signal positive control and does not estimate bias under finite noisy acquisition.",
            "The pilot uses neither latent phase nor target labels and changes only the observed calibration offset.",
            "Correct repair is algebraically required to match the sealed centered-only front-end input under the declared shared-bias model.",
            "Wrong-sign recentering doubles the deterministic bias; the observed orthogonal-axis action is the target-changing specificity control.",
            "Clean, centered, and full-positive source posteriors are pinned to the sealed source campaign.",
            "No model, front end, head, bias estimator, denoiser, action, observer, probe, or noise process is trained or fitted.",
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
            "data/experiments/tinyllm_bias_reference_recentering/"
            "20260810_d10_preregistered_v2"
        ),
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--conditions", type=_strings, default=CONDITIONS)
    parser.add_argument("--seeds", type=_ints, default=SEEDS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--required-seed-passes", type=int, default=4)
    parser.add_argument("--maximum-control-seed-passes", type=int, default=1)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = BiasReferenceRecenteringConfig(
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
