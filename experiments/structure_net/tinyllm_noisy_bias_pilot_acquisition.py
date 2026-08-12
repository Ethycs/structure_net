#!/usr/bin/env python3
"""Titrate finite noisy zero-signal pilots for frozen TinyLLM bias repair."""

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

import experiments.structure_net.tinyllm_acquisition_draw_stability as acquisition
import experiments.structure_net.tinyllm_bias_reference_recentering as repair


SCHEMA_VERSION = "nal.tinyllm-noisy-bias-pilot-acquisition.v1"
HYPOTHESIS_ID = "tinyllm-noisy-bias-pilot-acquisition-v1"
EVIDENCE_ROLE = "preregistered_frozen_reused_draw_bias_pilot_titration"
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-noisy-bias-pilot-acquisition-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "2e19e9c7bf3908c97e29fd614a1801ef9e3eaec568a3babf3a1fe24adfa9830b"
)
SOURCE_REPAIR_RUNNER_SHA256 = (
    "fd6ea5108ccd733e360010c83a1a4a411512cbed239e3c9356a6a6bb77a6996a"
)
SOURCE_REPAIR_CAMPAIGN_SHA256 = (
    "1996ac4c2534b62a25a2f52ceadfd21055a91bdadc81f38ccf01c6855da2b7d0"
)
SOURCE_REPAIR_IMPLEMENTATION_SHA256 = (
    "059d4ace65402fb296bcf35bf614aa411e085cc34abaf8513e077678a4828e15"
)
SOURCE_REPAIR_RESULT_MANIFEST_SHA256 = (
    "7dbbe3a49f4e3ebac36e891ec63d5336ff3be2e176e26f1a610cbfceecaabb4e"
)
SOURCE_REPAIR_CONTRACT_SHA256 = (
    "6bed75b6cd9a15be35f21e53463efa28bcc2f775f1490f31414e005398894004"
)
SOURCE_ACQUISITION_RUNNER_SHA256 = (
    "54c293d94582e4aa826772ac9c9a3791b5ed66c01aa9635fef75f433f7fe4e0d"
)
SOURCE_ACQUISITION_CAMPAIGN_SHA256 = (
    "968f85010129d761268b4816d85ddd2ab578bbc93307e8a936e58fa891e89d93"
)
SOURCE_ACQUISITION_IMPLEMENTATION_SHA256 = (
    "a0eae3da0dfcf74328ff0f2fa264a8e712b61f901337608f9e23ef93657d0440"
)
SOURCE_ACQUISITION_RESULT_MANIFEST_SHA256 = (
    "d13e52a07423e507cef034c78b734219b85abc8468feae6313b52148fa95b163"
)
SOURCE_ACQUISITION_ARRAY_SHA256 = (
    "57eca80cccf1b916a60d79d5982bdbffe3b515cee7dfbee7645830448779aace"
)
SOURCE_DVC_ROOT = "1de07aeb227a8093fa5973d37d63f9a6.dir"
SOURCE_LAKEFS_COMMIT = (
    "23a11ba9918f2adcf4397c619e8b942f7539e1f98bb52962f98be6f520e7c181"
)
CONDITIONS = repair.CONDITIONS
SEEDS = repair.SEEDS
REGIMES = repair.REGIMES
COUNTS = (1, 4, 16, 64, 256)
DRAW_COUNT = 16


@dataclass(frozen=True)
class NoisyBiasPilotConfig:
    source_repair_root: str = (
        "data/experiments/tinyllm_bias_reference_recentering/"
        "20260810_d10_preregistered_v2"
    )
    source_acquisition_root: str = (
        "data/experiments/tinyllm_acquisition_draw_stability/"
        "20260810_d16_preregistered"
    )
    conditions: tuple[str, ...] = CONDITIONS
    seeds: tuple[int, ...] = SEEDS
    counts: tuple[int, ...] = COUNTS
    draw_count: int = DRAW_COUNT
    selected_noise_sigma: float = 0.03125
    pilot_noise_sigma: float = 0.03125 / math.sqrt(2.0)
    acquisition_split: str = "composition"
    acquisition_channels: tuple[int, int] = (0, 1)
    acquisition_draw_seed_root: int = 81_027_026
    array_mean_tolerance: float = 0.03
    array_std_minimum: float = 0.95
    array_std_maximum: float = 1.05
    array_correlation_ceiling: float = 0.05
    source_replay_tolerance: float = 2e-6
    natural_accuracy_loss_ceiling: float = 0.05
    natural_circular_error_increase_ceiling: float = math.pi / 16.0
    natural_cross_entropy_increase_ceiling: float = 0.10
    required_seed_passes: int = 4
    required_draw_passes: int = 15
    maximum_control_seed_passes: int = 1
    batch_size: int = 256
    sample_limit: int | None = None
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2")
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.conditions or set(self.conditions).difference(CONDITIONS):
            raise ValueError("unknown or empty structured condition")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("checkpoint seeds must be non-empty and distinct")
        if self.counts != tuple(sorted(set(self.counts))):
            raise ValueError("pilot counts must be ordered and distinct")
        if not self.counts or min(self.counts) < 1 or max(self.counts) > 256:
            raise ValueError("pilot counts are outside the frozen array")
        if not 1 <= self.draw_count <= DRAW_COUNT:
            raise ValueError("draw count is outside the frozen array")
        if self.selected_noise_sigma != 0.03125:
            raise ValueError("the selected bias magnitude is frozen")
        if self.pilot_noise_sigma != 0.03125 / math.sqrt(2.0):
            raise ValueError("the pilot measurement noise scale is frozen")
        if self.acquisition_split != "composition":
            raise ValueError("the acquisition split is frozen")
        if self.acquisition_channels != (0, 1):
            raise ValueError("the acquisition channels are frozen")
        if self.acquisition_draw_seed_root != 81_027_026:
            raise ValueError("the acquisition seed root is frozen")
        if not self.devices or len(set(self.devices)) != len(self.devices):
            raise ValueError("worker devices must be non-empty and distinct")
        if self.batch_size < 1:
            raise ValueError("batch size must be positive")
        if self.sample_limit is not None and self.sample_limit < 8:
            raise ValueError("sample limit must be at least eight")
        if not 1 <= self.required_seed_passes <= len(self.seeds):
            raise ValueError("required seed count is outside selected population")
        if not 1 <= self.required_draw_passes <= self.draw_count:
            raise ValueError("required draw count is outside selected population")
        if not 0 <= self.maximum_control_seed_passes <= len(self.seeds):
            raise ValueError("control seed ceiling is outside selected population")
        if not self.allow_underpowered:
            expected = (
                self.conditions == CONDITIONS
                and self.seeds == SEEDS
                and self.counts == COUNTS
                and self.draw_count == DRAW_COUNT
                and self.required_seed_passes == 4
                and self.required_draw_passes == 15
                and self.maximum_control_seed_passes == 1
                and self.batch_size == 256
                and self.sample_limit is None
                and self.devices == ("cuda:0", "cuda:1", "cuda:2")
            )
            if not expected:
                raise ValueError("primary noisy-pilot configuration is fixed")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _array_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
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


def _json_config(config: NoisyBiasPilotConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _source_digests() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "preregistration": PREREGISTRATION_PATH,
        "source_repair_runner": Path(repair.__file__),
        "source_acquisition_runner": Path(acquisition.__file__),
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "source_repair_runner": SOURCE_REPAIR_RUNNER_SHA256,
        "source_acquisition_runner": SOURCE_ACQUISITION_RUNNER_SHA256,
    }
    for name, digest in expected.items():
        if _sha256(paths[name]) != digest:
            raise ValueError(f"frozen {name} changed: {paths[name]}")
    return {name: _sha256(path) for name, path in paths.items()}


def _implementation_digest(digests: Mapping[str, str] | None = None) -> str:
    return _json_hash(dict(digests or _source_digests()))


def _repair_config(config: NoisyBiasPilotConfig) -> repair.BiasReferenceRecenteringConfig:
    return repair.BiasReferenceRecenteringConfig(
        source_component_root=(
            "data/experiments/tinyllm_bias_component_causal_decomposition/"
            "20260810_d10_preregistered"
        ),
        conditions=config.conditions,
        seeds=config.seeds,
        batch_size=config.batch_size,
        sample_limit=config.sample_limit,
        required_seed_passes=min(config.required_seed_passes, len(config.seeds)),
        maximum_control_seed_passes=min(
            config.maximum_control_seed_passes, len(config.seeds)
        ),
        device=config.devices[0],
        allow_underpowered=True,
    )


def _load_repair_source(
    config: NoisyBiasPilotConfig,
) -> tuple[
    dict[str, Any],
    Path,
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int], Path],
]:
    root = Path(config.source_repair_root)
    campaign_path = root / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    population = campaign.get("population", {}).get("arms", {})
    if (
        _sha256(campaign_path) != SOURCE_REPAIR_CAMPAIGN_SHA256
        or campaign.get("schema_version") != repair.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != repair.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256")
        != SOURCE_REPAIR_IMPLEMENTATION_SHA256
        or campaign.get("result_manifest_sha256")
        != SOURCE_REPAIR_RESULT_MANIFEST_SHA256
        or campaign.get("intervention_contract_sha256")
        != SOURCE_REPAIR_CONTRACT_SHA256
        or campaign.get("aggregates", {}).get("classification")
        != "observed_bias_reference_repair_specific"
        or campaign.get("aggregates", {}).get("primary_hypothesis_pass") is not True
        or campaign.get("aggregates", {}).get("valid") is not True
        or population.get("analytic_calibrated", {}).get("variants")
        != {
            "recenter_correct": {"natural_utility_pass_count": 5},
            "recenter_target_changing": {"natural_utility_pass_count": 0},
            "recenter_wrong_sign": {"natural_utility_pass_count": 1},
            "source_full_plus": {"natural_utility_pass_count": 1},
        }
        or population.get("learned_calibrated_equivariant", {}).get("variants")
        != {
            "recenter_correct": {"natural_utility_pass_count": 5},
            "recenter_target_changing": {"natural_utility_pass_count": 0},
            "recenter_wrong_sign": {"natural_utility_pass_count": 0},
            "source_full_plus": {"natural_utility_pass_count": 3},
        }
        or len(campaign.get("results", [])) != 10
    ):
        raise ValueError(f"invalid exact-pilot source {campaign_path}")
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
            or detail.get("implementation_sha256")
            != SOURCE_REPAIR_IMPLEMENTATION_SHA256
            or detail.get("gates", {}).get("validity") is not True
        ):
            raise ValueError(f"invalid exact-pilot result {result_path}")
        details[cell] = detail
        diagnostics[cell] = diagnostics_path
    if set(details) != expected_cells:
        raise ValueError("exact-pilot source population changed")
    return campaign, campaign_path, details, diagnostics


def _load_acquisition_source(
    config: NoisyBiasPilotConfig,
) -> tuple[dict[str, Any], Path, np.ndarray]:
    root = Path(config.source_acquisition_root)
    campaign_path = root / "campaign_results.json"
    array_path = root / "acquisition_draw_errors.npz"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if (
        _sha256(campaign_path) != SOURCE_ACQUISITION_CAMPAIGN_SHA256
        or _sha256(array_path) != SOURCE_ACQUISITION_ARRAY_SHA256
        or campaign.get("schema_version") != acquisition.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != acquisition.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256")
        != SOURCE_ACQUISITION_IMPLEMENTATION_SHA256
        or campaign.get("result_manifest_sha256")
        != SOURCE_ACQUISITION_RESULT_MANIFEST_SHA256
        or campaign.get("artifacts", {}).get("draw_arrays_sha256")
        != SOURCE_ACQUISITION_ARRAY_SHA256
        or campaign.get("configuration", {}).get("draw_count") != 16
        or campaign.get("configuration", {}).get("draw_seed_root") != 81_027_026
        or campaign.get("configuration", {}).get("replicate_counts") != [64, 256]
        or campaign.get("aggregates", {}).get("valid") is not True
    ):
        raise ValueError(f"invalid acquisition source {campaign_path}")
    with np.load(array_path, allow_pickle=False) as loaded:
        errors = loaded[f"{config.acquisition_split}__errors"].copy()
        seed_root = loaded["draw_seed_root"].copy()
    if (
        errors.shape != (16, 256, 512)
        or seed_root.tolist() != [config.acquisition_draw_seed_root]
    ):
        raise ValueError("acquisition array structure changed")
    return campaign, array_path, errors


def build_pilot_arrays(
    errors: np.ndarray, config: NoisyBiasPilotConfig
) -> dict[str, np.ndarray]:
    source_streams = np.ascontiguousarray(
        errors[:DRAW_COUNT, :, list(config.acquisition_channels)],
        dtype=np.float64,
    )
    source_prefix_means = np.stack(
        [source_streams[:, :count].mean(axis=1) for count in COUNTS], axis=1
    )
    selected = source_streams[: config.draw_count].copy()
    noise_means = np.stack(
        [selected[:, :count].mean(axis=1) for count in config.counts], axis=1
    )
    bias = np.asarray([config.selected_noise_sigma, 0.0], dtype=np.float64)
    estimates = bias[None, None, :] + config.pilot_noise_sigma * noise_means
    return {
        "counts": np.asarray(config.counts, dtype=np.int64),
        "source_audit_counts": np.asarray(COUNTS, dtype=np.int64),
        "source_audit_standard_normal_streams": source_streams,
        "source_audit_standard_normal_prefix_means": source_prefix_means,
        "standard_normal_streams": selected,
        "standard_normal_prefix_means": noise_means,
        "pilot_estimates": estimates,
    }


def pilot_contract(
    arrays: Mapping[str, np.ndarray], config: NoisyBiasPilotConfig
) -> dict[str, Any]:
    source_counts = arrays["source_audit_counts"]
    streams = arrays["source_audit_standard_normal_streams"]
    source_means = arrays["source_audit_standard_normal_prefix_means"]
    means = arrays["standard_normal_prefix_means"]
    estimates = arrays["pilot_estimates"]
    flattened = streams.reshape(-1, 2)
    channel_means = flattened.mean(axis=0)
    channel_stds = flattened.std(axis=0)
    correlation = float(np.corrcoef(flattened.T)[0, 1])
    prefix_rmse = np.sqrt(np.mean(np.square(source_means), axis=(0, 2)))
    expected = (
        np.asarray([config.selected_noise_sigma, 0.0])[None, None, :]
        + config.pilot_noise_sigma * means
    )
    reconstruction_error = float(np.max(np.abs(estimates - expected)))
    return {
        "source_array_sha256": SOURCE_ACQUISITION_ARRAY_SHA256,
        "source_draw_seed_root": config.acquisition_draw_seed_root,
        "source_split": config.acquisition_split,
        "source_channels": list(config.acquisition_channels),
        "source_draw_count": DRAW_COUNT,
        "source_counts": source_counts.tolist(),
        "evaluation_draw_count": config.draw_count,
        "evaluation_counts": list(config.counts),
        "channel_means": channel_means.tolist(),
        "channel_standard_deviations": channel_stds.tolist(),
        "cross_channel_correlation": correlation,
        "prefix_mean_rmse": {
            str(count): float(value)
            for count, value in zip(source_counts, prefix_rmse)
        },
        "estimate_reconstruction_maximum_absolute_error": reconstruction_error,
        "pilot_array_content_sha256": _array_digest(arrays),
        "no_new_random_draws": True,
        "pass": bool(
            np.max(np.abs(channel_means)) <= config.array_mean_tolerance
            and np.min(channel_stds) >= config.array_std_minimum
            and np.max(channel_stds) <= config.array_std_maximum
            and abs(correlation) <= config.array_correlation_ceiling
            and reconstruction_error == 0.0
            and prefix_rmse[-1] < prefix_rmse[0]
            and streams.shape == (DRAW_COUNT, 256, 2)
            and source_means.shape == (DRAW_COUNT, len(COUNTS), 2)
            and np.isfinite(streams).all()
            and np.isfinite(estimates).all()
        ),
    }


def _prepare_sources(
    config: NoisyBiasPilotConfig,
) -> tuple[
    dict[str, Any],
    Path,
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int], Path],
    Mapping[str, Any],
    Mapping[str, Mapping[str, torch.Tensor]],
    dict[str, np.ndarray],
    dict[str, Any],
    Path,
]:
    repair_campaign, repair_path, repair_details, repair_diagnostics = (
        _load_repair_source(config)
    )
    _, acquisition_path, errors = _load_acquisition_source(config)
    pilot_arrays = build_pilot_arrays(errors, config)
    contract = pilot_contract(pilot_arrays, config)
    if not contract["pass"]:
        raise ValueError(f"pilot acquisition contract failed: {contract}")
    repair_config = _repair_config(config)
    base_config = repair._base_config(repair_config)
    base_arrays = repair.bias._load_base_noise(base_config)
    components = repair.bias.construct_components(base_arrays, base_config)
    runtime = repair.bias.dose._load_runtime_sources(
        repair.bias._dose_config(base_config)
    )
    return (
        repair_campaign,
        repair_path,
        repair_details,
        repair_diagnostics,
        runtime,
        components,
        pilot_arrays,
        contract,
        acquisition_path,
    )


def prepare_campaign(config: NoisyBiasPilotConfig, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    source_digests = _source_digests()
    implementation = _implementation_digest(source_digests)
    (
        _,
        repair_path,
        _,
        _,
        runtime,
        _,
        pilot_arrays,
        contract,
        acquisition_path,
    ) = _prepare_sources(config)
    array_path = output / "pilot_acquisition_arrays.npz"
    if array_path.is_file():
        with np.load(array_path, allow_pickle=False) as loaded:
            if set(loaded.files) != set(pilot_arrays) or any(
                not np.array_equal(loaded[name], value)
                for name, value in pilot_arrays.items()
            ):
                raise ValueError(f"incompatible pilot artifact {array_path}")
    else:
        _write_npz(array_path, pilot_arrays)
    value = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "prepared",
        "configuration": _json_config(config),
        "implementation_sha256": implementation,
        "source_digests": source_digests,
        "source_repair_campaign": str(repair_path),
        "source_repair_campaign_sha256": SOURCE_REPAIR_CAMPAIGN_SHA256,
        "source_acquisition_arrays": str(acquisition_path),
        "source_acquisition_arrays_sha256": SOURCE_ACQUISITION_ARRAY_SHA256,
        "dataset_hashes": runtime["dataset_hashes"],
        "pilot_contract": contract,
        "pilot_contract_sha256": _json_hash(contract),
        "pilot_arrays": str(array_path),
        "pilot_arrays_sha256": _sha256(array_path),
        "pilot_array_content_sha256": contract["pilot_array_content_sha256"],
    }
    _write_json(output / "prepared_manifest.json", value)
    return value


def _load_prepared(
    config: NoisyBiasPilotConfig, output: Path, implementation: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = output / "prepared_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    array_path = Path(manifest["pilot_arrays"])
    with np.load(array_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    contract = pilot_contract(arrays, config)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("hypothesis_id") != HYPOTHESIS_ID
        or manifest.get("configuration") != _json_config(config)
        or manifest.get("implementation_sha256") != implementation
        or manifest.get("pilot_contract") != contract
        or manifest.get("pilot_contract_sha256") != _json_hash(contract)
        or manifest.get("pilot_arrays_sha256") != _sha256(array_path)
        or manifest.get("pilot_array_content_sha256") != _array_digest(arrays)
        or not contract["pass"]
    ):
        raise ValueError("prepared pilot manifest changed")
    return manifest, arrays


def _cell_order(config: NoisyBiasPilotConfig) -> list[tuple[str, int]]:
    return [
        (condition, seed)
        for condition in config.conditions
        for seed in config.seeds
    ]


def _worker_cells(
    config: NoisyBiasPilotConfig, worker_index: int
) -> list[tuple[str, int]]:
    if not 0 <= worker_index < len(config.devices):
        raise ValueError("worker index is outside configured devices")
    return [
        cell
        for index, cell in enumerate(_cell_order(config))
        if index % len(config.devices) == worker_index
    ]


def _fingerprint(
    config: NoisyBiasPilotConfig,
    implementation: str,
    condition: str,
    seed: int,
    device: str,
    provenance: Mapping[str, Any],
    dataset_hashes: Mapping[str, str],
    pilot_contract_sha256: str,
    pilot_arrays_sha256: str,
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
            "device": device,
            "provenance": provenance,
            "dataset_hashes": dict(dataset_hashes),
            "pilot_contract_sha256": pilot_contract_sha256,
            "pilot_arrays_sha256": pilot_arrays_sha256,
            "source_repair_campaign_sha256": SOURCE_REPAIR_CAMPAIGN_SHA256,
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
        raise ValueError(f"incompatible completed noisy-pilot result {path}")
    return value


def _metric_error(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return max(abs(float(left[key]) - float(right[key])) for key in left)


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
    pilot_arrays: Mapping[str, np.ndarray],
    pilot_contract_sha256: str,
    pilot_arrays_sha256: str,
    config: NoisyBiasPilotConfig,
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
        str(device),
        provenance,
        runtime["dataset_hashes"],
        pilot_contract_sha256,
        pilot_arrays_sha256,
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
    capture_config = repair._capture_config(_repair_config(config))
    with np.load(source_diagnostics_path, allow_pickle=False) as loaded:
        stored = {name: loaded[name] for name in loaded.files}
    estimates = torch.from_numpy(pilot_arrays["pilot_estimates"]).float()
    regime_results: dict[str, Any] = {}
    diagnostics: dict[str, np.ndarray] = {}
    maximum_clean_replay = 0.0
    maximum_source_metric_replay = 0.0
    for regime in REGIMES:
        dataset = runtime["datasets"][regime]
        count_examples = len(dataset.calibration)
        input_ids = dataset.paired.circle.input_ids
        sensor = repair.bias.dose.source.calibrated.decode_sensor_tokens(
            input_ids, runtime["task"]
        )
        full_plus_sensor = sensor.clone()
        full_plus_sensor[..., :2] += components[regime]["full_plus"][
            :count_examples
        ]
        _, clean_posterior, _ = repair.bias.dose.source._capture(
            system,
            input_ids,
            sensor,
            dataset.calibration,
            runtime["task"],
            capture_config,
            device,
        )
        stored_clean = torch.from_numpy(
            stored[f"{regime}__clean_posterior"][:count_examples]
        ).double()
        stored_exact = torch.from_numpy(
            stored[f"{regime}__recenter_correct__posterior"][:count_examples]
        ).double()
        stored_full = torch.from_numpy(
            stored[f"{regime}__source_full_plus_posterior"][:count_examples]
        ).double()
        clean_replay = float((clean_posterior - stored_clean).abs().max())
        maximum_clean_replay = max(maximum_clean_replay, clean_replay)
        clean_metrics = repair.bias.dose.source.closure.posterior_metrics(
            clean_posterior, dataset
        )
        exact_metrics = repair.bias.dose.source.closure.posterior_metrics(
            stored_exact, dataset
        )
        full_metrics = repair.bias.dose.source.closure.posterior_metrics(
            stored_full, dataset
        )
        metric_replay = 0.0
        if config.sample_limit is None:
            metric_replay = max(
                _metric_error(
                    exact_metrics,
                    source_detail["regimes"][regime]["variants"][
                        "recenter_correct"
                    ]["task_metrics"],
                ),
                _metric_error(
                    full_metrics,
                    source_detail["regimes"][regime]["variants"][
                        "source_full_plus"
                    ]["task_metrics"],
                ),
            )
        maximum_source_metric_replay = max(
            maximum_source_metric_replay, metric_replay
        )
        draw_records: dict[str, Any] = {}
        selected_posteriors: dict[str, torch.Tensor] = {}
        for draw_index in range(config.draw_count):
            count_records: dict[str, Any] = {}
            for count_index, pilot_count in enumerate(config.counts):
                calibration = dataset.calibration.clone()
                calibration[:, 4:6] += estimates[
                    draw_index, count_index
                ][None, :]
                _, posterior, _ = repair.bias.dose.source._capture(
                    system,
                    input_ids,
                    full_plus_sensor,
                    calibration,
                    runtime["task"],
                    capture_config,
                    device,
                )
                metrics = repair.bias.dose.source.closure.posterior_metrics(
                    posterior, dataset
                )
                passed, gate = repair.bias.dose.source._natural_gate(
                    metrics, clean_metrics, capture_config
                )
                count_records[str(pilot_count)] = {
                    "task_metrics": metrics,
                    "natural_utility": gate,
                    "natural_utility_pass": passed,
                    "posterior_js_from_clean": (
                        repair.bias.dose.source.closure.jensen_shannon(
                            posterior, clean_posterior
                        )
                    ),
                    "pilot_estimate": estimates[
                        draw_index, count_index
                    ].double().tolist(),
                }
                if draw_index == 0 and pilot_count in (
                    config.counts[0],
                    config.counts[-1],
                ):
                    selected_posteriors[f"draw0_m{pilot_count}"] = posterior
            draw_records[f"draw_{draw_index:02d}"] = {
                "counts": count_records
            }
        wrong_calibration = dataset.calibration.clone()
        wrong_calibration[:, 4:6] -= estimates[0, -1][None, :]
        _, wrong_posterior, _ = repair.bias.dose.source._capture(
            system,
            input_ids,
            full_plus_sensor,
            wrong_calibration,
            runtime["task"],
            capture_config,
            device,
        )
        wrong_metrics = repair.bias.dose.source.closure.posterior_metrics(
            wrong_posterior, dataset
        )
        wrong_pass, wrong_gate = repair.bias.dose.source._natural_gate(
            wrong_metrics, clean_metrics, capture_config
        )
        regime_results[regime] = {
            "clean_task_metrics": clean_metrics,
            "clean_posterior_replay_maximum_absolute_error": clean_replay,
            "source_metric_replay_maximum_absolute_error": metric_replay,
            "source_exact_pilot_task_metrics": exact_metrics,
            "source_full_plus_task_metrics": full_metrics,
            "draws": draw_records,
            "wrong_sign_draw0_m256": {
                "task_metrics": wrong_metrics,
                "natural_utility": wrong_gate,
                "natural_utility_pass": wrong_pass,
                "posterior_js_from_clean": (
                    repair.bias.dose.source.closure.jensen_shannon(
                        wrong_posterior, clean_posterior
                    )
                ),
            },
        }
        diagnostics[f"{regime}__clean_posterior"] = clean_posterior.float().numpy()
        diagnostics[f"{regime}__wrong_sign_draw0_m256_posterior"] = (
            wrong_posterior.float().numpy()
        )
        for name, posterior in selected_posteriors.items():
            diagnostics[f"{regime}__{name}__posterior"] = (
                posterior.float().numpy()
            )

    draw_count_seed_gates = {
        f"draw_{draw_index:02d}": {
            str(pilot_count): all(
                regime_results[regime]["draws"][f"draw_{draw_index:02d}"][
                    "counts"
                ][str(pilot_count)]["natural_utility_pass"]
                for regime in REGIMES
            )
            for pilot_count in config.counts
        }
        for draw_index in range(config.draw_count)
    }
    wrong_sign_seed_gate = all(
        regime_results[regime]["wrong_sign_draw0_m256"][
            "natural_utility_pass"
        ]
        for regime in REGIMES
    )
    source_exact_seed_gate = source_detail["variant_seed_gates"][
        "recenter_correct"
    ]
    source_full_seed_gate = source_detail["variant_seed_gates"][
        "source_full_plus"
    ]
    state_unchanged = bool(
        repair.bias.dose.source.calibrated._state_digest(system.model)
        == provenance["model_state_sha256"]
        and repair.bias.dose.source.calibrated._module_digest(system)
        == provenance["system_state_sha256"]
    )
    gates = {
        "clean_posterior_replay": (
            maximum_clean_replay <= config.source_replay_tolerance
        ),
        "source_metric_replay": (
            maximum_source_metric_replay <= config.source_replay_tolerance
        ),
        "state_unchanged": state_unchanged,
        "finite": repair.bias.dose.source._finite(regime_results),
    }
    gates["validity"] = bool(all(gates.values()))
    diagnostics_path = result_dir / "diagnostics.npz"
    _write_npz(diagnostics_path, diagnostics)
    value = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-noisy-bias-pilot-{condition}-seed{seed}",
        "status": "completed",
        "evidence_role": EVIDENCE_ROLE,
        "completed_at": _utc_now(),
        "condition": condition,
        "seed": seed,
        "device": str(device),
        "configuration": _json_config(config),
        "implementation_sha256": implementation,
        "scientific_fingerprint": fingerprint,
        "dataset_hashes": runtime["dataset_hashes"],
        "pilot_contract_sha256": pilot_contract_sha256,
        "pilot_arrays_sha256": pilot_arrays_sha256,
        "provenance": {
            **dict(provenance),
            "source_repair_result": str(source_detail_path),
            "source_repair_result_sha256": source_detail_sha256,
            "source_repair_diagnostics": str(source_diagnostics_path),
            "source_repair_diagnostics_sha256": source_diagnostics_sha256,
        },
        "regimes": regime_results,
        "draw_count_seed_gates": draw_count_seed_gates,
        "source_exact_pilot_seed_gate": source_exact_seed_gate,
        "source_full_plus_seed_gate": source_full_seed_gate,
        "wrong_sign_draw0_m256_seed_gate": wrong_sign_seed_gate,
        "gates": gates,
        "analysis_seconds": time.perf_counter() - started,
        "artifacts": {
            "result": str(result_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": _sha256(diagnostics_path),
        },
    }
    _write_json(result_path, value)
    count_passes = {
        str(pilot_count): sum(
            draw_count_seed_gates[f"draw_{draw_index:02d}"][str(pilot_count)]
            for draw_index in range(config.draw_count)
        )
        for pilot_count in config.counts
    }
    print(
        f"{condition} seed {seed}: draw_passes={count_passes} "
        f"wrong={wrong_sign_seed_gate} valid={gates['validity']}",
        flush=True,
    )
    return value


def _load_runtime_for_worker(
    config: NoisyBiasPilotConfig,
) -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int], Path],
    Mapping[str, Any],
    Mapping[str, Mapping[str, torch.Tensor]],
]:
    _, _, details, diagnostics = _load_repair_source(config)
    repair_config = _repair_config(config)
    base_config = repair._base_config(repair_config)
    arrays = repair.bias._load_base_noise(base_config)
    components = repair.bias.construct_components(arrays, base_config)
    runtime = repair.bias.dose._load_runtime_sources(
        repair.bias._dose_config(base_config)
    )
    return details, diagnostics, runtime, components


def run_worker(
    config: NoisyBiasPilotConfig, output: Path, worker_index: int
) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True)
    source_digests = _source_digests()
    implementation = _implementation_digest(source_digests)
    manifest, pilot_arrays = _load_prepared(config, output, implementation)
    details, diagnostics, runtime, components = _load_runtime_for_worker(config)
    device = torch.device(config.devices[worker_index])
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    results = []
    for condition, seed in _worker_cells(config, worker_index):
        system, provenance = repair.bias.dose._load_system(
            runtime, condition, seed, device
        )
        source_detail = details[(condition, seed)]
        result = analyze_cell(
            system=system,
            condition=condition,
            seed=seed,
            provenance=provenance,
            source_detail=source_detail,
            source_detail_path=Path(source_detail["artifacts"]["result"]),
            source_diagnostics_path=diagnostics[(condition, seed)],
            runtime=runtime,
            components=components,
            pilot_arrays=pilot_arrays,
            pilot_contract_sha256=manifest["pilot_contract_sha256"],
            pilot_arrays_sha256=manifest["pilot_arrays_sha256"],
            config=config,
            implementation=implementation,
            output=output,
            device=device,
        )
        results.append(result)
        del system
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if _implementation_digest() != implementation:
            raise RuntimeError("noisy-pilot implementation changed")
    worker_manifest = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "worker_completed",
        "worker_index": worker_index,
        "device": str(device),
        "configuration": _json_config(config),
        "implementation_sha256": implementation,
        "cells": [list(cell) for cell in _worker_cells(config, worker_index)],
        "results": [result["artifacts"]["result"] for result in results],
    }
    _write_json(output / f"worker_{worker_index}_manifest.json", worker_manifest)
    return worker_manifest


def _load_completed_results(
    config: NoisyBiasPilotConfig, output: Path, implementation: str
) -> list[dict[str, Any]]:
    values = []
    for cell_index, (condition, seed) in enumerate(_cell_order(config)):
        worker_index = cell_index % len(config.devices)
        path = output / "runs" / condition / f"seed_{seed}" / "result.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        diagnostics_path = Path(value.get("artifacts", {}).get("diagnostics", ""))
        if (
            value.get("status") != "completed"
            or value.get("schema_version") != SCHEMA_VERSION
            or value.get("hypothesis_id") != HYPOTHESIS_ID
            or value.get("condition") != condition
            or value.get("seed") != seed
            or value.get("device") != config.devices[worker_index]
            or value.get("configuration") != _json_config(config)
            or value.get("implementation_sha256") != implementation
            or value.get("gates", {}).get("validity") is not True
            or not diagnostics_path.is_file()
            or _sha256(diagnostics_path)
            != value.get("artifacts", {}).get("diagnostics_sha256")
        ):
            raise ValueError(f"invalid completed worker result {path}")
        values.append(value)
    return values


def aggregate_results(
    results: list[Mapping[str, Any]], config: NoisyBiasPilotConfig
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for condition in config.conditions:
        selected = [item for item in results if item["condition"] == condition]
        draws: dict[str, Any] = {}
        for draw_index in range(config.draw_count):
            draw_key = f"draw_{draw_index:02d}"
            draws[draw_key] = {}
            for pilot_count in config.counts:
                seed_passes = sum(
                    item["draw_count_seed_gates"][draw_key][str(pilot_count)]
                    for item in selected
                )
                draws[draw_key][str(pilot_count)] = {
                    "seed_passes": seed_passes,
                    "population_pass": seed_passes >= config.required_seed_passes,
                }
        arms[condition] = {
            "draws": draws,
            "source_exact_pilot_passes": sum(
                item["source_exact_pilot_seed_gate"] for item in selected
            ),
            "source_full_plus_passes": sum(
                item["source_full_plus_seed_gate"] for item in selected
            ),
            "wrong_sign_draw0_m256_passes": sum(
                item["wrong_sign_draw0_m256_seed_gate"] for item in selected
            ),
        }
    counts: dict[str, Any] = {}
    reliable_count = None
    for pilot_count in config.counts:
        complete_vector = [
            all(
                arms[condition]["draws"][f"draw_{draw_index:02d}"][
                    str(pilot_count)
                ]["population_pass"]
                for condition in config.conditions
            )
            for draw_index in range(config.draw_count)
        ]
        arm_draw_passes = {
            condition: sum(
                arms[condition]["draws"][f"draw_{draw_index:02d}"][
                    str(pilot_count)
                ]["population_pass"]
                for draw_index in range(config.draw_count)
            )
            for condition in config.conditions
        }
        complete_passes = sum(complete_vector)
        counts[str(pilot_count)] = {
            "arm_population_draw_passes": arm_draw_passes,
            "complete_draw_pass_vector": complete_vector,
            "complete_draw_passes": complete_passes,
            "draws": config.draw_count,
        }
        if reliable_count is None and complete_passes >= config.required_draw_passes:
            reliable_count = pilot_count
    full_population = (
        config.conditions == CONDITIONS and config.seeds == SEEDS
    )
    full_source_counts_pass = bool(
        not full_population
        or (
            arms["analytic_calibrated"]["source_full_plus_passes"] == 1
            and arms["learned_calibrated_equivariant"][
                "source_full_plus_passes"
            ]
            == 3
        )
    )
    source_controls_pass = bool(
        set(config.conditions) == set(CONDITIONS)
        and arms["analytic_calibrated"]["source_exact_pilot_passes"]
        == len(config.seeds)
        and arms["learned_calibrated_equivariant"][
            "source_exact_pilot_passes"
        ]
        == len(config.seeds)
        and full_source_counts_pass
        and arms["analytic_calibrated"]["wrong_sign_draw0_m256_passes"]
        <= config.maximum_control_seed_passes
        and arms["learned_calibrated_equivariant"][
            "wrong_sign_draw0_m256_passes"
        ]
        <= config.maximum_control_seed_passes
    )
    return {
        "arms": arms,
        "counts": counts,
        "smallest_reliable_count": reliable_count,
        "source_controls_pass": source_controls_pass,
        "integrity_valid": all(item["gates"]["validity"] for item in results),
    }


def classify_campaign(
    *,
    integrity_valid: bool,
    pilot_contract_pass: bool,
    source_controls_pass: bool,
    analytic_m256_draw_passes: int,
    learned_m256_draw_passes: int,
    required_draw_passes: int,
) -> tuple[str, bool]:
    if not integrity_valid or not pilot_contract_pass or not source_controls_pass:
        return "invalid", False
    analytic = analytic_m256_draw_passes >= required_draw_passes
    learned = learned_m256_draw_passes >= required_draw_passes
    if analytic and learned:
        return "finite_noisy_pilot_repair_reliable", True
    if analytic != learned:
        return "finite_noisy_pilot_arm_asymmetry", False
    return "finite_noisy_pilot_insufficient", False


def _result_entry(result: Mapping[str, Any]) -> dict[str, Any]:
    result_path = Path(result["artifacts"]["result"])
    diagnostics_path = Path(result["artifacts"]["diagnostics"])
    return {
        "condition": result["condition"],
        "seed": result["seed"],
        "device": result["device"],
        "path": str(result_path),
        "result_sha256": _sha256(result_path),
        "diagnostics_path": str(diagnostics_path),
        "diagnostics_sha256": _sha256(diagnostics_path),
        "scientific_fingerprint": result["scientific_fingerprint"],
        "validity": result["gates"]["validity"],
    }


def aggregate_campaign(
    config: NoisyBiasPilotConfig, output: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    source_digests = _source_digests()
    implementation = _implementation_digest(source_digests)
    manifest, _ = _load_prepared(config, output, implementation)
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        entries = existing.get("results", [])
        if (
            existing.get("status") == "completed"
            and existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("configuration") == _json_config(config)
            and existing.get("implementation_sha256") == implementation
            and existing.get("result_manifest_sha256") == _json_hash(entries)
            and all(
                Path(entry["path"]).is_file()
                and _sha256(Path(entry["path"])) == entry["result_sha256"]
                and Path(entry["diagnostics_path"]).is_file()
                and _sha256(Path(entry["diagnostics_path"]))
                == entry["diagnostics_sha256"]
                for entry in entries
            )
        ):
            print("campaign already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed campaign {campaign_path}")
    repair_campaign, repair_path, _, _ = _load_repair_source(config)
    acquisition_campaign, acquisition_path, _ = _load_acquisition_source(config)
    results = _load_completed_results(config, output, implementation)
    population = aggregate_results(results, config)
    m256 = population["counts"][str(config.counts[-1])]
    classification, primary_pass = classify_campaign(
        integrity_valid=population["integrity_valid"],
        pilot_contract_pass=manifest["pilot_contract"]["pass"],
        source_controls_pass=population["source_controls_pass"],
        analytic_m256_draw_passes=m256["arm_population_draw_passes"][
            "analytic_calibrated"
        ],
        learned_m256_draw_passes=m256["arm_population_draw_passes"][
            "learned_calibrated_equivariant"
        ],
        required_draw_passes=config.required_draw_passes,
    )
    primary_evaluable = not config.allow_underpowered
    recorded_classification = (
        classification
        if primary_evaluable
        else f"underpowered_shakedown__{classification}"
    )
    entries = [_result_entry(result) for result in results]
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": EVIDENCE_ROLE,
        "completed_at": _utc_now(),
        "configuration": _json_config(config),
        "implementation_sha256": implementation,
        "source_digests": source_digests,
        "source_repair_campaign": str(repair_path),
        "source_repair_campaign_sha256": SOURCE_REPAIR_CAMPAIGN_SHA256,
        "source_repair_result_manifest_sha256": (
            SOURCE_REPAIR_RESULT_MANIFEST_SHA256
        ),
        "source_repair_contract_sha256": SOURCE_REPAIR_CONTRACT_SHA256,
        "source_acquisition_campaign": str(
            Path(config.source_acquisition_root) / "campaign_results.json"
        ),
        "source_acquisition_campaign_sha256": (
            SOURCE_ACQUISITION_CAMPAIGN_SHA256
        ),
        "source_acquisition_result_manifest_sha256": (
            SOURCE_ACQUISITION_RESULT_MANIFEST_SHA256
        ),
        "source_acquisition_arrays": str(acquisition_path),
        "source_acquisition_arrays_sha256": SOURCE_ACQUISITION_ARRAY_SHA256,
        "source_dvc_root": SOURCE_DVC_ROOT,
        "source_lakefs_commit": SOURCE_LAKEFS_COMMIT,
        "dataset_hashes": repair_campaign["dataset_hashes"],
        "pilot_contract": manifest["pilot_contract"],
        "pilot_contract_sha256": manifest["pilot_contract_sha256"],
        "pilot_arrays": manifest["pilot_arrays"],
        "pilot_arrays_sha256": manifest["pilot_arrays_sha256"],
        "pilot_array_content_sha256": manifest["pilot_array_content_sha256"],
        "population": population,
        "aggregates": {
            "classification": recorded_classification,
            "shakedown_outcome_classification": (
                None if primary_evaluable else classification
            ),
            "primary_hypothesis_pass": bool(
                primary_evaluable and primary_pass
            ),
            "primary_evaluable": primary_evaluable,
            "valid": bool(
                population["integrity_valid"]
                and manifest["pilot_contract"]["pass"]
                and population["source_controls_pass"]
            ),
            "integrity_valid": population["integrity_valid"],
            "pilot_contract_pass": manifest["pilot_contract"]["pass"],
            "source_controls_pass": population["source_controls_pass"],
            "smallest_reliable_count": population["smallest_reliable_count"],
            "required_seed_passes": config.required_seed_passes,
            "required_draw_passes": config.required_draw_passes,
            "maximum_control_seed_passes": config.maximum_control_seed_passes,
        },
        "results": entries,
        "result_manifest_sha256": _json_hash(entries),
        "summary": {
            "requested": len(config.conditions) * len(config.seeds),
            "completed": len(results),
            "failed": 0,
            "excluded": 0,
            "trained_models": 0,
            "trained_frontends": 0,
            "trained_task_heads": 0,
            "fitted_bias_estimators": 0,
            "fitted_denoisers": 0,
            "fitted_observers": 0,
            "fitted_probes": 0,
            "new_random_draws": 0,
            "reused_acquisition_draws": config.draw_count,
            "evaluated_pilot_counts": len(config.counts),
            "worker_count": len(config.devices),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "devices": list(config.devices),
            "cuda_available": torch.cuda.is_available(),
            "deterministic_algorithms": True,
        },
        "analysis_seconds": time.perf_counter() - started
        + sum(float(result["analysis_seconds"]) for result in results),
        "method_boundaries": [
            "The pilot streams are reused sealed Gaussian arrays generated independently of the evaluation sensor-noise arrays; no new randomness is introduced.",
            "The two selected scalar streams are interpreted as planar zero-signal pilot noise with per-axis sigma=0.03125/sqrt(2).",
            "Each draw supplies one global pilot estimate shared across examples, systems, and both evaluation shifts.",
            "The study tests only independent unbiased homoscedastic Gaussian pilot error, not drift, heavy tails, correlation, or example-dependent bias.",
            "The exact-pilot target-changing control is pinned from the source campaign rather than rerun.",
            "No model, front end, head, bias estimator, denoiser, observer, probe, or noise process is trained or fitted.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "pilot_arrays": manifest["pilot_arrays"],
        },
    }
    del acquisition_campaign
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
            "data/experiments/tinyllm_noisy_bias_pilot_acquisition/"
            "20260810_d16_preregistered"
        ),
    )
    parser.add_argument("--conditions", type=_strings, default=CONDITIONS)
    parser.add_argument("--seeds", type=_ints, default=SEEDS)
    parser.add_argument("--counts", type=_ints, default=COUNTS)
    parser.add_argument("--draw-count", type=int, default=DRAW_COUNT)
    parser.add_argument("--devices", type=_strings, default=("cuda:0", "cuda:1", "cuda:2"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--required-seed-passes", type=int, default=4)
    parser.add_argument("--required-draw-passes", type=int, default=15)
    parser.add_argument("--maximum-control-seed-passes", type=int, default=1)
    parser.add_argument("--allow-underpowered", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--worker-index", type=int)
    mode.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    config = NoisyBiasPilotConfig(
        conditions=args.conditions,
        seeds=args.seeds,
        counts=args.counts,
        draw_count=args.draw_count,
        devices=args.devices,
        batch_size=args.batch_size,
        sample_limit=args.sample_limit,
        required_seed_passes=args.required_seed_passes,
        required_draw_passes=args.required_draw_passes,
        maximum_control_seed_passes=args.maximum_control_seed_passes,
        allow_underpowered=args.allow_underpowered,
    )
    if args.prepare_only:
        value = prepare_campaign(config, args.output)
        print(json.dumps(value["pilot_contract"], indent=2, sort_keys=True))
        return 0
    if args.worker_index is not None:
        value = run_worker(config, args.output, args.worker_index)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.aggregate_only:
        value = aggregate_campaign(config, args.output)
        print(json.dumps(value["aggregates"], indent=2, sort_keys=True))
        return 0
    prepare_campaign(config, args.output)
    for worker_index in range(len(config.devices)):
        run_worker(config, args.output, worker_index)
    value = aggregate_campaign(config, args.output)
    print(json.dumps(value["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
