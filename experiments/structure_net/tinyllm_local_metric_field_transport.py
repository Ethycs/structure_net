#!/usr/bin/env python3
"""Test nuisance-group transport of TinyLLM's local task-metric field."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import experiments.structure_net.tinyllm_local_continuation_tangent_kernel as local
import experiments.structure_net.tinyllm_frozen_writer_capacity as writer
import experiments.structure_net.tinyllm_fixed_semantic_gauge_writer as fixed
import experiments.structure_net.tinyllm_cross_seed_causal_carrier_transport as transport
import experiments.structure_net.tinyllm_deck_action_descrambler as deck
import experiments.structure_net.tinyllm_nuisance_support_scaling as nuisance
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-c2-local-metric-field-transport.v1"
HYPOTHESIS_ID = "tinyllm-c2-local-metric-field-transport-v1"
WRITER_CAMPAIGN_SHA256 = (
    "7ab6079d56bc8dc78cc5e1f4011c581b8f507bcd8ec20573bbf8b11227df8d1b"
)
WRITER_IMPLEMENTATION_SHA256 = (
    "d53edaedd49ae553af9f8393d92254664239e5100246ac0fd3a06cb420ca80ed"
)
CORRECTIVE_CAMPAIGN_SHA256 = (
    "8b2f25dc61d1ffb93ba9dc66f1a85f36c07bda5a43ed3ed34a9627ca1485c12a"
)
CORRECTIVE_IMPLEMENTATION_SHA256 = (
    "31ef191a31bbd5d509fb912cd5164385d846039a3b7c98aca04e6f5e29835c38"
)
GROUP_ARMS = ("amplitude", "orientation", "offset", "composed")
REGIMES = ("composition", "extrapolation")
FRESH_COHORT_SEEDS = {"composition": 430_007, "extrapolation": 430_008}


@dataclass(frozen=True)
class LocalMetricFieldTransportConfig:
    writer_root: str = (
        "data/experiments/tinyllm_frozen_writer_capacity/"
        "20260807_d6_preregistered_diagnostic"
    )
    corrective_root: str = (
        "data/experiments/tinyllm_local_continuation_tangent_kernel/"
        "20260807_d6_corrective_v2"
    )
    seeds: tuple[int, ...] = fixed.PRIMARY_SEEDS
    orbit_count: int = 64
    phase_count: int = 16
    nuisance_replicates: int = 4
    carrier_rank: int = 3
    fourier_order: int = 4
    group_arms: tuple[str, ...] = GROUP_ARMS
    jacobian_rtol: float = 1e-6
    jacobian_atol: float = 1e-10
    finite_difference_step: float = 1e-2
    finite_difference_tolerance: float = 0.05
    decomposition_tolerance: float = 1e-6
    clipping_fraction_ceiling: float = 0.01
    changed_token_fraction_floor: float = 0.02
    median_kernel_cosine_floor: float = 0.95
    p10_kernel_cosine_floor: float = 0.90
    p95_projector_distance_ceiling: float = math.sqrt(1.0 - 0.90**2)
    shuffled_cosine_margin: float = 0.10
    causal_control_margin_bins: float = 0.05
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != fixed.PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("primary metric-field seeds are fixed to 7,29,53")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.orbit_count != 64:
            raise ValueError("fresh metric-field cohorts fix 64 exact orbits")
        if self.phase_count != 16 or self.nuisance_replicates != 4:
            raise ValueError("fresh cohorts fix 16 phases and four nuisance replicates")
        if self.phase_count * self.nuisance_replicates != self.orbit_count:
            raise ValueError("phase and replicate counts must tile the orbit cohort")
        if self.carrier_rank != 3 or self.fourier_order != 4:
            raise ValueError("carrier rank and writer order are fixed to three and four")
        if tuple(self.group_arms) != GROUP_ARMS:
            raise ValueError("group arms are fixed to amplitude, orientation, offset, composed")
        positive = (
            self.jacobian_rtol,
            self.jacobian_atol,
            self.finite_difference_step,
            self.finite_difference_tolerance,
            self.decomposition_tolerance,
            self.clipping_fraction_ceiling,
            self.changed_token_fraction_floor,
            self.median_kernel_cosine_floor,
            self.p10_kernel_cosine_floor,
            self.p95_projector_distance_ceiling,
            self.shuffled_cosine_margin,
            self.causal_control_margin_bins,
        )
        if min(positive) <= 0.0:
            raise ValueError("all numerical thresholds must be positive")


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


def _json_compatible(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(local.__file__),
        Path(writer.__file__),
        Path(fixed.__file__),
        Path(transport.__file__),
        Path(transport.rank.__file__),
        Path(deck.__file__),
        Path(nuisance.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evidence_role(config: LocalMetricFieldTransportConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_fresh_cohort_underpowered_mechanistic_diagnostic"
    )


def _tensor_digest(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def phase_matched_shift_indices(
    phase_count: int, nuisance_replicates: int
) -> torch.Tensor:
    """Cycle nuisance replicate while preserving each quotient phase."""
    indices = torch.arange(phase_count * nuisance_replicates).reshape(
        phase_count, nuisance_replicates
    )
    return indices.roll(-1, dims=1).reshape(-1)


def _repeat_orbits(values: Mapping[str, np.ndarray], k: int) -> dict[str, np.ndarray]:
    return {name: np.repeat(value, k, axis=0) for name, value in values.items()}


def _group_sensor(
    base: np.ndarray,
    amplitude: np.ndarray,
    orientation: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """Apply rotate, scale, translate to an observed three-channel history."""
    cosine = np.cos(orientation)
    sine = np.sin(orientation)
    result = base.copy()
    x_value = base[..., 0]
    y_value = base[..., 1]
    result[..., 0] = cosine * x_value - sine * y_value
    result[..., 1] = sine * x_value + cosine * y_value
    return amplitude * result + offset


def _dataset_from_sensor(
    task: CircleTaskConfig,
    sensor_values: np.ndarray,
    full_values: Mapping[str, np.ndarray],
    directions: np.ndarray,
    future: np.ndarray,
    theta_flat: np.ndarray,
    branch: np.ndarray,
    orbit_count: int,
) -> deck.OrbitDataset:
    input_ids = torch.from_numpy(deck.io_source._serialize_sensor_values(sensor_values, task))
    sensor = deck.io_source.decode_sensor_tokens(input_ids, task)
    calibration = deck.calibrated.calibration_packet(
        full_values, torch.tensor(directions, dtype=torch.float32)
    )
    posterior, bins = deck.ladder._targets(future, 2, task.phase_bins)
    return deck.OrbitDataset(
        input_ids=input_ids,
        sensor=sensor,
        calibration=calibration,
        phase=torch.tensor(future, dtype=torch.float32),
        quotient_phase=torch.tensor(theta_flat, dtype=torch.float32),
        branch=torch.tensor(branch, dtype=torch.long),
        target_posteriors=posterior,
        target_bins=bins,
        orbit_count=orbit_count,
        k=2,
    )


def generate_group_paired_orbits(
    task: CircleTaskConfig,
    config: LocalMetricFieldTransportConfig,
    *,
    seed: int,
    regime: str,
) -> tuple[dict[str, deck.OrbitDataset], dict[str, Any]]:
    """Generate fresh exact C2 orbits and exact observed group transforms."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime}")
    phase_centers = (
        np.arange(config.phase_count, dtype=np.float64) + 0.5
    ) * (2.0 * math.pi / config.phase_count)
    theta = np.repeat(phase_centers, config.nuisance_replicates)
    orbit_count = len(theta)
    branch = np.tile(np.arange(2, dtype=np.int64), orbit_count)
    future = np.stack((theta / 2.0, (theta + 2.0 * math.pi) / 2.0), axis=1).reshape(-1)
    theta_flat = np.repeat(theta, 2)

    base_generator = np.random.default_rng(seed + 11_003)
    base_values = nuisance._nuisance_values(
        "N3", "interpolation", orbit_count, base_generator
    )
    base_directions = nuisance._sample_directions(
        "N3", "interpolation", orbit_count, base_generator
    )
    values = _repeat_orbits(base_values, 2)
    directions = np.repeat(base_directions, 2)
    current = np.remainder(future - directions * task.future_delta, 2.0 * math.pi)
    history = np.arange(task.sensor_steps, dtype=np.float64) - (task.sensor_steps - 1)
    angles = current[:, None] + directions[:, None] * history[None, :] * values["speed"]
    harmonic = values["harmonic_strength"] * np.cos(
        values["harmonic_order"] * angles + values["harmonic_phase"]
    )
    base_sensor = np.stack((np.cos(angles), np.sin(angles), harmonic), axis=-1)
    normalized_history = history / max(1.0, float(task.sensor_steps - 1))
    base_sensor += values["drift"] * normalized_history[None, :, None]
    orbit_noise = base_generator.normal(0.0, 1.0, (orbit_count, task.sensor_steps, 3))
    base_sensor += np.repeat(orbit_noise, 2, axis=0) * values["noise_scale"]

    group_generator = np.random.default_rng(seed + 29_011)
    group_values_base = nuisance._nuisance_values(
        "N3", regime, orbit_count, group_generator
    )
    group_values = _repeat_orbits(group_values_base, 2)
    ones = np.ones_like(group_values["amplitude"])
    zeros_orientation = np.zeros_like(group_values["orientation"])
    zeros_offset = np.zeros_like(group_values["offset"])
    group_parameters = {
        "reference": (ones, zeros_orientation, zeros_offset),
        "amplitude": (group_values["amplitude"], zeros_orientation, zeros_offset),
        "orientation": (ones, group_values["orientation"], zeros_offset),
        "offset": (ones, zeros_orientation, group_values["offset"]),
        "composed": (
            group_values["amplitude"],
            group_values["orientation"],
            group_values["offset"],
        ),
    }

    datasets: dict[str, deck.OrbitDataset] = {}
    sensors: dict[str, np.ndarray] = {}
    summaries: dict[str, Any] = {}
    for arm, (amplitude, orientation, offset) in group_parameters.items():
        sensor_values = _group_sensor(base_sensor, amplitude, orientation, offset)
        sensors[arm] = sensor_values
        full_values = {name: value.copy() for name, value in values.items()}
        full_values["amplitude"] = amplitude
        full_values["orientation"] = orientation
        full_values["offset"] = offset
        dataset = _dataset_from_sensor(
            task,
            sensor_values,
            full_values,
            directions,
            future,
            theta_flat,
            branch,
            orbit_count,
        )
        datasets[arm] = dataset
        summaries[arm] = {
            "clipping_fraction": float(
                np.mean(np.abs(sensor_values) > task.quantization_limit)
            ),
            "nonfinite_count": int(np.size(sensor_values) - np.isfinite(sensor_values).sum()),
            "input_sha256": _tensor_digest(dataset.input_ids),
        }

    reference_tokens = datasets["reference"].input_ids[:, 1:-1]
    for arm in GROUP_ARMS:
        tokens = datasets[arm].input_ids[:, 1:-1]
        summaries[arm]["changed_token_fraction"] = float(
            (tokens != reference_tokens).double().mean()
        )
    summaries["reference"]["changed_token_fraction"] = 0.0
    summaries["group_parameter_sha256"] = hashlib.sha256(
        b"".join(
            np.ascontiguousarray(group_values[name]).tobytes()
            for name in ("amplitude", "orientation", "offset")
        )
    ).hexdigest()
    summaries["quotient_phase_sha256"] = _tensor_digest(
        datasets["reference"].quotient_phase
    )
    summaries["target_sha256"] = _tensor_digest(
        datasets["reference"].target_posteriors,
        datasets["reference"].target_bins,
    )
    summaries["pair_contract"] = bool(
        all(
            torch.equal(datasets["reference"].target_posteriors, datasets[arm].target_posteriors)
            and torch.equal(datasets["reference"].target_bins, datasets[arm].target_bins)
            and torch.equal(datasets["reference"].quotient_phase, datasets[arm].quotient_phase)
            for arm in GROUP_ARMS
        )
    )
    return datasets, summaries


def projector_geometry(
    reference: torch.Tensor,
    transformed: torch.Tensor,
    shuffled_indices: torch.Tensor,
) -> dict[str, torch.Tensor | float]:
    """Compare paired rank-two projectors through their rank-one kernels."""
    reference = 0.5 * (reference.double() + reference.double().transpose(-1, -2))
    transformed = 0.5 * (
        transformed.double() + transformed.double().transpose(-1, -2)
    )
    _, reference_vectors = torch.linalg.eigh(reference)
    _, transformed_vectors = torch.linalg.eigh(transformed)
    reference_kernel = reference_vectors[:, :, 0]
    transformed_kernel = transformed_vectors[:, :, 0]
    paired_cosine = (reference_kernel * transformed_kernel).sum(1).abs()
    shuffled_cosine = (
        reference_kernel[shuffled_indices.to(reference.device)] * transformed_kernel
    ).sum(1).abs()
    distance = torch.linalg.matrix_norm(transformed - reference, dim=(-2, -1)) / math.sqrt(2.0)
    return {
        "paired_kernel_cosine": paired_cosine,
        "shuffled_kernel_cosine": shuffled_cosine,
        "projector_distance": distance,
        "median_kernel_cosine": float(paired_cosine.median()),
        "p10_kernel_cosine": float(torch.quantile(paired_cosine, 0.10)),
        "p95_projector_distance": float(torch.quantile(distance, 0.95)),
        "shuffled_median_kernel_cosine": float(shuffled_cosine.median()),
        "paired_over_shuffled_median_margin": float(
            paired_cosine.median() - shuffled_cosine.median()
        ),
    }


def random_rank_two_projectors(
    count: int, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    kernel = torch.randn((count, 3), generator=generator, dtype=torch.float64)
    kernel /= torch.linalg.vector_norm(kernel, dim=1, keepdim=True).clamp_min(1e-24)
    kernel = kernel.to(device)
    identity = torch.eye(3, dtype=torch.float64, device=device).expand(count, -1, -1)
    return identity - kernel[:, :, None] * kernel[:, None, :]


def _random_seed(seed: int, regime: str, arm: str) -> int:
    payload = f"{HYPOTHESIS_ID}:{seed}:{regime}:{arm}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _cell_summary(
    jacobian: torch.Tensor,
    decomposition: Mapping[str, torch.Tensor],
    finite_error: torch.Tensor,
) -> dict[str, Any]:
    return {
        "minimum_jacobian_rank": int(decomposition["rank"].min()),
        "maximum_jacobian_rank": int(decomposition["rank"].max()),
        "mean_singular_values": decomposition["singular_values"].mean(0).cpu().tolist(),
        "maximum_finite_difference_relative_error": float(finite_error.max()),
        "mean_finite_difference_relative_error": float(finite_error.mean()),
        "maximum_decomposition_relative_error": float(
            decomposition["relative_decomposition_error"].max()
        ),
        "maximum_relative_tangent_mismatch": float(
            decomposition["relative_tangent_mismatch"].max()
        ),
        "maximum_relative_kernel_leakage": float(
            decomposition["relative_kernel_leakage"].max()
        ),
        "jacobian_frobenius_mean": float(
            torch.linalg.matrix_norm(jacobian.double(), dim=(-2, -1)).mean()
        ),
    }


def classify_checkpoint(
    *,
    valid: bool,
    local_tangent_pass: bool,
    geometry_pass: bool,
    causal_transport_pass: bool,
) -> str:
    if not valid:
        return "invalid"
    if geometry_pass and causal_transport_pass:
        return "invariant_metric_field_transport_supported"
    if causal_transport_pass:
        return "causal_transport_without_geometric_invariance"
    if geometry_pass:
        return "geometric_invariance_without_causal_transport"
    if not local_tangent_pass:
        return "local_tangent_not_fresh_cohort_sufficient"
    return "nuisance_specific_metric_field"


def _load_sources(
    config: LocalMetricFieldTransportConfig,
) -> tuple[
    dict[str, Any],
    Path,
    dict[int, tuple[dict[str, Any], Path]],
    dict[str, Any],
    Path,
]:
    writer_path = Path(config.writer_root) / "campaign_results.json"
    writer_campaign = json.loads(writer_path.read_text())
    if (
        _sha256(writer_path) != WRITER_CAMPAIGN_SHA256
        or writer_campaign.get("schema_version") != writer.SCHEMA_VERSION
        or writer_campaign.get("hypothesis_id") != writer.HYPOTHESIS_ID
        or writer_campaign.get("status") != "completed"
        or writer_campaign.get("implementation_sha256") != WRITER_IMPLEMENTATION_SHA256
        or writer_campaign.get("aggregates", {}).get("conclusion")
        != "small_writer_insufficient"
    ):
        raise ValueError(f"invalid writer campaign {writer_path}")
    details: dict[int, tuple[dict[str, Any], Path]] = {}
    for entry in writer_campaign["results"]:
        path = Path(entry["path"])
        detail = json.loads(path.read_text())
        if (
            _sha256(path) != entry.get("result_sha256")
            or detail.get("scientific_fingerprint") != entry.get("scientific_fingerprint")
            or detail.get("classification") != "small_writer_insufficient"
            or "fourier_m04" not in detail.get("alignment_fit", {}).get("mappings", {})
        ):
            raise ValueError(f"invalid writer result {path}")
        details[int(detail["seed"])] = (detail, path)
    if not set(config.seeds).issubset(details):
        raise ValueError("writer campaign lacks a requested checkpoint")

    corrective_path = Path(config.corrective_root) / "campaign_results.json"
    corrective = json.loads(corrective_path.read_text())
    if (
        _sha256(corrective_path) != CORRECTIVE_CAMPAIGN_SHA256
        or corrective.get("schema_version") != local.SCHEMA_VERSION
        or corrective.get("hypothesis_id") != local.HYPOTHESIS_ID
        or corrective.get("status") != "completed"
        or corrective.get("implementation_sha256") != CORRECTIVE_IMPLEMENTATION_SHA256
        or corrective.get("aggregates", {}).get("conclusion")
        != "mixed_local_continuation_geometry"
    ):
        raise ValueError(f"invalid corrective campaign {corrective_path}")
    return writer_campaign, writer_path, details, corrective, corrective_path


def _fingerprint(
    config: LocalMetricFieldTransportConfig,
    seed: int,
    writer_result_sha256: str,
    checkpoint_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "fresh_cohort_seeds": FRESH_COHORT_SEEDS,
        "seed": seed,
        "writer_campaign_sha256": WRITER_CAMPAIGN_SHA256,
        "corrective_campaign_sha256": CORRECTIVE_CAMPAIGN_SHA256,
        "writer_result_sha256": writer_result_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _campaign_is_reusable(
    value: Mapping[str, Any],
    config: LocalMetricFieldTransportConfig,
    implementation: str,
) -> bool:
    if not (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("evidence_role") == _evidence_role(config)
        and value.get("implementation_sha256") == implementation
        and value.get("configuration") == _json_compatible(asdict(config))
        and int(value.get("summary", {}).get("completed", -1)) == len(config.seeds)
    ):
        return False
    for entry in value.get("results", []):
        path = Path(entry.get("path", ""))
        arrays = Path(entry.get("arrays", ""))
        if (
            not path.is_file()
            or not arrays.is_file()
            or _sha256(path) != entry.get("result_sha256")
            or _sha256(arrays) != entry.get("arrays_sha256")
        ):
            return False
    return len(value.get("results", [])) == len(config.seeds)


def _reusable_seed_result(
    result_path: Path,
    arrays_path: Path,
    config: LocalMetricFieldTransportConfig,
    implementation: str,
    seed: int,
    fingerprint: str,
) -> dict[str, Any] | None:
    """Return a fingerprint-matched completed seed without rewriting its bytes."""
    if not result_path.is_file():
        return None
    existing = json.loads(result_path.read_text())
    valid = bool(
        existing.get("schema_version") == SCHEMA_VERSION
        and existing.get("hypothesis_id") == HYPOTHESIS_ID
        and existing.get("status") == "completed"
        and existing.get("evidence_role") == _evidence_role(config)
        and existing.get("implementation_sha256") == implementation
        and existing.get("scientific_fingerprint") == fingerprint
        and existing.get("configuration") == _json_compatible(asdict(config))
        and int(existing.get("seed", -1)) == seed
        and existing.get("artifacts", {}).get("result") == str(result_path)
        and existing.get("artifacts", {}).get("arrays") == str(arrays_path)
        and arrays_path.is_file()
        and _sha256(arrays_path)
        == existing.get("artifacts", {}).get("arrays_sha256")
    )
    if not valid:
        raise ValueError(f"incompatible completed result {result_path}; use a new root")
    return existing


def run_campaign(
    config: LocalMetricFieldTransportConfig, output: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    implementation = _implementation_digest()
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text())
        if _campaign_is_reusable(existing, config, implementation):
            print("aggregate already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed aggregate {campaign_path}")

    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    writer_campaign, writer_path, writer_details, corrective, corrective_path = (
        _load_sources(config)
    )
    paired = {
        regime: generate_group_paired_orbits(
            task,
            config,
            seed=FRESH_COHORT_SEEDS[regime],
            regime=regime,
        )
        for regime in REGIMES
    }
    shift_indices = phase_matched_shift_indices(
        config.phase_count, config.nuisance_replicates
    )

    fixed_config = fixed.FixedSemanticGaugeWriterConfig(
        seeds=config.seeds,
        orbit_count=config.orbit_count,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )
    transport_config = fixed._transport_config(fixed_config)
    rank_config = transport._rank_config(transport_config)
    bridge = transport.rank._bridge_config(rank_config)

    results: list[dict[str, Any]] = []
    reused = 0
    for seed in config.seeds:
        seed_started = time.perf_counter()
        prior, prior_path = writer_details[seed]
        fingerprint = _fingerprint(
            config,
            seed,
            _sha256(prior_path),
            prior["provenance"]["checkpoint_sha256"],
        )
        result_path = output / "runs" / f"seed_{seed}" / "result.json"
        arrays_path = output / "runs" / f"seed_{seed}" / "metric_field_arrays.npz"
        existing = _reusable_seed_result(
            result_path,
            arrays_path,
            config,
            implementation,
            seed,
            fingerprint,
        )
        if existing is not None:
            results.append(existing)
            reused += 1
            print(f"resuming {existing['experiment_id']}", flush=True)
            continue

        system, provenance = transport.rank.deck.load_source(task, bridge, 2, seed, device)
        if provenance["checkpoint_sha256"] != prior["provenance"]["checkpoint_sha256"]:
            raise ValueError(f"checkpoint mismatch for seed {seed}")
        frozen, frozen_path = transport.rank._load_character_source(
            rank_config, seed, provenance["checkpoint_sha256"]
        )
        basis, basis_summary = transport._fit_seed_basis(
            system,
            task,
            transport_config,
            rank_config,
            bridge,
            frozen,
            seed,
            device,
        )
        readout_predecessor = transport._load_readout_predecessor(
            transport_config, seed, provenance["checkpoint_sha256"]
        )
        mapping_record = prior["alignment_fit"]["mappings"]["fourier_m04"]
        mapping = {
            "linear": torch.tensor(
                mapping_record["linear"], dtype=torch.float64, device=device
            ),
            "intercept": torch.tensor(
                mapping_record["intercept"], dtype=torch.float64, device=device
            ),
        }

        cells: dict[str, dict[str, Mapping[str, Any]]] = {}
        analyses: dict[str, dict[str, dict[str, Any]]] = {}
        array_payload: dict[str, np.ndarray] = {}
        numerical_valid = True
        full_control_pass = True
        local_tangent_pass = True
        geometry_cells: list[dict[str, Any]] = []
        causal_cells: list[dict[str, Any]] = []
        for regime in REGIMES:
            datasets, input_summary = paired[regime]
            cells[regime] = {}
            analyses[regime] = {}
            theta = (
                datasets["reference"]
                .quotient_phase.to(device)
                .reshape(config.orbit_count, 2)[:, 0]
                .double()
            )
            features = writer.fourier_features(theta, config.fourier_order)
            predicted = transport.apply_affine(features, mapping)
            for name, dataset in datasets.items():
                cell = transport._extract_cell(
                    system,
                    task,
                    transport_config,
                    bridge,
                    dataset,
                    "heldout_a",
                    regime,
                    device,
                )
                cell["cohort"] = "fresh_group_pair"
                cell["evaluation_seed"] = FRESH_COHORT_SEEDS[regime]
                cells[regime][name] = cell
                target = transport._coordinates(cell, basis)
                residual = target - predicted
                _, jacobian = local.continuation_moment_jacobian(
                    system,
                    cell["target_cut"],
                    cell["propagated"],
                    predicted,
                    basis,
                    task,
                    rank_config.continuation_batch_size,
                )
                decomposition = local.tangent_kernel_decomposition(
                    jacobian,
                    residual,
                    rtol=config.jacobian_rtol,
                    atol=config.jacobian_atol,
                )
                finite_error = local.finite_difference_error(
                    system,
                    task,
                    cell["target_cut"],
                    cell["propagated"],
                    predicted,
                    basis,
                    jacobian,
                    residual,
                    config.finite_difference_step,
                    rank_config.continuation_batch_size,
                )
                summary = _cell_summary(jacobian, decomposition, finite_error)
                analyses[regime][name] = {
                    "target": target,
                    "residual": residual,
                    "jacobian": jacobian,
                    "projector": decomposition["projector"],
                    "local_tangent": decomposition["tangent"],
                    "summary": summary,
                }
                prefix = f"{regime}__{name}"
                array_payload[f"{prefix}__jacobian"] = jacobian.cpu().numpy()
                array_payload[f"{prefix}__projector"] = decomposition[
                    "projector"
                ].cpu().numpy()
                array_payload[f"{prefix}__target"] = target.cpu().numpy()
                array_payload[f"{prefix}__residual"] = residual.cpu().numpy()
                numerical_valid = bool(
                    numerical_valid
                    and summary["minimum_jacobian_rank"] == 2
                    and summary["maximum_jacobian_rank"] == 2
                    and summary["maximum_finite_difference_relative_error"]
                    <= config.finite_difference_tolerance
                    and summary["maximum_decomposition_relative_error"]
                    <= config.decomposition_tolerance
                    and summary["maximum_relative_tangent_mismatch"]
                    <= config.decomposition_tolerance
                    and summary["maximum_relative_kernel_leakage"]
                    <= config.decomposition_tolerance
                )

            reference = analyses[regime]["reference"]
            for arm in GROUP_ARMS:
                transformed = analyses[regime][arm]
                geometry = projector_geometry(
                    reference["projector"],
                    transformed["projector"],
                    shift_indices,
                )
                geometry_pass = bool(
                    geometry["median_kernel_cosine"]
                    >= config.median_kernel_cosine_floor
                    and geometry["p10_kernel_cosine"]
                    >= config.p10_kernel_cosine_floor
                    and geometry["p95_projector_distance"]
                    <= config.p95_projector_distance_ceiling
                    and geometry["paired_over_shuffled_median_margin"]
                    >= config.shuffled_cosine_margin
                )
                random_projector = random_rank_two_projectors(
                    config.orbit_count,
                    _random_seed(seed, regime, arm),
                    device,
                )
                residual = transformed["residual"]
                reference_projector = reference["projector"]
                shuffled_projector = reference_projector[
                    shift_indices.to(reference_projector.device)
                ]
                coordinate_states = {
                    "predicted": predicted,
                    "full": transformed["target"],
                    "local_tangent": predicted + transformed["local_tangent"],
                    "transported_tangent": predicted
                    + torch.einsum("nij,nj->ni", reference_projector, residual),
                    "shuffled_tangent": predicted
                    + torch.einsum("nij,nj->ni", shuffled_projector, residual),
                    "random_tangent": predicted
                    + torch.einsum("nij,nj->ni", random_projector, residual),
                }
                states = local.evaluate_states(
                    system,
                    task,
                    config,
                    cells[regime][arm],
                    basis,
                    coordinate_states,
                    readout_predecessor["rotation_bins"],
                )
                full_control_pass = bool(
                    full_control_pass
                    and states["full"]["continuous"]["continuous_pass"]
                )
                local_tangent_pass = bool(
                    local_tangent_pass
                    and states["local_tangent"]["continuous"]["continuous_pass"]
                )
                input_record = input_summary[arm]
                input_pass = bool(
                    input_record["clipping_fraction"]
                    <= config.clipping_fraction_ceiling
                    and input_record["nonfinite_count"] == 0
                    and input_record["changed_token_fraction"]
                    >= config.changed_token_fraction_floor
                    and input_summary["pair_contract"]
                )
                numerical_valid = bool(numerical_valid and input_pass)
                geometry_record = {
                    "regime": regime,
                    "arm": arm,
                    "input": input_record,
                    "geometry": {
                        name: value
                        for name, value in geometry.items()
                        if isinstance(value, float)
                    },
                    "geometry_pass": geometry_pass,
                    "jacobian": transformed["summary"],
                    "reference_jacobian": reference["summary"],
                }
                causal_record = {
                    "regime": regime,
                    "arm": arm,
                    "states": states,
                }
                geometry_cells.append(geometry_record)
                causal_cells.append(causal_record)
                array_payload[
                    f"{regime}__{arm}__paired_kernel_cosine"
                ] = geometry["paired_kernel_cosine"].cpu().numpy()
                array_payload[
                    f"{regime}__{arm}__shuffled_kernel_cosine"
                ] = geometry["shuffled_kernel_cosine"].cpu().numpy()
                array_payload[
                    f"{regime}__{arm}__projector_distance"
                ] = geometry["projector_distance"].cpu().numpy()

        geometry_pass = all(cell["geometry_pass"] for cell in geometry_cells)
        regime_causal: dict[str, Any] = {}
        causal_transport_pass = True
        for regime in REGIMES:
            selected = [cell for cell in causal_cells if cell["regime"] == regime]
            transported_all = all(
                cell["states"]["transported_tangent"]["continuous"][
                    "continuous_pass"
                ]
                for cell in selected
            )
            shuffled_failure = any(
                not cell["states"]["shuffled_tangent"]["continuous"][
                    "continuous_pass"
                ]
                for cell in selected
            )
            random_failure = any(
                not cell["states"]["random_tangent"]["continuous"]["continuous_pass"]
                for cell in selected
            )
            means = {
                state: sum(
                    cell["states"][state]["continuous"]["mean_moment_shift_bins"]
                    for cell in selected
                )
                / len(selected)
                for state in (
                    "predicted",
                    "local_tangent",
                    "transported_tangent",
                    "shuffled_tangent",
                    "random_tangent",
                )
            }
            shuffled_margin = means["shuffled_tangent"] - means["transported_tangent"]
            random_margin = means["random_tangent"] - means["transported_tangent"]
            regime_pass = bool(
                transported_all
                and shuffled_failure
                and random_failure
                and shuffled_margin >= config.causal_control_margin_bins
                and random_margin >= config.causal_control_margin_bins
            )
            regime_causal[regime] = {
                "transported_all_cells_pass": transported_all,
                "shuffled_any_failure": shuffled_failure,
                "random_any_failure": random_failure,
                "aggregate_mean_shift_bins": means,
                "shuffled_margin_bins": shuffled_margin,
                "random_margin_bins": random_margin,
                "causal_transport_pass": regime_pass,
            }
            causal_transport_pass = bool(causal_transport_pass and regime_pass)

        valid = bool(numerical_valid and full_control_pass)
        classification = classify_checkpoint(
            valid=valid,
            local_tangent_pass=local_tangent_pass,
            geometry_pass=geometry_pass,
            causal_transport_pass=bool(causal_transport_pass and local_tangent_pass),
        )
        _write_npz(arrays_path, array_payload)
        arrays_sha256 = _sha256(arrays_path)
        result = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_id": f"tinyllm-c2-local-metric-field-transport-seed{seed}",
            "status": "completed",
            "evidence_role": _evidence_role(config),
            "completed_at": _utc_now(),
            "seed": seed,
            "configuration": asdict(config),
            "scientific_fingerprint": fingerprint,
            "implementation_sha256": implementation,
            "provenance": {
                "writer_campaign": str(writer_path),
                "writer_campaign_sha256": _sha256(writer_path),
                "writer_result": str(prior_path),
                "writer_result_sha256": _sha256(prior_path),
                "corrective_campaign": str(corrective_path),
                "corrective_campaign_sha256": _sha256(corrective_path),
                "checkpoint": provenance["checkpoint"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "frontend_checkpoint": provenance["frontend_checkpoint"],
                "frontend_checkpoint_sha256": provenance[
                    "frontend_checkpoint_sha256"
                ],
                "character_result": str(frozen_path),
                "character_result_sha256": _sha256(frozen_path),
                "readout": readout_predecessor,
            },
            "basis": basis_summary,
            "writer_mapping_sha256": _tensor_digest(
                mapping["linear"], mapping["intercept"]
            ),
            "fresh_inputs": {
                regime: paired[regime][1] for regime in REGIMES
            },
            "geometry_cells": geometry_cells,
            "causal_cells": causal_cells,
            "causal_by_regime": regime_causal,
            "gates": {
                "input_and_pair_contract": all(
                    cell["input"]["clipping_fraction"]
                    <= config.clipping_fraction_ceiling
                    and cell["input"]["nonfinite_count"] == 0
                    and cell["input"]["changed_token_fraction"]
                    >= config.changed_token_fraction_floor
                    for cell in geometry_cells
                )
                and all(paired[regime][1]["pair_contract"] for regime in REGIMES),
                "numerical_contract": numerical_valid,
                "full_control_pass": full_control_pass,
                "local_tangent_fresh_cohort_pass": local_tangent_pass,
                "geometric_transport_pass": geometry_pass,
                "causal_transport_pass": causal_transport_pass,
                "primary_checkpoint_pass": bool(
                    classification == "invariant_metric_field_transport_supported"
                ),
            },
            "classification": classification,
            "primary_metric": float(
                classification == "invariant_metric_field_transport_supported"
            ),
            "analysis_seconds": time.perf_counter() - seed_started,
            "artifacts": {
                "result": str(result_path),
                "arrays": str(arrays_path),
                "arrays_sha256": arrays_sha256,
            },
        }
        _write_json(result_path, result)
        results.append(result)
        print(f"seed {seed}: {classification}", flush=True)
        if _implementation_digest() != implementation:
            raise RuntimeError("analysis implementation changed during campaign")

    classifications = [result["classification"] for result in results]
    support_count = classifications.count("invariant_metric_field_transport_supported")
    if any(name == "invalid" for name in classifications):
        conclusion = "invalid_campaign"
    elif len(config.seeds) == 3 and support_count == 3:
        conclusion = "supported_invariant_metric_field_transport_three_of_three"
    elif len(set(classifications)) == 1:
        conclusion = classifications[0]
    else:
        conclusion = "checkpoint_stratified_metric_field_transport"
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "fresh_cohort_seeds": FRESH_COHORT_SEEDS,
        "implementation_sha256": implementation,
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
        "provenance": {
            "writer_campaign": str(writer_path),
            "writer_campaign_sha256": _sha256(writer_path),
            "writer_implementation_sha256": writer_campaign["implementation_sha256"],
            "corrective_campaign": str(corrective_path),
            "corrective_campaign_sha256": _sha256(corrective_path),
            "corrective_implementation_sha256": corrective["implementation_sha256"],
        },
        "summary": {
            "requested": len(config.seeds),
            "scheduled": len(config.seeds) - reused,
            "completed": len(results),
            "failed": 0,
            "excluded": 0,
            "retries": 0,
            "reused": reused,
            "trained_models": 0,
            "fitted_predictive_observers": 0,
            "fitted_writers": 0,
            "fresh_group_cells": len(config.seeds) * len(REGIMES) * len(GROUP_ARMS),
            "fresh_jacobian_fields": len(config.seeds)
            * len(REGIMES)
            * (len(GROUP_ARMS) + 1),
        },
        "aggregates": {
            "conclusion": conclusion,
            "primary_checkpoint_pass_count": support_count,
            "required_checkpoint_pass_count": 3,
            "classification_counts": {
                name: classifications.count(name) for name in sorted(set(classifications))
            },
            "classification_by_seed": {
                str(result["seed"]): result["classification"] for result in results
            },
            "gate_counts": {
                name: sum(bool(result["gates"][name]) for result in results)
                for name in (
                    "input_and_pair_contract",
                    "numerical_contract",
                    "full_control_pass",
                    "local_tangent_fresh_cohort_pass",
                    "geometric_transport_pass",
                    "causal_transport_pass",
                    "primary_checkpoint_pass",
                )
            },
        },
        "results": [
            {
                "experiment_id": result["experiment_id"],
                "seed": result["seed"],
                "scientific_fingerprint": result["scientific_fingerprint"],
                "classification": result["classification"],
                "gates": result["gates"],
                "path": result["artifacts"]["result"],
                "result_sha256": _sha256(Path(result["artifacts"]["result"])),
                "arrays": result["artifacts"]["arrays"],
                "arrays_sha256": _sha256(Path(result["artifacts"]["arrays"])),
            }
            for result in results
        ],
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "The observed similarity group covers scale, planar orientation, and constant offset only.",
            "Tokenization approximates the exact continuous sensor action and is bounded by explicit input contracts.",
            "Exact residual amplitudes are diagnostic and unavailable at inference time.",
            "Carrier bases, writer, readout, and checkpoints were selected in prior work.",
            "Three selected checkpoints do not establish population prevalence.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "seed_*" / "result.json"),
            "arrays": str(output / "runs" / "seed_*" / "metric_field_arrays.npz"),
        },
    }
    _write_json(campaign_path, campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_local_metric_field_transport/"
            "20260807_d6_fresh_cohort"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=_ints, default=fixed.PRIMARY_SEEDS)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = LocalMetricFieldTransportConfig(
        seeds=args.seeds,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
