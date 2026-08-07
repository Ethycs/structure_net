#!/usr/bin/env python3
"""Causally test nuisance-fiber transport of TinyLLM's local task tangent."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
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

from experiments.structure_net import tinyllm_local_continuation_tangent_kernel as local
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-c2-nuisance-fiber-task-metric-transport.v1"
HYPOTHESIS_ID = "tinyllm-c2-nuisance-fiber-task-metric-transport-v1"
WRITER_CAMPAIGN_SHA256 = (
    "7ab6079d56bc8dc78cc5e1f4011c581b8f507bcd8ec20573bbf8b11227df8d1b"
)
LOCAL_METHOD_CAMPAIGN_SHA256 = (
    "8b2f25dc61d1ffb93ba9dc66f1a85f36c07bda5a43ed3ed34a9627ca1485c12a"
)
FRESH_COHORT_SEEDS = {
    "fresh_1": {
        "composition": {"quotient": 910101, "source": 910111, "target": 910121},
        "extrapolation": {"quotient": 910102, "source": 910112, "target": 910122},
    },
    "fresh_2": {
        "composition": {"quotient": 920101, "source": 920111, "target": 920121},
        "extrapolation": {"quotient": 920102, "source": 920112, "target": 920122},
    },
}
REGIMES = local.REGIMES
PRIMARY_SEEDS = local.fixed.PRIMARY_SEEDS


@dataclass(frozen=True)
class NuisanceFiberMetricTransportConfig:
    writer_root: str = (
        "data/experiments/tinyllm_frozen_writer_capacity/"
        "20260807_d6_preregistered_diagnostic"
    )
    local_method_root: str = (
        "data/experiments/tinyllm_local_continuation_tangent_kernel/"
        "20260807_d6_corrective_v2"
    )
    seeds: tuple[int, ...] = PRIMARY_SEEDS
    orbit_count: int = 64
    random_controls: int = 8
    fourier_order: int = 4
    jacobian_rtol: float = 1e-6
    jacobian_atol: float = 1e-10
    decomposition_tolerance: float = 1e-6
    exact_action_tolerance: float = 1e-4
    kernel_median_cosine_floor: float = 0.90
    kernel_p10_cosine_floor: float = 0.75
    transported_local_margin_bins: float = 0.05
    specificity_margin_bins: float = 0.125
    kernel_mean_movement_ceiling_bins: float = 0.05
    kernel_p95_movement_ceiling_bins: float = 0.20
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("primary diagnostic seeds are fixed to 7,29,53")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.orbit_count != 64 and not self.allow_underpowered:
            raise ValueError("primary diagnostic fixes 64 exact orbits per cell")
        if self.orbit_count < 8:
            raise ValueError("at least eight exact orbits are required")
        if self.random_controls != 8 and not self.allow_underpowered:
            raise ValueError("primary diagnostic fixes eight random controls")
        if self.random_controls < 2:
            raise ValueError("at least two random controls are required")
        if self.fourier_order != 4:
            raise ValueError("the predecessor writer is fixed to Fourier order four")
        positive = (
            self.jacobian_rtol,
            self.jacobian_atol,
            self.decomposition_tolerance,
            self.exact_action_tolerance,
            self.kernel_median_cosine_floor,
            self.kernel_p10_cosine_floor,
            self.transported_local_margin_bins,
            self.specificity_margin_bins,
            self.kernel_mean_movement_ceiling_bins,
            self.kernel_p95_movement_ceiling_bins,
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
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
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
        Path(local.predecessor.__file__),
        Path(local.fixed.__file__),
        Path(local.transport.__file__),
        Path(local.transport.rank.__file__),
        Path(local.readout.__file__),
        Path(local.coupling.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evidence_role(config: NuisanceFiberMetricTransportConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_post_outcome_underpowered_mechanistic_diagnostic"
    )


def _local_config(
    config: NuisanceFiberMetricTransportConfig,
) -> local.LocalContinuationConfig:
    return local.LocalContinuationConfig(
        predecessor_root=config.writer_root,
        seeds=config.seeds,
        orbit_count=config.orbit_count,
        random_controls=config.random_controls,
        fourier_order=config.fourier_order,
        jacobian_rtol=config.jacobian_rtol,
        jacobian_atol=config.jacobian_atol,
        decomposition_tolerance=config.decomposition_tolerance,
        kernel_mean_movement_ceiling_bins=config.kernel_mean_movement_ceiling_bins,
        kernel_p95_movement_ceiling_bins=config.kernel_p95_movement_ceiling_bins,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
        post_outcome_corrective_replication=not config.allow_underpowered,
    )


def _load_local_method_contract(
    config: NuisanceFiberMetricTransportConfig,
) -> tuple[dict[str, Any], Path]:
    path = Path(config.local_method_root) / "campaign_results.json"
    value = json.loads(path.read_text())
    if (
        _sha256(path) != LOCAL_METHOD_CAMPAIGN_SHA256
        or value.get("schema_version") != local.SCHEMA_VERSION
        or value.get("status") != "completed"
        or int(value.get("summary", {}).get("completed", -1)) != 3
    ):
        raise ValueError(f"invalid local-method campaign {path}")
    return value, path


def fresh_datasets(
    task: CircleTaskConfig,
    orbit_count: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for cohort, regimes in FRESH_COHORT_SEEDS.items():
        result[cohort] = {}
        for regime, seeds in regimes.items():
            theta = np.random.default_rng(seeds["quotient"]).uniform(
                0.0, 2.0 * math.pi, orbit_count
            )
            source = local.transport.rank.deck.generate_exact_orbits(
                task,
                k=2,
                orbit_count=orbit_count,
                seed=seeds["source"],
                regime=regime,
                quotient_phases=theta,
            )
            target = local.transport.rank.deck.generate_exact_orbits(
                task,
                k=2,
                orbit_count=orbit_count,
                seed=seeds["target"],
                regime=regime,
                quotient_phases=theta,
            )
            if not torch.equal(source.quotient_phase, target.quotient_phase):
                raise AssertionError("phase-matched nuisance pair construction failed")
            result[cohort][regime] = {"source": source, "target": target}
    return result


def apply_calibration_action(dataset: Any, task: CircleTaskConfig) -> Any:
    """Apply the locked nonidentity Gcal action while preserving the carrier."""
    angle = 0.71
    scale = 1.6
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=dataset.sensor.dtype,
    )
    extra_offset = torch.tensor(
        [0.31, -0.17], dtype=dataset.sensor.dtype
    )
    extra_drift = torch.tensor(
        [-0.05, 0.08], dtype=dataset.sensor.dtype
    )
    history = torch.arange(task.sensor_steps, dtype=dataset.sensor.dtype)
    history = history / max(1.0, float(task.sensor_steps - 1)) - 1.0
    sensor = dataset.sensor.clone()
    sensor[..., :2] = (
        scale * torch.einsum("ij,btj->bti", rotation, sensor[..., :2])
        + extra_offset[None, None, :]
        + extra_drift[None, None, :] * history[None, :, None]
    )
    calibration = dataset.calibration.clone()
    calibration[:, :2] = torch.einsum(
        "ij,bj->bi", rotation, dataset.calibration[:, :2]
    )
    calibration[:, 3:4] = scale * dataset.calibration[:, 3:4]
    calibration[:, 4:6] = (
        scale
        * torch.einsum("ij,bj->bi", rotation, dataset.calibration[:, 4:6])
        + extra_offset[None, :]
    )
    calibration[:, 6:8] = (
        scale
        * torch.einsum("ij,bj->bi", rotation, dataset.calibration[:, 6:8])
        + extra_drift[None, :]
    )
    return replace(dataset, sensor=sensor, calibration=calibration)


def kernel_line_cosines(
    source_jacobian: torch.Tensor, target_jacobian: torch.Tensor
) -> torch.Tensor:
    source_kernel = torch.linalg.svd(source_jacobian.double()).Vh[:, -1, :]
    target_kernel = torch.linalg.svd(target_jacobian.double()).Vh[:, -1, :]
    return torch.einsum("ni,ni->n", source_kernel, target_kernel).abs()


def projectors_from_jacobian(
    jacobian: torch.Tensor, *, rtol: float, atol: float
) -> tuple[torch.Tensor, torch.Tensor]:
    pseudoinverse = torch.linalg.pinv(jacobian.double(), rtol=rtol, atol=atol)
    projector = pseudoinverse @ jacobian.double()
    rank = torch.linalg.matrix_rank(jacobian.double(), rtol=rtol, atol=atol)
    return projector, rank


def _norm_match(vector: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    target = torch.linalg.vector_norm(reference, dim=-1, keepdim=True)
    fallback = torch.zeros_like(vector)
    fallback[..., 0] = 1.0
    direction = torch.where(norm > 1e-12, vector / norm.clamp_min(1e-12), fallback)
    return direction * target


def random_projector_tangents(
    residual: torch.Tensor,
    reference: torch.Tensor,
    count: int,
    seed: int,
) -> torch.Tensor:
    """Return deterministic rank-two-projector directions, norm matched."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    normals = torch.randn(
        count, len(residual), 3, generator=generator, dtype=torch.float64
    ).to(residual.device)
    normals = normals / torch.linalg.vector_norm(
        normals, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    random = residual.double().unsqueeze(0) - (
        residual.double().unsqueeze(0) * normals
    ).sum(-1, keepdim=True) * normals
    return _norm_match(random, reference.double().unsqueeze(0))


def _control_seed(seed: int, cohort: str, regime: str, family: str) -> int:
    payload = f"{HYPOTHESIS_ID}:{seed}:{cohort}:{regime}:{family}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _aggregate_mean(cells: Sequence[Mapping[str, Any]], state: str) -> float:
    return float(
        np.mean(
            [cell["states"][state]["continuous"]["mean_moment_shift_bins"] for cell in cells]
        )
    )


def checkpoint_gates(
    cells: Sequence[Mapping[str, Any]],
    action_contract_passed: bool,
    config: NuisanceFiberMetricTransportConfig,
) -> dict[str, Any]:
    numerical_target = all(
        cell["numerical"]["source_rank_two"]
        and cell["numerical"]["target_rank_two"]
        and cell["numerical"]["jacobians_finite"]
        and cell["numerical"]["maximum_decomposition_error"]
        <= config.decomposition_tolerance
        and not cell["states"]["zero"]["continuous"]["continuous_pass"]
        and not cell["states"]["predicted"]["continuous"]["continuous_pass"]
        and cell["states"]["exact"]["continuous"]["continuous_pass"]
        and cell["states"]["full"]["continuous"]["continuous_pass"]
        and cell["states"]["local_tangent"]["continuous"]["continuous_pass"]
        and cell["states"]["local_kernel"]["movement_from_predicted"]["mean_bins"]
        <= config.kernel_mean_movement_ceiling_bins
        and cell["states"]["local_kernel"]["movement_from_predicted"]["p95_bins"]
        <= config.kernel_p95_movement_ceiling_bins
        for cell in cells
    )
    geometric = all(
        cell["geometry"]["median_kernel_line_absolute_cosine"]
        >= config.kernel_median_cosine_floor
        and cell["geometry"]["p10_kernel_line_absolute_cosine"]
        >= config.kernel_p10_cosine_floor
        for cell in cells
    )
    transported_mean = _aggregate_mean(cells, "transported_tangent")
    local_mean = _aggregate_mean(cells, "local_tangent")
    causal = bool(
        all(
            cell["states"]["transported_tangent"]["continuous"]["continuous_pass"]
            for cell in cells
        )
        and transported_mean <= local_mean + config.transported_local_margin_bins
    )
    shuffled_mean = _aggregate_mean(cells, "shuffled_tangent")
    shuffled_checkpoint_pass = all(
        cell["states"]["shuffled_tangent"]["continuous"]["continuous_pass"]
        for cell in cells
    )
    random_means = [
        _aggregate_mean(cells, f"random_tangent_{index:02d}")
        for index in range(config.random_controls)
    ]
    random_checkpoint_passes = [
        all(
            cell["states"][f"random_tangent_{index:02d}"]["continuous"][
                "continuous_pass"
            ]
            for cell in cells
        )
        for index in range(config.random_controls)
    ]
    specificity = bool(
        not shuffled_checkpoint_pass
        and transported_mean + config.specificity_margin_bins <= shuffled_mean
        and sum(random_checkpoint_passes) <= 1
        and transported_mean + config.specificity_margin_bins
        <= float(np.median(random_means))
    )
    return {
        "exact_calibration_action_contract": action_contract_passed,
        "numerical_target_control_contract": numerical_target,
        "geometric_transport": geometric,
        "causal_transport": causal,
        "specificity": specificity,
        "local_tangent_aggregate_mean_shift_bins": local_mean,
        "transported_tangent_aggregate_mean_shift_bins": transported_mean,
        "shuffled_tangent_aggregate_mean_shift_bins": shuffled_mean,
        "random_tangent_aggregate_mean_shift_bins": random_means,
        "random_checkpoint_passes": random_checkpoint_passes,
        "shuffled_checkpoint_pass": shuffled_checkpoint_pass,
    }


def classify_checkpoint(gates: Mapping[str, Any]) -> str:
    if not gates["exact_calibration_action_contract"]:
        return "invalid_exact_action_contract"
    if not gates["numerical_target_control_contract"]:
        return "invalid_numerical_or_target_controls"
    if all(
        gates[name]
        for name in ("geometric_transport", "causal_transport", "specificity")
    ):
        return "nuisance_fiber_task_metric_transport_confirmed"
    if gates["causal_transport"] and not gates["geometric_transport"]:
        return "causal_equivalence_without_projector_equality"
    if gates["geometric_transport"] and not gates["causal_transport"]:
        return "projector_geometry_without_causal_transport"
    return "nuisance_contextual_local_task_metric"


def _campaign_is_reusable(
    value: Mapping[str, Any],
    config: NuisanceFiberMetricTransportConfig,
    implementation: str,
) -> bool:
    return bool(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("evidence_role") == _evidence_role(config)
        and value.get("implementation_sha256") == implementation
        and value.get("configuration") == _json_compatible(asdict(config))
        and int(value.get("summary", {}).get("completed", -1)) == len(config.seeds)
        and len(value.get("results", [])) == len(config.seeds)
        and all(
            Path(item.get("path", "")).is_file()
            and _sha256(Path(item["path"])) == item.get("result_sha256")
            and Path(item.get("arrays", "")).is_file()
            and _sha256(Path(item["arrays"])) == item.get("arrays_sha256")
            for item in value.get("results", [])
        )
    )


def run_campaign(
    config: NuisanceFiberMetricTransportConfig, output: Path
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
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
    local_config = _local_config(config)
    writer_campaign, writer_path, writer_details = local._load_predecessor(local_config)
    if _sha256(writer_path) != WRITER_CAMPAIGN_SHA256:
        raise ValueError("writer predecessor campaign hash changed")
    method_campaign, method_path = _load_local_method_contract(config)
    base_config = local.fixed.FixedSemanticGaugeWriterConfig(
        seeds=config.seeds,
        orbit_count=config.orbit_count,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )
    transport_config = local.fixed._transport_config(base_config)
    rank_config = local.transport._rank_config(transport_config)
    bridge = local.transport.rank._bridge_config(rank_config)
    datasets = fresh_datasets(task, config.orbit_count)

    results: list[dict[str, Any]] = []
    for seed in config.seeds:
        seed_started = time.perf_counter()
        prior, prior_path = writer_details[seed]
        system, provenance = local.transport.rank.deck.load_source(
            task, bridge, 2, seed, device
        )
        system.model.eval()
        for parameter in system.model.parameters():
            parameter.requires_grad_(False)
        if provenance["checkpoint_sha256"] != prior["provenance"]["checkpoint_sha256"]:
            raise ValueError(f"checkpoint mismatch for seed {seed}")
        frozen, frozen_path = local.transport.rank._load_character_source(
            rank_config, seed, provenance["checkpoint_sha256"]
        )
        basis, basis_summary = local.transport._fit_seed_basis(
            system,
            task,
            transport_config,
            rank_config,
            bridge,
            frozen,
            seed,
            device,
        )
        readout_predecessor = local.transport._load_readout_predecessor(
            transport_config, seed, provenance["checkpoint_sha256"]
        )
        mapping_record = prior["alignment_fit"]["mappings"]["fourier_m04"]
        mapping = {
            "linear": torch.tensor(mapping_record["linear"], dtype=torch.float64, device=device),
            "intercept": torch.tensor(
                mapping_record["intercept"], dtype=torch.float64, device=device
            ),
        }
        cells: list[dict[str, Any]] = []
        arrays: dict[str, np.ndarray] = {}
        maximum_action_error = 0.0
        action_by_cell: list[dict[str, Any]] = []
        for cohort in FRESH_COHORT_SEEDS:
            for regime in REGIMES:
                pair = datasets[cohort][regime]
                # The shared extractor uses the cohort label only to attach the
                # predecessor's evaluation-seed metadata. Reuse its two valid
                # slots, then retain the fresh label/seeds in this experiment.
                extractor_cohort = {
                    "fresh_1": "heldout_a",
                    "fresh_2": "heldout_b",
                }[cohort]
                source_cell = local.transport._extract_cell(
                    system,
                    task,
                    transport_config,
                    bridge,
                    pair["source"],
                    extractor_cohort,
                    regime,
                    device,
                )
                target_cell = local.transport._extract_cell(
                    system,
                    task,
                    transport_config,
                    bridge,
                    pair["target"],
                    extractor_cohort,
                    regime,
                    device,
                )
                action_dataset = apply_calibration_action(pair["target"], task)
                action_cell = local.transport._extract_cell(
                    system,
                    task,
                    transport_config,
                    bridge,
                    action_dataset,
                    extractor_cohort,
                    regime,
                    device,
                )
                theta = (
                    pair["target"].quotient_phase.to(device)
                    .reshape(config.orbit_count, 2)[:, 0]
                    .double()
                )
                features = local.predecessor.fourier_features(theta, config.fourier_order)
                predicted = local.transport.apply_affine(features, mapping)
                target = local.transport._coordinates(target_cell, basis)
                residual = target - predicted
                _, source_jacobian = local.continuation_moment_jacobian(
                    system,
                    source_cell["target_cut"],
                    source_cell["propagated"],
                    predicted,
                    basis,
                    task,
                    rank_config.continuation_batch_size,
                )
                _, target_jacobian = local.continuation_moment_jacobian(
                    system,
                    target_cell["target_cut"],
                    target_cell["propagated"],
                    predicted,
                    basis,
                    task,
                    rank_config.continuation_batch_size,
                )
                _, action_jacobian = local.continuation_moment_jacobian(
                    system,
                    action_cell["target_cut"],
                    action_cell["propagated"],
                    predicted,
                    basis,
                    task,
                    rank_config.continuation_batch_size,
                )
                source_projector, source_rank = projectors_from_jacobian(
                    source_jacobian, rtol=config.jacobian_rtol, atol=config.jacobian_atol
                )
                target_projector, target_rank = projectors_from_jacobian(
                    target_jacobian, rtol=config.jacobian_rtol, atol=config.jacobian_atol
                )
                action_projector, _ = projectors_from_jacobian(
                    action_jacobian, rtol=config.jacobian_rtol, atol=config.jacobian_atol
                )
                decomposition = local.tangent_kernel_decomposition(
                    target_jacobian,
                    residual,
                    rtol=config.jacobian_rtol,
                    atol=config.jacobian_atol,
                )
                transported = torch.einsum("nij,nj->ni", source_projector, residual)
                generator = torch.Generator(device="cpu").manual_seed(
                    _control_seed(seed, cohort, regime, "shuffle")
                )
                permutation = torch.randperm(config.orbit_count, generator=generator).to(device)
                shuffled = torch.einsum(
                    "nij,nj->ni", source_projector[permutation], residual
                )
                shuffled = _norm_match(shuffled, transported)
                random = random_projector_tangents(
                    residual,
                    transported,
                    config.random_controls,
                    _control_seed(seed, cohort, regime, "random"),
                )
                coordinate_states: dict[str, torch.Tensor] = {
                    "predicted": predicted,
                    "full": target,
                    "local_tangent": predicted + decomposition["tangent"],
                    "transported_tangent": predicted + transported,
                    "local_kernel": predicted + decomposition["kernel"],
                    "shuffled_tangent": predicted + shuffled,
                }
                for index in range(config.random_controls):
                    coordinate_states[f"random_tangent_{index:02d}"] = predicted + random[index]
                states = local.evaluate_states(
                    system,
                    task,
                    local_config,
                    target_cell,
                    basis,
                    coordinate_states,
                    readout_predecessor["rotation_bins"],
                )
                cosines = kernel_line_cosines(source_jacobian, target_jacobian)
                carrier = local.fixed.ladder.AnalyticPhaseCarrier(task)
                target_carrier = carrier(
                    pair["target"].sensor.to(device), pair["target"].calibration.to(device)
                )
                action_carrier = carrier(
                    action_dataset.sensor.to(device), action_dataset.calibration.to(device)
                )
                action_errors = {
                    "analytic_carrier_maximum_absolute_error": float(
                        (target_carrier - action_carrier).abs().max()
                    ),
                    "propagated_state_maximum_absolute_error": float(
                        (target_cell["propagated"] - action_cell["propagated"]).abs().max()
                    ),
                    "predicted_coordinate_maximum_absolute_error": 0.0,
                    "jacobian_maximum_absolute_error": float(
                        (target_jacobian - action_jacobian).abs().max()
                    ),
                    "projector_maximum_absolute_error": float(
                        (target_projector - action_projector).abs().max()
                    ),
                }
                action_errors["maximum_absolute_error"] = max(action_errors.values())
                maximum_action_error = max(
                    maximum_action_error, action_errors["maximum_absolute_error"]
                )
                action_by_cell.append(
                    {"cohort": cohort, "regime": regime, **action_errors}
                )
                maximum_decomposition = max(
                    float(decomposition["relative_decomposition_error"].max()),
                    float(decomposition["relative_tangent_mismatch"].max()),
                    float(decomposition["relative_kernel_leakage"].max()),
                )
                cells.append(
                    {
                        "cohort": cohort,
                        "regime": regime,
                        "seeds": FRESH_COHORT_SEEDS[cohort][regime],
                        "geometry": {
                            "mean_kernel_line_absolute_cosine": float(cosines.mean()),
                            "median_kernel_line_absolute_cosine": float(cosines.median()),
                            "p10_kernel_line_absolute_cosine": float(
                                torch.quantile(cosines, 0.10)
                            ),
                            "mean_projector_frobenius_distance": float(
                                torch.linalg.matrix_norm(
                                    source_projector - target_projector, dim=(-2, -1)
                                ).mean()
                            ),
                        },
                        "numerical": {
                            "source_rank_two": bool(torch.all(source_rank == 2)),
                            "target_rank_two": bool(torch.all(target_rank == 2)),
                            "jacobians_finite": bool(
                                torch.isfinite(source_jacobian).all()
                                and torch.isfinite(target_jacobian).all()
                            ),
                            "maximum_decomposition_error": maximum_decomposition,
                        },
                        "action_contract": action_errors,
                        "states": states,
                    }
                )
                prefix = f"{cohort}__{regime}"
                arrays.update(
                    {
                        f"{prefix}__source_jacobian": source_jacobian.cpu().numpy(),
                        f"{prefix}__target_jacobian": target_jacobian.cpu().numpy(),
                        f"{prefix}__source_projector": source_projector.cpu().numpy(),
                        f"{prefix}__target_projector": target_projector.cpu().numpy(),
                        f"{prefix}__target_residual": residual.cpu().numpy(),
                        f"{prefix}__local_tangent": decomposition["tangent"].cpu().numpy(),
                        f"{prefix}__transported_tangent": transported.cpu().numpy(),
                        f"{prefix}__kernel_line_cosine": cosines.cpu().numpy(),
                    }
                )

        action_passed = maximum_action_error <= config.exact_action_tolerance
        gates = checkpoint_gates(cells, action_passed, config)
        classification = classify_checkpoint(gates)
        run_dir = output / "runs" / f"seed_{seed}"
        arrays_path = run_dir / "transport_arrays.npz"
        _write_npz(arrays_path, arrays)
        result_path = run_dir / "result.json"
        fingerprint_payload = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "configuration": asdict(config),
            "fresh_cohort_seeds": FRESH_COHORT_SEEDS,
            "seed": seed,
            "writer_campaign_sha256": WRITER_CAMPAIGN_SHA256,
            "local_method_campaign_sha256": LOCAL_METHOD_CAMPAIGN_SHA256,
            "predecessor_result_sha256": _sha256(prior_path),
            "checkpoint_sha256": provenance["checkpoint_sha256"],
        }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        result = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_id": f"tinyllm-c2-nuisance-fiber-task-metric-seed{seed}",
            "status": "completed",
            "evidence_role": _evidence_role(config),
            "completed_at": _utc_now(),
            "seed": seed,
            "configuration": asdict(config),
            "fresh_cohort_seeds": FRESH_COHORT_SEEDS,
            "scientific_fingerprint": fingerprint,
            "implementation_sha256": implementation,
            "provenance": {
                "writer_campaign": str(writer_path),
                "writer_campaign_sha256": _sha256(writer_path),
                "writer_result": str(prior_path),
                "writer_result_sha256": _sha256(prior_path),
                "local_method_campaign": str(method_path),
                "local_method_campaign_sha256": _sha256(method_path),
                "checkpoint": provenance["checkpoint"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "character_result": str(frozen_path),
                "character_result_sha256": _sha256(frozen_path),
                "readout_result": readout_predecessor["result"],
                "readout_result_sha256": readout_predecessor["result_sha256"],
            },
            "basis": basis_summary,
            "exact_action_contract": {
                "group": "(R>0 x SO(2)) semidirect (R2_offset x R2_drift)",
                "representation": "rho(g)=I3 on the neutral C2 carrier",
                "locked_action": {
                    "rotation_radians": 0.71,
                    "scale": 1.6,
                    "offset": [0.31, -0.17],
                    "drift": [-0.05, 0.08],
                },
                "maximum_absolute_error": maximum_action_error,
                "by_cell": action_by_cell,
                "passed": action_passed,
            },
            "heldout_cells": cells,
            "gates": gates,
            "classification": classification,
            "primary_metric": float(
                classification == "nuisance_fiber_task_metric_transport_confirmed"
            ),
            "analysis_seconds": time.perf_counter() - seed_started,
            "artifacts": {
                "result": str(result_path),
                "arrays": str(arrays_path),
                "arrays_sha256": _sha256(arrays_path),
            },
            "method_boundaries": [
                "The exact calibration action is a positive contract, not the primary nuisance test.",
                "The broader N3 nuisance relation is an observation groupoid, not one claimed finite-dimensional group.",
                "Held-out exact residuals define diagnostic causal patches and are not deployable inputs.",
                "Three selected checkpoints do not establish population prevalence.",
            ],
        }
        _write_json(result_path, result)
        results.append(result)
        print(f"seed {seed}: {classification}", flush=True)
        if _implementation_digest() != implementation:
            raise RuntimeError("implementation changed during campaign")

    gate_names = (
        "exact_calibration_action_contract",
        "numerical_target_control_contract",
        "geometric_transport",
        "causal_transport",
        "specificity",
    )
    gate_counts = {
        name: sum(bool(result["gates"][name]) for result in results)
        for name in gate_names
    }
    confirmed = bool(
        not config.allow_underpowered
        and len(results) == 3
        and all(gate_counts[name] == 3 for name in gate_names)
    )
    classifications = [result["classification"] for result in results]
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
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        },
        "provenance": {
            "writer_campaign": str(writer_path),
            "writer_campaign_sha256": _sha256(writer_path),
            "writer_implementation_sha256": writer_campaign["implementation_sha256"],
            "local_method_campaign": str(method_path),
            "local_method_campaign_sha256": _sha256(method_path),
            "local_method_implementation_sha256": method_campaign["implementation_sha256"],
        },
        "summary": {
            "requested": len(config.seeds),
            "completed": len(results),
            "failed": 0,
            "excluded": 0,
            "trained_models": 0,
            "fitted_writers": 0,
            "fitted_predictive_observers": 0,
            "fresh_matched_cells": len(results) * len(FRESH_COHORT_SEEDS) * len(REGIMES),
        },
        "aggregates": {
            "confirmed": confirmed,
            "conclusion": (
                "systems_lifecycle_only_not_quality_evidence"
                if config.allow_underpowered
                else (
                    "confirmed_nuisance_fiber_task_metric_transport"
                    if confirmed
                    else "not_confirmed_nuisance_fiber_task_metric_transport"
                )
            ),
            "gate_counts": gate_counts,
            "classification_by_seed": {
                str(result["seed"]): result["classification"] for result in results
            },
            "classification_counts": {
                name: classifications.count(name) for name in sorted(set(classifications))
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
                "arrays_sha256": result["artifacts"]["arrays_sha256"],
            }
            for result in results
        ],
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "This is a preregistered post-outcome diagnostic on three selected frozen checkpoints.",
            "No group action or coordinate alignment is fitted.",
            "The exact subgroup contract and broader nuisance-groupoid test are reported separately.",
            "Systems-only shakedowns are never pooled.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "seed_*" / "result.json"),
            "arrays": str(output / "runs" / "seed_*" / "transport_arrays.npz"),
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
            "data/experiments/tinyllm_nuisance_fiber_task_metric_transport/"
            "20260807_d6_preregistered_diagnostic"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=_ints, default=PRIMARY_SEEDS)
    parser.add_argument("--orbit-count", type=int, default=64)
    parser.add_argument("--random-controls", type=int, default=8)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = NuisanceFiberMetricTransportConfig(
        seeds=args.seeds,
        orbit_count=args.orbit_count,
        random_controls=args.random_controls,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
