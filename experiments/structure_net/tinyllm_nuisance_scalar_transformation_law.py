#!/usr/bin/env python3
"""Test whether TinyLLM's signed correction scalar is nuisance invariant."""

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

import experiments.structure_net.tinyllm_local_metric_field_transport as group
import experiments.structure_net.tinyllm_local_task_tangent as local
import experiments.structure_net.tinyllm_source_task_covector_portability as source
import experiments.structure_net.tinyllm_fixed_semantic_gauge_writer as fixed
import experiments.structure_net.tinyllm_cross_seed_causal_carrier_transport as transport
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-c2-nuisance-scalar-transformation-law.v1"
HYPOTHESIS_ID = "tinyllm-c2-nuisance-scalar-transformation-law-v1"
GROUP_CAMPAIGN_SHA256 = (
    "2aa80cff5882cb79754d732e9012c810c2624934bc5be9e03c9e77de4f525f55"
)
GROUP_IMPLEMENTATION_SHA256 = (
    "e3a05118971bba51e50e07e0fc426a8b321f0ecf823ded7dfec6b3539b630bfd"
)
GROUP_ARMS = group.GROUP_ARMS
REGIMES = group.REGIMES


@dataclass(frozen=True)
class NuisanceScalarTransformationConfig:
    group_root: str = (
        "data/experiments/tinyllm_local_metric_field_transport/"
        "20260807_d6_fresh_cohort"
    )
    seeds: tuple[int, ...] = fixed.PRIMARY_SEEDS
    orbit_count: int = 64
    phase_count: int = 16
    nuisance_replicates: int = 4
    carrier_rank: int = 3
    writer_order: int = 4
    fine_step_std: float = 0.025
    coarse_step_std: float = 0.05
    coordinate_scale_floor: float = 1e-8
    derivative_cosine_floor: float = 0.98
    derivative_relative_l2_ceiling: float = 0.15
    signed_error_r2_floor: float = 0.50
    residual_mae_fraction_ceiling: float = 0.50
    sign_agreement_floor: float = 0.75
    sign_magnitude_floor_bins: float = 0.01
    invariance_r2_floor: float = 0.90
    invariance_relative_l2_ceiling: float = math.sqrt(0.10)
    invariance_sign_agreement_floor: float = 0.90
    shuffled_r2_margin: float = 0.10
    replay_tolerance: float = 1e-6
    basis_gauge_replay_tolerance: float = 1e-5
    continuation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != fixed.PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("primary nuisance-scalar seeds are fixed to 7,29,53")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.orbit_count != 64:
            raise ValueError("the existing group cohort fixes 64 exact orbits")
        if self.phase_count != 16 or self.nuisance_replicates != 4:
            raise ValueError("the existing group cohort fixes 16 phases x 4 nuisances")
        if self.phase_count * self.nuisance_replicates != self.orbit_count:
            raise ValueError("phase and nuisance counts must tile the cohort")
        if self.carrier_rank != 3 or self.writer_order != 4:
            raise ValueError("carrier rank and writer order are fixed to three and four")
        if not 0.0 < self.fine_step_std < self.coarse_step_std:
            raise ValueError("finite-difference steps must be positive and ordered")


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
        Path(group.__file__),
        Path(local.__file__),
        Path(source.__file__),
        Path(source.jacobian.__file__),
        Path(fixed.__file__),
        Path(transport.__file__),
        Path(transport.rank.__file__),
        Path(transport.coupling.__file__),
        Path(local.readout.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evidence_role(config: NuisanceScalarTransformationConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_existing_artifact_post_outcome_underpowered_mechanistic_evidence"
    )


def scalar_pair_metrics(
    reference: torch.Tensor,
    transformed: torch.Tensor,
    sign_floor: float,
) -> dict[str, Any]:
    reference = reference.reshape(-1).double()
    transformed = transformed.reshape(-1).double()
    if reference.shape != transformed.shape or not len(reference):
        raise ValueError("scalar pairs must be non-empty and row aligned")
    difference = reference - transformed
    target_norm = torch.linalg.vector_norm(transformed).clamp_min(1e-24)
    target_rms = torch.sqrt(transformed.square().mean())
    mask = transformed.abs() >= sign_floor
    centered_reference = reference - reference.mean()
    centered_transformed = transformed - transformed.mean()
    correlation_denominator = (
        torch.linalg.vector_norm(centered_reference)
        * torch.linalg.vector_norm(centered_transformed)
    )
    correlation = (
        float((centered_reference * centered_transformed).sum() / correlation_denominator)
        if float(correlation_denominator) > 1e-24
        else 0.0
    )
    return {
        "zero_referenced_r2": float(
            1.0 - difference.square().sum() / transformed.square().sum().clamp_min(1e-24)
        ),
        "relative_l2": float(torch.linalg.vector_norm(difference) / target_norm),
        "mae": float(difference.abs().mean()),
        "rmse": float(torch.sqrt(difference.square().mean())),
        "target_rms": float(target_rms),
        "action_difference_rms": float(torch.sqrt(difference.square().mean())),
        "sign_agreement": (
            float((reference[mask].sign() == transformed[mask].sign()).double().mean())
            if bool(mask.any())
            else 1.0
        ),
        "sign_evaluation_count": int(mask.sum()),
        "correlation": correlation,
    }


def invariance_pass(
    metrics: Mapping[str, Any], config: NuisanceScalarTransformationConfig
) -> bool:
    return bool(
        metrics["zero_referenced_r2"] >= config.invariance_r2_floor
        and metrics["relative_l2"] <= config.invariance_relative_l2_ceiling
        and metrics["sign_agreement"] >= config.invariance_sign_agreement_floor
    )


def cell_gate(
    paired: Mapping[str, Any],
    shuffled: Mapping[str, Any],
    config: NuisanceScalarTransformationConfig,
) -> dict[str, Any]:
    margin = float(
        paired["zero_referenced_r2"] - shuffled["zero_referenced_r2"]
    )
    return {
        "invariance_pass": invariance_pass(paired, config),
        "paired_over_shuffled_r2_margin": margin,
        "correspondence_specific": bool(margin >= config.shuffled_r2_margin),
    }


def align_basis_to_stored_gauge(
    basis: torch.Tensor,
    regenerated_coordinates: torch.Tensor,
    stored_coordinates: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Recover only the predecessor's arbitrary rank-three coordinate gauge."""
    regenerated = regenerated_coordinates.detach().cpu().double()
    stored = stored_coordinates.detach().cpu().double()
    if regenerated.shape != stored.shape or regenerated.ndim != 2:
        raise ValueError("basis-gauge coordinates must be row-aligned matrices")
    transform = torch.linalg.lstsq(stored, regenerated).solution
    identity = torch.eye(transform.shape[0], dtype=torch.float64)
    aligned = transform.to(device=basis.device) @ basis.double()
    replay = stored @ transform
    return aligned, {
        "transform": transform.tolist(),
        "maximum_orthogonality_error": float(
            (transform.T @ transform - identity).abs().max()
        ),
        "maximum_gauge_fit_error": float((replay - regenerated).abs().max()),
        "aligned_basis_orthogonality_error": float(
            (
                aligned.detach().cpu() @ aligned.detach().cpu().T
                - identity
            )
            .abs()
            .max()
        ),
    }


def classify_checkpoint(
    *, valid: bool, invariant: bool, correspondence_specific: bool
) -> str:
    if not valid:
        return "invalid"
    if invariant and correspondence_specific:
        return "scalar_nuisance_invariant_and_correspondence_specific"
    if invariant:
        return "scalar_nuisance_invariant_nonspecific"
    return "scalar_action_dependent"


def _load_group_campaign(
    config: NuisanceScalarTransformationConfig,
) -> tuple[dict[str, Any], Path, dict[int, tuple[dict[str, Any], Path, Path]]]:
    path = Path(config.group_root) / "campaign_results.json"
    campaign = json.loads(path.read_text(encoding="utf-8"))
    gate_counts = campaign.get("aggregates", {}).get("gate_counts", {})
    if (
        _sha256(path) != GROUP_CAMPAIGN_SHA256
        or campaign.get("schema_version") != group.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != group.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256") != GROUP_IMPLEMENTATION_SHA256
        or gate_counts.get("input_and_pair_contract") != 3
        or gate_counts.get("numerical_contract") != 3
        or gate_counts.get("full_control_pass") != 3
        or gate_counts.get("local_tangent_fresh_cohort_pass") != 3
        or int(campaign.get("summary", {}).get("trained_models", -1)) != 0
    ):
        raise ValueError(f"invalid local metric-field group campaign {path}")
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if not set(config.seeds).issubset(entries):
        raise ValueError("group campaign lacks a requested checkpoint")
    details: dict[int, tuple[dict[str, Any], Path, Path]] = {}
    for seed in config.seeds:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        arrays_path = Path(entry["arrays"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        if (
            _sha256(detail_path) != entry.get("result_sha256")
            or _sha256(arrays_path) != entry.get("arrays_sha256")
            or detail.get("implementation_sha256") != GROUP_IMPLEMENTATION_SHA256
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or detail.get("gates", {}).get("input_and_pair_contract") is not True
            or detail.get("gates", {}).get("numerical_contract") is not True
            or detail.get("gates", {}).get("full_control_pass") is not True
            or detail.get("gates", {}).get("local_tangent_fresh_cohort_pass")
            is not True
        ):
            raise ValueError(f"invalid local metric-field group result {detail_path}")
        details[seed] = (detail, detail_path, arrays_path)
    return campaign, path, details


def _group_config(config: NuisanceScalarTransformationConfig) -> Any:
    return group.LocalMetricFieldTransportConfig(
        seeds=config.seeds,
        orbit_count=config.orbit_count,
        phase_count=config.phase_count,
        nuisance_replicates=config.nuisance_replicates,
        carrier_rank=config.carrier_rank,
        fourier_order=config.writer_order,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )


def _fingerprint(
    config: NuisanceScalarTransformationConfig,
    seed: int,
    group_result_sha256: str,
    checkpoint_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "group_campaign_sha256": GROUP_CAMPAIGN_SHA256,
        "group_cohort_seeds": group.FRESH_COHORT_SEEDS,
        "seed": seed,
        "group_result_sha256": group_result_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reusable_seed_result(
    result_path: Path,
    arrays_path: Path,
    config: NuisanceScalarTransformationConfig,
    implementation: str,
    seed: int,
    fingerprint: str,
) -> dict[str, Any] | None:
    if not result_path.is_file():
        return None
    value = json.loads(result_path.read_text(encoding="utf-8"))
    if not (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("evidence_role") == _evidence_role(config)
        and value.get("implementation_sha256") == implementation
        and value.get("configuration") == _json_compatible(asdict(config))
        and int(value.get("seed", -1)) == seed
        and value.get("scientific_fingerprint") == fingerprint
        and value.get("artifacts", {}).get("result") == str(result_path)
        and value.get("artifacts", {}).get("arrays") == str(arrays_path)
        and arrays_path.is_file()
        and _sha256(arrays_path) == value.get("artifacts", {}).get("arrays_sha256")
    ):
        raise ValueError(f"incompatible completed result {result_path}; use a new root")
    return value


def _campaign_is_reusable(
    value: Mapping[str, Any],
    config: NuisanceScalarTransformationConfig,
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
    return len(value.get("results", [])) == len(config.seeds) and all(
        Path(item.get("path", "")).is_file()
        and Path(item.get("arrays", "")).is_file()
        and _sha256(Path(item["path"])) == item.get("result_sha256")
        and _sha256(Path(item["arrays"])) == item.get("arrays_sha256")
        for item in value.get("results", [])
    )


@torch.no_grad()
def run_campaign(
    config: NuisanceScalarTransformationConfig, output: Path
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    implementation = _implementation_digest()
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if _campaign_is_reusable(existing, config, implementation):
            print("aggregate already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed aggregate {campaign_path}")

    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    group_campaign, group_path, group_details = _load_group_campaign(config)
    group_config = _group_config(config)
    _, writer_path, writer_details, _, corrective_path = group._load_sources(
        group_config
    )
    paired = {
        regime: group.generate_group_paired_orbits(
            task,
            group_config,
            seed=group.FRESH_COHORT_SEEDS[regime],
            regime=regime,
        )
        for regime in REGIMES
    }
    shift_indices = group.phase_matched_shift_indices(
        config.phase_count, config.nuisance_replicates
    ).to(device)
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
        group_detail, group_result_path, group_arrays_path = group_details[seed]
        with np.load(group_arrays_path) as stored_archive:
            stored_group_arrays = {
                name: stored_archive[name].copy() for name in stored_archive.files
            }
        prior, prior_path = writer_details[seed]
        fingerprint = _fingerprint(
            config,
            seed,
            _sha256(group_result_path),
            group_detail["provenance"]["checkpoint_sha256"],
        )
        result_path = output / "runs" / f"seed_{seed}" / "result.json"
        arrays_path = output / "runs" / f"seed_{seed}" / "scalar_law_arrays.npz"
        existing = _reusable_seed_result(
            result_path, arrays_path, config, implementation, seed, fingerprint
        )
        if existing is not None:
            results.append(existing)
            reused += 1
            print(f"resuming {existing['experiment_id']}", flush=True)
            continue

        system, provenance = transport.rank.deck.load_source(
            task, bridge, 2, seed, device
        )
        if provenance["checkpoint_sha256"] != group_detail["provenance"]["checkpoint_sha256"]:
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
        if source.decomposition._numeric_max_difference(
            basis_summary, group_detail["basis"]
        ) > config.replay_tolerance:
            raise ValueError(f"basis replay mismatch for seed {seed}")
        gauge_datasets = paired["composition"][0]
        gauge_cell = transport._extract_cell(
            system,
            task,
            transport_config,
            bridge,
            gauge_datasets["reference"],
            "heldout_a",
            "composition",
            device,
        )
        regenerated_gauge_coordinates = transport._coordinates(gauge_cell, basis)
        stored_gauge_coordinates = torch.tensor(
            stored_group_arrays["composition__reference__target"],
            dtype=torch.float64,
        )
        basis, basis_gauge_replay = align_basis_to_stored_gauge(
            basis,
            regenerated_gauge_coordinates,
            stored_gauge_coordinates,
        )
        gauge_contract = bool(
            basis_gauge_replay["maximum_orthogonality_error"]
            <= config.basis_gauge_replay_tolerance
            and basis_gauge_replay["maximum_gauge_fit_error"]
            <= config.basis_gauge_replay_tolerance
            and basis_gauge_replay["aligned_basis_orthogonality_error"]
            <= config.basis_gauge_replay_tolerance
        )
        if not gauge_contract:
            raise ValueError(f"basis gauge replay mismatch for seed {seed}")
        mapping_record = prior["alignment_fit"]["mappings"]["fourier_m04"]
        mapping = {
            "linear": torch.tensor(
                mapping_record["linear"], dtype=torch.float64, device=device
            ),
            "intercept": torch.tensor(
                mapping_record["intercept"], dtype=torch.float64, device=device
            ),
        }
        writer_mapping_sha256 = group._tensor_digest(
            mapping["linear"], mapping["intercept"]
        )
        if writer_mapping_sha256 != group_detail["writer_mapping_sha256"]:
            raise ValueError(f"writer mapping replay mismatch for seed {seed}")

        alignment_coordinates = []
        for regime in REGIMES:
            dataset = transport.rank.deck.generate_exact_orbits(
                task,
                k=2,
                orbit_count=config.orbit_count,
                seed=fixed.COHORT_SEEDS["alignment_fit"][regime],
                regime=regime,
            )
            cell = transport._extract_cell(
                system,
                task,
                transport_config,
                bridge,
                dataset,
                "alignment_fit",
                regime,
                device,
            )
            alignment_coordinates.append(transport._coordinates(cell, basis))
        coordinate_scale = torch.cat(alignment_coordinates).std(
            0, unbiased=False
        ).clamp_min(config.coordinate_scale_floor)

        cell_records: list[dict[str, Any]] = []
        numerical_pass = True
        target_control_pass = True
        input_pair_pass = True
        input_replay_pass = True
        target_coordinate_replay_pass = True
        maximum_target_coordinate_replay_error = 0.0
        arrays: dict[str, np.ndarray] = {}
        for regime in REGIMES:
            datasets, input_summary = paired[regime]
            input_pair_pass = bool(input_pair_pass and input_summary["pair_contract"])
            input_replay_pass = bool(
                input_replay_pass
                and input_summary == group_detail["fresh_inputs"][regime]
            )
            theta = (
                datasets["reference"]
                .quotient_phase.to(device)
                .reshape(config.orbit_count, 2)[:, 0]
                .double()
            )
            features = group.writer.fourier_features(theta, config.writer_order)
            predicted = transport.apply_affine(features, mapping)
            derivatives: dict[str, dict[str, torch.Tensor]] = {}
            controls: dict[str, dict[str, Any]] = {}
            for arm, dataset in datasets.items():
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
                target = transport._coordinates(cell, basis)
                stored_target = torch.tensor(
                    stored_group_arrays[f"{regime}__{arm}__target"],
                    dtype=torch.float64,
                    device=device,
                )
                target_replay_error = float((target - stored_target).abs().max())
                maximum_target_coordinate_replay_error = max(
                    maximum_target_coordinate_replay_error, target_replay_error
                )
                target_coordinate_replay_pass = bool(
                    target_coordinate_replay_pass
                    and target_replay_error <= config.basis_gauge_replay_tolerance
                )
                derivative = local.finite_difference_cell(
                    system,
                    task,
                    config,
                    cell,
                    basis,
                    predicted,
                    target,
                    coordinate_scale,
                )
                linearization = source.jacobian.linearization_metrics(
                    derivative["fine_gradient"],
                    derivative["coarse_gradient"],
                    derivative["predicted_delta"],
                    derivative["observed_delta"],
                    config,
                )
                states = local.evaluate_science_states(
                    system,
                    task,
                    config,
                    cell,
                    basis,
                    {"order4": predicted, "direct_rank3": target},
                )
                control = bool(
                    not states["zero"]["continuous"]["continuous_pass"]
                    and states["exact"]["continuous"]["continuous_pass"]
                    and states["direct_rank3"]["continuous"]["continuous_pass"]
                )
                finite = all(
                    bool(torch.isfinite(value).all())
                    for value in derivative.values()
                )
                numerical_pass = bool(
                    numerical_pass and finite and linearization["adequate"]
                )
                target_control_pass = bool(target_control_pass and control)
                derivatives[arm] = derivative
                controls[arm] = {
                    "finite": finite,
                    "linearization": linearization,
                    "target_control_pass": control,
                }
                prefix = f"{regime}__{arm}"
                arrays[f"{prefix}__signed_scalar"] = (
                    derivative["observed_delta"].detach().cpu().numpy()
                )
                arrays[f"{prefix}__fine_gradient"] = (
                    derivative["fine_gradient"].detach().cpu().numpy()
                )

            reference = derivatives["reference"]["observed_delta"]
            for arm in GROUP_ARMS:
                transformed = derivatives[arm]["observed_delta"]
                paired_metrics = scalar_pair_metrics(
                    reference, transformed, config.sign_magnitude_floor_bins
                )
                shuffled_metrics = scalar_pair_metrics(
                    reference[shift_indices],
                    transformed,
                    config.sign_magnitude_floor_bins,
                )
                gate = cell_gate(paired_metrics, shuffled_metrics, config)
                cell_records.append(
                    {
                        "regime": regime,
                        "arm": arm,
                        "paired": paired_metrics,
                        "phase_matched_shuffled": shuffled_metrics,
                        "gate": gate,
                        "reference_finite": controls["reference"]["finite"],
                        "transformed_finite": controls[arm]["finite"],
                        "reference_linearization": controls["reference"]["linearization"],
                        "transformed_linearization": controls[arm]["linearization"],
                    }
                )

        invariant = all(item["gate"]["invariance_pass"] for item in cell_records)
        correspondence = all(
            item["gate"]["correspondence_specific"] for item in cell_records
        )
        valid = bool(
            input_pair_pass
            and input_replay_pass
            and target_coordinate_replay_pass
            and numerical_pass
            and target_control_pass
            and bool(torch.isfinite(coordinate_scale).all())
            and float(coordinate_scale.min()) > config.coordinate_scale_floor
        )
        classification = classify_checkpoint(
            valid=valid,
            invariant=invariant,
            correspondence_specific=correspondence,
        )
        _write_npz(arrays_path, arrays)
        arrays_sha256 = _sha256(arrays_path)
        result = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_id": f"tinyllm-c2-nuisance-scalar-law-seed{seed}",
            "status": "completed",
            "evidence_role": _evidence_role(config),
            "completed_at": _utc_now(),
            "seed": seed,
            "configuration": asdict(config),
            "scientific_fingerprint": fingerprint,
            "implementation_sha256": implementation,
            "provenance": {
                "group_campaign": str(group_path),
                "group_campaign_sha256": _sha256(group_path),
                "group_result": str(group_result_path),
                "group_result_sha256": _sha256(group_result_path),
                "group_arrays": str(group_arrays_path),
                "group_arrays_sha256": _sha256(group_arrays_path),
                "writer_campaign": str(writer_path),
                "writer_result": str(prior_path),
                "corrective_campaign": str(corrective_path),
                "checkpoint": provenance["checkpoint"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "character_result": str(frozen_path),
                "character_result_sha256": _sha256(frozen_path),
            },
            "basis": basis_summary,
            "basis_gauge_replay": {
                **basis_gauge_replay,
                "maximum_all_cell_target_replay_error": (
                    maximum_target_coordinate_replay_error
                ),
            },
            "writer_mapping_sha256": writer_mapping_sha256,
            "coordinate_scale": coordinate_scale.detach().cpu().tolist(),
            "cells": cell_records,
            "gates": {
                "provenance_contract": True,
                "input_and_pair_contract": input_pair_pass,
                "input_identity_replay_contract": input_replay_pass,
                "basis_gauge_replay_contract": bool(
                    gauge_contract and target_coordinate_replay_pass
                ),
                "local_linearization_contract": numerical_pass,
                "target_control_contract": target_control_pass,
                "scalar_invariance_all_cells_pass": invariant,
                "correspondence_specificity_all_cells_pass": correspondence,
                "nuisance_scalar_transformation_gate": bool(valid and invariant),
            },
            "classification": classification,
            "primary_metric": float(valid and invariant),
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
            raise RuntimeError("nuisance-scalar implementation changed during campaign")

    classifications = [result["classification"] for result in results]
    pass_count = sum(
        result["gates"]["nuisance_scalar_transformation_gate"] for result in results
    )
    if any(name == "invalid" for name in classifications):
        conclusion = "invalid_campaign"
    elif len(config.seeds) == 3 and pass_count == 3:
        conclusion = "supported_scalar_nuisance_invariance_three_of_three"
    elif len(set(classifications)) == 1:
        conclusion = classifications[0]
    else:
        conclusion = "checkpoint_stratified_scalar_transformation_law"
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "group_cohort_seeds": group.FRESH_COHORT_SEEDS,
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
            "group_campaign": str(group_path),
            "group_campaign_sha256": _sha256(group_path),
            "group_implementation_sha256": group_campaign["implementation_sha256"],
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
            "fitted_writers": 0,
            "fitted_encoders": 0,
            "fitted_observers": 0,
            "group_cells": len(config.seeds) * len(REGIMES) * len(GROUP_ARMS),
            "signed_scalar_fields": len(config.seeds)
            * len(REGIMES)
            * (len(GROUP_ARMS) + 1),
        },
        "aggregates": {
            "conclusion": conclusion,
            "primary_checkpoint_pass_count": pass_count,
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
                    "provenance_contract",
                    "input_and_pair_contract",
                    "input_identity_replay_contract",
                    "basis_gauge_replay_contract",
                    "local_linearization_contract",
                    "target_control_contract",
                    "scalar_invariance_all_cells_pass",
                    "correspondence_specificity_all_cells_pass",
                    "nuisance_scalar_transformation_gate",
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
            "This is a preregistered derived diagnostic on an already analyzed exact-group cohort, not fresh evidence.",
            "The signed target uses exact direct-rank-three continuation outcomes unavailable at inference time.",
            "The observed similarity group covers positive scale, planar rotation, and constant offset only.",
            "Failure of scalar invariance rules out an invariant output type, not every action-conditioned equivariant sensor.",
            "Three selected checkpoints do not establish population prevalence.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "seed_*" / "result.json"),
            "arrays": str(output / "runs" / "seed_*" / "scalar_law_arrays.npz"),
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
            "data/experiments/tinyllm_nuisance_scalar_transformation_law/"
            "20260807_d6_existing_group"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=_ints, default=fixed.PRIMARY_SEEDS)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = NuisanceScalarTransformationConfig(
        seeds=args.seeds,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
