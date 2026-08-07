#!/usr/bin/env python3
"""Causally transport the stable TinyLLM C2 carrier across checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import torch

import experiments.structure_net.tinyllm_continuous_readout_calibration as readout
import experiments.structure_net.tinyllm_defect_subspace_rank as rank
import experiments.structure_net.tinyllm_reynolds_character_coupling as coupling
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-cross-seed-causal-carrier-transport.v1"
HYPOTHESIS_ID = "tinyllm-c2-cross-seed-causal-carrier-transport-v1"
PRIMARY_SEEDS = (7, 29, 53)
REGIMES = ("composition", "extrapolation")
COHORT_SEEDS = {
    "alignment_fit": {"composition": 130007, "extrapolation": 130008},
    "heldout_a": {"composition": 230007, "extrapolation": 230008},
    "heldout_b": {"composition": 330007, "extrapolation": 330008},
}
HELDOUT_COHORTS = ("heldout_a", "heldout_b")


@dataclass(frozen=True)
class CrossSeedCarrierTransportConfig:
    source_root: str = rank.DefectSubspaceRankConfig.source_root
    character_root: str = rank.DefectSubspaceRankConfig.character_root
    readout_root: str = (
        "data/experiments/tinyllm_continuous_readout_calibration/"
        "20260806_d6_corrective_v3_6a88480c"
    )
    seeds: tuple[int, ...] = PRIMARY_SEEDS
    orbit_count: int = 64
    carrier_rank: int = 3
    first_blocks: int = 6
    coordinate_r2_floor: float = 0.80
    specificity_margin: float = 0.20
    whitening_ridge: float = 1e-6
    affine_ridge: float = 1e-6
    alignment_loss_ceiling: float = 0.005
    mean_shift_ceiling_bins: float = 0.125
    p95_shift_ceiling_bins: float = 0.50
    winding_tolerance: float = 0.10
    accuracy_loss_ceiling: float = 0.03
    task_effect_floor: float = 1e-6
    decomposition_tolerance: float = 1e-6
    activation_batch_size: int = 256
    continuation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("confirmatory transport seeds are fixed to 7,29,53")
        if self.orbit_count < 8:
            raise ValueError("at least eight exact orbits are required")
        if self.carrier_rank != 3 and not self.allow_underpowered:
            raise ValueError("confirmatory common carrier rank is fixed to three")
        if not 0.0 < self.coordinate_r2_floor < 1.0:
            raise ValueError("coordinate R2 floor must lie between zero and one")
        if not 0.0 < self.specificity_margin < 1.0:
            raise ValueError("specificity margin must lie between zero and one")
        if min(self.whitening_ridge, self.affine_ridge) <= 0.0:
            raise ValueError("alignment ridges must be positive")


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


def _json_compatible(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(rank.__file__),
        Path(readout.__file__),
        Path(coupling.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evidence_role(config: CrossSeedCarrierTransportConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_underpowered_mechanistic_evidence"
    )


def _rank_config(config: CrossSeedCarrierTransportConfig) -> rank.DefectSubspaceRankConfig:
    return rank.DefectSubspaceRankConfig(
        source_root=config.source_root,
        character_root=config.character_root,
        seeds=rank.PRIMARY_SEEDS,
        orbit_count=config.orbit_count,
        first_blocks=config.first_blocks,
        ranks=rank.DEFAULT_RANKS,
        task_effect_floor=config.task_effect_floor,
        decomposition_tolerance=config.decomposition_tolerance,
        activation_batch_size=config.activation_batch_size,
        continuation_batch_size=config.continuation_batch_size,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )


def _readout_config(
    config: CrossSeedCarrierTransportConfig,
) -> readout.ContinuousReadoutCalibrationConfig:
    return readout.ContinuousReadoutCalibrationConfig(
        source_root=config.source_root,
        character_root=config.character_root,
        seeds=readout.PRIMARY_SEEDS,
        orbit_count=config.orbit_count,
        first_blocks=config.first_blocks,
        ranks=readout.RANKS,
        alignment_loss_ceiling=config.alignment_loss_ceiling,
        mean_shift_ceiling_bins=config.mean_shift_ceiling_bins,
        p95_shift_ceiling_bins=config.p95_shift_ceiling_bins,
        winding_tolerance=config.winding_tolerance,
        accuracy_loss_ceiling=config.accuracy_loss_ceiling,
        task_effect_floor=config.task_effect_floor,
        decomposition_tolerance=config.decomposition_tolerance,
        activation_batch_size=config.activation_batch_size,
        continuation_batch_size=config.continuation_batch_size,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
        post_outcome_corrective_replication=not config.allow_underpowered,
    )


def fit_whitened_orthogonal(
    source: torch.Tensor, target: torch.Tensor, ridge: float
) -> dict[str, torch.Tensor]:
    """Fit an affine map that is orthogonal after separate source/target whitening."""
    source = source.double()
    target = target.double()
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
        raise ValueError("paired coordinates must have equal observation-by-feature shape")
    if source.shape[0] <= source.shape[1]:
        raise ValueError("alignment needs more observations than features")
    source_mean = source.mean(0)
    target_mean = target.mean(0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    def factors(centered: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        covariance = centered.T @ centered / float(centered.shape[0] - 1)
        values, vectors = torch.linalg.eigh(covariance)
        floor = values.max().clamp_min(1e-24) * ridge
        values = values.clamp_min(floor)
        whitening = (vectors * values.rsqrt().unsqueeze(0)) @ vectors.T
        coloring = (vectors * values.sqrt().unsqueeze(0)) @ vectors.T
        return whitening, coloring

    source_whitening, _ = factors(source_centered)
    target_whitening, target_coloring = factors(target_centered)
    source_white = source_centered @ source_whitening
    target_white = target_centered @ target_whitening
    left, singular, right_h = torch.linalg.svd(
        source_white.T @ target_white, full_matrices=False
    )
    rotation = left @ right_h
    linear = source_whitening @ rotation @ target_coloring
    intercept = target_mean - source_mean @ linear
    return {
        "linear": linear,
        "intercept": intercept,
        "whitened_cross_singular_values": singular
        / float(source.shape[0] - 1),
    }


def fit_affine_ridge(
    source: torch.Tensor, target: torch.Tensor, ridge: float
) -> dict[str, torch.Tensor]:
    source = source.double()
    target = target.double()
    ones = torch.ones((source.shape[0], 1), dtype=source.dtype, device=source.device)
    design = torch.cat((source, ones), dim=1)
    penalty = torch.eye(design.shape[1], dtype=source.dtype, device=source.device)
    penalty[-1, -1] = 0.0
    weights = torch.linalg.solve(
        design.T @ design + ridge * penalty,
        design.T @ target,
    )
    return {"linear": weights[:-1], "intercept": weights[-1]}


def apply_affine(coordinates: torch.Tensor, mapping: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return coordinates.double() @ mapping["linear"] + mapping["intercept"]


def coordinate_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    predicted = predicted.double()
    target = target.double()
    residual = (predicted - target).square().sum()
    total = (target - target.mean(0)).square().sum().clamp_min(1e-24)
    predicted_norm = torch.linalg.vector_norm(predicted, dim=1)
    target_norm = torch.linalg.vector_norm(target, dim=1)
    cosine = (predicted * target).sum(1) / (predicted_norm * target_norm).clamp_min(1e-24)
    return {
        "variance_explained": float(1.0 - residual / total),
        "normalized_rmse": float(torch.sqrt(residual / total)),
        "mean_row_cosine": float(cosine.mean()),
    }


def regime_preserving_permutation(
    count_per_regime: int, source_seed: int, target_seed: int
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(
        9_000_000 + source_seed * 1009 + target_seed * 9176
    )
    parts = []
    for regime_index in range(len(REGIMES)):
        offset = regime_index * count_per_regime
        parts.append(torch.randperm(count_per_regime, generator=generator) + offset)
    return torch.cat(parts)


def _load_readout_predecessor(
    config: CrossSeedCarrierTransportConfig, seed: int, checkpoint_sha256: str
) -> dict[str, Any]:
    root = Path(config.readout_root)
    campaign_path = root / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text())
    references = {int(item["seed"]): item for item in campaign.get("results", [])}
    reference = references.get(seed, {})
    result_path = root / "runs" / f"seed_{seed}" / "result.json"
    result = json.loads(result_path.read_text())
    if (
        campaign.get("schema_version") != readout.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != readout.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role")
        != "post_outcome_corrective_replication_evidence"
        or campaign.get("configuration", {}).get(
            "post_outcome_corrective_replication"
        )
        is not True
        or campaign.get("aggregates", {}).get("confirmed") is not True
        or result.get("schema_version") != readout.SCHEMA_VERSION
        or result.get("hypothesis_id") != readout.HYPOTHESIS_ID
        or result.get("status") != "completed"
        or result.get("evidence_role")
        != "post_outcome_corrective_replication_evidence"
        or result.get("implementation_sha256")
        != campaign.get("implementation_sha256")
        or int(result.get("seed", -1)) != seed
        or result.get("provenance", {}).get("checkpoint_sha256")
        != checkpoint_sha256
        or result.get("scientific_fingerprint")
        != reference.get("scientific_fingerprint")
        or _sha256(result_path) != reference.get("result_sha256")
    ):
        raise ValueError(f"invalid readout predecessor for seed {seed}")
    return {
        "campaign": str(campaign_path),
        "campaign_sha256": _sha256(campaign_path),
        "result": str(result_path),
        "result_sha256": _sha256(result_path),
        "rotation_bins": float(
            result["source_selection"]["selected_calibration"]["rotation_bins"]
        ),
    }


@torch.no_grad()
def _extract_cell(
    system: Any,
    task: CircleTaskConfig,
    config: CrossSeedCarrierTransportConfig,
    bridge: Any,
    dataset: Any,
    cohort: str,
    regime: str,
    device: torch.device,
) -> dict[str, Any]:
    source_cut, target_cut = rank.koopman.one_step_transitions(config.first_blocks)[0]
    captured = rank.deck.capture_sequences(system, dataset, bridge, device)
    source = torch.from_numpy(captured[source_cut]).to(device)
    source = source.reshape((config.orbit_count, 2) + source.shape[1:])
    block = system.model.transformer["h"][0]
    propagated, _, exact, error = rank.heads.exact_head_defects(block, source)
    defect = exact - propagated
    baseline_posterior = coupling._continue_tensor(
        system,
        "full",
        torch.from_numpy(captured["full"]).to(device),
        task,
        config.continuation_batch_size,
    )
    baseline = rank.deck.output_diagnostics(
        baseline_posterior.double().cpu().numpy(), dataset, include_topology=False
    )
    return {
        "cohort": cohort,
        "regime": regime,
        "evaluation_seed": COHORT_SEEDS[cohort][regime],
        "target_cut": target_cut,
        "dataset": dataset,
        "baseline": baseline,
        "propagated": propagated,
        "exact": exact,
        "full_defect": defect,
        "decomposition_relative_error": error,
    }


def _fit_seed_basis(
    system: Any,
    task: CircleTaskConfig,
    config: CrossSeedCarrierTransportConfig,
    rank_config: rank.DefectSubspaceRankConfig,
    bridge: Any,
    frozen: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    cells = [
        rank._defect_cell(
            system,
            task,
            rank_config,
            bridge,
            frozen,
            seed,
            "source_selection",
            regime,
            device,
        )
        for regime in REGIMES
    ]
    matrix = torch.cat(
        [cell["full_defect"].reshape(config.orbit_count, -1) for cell in cells],
        dim=0,
    )
    basis, energy = rank.fit_right_singular_basis(
        matrix, rank_config.numerical_rank_rtol
    )
    selected = basis[: config.carrier_rank]
    return selected, {
        "numerical_rank": int(basis.shape[0]),
        "features": int(basis.shape[1]),
        "carrier_rank": config.carrier_rank,
        "carrier_cumulative_energy": float(energy[: config.carrier_rank].sum()),
        "orthogonality_error": float(
            torch.linalg.matrix_norm(
                selected @ selected.T
                - torch.eye(
                    config.carrier_rank,
                    device=selected.device,
                    dtype=selected.dtype,
                )
            )
        ),
    }


def _coordinates(cell: Mapping[str, Any], basis: torch.Tensor) -> torch.Tensor:
    flat = cell["full_defect"].reshape(cell["full_defect"].shape[0], -1)
    return flat.double() @ basis.double().T


def _mapping_summary(mapping: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    output = {
        "linear": mapping["linear"].detach().cpu().tolist(),
        "intercept": mapping["intercept"].detach().cpu().tolist(),
    }
    if "whitened_cross_singular_values" in mapping:
        output["whitened_cross_singular_values"] = mapping[
            "whitened_cross_singular_values"
        ].detach().cpu().tolist()
    return output


@torch.no_grad()
def _evaluate_transport_cell(
    target_system: Any,
    task: CircleTaskConfig,
    config: CrossSeedCarrierTransportConfig,
    source_cell: Mapping[str, Any],
    target_cell: Mapping[str, Any],
    source_basis: torch.Tensor,
    target_basis: torch.Tensor,
    mappings: Mapping[str, Mapping[str, torch.Tensor]],
    target_rotation_bins: float,
) -> dict[str, Any]:
    source_coordinates = _coordinates(source_cell, source_basis)
    target_coordinates = _coordinates(target_cell, target_basis)
    predicted_coordinates = {
        name: apply_affine(source_coordinates, mapping).to(target_basis.device)
        for name, mapping in mappings.items()
    }
    coordinate_records = {
        name: coordinate_metrics(value, target_coordinates)
        for name, value in predicted_coordinates.items()
    }

    target_flat = target_cell["full_defect"].reshape(config.orbit_count, -1)
    direct_defect = (target_coordinates @ target_basis.double()).to(
        dtype=target_flat.dtype
    ).reshape_as(target_cell["full_defect"])
    states = {
        "zero": target_cell["propagated"],
        "exact": target_cell["exact"],
        "direct_rank3": target_cell["propagated"] + direct_defect,
    }
    for name, coordinates in predicted_coordinates.items():
        predicted_defect = (coordinates @ target_basis.double()).to(
            dtype=target_flat.dtype
        ).reshape_as(target_cell["full_defect"])
        states[name] = target_cell["propagated"] + predicted_defect

    keys = tuple(states)
    repeated = torch.cat(
        [coupling._repeat_patch(states[key], 2) for key in keys], dim=0
    )
    combined = coupling._continue_tensor(
        target_system,
        target_cell["target_cut"],
        repeated,
        task,
        config.continuation_batch_size,
    )
    per_state = config.orbit_count * 2
    raw_posteriors = dict(zip(keys, combined.split(per_state)))
    orbit_posteriors = {
        key: value.reshape(config.orbit_count, 2, -1)[:, 0]
        for key, value in raw_posteriors.items()
    }
    exact_repeated = orbit_posteriors["exact"][:, None].expand(-1, 2, -1).reshape(
        per_state, -1
    )
    exact_diagnostics = coupling._diagnostics(
        exact_repeated, target_cell["dataset"]
    )
    targets = torch.as_tensor(
        target_cell["dataset"].target_bins.reshape(config.orbit_count, 2)[:, 0],
        device=target_basis.device,
    )
    bins = len(task.answer_token_ids)
    readout_config = _readout_config(config)
    state_records = {}
    for key, posterior in orbit_posteriors.items():
        repeated_posterior = posterior[:, None].expand(-1, 2, -1).reshape(
            per_state, -1
        )
        diagnostics = coupling._diagnostics(
            repeated_posterior, target_cell["dataset"]
        )
        angle, _ = readout.posterior_moment(posterior)
        discrete = readout.discrete_readout_metrics(
            angle,
            targets,
            bins,
            target_rotation_bins,
            target_cell["baseline"]["exact_bin_accuracy"],
            config.accuracy_loss_ceiling,
        )
        continuous = readout.continuous_metrics(
            posterior,
            orbit_posteriors["exact"],
            diagnostics,
            exact_diagnostics,
            readout_config,
        )
        fisher = rank.heads._effect_preservation(
            posterior,
            orbit_posteriors["exact"],
            orbit_posteriors["zero"],
            config.task_effect_floor,
        )
        state_records[key] = {
            "continuous": continuous,
            "discrete": discrete,
            "diagnostics": diagnostics,
            "fisher_effect": fisher,
            "joint_pass": bool(continuous["continuous_pass"] and discrete["discrete_pass"]),
        }
    return {
        "cohort": target_cell["cohort"],
        "regime": target_cell["regime"],
        "evaluation_seed": target_cell["evaluation_seed"],
        "coordinate_metrics": coordinate_records,
        "states": state_records,
        "decomposition_relative_error": float(
            target_cell["decomposition_relative_error"]
        ),
    }


def _fingerprint(
    config: CrossSeedCarrierTransportConfig,
    task: CircleTaskConfig,
    source_seed: int,
    target_seed: int,
    source_provenance: Mapping[str, Any],
    target_provenance: Mapping[str, Any],
    source_readout: Mapping[str, Any],
    target_readout: Mapping[str, Any],
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "task": asdict(task),
        "cohort_seeds": COHORT_SEEDS,
        "source_seed": source_seed,
        "target_seed": target_seed,
        "source_checkpoint_sha256": source_provenance["checkpoint_sha256"],
        "target_checkpoint_sha256": target_provenance["checkpoint_sha256"],
        "source_readout_sha256": source_readout["result_sha256"],
        "target_readout_sha256": target_readout["result_sha256"],
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode()).hexdigest()


def _pair_gates(
    heldout: Sequence[Mapping[str, Any]],
    config: CrossSeedCarrierTransportConfig,
) -> dict[str, Any]:
    controls = all(
        not cell["states"]["zero"]["joint_pass"]
        and cell["states"]["exact"]["joint_pass"]
        and cell["states"]["direct_rank3"]["joint_pass"]
        and cell["decomposition_relative_error"] <= config.decomposition_tolerance
        for cell in heldout
    )
    paired_coordinates = all(
        cell["coordinate_metrics"]["paired"]["variance_explained"]
        >= config.coordinate_r2_floor
        for cell in heldout
    )
    paired_causal = all(cell["states"]["paired"]["joint_pass"] for cell in heldout)
    shuffled_fails = any(
        cell["coordinate_metrics"]["shuffled"]["variance_explained"]
        < config.coordinate_r2_floor
        or not cell["states"]["shuffled"]["joint_pass"]
        for cell in heldout
    )
    paired_worst = min(
        cell["coordinate_metrics"]["paired"]["variance_explained"]
        for cell in heldout
    )
    shuffled_worst = min(
        cell["coordinate_metrics"]["shuffled"]["variance_explained"]
        for cell in heldout
    )
    specificity = bool(
        shuffled_fails
        and paired_worst - shuffled_worst >= config.specificity_margin
    )
    return {
        "target_control_contract": controls,
        "paired_coordinate_transport": paired_coordinates,
        "paired_causal_transport": paired_causal,
        "shuffled_specificity": specificity,
        "paired_worst_variance_explained": paired_worst,
        "shuffled_worst_variance_explained": shuffled_worst,
        "specificity_margin": paired_worst - shuffled_worst,
    }


def _campaign_is_reusable(
    campaign: Mapping[str, Any],
    config: CrossSeedCarrierTransportConfig,
    implementation: str,
) -> bool:
    required_pairs = len(config.seeds) * (len(config.seeds) - 1)
    if (
        campaign.get("schema_version") != SCHEMA_VERSION
        or campaign.get("hypothesis_id") != HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != _evidence_role(config)
        or campaign.get("implementation_sha256") != implementation
        or campaign.get("configuration") != _json_compatible(asdict(config))
        or int(campaign.get("summary", {}).get("completed", -1))
        != required_pairs
        or len(campaign.get("results", [])) != required_pairs
    ):
        return False
    fingerprints = set()
    pairs = set()
    for entry in campaign["results"]:
        path = Path(entry.get("path", ""))
        if not path.is_file() or _sha256(path) != entry.get("result_sha256"):
            return False
        detail = json.loads(path.read_text())
        pair = (int(entry["source_seed"]), int(entry["target_seed"]))
        fingerprint = entry.get("scientific_fingerprint")
        if (
            pair[0] == pair[1]
            or pair in pairs
            or not fingerprint
            or fingerprint in fingerprints
            or detail.get("schema_version") != SCHEMA_VERSION
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != _evidence_role(config)
            or detail.get("implementation_sha256") != implementation
            or detail.get("scientific_fingerprint") != fingerprint
        ):
            return False
        pairs.add(pair)
        fingerprints.add(fingerprint)
    return pairs == {
        (source, target)
        for source in config.seeds
        for target in config.seeds
        if source != target
    }


def run_campaign(
    config: CrossSeedCarrierTransportConfig, output: Path
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    implementation = _implementation_digest()
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing_campaign = json.loads(campaign_path.read_text())
        if _campaign_is_reusable(existing_campaign, config, implementation):
            print("aggregate already complete; leaving bytes unchanged", flush=True)
            return existing_campaign
        raise ValueError(
            f"incompatible completed aggregate {campaign_path}; use a new root"
        )
    rank_config = _rank_config(config)
    bridge = rank._bridge_config(rank_config)

    datasets = {
        cohort: {
            regime: rank.deck.generate_exact_orbits(
                task,
                k=2,
                orbit_count=config.orbit_count,
                seed=COHORT_SEEDS[cohort][regime],
                regime=regime,
            )
            for regime in REGIMES
        }
        for cohort in COHORT_SEEDS
    }

    seed_records: dict[int, dict[str, Any]] = {}
    for seed in config.seeds:
        system, provenance = rank.deck.load_source(task, bridge, 2, seed, device)
        frozen, frozen_path = rank._load_character_source(
            rank_config, seed, provenance["checkpoint_sha256"]
        )
        if any(
            int(frozen["regimes"][regime]["synthesis_front_index"]) != 0
            for regime in REGIMES
        ):
            raise ValueError(f"seed {seed} is not a stable block-0 front")
        basis, basis_summary = _fit_seed_basis(
            system,
            task,
            config,
            rank_config,
            bridge,
            frozen,
            seed,
            device,
        )
        readout_predecessor = _load_readout_predecessor(
            config, seed, provenance["checkpoint_sha256"]
        )
        cells = {
            cohort: {
                regime: _extract_cell(
                    system,
                    task,
                    config,
                    bridge,
                    datasets[cohort][regime],
                    cohort,
                    regime,
                    device,
                )
                for regime in REGIMES
            }
            for cohort in COHORT_SEEDS
        }
        seed_records[seed] = {
            "system": system,
            "provenance": {
                **provenance,
                "character_result": str(frozen_path),
                "character_result_sha256": _sha256(frozen_path),
            },
            "basis": basis,
            "basis_summary": basis_summary,
            "readout": readout_predecessor,
            "cells": cells,
        }
        print(f"seed {seed} carriers extracted", flush=True)
        if _implementation_digest() != implementation:
            raise RuntimeError("analysis implementation changed during campaign")

    pair_results = []
    reused = 0
    for source_seed in config.seeds:
        for target_seed in config.seeds:
            if source_seed == target_seed:
                continue
            source_record = seed_records[source_seed]
            target_record = seed_records[target_seed]
            source_fit = torch.cat(
                [
                    _coordinates(
                        source_record["cells"]["alignment_fit"][regime],
                        source_record["basis"],
                    )
                    for regime in REGIMES
                ],
                dim=0,
            )
            target_fit = torch.cat(
                [
                    _coordinates(
                        target_record["cells"]["alignment_fit"][regime],
                        target_record["basis"],
                    )
                    for regime in REGIMES
                ],
                dim=0,
            )
            paired = fit_whitened_orthogonal(
                source_fit, target_fit, config.whitening_ridge
            )
            permutation = regime_preserving_permutation(
                config.orbit_count, source_seed, target_seed
            ).to(target_fit.device)
            shuffled = fit_whitened_orthogonal(
                source_fit, target_fit[permutation], config.whitening_ridge
            )
            ridge_mapping = fit_affine_ridge(
                source_fit, target_fit, config.affine_ridge
            )
            mappings = {
                "paired": paired,
                "shuffled": shuffled,
                "affine_ridge": ridge_mapping,
            }
            heldout = []
            for cohort in HELDOUT_COHORTS:
                for regime in REGIMES:
                    heldout.append(
                        _evaluate_transport_cell(
                            target_record["system"],
                            task,
                            config,
                            source_record["cells"][cohort][regime],
                            target_record["cells"][cohort][regime],
                            source_record["basis"],
                            target_record["basis"],
                            mappings,
                            target_record["readout"]["rotation_bins"],
                        )
                    )
            gates = _pair_gates(heldout, config)
            fingerprint = _fingerprint(
                config,
                task,
                source_seed,
                target_seed,
                source_record["provenance"],
                target_record["provenance"],
                source_record["readout"],
                target_record["readout"],
            )
            result_path = (
                output
                / "runs"
                / f"source_{source_seed}_target_{target_seed}"
                / "result.json"
            )
            result = {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "experiment_id": f"tinyllm-c2-carrier-transport-{source_seed}-to-{target_seed}",
                "status": "completed",
                "evidence_role": _evidence_role(config),
                "completed_at": _utc_now(),
                "source_seed": source_seed,
                "target_seed": target_seed,
                "configuration": asdict(config),
                "scientific_fingerprint": fingerprint,
                "implementation_sha256": implementation,
                "provenance": {
                    "source": source_record["provenance"],
                    "target": target_record["provenance"],
                    "source_readout": source_record["readout"],
                    "target_readout": target_record["readout"],
                },
                "source_basis": source_record["basis_summary"],
                "target_basis": target_record["basis_summary"],
                "alignment_fit": {
                    "paired_metrics": coordinate_metrics(
                        apply_affine(source_fit, paired), target_fit
                    ),
                    "shuffled_metrics_against_true_pairs": coordinate_metrics(
                        apply_affine(source_fit, shuffled), target_fit
                    ),
                    "affine_ridge_metrics": coordinate_metrics(
                        apply_affine(source_fit, ridge_mapping), target_fit
                    ),
                    "paired_map": _mapping_summary(paired),
                    "shuffled_map": _mapping_summary(shuffled),
                    "affine_ridge_map": _mapping_summary(ridge_mapping),
                    "permutation_sha256": hashlib.sha256(
                        permutation.cpu().numpy().tobytes()
                    ).hexdigest(),
                },
                "heldout_cells": heldout,
                "gates": gates,
                "primary_metric": float(
                    all(
                        gates[name]
                        for name in (
                            "target_control_contract",
                            "paired_coordinate_transport",
                            "paired_causal_transport",
                            "shuffled_specificity",
                        )
                    )
                ),
                "analysis_seconds": time.perf_counter() - started,
                "artifacts": {"result": str(result_path)},
            }
            if result_path.is_file():
                existing = json.loads(result_path.read_text())
                if (
                    existing.get("schema_version") == SCHEMA_VERSION
                    and existing.get("hypothesis_id") == HYPOTHESIS_ID
                    and existing.get("status") == "completed"
                    and existing.get("evidence_role") == _evidence_role(config)
                    and existing.get("implementation_sha256") == implementation
                    and existing.get("scientific_fingerprint") == fingerprint
                    and int(existing.get("source_seed", -1)) == source_seed
                    and int(existing.get("target_seed", -1)) == target_seed
                ):
                    result = existing
                    reused += 1
                    print(
                        f"resuming {existing['experiment_id']}", flush=True
                    )
                else:
                    raise ValueError(
                        f"incompatible completed result {result_path}; use a new root"
                    )
            else:
                _write_json(result_path, result)
            pair_results.append(result)
            print(f"source {source_seed} -> target {target_seed} complete", flush=True)
            if _implementation_digest() != implementation:
                raise RuntimeError("analysis implementation changed during campaign")

    gate_names = (
        "target_control_contract",
        "paired_coordinate_transport",
        "paired_causal_transport",
        "shuffled_specificity",
    )
    gate_counts = {
        name: sum(bool(result["gates"][name]) for result in pair_results)
        for name in gate_names
    }
    required_pairs = len(config.seeds) * (len(config.seeds) - 1)
    confirmed = bool(
        not config.allow_underpowered
        and required_pairs == 6
        and all(gate_counts[name] == required_pairs for name in gate_names)
    )
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "cohort_seeds": COHORT_SEEDS,
        "implementation_sha256": implementation,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu"
            ),
        },
        "summary": {
            "requested": required_pairs,
            "scheduled": required_pairs - reused,
            "completed": len(pair_results),
            "failed": 0,
            "excluded": 0,
            "retries": 0,
            "reused": reused,
            "trained_models": 0,
            "fitted_predictive_observers": 0,
            "fitted_coordinate_maps": required_pairs * 3,
        },
        "aggregates": {
            "gate_counts": gate_counts,
            "required_pairs": required_pairs,
            "confirmed": confirmed,
            "conclusion": (
                "systems_lifecycle_only_not_quality_evidence"
                if config.allow_underpowered
                else (
                    "confirmed_cross_seed_causal_carrier_transport"
                    if confirmed
                    else "not_confirmed_cross_seed_causal_carrier_transport"
                )
            ),
            "per_pair": {
                f"{result['source_seed']}->{result['target_seed']}": result["gates"]
                for result in pair_results
            },
        },
        "results": [
            {
                "experiment_id": result["experiment_id"],
                "source_seed": result["source_seed"],
                "target_seed": result["target_seed"],
                "scientific_fingerprint": result["scientific_fingerprint"],
                "path": result["artifacts"]["result"],
                "result_sha256": _sha256(Path(result["artifacts"]["result"])),
                "primary_metric": result["primary_metric"],
                "gates": result["gates"],
            }
            for result in pair_results
        ],
        "method_boundaries": [
            "Only three previously selected stable C2 block-0 checkpoints are tested.",
            "Coordinate maps are fitted on paired source orbits and are not independently computable encoders.",
            "The intervention is an off-manifold frozen residual patch conditioned on the target continuation.",
            "Post-synthesis carrier transport does not align individual neurons or attention heads.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "source_*_target_*" / "result.json"),
        },
        "analysis_seconds": time.perf_counter() - started,
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
            "data/experiments/tinyllm_cross_seed_causal_carrier_transport/"
            "20260806_d6_preregistered"
        ),
    )
    parser.add_argument("--source-root", default=CrossSeedCarrierTransportConfig.source_root)
    parser.add_argument("--character-root", default=CrossSeedCarrierTransportConfig.character_root)
    parser.add_argument("--readout-root", default=CrossSeedCarrierTransportConfig.readout_root)
    parser.add_argument("--seeds", default=",".join(map(str, PRIMARY_SEEDS)))
    parser.add_argument("--orbit-count", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = CrossSeedCarrierTransportConfig(
        source_root=args.source_root,
        character_root=args.character_root,
        readout_root=args.readout_root,
        seeds=_ints(args.seeds),
        orbit_count=args.orbit_count,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
