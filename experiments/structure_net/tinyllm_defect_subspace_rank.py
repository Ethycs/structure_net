#!/usr/bin/env python3
"""Causally titrate the stable TinyLLM C2 Reynolds-defect subspace rank."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Optional, Sequence

import torch

import experiments.structure_net.tinyllm_c2_attention_head_decomposition as heads
import experiments.structure_net.tinyllm_deck_action_descrambler as deck
import experiments.structure_net.tinyllm_reynolds_character_coupling as coupling
import experiments.structure_net.tinyllm_reynolds_koopman as koopman
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-defect-subspace-rank.v1"
HYPOTHESIS_ID = "tinyllm-c2-defect-subspace-rank-v1"
REGIMES = ("composition", "extrapolation")
HELDOUT_COHORTS = ("heldout_a", "heldout_b")
PRIMARY_SEEDS = (7, 29, 53)
SOURCE_SEEDS = (7, 17, 29, 41, 53)
DEFAULT_RANKS = (1, 2, 4, 8, 16, 32, 64)


@dataclass(frozen=True)
class DefectSubspaceRankConfig:
    source_root: str = "data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered"
    character_root: str = "data/experiments/tinyllm_reynolds_character_coupling/20260806_d6_preregistered"
    seeds: tuple[int, ...] = PRIMARY_SEEDS
    orbit_count: int = 64
    first_blocks: int = 6
    ranks: tuple[int, ...] = DEFAULT_RANKS
    sufficient_fisher_threshold: float = 0.90
    complement_fisher_ceiling: float = 0.50
    task_effect_floor: float = 1e-6
    decomposition_tolerance: float = 1e-6
    orthogonality_tolerance: float = 1e-8
    numerical_rank_rtol: float = 1e-10
    activation_batch_size: int = 256
    continuation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("confirmatory seeds are fixed to 7,29,53")
        if self.orbit_count < 8:
            raise ValueError("at least eight exact orbits are required")
        if not self.ranks or any(rank < 1 for rank in self.ranks):
            raise ValueError("ranks must be positive")
        if tuple(sorted(set(self.ranks))) != self.ranks:
            raise ValueError("ranks must be unique and increasing")
        if self.ranks[0] != 1:
            raise ValueError("the preregistered titration must start at rank one")
        if not 0.0 < self.complement_fisher_ceiling < self.sufficient_fisher_threshold <= 1.0:
            raise ValueError("Fisher thresholds must order complement below sufficiency")
        if not 0.0 < self.numerical_rank_rtol < 1.0:
            raise ValueError("numerical rank tolerance must lie between zero and one")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(heads.__file__),
        Path(deck.__file__),
        Path(coupling.__file__),
        Path(koopman.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _bridge_config(config: DefectSubspaceRankConfig) -> deck.DeckDescramblerConfig:
    return deck.DeckDescramblerConfig(
        source_root=config.source_root,
        # The inherited loader validates the original five-seed source
        # campaign; the analysis itself remains restricted to PRIMARY_SEEDS.
        seeds=SOURCE_SEEDS if not config.allow_underpowered else config.seeds,
        degrees=(2,),
        fit_orbits=config.orbit_count,
        evaluation_orbits=config.orbit_count,
        map_points=24,
        first_blocks=config.first_blocks,
        activation_batch_size=config.activation_batch_size,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )


def fit_right_singular_basis(
    matrix: torch.Tensor, relative_tolerance: float = 1e-10
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the numerical row-space basis and normalized retained energy."""
    if matrix.ndim != 2:
        raise ValueError("matrix must have shape observations by features")
    _, singular, vh = torch.linalg.svd(matrix.double(), full_matrices=False)
    if singular.numel() == 0 or float(singular[0]) == 0.0:
        raise ValueError("matrix has no nonzero singular direction")
    numerical_rank = int((singular > singular[0] * relative_tolerance).sum().item())
    vh = vh[:numerical_rank]
    energy = singular[:numerical_rank].square()
    energy = energy / energy.sum().clamp_min(1e-24)
    return vh, energy


def project_onto_basis(matrix: torch.Tensor, basis: torch.Tensor, rank: int) -> torch.Tensor:
    """Project rows of matrix onto the first rank feature directions."""
    if matrix.ndim != 2 or basis.ndim != 2:
        raise ValueError("matrix and basis must be two-dimensional")
    effective = min(rank, basis.shape[0])
    selected = basis[:effective].to(device=matrix.device, dtype=matrix.dtype)
    return (matrix @ selected.T) @ selected


def _random_direction(features: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(features, generator=generator, dtype=torch.float64)
    value = value / torch.linalg.vector_norm(value).clamp_min(1e-24)
    return value.to(device)


def random_source_direction(basis: torch.Tensor, seed: int) -> torch.Tensor:
    """Draw a deterministic unit direction inside an orthonormal source span."""
    coefficients = _random_direction(basis.shape[0], seed, basis.device)
    value = coefficients.to(basis.dtype) @ basis
    return value / torch.linalg.vector_norm(value).clamp_min(1e-24)


def _load_character_source(
    config: DefectSubspaceRankConfig, seed: int, checkpoint_sha256: str
) -> tuple[dict[str, Any], Path]:
    path = Path(config.character_root) / "runs" / "k2" / f"seed_{seed}" / "result.json"
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != coupling.SCHEMA_VERSION
        or value.get("hypothesis_id") != coupling.HYPOTHESIS_ID
        or value.get("status") != "completed"
        or int(value.get("seed", -1)) != seed
        or int(value.get("k", -1)) != 2
        or value.get("provenance", {}).get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError(f"invalid character source {path}")
    return value, path


def _fingerprint(
    config: DefectSubspaceRankConfig,
    task: CircleTaskConfig,
    seed: int,
    checkpoint_sha256: str,
    character_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "task": asdict(task),
        "cohorts": heads.COHORTS,
        "seed": seed,
        "checkpoint_sha256": checkpoint_sha256,
        "character_sha256": character_sha256,
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode()).hexdigest()


@torch.no_grad()
def _defect_cell(
    system: Any,
    task: CircleTaskConfig,
    config: DefectSubspaceRankConfig,
    bridge: deck.DeckDescramblerConfig,
    frozen: Mapping[str, Any],
    seed: int,
    cohort: str,
    regime: str,
    device: torch.device,
) -> dict[str, Any]:
    regime_index = REGIMES.index(regime)
    offset, stride = heads.COHORTS[cohort]
    evaluation_seed = seed + offset + stride * regime_index
    transition_index = int(frozen["regimes"][regime]["synthesis_front_index"])
    if transition_index != 0:
        raise ValueError(f"seed {seed} is not a stable block-0 attention front")
    source_cut, target_cut = koopman.one_step_transitions(config.first_blocks)[transition_index]
    dataset = deck.generate_exact_orbits(
        task, k=2, orbit_count=config.orbit_count, seed=evaluation_seed, regime=regime
    )
    captured = deck.capture_sequences(system, dataset, bridge, device)
    source = torch.from_numpy(captured[source_cut]).to(device)
    source = source.reshape((config.orbit_count, 2) + source.shape[1:])
    block = system.model.transformer["h"][0]
    propagated, defects, exact, error = heads.exact_head_defects(block, source)
    full_defect = exact - propagated
    baseline_posterior = coupling._continue_tensor(
        system,
        "full",
        torch.from_numpy(captured["full"]).to(device),
        task,
        config.continuation_batch_size,
    )
    baseline = deck.output_diagnostics(
        baseline_posterior.double().cpu().numpy(), dataset, include_topology=False
    )
    return {
        "cohort": cohort,
        "regime": regime,
        "evaluation_seed": evaluation_seed,
        "target_cut": target_cut,
        "dataset": dataset,
        "baseline": baseline,
        "propagated": propagated,
        "exact": exact,
        "full_defect": full_defect,
        "decomposition_relative_error": error,
    }


def _rank_key(rank: int) -> str:
    return f"rank_{rank}"


@torch.no_grad()
def evaluate_heldout_cell(
    system: Any,
    task: CircleTaskConfig,
    config: DefectSubspaceRankConfig,
    cell: Mapping[str, Any],
    basis: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    defect = cell["full_defect"]
    propagated = cell["propagated"]
    exact = cell["exact"]
    shape = defect.shape
    flat = defect.reshape(config.orbit_count, -1)
    ranks = tuple(rank for rank in config.ranks if rank < basis.shape[0]) + (basis.shape[0],)
    ranks = tuple(dict.fromkeys(ranks))
    states: dict[str, torch.Tensor] = {"zero": propagated, "exact": exact}
    for rank in ranks:
        projected = project_onto_basis(flat, basis, rank).reshape(shape)
        states[_rank_key(rank)] = propagated + projected
    rank_one = project_onto_basis(flat, basis, 1).reshape(shape)
    states["rank_1_complement"] = propagated + (defect - rank_one)
    random = random_source_direction(basis, seed).to(flat.dtype)
    random_projection = ((flat @ random[:, None]) @ random[None, :]).reshape(shape)
    states["random_rank_1"] = propagated + random_projection

    keys = tuple(states)
    repeated = torch.cat([coupling._repeat_patch(states[key], 2) for key in keys])
    combined = coupling._continue_tensor(
        system,
        cell["target_cut"],
        repeated,
        task,
        config.continuation_batch_size,
    )
    per_state = config.orbit_count * 2
    posteriors = dict(zip(keys, combined.split(per_state)))
    zero_orbit = posteriors["zero"].reshape(config.orbit_count, 2, -1)[:, 0]
    exact_orbit = posteriors["exact"].reshape(config.orbit_count, 2, -1)[:, 0]
    records = {}
    for key, posterior in posteriors.items():
        diagnostics = coupling._diagnostics(posterior, cell["dataset"])
        causal_pass, _ = coupling._causal_classification(diagnostics, cell["baseline"], 2)
        orbit_posterior = posterior.reshape(config.orbit_count, 2, -1)[:, 0]
        effect = heads._effect_preservation(
            orbit_posterior, exact_orbit, zero_orbit, config.task_effect_floor
        )
        records[key] = {
            "causal_pass": causal_pass,
            "effect_preservation": effect,
            "diagnostics": diagnostics,
        }
    return {
        "cohort": cell["cohort"],
        "regime": cell["regime"],
        "evaluation_seed": cell["evaluation_seed"],
        "target_cut": cell["target_cut"],
        "decomposition_relative_error": cell["decomposition_relative_error"],
        "available_source_rank": int(basis.shape[0]),
        "interventions": records,
    }


def _is_sufficient(record: Mapping[str, Any], threshold: float) -> bool:
    effect = record["effect_preservation"]
    return bool(
        record["causal_pass"]
        and not effect["degenerate"]
        and effect["preserved_fraction"] is not None
        and effect["preserved_fraction"] >= threshold
    )


def select_minimum_rank(
    cells: Sequence[Mapping[str, Any]], ranks: Sequence[int], threshold: float
) -> Optional[int]:
    for rank in ranks:
        key = _rank_key(rank)
        if all(key in cell["interventions"] and _is_sufficient(cell["interventions"][key], threshold) for cell in cells):
            return int(rank)
    return None


def _seed_gates(
    cells: Sequence[Mapping[str, Any]], config: DefectSubspaceRankConfig
) -> dict[str, Any]:
    endpoint = all(
        not cell["interventions"]["zero"]["causal_pass"]
        and cell["interventions"]["exact"]["causal_pass"]
        and cell["decomposition_relative_error"] <= config.decomposition_tolerance
        for cell in cells
    )
    rank_one = all(
        _is_sufficient(cell["interventions"]["rank_1"], config.sufficient_fisher_threshold)
        for cell in cells
    )
    complement = all(
        not cell["interventions"]["rank_1_complement"]["effect_preservation"]["degenerate"]
        and cell["interventions"]["rank_1_complement"]["effect_preservation"]["preserved_fraction"]
        <= config.complement_fisher_ceiling
        for cell in cells
    )
    random_specificity = all(
        not _is_sufficient(cell["interventions"]["random_rank_1"], config.sufficient_fisher_threshold)
        for cell in cells
    )
    available = int(cells[0]["available_source_rank"])
    tested_ranks = tuple(rank for rank in config.ranks if rank < available) + (available,)
    tested_ranks = tuple(dict.fromkeys(tested_ranks))
    return {
        "exact_endpoint_replication": endpoint,
        "rank_one_sufficiency": rank_one,
        "rank_one_complement_weak": complement,
        "random_rank_one_specificity": random_specificity,
        "minimum_fixed_sufficient_rank": select_minimum_rank(
            cells, tested_ranks, config.sufficient_fisher_threshold
        ),
        "tested_ranks": list(tested_ranks),
    }


def analyze_cell(
    task: CircleTaskConfig,
    config: DefectSubspaceRankConfig,
    seed: int,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    bridge = _bridge_config(config)
    system, provenance = deck.load_source(task, bridge, 2, seed, device)
    frozen, frozen_path = _load_character_source(config, seed, provenance["checkpoint_sha256"])
    frozen_sha = _sha256(frozen_path)
    source_cells = [
        _defect_cell(system, task, config, bridge, frozen, seed, "source_selection", regime, device)
        for regime in REGIMES
    ]
    source_matrix = torch.cat(
        [cell["full_defect"].reshape(config.orbit_count, -1) for cell in source_cells], dim=0
    )
    basis, energy = fit_right_singular_basis(
        source_matrix, config.numerical_rank_rtol
    )
    gram = basis @ basis.T
    orthogonality_error = float(
        torch.max(torch.abs(gram - torch.eye(len(basis), dtype=gram.dtype, device=gram.device))).cpu()
    )
    heldout = []
    for cohort in HELDOUT_COHORTS:
        for regime in REGIMES:
            cell = _defect_cell(system, task, config, bridge, frozen, seed, cohort, regime, device)
            heldout.append(evaluate_heldout_cell(system, task, config, cell, basis, seed + 17011))
            print(f"seed {seed} {cohort} {regime} complete", flush=True)
    gates = _seed_gates(heldout, config)
    gates["basis_contract"] = bool(
        orthogonality_error <= config.orthogonality_tolerance
        and max(cell["decomposition_relative_error"] for cell in source_cells)
        <= config.decomposition_tolerance
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-defect-subspace-rank-seed{seed}",
        "status": "completed",
        "evidence_role": "systems_lifecycle_only_not_quality_evidence" if config.allow_underpowered else "preregistered_primary_evidence",
        "completed_at": _utc_now(),
        "seed": seed,
        "configuration": asdict(config),
        "scientific_fingerprint": _fingerprint(
            config, task, seed, provenance["checkpoint_sha256"], frozen_sha
        ),
        "implementation_sha256": _implementation_digest(),
        "provenance": {
            **provenance,
            "character_result": str(frozen_path),
            "character_result_sha256": frozen_sha,
        },
        "source_basis": {
            "candidate_rank": int(source_matrix.shape[0]),
            "rank": int(basis.shape[0]),
            "features": int(basis.shape[1]),
            "orthogonality_max_error": orthogonality_error,
            "singular_energy": [float(value.cpu()) for value in energy],
            "cumulative_energy": [float(value) for value in energy.cpu().cumsum(0)],
        },
        "heldout_cells": heldout,
        "gates": gates,
        "all_primary_gates_passed": all(
            gates[name]
            for name in (
                "basis_contract",
                "exact_endpoint_replication",
                "rank_one_sufficiency",
                "rank_one_complement_weak",
                "random_rank_one_specificity",
            )
        ),
        "analysis_seconds": time.perf_counter() - started,
    }
    _write_json(output / "runs" / f"seed_{seed}" / "result.json", result)
    return result


def aggregate(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gate_names = (
        "basis_contract",
        "exact_endpoint_replication",
        "rank_one_sufficiency",
        "rank_one_complement_weak",
        "random_rank_one_specificity",
    )
    counts = {name: sum(bool(run["gates"][name]) for run in runs) for name in gate_names}
    passes = {name: count == len(runs) for name, count in counts.items()}
    return {
        "primary_seed_count": len(runs),
        "gate_counts": counts,
        "gate_passes": passes,
        "confirmed": all(passes.values()),
        "per_seed": {
            str(run["seed"]): {
                "gates": run["gates"],
                "rank_one_source_energy": run["source_basis"]["cumulative_energy"][0],
            }
            for run in runs
        },
    }


def run_campaign(config: DefectSubspaceRankConfig, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    implementation = _implementation_digest()
    runs = []
    reused = 0
    for seed in config.seeds:
        path = output / "runs" / f"seed_{seed}" / "result.json"
        if path.is_file():
            existing = json.loads(path.read_text())
            checkpoint = Path(config.source_root) / "runs" / "k2" / f"seed_{seed}" / "model.pt"
            character = Path(config.character_root) / "runs" / "k2" / f"seed_{seed}" / "result.json"
            expected = _fingerprint(config, task, seed, _sha256(checkpoint), _sha256(character))
            if (
                existing.get("status") == "completed"
                and existing.get("schema_version") == SCHEMA_VERSION
                and existing.get("scientific_fingerprint") == expected
                and existing.get("implementation_sha256") == implementation
            ):
                runs.append(existing)
                reused += 1
                print(f"resuming {existing['experiment_id']}", flush=True)
                continue
        runs.append(analyze_cell(task, config, seed, output, device))
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": "systems_lifecycle_only_not_quality_evidence" if config.allow_underpowered else "preregistered_primary_evidence",
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "implementation_sha256": implementation,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        "summary": {
            "requested": len(config.seeds),
            "scheduled": len(config.seeds) - reused,
            "completed": len(runs),
            "failed": 0,
            "reused": reused,
            "retries": 0,
            "excluded": 0,
            "trained_models": 0,
            "fitted_predictive_observers": 0,
        },
        "aggregates": aggregate(runs),
        "results": [
            {
                "experiment_id": run["experiment_id"],
                "seed": run["seed"],
                "scientific_fingerprint": run["scientific_fingerprint"],
                "analysis_seconds": run["analysis_seconds"],
                "gates": run["gates"],
                "path": str(output / "runs" / f"seed_{run['seed']}" / "result.json"),
            }
            for run in runs
        ],
        "execution": {
            "runner": "single_process_sequential_frozen_checkpoint_analysis",
            "scheduler": None,
        },
        "method_boundaries": [
            "The source-cohort SVD is a geometric basis fit, not a predictive observer.",
            "Held-out projections use the exact held-out defect and test representational sufficiency, not independent computability.",
            "The basis is checkpoint-local and is not aligned across independently initialized seeds.",
            "All task effects remain conditioned on the frozen downstream decoder.",
        ],
    }
    _write_json(output / "campaign_results.json", campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=DefectSubspaceRankConfig.source_root)
    parser.add_argument("--character-root", default=DefectSubspaceRankConfig.character_root)
    parser.add_argument("--output", type=Path, default=Path("data/experiments/tinyllm_defect_subspace_rank/20260806_d6_preregistered"))
    parser.add_argument("--seeds", default="7,29,53")
    parser.add_argument("--orbit-count", type=int, default=64)
    parser.add_argument("--first-blocks", type=int, default=6)
    parser.add_argument("--ranks", default="1,2,4,8,16,32,64")
    parser.add_argument("--sufficient-fisher-threshold", type=float, default=0.90)
    parser.add_argument("--complement-fisher-ceiling", type=float, default=0.50)
    parser.add_argument("--task-effect-floor", type=float, default=1e-6)
    parser.add_argument("--decomposition-tolerance", type=float, default=1e-6)
    parser.add_argument("--orthogonality-tolerance", type=float, default=1e-8)
    parser.add_argument("--numerical-rank-rtol", type=float, default=1e-10)
    parser.add_argument("--activation-batch-size", type=int, default=256)
    parser.add_argument("--continuation-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args(argv)
    config = DefectSubspaceRankConfig(
        source_root=args.source_root,
        character_root=args.character_root,
        seeds=_ints(args.seeds),
        orbit_count=args.orbit_count,
        first_blocks=args.first_blocks,
        ranks=_ints(args.ranks),
        sufficient_fisher_threshold=args.sufficient_fisher_threshold,
        complement_fisher_ceiling=args.complement_fisher_ceiling,
        task_effect_floor=args.task_effect_floor,
        decomposition_tolerance=args.decomposition_tolerance,
        orthogonality_tolerance=args.orthogonality_tolerance,
        numerical_rank_rtol=args.numerical_rank_rtol,
        activation_batch_size=args.activation_batch_size,
        continuation_batch_size=args.continuation_batch_size,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
