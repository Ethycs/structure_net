#!/usr/bin/env python3
"""Test whether decoder-boundary normals compress the TinyLLM C2 defect."""

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
import experiments.structure_net.tinyllm_defect_subspace_rank as rank
import experiments.structure_net.tinyllm_io_correspondence as io_source
import experiments.structure_net.tinyllm_reynolds_character_coupling as coupling
import experiments.structure_net.tinyllm_reynolds_koopman as koopman
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-defect-boundary-basis.v1"
HYPOTHESIS_ID = "tinyllm-c2-defect-boundary-basis-v1"
PRIMARY_SEEDS = (7, 29, 53)
SOURCE_SEEDS = (7, 17, 29, 41, 53)
BASE_RANKS = {7: 1, 29: 4, 53: 2}
CORRECTION_COUNTS = (1, 2, 4)


@dataclass(frozen=True)
class DefectBoundaryBasisConfig:
    source_root: str = "data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered"
    character_root: str = "data/experiments/tinyllm_reynolds_character_coupling/20260806_d6_preregistered"
    predecessor_root: str = "data/experiments/tinyllm_defect_subspace_rank/20260806_d6_preregistered"
    seeds: tuple[int, ...] = PRIMARY_SEEDS
    orbit_count: int = 64
    first_blocks: int = 6
    correction_counts: tuple[int, ...] = CORRECTION_COUNTS
    sufficient_fisher_threshold: float = 0.90
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
        if any(seed not in BASE_RANKS for seed in self.seeds):
            raise ValueError("every seed must have a frozen predecessor base rank")
        if self.orbit_count < 8:
            raise ValueError("at least eight exact orbits are required")
        if self.correction_counts != tuple(sorted(set(self.correction_counts))):
            raise ValueError("correction counts must be unique and increasing")
        if not self.correction_counts or self.correction_counts[0] != 1:
            raise ValueError("the preregistered correction titration starts at one")
        if not 0.0 < self.sufficient_fisher_threshold <= 1.0:
            raise ValueError("Fisher threshold must lie in (0,1]")
        if not 0.0 < self.numerical_rank_rtol < 1.0:
            raise ValueError("numerical rank tolerance must lie in (0,1)")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _json_compatible(value: Any) -> Any:
    """Return the strict-JSON value used when comparing reloaded artifacts."""
    return json.loads(json.dumps(value, allow_nan=False))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().double().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(rank.__file__),
        Path(heads.__file__),
        Path(deck.__file__),
        Path(coupling.__file__),
        Path(koopman.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _bridge_config(config: DefectBoundaryBasisConfig) -> deck.DeckDescramblerConfig:
    return deck.DeckDescramblerConfig(
        source_root=config.source_root,
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


def _load_predecessor(
    config: DefectBoundaryBasisConfig, seed: int, checkpoint_sha256: str
) -> tuple[dict[str, Any], Path]:
    path = Path(config.predecessor_root) / "runs" / f"seed_{seed}" / "result.json"
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != rank.SCHEMA_VERSION
        or value.get("hypothesis_id") != rank.HYPOTHESIS_ID
        or value.get("status") != "completed"
        or int(value.get("seed", -1)) != seed
        or value.get("provenance", {}).get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError(f"invalid rank predecessor {path}")
    return value, path


def _fingerprint(
    config: DefectBoundaryBasisConfig,
    task: CircleTaskConfig,
    seed: int,
    checkpoint_sha256: str,
    predecessor_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "task": asdict(task),
        "cohorts": heads.COHORTS,
        "base_ranks": BASE_RANKS,
        "seed": seed,
        "checkpoint_sha256": checkpoint_sha256,
        "predecessor_sha256": predecessor_sha256,
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode()).hexdigest()


def _continue_logits(
    system: Any,
    cut: str,
    values: torch.Tensor,
    task: CircleTaskConfig,
) -> torch.Tensor:
    """Differentiably continue residual states to the answer-token logits."""
    value = values
    if cut == "full":
        next_block = len(system.model.transformer["h"])
    else:
        pieces = cut.split("_")
        block_index = int(pieces[1])
        if cut.endswith("post_attention"):
            block = system.model.transformer["h"][block_index]
            value = value + block.mlp(block.ln_2(value))
        next_block = block_index + 1
    for block in system.model.transformer["h"][next_block:]:
        value = block(value)
    answer = torch.tensor(task.answer_token_ids, device=value.device)
    return io_source._task_logits(system.model, value[:, -1, :], answer)


def select_winner_and_challenger(
    exact_logits: torch.Tensor, base_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the exact winner and the strongest different class at the base."""
    if exact_logits.shape != base_logits.shape or exact_logits.ndim != 2:
        raise ValueError("logit matrices must have identical batch-by-class shape")
    winners = exact_logits.argmax(-1)
    masked = base_logits.detach().clone()
    masked.scatter_(1, winners[:, None], -torch.inf)
    challengers = masked.argmax(-1)
    return winners, challengers


def project_residual_onto_normals(
    residual: torch.Tensor, normals: torch.Tensor, epsilon: float = 1e-12
) -> torch.Tensor:
    """Return each residual's signed Euclidean projection onto its normal."""
    if residual.shape != normals.shape or residual.ndim != 2:
        raise ValueError("residual and normals must have identical row shape")
    denominator = normals.square().sum(-1, keepdim=True).clamp_min(epsilon)
    coefficient = (residual * normals).sum(-1, keepdim=True) / denominator
    return coefficient * normals


def residualize_rows(rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Remove components in an orthonormal row basis."""
    if rows.ndim != 2 or basis.ndim != 2 or rows.shape[1] != basis.shape[1]:
        raise ValueError("rows and basis must share their feature dimension")
    return rows - (rows @ basis.T) @ basis


def combine_bases(base: torch.Tensor, correction: torch.Tensor, count: int) -> torch.Tensor:
    """Concatenate nested orthonormal base and correction rows."""
    if count < 1 or count > len(correction):
        raise ValueError("requested correction rank is unavailable")
    result = torch.cat((base, correction[:count]))
    gram = result @ result.T
    error = torch.max(
        torch.abs(gram - torch.eye(len(result), dtype=gram.dtype, device=gram.device))
    )
    if float(error) > 1e-7:
        raise ValueError(f"combined basis is not orthonormal: {float(error):.3e}")
    return result


def random_residual_basis(
    geometric_basis: torch.Tensor, base_rank: int, count: int, seed: int
) -> torch.Tensor:
    """Draw deterministic orthonormal rows inside the remaining source span."""
    residual_basis = geometric_basis[base_rank:]
    if count > len(residual_basis):
        raise ValueError("random correction rank exceeds the residual source span")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    coefficients = torch.randn(
        len(residual_basis), count, generator=generator, dtype=torch.float64
    ).to(residual_basis.device)
    q, _ = torch.linalg.qr(coefficients, mode="reduced")
    return q.T.to(residual_basis.dtype) @ residual_basis


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().double().cpu().flatten()
    return {
        "minimum": float(flat.min()),
        "q10": float(torch.quantile(flat, 0.10)),
        "median": float(torch.quantile(flat, 0.50)),
        "q90": float(torch.quantile(flat, 0.90)),
        "maximum": float(flat.max()),
        "mean": float(flat.mean()),
    }


def _boundary_components(
    system: Any,
    task: CircleTaskConfig,
    cell: Mapping[str, Any],
    geometric_basis: torch.Tensor,
    base_rank: int,
) -> dict[str, Any]:
    defect = cell["full_defect"]
    shape = defect.shape
    flat = defect.reshape(len(defect), -1)
    base_basis = geometric_basis[:base_rank].to(flat.dtype)
    projected = rank.project_onto_basis(flat, base_basis, base_rank).reshape(shape)
    base_state = (cell["propagated"] + projected).detach().requires_grad_(True)
    with torch.no_grad():
        exact_logits = _continue_logits(system, cell["target_cut"], cell["exact"], task)
    base_logits = _continue_logits(system, cell["target_cut"], base_state, task)
    winners, challengers = select_winner_and_challenger(exact_logits, base_logits)
    rows = torch.arange(len(base_logits), device=base_logits.device)
    base_margin = base_logits[rows, winners] - base_logits[rows, challengers]
    normals = torch.autograd.grad(base_margin.sum(), base_state)[0].detach().reshape(len(flat), -1)
    residual = residualize_rows(flat, base_basis)
    paired = project_residual_onto_normals(residual, normals)
    paired = residualize_rows(paired, base_basis)
    with torch.no_grad():
        exact_margin = exact_logits[rows, winners] - exact_logits[rows, challengers]
    normal_norm = torch.linalg.vector_norm(normals, dim=-1)
    residual_norm = torch.linalg.vector_norm(residual, dim=-1)
    signed = (residual * normals).sum(-1)
    cosine = signed / (normal_norm * residual_norm).clamp_min(1e-12)
    return {
        "residual": residual.double(),
        "normals": normals.double(),
        "paired": paired.double(),
        "summary": {
            "base_margin": _quantiles(base_margin),
            "exact_margin": _quantiles(exact_margin),
            "normal_norm": _quantiles(normal_norm),
            "residual_normal_cosine": _quantiles(cosine),
            "signed_linear_margin_contribution": _quantiles(signed),
            "winner_changed_from_base_to_exact": float(
                (base_logits.argmax(-1) != winners).double().mean().detach().cpu()
            ),
        },
    }


def _posterior_summary(posterior: torch.Tensor, exact: torch.Tensor) -> dict[str, Any]:
    top_two = posterior.topk(2, dim=-1).values
    margin = top_two[:, 0] - top_two[:, 1]
    return {
        "exact_winner_agreement": float(
            (posterior.argmax(-1) == exact.argmax(-1)).double().mean().cpu()
        ),
        "top_two_probability_margin": _quantiles(margin),
    }


@torch.no_grad()
def evaluate_heldout_cell(
    system: Any,
    task: CircleTaskConfig,
    config: DefectBoundaryBasisConfig,
    cell: Mapping[str, Any],
    bases: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    defect = cell["full_defect"]
    flat = defect.reshape(config.orbit_count, -1)
    states: dict[str, torch.Tensor] = {
        "zero": cell["propagated"],
        "exact": cell["exact"],
    }
    for name, basis in bases.items():
        projected = rank.project_onto_basis(flat, basis, len(basis)).reshape_as(defect)
        states[name] = cell["propagated"] + projected

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
    orbit_posteriors = {
        key: value.reshape(config.orbit_count, 2, -1)[:, 0]
        for key, value in posteriors.items()
    }
    zero = orbit_posteriors["zero"]
    exact = orbit_posteriors["exact"]
    records = {}
    for key, posterior in posteriors.items():
        diagnostics = coupling._diagnostics(posterior, cell["dataset"])
        causal_pass, _ = coupling._causal_classification(diagnostics, cell["baseline"], 2)
        orbit = orbit_posteriors[key]
        records[key] = {
            "causal_pass": causal_pass,
            "effect_preservation": heads._effect_preservation(
                orbit, exact, zero, config.task_effect_floor
            ),
            "diagnostics": diagnostics,
            **_posterior_summary(orbit, exact),
        }
    return {
        "cohort": cell["cohort"],
        "regime": cell["regime"],
        "evaluation_seed": cell["evaluation_seed"],
        "target_cut": cell["target_cut"],
        "decomposition_relative_error": cell["decomposition_relative_error"],
        "interventions": records,
    }


def _is_sufficient(record: Mapping[str, Any], threshold: float) -> bool:
    return rank._is_sufficient(record, threshold)


def minimum_fixed_correction(
    cells: Sequence[Mapping[str, Any]],
    family: str,
    counts: Sequence[int],
    threshold: float,
) -> Optional[int]:
    for count in counts:
        key = f"{family}_plus_{count}"
        if all(
            key in cell["interventions"]
            and _is_sufficient(cell["interventions"][key], threshold)
            for cell in cells
        ):
            return int(count)
    return None


def _failure_cells(
    cells: Sequence[Mapping[str, Any]], key: str, threshold: float
) -> list[str]:
    return sorted(
        f"{cell['cohort']}:{cell['regime']}"
        for cell in cells
        if not _is_sufficient(cell["interventions"][key], threshold)
    )


def _predecessor_failure_cells(
    predecessor: Mapping[str, Any], base_rank: int, threshold: float
) -> list[str]:
    key = f"rank_{base_rank}"
    return sorted(
        f"{cell['cohort']}:{cell['regime']}"
        for cell in predecessor["heldout_cells"]
        if not _is_sufficient(cell["interventions"][key], threshold)
    )


def _seed_gates(
    cells: Sequence[Mapping[str, Any]],
    predecessor_failures: Sequence[str],
    basis_contract: bool,
    config: DefectBoundaryBasisConfig,
) -> dict[str, Any]:
    threshold = config.sufficient_fisher_threshold
    endpoint = all(
        not cell["interventions"]["zero"]["causal_pass"]
        and cell["interventions"]["exact"]["causal_pass"]
        and cell["decomposition_relative_error"] <= config.decomposition_tolerance
        for cell in cells
    )
    boundary_one = all(
        "boundary_plus_1" in cell["interventions"]
        and _is_sufficient(cell["interventions"]["boundary_plus_1"], threshold)
        for cell in cells
    )
    indexed = {f"{cell['cohort']}:{cell['regime']}": cell for cell in cells}
    shuffled_specific = any(
        not _is_sufficient(indexed[name]["interventions"]["shuffled_plus_1"], threshold)
        for name in predecessor_failures
    )
    random_specific = any(
        not _is_sufficient(indexed[name]["interventions"]["random_plus_1"], threshold)
        for name in predecessor_failures
    )
    base_failures = _failure_cells(cells, "base_geometric", threshold)
    return {
        "basis_contract": basis_contract,
        "exact_endpoint_replication": endpoint,
        "predecessor_failure_replication": base_failures == list(predecessor_failures),
        "boundary_rank_one_sufficiency": boundary_one,
        "shuffled_and_random_specificity": shuffled_specific and random_specific,
        "declared_predecessor_failure_cells": list(predecessor_failures),
        "observed_base_failure_cells": base_failures,
        "minimum_boundary_correction": minimum_fixed_correction(
            cells, "boundary", config.correction_counts, threshold
        ),
        "minimum_geometric_correction": minimum_fixed_correction(
            cells, "geometric", config.correction_counts, threshold
        ),
        "minimum_shuffled_correction": minimum_fixed_correction(
            cells, "shuffled", config.correction_counts, threshold
        ),
        "minimum_random_correction": minimum_fixed_correction(
            cells, "random", config.correction_counts, threshold
        ),
    }


def analyze_seed(
    task: CircleTaskConfig,
    config: DefectBoundaryBasisConfig,
    seed: int,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    bridge = _bridge_config(config)
    system, provenance = deck.load_source(task, bridge, 2, seed, device)
    system.model.eval()
    for parameter in system.model.parameters():
        parameter.requires_grad_(False)
    predecessor, predecessor_path = _load_predecessor(
        config, seed, provenance["checkpoint_sha256"]
    )
    predecessor_sha = _sha256(predecessor_path)
    character, _ = rank._load_character_source(
        rank.DefectSubspaceRankConfig(
            source_root=config.source_root,
            character_root=config.character_root,
            seeds=(seed,),
            orbit_count=config.orbit_count,
            first_blocks=config.first_blocks,
            device=config.device,
            allow_underpowered=True,
        ),
        seed,
        provenance["checkpoint_sha256"],
    )
    base_rank = BASE_RANKS[seed]
    source_cells = [
        rank._defect_cell(
            system,
            task,
            rank.DefectSubspaceRankConfig(
                source_root=config.source_root,
                character_root=config.character_root,
                seeds=(seed,),
                orbit_count=config.orbit_count,
                first_blocks=config.first_blocks,
                activation_batch_size=config.activation_batch_size,
                continuation_batch_size=config.continuation_batch_size,
                device=config.device,
                allow_underpowered=True,
            ),
            bridge,
            character,
            seed,
            "source_selection",
            regime,
            device,
        )
        for regime in rank.REGIMES
    ]
    source_matrix = torch.cat(
        [cell["full_defect"].reshape(config.orbit_count, -1) for cell in source_cells]
    )
    geometric_basis, geometric_energy = rank.fit_right_singular_basis(
        source_matrix, config.numerical_rank_rtol
    )
    base_basis = geometric_basis[:base_rank]

    components = [
        _boundary_components(system, task, cell, geometric_basis, base_rank)
        for cell in source_cells
    ]
    residual = torch.cat([item["residual"] for item in components])
    normals = torch.cat([item["normals"] for item in components])
    paired = torch.cat([item["paired"] for item in components])
    paired = residualize_rows(paired, base_basis)
    boundary_basis, boundary_energy = rank.fit_right_singular_basis(
        paired, config.numerical_rank_rtol
    )

    generator = torch.Generator(device="cpu").manual_seed(seed + 23003)
    permutation = torch.randperm(len(normals), generator=generator).to(normals.device)
    shuffled = project_residual_onto_normals(residual, normals[permutation])
    shuffled = residualize_rows(shuffled, base_basis)
    shuffled_basis, shuffled_energy = rank.fit_right_singular_basis(
        shuffled, config.numerical_rank_rtol
    )
    maximum_correction = max(config.correction_counts)
    random_basis = random_residual_basis(
        geometric_basis, base_rank, maximum_correction, seed + 31013
    )

    bases: dict[str, torch.Tensor] = {"base_geometric": base_basis}
    for count in config.correction_counts:
        bases[f"geometric_plus_{count}"] = geometric_basis[: base_rank + count]
        if count <= len(boundary_basis):
            bases[f"boundary_plus_{count}"] = combine_bases(
                base_basis, boundary_basis, count
            )
        if count <= len(shuffled_basis):
            bases[f"shuffled_plus_{count}"] = combine_bases(
                base_basis, shuffled_basis, count
            )
        bases[f"random_plus_{count}"] = combine_bases(base_basis, random_basis, count)

    rank_config = rank.DefectSubspaceRankConfig(
        source_root=config.source_root,
        character_root=config.character_root,
        seeds=(seed,),
        orbit_count=config.orbit_count,
        first_blocks=config.first_blocks,
        activation_batch_size=config.activation_batch_size,
        continuation_batch_size=config.continuation_batch_size,
        device=config.device,
        allow_underpowered=True,
    )
    heldout = []
    for cohort in rank.HELDOUT_COHORTS:
        for regime in rank.REGIMES:
            cell = rank._defect_cell(
                system,
                task,
                rank_config,
                bridge,
                character,
                seed,
                cohort,
                regime,
                device,
            )
            heldout.append(evaluate_heldout_cell(system, task, config, cell, bases))
            print(f"seed {seed} {cohort} {regime} complete", flush=True)

    basis_errors = {}
    for name, basis in bases.items():
        gram = basis @ basis.T
        basis_errors[name] = float(
            torch.max(
                torch.abs(
                    gram
                    - torch.eye(len(basis), dtype=gram.dtype, device=gram.device)
                )
            ).cpu()
        )
    inherited_energy = predecessor["source_basis"]["singular_energy"]
    energy_error = max(
        abs(float(left.cpu()) - float(right))
        for left, right in zip(geometric_energy, inherited_energy)
    )
    basis_contract = bool(
        max(basis_errors.values()) <= config.orthogonality_tolerance
        and energy_error <= 1e-10
        and max(cell["decomposition_relative_error"] for cell in source_cells)
        <= config.decomposition_tolerance
        and all(
            not record["effect_preservation"]["degenerate"]
            for cell in heldout
            for record in cell["interventions"].values()
        )
    )
    predecessor_failures = _predecessor_failure_cells(
        predecessor, base_rank, config.sufficient_fisher_threshold
    )
    gates = _seed_gates(
        heldout, predecessor_failures, basis_contract, config
    )
    primary_names = (
        "basis_contract",
        "exact_endpoint_replication",
        "predecessor_failure_replication",
        "boundary_rank_one_sufficiency",
        "shuffled_and_random_specificity",
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-defect-boundary-basis-seed{seed}",
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else "preregistered_underpowered_mechanistic_evidence"
        ),
        "completed_at": _utc_now(),
        "seed": seed,
        "configuration": asdict(config),
        "scientific_fingerprint": _fingerprint(
            config,
            task,
            seed,
            provenance["checkpoint_sha256"],
            predecessor_sha,
        ),
        "implementation_sha256": _implementation_digest(),
        "provenance": {
            **provenance,
            "predecessor_result": str(predecessor_path),
            "predecessor_sha256": predecessor_sha,
        },
        "source_basis": {
            "features": int(geometric_basis.shape[1]),
            "geometric_rank": int(len(geometric_basis)),
            "base_rank": base_rank,
            "boundary_rank": int(len(boundary_basis)),
            "shuffled_rank": int(len(shuffled_basis)),
            "geometric_energy_max_error_from_predecessor": energy_error,
            "boundary_singular_energy": [float(value.cpu()) for value in boundary_energy],
            "boundary_cumulative_energy": [
                float(value) for value in boundary_energy.cpu().cumsum(0)
            ],
            "shuffled_singular_energy": [float(value.cpu()) for value in shuffled_energy],
            "basis_orthogonality_max_errors": basis_errors,
            "basis_sha256": {
                "geometric": _tensor_sha256(geometric_basis),
                "boundary": _tensor_sha256(boundary_basis),
                "shuffled": _tensor_sha256(shuffled_basis),
                "random": _tensor_sha256(random_basis),
            },
            "source_boundary_cells": [
                {"regime": regime, **item["summary"]}
                for regime, item in zip(rank.REGIMES, components)
            ],
        },
        "heldout_cells": heldout,
        "gates": gates,
        "all_primary_gates_passed": all(bool(gates[name]) for name in primary_names),
        "analysis_seconds": time.perf_counter() - started,
    }
    _write_json(output / "runs" / f"seed_{seed}" / "result.json", result)
    return result


def aggregate(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gate_names = (
        "basis_contract",
        "exact_endpoint_replication",
        "predecessor_failure_replication",
        "boundary_rank_one_sufficiency",
        "shuffled_and_random_specificity",
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
                "base_rank": run["source_basis"]["base_rank"],
                "boundary_cumulative_energy": run["source_basis"][
                    "boundary_cumulative_energy"
                ],
                "gates": run["gates"],
            }
            for run in runs
        },
    }


def run_campaign(
    config: DefectBoundaryBasisConfig, output: Path
) -> dict[str, Any]:
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
            predecessor = Path(config.predecessor_root) / "runs" / f"seed_{seed}" / "result.json"
            expected = _fingerprint(
                config, task, seed, _sha256(checkpoint), _sha256(predecessor)
            )
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
        runs.append(analyze_seed(task, config, seed, output, device))

    campaign_path = output / "campaign_results.json"
    if reused == len(config.seeds) and campaign_path.is_file():
        existing_campaign = json.loads(campaign_path.read_text())
        expected_fingerprints = [run["scientific_fingerprint"] for run in runs]
        observed_fingerprints = [
            item["scientific_fingerprint"] for item in existing_campaign.get("results", [])
        ]
        if (
            existing_campaign.get("schema_version") == SCHEMA_VERSION
            and existing_campaign.get("implementation_sha256") == implementation
            and existing_campaign.get("configuration")
            == _json_compatible(asdict(config))
            and observed_fingerprints == expected_fingerprints
        ):
            print("aggregate already complete; leaving bytes unchanged", flush=True)
            return existing_campaign

    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else "preregistered_underpowered_mechanistic_evidence"
        ),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "implementation_sha256": implementation,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
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
            "The source-only basis is conditioned on frozen decoder top-two margin normals.",
            "No target label or held-out gradient enters basis construction.",
            "Held-out patches use the exact held-out defect and test representational sufficiency, not independent computability.",
            "The first-order basis may miss decoder curvature between base and exact states.",
            "Bases are checkpoint-local and all effects remain conditioned on the frozen decoder.",
        ],
    }
    _write_json(campaign_path, campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=DefectBoundaryBasisConfig.source_root)
    parser.add_argument("--character-root", default=DefectBoundaryBasisConfig.character_root)
    parser.add_argument("--predecessor-root", default=DefectBoundaryBasisConfig.predecessor_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_defect_boundary_basis/20260806_d6_preregistered"
        ),
    )
    parser.add_argument("--seeds", default="7,29,53")
    parser.add_argument("--orbit-count", type=int, default=64)
    parser.add_argument("--first-blocks", type=int, default=6)
    parser.add_argument("--correction-counts", default="1,2,4")
    parser.add_argument("--sufficient-fisher-threshold", type=float, default=0.90)
    parser.add_argument("--task-effect-floor", type=float, default=1e-6)
    parser.add_argument("--decomposition-tolerance", type=float, default=1e-6)
    parser.add_argument("--orthogonality-tolerance", type=float, default=1e-8)
    parser.add_argument("--numerical-rank-rtol", type=float, default=1e-10)
    parser.add_argument("--activation-batch-size", type=int, default=256)
    parser.add_argument("--continuation-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args(argv)
    config = DefectBoundaryBasisConfig(
        source_root=args.source_root,
        character_root=args.character_root,
        predecessor_root=args.predecessor_root,
        seeds=_ints(args.seeds),
        orbit_count=args.orbit_count,
        first_blocks=args.first_blocks,
        correction_counts=_ints(args.correction_counts),
        sufficient_fisher_threshold=args.sufficient_fisher_threshold,
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
