#!/usr/bin/env python3
"""Exactly decompose C2 quotient synthesis into TinyLLM attention heads."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.nn import functional as F

import experiments.structure_net.tinyllm_deck_action_descrambler as deck
import experiments.structure_net.tinyllm_reynolds_character_coupling as coupling
import experiments.structure_net.tinyllm_reynolds_koopman as koopman
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-c2-attention-head-decomposition.v1"
HYPOTHESIS_ID = "tinyllm-c2-sparse-attention-head-synthesis-v1"
REGIMES = ("composition", "extrapolation")
COHORTS = {
    "source_selection": (2501, 101),
    "heldout_a": (4501, 101),
    "heldout_b": (9101, 307),
}
PRIMARY_SEEDS = (7, 29, 53)


@dataclass(frozen=True)
class HeadDecompositionConfig:
    source_root: str = "data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered"
    character_root: str = "data/experiments/tinyllm_reynolds_character_coupling/20260806_d6_preregistered"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    primary_seeds: tuple[int, ...] = PRIMARY_SEEDS
    orbit_count: int = 64
    first_blocks: int = 6
    sparse_head_ceiling: int = 2
    sufficient_fisher_threshold: float = 0.90
    complement_fisher_ceiling: float = 0.50
    specificity_margin: float = 0.20
    task_effect_floor: float = 1e-6
    decomposition_tolerance: float = 1e-6
    activation_batch_size: int = 256
    continuation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if len(self.seeds) < 5 and not self.allow_underpowered:
            raise ValueError("confirmatory campaign requires five seeds")
        if not set(self.primary_seeds).issubset(self.seeds):
            raise ValueError("primary seeds must be included in seeds")
        if tuple(self.primary_seeds) != PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("confirmatory primary cohort is fixed to seeds 7,29,53")
        if self.orbit_count < 8:
            raise ValueError("at least eight exact orbits are required")
        if self.first_blocks < 1:
            raise ValueError("at least one transformer block is required")
        if not 1 <= self.sparse_head_ceiling <= 6:
            raise ValueError("sparse head ceiling must be between one and six")
        if not 0.0 < self.complement_fisher_ceiling < self.sufficient_fisher_threshold <= 1.0:
            raise ValueError("Fisher thresholds must order complement below sufficiency")
        if self.specificity_margin < 0.0:
            raise ValueError("specificity margin must be nonnegative")


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
    for path in (Path(__file__), Path(deck.__file__), Path(coupling.__file__), Path(koopman.__file__)):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _bridge_config(config: HeadDecompositionConfig) -> deck.DeckDescramblerConfig:
    return deck.DeckDescramblerConfig(
        source_root=config.source_root,
        seeds=config.seeds,
        degrees=(2,),
        fit_orbits=config.orbit_count,
        evaluation_orbits=config.orbit_count,
        map_points=24,
        first_blocks=config.first_blocks,
        activation_batch_size=config.activation_batch_size,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )


def _load_character_source(
    config: HeadDecompositionConfig,
    seed: int,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], Path]:
    path = Path(config.character_root) / "runs" / "k2" / f"seed_{seed}" / "result.json"
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != coupling.SCHEMA_VERSION
        or value.get("hypothesis_id") != coupling.HYPOTHESIS_ID
        or value.get("status") != "completed"
        or int(value.get("k", -1)) != 2
        or int(value.get("seed", -1)) != seed
        or value.get("provenance", {}).get("checkpoint_sha256") != checkpoint_sha256
        or int(value.get("configuration", {}).get("first_blocks", -1)) != config.first_blocks
    ):
        raise ValueError(f"invalid character-coupling source {path}")
    return value, path


def _scientific_fingerprint(
    config: HeadDecompositionConfig,
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
        "cohorts": COHORTS,
        "seed": seed,
        "checkpoint_sha256": checkpoint_sha256,
        "character_sha256": character_sha256,
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode()).hexdigest()


@torch.no_grad()
def projected_attention_heads(block: Any, values: torch.Tensor) -> torch.Tensor:
    """Return bias-free projected head outputs with shape B,H,T,C."""
    normalized = block.ln_1(values)
    attention = block.attn
    batch, sequence, channels = normalized.shape
    query, key, content = attention.c_attn(normalized).split(attention.n_embd, dim=-1)
    head_count = int(attention.n_head)
    head_width = channels // head_count

    def split_heads(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(batch, sequence, head_count, head_width).transpose(1, 2)

    query, key, content = map(split_heads, (query, key, content))
    attended = F.scaled_dot_product_attention(
        query, key, content, dropout_p=0.0, is_causal=True
    )
    contributions = []
    for head in range(head_count):
        start, stop = head * head_width, (head + 1) * head_width
        contributions.append(
            F.linear(attended[:, head], attention.c_proj.weight[:, start:stop], bias=None)
        )
    return torch.stack(contributions, dim=1)


@torch.no_grad()
def exact_head_defects(
    block: Any,
    sheets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return F(b), per-head Reynolds defects, exact mean F(h), and closure error."""
    orbit_count, degree = sheets.shape[:2]
    barycenter = sheets.mean(1)
    propagated = barycenter + block.attn(block.ln_1(barycenter))
    exact_members = sheets.flatten(0, 1)
    exact = exact_members + block.attn(block.ln_1(exact_members))
    exact = exact.reshape_as(sheets).mean(1)
    member_heads = projected_attention_heads(block, exact_members)
    member_heads = member_heads.reshape(
        (orbit_count, degree) + member_heads.shape[1:]
    ).mean(1)
    barycenter_heads = projected_attention_heads(block, barycenter)
    defects = member_heads - barycenter_heads
    reconstruction = propagated + defects.sum(1)
    denominator = torch.linalg.vector_norm(exact - propagated).clamp_min(1e-12)
    error = float(torch.linalg.vector_norm(reconstruction - exact).cpu() / denominator.cpu())
    return propagated, defects, exact, error


def subset_heads(mask: int, head_count: int) -> tuple[int, ...]:
    return tuple(head for head in range(head_count) if mask & (1 << head))


def subset_states(
    propagated: torch.Tensor,
    defects: torch.Tensor,
) -> dict[int, torch.Tensor]:
    head_count = defects.shape[1]
    states = {}
    for mask in range(1 << head_count):
        heads = subset_heads(mask, head_count)
        states[mask] = (
            propagated
            if not heads
            else propagated + defects[:, heads].sum(1)
        )
    return states


def _fisher_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double()
    right = right.double()
    left = left / left.sum(-1, keepdim=True).clamp_min(1e-15)
    right = right / right.sum(-1, keepdim=True).clamp_min(1e-15)
    return float(coupling.fisher_rao_squared(left, right).mean().cpu())


def _effect_preservation(
    posterior: torch.Tensor,
    exact: torch.Tensor,
    propagated: torch.Tensor,
    floor: float,
) -> dict[str, Any]:
    denominator = _fisher_distance(propagated, exact)
    remaining = _fisher_distance(posterior, exact)
    degenerate = denominator < floor
    return {
        "exact_fisher_effect": denominator,
        "remaining_fisher_error": remaining,
        "preserved_fraction": None if degenerate else 1.0 - remaining / denominator,
        "degenerate": degenerate,
    }


def shapley_values(values: Mapping[int, float], head_count: int) -> list[float]:
    """Exact Shapley values for a complete subset-value map."""
    denominator = math.factorial(head_count)
    output = []
    for head in range(head_count):
        contribution = 0.0
        bit = 1 << head
        for mask in range(1 << head_count):
            if mask & bit:
                continue
            size = int(mask.bit_count())
            weight = (
                math.factorial(size)
                * math.factorial(head_count - size - 1)
                / denominator
            )
            contribution += weight * (values[mask | bit] - values[mask])
        output.append(contribution)
    return output


@torch.no_grad()
def _continue_subsets(
    system: Any,
    target_cut: str,
    states: Mapping[int, torch.Tensor],
    task: CircleTaskConfig,
    config: HeadDecompositionConfig,
) -> dict[int, torch.Tensor]:
    masks = tuple(states)
    per_mask = len(next(iter(states.values()))) * 2
    combined = torch.cat(
        [coupling._repeat_patch(states[mask], 2) for mask in masks], dim=0
    )
    posterior = coupling._continue_tensor(
        system, target_cut, combined, task, config.continuation_batch_size
    )
    return dict(zip(masks, posterior.split(per_mask)))


@torch.no_grad()
def analyze_cohort_regime(
    system: Any,
    task: CircleTaskConfig,
    config: HeadDecompositionConfig,
    bridge: deck.DeckDescramblerConfig,
    frozen: Mapping[str, Any],
    seed: int,
    cohort: str,
    regime: str,
    device: torch.device,
) -> dict[str, Any]:
    regime_index = REGIMES.index(regime)
    offset, stride = COHORTS[cohort]
    evaluation_seed = seed + offset + stride * regime_index
    front = frozen["regimes"][regime]["synthesis_front_index"]
    if front is None:
        raise ValueError(f"missing frozen synthesis front for seed {seed} {regime}")
    transition_index = int(front)
    if transition_index % 2:
        raise ValueError("degree-two frozen synthesis front is not attention")
    transitions = koopman.one_step_transitions(config.first_blocks)
    source_cut, target_cut = transitions[transition_index]
    block = system.model.transformer["h"][transition_index // 2]

    dataset = deck.generate_exact_orbits(
        task,
        k=2,
        orbit_count=config.orbit_count,
        seed=evaluation_seed,
        regime=regime,
    )
    captured = deck.capture_sequences(system, dataset, bridge, device)
    source = torch.from_numpy(captured[source_cut]).to(device)
    source = source.reshape((config.orbit_count, 2) + source.shape[1:])
    propagated, defects, exact, decomposition_error = exact_head_defects(block, source)
    states = subset_states(propagated, defects)
    posteriors = _continue_subsets(system, target_cut, states, task, config)

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
    head_count = int(defects.shape[1])
    full_mask = (1 << head_count) - 1
    propagated_orbit = posteriors[0].reshape(config.orbit_count, 2, -1)[:, 0]
    exact_orbit = posteriors[full_mask].reshape(config.orbit_count, 2, -1)[:, 0]
    subset_records = {}
    values = {}
    for mask, posterior in posteriors.items():
        diagnostics = coupling._diagnostics(posterior, dataset)
        causal_pass, _ = coupling._causal_classification(diagnostics, baseline, 2)
        orbit_posterior = posterior.reshape(config.orbit_count, 2, -1)[:, 0]
        effect = _effect_preservation(
            orbit_posterior,
            exact_orbit,
            propagated_orbit,
            config.task_effect_floor,
        )
        value = (
            -1e9
            if effect["preserved_fraction"] is None
            else float(effect["preserved_fraction"])
        )
        values[mask] = value
        subset_records[str(mask)] = {
            "mask": mask,
            "heads": list(subset_heads(mask, head_count)),
            "cardinality": int(mask.bit_count()),
            "causal_pass": causal_pass,
            "effect_preservation": effect,
            "diagnostics": diagnostics,
        }
    zero_pass = bool(subset_records["0"]["causal_pass"])
    exact_pass = bool(subset_records[str(full_mask)]["causal_pass"])
    effect_degenerate = bool(
        subset_records[str(full_mask)]["effect_preservation"]["degenerate"]
    )
    defect_norms = torch.linalg.vector_norm(
        defects.double().reshape(config.orbit_count, head_count, -1), dim=(0, 2)
    )
    total_norm = float(defect_norms.sum().cpu())
    return {
        "cohort": cohort,
        "regime": regime,
        "evaluation_seed": evaluation_seed,
        "transition_index": transition_index,
        "source_cut": source_cut,
        "target_cut": target_cut,
        "baseline": baseline,
        "head_count": head_count,
        "decomposition_relative_error": decomposition_error,
        "endpoint_replication": bool(not zero_pass and exact_pass and not effect_degenerate),
        "zero_causal_pass": zero_pass,
        "exact_causal_pass": exact_pass,
        "effect_degenerate": effect_degenerate,
        "per_head_defect_norm": [float(value.cpu()) for value in defect_norms],
        "per_head_defect_norm_fraction": [
            float(value.cpu()) / max(total_norm, 1e-12) for value in defect_norms
        ],
        "shapley_fisher_preservation": (
            None if effect_degenerate else shapley_values(values, head_count)
        ),
        "subsets": subset_records,
    }


def _source_selection(
    cohorts: Mapping[str, Mapping[str, Mapping[str, Any]]],
    config: HeadDecompositionConfig,
) -> dict[str, Any]:
    source = cohorts["source_selection"]
    head_count = int(source["composition"]["head_count"])
    candidates = []
    for mask in range(1 << head_count):
        records = [source[regime]["subsets"][str(mask)] for regime in REGIMES]
        sufficient = all(
            record["causal_pass"]
            and not record["effect_preservation"]["degenerate"]
            and record["effect_preservation"]["preserved_fraction"]
            >= config.sufficient_fisher_threshold
            for record in records
        )
        if sufficient:
            candidates.append(
                (
                    int(mask.bit_count()),
                    -min(
                        float(record["effect_preservation"]["preserved_fraction"])
                        for record in records
                    ),
                    subset_heads(mask, head_count),
                    mask,
                )
            )
    if not candidates:
        return {
            "selected_mask": None,
            "selected_heads": None,
            "cardinality": None,
            "source_sufficient_subset_count": 0,
        }
    candidates.sort()
    _, negative_minimum, heads, mask = candidates[0]
    return {
        "selected_mask": mask,
        "selected_heads": list(heads),
        "cardinality": len(heads),
        "source_minimum_fisher_preservation": -negative_minimum,
        "source_sufficient_subset_count": len(candidates),
    }


def _heldout_summary(
    cohorts: Mapping[str, Mapping[str, Mapping[str, Any]]],
    selection: Mapping[str, Any],
    config: HeadDecompositionConfig,
) -> dict[str, Any]:
    mask = selection["selected_mask"]
    if mask is None:
        return {
            "endpoint_replication": False,
            "fixed_subset_sufficiency": False,
            "complement_dominance": False,
            "subset_specificity": False,
            "selected_worst_preservation": None,
            "alternative_median_worst_preservation": None,
            "specificity_margin": None,
        }
    head_count = int(cohorts["source_selection"]["composition"]["head_count"])
    full_mask = (1 << head_count) - 1
    complement = full_mask ^ int(mask)
    heldout_cells = [
        cohorts[cohort][regime]
        for cohort in ("heldout_a", "heldout_b")
        for regime in REGIMES
    ]
    endpoint = all(cell["endpoint_replication"] for cell in heldout_cells)
    selected_records = [cell["subsets"][str(mask)] for cell in heldout_cells]
    complement_records = [cell["subsets"][str(complement)] for cell in heldout_cells]
    sufficient = bool(
        endpoint
        and all(
            record["causal_pass"]
            and not record["effect_preservation"]["degenerate"]
            and record["effect_preservation"]["preserved_fraction"]
            >= config.sufficient_fisher_threshold
            for record in selected_records
        )
    )
    dominance = bool(
        endpoint
        and all(
            not record["effect_preservation"]["degenerate"]
            and record["effect_preservation"]["preserved_fraction"]
            <= config.complement_fisher_ceiling
            for record in complement_records
        )
    )
    selected_worst = min(
        float(record["effect_preservation"]["preserved_fraction"])
        for record in selected_records
        if record["effect_preservation"]["preserved_fraction"] is not None
    )
    alternatives = []
    cardinality = int(mask).bit_count()
    for candidate in range(1 << head_count):
        if candidate == mask or candidate.bit_count() != cardinality:
            continue
        records = [cell["subsets"][str(candidate)] for cell in heldout_cells]
        if all(
            record["effect_preservation"]["preserved_fraction"] is not None
            for record in records
        ):
            alternatives.append(
                min(
                    float(record["effect_preservation"]["preserved_fraction"])
                    for record in records
                )
            )
    alternative_median = None if not alternatives else float(np.median(alternatives))
    margin = None if alternative_median is None else selected_worst - alternative_median
    specificity = bool(
        margin is not None and margin >= config.specificity_margin
    )
    return {
        "endpoint_replication": endpoint,
        "fixed_subset_sufficiency": sufficient,
        "complement_dominance": dominance,
        "subset_specificity": specificity,
        "selected_mask": mask,
        "selected_heads": selection["selected_heads"],
        "complement_mask": complement,
        "complement_heads": list(subset_heads(complement, head_count)),
        "selected_worst_preservation": selected_worst,
        "alternative_median_worst_preservation": alternative_median,
        "specificity_margin": margin,
    }


def analyze_cell(
    task: CircleTaskConfig,
    config: HeadDecompositionConfig,
    seed: int,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    bridge = _bridge_config(config)
    system, source_provenance = deck.load_source(task, bridge, 2, seed, device)
    frozen, frozen_path = _load_character_source(
        config, seed, source_provenance["checkpoint_sha256"]
    )
    frozen_sha = _sha256(frozen_path)
    fingerprint = _scientific_fingerprint(
        config,
        task,
        seed,
        source_provenance["checkpoint_sha256"],
        frozen_sha,
    )
    cohorts = {}
    for cohort in COHORTS:
        cohorts[cohort] = {}
        for regime in REGIMES:
            cohorts[cohort][regime] = analyze_cohort_regime(
                system,
                task,
                config,
                bridge,
                frozen,
                seed,
                cohort,
                regime,
                device,
            )
            print(f"seed {seed} {cohort} {regime} complete", flush=True)
    selection = _source_selection(cohorts, config)
    heldout = _heldout_summary(cohorts, selection, config)
    maximum_decomposition_error = max(
        float(cohorts[cohort][regime]["decomposition_relative_error"])
        for cohort in COHORTS
        for regime in REGIMES
    )
    gates = {
        "exact_decomposition": maximum_decomposition_error <= config.decomposition_tolerance,
        "sparse_source_selection": bool(
            selection["cardinality"] is not None
            and int(selection["cardinality"]) <= config.sparse_head_ceiling
        ),
        "heldout_endpoint_replication": heldout["endpoint_replication"],
        "fixed_subset_sufficiency": heldout["fixed_subset_sufficiency"],
        "complement_dominance": heldout["complement_dominance"],
        "subset_specificity": heldout["subset_specificity"],
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-c2-attention-head-decomposition-seed{seed}",
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else "preregistered_primary_evidence"
        ),
        "completed_at": _utc_now(),
        "seed": seed,
        "primary_seed": seed in config.primary_seeds,
        "configuration": asdict(config),
        "scientific_fingerprint": fingerprint,
        "provenance": {
            **source_provenance,
            "character_result": str(frozen_path),
            "character_result_sha256": frozen_sha,
            "character_implementation_sha256": frozen["implementation_sha256"],
        },
        "cohorts": cohorts,
        "source_selection": selection,
        "heldout": heldout,
        "maximum_decomposition_relative_error": maximum_decomposition_error,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "implementation_sha256": _implementation_digest(),
        "analysis_seconds": time.perf_counter() - started,
    }
    _write_json(output / "runs" / f"seed_{seed}" / "result.json", result)
    return result


def aggregate(
    runs: Sequence[Mapping[str, Any]], config: HeadDecompositionConfig
) -> dict[str, Any]:
    primary = [run for run in runs if int(run["seed"]) in config.primary_seeds]
    gate_names = (
        "exact_decomposition",
        "sparse_source_selection",
        "heldout_endpoint_replication",
        "fixed_subset_sufficiency",
        "complement_dominance",
        "subset_specificity",
    )
    gate_counts = {
        gate: sum(bool(run["gates"][gate]) for run in primary) for gate in gate_names
    }
    gate_passes = {gate: count == len(config.primary_seeds) for gate, count in gate_counts.items()}
    return {
        "primary_seed_count": len(config.primary_seeds),
        "primary_seeds": list(config.primary_seeds),
        "gate_counts": gate_counts,
        "gate_passes": gate_passes,
        "confirmed": all(gate_passes.values()),
        "per_seed": {
            str(run["seed"]): {
                "primary_seed": run["primary_seed"],
                "source_selection": run["source_selection"],
                "heldout": run["heldout"],
                "gates": run["gates"],
                "maximum_decomposition_relative_error": run[
                    "maximum_decomposition_relative_error"
                ],
            }
            for run in runs
        },
    }


def run_campaign(
    config: HeadDecompositionConfig, output: Path
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    runs = []
    reused = 0
    implementation = _implementation_digest()
    for seed in config.seeds:
        path = output / "runs" / f"seed_{seed}" / "result.json"
        if path.is_file():
            existing = json.loads(path.read_text())
            if (
                existing.get("schema_version") == SCHEMA_VERSION
                and existing.get("status") == "completed"
                and existing.get("implementation_sha256") == implementation
                and existing.get("scientific_fingerprint")
            ):
                source = Path(existing["provenance"]["checkpoint"])
                character = Path(existing["provenance"]["character_result"])
                expected = _scientific_fingerprint(
                    config,
                    task,
                    seed,
                    existing["provenance"]["checkpoint_sha256"],
                    existing["provenance"]["character_result_sha256"],
                )
                if (
                    source.is_file()
                    and character.is_file()
                    and _sha256(source) == existing["provenance"]["checkpoint_sha256"]
                    and _sha256(character) == existing["provenance"]["character_result_sha256"]
                    and existing["scientific_fingerprint"] == expected
                ):
                    runs.append(existing)
                    reused += 1
                    print(f"resuming {existing['experiment_id']}", flush=True)
                    continue
        result = analyze_cell(task, config, seed, output, device)
        runs.append(result)
        print(result["experiment_id"], f"{result['analysis_seconds']:.1f}s", flush=True)

    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else "preregistered_primary_evidence"
        ),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "task_config": asdict(task),
        "cohorts": COHORTS,
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
            "reused": reused,
            "completed": len(runs),
            "failed": 0,
            "trained_models": 0,
        },
        "aggregates": aggregate(runs, config),
        "results": [
            {
                "experiment_id": run["experiment_id"],
                "seed": run["seed"],
                "primary_seed": run["primary_seed"],
                "scientific_fingerprint": run["scientific_fingerprint"],
                "analysis_seconds": run["analysis_seconds"],
                "gates": run["gates"],
                "path": str(output / "runs" / f"seed_{run['seed']}" / "result.json"),
            }
            for run in runs
        ],
        "method_boundaries": [
            "Head indices are checkpoint-local and are not aligned across independently trained seeds.",
            "Subset states are exact additive attention-output interventions but can be off the natural activation manifold.",
            "Fisher preservation is conditioned on the frozen task decoder.",
            "Repeated barycenter patches imply within-orbit identity but not global branch erasure.",
            "The source subset is selected on a known cohort family; held-out outcomes are new head-level measurements, not new model seeds.",
        ],
    }
    _write_json(output / "campaign_results.json", campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=HeadDecompositionConfig.source_root)
    parser.add_argument("--character-root", default=HeadDecompositionConfig.character_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_c2_attention_head_decomposition/"
            "20260806_d6_preregistered"
        ),
    )
    parser.add_argument("--seeds", default="7,17,29,41,53")
    parser.add_argument("--primary-seeds", default="7,29,53")
    parser.add_argument("--orbit-count", type=int, default=64)
    parser.add_argument("--first-blocks", type=int, default=6)
    parser.add_argument("--sparse-head-ceiling", type=int, default=2)
    parser.add_argument("--sufficient-fisher-threshold", type=float, default=0.90)
    parser.add_argument("--complement-fisher-ceiling", type=float, default=0.50)
    parser.add_argument("--specificity-margin", type=float, default=0.20)
    parser.add_argument("--task-effect-floor", type=float, default=1e-6)
    parser.add_argument("--decomposition-tolerance", type=float, default=1e-6)
    parser.add_argument("--activation-batch-size", type=int, default=256)
    parser.add_argument("--continuation-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args(argv)
    config = HeadDecompositionConfig(
        source_root=args.source_root,
        character_root=args.character_root,
        seeds=_ints(args.seeds),
        primary_seeds=_ints(args.primary_seeds),
        orbit_count=args.orbit_count,
        first_blocks=args.first_blocks,
        sparse_head_ceiling=args.sparse_head_ceiling,
        sufficient_fisher_threshold=args.sufficient_fisher_threshold,
        complement_fisher_ceiling=args.complement_fisher_ceiling,
        specificity_margin=args.specificity_margin,
        task_effect_floor=args.task_effect_floor,
        decomposition_tolerance=args.decomposition_tolerance,
        activation_batch_size=args.activation_batch_size,
        continuation_batch_size=args.continuation_batch_size,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    print(args.output / "campaign_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
