#!/usr/bin/env python3
"""Causally localize TinyLLM Reynolds defects to attention-head subsets."""

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
import torch.nn.functional as F

import experiments.structure_net.tinyllm_deck_action_descrambler as deck
import experiments.structure_net.tinyllm_reynolds_character_coupling as coupling
import experiments.structure_net.tinyllm_reynolds_koopman as koopman
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-attention-head-defect.v1"
HYPOTHESIS_ID = "tinyllm-attention-head-defect-sparsity-v1"
REGIMES = ("composition", "extrapolation")


@dataclass(frozen=True)
class AttentionHeadDefectConfig:
    source_root: str = "data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered"
    character_root: str = "data/experiments/tinyllm_reynolds_character_coupling/20260806_d6_preregistered"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    orbit_count: int = 64
    first_blocks: int = 6
    sparse_subset_size: int = 2
    sufficient_fisher_threshold: float = 0.90
    necessary_fisher_threshold: float = 0.70
    task_effect_floor: float = 1e-6
    reconstruction_tolerance: float = 1e-6
    activation_batch_size: int = 256
    continuation_batch_size: int = 256
    device: str = "cpu"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if len(self.seeds) < 5 and not self.allow_underpowered:
            raise ValueError("confirmatory campaign requires five seeds")
        if self.orbit_count < 8:
            raise ValueError("at least eight exact orbits are required")
        if self.first_blocks < 1:
            raise ValueError("at least one transformer block is required")
        if self.sparse_subset_size != 2:
            raise ValueError("the preregistered sparse subset size is two")
        if not 0.0 < self.necessary_fisher_threshold < self.sufficient_fisher_threshold <= 1.0:
            raise ValueError("Fisher thresholds must satisfy 0 < necessary < sufficient <= 1")
        if self.task_effect_floor <= 0.0 or self.reconstruction_tolerance <= 0.0:
            raise ValueError("numeric floors and tolerances must be positive")


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
    for path in (Path(__file__), Path(deck.__file__), Path(coupling.__file__)):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _bridge_config(config: AttentionHeadDefectConfig) -> deck.DeckDescramblerConfig:
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


def intervention_subsets(n_head: int) -> tuple[tuple[int, ...], ...]:
    """Return empty, singleton, pair, leave-one-out, and full subsets."""
    if n_head < 3:
        raise ValueError("at least three heads are required")
    values: list[tuple[int, ...]] = [()]
    values.extend((head,) for head in range(n_head))
    values.extend(itertools.combinations(range(n_head), 2))
    values.extend(tuple(index for index in range(n_head) if index != head) for head in range(n_head))
    values.append(tuple(range(n_head)))
    return tuple(dict.fromkeys(values))


def _subset_key(subset: Sequence[int], n_head: int) -> str:
    value = tuple(subset)
    if not value:
        return "none"
    if len(value) == n_head:
        return "all"
    missing = tuple(index for index in range(n_head) if index not in value)
    if len(missing) == 1 and len(value) == n_head - 1:
        return f"all_except_h{missing[0]}"
    return "_".join(f"h{index}" for index in value)


@torch.no_grad()
def projected_head_outputs(attention: Any, value: torch.Tensor) -> torch.Tensor:
    """Return bias-free post-projection contributions as [batch, head, token, channel]."""
    batch, sequence, channels = value.shape
    query, key, content = attention.c_attn(value).split(attention.n_embd, dim=-1)
    head_size = channels // attention.n_head

    def split(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(batch, sequence, attention.n_head, head_size).transpose(1, 2)

    query, key, content = map(split, (query, key, content))
    attended = F.scaled_dot_product_attention(
        query, key, content, dropout_p=0.0, is_causal=True
    )
    contributions = []
    for head in range(attention.n_head):
        weight = attention.c_proj.weight[:, head * head_size : (head + 1) * head_size]
        contributions.append(F.linear(attended[:, head], weight, bias=None))
    return torch.stack(contributions, dim=1)


@torch.no_grad()
def attention_head_defects(block: Any, sheets: torch.Tensor) -> dict[str, torch.Tensor | float]:
    """Exactly decompose an attention-residual Reynolds defect by projected head."""
    if sheets.ndim != 4:
        raise ValueError("sheets must have shape [orbit, member, token, channel]")
    orbit_count, members = sheets.shape[:2]
    barycenter = sheets.mean(1)
    flat = sheets.flatten(0, 1)
    normalized_sheets = block.ln_1(flat)
    normalized_barycenter = block.ln_1(barycenter)
    sheet_heads = projected_head_outputs(block.attn, normalized_sheets).reshape(
        orbit_count, members, block.attn.n_head, *sheets.shape[2:]
    )
    barycenter_heads = projected_head_outputs(block.attn, normalized_barycenter)
    defects = sheet_heads.mean(1) - barycenter_heads
    propagated = barycenter + block.attn(normalized_barycenter)
    actual = (
        flat + block.attn(normalized_sheets)
    ).reshape_as(sheets).mean(1)
    full_defect = actual - propagated

    sheet_reconstruction = sheet_heads.sum(2)
    bias = block.attn.c_proj.bias
    if bias is not None:
        sheet_reconstruction = sheet_reconstruction + bias
    ordinary = block.attn(normalized_sheets).reshape(
        orbit_count, members, *sheets.shape[2:]
    )
    ordinary_error = torch.linalg.vector_norm(
        (sheet_reconstruction - ordinary).double()
    ) / torch.linalg.vector_norm(ordinary.double()).clamp_min(1e-12)
    defect_error = torch.linalg.vector_norm(
        (defects.sum(1) - full_defect).double()
    ) / torch.linalg.vector_norm(full_defect.double()).clamp_min(1e-12)
    return {
        "barycenter": barycenter,
        "propagated": propagated,
        "actual": actual,
        "full_defect": full_defect,
        "head_defects": defects,
        "ordinary_attention_relative_error": float(ordinary_error.cpu()),
        "defect_relative_error": float(defect_error.cpu()),
    }


def _head_geometry(defects: torch.Tensor, full_defect: torch.Tensor) -> list[dict[str, float]]:
    full = full_defect.double().reshape(-1)
    full_norm = torch.linalg.vector_norm(full).clamp_min(1e-12)
    output = []
    for head in range(defects.shape[1]):
        value = defects[:, head].double().reshape(-1)
        norm = torch.linalg.vector_norm(value)
        output.append(
            {
                "head": head,
                "norm_ratio_to_full": float((norm / full_norm).cpu()),
                "cosine_to_full": float(
                    (torch.dot(value, full) / (norm * full_norm).clamp_min(1e-12)).cpu()
                ),
            }
        )
    return output


def _load_character_source(
    config: AttentionHeadDefectConfig,
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
        or value["provenance"]["checkpoint_sha256"] != checkpoint_sha256
    ):
        raise ValueError(f"invalid character source {path}")
    return value, path


def _fingerprint(
    config: AttentionHeadDefectConfig,
    task: CircleTaskConfig,
    seed: int,
    provenance: Mapping[str, Any],
) -> str:
    value = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "task": asdict(task),
        "seed": seed,
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "character_result_sha256": provenance["character_result_sha256"],
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@torch.no_grad()
def analyze_regime(
    system: Any,
    task: CircleTaskConfig,
    config: AttentionHeadDefectConfig,
    seed: int,
    regime: str,
    source_record: Mapping[str, Any],
    bridge: deck.DeckDescramblerConfig,
    device: torch.device,
) -> dict[str, Any]:
    regime_index = REGIMES.index(regime)
    transition_index = source_record["regimes"][regime]["synthesis_front_index"]
    if transition_index is None:
        raise ValueError(f"missing synthesis front for seed {seed} {regime}")
    transition_index = int(transition_index)
    if transition_index % 2:
        raise ValueError("the degree-two synthesis front must be an attention sublayer")
    source_cut, target_cut = koopman.one_step_transitions(config.first_blocks)[transition_index]
    if target_cut != source_record["regimes"][regime]["synthesis_target_cut"]:
        raise ValueError("synthesis target cut does not match frozen comparator")

    dataset = deck.generate_exact_orbits(
        task,
        k=2,
        orbit_count=config.orbit_count,
        seed=seed + 2501 + 101 * regime_index,
        regime=regime,
    )
    captured = deck.capture_sequences(system, dataset, bridge, device)
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
    source = torch.from_numpy(captured[source_cut]).to(device)
    source = source.reshape((config.orbit_count, 2) + source.shape[1:])
    block = system.model.transformer["h"][transition_index // 2]
    decomposition = attention_head_defects(block, source)
    propagated = decomposition["propagated"]
    full_defect = decomposition["full_defect"]
    defects = decomposition["head_defects"]
    assert isinstance(propagated, torch.Tensor)
    assert isinstance(full_defect, torch.Tensor)
    assert isinstance(defects, torch.Tensor)

    subsets = intervention_subsets(block.attn.n_head)
    states = []
    keys = []
    for subset in subsets:
        state = propagated
        if subset:
            state = state + defects[:, tuple(subset)].sum(1)
        states.append(state)
        keys.append(_subset_key(subset, block.attn.n_head))
    repeated = torch.cat([coupling._repeat_patch(state, 2) for state in states])
    combined = coupling._continue_tensor(
        system, target_cut, repeated, task, config.continuation_batch_size
    )
    lengths = [config.orbit_count * 2] * len(states)
    posteriors = dict(zip(keys, combined.split(lengths)))
    empty_orbits = posteriors["none"].reshape(config.orbit_count, 2, -1)[:, 0]
    full_orbits = posteriors["all"].reshape(config.orbit_count, 2, -1)[:, 0]

    records: dict[str, Any] = {}
    subset_by_key = dict(zip(keys, subsets))
    for key in keys:
        posterior = posteriors[key]
        diagnostics = coupling._diagnostics(posterior, dataset)
        causal_pass, _ = coupling._causal_classification(diagnostics, baseline, 2)
        effect = coupling.task_effect_approximation(
            empty_orbits,
            full_orbits,
            posterior.reshape(config.orbit_count, 2, -1)[:, 0],
            config.task_effect_floor,
        )
        records[key] = {
            "subset": list(subset_by_key[key]),
            "causal_pass": causal_pass,
            "diagnostics": diagnostics,
            "fisher_effect": effect,
        }

    sufficient = sorted(
        key
        for key, record in records.items()
        if 0 < len(record["subset"]) <= config.sparse_subset_size
        and record["causal_pass"]
        and not record["fisher_effect"]["degenerate"]
        and record["fisher_effect"]["explained_fraction"]
        >= config.sufficient_fisher_threshold
    )
    necessary = []
    for head in range(block.attn.n_head):
        record = records[f"all_except_h{head}"]
        if (
            not record["causal_pass"]
            or record["fisher_effect"]["degenerate"]
            or record["fisher_effect"]["explained_fraction"]
            < config.necessary_fisher_threshold
        ):
            necessary.append(head)
    pairs = [
        record
        for record in records.values()
        if len(record["subset"]) == 2
    ]
    best_pair = max(
        pairs,
        key=lambda record: -math.inf
        if record["fisher_effect"]["explained_fraction"] is None
        else record["fisher_effect"]["explained_fraction"],
    )
    contract_pass = bool(
        decomposition["ordinary_attention_relative_error"] <= config.reconstruction_tolerance
        and decomposition["defect_relative_error"] <= config.reconstruction_tolerance
    )
    endpoint_pass = bool(not records["none"]["causal_pass"] and records["all"]["causal_pass"])
    return {
        "regime": regime,
        "evaluation_seed": seed + 2501 + 101 * regime_index,
        "transition_index": transition_index,
        "source_cut": source_cut,
        "target_cut": target_cut,
        "baseline": baseline,
        "ordinary_attention_relative_error": decomposition["ordinary_attention_relative_error"],
        "defect_relative_error": decomposition["defect_relative_error"],
        "contract_pass": contract_pass,
        "endpoint_pass": endpoint_pass,
        "head_geometry": _head_geometry(defects, full_defect),
        "sufficient_sparse_subsets": sufficient,
        "necessary_heads": necessary,
        "best_pair": {
            "subset": best_pair["subset"],
            "causal_pass": best_pair["causal_pass"],
            "fisher_effect": best_pair["fisher_effect"],
        },
        "interventions": records,
    }


def seed_gates(regimes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    sufficient_sets = {
        regime: set(regimes[regime]["sufficient_sparse_subsets"])
        for regime in REGIMES
    }
    necessary_sets = {
        regime: set(regimes[regime]["necessary_heads"])
        for regime in REGIMES
    }
    union = sufficient_sets["composition"] | sufficient_sets["extrapolation"]
    intersection = sufficient_sets["composition"] & sufficient_sets["extrapolation"]
    jaccard = 1.0 if not union else len(intersection) / len(union)
    return {
        "exact_contract_and_endpoint": all(
            regimes[regime]["contract_pass"] and regimes[regime]["endpoint_pass"]
            for regime in REGIMES
        ),
        "sparse_sufficiency": all(sufficient_sets[regime] for regime in REGIMES),
        "shift_stable_sparse_circuit": bool(intersection),
        "individual_necessity": bool(
            necessary_sets["composition"] & necessary_sets["extrapolation"]
        ),
        "common_sufficient_subsets": sorted(intersection),
        "common_necessary_heads": sorted(
            necessary_sets["composition"] & necessary_sets["extrapolation"]
        ),
        "sufficient_subset_jaccard": jaccard,
    }


def analyze_cell(
    task: CircleTaskConfig,
    config: AttentionHeadDefectConfig,
    seed: int,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    bridge = _bridge_config(config)
    system, source_provenance = deck.load_source(task, bridge, 2, seed, device)
    source_record, source_path = _load_character_source(
        config, seed, source_provenance["checkpoint_sha256"]
    )
    provenance = {
        **source_provenance,
        "character_result": str(source_path),
        "character_result_sha256": _sha256(source_path),
    }
    fingerprint = _fingerprint(config, task, seed, provenance)
    regimes = {
        regime: analyze_regime(
            system, task, config, seed, regime, source_record, bridge, device
        )
        for regime in REGIMES
    }
    gates = seed_gates(regimes)
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-attention-head-defect-seed{seed}",
        "status": "completed",
        "evidence_role": "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_primary_evidence",
        "completed_at": _utc_now(),
        "seed": seed,
        "configuration": asdict(config),
        "scientific_fingerprint": fingerprint,
        "implementation_sha256": _implementation_digest(),
        "provenance": provenance,
        "regimes": regimes,
        "gates": gates,
        "all_primary_gates_passed": all(
            bool(gates[name])
            for name in (
                "exact_contract_and_endpoint",
                "sparse_sufficiency",
                "shift_stable_sparse_circuit",
                "individual_necessity",
            )
        ),
        "analysis_seconds": time.perf_counter() - started,
    }
    _write_json(output / "runs" / f"seed_{seed}" / "result.json", result)
    return result


def aggregate(
    runs: Sequence[Mapping[str, Any]], config: AttentionHeadDefectConfig
) -> dict[str, Any]:
    required = 4 if len(config.seeds) >= 5 else len(config.seeds)
    gate_names = (
        "exact_contract_and_endpoint",
        "sparse_sufficiency",
        "shift_stable_sparse_circuit",
        "individual_necessity",
    )
    counts = {
        gate: sum(bool(run["gates"][gate]) for run in runs) for gate in gate_names
    }
    passes = {gate: count >= required for gate, count in counts.items()}
    return {
        "required_seed_count": required,
        "gate_counts": counts,
        "gate_passes": passes,
        "confirmed": all(passes.values()),
        "per_seed": {
            str(run["seed"]): {
                "gates": run["gates"],
                "target_cuts": {
                    regime: run["regimes"][regime]["target_cut"]
                    for regime in REGIMES
                },
                "sufficient_subsets": {
                    regime: run["regimes"][regime]["sufficient_sparse_subsets"]
                    for regime in REGIMES
                },
                "necessary_heads": {
                    regime: run["regimes"][regime]["necessary_heads"]
                    for regime in REGIMES
                },
            }
            for run in runs
        },
    }


def run_campaign(
    config: AttentionHeadDefectConfig, output: Path
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    runs = []
    for seed in config.seeds:
        path = output / "runs" / f"seed_{seed}" / "result.json"
        if path.is_file():
            existing = json.loads(path.read_text())
            checkpoint = Path(config.source_root) / "runs" / "k2" / f"seed_{seed}" / "model.pt"
            character = Path(config.character_root) / "runs" / "k2" / f"seed_{seed}" / "result.json"
            provenance = {
                "checkpoint_sha256": _sha256(checkpoint),
                "character_result_sha256": _sha256(character),
            }
            expected = _fingerprint(config, task, seed, provenance)
            if (
                existing.get("status") == "completed"
                and existing.get("schema_version") == SCHEMA_VERSION
                and existing.get("scientific_fingerprint") == expected
                and existing.get("implementation_sha256") == _implementation_digest()
            ):
                runs.append(existing)
                print(f"resuming {existing['experiment_id']}", flush=True)
                continue
        result = analyze_cell(task, config, seed, output, device)
        runs.append(result)
        print(result["experiment_id"], f"{result['analysis_seconds']:.1f}s", flush=True)
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_primary_evidence",
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "implementation_sha256": _implementation_digest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
        },
        "summary": {
            "requested": len(config.seeds),
            "completed": len(runs),
            "failed": 0,
            "trained_models": 0,
            "interventions_per_shift": len(intervention_subsets(6)),
        },
        "aggregates": aggregate(runs, config),
        "method_boundaries": [
            "Head components are defined after learned output-projection slices and do not identify unique within-head neurons.",
            "The campaign reuses predecessor orbit cohorts and is a causal decomposition, not an independent front replication.",
            "Patching a head defect tests downstream sufficiency, not independent upstream computability.",
            "Repeated patches make within-orbit branch identity constant by construction.",
            "The task gate remains conditioned on the frozen decoder.",
        ],
    }
    _write_json(output / "campaign_results.json", campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=AttentionHeadDefectConfig.source_root)
    parser.add_argument("--character-root", default=AttentionHeadDefectConfig.character_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_attention_head_defect/20260806_d6_preregistered"
        ),
    )
    parser.add_argument("--seeds", default="7,17,29,41,53")
    parser.add_argument("--orbit-count", type=int, default=64)
    parser.add_argument("--first-blocks", type=int, default=6)
    parser.add_argument("--sufficient-fisher-threshold", type=float, default=0.90)
    parser.add_argument("--necessary-fisher-threshold", type=float, default=0.70)
    parser.add_argument("--task-effect-floor", type=float, default=1e-6)
    parser.add_argument("--reconstruction-tolerance", type=float, default=1e-6)
    parser.add_argument("--activation-batch-size", type=int, default=256)
    parser.add_argument("--continuation-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args(argv)
    config = AttentionHeadDefectConfig(
        source_root=args.source_root,
        character_root=args.character_root,
        seeds=_ints(args.seeds),
        orbit_count=args.orbit_count,
        first_blocks=args.first_blocks,
        sufficient_fisher_threshold=args.sufficient_fisher_threshold,
        necessary_fisher_threshold=args.necessary_fisher_threshold,
        task_effect_floor=args.task_effect_floor,
        reconstruction_tolerance=args.reconstruction_tolerance,
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
