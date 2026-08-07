#!/usr/bin/env python3
"""Causally decompose TinyLLM Reynolds defects into deck-character couplings."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Mapping, Optional, Sequence
import time

import numpy as np
import torch

import experiments.structure_net.tinyllm_deck_action_descrambler as deck
import experiments.structure_net.tinyllm_io_correspondence as io_source
import experiments.structure_net.tinyllm_reynolds_koopman as koopman
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-reynolds-character-coupling.v1"
HYPOTHESIS_ID = "tinyllm-reynolds-character-coupling-synthesis-v1"
REGIMES = ("composition", "extrapolation")
CONDITIONS = ("exact", "shuffled_membership", "matched_random_directions")


@dataclass(frozen=True)
class CharacterCouplingConfig:
    source_root: str = "data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered"
    causal_root: str = "data/experiments/tinyllm_deck_action_descrambler/20260806_d6_preregistered"
    koopman_root: str = "data/experiments/tinyllm_reynolds_koopman/20260806_d6_preregistered"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    degrees: tuple[int, ...] = (2, 3)
    orbit_count: int = 64
    first_blocks: int = 6
    finite_difference_eta: float = 0.25
    quadratic_fisher_threshold: float = 0.70
    task_effect_floor: float = 1e-6
    activation_batch_size: int = 256
    continuation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if set(self.degrees).difference((2, 3)):
            raise ValueError("degrees must be drawn from 2,3")
        if len(self.seeds) < 5 and not self.allow_underpowered:
            raise ValueError("confirmatory campaign requires five seeds")
        if self.orbit_count < 8:
            raise ValueError("at least eight exact orbits are required")
        if self.first_blocks < 1:
            raise ValueError("at least one transformer block is required")
        if not 0.0 < self.finite_difference_eta <= 0.5:
            raise ValueError("finite-difference eta must be in (0,0.5]")
        if not 0.0 < self.quadratic_fisher_threshold <= 1.0:
            raise ValueError("quadratic threshold must be in (0,1]")


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
    for path in (Path(__file__), Path(deck.__file__), Path(koopman.__file__)):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _bridge_config(config: CharacterCouplingConfig) -> deck.DeckDescramblerConfig:
    return deck.DeckDescramblerConfig(
        source_root=config.source_root,
        seeds=config.seeds,
        degrees=config.degrees,
        fit_orbits=config.orbit_count,
        evaluation_orbits=config.orbit_count,
        map_points=24,
        first_blocks=config.first_blocks,
        activation_batch_size=config.activation_batch_size,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )


def _apply_sublayer(system: Any, transition_index: int, values: torch.Tensor) -> torch.Tensor:
    block = system.model.transformer["h"][transition_index // 2]
    if transition_index % 2 == 0:
        return values + block.attn(block.ln_1(values))
    return values + block.mlp(block.ln_2(values))


@torch.no_grad()
def _continue_tensor(
    system: Any,
    cut: str,
    values: torch.Tensor,
    task: CircleTaskConfig,
    batch_size: int,
) -> torch.Tensor:
    device = values.device
    answer = torch.tensor(task.answer_token_ids, device=device)
    output = []
    for start in range(0, len(values), batch_size):
        value = values[start : start + batch_size]
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
        logits = io_source._task_logits(system.model, value[:, -1, :], answer)
        output.append(torch.softmax(logits, -1))
    return torch.cat(output)


def _repeat_patch(values: torch.Tensor, k: int) -> torch.Tensor:
    return values[:, None].expand((-1, k) + (-1,) * (values.ndim - 1)).flatten(0, 1)


def _condition_sheets(
    source: torch.Tensor,
    target: torch.Tensor,
    condition: str,
    seed: int,
    apply: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if condition == "exact":
        return source, target
    orbit_count, k = source.shape[:2]
    if condition == "shuffled_membership":
        generator = np.random.default_rng(seed)
        source_out, target_out = source.clone(), target.clone()
        for sheet in range(1, k):
            order = torch.tensor(generator.permutation(orbit_count), device=source.device)
            source_out[:, sheet] = source[order, sheet]
            target_out[:, sheet] = target[order, sheet]
        return source_out, target_out
    if condition != "matched_random_directions":
        raise ValueError(condition)
    barycenter = source.mean(1, keepdim=True)
    exact_delta = source - barycenter
    generator = torch.Generator(device=source.device)
    generator.manual_seed(seed)
    random_delta = torch.randn(source.shape, generator=generator, device=source.device, dtype=source.dtype)
    random_delta -= random_delta.mean(1, keepdim=True)
    axes = tuple(range(1, source.ndim))
    exact_norm = torch.square(exact_delta).sum(dim=axes, keepdim=True).sqrt()
    random_norm = torch.square(random_delta).sum(dim=axes, keepdim=True).sqrt().clamp_min(1e-12)
    random_delta = random_delta * exact_norm / random_norm
    source_out = barycenter + random_delta
    target_out = apply(source_out.flatten(0, 1)).reshape_as(source_out)
    return source_out, target_out


def character_metrics(sheets: torch.Tensor) -> dict[str, Any]:
    orbit_count, k = sheets.shape[:2]
    flat = sheets.reshape(orbit_count, k, -1).double()
    members = torch.arange(k, device=sheets.device, dtype=torch.float64)
    components = []
    energies = []
    for character in range(k):
        weights = torch.exp(-2j * math.pi * character * members / k)
        component = torch.einsum("j,njd->nd", weights, flat.to(torch.complex128)) / k
        components.append(component)
        energies.append(float(torch.square(torch.abs(component)).sum().cpu()))
    reconstructed = torch.stack(
        [
            sum(
                components[r] * complex(math.cos(2.0 * math.pi * r * j / k), math.sin(2.0 * math.pi * r * j / k))
                for r in range(k)
            )
            for j in range(k)
        ],
        dim=1,
    )
    error = torch.linalg.vector_norm(reconstructed - flat.to(torch.complex128)) / torch.linalg.vector_norm(flat).clamp_min(1e-12)
    total = max(1e-12, sum(energies))
    return {
        "energy_fractions": {str(index): value / total for index, value in enumerate(energies)},
        "reconstruction_error": float(error.cpu()),
        "neutral_quadratic_pairs": [[r, (-r) % k] for r in range(1, k)],
    }


@torch.no_grad()
def taylor_defect(
    sheets: torch.Tensor,
    apply: Any,
    eta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return propagated state, exact defect, neutral quadratic, and cubic terms."""
    barycenter = sheets.mean(1)
    delta = sheets - barycenter[:, None]
    center = apply(barycenter)
    shape = sheets.shape

    plus = apply((barycenter[:, None] + eta * delta).flatten(0, 1)).reshape(shape)
    minus = apply((barycenter[:, None] - eta * delta).flatten(0, 1)).reshape(shape)
    second = (plus - 2.0 * center[:, None] + minus) / eta**2
    quadratic = 0.5 * second.mean(1)

    plus_two = apply((barycenter[:, None] + 2.0 * eta * delta).flatten(0, 1)).reshape(shape)
    minus_two = apply((barycenter[:, None] - 2.0 * eta * delta).flatten(0, 1)).reshape(shape)
    third = (plus_two - 2.0 * plus + 2.0 * minus - minus_two) / (2.0 * eta**3)
    cubic = third.mean(1) / 6.0
    return center, delta, quadratic, cubic


def residual_approximation(exact: torch.Tensor, approximation: torch.Tensor) -> dict[str, float]:
    exact_flat = exact.double().reshape(-1)
    approximation_flat = approximation.double().reshape(-1)
    denominator = float(torch.square(exact_flat).sum().cpu())
    residual = float(torch.square(exact_flat - approximation_flat).sum().cpu())
    cosine = float(
        torch.dot(exact_flat, approximation_flat).cpu()
        / (torch.linalg.vector_norm(exact_flat) * torch.linalg.vector_norm(approximation_flat)).clamp_min(1e-12).cpu()
    )
    return {
        "explained_fraction": 1.0 - residual / max(1e-12, denominator),
        "cosine": cosine,
        "exact_norm": math.sqrt(denominator),
        "approximation_norm": math.sqrt(float(torch.square(approximation_flat).sum().cpu())),
    }


def fisher_rao_squared(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    affinity = torch.sqrt(left.clamp_min(0) * right.clamp_min(0)).sum(-1).clamp(-1.0, 1.0)
    return torch.square(2.0 * torch.arccos(affinity))


def task_effect_approximation(
    propagated: torch.Tensor,
    actual: torch.Tensor,
    approximation: torch.Tensor,
    floor: float,
) -> dict[str, Any]:
    denominator = float(fisher_rao_squared(propagated, actual).mean().cpu())
    numerator = float(fisher_rao_squared(approximation, actual).mean().cpu())
    degenerate = denominator < floor
    return {
        "actual_fisher_effect": denominator,
        "remaining_fisher_error": numerator,
        "explained_fraction": None if degenerate else 1.0 - numerator / denominator,
        "degenerate": degenerate,
    }


def _causal_classification(metrics: Mapping[str, Any], baseline: Mapping[str, Any], k: int) -> tuple[bool, str]:
    accuracy_loss = float(baseline["exact_bin_accuracy"] - metrics["exact_bin_accuracy"])
    passed = bool(
        metrics["circular_alignment"] >= 0.90
        and metrics["sampling_resolved"]
        and abs(metrics["winding_degree"] - k) <= 0.10
        and accuracy_loss <= 0.03
    )
    return passed, "pass" if passed else "fail"


def _regime_name(propagated: bool, actual: bool) -> str:
    return {
        (False, False): "cover_required_after_sublayer",
        (False, True): "invariant_synthesis",
        (True, True): "quotient_already_closed",
        (True, False): "quotient_corruption",
    }[(propagated, actual)]


def _diagnostics(posterior: torch.Tensor, dataset: deck.OrbitDataset) -> dict[str, Any]:
    result = deck.output_diagnostics(posterior.double().cpu().numpy(), dataset, include_topology=False)
    result["within_orbit_branch_balanced_accuracy"] = 1.0 / dataset.k
    result["within_orbit_state_identity"] = True
    return result


def _load_comparators(config: CharacterCouplingConfig, k: int, seed: int, checkpoint_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    causal_path = Path(config.causal_root) / "runs" / f"k{k}" / f"seed_{seed}" / "result.json"
    koopman_path = Path(config.koopman_root) / "runs" / f"k{k}" / f"seed_{seed}" / "result.json"
    causal = json.loads(causal_path.read_text())
    reynolds = json.loads(koopman_path.read_text())
    if causal.get("status") != "completed" or causal["provenance"]["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError(f"invalid causal comparator {causal_path}")
    if reynolds.get("status") != "completed" or reynolds["provenance"]["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError(f"invalid Reynolds comparator {koopman_path}")
    ordered = deck.cut_names(int(causal["configuration"]["first_blocks"]))
    fronts = {
        regime: next(
            (
                cut for cut in ordered
                if causal["cuts"][cut]["causal"][regime]["orbit_average"]["causal_classification"] == "preserved"
            ),
            None,
        )
        for regime in REGIMES
    }
    return {
        "path": str(causal_path), "sha256": _sha256(causal_path), "fronts": fronts
    }, {
        "path": str(koopman_path), "sha256": _sha256(koopman_path),
        "autonomous_one_step_near_front": reynolds["autonomous_one_step_near_front"],
    }


def _cut_depth(cut: Optional[str], transitions: Sequence[tuple[str, str]]) -> Optional[int]:
    if cut is None:
        return None
    if cut == "full":
        return len(transitions)
    for index, (_, target) in enumerate(transitions):
        if target == cut:
            return index + 1
    return None


@torch.no_grad()
def analyze_transition(
    system: Any,
    dataset: deck.OrbitDataset,
    source: np.ndarray,
    target: np.ndarray,
    target_cut: str,
    transition_index: int,
    condition: str,
    seed: int,
    baseline: Mapping[str, Any],
    task: CircleTaskConfig,
    config: CharacterCouplingConfig,
    device: torch.device,
) -> dict[str, Any]:
    n, k = dataset.orbit_count, dataset.k
    source_t = torch.from_numpy(source).to(device).reshape((n, k) + source.shape[1:])
    target_t = torch.from_numpy(target).to(device).reshape((n, k) + target.shape[1:])
    apply = lambda value: _apply_sublayer(system, transition_index, value)
    source_t, target_t = _condition_sheets(source_t, target_t, condition, seed, apply)
    propagated, _, quadratic, cubic = taylor_defect(
        source_t, apply, config.finite_difference_eta
    )
    actual = target_t.mean(1)
    defect = actual - propagated
    patches = {
        "propagated": propagated,
        "actual_barycenter": actual,
        "quadratic": propagated + quadratic,
        "quadratic_plus_cubic": propagated + quadratic + cubic,
    }
    posteriors = {
        name: _continue_tensor(
            system, target_cut, _repeat_patch(value, k), task, config.continuation_batch_size
        )
        for name, value in patches.items()
    }
    diagnostics = {name: _diagnostics(value, dataset) for name, value in posteriors.items()}
    propagated_pass, _ = _causal_classification(diagnostics["propagated"], baseline, k)
    actual_pass, _ = _causal_classification(diagnostics["actual_barycenter"], baseline, k)
    orbit_posteriors = {
        name: value.reshape(n, k, -1)[:, 0] for name, value in posteriors.items()
    }
    quadratic_task = task_effect_approximation(
        orbit_posteriors["propagated"], orbit_posteriors["actual_barycenter"],
        orbit_posteriors["quadratic"], config.task_effect_floor,
    )
    cubic_task = task_effect_approximation(
        orbit_posteriors["propagated"], orbit_posteriors["actual_barycenter"],
        orbit_posteriors["quadratic_plus_cubic"], config.task_effect_floor,
    )
    return {
        "condition": condition,
        "transition_index": transition_index,
        "target_cut": target_cut,
        "regime": _regime_name(propagated_pass, actual_pass),
        "propagated_pass": propagated_pass,
        "actual_barycenter_pass": actual_pass,
        "character_decomposition": character_metrics(source_t),
        "residual_approximation": {
            "quadratic": residual_approximation(defect, quadratic),
            "quadratic_plus_cubic": residual_approximation(defect, quadratic + cubic),
        },
        "task_effect_approximation": {
            "quadratic": quadratic_task,
            "quadratic_plus_cubic": cubic_task,
        },
        "diagnostics": diagnostics,
    }


def _first_synthesis(records: Sequence[Mapping[str, Any]]) -> Optional[int]:
    return next((int(item["transition_index"]) for item in records if item["regime"] == "invariant_synthesis"), None)


def _control_reproduces(records: Sequence[Mapping[str, Any]], exact_front: Optional[int], config: CharacterCouplingConfig) -> bool:
    if exact_front is None:
        return False
    return any(
        item["regime"] == "invariant_synthesis"
        and abs(int(item["transition_index"]) - exact_front) <= 1
        and not item["task_effect_approximation"]["quadratic"]["degenerate"]
        and item["task_effect_approximation"]["quadratic"]["explained_fraction"] >= config.quadratic_fisher_threshold
        for item in records
    )


def analyze_cell(task: CircleTaskConfig, config: CharacterCouplingConfig, k: int, seed: int, output: Path, device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    bridge = _bridge_config(config)
    system, provenance = deck.load_source(task, bridge, k, seed, device)
    causal, reynolds = _load_comparators(config, k, seed, provenance["checkpoint_sha256"])
    transitions = koopman.one_step_transitions(config.first_blocks)
    regimes = {}
    for regime_index, regime in enumerate(REGIMES):
        dataset = deck.generate_exact_orbits(
            task, k=k, orbit_count=config.orbit_count,
            seed=seed + 2501 + 101 * regime_index, regime=regime,
        )
        captured = deck.capture_sequences(system, dataset, bridge, device)
        baseline_posterior = _continue_tensor(
            system, "full", torch.from_numpy(captured["full"]).to(device),
            task, config.continuation_batch_size,
        )
        baseline = deck.output_diagnostics(
            baseline_posterior.double().cpu().numpy(), dataset, include_topology=False
        )
        conditions = {}
        for condition_index, condition in enumerate(CONDITIONS):
            records = []
            for transition_index, (source_cut, target_cut) in enumerate(transitions):
                records.append(
                    analyze_transition(
                        system, dataset, captured[source_cut], captured[target_cut], target_cut,
                        transition_index, condition,
                        seed + 100_003 * regime_index + 10_007 * condition_index + transition_index,
                        baseline, task, config, device,
                    )
                )
            conditions[condition] = records
            print(f"k{k} seed {seed} {regime} {condition} complete", flush=True)
        exact_front = _first_synthesis(conditions["exact"])
        known_depth = _cut_depth(causal["fronts"][regime], transitions)
        exact_item = None if exact_front is None else conditions["exact"][exact_front]
        quadratic = None if exact_item is None else exact_item["task_effect_approximation"]["quadratic"]
        control_reproduction = {
            condition: _control_reproduces(conditions[condition], exact_front, config)
            for condition in CONDITIONS[1:]
        }
        regimes[regime] = {
            "baseline": baseline,
            "known_causal_front": causal["fronts"][regime],
            "known_causal_depth": known_depth,
            "synthesis_front_index": exact_front,
            "synthesis_target_cut": None if exact_front is None else transitions[exact_front][1],
            "front_distance": None if exact_front is None or known_depth is None else abs((exact_front + 1) - known_depth),
            "quadratic_at_synthesis": quadratic,
            "control_reproduction": control_reproduction,
            "conditions": conditions,
        }
    exact_sequences = {
        regime: [item["regime"] for item in regimes[regime]["conditions"]["exact"]]
        for regime in REGIMES
    }
    gates = {
        "shift_stable_causal_regime": exact_sequences["composition"] == exact_sequences["extrapolation"],
        "causal_front_localization": all(
            regimes[regime]["front_distance"] is not None and regimes[regime]["front_distance"] <= 1
            for regime in REGIMES
        ),
        "neutral_quadratic_sufficiency": all(
            regimes[regime]["quadratic_at_synthesis"] is not None
            and not regimes[regime]["quadratic_at_synthesis"]["degenerate"]
            and regimes[regime]["quadratic_at_synthesis"]["explained_fraction"] >= config.quadratic_fisher_threshold
            for regime in REGIMES
        ),
        "control_specificity": all(
            not reproduced
            for regime in REGIMES
            for reproduced in regimes[regime]["control_reproduction"].values()
        ),
    }
    root = output / "runs" / f"k{k}" / f"seed_{seed}"
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-reynolds-character-coupling-k{k}-seed{seed}",
        "status": "completed",
        "completed_at": _utc_now(),
        "k": k,
        "seed": seed,
        "configuration": asdict(config),
        "provenance": {**provenance, "causal_comparator": causal, "reynolds_comparator": reynolds},
        "regimes": regimes,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "implementation_sha256": _implementation_digest(),
        "analysis_seconds": time.perf_counter() - started,
    }
    _write_json(root / "result.json", result)
    return result


def aggregate(runs: Sequence[Mapping[str, Any]], config: CharacterCouplingConfig) -> dict[str, Any]:
    required = 4 if len(config.seeds) >= 5 else len(config.seeds)
    gate_names = (
        "shift_stable_causal_regime", "causal_front_localization",
        "neutral_quadratic_sufficiency", "control_specificity",
    )
    degrees = {}
    for k in config.degrees:
        selected = [item for item in runs if int(item["k"]) == k]
        counts = {gate: sum(bool(item["gates"][gate]) for item in selected) for gate in gate_names}
        degrees[str(k)] = {
            "gate_counts": counts,
            "gate_passes": {gate: count >= required for gate, count in counts.items()},
            "per_seed": {
                str(item["seed"]): {
                    "gates": item["gates"],
                    "regimes": {
                        regime: {
                            key: item["regimes"][regime][key]
                            for key in (
                                "known_causal_front", "known_causal_depth", "synthesis_front_index",
                                "synthesis_target_cut", "front_distance", "quadratic_at_synthesis",
                                "control_reproduction",
                            )
                        }
                        for regime in REGIMES
                    },
                }
                for item in selected
            },
        }
        degrees[str(k)]["confirmed"] = all(degrees[str(k)]["gate_passes"].values())
    return {
        "required_seed_count": required,
        "degrees": degrees,
        "confirmed": all(degrees[str(k)]["confirmed"] for k in config.degrees),
    }


def run_campaign(config: CharacterCouplingConfig, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    runs = []
    for k in config.degrees:
        for seed in config.seeds:
            path = output / "runs" / f"k{k}" / f"seed_{seed}" / "result.json"
            if path.is_file():
                existing = json.loads(path.read_text())
                if existing.get("implementation_sha256") == _implementation_digest():
                    runs.append(existing)
                    print(f"resuming {existing['experiment_id']}", flush=True)
                    continue
            result = analyze_cell(task, config, k, seed, output, device)
            runs.append(result)
            print(result["experiment_id"], f"{result['analysis_seconds']:.1f}s", flush=True)
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "implementation_sha256": _implementation_digest(),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        "summary": {"requested": len(config.degrees) * len(config.seeds), "completed": len(runs)},
        "aggregates": aggregate(runs, config),
        "method_boundaries": [
            "Finite differences estimate local derivatives along observed exact modes and are not interval certificates.",
            "The Fisher endpoint measures downstream task-posterior effect rather than residual equality.",
            "Within-orbit branch chance follows exactly from repeating one patched state across every fiber member.",
            "The character-neutral group average does not identify a unique coordinate basis inside an isotypic component.",
        ],
    }
    _write_json(output / "campaign_results.json", campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=CharacterCouplingConfig.source_root)
    parser.add_argument("--causal-root", default=CharacterCouplingConfig.causal_root)
    parser.add_argument("--koopman-root", default=CharacterCouplingConfig.koopman_root)
    parser.add_argument("--output", type=Path, default=Path("data/experiments/tinyllm_reynolds_character_coupling/20260806_d6_preregistered"))
    parser.add_argument("--seeds", default="7,17,29,41,53")
    parser.add_argument("--degrees", default="2,3")
    parser.add_argument("--orbit-count", type=int, default=64)
    parser.add_argument("--first-blocks", type=int, default=6)
    parser.add_argument("--finite-difference-eta", type=float, default=0.25)
    parser.add_argument("--quadratic-fisher-threshold", type=float, default=0.70)
    parser.add_argument("--task-effect-floor", type=float, default=1e-6)
    parser.add_argument("--activation-batch-size", type=int, default=256)
    parser.add_argument("--continuation-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args(argv)
    config = CharacterCouplingConfig(
        source_root=args.source_root, causal_root=args.causal_root, koopman_root=args.koopman_root,
        seeds=_ints(args.seeds), degrees=_ints(args.degrees), orbit_count=args.orbit_count,
        first_blocks=args.first_blocks, finite_difference_eta=args.finite_difference_eta,
        quadratic_fisher_threshold=args.quadratic_fisher_threshold,
        task_effect_floor=args.task_effect_floor, activation_batch_size=args.activation_batch_size,
        continuation_batch_size=args.continuation_batch_size, device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    print(args.output / "campaign_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
