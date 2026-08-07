#!/usr/bin/env python3
"""Causally ablate TinyLLM deck irreps at frozen quotient-synthesis fronts."""

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
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

import experiments.structure_net.tinyllm_deck_action_descrambler as deck
import experiments.structure_net.tinyllm_reynolds_character_coupling as coupling
import experiments.structure_net.tinyllm_reynolds_koopman as koopman
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-deck-irrep-fusion-ablation.v1"
HYPOTHESIS_ID = "tinyllm-deck-irrep-fusion-ablation-v1"
REGIMES = ("composition", "extrapolation")


@dataclass(frozen=True)
class IrrepFusionConfig:
    source_root: str = "data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered"
    character_root: str = "data/experiments/tinyllm_reynolds_character_coupling/20260806_d6_preregistered"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    degrees: tuple[int, ...] = (2, 3)
    orbit_count: int = 64
    first_blocks: int = 6
    amplitude_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    phase_steps: int = 12
    task_effect_floor: float = 1e-6
    radial_threshold: float = 0.10
    phase_sensitive_threshold: float = 0.25
    control_reproduction_threshold: float = 0.70
    group_state_tolerance: float = 1e-5
    group_posterior_tolerance: float = 1e-7
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
        if tuple(sorted(self.amplitude_grid)) != self.amplitude_grid:
            raise ValueError("amplitude grid must be sorted")
        if not self.amplitude_grid or self.amplitude_grid[0] != 0.0 or self.amplitude_grid[-1] != 1.0:
            raise ValueError("amplitude grid must include endpoints 0 and 1")
        if self.phase_steps < 6 or self.phase_steps % 3:
            raise ValueError("phase steps must be a multiple of three and at least six")
        if not 0.0 <= self.radial_threshold < self.phase_sensitive_threshold:
            raise ValueError("phase phenotype thresholds are invalid")


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
    value = hashlib.sha256()
    for path in (Path(__file__), Path(coupling.__file__), Path(deck.__file__)):
        value.update(str(path).encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def _bridge_config(config: IrrepFusionConfig) -> deck.DeckDescramblerConfig:
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


def deck_fourier_components(sheets: torch.Tensor) -> torch.Tensor:
    """Return c_r with shape (orbit, character, flattened-state)."""
    orbit_count, k = sheets.shape[:2]
    flat = sheets.reshape(orbit_count, k, -1).to(torch.float64)
    member = torch.arange(k, device=sheets.device, dtype=torch.float64)
    character = torch.arange(k, device=sheets.device, dtype=torch.float64)
    weights = torch.exp(-2j * math.pi * character[:, None] * member[None, :] / k)
    return torch.einsum("rj,njd->nrd", weights, flat.to(torch.complex128)) / k


def reconstruct_character_phase(
    sheets: torch.Tensor,
    theta: float,
) -> tuple[torch.Tensor, float]:
    """Rotate the real C2/C3 carrier and reconstruct real deck sheets."""
    orbit_count, k = sheets.shape[:2]
    if k not in (2, 3):
        raise ValueError("only C2 and C3 carriers are supported")
    components = deck_fourier_components(sheets).clone()
    if k == 2:
        components[:, 1] *= complex(math.cos(theta), math.sin(theta))
    else:
        components[:, 1] *= complex(math.cos(theta), math.sin(theta))
        components[:, 2] *= complex(math.cos(theta), -math.sin(theta))
    member = torch.arange(k, device=sheets.device, dtype=torch.float64)
    character = torch.arange(k, device=sheets.device, dtype=torch.float64)
    basis = torch.exp(2j * math.pi * character[:, None] * member[None, :] / k)
    reconstructed = torch.einsum("rj,nrd->njd", basis, components)
    imaginary = float(reconstructed.imag.abs().max().cpu())
    real = reconstructed.real.reshape_as(sheets).to(dtype=sheets.dtype)
    return real, imaginary


def substitute_orbit_carrier(sheets: torch.Tensor) -> torch.Tensor:
    """Keep each barycenter while cyclically substituting a norm-matched carrier."""
    barycenter = sheets.mean(1, keepdim=True)
    delta = sheets - barycenter
    substitute = delta.roll(1, dims=0)
    axes = tuple(range(1, delta.ndim))
    target_norm = torch.square(delta).sum(dim=axes, keepdim=True).sqrt()
    source_norm = torch.square(substitute).sum(dim=axes, keepdim=True).sqrt().clamp_min(1e-12)
    substitute = substitute * target_norm / source_norm
    return barycenter + substitute


def phase_phenotype(
    sensitivity: Optional[float],
    radial_threshold: float,
    phase_sensitive_threshold: float,
) -> str:
    if sensitivity is None:
        return "degenerate"
    if sensitivity <= radial_threshold:
        return "radial"
    if sensitivity >= phase_sensitive_threshold:
        return "finite_group_phase_sensitive"
    return "mixed"


def _effect_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    # Float32 Bhattacharyya affinities can make identical posteriors appear one
    # machine epsilon apart after arccos.  The group contract needs a genuine
    # numerical-zero check, so compute all Fisher endpoints in float64.
    return float(coupling.fisher_rao_squared(left.double(), right.double()).mean().cpu())


def _effect_preservation(
    posterior: torch.Tensor,
    exact: torch.Tensor,
    propagated: torch.Tensor,
    floor: float,
) -> dict[str, Any]:
    denominator = _effect_distance(propagated, exact)
    error = _effect_distance(posterior, exact)
    degenerate = denominator < floor
    return {
        "exact_fisher_effect": denominator,
        "remaining_fisher_error": error,
        "preserved_fraction": None if degenerate else 1.0 - error / denominator,
        "degenerate": degenerate,
    }


def _alpha_name(alpha: float) -> str:
    return f"alpha_{alpha:.2f}".replace(".", "p")


def _phase_name(index: int) -> str:
    return f"phase_{index:02d}"


def _load_frozen_result(
    config: IrrepFusionConfig,
    k: int,
    seed: int,
    checkpoint_sha: str,
) -> tuple[dict[str, Any], Path]:
    path = Path(config.character_root) / "runs" / f"k{k}" / f"seed_{seed}" / "result.json"
    result = json.loads(path.read_text())
    if result.get("status") != "completed":
        raise ValueError(f"incomplete character comparator {path}")
    if result.get("k") != k or result.get("seed") != seed:
        raise ValueError(f"identity mismatch in {path}")
    if result["provenance"]["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError(f"checkpoint mismatch in {path}")
    if (
        int(result["configuration"]["orbit_count"]) != config.orbit_count
        and not config.allow_underpowered
    ):
        raise ValueError("orbit count must match the frozen character campaign")
    if int(result["configuration"]["first_blocks"]) != config.first_blocks:
        raise ValueError("cut set must match the frozen character campaign")
    return result, path


def _posterior_batches(
    system: Any,
    target_cut: str,
    patches: Mapping[str, torch.Tensor],
    k: int,
    task: CircleTaskConfig,
    config: IrrepFusionConfig,
) -> dict[str, torch.Tensor]:
    names = tuple(patches)
    lengths = [len(patches[name]) * k for name in names]
    combined = torch.cat([coupling._repeat_patch(patches[name], k) for name in names])
    posterior = coupling._continue_tensor(
        system, target_cut, combined, task, config.continuation_batch_size
    )
    return dict(zip(names, posterior.split(lengths)))


@torch.no_grad()
def analyze_regime(
    system: Any,
    dataset: deck.OrbitDataset,
    captured: Mapping[str, np.ndarray],
    frozen: Mapping[str, Any],
    regime: str,
    task: CircleTaskConfig,
    config: IrrepFusionConfig,
    device: torch.device,
) -> dict[str, Any]:
    k = dataset.k
    orbit_count = dataset.orbit_count
    transitions = koopman.one_step_transitions(config.first_blocks)
    front = frozen["regimes"][regime]["synthesis_front_index"]
    if front is None:
        raise ValueError(f"frozen {regime} cell has no synthesis front")
    transition_index = int(front)
    source_cut, target_cut = transitions[transition_index]
    source = torch.from_numpy(captured[source_cut]).to(device)
    source = source.reshape((orbit_count, k) + source.shape[1:])
    captured_target = torch.from_numpy(captured[target_cut]).to(device)
    captured_target = captured_target.reshape((orbit_count, k) + captured_target.shape[1:])
    apply = lambda value: coupling._apply_sublayer(system, transition_index, value)

    barycenter = source.mean(1)
    delta = source - barycenter[:, None]
    propagated = apply(barycenter)
    exact_members = apply(source.flatten(0, 1)).reshape_as(source)
    exact = exact_members.mean(1)
    transition_error = float(
        torch.linalg.vector_norm(exact_members - captured_target)
        / torch.linalg.vector_norm(captured_target).clamp_min(1e-12)
    )

    patches: dict[str, torch.Tensor] = {}
    for alpha in config.amplitude_grid:
        members = barycenter[:, None] + alpha * delta
        patches[_alpha_name(alpha)] = apply(members.flatten(0, 1)).reshape_as(source).mean(1)
    exact_defect = exact - propagated
    for alpha in config.amplitude_grid[1:-1]:
        patches[f"quadratic_{_alpha_name(alpha)}"] = propagated + alpha**2 * exact_defect

    substituted = substitute_orbit_carrier(source)
    patches["substituted_carrier"] = apply(substituted.flatten(0, 1)).reshape_as(source).mean(1)

    phase_records: list[dict[str, Any]] = []
    phase_indices = range(config.phase_steps) if k == 3 else (0, config.phase_steps // 2)
    for index in phase_indices:
        theta = 2.0 * math.pi * index / config.phase_steps
        rotated, imaginary = reconstruct_character_phase(source, theta)
        phase_value = apply(rotated.flatten(0, 1)).reshape_as(source).mean(1)
        name = _phase_name(index)
        patches[name] = phase_value
        phase_records.append({
            "name": name,
            "index": index,
            "theta": theta,
            "maximum_reconstruction_imaginary": imaginary,
        })

    posteriors = _posterior_batches(system, target_cut, patches, k, task, config)
    baseline_posterior = coupling._continue_tensor(
        system, "full", torch.from_numpy(captured["full"]).to(device),
        task, config.continuation_batch_size,
    )
    baseline = deck.output_diagnostics(
        baseline_posterior.double().cpu().numpy(), dataset, include_topology=False
    )
    diagnostics: dict[str, Any] = {}
    causal_passes: dict[str, bool] = {}
    orbit_posteriors: dict[str, torch.Tensor] = {}
    for name, posterior in posteriors.items():
        diagnostics[name] = coupling._diagnostics(posterior, dataset)
        causal_passes[name], _ = coupling._causal_classification(diagnostics[name], baseline, k)
        orbit_posteriors[name] = posterior.reshape(orbit_count, k, -1)[:, 0]

    zero_name, exact_name = _alpha_name(0.0), _alpha_name(1.0)
    zero_posterior = orbit_posteriors[zero_name]
    exact_posterior = orbit_posteriors[exact_name]
    effect = _effect_distance(zero_posterior, exact_posterior)
    degenerate = effect < config.task_effect_floor

    amplitude = []
    first_passing_alpha: Optional[float] = None
    for alpha in config.amplitude_grid:
        name = _alpha_name(alpha)
        preservation = _effect_preservation(
            orbit_posteriors[name], exact_posterior, zero_posterior, config.task_effect_floor
        )
        if first_passing_alpha is None and causal_passes[name]:
            first_passing_alpha = alpha
        record: dict[str, Any] = {
            "alpha": alpha,
            "name": name,
            "causal_pass": causal_passes[name],
            "effect_preservation": preservation,
            "diagnostics": diagnostics[name],
        }
        if 0.0 < alpha < 1.0:
            qname = f"quadratic_{name}"
            record["quadratic_homogeneity"] = coupling.task_effect_approximation(
                zero_posterior,
                orbit_posteriors[name],
                orbit_posteriors[qname],
                config.task_effect_floor,
            )
        amplitude.append(record)

    substituted_effect = _effect_preservation(
        orbit_posteriors["substituted_carrier"], exact_posterior,
        zero_posterior, config.task_effect_floor,
    )
    substituted_reproduces = bool(
        causal_passes["substituted_carrier"]
        and not substituted_effect["degenerate"]
        and substituted_effect["preserved_fraction"] >= config.control_reproduction_threshold
    )

    deck_indices = {0, config.phase_steps // 3, 2 * config.phase_steps // 3} if k == 3 else {0, config.phase_steps // 2}
    nondeck_sensitivities = []
    group_contract_pass = True
    for record in phase_records:
        name = record["name"]
        state_relative_error = float(
            torch.linalg.vector_norm(patches[name] - exact)
            / torch.linalg.vector_norm(exact).clamp_min(1e-12)
        )
        posterior_distance = _effect_distance(orbit_posteriors[name], exact_posterior)
        record.update({
            "is_deck_rotation": record["index"] in deck_indices,
            "state_relative_error_from_exact": state_relative_error,
            "posterior_fisher_distance_from_exact": posterior_distance,
            "normalized_phase_sensitivity": None if degenerate else posterior_distance / effect,
            "causal_pass": causal_passes[name],
            "diagnostics": diagnostics[name],
        })
        if record["is_deck_rotation"]:
            group_contract_pass = bool(
                group_contract_pass
                and state_relative_error <= config.group_state_tolerance
                and posterior_distance <= config.group_posterior_tolerance
                and record["maximum_reconstruction_imaginary"] <= 1e-6
            )
        elif not degenerate:
            nondeck_sensitivities.append(float(record["normalized_phase_sensitivity"]))

    sensitivity = None
    if k == 3 and nondeck_sensitivities:
        sensitivity = float(np.median(nondeck_sensitivities))
    phenotype = None if k == 2 else phase_phenotype(
        sensitivity, config.radial_threshold, config.phase_sensitive_threshold
    )
    charged_necessity = bool(not causal_passes[zero_name] and causal_passes[exact_name])
    return {
        "regime": regime,
        "frozen_transition_index": transition_index,
        "source_cut": source_cut,
        "target_cut": target_cut,
        "captured_transition_relative_error": transition_error,
        "baseline": baseline,
        "exact_causal_effect": effect,
        "effect_degenerate": degenerate,
        "first_passing_alpha": first_passing_alpha,
        "amplitude_path": amplitude,
        "substituted_carrier": {
            "causal_pass": causal_passes["substituted_carrier"],
            "effect_preservation": substituted_effect,
            "reproduces_synthesis": substituted_reproduces,
            "diagnostics": diagnostics["substituted_carrier"],
        },
        "phase_intervention": {
            "records": phase_records,
            "median_normalized_continuous_phase_sensitivity": sensitivity,
            "phenotype": phenotype,
        },
        "gates": {
            "exact_group_contract": group_contract_pass,
            "charged_mode_necessity": charged_necessity,
            "orbit_specific_carrier": not substituted_reproduces,
            "finite_c3_phase_mechanism": bool(
                k == 3 and not degenerate and phenotype == "finite_group_phase_sensitive"
            ),
        },
    }


def analyze_cell(
    task: CircleTaskConfig,
    config: IrrepFusionConfig,
    k: int,
    seed: int,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    bridge = _bridge_config(config)
    system, provenance = deck.load_source(task, bridge, k, seed, device)
    frozen, frozen_path = _load_frozen_result(
        config, k, seed, provenance["checkpoint_sha256"]
    )
    regimes = {}
    for regime_index, regime in enumerate(REGIMES):
        dataset = deck.generate_exact_orbits(
            task,
            k=k,
            orbit_count=config.orbit_count,
            seed=seed + 2501 + 101 * regime_index,
            regime=regime,
        )
        captured = deck.capture_sequences(system, dataset, bridge, device)
        regimes[regime] = analyze_regime(
            system, dataset, captured, frozen, regime, task, config, device
        )
        print(f"k{k} seed {seed} {regime} complete", flush=True)

    gates = {
        "exact_group_contract": all(
            regimes[regime]["gates"]["exact_group_contract"] for regime in REGIMES
        ),
        "charged_mode_necessity": all(
            regimes[regime]["gates"]["charged_mode_necessity"] for regime in REGIMES
        ),
        "orbit_specific_carrier": all(
            regimes[regime]["gates"]["orbit_specific_carrier"] for regime in REGIMES
        ),
    }
    if k == 3:
        gates.update({
            "finite_c3_phase_mechanism": all(
                regimes[regime]["gates"]["finite_c3_phase_mechanism"] for regime in REGIMES
            ),
            "shift_stable_phase_phenotype": (
                regimes["composition"]["phase_intervention"]["phenotype"]
                == regimes["extrapolation"]["phase_intervention"]["phenotype"]
                != "degenerate"
            ),
        })
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-deck-irrep-fusion-ablation-k{k}-seed{seed}",
        "status": "completed",
        "completed_at": _utc_now(),
        "k": k,
        "seed": seed,
        "configuration": asdict(config),
        "provenance": {
            **provenance,
            "frozen_character_result": str(frozen_path),
            "frozen_character_result_sha256": _sha256(frozen_path),
            "frozen_character_implementation_sha256": frozen["implementation_sha256"],
        },
        "regimes": regimes,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "implementation_sha256": _implementation_digest(),
        "analysis_seconds": time.perf_counter() - started,
    }
    _write_json(output / "runs" / f"k{k}" / f"seed_{seed}" / "result.json", result)
    return result


def aggregate(runs: Sequence[Mapping[str, Any]], config: IrrepFusionConfig) -> dict[str, Any]:
    required = 4 if len(config.seeds) >= 5 else len(config.seeds)
    degrees = {}
    for k in config.degrees:
        selected = [item for item in runs if int(item["k"]) == k]
        gate_names = (
            "exact_group_contract",
            "charged_mode_necessity",
            "orbit_specific_carrier",
        ) + (("finite_c3_phase_mechanism", "shift_stable_phase_phenotype") if k == 3 else ())
        counts = {gate: sum(bool(item["gates"][gate]) for item in selected) for gate in gate_names}
        gate_passes = {gate: count >= required for gate, count in counts.items()}
        degrees[str(k)] = {
            "gate_counts": counts,
            "gate_passes": gate_passes,
            "confirmed": all(gate_passes.values()),
            "per_seed": {
                str(item["seed"]): {
                    "gates": item["gates"],
                    "regimes": {
                        regime: {
                            "frozen_transition_index": item["regimes"][regime]["frozen_transition_index"],
                            "target_cut": item["regimes"][regime]["target_cut"],
                            "first_passing_alpha": item["regimes"][regime]["first_passing_alpha"],
                            "substituted_carrier_reproduces": item["regimes"][regime]["substituted_carrier"]["reproduces_synthesis"],
                            "phase_sensitivity": item["regimes"][regime]["phase_intervention"]["median_normalized_continuous_phase_sensitivity"],
                            "phase_phenotype": item["regimes"][regime]["phase_intervention"]["phenotype"],
                        }
                        for regime in REGIMES
                    },
                }
                for item in selected
            },
        }
    return {
        "required_seed_count": required,
        "degrees": degrees,
        "confirmed": all(degrees[str(k)]["confirmed"] for k in config.degrees),
    }


def run_campaign(config: IrrepFusionConfig, output: Path) -> dict[str, Any]:
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
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        "summary": {
            "requested": len(config.degrees) * len(config.seeds),
            "completed": len(runs),
            "failed": 0,
        },
        "aggregates": aggregate(runs, config),
        "method_boundaries": [
            "Continuous C3 phase rotations are off-orbit interventions within the observed real isotypic carrier.",
            "The diagnostic uses frozen synthesis fronts and does not reselect them on these outcomes.",
            "Fisher endpoints measure downstream task effects rather than residual equality.",
            "Repeated barycenter patches imply within-orbit branch chance but not global branch erasure.",
        ],
    }
    _write_json(output / "campaign_results.json", campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=IrrepFusionConfig.source_root)
    parser.add_argument("--character-root", default=IrrepFusionConfig.character_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/tinyllm_irrep_fusion_ablation/20260806_d6_preregistered"),
    )
    parser.add_argument("--seeds", default="7,17,29,41,53")
    parser.add_argument("--degrees", default="2,3")
    parser.add_argument("--orbit-count", type=int, default=64)
    parser.add_argument("--first-blocks", type=int, default=6)
    parser.add_argument("--amplitude-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--phase-steps", type=int, default=12)
    parser.add_argument("--task-effect-floor", type=float, default=1e-6)
    parser.add_argument("--radial-threshold", type=float, default=0.10)
    parser.add_argument("--phase-sensitive-threshold", type=float, default=0.25)
    parser.add_argument("--control-reproduction-threshold", type=float, default=0.70)
    parser.add_argument("--group-state-tolerance", type=float, default=1e-5)
    parser.add_argument("--group-posterior-tolerance", type=float, default=1e-7)
    parser.add_argument("--activation-batch-size", type=int, default=256)
    parser.add_argument("--continuation-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args(argv)
    config = IrrepFusionConfig(
        source_root=args.source_root,
        character_root=args.character_root,
        seeds=_ints(args.seeds),
        degrees=_ints(args.degrees),
        orbit_count=args.orbit_count,
        first_blocks=args.first_blocks,
        amplitude_grid=_floats(args.amplitude_grid),
        phase_steps=args.phase_steps,
        task_effect_floor=args.task_effect_floor,
        radial_threshold=args.radial_threshold,
        phase_sensitive_threshold=args.phase_sensitive_threshold,
        control_reproduction_threshold=args.control_reproduction_threshold,
        group_state_tolerance=args.group_state_tolerance,
        group_posterior_tolerance=args.group_posterior_tolerance,
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
