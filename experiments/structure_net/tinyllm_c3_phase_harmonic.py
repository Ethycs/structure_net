#!/usr/bin/env python3
"""Causally decompose TinyLLM C3 carrier-phase responses into allowed harmonics."""

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
import experiments.structure_net.tinyllm_irrep_fusion_ablation as irrep
import experiments.structure_net.tinyllm_reynolds_character_coupling as coupling
import experiments.structure_net.tinyllm_reynolds_koopman as koopman
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-c3-phase-harmonic.v1"
HYPOTHESIS_ID = "tinyllm-c3-phase-harmonic-fusion-v1"
REGIMES = ("composition", "extrapolation")


@dataclass(frozen=True)
class C3PhaseHarmonicConfig:
    source_root: str = "data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered"
    irrep_root: str = "data/experiments/tinyllm_irrep_fusion_ablation/20260806_d6_preregistered"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    primary_phase_seeds: tuple[int, ...] = (17, 29, 53)
    orbit_count: int = 64
    first_blocks: int = 6
    phase_steps: int = 24
    task_effect_floor: float = 1e-6
    effect_threshold: float = 0.70
    periodicity_tolerance: float = 1e-5
    forbidden_energy_tolerance: float = 1e-8
    reconstruction_tolerance: float = 1e-5
    activation_batch_size: int = 256
    continuation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if len(self.seeds) < 5 and not self.allow_underpowered:
            raise ValueError("confirmatory campaign requires five seeds")
        if not set(self.primary_phase_seeds).issubset(self.seeds):
            raise ValueError("primary phase seeds must be included in seeds")
        if self.orbit_count < 8:
            raise ValueError("at least eight exact orbits are required")
        if self.phase_steps != 24:
            raise ValueError("the preregistered phase grid has exactly 24 steps")
        if not 0.0 < self.effect_threshold <= 1.0:
            raise ValueError("effect threshold must be in (0,1]")


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
    for path in (Path(__file__), Path(irrep.__file__), Path(coupling.__file__), Path(deck.__file__)):
        value.update(str(path).encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def _bridge_config(config: C3PhaseHarmonicConfig) -> deck.DeckDescramblerConfig:
    return deck.DeckDescramblerConfig(
        source_root=config.source_root,
        seeds=config.seeds,
        degrees=(3,),
        fit_orbits=config.orbit_count,
        evaluation_orbits=config.orbit_count,
        map_points=24,
        first_blocks=config.first_blocks,
        activation_batch_size=config.activation_batch_size,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )


def phase_dft(values: torch.Tensor) -> torch.Tensor:
    """Return unit-normalized complex DFT coefficients along phase axis zero."""
    return torch.fft.fft(values.double(), dim=0) / values.shape[0]


def allowed_harmonic_reconstructions(
    values: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Reconstruct theta=0 from nested C3-allowed Fourier pairs."""
    steps = values.shape[0]
    if steps != 24:
        raise ValueError("expected the preregistered 24-point phase grid")
    coefficients = phase_dft(values)
    reconstructions: dict[str, torch.Tensor] = {"phase_twirl": coefficients[0].real}
    current = coefficients[0].real.clone()
    for frequency in (3, 6, 9, 12):
        contribution = coefficients[frequency].real if frequency == steps // 2 else (
            coefficients[frequency] + coefficients[-frequency]
        ).real
        reconstructions[f"twirl_plus_{frequency}"] = coefficients[0].real + contribution
        current = current + contribution
        reconstructions[f"allowed_prefix_{frequency}"] = current.clone()
    reconstructions["full_dft"] = coefficients.sum(0).real
    return reconstructions, coefficients


def spectral_metrics(values: torch.Tensor, coefficients: torch.Tensor) -> dict[str, Any]:
    steps = values.shape[0]
    flat = coefficients.reshape(steps, -1)
    energies = torch.square(torch.abs(flat)).sum(-1)
    variation = float(energies[1:].sum().cpu())
    forbidden = float(sum(energies[index] for index in range(1, steps) if index % 3).cpu())
    allowed_pairs = {}
    for frequency in (3, 6, 9, 12):
        energy = energies[frequency]
        if frequency != steps // 2:
            energy = energy + energies[-frequency]
        allowed_pairs[str(frequency)] = float(energy.cpu()) / max(1e-15, variation)
    reconstructed = coefficients.sum(0).real
    reconstruction_error = float(
        torch.linalg.vector_norm(reconstructed - values[0].double())
        / torch.linalg.vector_norm(values[0].double()).clamp_min(1e-15)
    )
    deck_stride = steps // 3
    periodicity = []
    for index in range(steps):
        reference = values[index]
        translated = values[(index + deck_stride) % steps]
        periodicity.append(float(
            torch.linalg.vector_norm(translated - reference)
            / torch.linalg.vector_norm(reference).clamp_min(1e-12)
        ))
    return {
        "variation_energy": variation,
        "forbidden_variation_energy_fraction": forbidden / max(1e-15, variation),
        "allowed_pair_energy_fractions": allowed_pairs,
        "theta_zero_reconstruction_relative_error": reconstruction_error,
        "maximum_deck_periodicity_relative_error": max(periodicity),
    }


def _load_irrep_result(
    config: C3PhaseHarmonicConfig,
    seed: int,
    checkpoint_sha: str,
) -> tuple[dict[str, Any], Path]:
    path = Path(config.irrep_root) / "runs" / "k3" / f"seed_{seed}" / "result.json"
    result = json.loads(path.read_text())
    if (
        result.get("status") != "completed"
        or result.get("schema_version") != irrep.SCHEMA_VERSION
        or int(result.get("k", -1)) != 3
        or int(result.get("seed", -1)) != seed
        or result["provenance"]["checkpoint_sha256"] != checkpoint_sha
    ):
        raise ValueError(f"invalid irrep comparator {path}")
    if int(result["configuration"]["orbit_count"]) != config.orbit_count and not config.allow_underpowered:
        raise ValueError("orbit count must match the frozen irrep campaign")
    for regime in REGIMES:
        expected = seed in config.primary_phase_seeds
        observed = result["regimes"][regime]["phase_intervention"]["phenotype"] == "finite_group_phase_sensitive"
        if expected and not observed:
            raise ValueError(f"primary stratum mismatch for seed {seed} {regime}")
    return result, path


def _condition_posteriors(
    system: Any,
    target_cut: str,
    patches: Mapping[str, torch.Tensor],
    task: CircleTaskConfig,
    config: C3PhaseHarmonicConfig,
) -> dict[str, torch.Tensor]:
    names = tuple(patches)
    lengths = [len(patches[name]) * 3 for name in names]
    combined = torch.cat([coupling._repeat_patch(patches[name].float(), 3) for name in names])
    posterior = coupling._continue_tensor(
        system, target_cut, combined, task, config.continuation_batch_size
    )
    return dict(zip(names, posterior.split(lengths)))


def _effect_record(
    posterior: torch.Tensor,
    exact: torch.Tensor,
    twirl: torch.Tensor,
    floor: float,
) -> dict[str, Any]:
    effect = irrep._effect_distance(twirl, exact)
    remaining = irrep._effect_distance(posterior, exact)
    degenerate = effect < floor
    return {
        "finite_phase_fisher_effect": effect,
        "remaining_fisher_error": remaining,
        "explained_fraction": None if degenerate else 1.0 - remaining / effect,
        "degenerate": degenerate,
    }


@torch.no_grad()
def analyze_regime(
    system: Any,
    dataset: deck.OrbitDataset,
    captured: Mapping[str, np.ndarray],
    frozen: Mapping[str, Any],
    regime: str,
    task: CircleTaskConfig,
    config: C3PhaseHarmonicConfig,
    device: torch.device,
) -> dict[str, Any]:
    orbit_count = dataset.orbit_count
    transition_index = int(frozen["regimes"][regime]["frozen_transition_index"])
    source_cut, target_cut = koopman.one_step_transitions(config.first_blocks)[transition_index]
    source = torch.from_numpy(captured[source_cut]).to(device)
    source = source.reshape((orbit_count, 3) + source.shape[1:])
    apply = lambda value: coupling._apply_sublayer(system, transition_index, value)

    phase_values = []
    maximum_imaginary = 0.0
    for index in range(config.phase_steps):
        theta = 2.0 * math.pi * index / config.phase_steps
        rotated, imaginary = irrep.reconstruct_character_phase(source, theta)
        maximum_imaginary = max(maximum_imaginary, imaginary)
        phase_values.append(apply(rotated.flatten(0, 1)).reshape_as(source).mean(1))
    phase_tensor = torch.stack(phase_values)
    reconstructions, coefficients = allowed_harmonic_reconstructions(phase_tensor)
    spectrum = spectral_metrics(phase_tensor, coefficients)

    captured_target = torch.from_numpy(captured[target_cut]).to(device)
    captured_target = captured_target.reshape((orbit_count, 3) + captured_target.shape[1:]).mean(1)
    exact_transition_error = float(
        torch.linalg.vector_norm(phase_tensor[0] - captured_target)
        / torch.linalg.vector_norm(captured_target).clamp_min(1e-12)
    )

    patches = {"exact_phase_0": phase_tensor[0], **reconstructions}
    posteriors = _condition_posteriors(system, target_cut, patches, task, config)
    phase_posteriors_flat = coupling._continue_tensor(
        system,
        target_cut,
        torch.cat([coupling._repeat_patch(value, 3) for value in phase_values]),
        task,
        config.continuation_batch_size,
    )
    phase_posteriors = phase_posteriors_flat.reshape(config.phase_steps, orbit_count, 3, -1)[:, :, 0]
    posterior_spectrum = spectral_metrics(phase_posteriors, phase_dft(phase_posteriors))

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
    diagnostics = {}
    causal_passes = {}
    orbit_posteriors = {}
    for name, posterior in posteriors.items():
        diagnostics[name] = coupling._diagnostics(posterior, dataset)
        causal_passes[name], _ = coupling._causal_classification(diagnostics[name], baseline, 3)
        orbit_posteriors[name] = posterior.reshape(orbit_count, 3, -1)[:, 0]

    exact_posterior = orbit_posteriors["exact_phase_0"]
    twirl_posterior = orbit_posteriors["phase_twirl"]
    effects = {
        name: _effect_record(value, exact_posterior, twirl_posterior, config.task_effect_floor)
        for name, value in orbit_posteriors.items()
    }
    finite_effect = effects["phase_twirl"]["finite_phase_fisher_effect"]
    eligible = bool(causal_passes["exact_phase_0"] and finite_effect >= config.task_effect_floor)
    necessity = bool(eligible and not causal_passes["phase_twirl"])
    first = effects["allowed_prefix_3"]
    first_sufficient = bool(
        eligible
        and causal_passes["allowed_prefix_3"]
        and not first["degenerate"]
        and first["explained_fraction"] >= config.effect_threshold
    )
    prefix_order = ("phase_twirl", "allowed_prefix_3", "allowed_prefix_6", "allowed_prefix_9", "allowed_prefix_12")
    minimal_prefix = None
    for name in prefix_order:
        if name == "phase_twirl":
            sufficient = causal_passes[name]
        else:
            effect = effects[name]
            sufficient = bool(
                causal_passes[name]
                and not effect["degenerate"]
                and effect["explained_fraction"] >= config.effect_threshold
            )
        if sufficient:
            minimal_prefix = name
            break
    contract = bool(
        maximum_imaginary <= 1e-6
        and spectrum["maximum_deck_periodicity_relative_error"] <= config.periodicity_tolerance
        and spectrum["forbidden_variation_energy_fraction"] <= config.forbidden_energy_tolerance
        and spectrum["theta_zero_reconstruction_relative_error"] <= config.reconstruction_tolerance
    )
    condition_records = {
        name: {
            "causal_pass": causal_passes[name],
            "effect": effects[name],
            "diagnostics": diagnostics[name],
        }
        for name in patches
    }
    return {
        "regime": regime,
        "transition_index": transition_index,
        "source_cut": source_cut,
        "target_cut": target_cut,
        "exact_transition_relative_error": exact_transition_error,
        "maximum_reconstruction_imaginary": maximum_imaginary,
        "state_spectrum": spectrum,
        "posterior_spectrum": posterior_spectrum,
        "baseline": baseline,
        "finite_phase_fisher_effect": finite_effect,
        "minimal_sufficient_prefix": minimal_prefix,
        "conditions": condition_records,
        "gates": {
            "spectral_group_contract": contract,
            "eligible_endpoint": eligible,
            "finite_phase_necessity": necessity,
            "first_harmonic_sufficiency": first_sufficient,
        },
    }


def analyze_cell(
    task: CircleTaskConfig,
    config: C3PhaseHarmonicConfig,
    seed: int,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    bridge = _bridge_config(config)
    system, provenance = deck.load_source(task, bridge, 3, seed, device)
    frozen, frozen_path = _load_irrep_result(config, seed, provenance["checkpoint_sha256"])
    regimes = {}
    for regime_index, regime in enumerate(REGIMES):
        dataset = deck.generate_exact_orbits(
            task,
            k=3,
            orbit_count=config.orbit_count,
            seed=seed + 2501 + 101 * regime_index,
            regime=regime,
        )
        captured = deck.capture_sequences(system, dataset, bridge, device)
        regimes[regime] = analyze_regime(
            system, dataset, captured, frozen, regime, task, config, device
        )
        print(f"k3 seed {seed} {regime} complete", flush=True)
    gates = {
        "spectral_group_contract": all(regimes[r]["gates"]["spectral_group_contract"] for r in REGIMES),
        "eligible_endpoint": all(regimes[r]["gates"]["eligible_endpoint"] for r in REGIMES),
        "finite_phase_necessity": all(regimes[r]["gates"]["finite_phase_necessity"] for r in REGIMES),
        "first_harmonic_sufficiency": all(regimes[r]["gates"]["first_harmonic_sufficiency"] for r in REGIMES),
        "shift_stable_minimal_prefix": (
            regimes["composition"]["minimal_sufficient_prefix"]
            == regimes["extrapolation"]["minimal_sufficient_prefix"]
            is not None
        ),
    }
    primary = seed in config.primary_phase_seeds
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-c3-phase-harmonic-seed{seed}",
        "status": "completed",
        "completed_at": _utc_now(),
        "seed": seed,
        "primary_phase_stratum": primary,
        "configuration": asdict(config),
        "provenance": {
            **provenance,
            "frozen_irrep_result": str(frozen_path),
            "frozen_irrep_result_sha256": _sha256(frozen_path),
            "frozen_irrep_implementation_sha256": frozen["implementation_sha256"],
        },
        "regimes": regimes,
        "gates": gates,
        "primary_gates_passed": bool(primary and all(gates.values())),
        "implementation_sha256": _implementation_digest(),
        "analysis_seconds": time.perf_counter() - started,
    }
    _write_json(output / "runs" / f"seed_{seed}" / "result.json", result)
    return result


def aggregate(runs: Sequence[Mapping[str, Any]], config: C3PhaseHarmonicConfig) -> dict[str, Any]:
    selected = [run for run in runs if bool(run["primary_phase_stratum"])]
    mechanism_gates = (
        "eligible_endpoint",
        "finite_phase_necessity",
        "first_harmonic_sufficiency",
        "shift_stable_minimal_prefix",
    )
    contract_count = sum(bool(run["gates"]["spectral_group_contract"]) for run in runs)
    selected_counts = {
        gate: sum(bool(run["gates"][gate]) for run in selected) for gate in mechanism_gates
    }
    contract_pass = contract_count == len(runs)
    selected_passes = {gate: count == len(selected) for gate, count in selected_counts.items()}
    return {
        "spectral_group_contract_count": contract_count,
        "spectral_group_contract_pass": contract_pass,
        "primary_seed_count": len(selected),
        "primary_gate_counts": selected_counts,
        "primary_gate_passes": selected_passes,
        "confirmed": bool(contract_pass and all(selected_passes.values())),
        "per_seed": {
            str(run["seed"]): {
                "primary_phase_stratum": run["primary_phase_stratum"],
                "gates": run["gates"],
                "regimes": {
                    regime: {
                        "target_cut": run["regimes"][regime]["target_cut"],
                        "finite_phase_fisher_effect": run["regimes"][regime]["finite_phase_fisher_effect"],
                        "minimal_sufficient_prefix": run["regimes"][regime]["minimal_sufficient_prefix"],
                        "first_harmonic_effect_explained": run["regimes"][regime]["conditions"]["allowed_prefix_3"]["effect"]["explained_fraction"],
                        "twirl_causal_pass": run["regimes"][regime]["conditions"]["phase_twirl"]["causal_pass"],
                        "exact_causal_pass": run["regimes"][regime]["conditions"]["exact_phase_0"]["causal_pass"],
                    }
                    for regime in REGIMES
                },
            }
            for run in runs
        },
    }


def run_campaign(config: C3PhaseHarmonicConfig, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    runs = []
    for seed in config.seeds:
        path = output / "runs" / f"seed_{seed}" / "result.json"
        if path.is_file():
            existing = json.loads(path.read_text())
            if existing.get("implementation_sha256") == _implementation_digest():
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
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "implementation_sha256": _implementation_digest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        "summary": {"requested": len(config.seeds), "completed": len(runs), "failed": 0},
        "aggregates": aggregate(runs, config),
        "method_boundaries": [
            "The phase rotation is an off-orbit intervention in the observed real C3 carrier.",
            "The same predecessor cohort and frozen synthesis cuts are reused for mechanistic decomposition, not independent replication.",
            "A phase frequency does not uniquely identify Taylor order.",
            "Causal sufficiency is conditioned on the frozen downstream decoder and task gate.",
        ],
    }
    _write_json(output / "campaign_results.json", campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=C3PhaseHarmonicConfig.source_root)
    parser.add_argument("--irrep-root", default=C3PhaseHarmonicConfig.irrep_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/tinyllm_c3_phase_harmonic/20260806_d6_preregistered"),
    )
    parser.add_argument("--seeds", default="7,17,29,41,53")
    parser.add_argument("--primary-phase-seeds", default="17,29,53")
    parser.add_argument("--orbit-count", type=int, default=64)
    parser.add_argument("--first-blocks", type=int, default=6)
    parser.add_argument("--phase-steps", type=int, default=24)
    parser.add_argument("--task-effect-floor", type=float, default=1e-6)
    parser.add_argument("--effect-threshold", type=float, default=0.70)
    parser.add_argument("--periodicity-tolerance", type=float, default=1e-5)
    parser.add_argument("--forbidden-energy-tolerance", type=float, default=1e-8)
    parser.add_argument("--reconstruction-tolerance", type=float, default=1e-5)
    parser.add_argument("--activation-batch-size", type=int, default=256)
    parser.add_argument("--continuation-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args(argv)
    config = C3PhaseHarmonicConfig(
        source_root=args.source_root,
        irrep_root=args.irrep_root,
        seeds=_ints(args.seeds),
        primary_phase_seeds=_ints(args.primary_phase_seeds),
        orbit_count=args.orbit_count,
        first_blocks=args.first_blocks,
        phase_steps=args.phase_steps,
        task_effect_floor=args.task_effect_floor,
        effect_threshold=args.effect_threshold,
        periodicity_tolerance=args.periodicity_tolerance,
        forbidden_energy_tolerance=args.forbidden_energy_tolerance,
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
