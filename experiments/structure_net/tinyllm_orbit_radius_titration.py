#!/usr/bin/env python3
"""Titrate the exact group-orbit amplitude at frozen TinyLLM synthesis fronts."""

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


SCHEMA_VERSION = "nal.tinyllm-orbit-radius-titration.v1"
HYPOTHESIS_ID = "tinyllm-causal-orbit-radius-threshold-v1"
REGIMES = ("composition", "extrapolation")
PATHS = ("exact_orbit", "linear_chord", "quadratic_chord")


@dataclass(frozen=True)
class OrbitRadiusConfig:
    source_root: str = "data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered"
    character_root: str = "data/experiments/tinyllm_reynolds_character_coupling/20260806_d6_preregistered"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    degrees: tuple[int, ...] = (2, 3)
    orbit_count: int = 64
    first_blocks: int = 6
    radii: tuple[float, ...] = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
    threshold_tolerance: float = 0.125
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
        if not self.radii or self.radii[0] != 0.0 or self.radii[-1] != 1.0:
            raise ValueError("radii must start at 0 and end at 1")
        if any(left >= right for left, right in zip(self.radii, self.radii[1:])):
            raise ValueError("radii must be strictly increasing")
        if any(radius < 0.0 or radius > 1.0 for radius in self.radii):
            raise ValueError("radii must lie in [0,1]")
        if self.threshold_tolerance < 0.0:
            raise ValueError("threshold_tolerance cannot be negative")
        if self.task_effect_floor <= 0.0:
            raise ValueError("task_effect_floor must be positive")


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


def _scientific_fingerprint(
    config: OrbitRadiusConfig,
    task: CircleTaskConfig,
    k: int,
    seed: int,
    checkpoint_sha256: str,
    comparator_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "task": asdict(task),
        "k": k,
        "seed": seed,
        "checkpoint_sha256": checkpoint_sha256,
        "comparator_sha256": comparator_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _bridge_config(config: OrbitRadiusConfig) -> deck.DeckDescramblerConfig:
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


def crossing_summary(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize the causal pass/fail curve without smoothing or interpolation."""
    if not points:
        raise ValueError("at least one radius point is required")
    radii = [float(point["radius"]) for point in points]
    passed = [bool(point["causal_pass"]) for point in points]
    critical_index = next((index for index, value in enumerate(passed) if value), None)
    critical_radius = None if critical_index is None else radii[critical_index]
    endpoint_replication = not passed[0] and passed[-1]
    single_crossing = bool(
        endpoint_replication
        and critical_index is not None
        and passed == [index >= critical_index for index in range(len(passed))]
    )
    return {
        "endpoint_replication": endpoint_replication,
        "single_crossing": single_crossing,
        "critical_radius": critical_radius,
        "pass_sequence": passed,
    }


def radial_residual_metrics(
    defect: torch.Tensor,
    full_defect: torch.Tensor,
    radius: float,
) -> dict[str, Optional[float]]:
    flat = defect.double().reshape(-1)
    full = full_defect.double().reshape(-1)
    full_squared = float(torch.square(full).sum().cpu())
    full_norm = math.sqrt(full_squared)
    norm = float(torch.linalg.vector_norm(flat).cpu())
    cosine: Optional[float]
    if norm <= 1e-12 or full_norm <= 1e-12:
        cosine = None
    else:
        cosine = float(torch.dot(flat, full).cpu()) / (norm * full_norm)
    denominator = max(1e-12, full_squared)
    return {
        "norm_ratio_to_full": norm / max(1e-12, full_norm),
        "cosine_to_full": cosine,
        "normalized_squared_error_to_linear_chord": float(
            torch.square(flat - radius * full).sum().cpu()
        )
        / denominator,
        "normalized_squared_error_to_quadratic_chord": float(
            torch.square(flat - radius**2 * full).sum().cpu()
        )
        / denominator,
    }


def _load_character_comparator(
    config: OrbitRadiusConfig,
    k: int,
    seed: int,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], str, Path]:
    path = Path(config.character_root) / "runs" / f"k{k}" / f"seed_{seed}" / "result.json"
    result = json.loads(path.read_text())
    if result.get("status") != "completed":
        raise ValueError(f"character comparator is incomplete: {path}")
    if result.get("hypothesis_id") != coupling.HYPOTHESIS_ID:
        raise ValueError(f"wrong character comparator hypothesis: {path}")
    if result["provenance"]["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(f"checkpoint mismatch in character comparator: {path}")
    return result, _sha256(path), path


@torch.no_grad()
def _path_curve(
    system: Any,
    dataset: deck.OrbitDataset,
    baseline: Mapping[str, Any],
    task: CircleTaskConfig,
    config: OrbitRadiusConfig,
    target_cut: str,
    barycenter: torch.Tensor,
    propagated: torch.Tensor,
    full_defect: torch.Tensor,
    centered_sheets: torch.Tensor,
    apply: Any,
) -> dict[str, Any]:
    n, k = dataset.orbit_count, dataset.k
    states: dict[str, list[torch.Tensor]] = {path: [] for path in PATHS}
    exact_defects: list[torch.Tensor] = []
    for radius in config.radii:
        exact_target = apply(
            (barycenter[:, None] + radius * centered_sheets).flatten(0, 1)
        ).reshape_as(centered_sheets).mean(1)
        exact_defect = exact_target - propagated
        exact_defects.append(exact_defect)
        states["exact_orbit"].append(exact_target)
        states["linear_chord"].append(propagated + radius * full_defect)
        states["quadratic_chord"].append(propagated + radius**2 * full_defect)

    posteriors: dict[str, list[torch.Tensor]] = {path: [] for path in PATHS}
    for path in PATHS:
        for state in states[path]:
            posterior = coupling._continue_tensor(
                system,
                target_cut,
                coupling._repeat_patch(state, k),
                task,
                config.continuation_batch_size,
            )
            posteriors[path].append(posterior)

    reference_zero = posteriors["exact_orbit"][0].reshape(n, k, -1)[:, 0]
    reference_full = posteriors["exact_orbit"][-1].reshape(n, k, -1)[:, 0]
    curves: dict[str, Any] = {}
    for path in PATHS:
        points = []
        for index, radius in enumerate(config.radii):
            posterior = posteriors[path][index]
            diagnostics = coupling._diagnostics(posterior, dataset)
            causal_pass, _ = coupling._causal_classification(diagnostics, baseline, k)
            orbit_posterior = posterior.reshape(n, k, -1)[:, 0]
            point: dict[str, Any] = {
                "radius": radius,
                "causal_pass": causal_pass,
                "diagnostics": diagnostics,
                "task_effect": coupling.task_effect_approximation(
                    reference_zero,
                    reference_full,
                    orbit_posterior,
                    config.task_effect_floor,
                ),
            }
            if path == "exact_orbit":
                point["residual_geometry"] = radial_residual_metrics(
                    exact_defects[index], full_defect, radius
                )
            points.append(point)
        curves[path] = {"points": points, "crossing": crossing_summary(points)}
    return curves


def analyze_cell(
    task: CircleTaskConfig,
    config: OrbitRadiusConfig,
    k: int,
    seed: int,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    bridge = _bridge_config(config)
    system, source_provenance = deck.load_source(task, bridge, k, seed, device)
    comparator, comparator_sha, comparator_path = _load_character_comparator(
        config, k, seed, source_provenance["checkpoint_sha256"]
    )
    fingerprint = _scientific_fingerprint(
        config,
        task,
        k,
        seed,
        source_provenance["checkpoint_sha256"],
        comparator_sha,
    )
    transitions = koopman.one_step_transitions(config.first_blocks)
    regimes: dict[str, Any] = {}
    for regime_index, regime in enumerate(REGIMES):
        synthesis_index = comparator["regimes"][regime]["synthesis_front_index"]
        if synthesis_index is None:
            raise ValueError(f"no synthesis front for k{k} seed {seed} {regime}")
        synthesis_index = int(synthesis_index)
        source_cut, target_cut = transitions[synthesis_index]
        dataset = deck.generate_exact_orbits(
            task,
            k=k,
            orbit_count=config.orbit_count,
            seed=seed + 4501 + 101 * regime_index,
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
        source = source.reshape((dataset.orbit_count, k) + source.shape[1:])
        barycenter = source.mean(1)
        centered = source - barycenter[:, None]
        apply = lambda value: coupling._apply_sublayer(system, synthesis_index, value)
        propagated = apply(barycenter)
        actual = apply(source.flatten(0, 1)).reshape_as(source).mean(1)
        curves = _path_curve(
            system,
            dataset,
            baseline,
            task,
            config,
            target_cut,
            barycenter,
            propagated,
            actual - propagated,
            centered,
            apply,
        )
        regimes[regime] = {
            "evaluation_seed": seed + 4501 + 101 * regime_index,
            "synthesis_transition_index": synthesis_index,
            "source_cut": source_cut,
            "target_cut": target_cut,
            "baseline": baseline,
            "curves": curves,
        }

    exact_crossings = {
        regime: regimes[regime]["curves"]["exact_orbit"]["crossing"]
        for regime in REGIMES
    }
    critical = [exact_crossings[regime]["critical_radius"] for regime in REGIMES]
    gates = {
        "endpoint_replication": all(
            exact_crossings[regime]["endpoint_replication"] for regime in REGIMES
        ),
        "single_radial_crossing": all(
            exact_crossings[regime]["single_crossing"] for regime in REGIMES
        ),
        "shift_stable_threshold": bool(
            all(value is not None for value in critical)
            and abs(float(critical[0]) - float(critical[1]))
            <= config.threshold_tolerance + 1e-12
        ),
    }
    root = output / "runs" / f"k{k}" / f"seed_{seed}"
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-orbit-radius-k{k}-seed{seed}",
        "status": "completed",
        "completed_at": _utc_now(),
        "k": k,
        "seed": seed,
        "configuration": asdict(config),
        "scientific_fingerprint": fingerprint,
        "implementation_sha256": _implementation_digest(),
        "provenance": {
            **source_provenance,
            "character_comparator_path": str(comparator_path),
            "character_comparator_sha256": comparator_sha,
        },
        "regimes": regimes,
        "gates": gates,
        "all_primary_gates_passed": k == 2 and all(gates.values()),
        "analysis_seconds": time.perf_counter() - started,
    }
    _write_json(root / "result.json", result)
    return result


def aggregate(
    runs: Sequence[Mapping[str, Any]], config: OrbitRadiusConfig
) -> dict[str, Any]:
    required = 4 if len(config.seeds) >= 5 else len(config.seeds)
    gate_names = (
        "endpoint_replication",
        "single_radial_crossing",
        "shift_stable_threshold",
    )
    degrees: dict[str, Any] = {}
    for k in config.degrees:
        selected = [run for run in runs if int(run["k"]) == k]
        counts = {
            gate: sum(bool(run["gates"][gate]) for run in selected)
            for gate in gate_names
        }
        degrees[str(k)] = {
            "role": "primary" if k == 2 else "secondary_diagnostic",
            "gate_counts": counts,
            "gate_passes": {gate: count >= required for gate, count in counts.items()},
            "per_seed": {
                str(run["seed"]): {
                    "gates": run["gates"],
                    "critical_radii": {
                        regime: run["regimes"][regime]["curves"]["exact_orbit"]["crossing"]["critical_radius"]
                        for regime in REGIMES
                    },
                    "path_critical_radii": {
                        regime: {
                            path: run["regimes"][regime]["curves"][path]["crossing"]["critical_radius"]
                            for path in PATHS
                        }
                        for regime in REGIMES
                    },
                }
                for run in selected
            },
        }
    primary = degrees.get("2")
    confirmed = bool(primary and all(primary["gate_passes"].values()))
    return {
        "required_seed_count": required,
        "degrees": degrees,
        "primary_degree": 2,
        "confirmed": confirmed,
    }


def run_campaign(config: OrbitRadiusConfig, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    runs = []
    for k in config.degrees:
        for seed in config.seeds:
            root = output / "runs" / f"k{k}" / f"seed_{seed}"
            path = root / "result.json"
            if path.is_file():
                existing = json.loads(path.read_text())
                checkpoint = Path(config.source_root) / "runs" / f"k{k}" / f"seed_{seed}" / "model.pt"
                comparator = Path(config.character_root) / "runs" / f"k{k}" / f"seed_{seed}" / "result.json"
                expected = _scientific_fingerprint(
                    config, task, k, seed, _sha256(checkpoint), _sha256(comparator)
                )
                if (
                    existing.get("status") == "completed"
                    and existing.get("schema_version") == SCHEMA_VERSION
                    and existing.get("scientific_fingerprint") == expected
                    and existing.get("implementation_sha256") == _implementation_digest()
                ):
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
            "reused_checkpoints": len(runs),
            "trained_models": 0,
        },
        "aggregates": aggregate(runs, config),
        "method_boundaries": [
            "The intervention identifies sufficiency only along observed exact deck-orbit directions.",
            "The output gate is decoder-conditioned and task-specific.",
            "Repeated patched states make within-orbit branch chance exact but do not establish global branch absence.",
            "Degree-three curves are secondary and cannot promote the primary degree-two result.",
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
    parser.add_argument("--source-root", default=OrbitRadiusConfig.source_root)
    parser.add_argument("--character-root", default=OrbitRadiusConfig.character_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/tinyllm_orbit_radius_titration/20260806_d6_preregistered"),
    )
    parser.add_argument("--seeds", default="7,17,29,41,53")
    parser.add_argument("--degrees", default="2,3")
    parser.add_argument("--orbit-count", type=int, default=64)
    parser.add_argument("--first-blocks", type=int, default=6)
    parser.add_argument("--radii", default="0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1")
    parser.add_argument("--threshold-tolerance", type=float, default=0.125)
    parser.add_argument("--task-effect-floor", type=float, default=1e-6)
    parser.add_argument("--activation-batch-size", type=int, default=256)
    parser.add_argument("--continuation-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args(argv)
    config = OrbitRadiusConfig(
        source_root=args.source_root,
        character_root=args.character_root,
        seeds=_ints(args.seeds),
        degrees=_ints(args.degrees),
        orbit_count=args.orbit_count,
        first_blocks=args.first_blocks,
        radii=_floats(args.radii),
        threshold_tolerance=args.threshold_tolerance,
        task_effect_floor=args.task_effect_floor,
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
