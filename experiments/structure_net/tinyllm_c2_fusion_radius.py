#!/usr/bin/env python3
"""Measure the causal radius of degree-two deck-character fusion."""

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


SCHEMA_VERSION = "nal.tinyllm-c2-character-fusion-radius.v1"
HYPOTHESIS_ID = "tinyllm-c2-character-fusion-radius-v1"
REGIMES = ("composition", "extrapolation")
CONDITIONS = ("exact_character", "cross_orbit_transplant", "matched_random_symmetric")
EARLY_FRONT_SEEDS = (7, 29, 53)
LATER_FRONT_SEEDS = (17, 41)


@dataclass(frozen=True)
class FusionRadiusConfig:
    source_root: str = "data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered"
    character_root: str = "data/experiments/tinyllm_reynolds_character_coupling/20260806_d6_preregistered"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    orbit_count: int = 64
    first_blocks: int = 6
    radius_grid: tuple[float, ...] = (0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25)
    primary_max_radius: float = 1.0
    early_radius_ceiling: float = 0.5
    later_radius_floor: float = 0.75
    shift_radius_tolerance: float = 0.25
    control_fisher_threshold: float = 0.70
    task_effect_floor: float = 1e-6
    exchange_tolerance: float = 1e-6
    activation_batch_size: int = 256
    continuation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if len(self.seeds) < 5 and not self.allow_underpowered:
            raise ValueError("confirmatory campaign requires five seeds")
        if self.orbit_count < 8:
            raise ValueError("at least eight exact orbits are required")
        if self.first_blocks < 1:
            raise ValueError("at least one transformer block is required")
        radii = tuple(float(value) for value in self.radius_grid)
        if radii != tuple(sorted(set(radii))) or radii[0] != 0.0:
            raise ValueError("radius grid must be unique, sorted, and start at zero")
        if 1.0 not in radii or self.primary_max_radius not in radii:
            raise ValueError("radius grid must contain the observed and primary maximum radii")
        if self.primary_max_radius > 1.0:
            raise ValueError("off-manifold radii cannot define primary onset")
        if not 0.0 < self.control_fisher_threshold <= 1.0:
            raise ValueError("control Fisher threshold must be in (0,1]")


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


def _fingerprint(config: FusionRadiusConfig, seed: int, provenance: Mapping[str, Any]) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "seed": seed,
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "character_result_sha256": provenance["character_result_sha256"],
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode()).hexdigest()


def _bridge_config(config: FusionRadiusConfig) -> deck.DeckDescramblerConfig:
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
    config: FusionRadiusConfig, seed: int, checkpoint_sha256: str
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
    ):
        raise ValueError(f"invalid character-coupling source {path}")
    return value, path


def _radius_key(radius: float) -> str:
    return f"{radius:.3f}".rstrip("0").rstrip(".")


def symmetric_response(
    barycenter: torch.Tensor,
    deviation: torch.Tensor,
    radius: float,
    apply: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the propagated barycenter and exact even response at one radius."""
    propagated = apply(barycenter)
    plus = apply(barycenter + radius * deviation)
    minus = apply(barycenter - radius * deviation)
    response = 0.5 * (plus + minus) - propagated
    return propagated, response


def _matched_directions(
    exact: torch.Tensor, seed: int
) -> dict[str, torch.Tensor]:
    axes = tuple(range(1, exact.ndim))
    exact_norm = torch.square(exact).sum(dim=axes, keepdim=True).sqrt()

    shift = 1 + seed % max(1, len(exact) - 1)
    transplanted = torch.roll(exact, shifts=shift, dims=0)
    transplant_norm = torch.square(transplanted).sum(dim=axes, keepdim=True).sqrt().clamp_min(1e-12)
    transplanted = transplanted * exact_norm / transplant_norm

    generator = torch.Generator(device=exact.device)
    generator.manual_seed(seed)
    random = torch.randn(exact.shape, generator=generator, device=exact.device, dtype=exact.dtype)
    random_norm = torch.square(random).sum(dim=axes, keepdim=True).sqrt().clamp_min(1e-12)
    random = random * exact_norm / random_norm
    return {
        "exact_character": exact,
        "cross_orbit_transplant": transplanted,
        "matched_random_symmetric": random,
    }


def _residual_metrics(response: torch.Tensor, reference: torch.Tensor) -> dict[str, Optional[float]]:
    response_flat = response.double().reshape(-1)
    reference_flat = reference.double().reshape(-1)
    norm = torch.linalg.vector_norm(response_flat)
    reference_norm = torch.linalg.vector_norm(reference_flat)
    cosine = None
    if float(norm.cpu()) > 1e-12 and float(reference_norm.cpu()) > 1e-12:
        cosine = float(torch.dot(response_flat, reference_flat).cpu() / (norm * reference_norm).cpu())
    return {"norm": float(norm.cpu()), "cosine_to_exact_radius_one": cosine}


def _primary_radii(config: FusionRadiusConfig) -> list[float]:
    return [
        radius
        for radius in config.radius_grid
        if 0.0 < radius <= config.primary_max_radius
    ]


def _onset(records: Mapping[str, Mapping[str, Any]], config: FusionRadiusConfig) -> Optional[float]:
    return next(
        (
            radius
            for radius in _primary_radii(config)
            if bool(records[_radius_key(radius)]["causal_pass"])
        ),
        None,
    )


def _monotone_after_onset(
    records: Mapping[str, Mapping[str, Any]], onset: Optional[float], config: FusionRadiusConfig
) -> bool:
    if onset is None:
        return False
    return all(
        bool(records[_radius_key(radius)]["causal_pass"])
        for radius in _primary_radii(config)
        if radius >= onset
    )


def _local_slopes(records: dict[str, dict[str, Any]], config: FusionRadiusConfig) -> None:
    previous_radius: Optional[float] = None
    previous_norm: Optional[float] = None
    for radius in config.radius_grid:
        key = _radius_key(radius)
        current_norm = float(records[key]["response"]["norm"])
        slope = None
        if (
            previous_radius is not None
            and previous_norm is not None
            and previous_radius > 0.0
            and previous_norm > 1e-12
            and current_norm > 1e-12
        ):
            slope = math.log(current_norm / previous_norm) / math.log(radius / previous_radius)
        records[key]["local_log_slope"] = slope
        if radius > 0.0:
            previous_radius, previous_norm = radius, current_norm


@torch.no_grad()
def analyze_regime(
    system: Any,
    task: CircleTaskConfig,
    config: FusionRadiusConfig,
    seed: int,
    regime: str,
    source_record: Mapping[str, Any],
    bridge: deck.DeckDescramblerConfig,
    device: torch.device,
) -> dict[str, Any]:
    regime_index = REGIMES.index(regime)
    source_regime = source_record["regimes"][regime]
    transition_index = source_regime["synthesis_front_index"]
    if transition_index is None:
        raise ValueError(f"source synthesis front missing for seed {seed} {regime}")
    transition_index = int(transition_index)
    transitions = koopman.one_step_transitions(config.first_blocks)
    source_cut, target_cut = transitions[transition_index]
    if target_cut != source_regime["synthesis_target_cut"]:
        raise ValueError("source synthesis cut does not match transition table")

    dataset = deck.generate_exact_orbits(
        task,
        k=2,
        orbit_count=config.orbit_count,
        seed=seed + 9101 + 307 * regime_index,
        regime=regime,
    )
    captured = deck.capture_sequences(system, dataset, bridge, device)
    source = torch.from_numpy(captured[source_cut]).to(device)
    source = source.reshape((config.orbit_count, 2) + source.shape[1:])
    barycenter = source.mean(1)
    deviation = source[:, 0] - barycenter
    symmetry_error = float(torch.max(torch.abs(source[:, 1] - (barycenter - deviation))).cpu())
    apply = lambda value: coupling._apply_sublayer(system, transition_index, value)

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

    directions = _matched_directions(deviation, seed + 1009 * regime_index)
    condition_states: dict[str, dict[str, Any]] = {}
    condition_posteriors: dict[str, dict[str, torch.Tensor]] = {}
    condition_responses: dict[str, dict[str, torch.Tensor]] = {}
    exchange_error = 0.0
    for condition, direction in directions.items():
        records: dict[str, Any] = {}
        posteriors: dict[str, torch.Tensor] = {}
        responses: dict[str, torch.Tensor] = {}
        for radius in config.radius_grid:
            key = _radius_key(radius)
            propagated, response = symmetric_response(barycenter, direction, radius, apply)
            state = propagated + response
            posterior = coupling._continue_tensor(
                system,
                target_cut,
                coupling._repeat_patch(state, 2),
                task,
                config.continuation_batch_size,
            )
            diagnostics = coupling._diagnostics(posterior, dataset)
            causal_pass, causal_label = coupling._causal_classification(diagnostics, baseline, 2)
            records[key] = {
                "radius": radius,
                "causal_pass": causal_pass,
                "causal_label": causal_label,
                "diagnostics": diagnostics,
            }
            posteriors[key] = posterior
            responses[key] = response
            if condition == "exact_character":
                _, exchanged_response = symmetric_response(barycenter, -direction, radius, apply)
                exchanged = coupling._continue_tensor(
                    system,
                    target_cut,
                    coupling._repeat_patch(propagated + exchanged_response, 2),
                    task,
                    config.continuation_batch_size,
                )
                exchange_error = max(
                    exchange_error,
                    float(torch.max(torch.abs(posterior - exchanged)).cpu()),
                )
        condition_states[condition] = records
        condition_posteriors[condition] = posteriors
        condition_responses[condition] = responses

    exact_reference = condition_posteriors["exact_character"][_radius_key(1.0)]
    exact_reference_orbits = exact_reference.reshape(config.orbit_count, 2, -1)[:, 0]
    propagated_reference = condition_posteriors["exact_character"][_radius_key(0.0)]
    propagated_reference_orbits = propagated_reference.reshape(config.orbit_count, 2, -1)[:, 0]
    exact_response = condition_responses["exact_character"][_radius_key(1.0)]
    for condition in CONDITIONS:
        for radius in config.radius_grid:
            key = _radius_key(radius)
            posterior = condition_posteriors[condition][key]
            posterior_orbits = posterior.reshape(config.orbit_count, 2, -1)[:, 0]
            effect = coupling.task_effect_approximation(
                propagated_reference_orbits,
                exact_reference_orbits,
                posterior_orbits,
                config.task_effect_floor,
            )
            condition_states[condition][key]["fisher_effect_vs_exact"] = effect
            condition_states[condition][key]["response"] = _residual_metrics(
                condition_responses[condition][key], exact_response
            )
        _local_slopes(condition_states[condition], config)

    exact_records = condition_states["exact_character"]
    onset = _onset(exact_records, config)
    monotone = _monotone_after_onset(exact_records, onset, config)
    control_reproduction = {}
    for condition in CONDITIONS[1:]:
        control_reproduction[condition] = bool(
            onset is not None
            and any(
                not condition_states[condition][_radius_key(radius)]["fisher_effect_vs_exact"]["degenerate"]
                and condition_states[condition][_radius_key(radius)]["fisher_effect_vs_exact"]["explained_fraction"]
                >= config.control_fisher_threshold
                for radius in _primary_radii(config)
                if radius <= onset
            )
        )

    return {
        "regime": regime,
        "fresh_orbit_seed": seed + 9101 + 307 * regime_index,
        "source_cut": source_cut,
        "target_cut": target_cut,
        "transition_index": transition_index,
        "baseline": baseline,
        "source_c2_symmetry_error": symmetry_error,
        "onset_radius": onset,
        "monotone_after_onset": monotone,
        "control_reproduction": control_reproduction,
        "exchange_max_posterior_error": exchange_error,
        "conditions": condition_states,
    }


def _seed_gates(
    seed: int, regimes: Mapping[str, Mapping[str, Any]], config: FusionRadiusConfig
) -> dict[str, Any]:
    onsets = {regime: regimes[regime]["onset_radius"] for regime in REGIMES}
    nonmissing = all(value is not None for value in onsets.values())
    early = seed in EARLY_FRONT_SEEDS and nonmissing and all(
        float(value) <= config.early_radius_ceiling for value in onsets.values()
    )
    later = seed in LATER_FRONT_SEEDS and nonmissing and all(
        float(value) >= config.later_radius_floor for value in onsets.values()
    )
    shift_stable = bool(
        nonmissing
        and abs(float(onsets["composition"]) - float(onsets["extrapolation"]))
        <= config.shift_radius_tolerance
    )
    monotone = all(bool(regimes[regime]["monotone_after_onset"]) for regime in REGIMES)
    control_specificity = all(
        not reproduced
        for regime in REGIMES
        for reproduced in regimes[regime]["control_reproduction"].values()
    )
    exchange_invariance = all(
        float(regimes[regime]["exchange_max_posterior_error"]) <= config.exchange_tolerance
        for regime in REGIMES
    )
    return {
        "cohort_prediction": early if seed in EARLY_FRONT_SEEDS else later,
        "early_front_locality": early if seed in EARLY_FRONT_SEEDS else None,
        "later_front_finite_radius": later if seed in LATER_FRONT_SEEDS else None,
        "shift_stable_onset": shift_stable,
        "monotone_causal_response": monotone,
        "control_specificity": control_specificity,
        "exchange_invariance": exchange_invariance,
    }


def analyze_cell(
    task: CircleTaskConfig,
    config: FusionRadiusConfig,
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
        "character_implementation_sha256": source_record["implementation_sha256"],
    }
    fingerprint = _fingerprint(config, seed, provenance)
    regimes = {
        regime: analyze_regime(
            system, task, config, seed, regime, source_record, bridge, device
        )
        for regime in REGIMES
    }
    gates = _seed_gates(seed, regimes, config)
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-c2-fusion-radius-seed{seed}",
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else "preregistered_primary_evidence"
        ),
        "completed_at": _utc_now(),
        "seed": seed,
        "configuration": asdict(config),
        "scientific_fingerprint": fingerprint,
        "provenance": provenance,
        "regimes": regimes,
        "gates": gates,
        "implementation_sha256": _implementation_digest(),
        "analysis_seconds": time.perf_counter() - started,
    }
    _write_json(output / "runs" / f"seed_{seed}" / "result.json", result)
    return result


def aggregate(runs: Sequence[Mapping[str, Any]], config: FusionRadiusConfig) -> dict[str, Any]:
    by_seed = {int(run["seed"]): run for run in runs}
    early_pass = all(
        seed in by_seed and by_seed[seed]["gates"]["early_front_locality"] is True
        for seed in EARLY_FRONT_SEEDS
    )
    later_pass = all(
        seed in by_seed and by_seed[seed]["gates"]["later_front_finite_radius"] is True
        for seed in LATER_FRONT_SEEDS
    )
    required = 4 if len(config.seeds) >= 5 else len(config.seeds)
    counts = {
        gate: sum(bool(run["gates"][gate]) for run in runs)
        for gate in (
            "shift_stable_onset",
            "monotone_causal_response",
            "control_specificity",
            "exchange_invariance",
        )
    }
    gate_passes = {
        "early_front_locality": early_pass,
        "later_front_finite_radius": later_pass,
        "shift_stable_onset": counts["shift_stable_onset"] >= required,
        "monotone_causal_response": counts["monotone_causal_response"] >= required,
        "control_specificity": counts["control_specificity"] >= required,
        "exchange_invariance": counts["exchange_invariance"] == len(runs),
    }
    return {
        "required_seed_count": required,
        "gate_counts": counts,
        "gate_passes": gate_passes,
        "confirmed": all(gate_passes.values()),
        "per_seed": {
            str(seed): {
                "onsets": {
                    regime: by_seed[seed]["regimes"][regime]["onset_radius"]
                    for regime in REGIMES
                },
                "target_cuts": {
                    regime: by_seed[seed]["regimes"][regime]["target_cut"]
                    for regime in REGIMES
                },
                "gates": by_seed[seed]["gates"],
            }
            for seed in sorted(by_seed)
        },
    }


def run_campaign(config: FusionRadiusConfig, output: Path) -> dict[str, Any]:
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
            ):
                source = Path(existing["provenance"]["checkpoint"])
                character = Path(existing["provenance"]["character_result"])
                if (
                    source.is_file()
                    and character.is_file()
                    and _sha256(source) == existing["provenance"]["checkpoint_sha256"]
                    and _sha256(character) == existing["provenance"]["character_result_sha256"]
                    and existing.get("scientific_fingerprint")
                    == _fingerprint(config, seed, existing["provenance"])
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
        },
        "aggregates": aggregate(runs, config),
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
        "method_boundaries": [
            "The intervention follows observed residual-space character rays and does not identify a global group representation.",
            "Repeated patches make within-orbit identity constant but do not prove global absence of branch information.",
            "Fisher--Rao effects are conditioned on the frozen task decoder.",
            "Radius 1.25 is an artificial residual intervention and cannot define primary onset.",
            "Early and later cohorts were selected from a predecessor result and are evaluated here on fresh orbits, not new model seeds.",
        ],
    }
    _write_json(output / "campaign_results.json", campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=FusionRadiusConfig.source_root)
    parser.add_argument("--character-root", default=FusionRadiusConfig.character_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/tinyllm_c2_fusion_radius/20260806_d6_preregistered"),
    )
    parser.add_argument("--seeds", default="7,17,29,41,53")
    parser.add_argument("--orbit-count", type=int, default=64)
    parser.add_argument("--first-blocks", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--activation-batch-size", type=int, default=256)
    parser.add_argument("--continuation-batch-size", type=int, default=256)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args(argv)
    config = FusionRadiusConfig(
        source_root=args.source_root,
        character_root=args.character_root,
        seeds=_ints(args.seeds),
        orbit_count=args.orbit_count,
        first_blocks=args.first_blocks,
        device=args.device,
        activation_batch_size=args.activation_batch_size,
        continuation_batch_size=args.continuation_batch_size,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    print(args.output / "campaign_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
