#!/usr/bin/env python3
"""Correct the specificity control for the TinyLLM local metric-field audit."""

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
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "nal.tinyllm-c2-local-metric-field-specificity.v1"
HYPOTHESIS_ID = "tinyllm-c2-local-metric-field-specificity-v1"
SOURCE_SCHEMA = "nal.tinyllm-c2-local-metric-field-transport.v1"
SOURCE_HYPOTHESIS_ID = "tinyllm-c2-local-metric-field-transport-v1"
SOURCE_CAMPAIGN_SHA256 = (
    "2aa80cff5882cb79754d732e9012c810c2624934bc5be9e03c9e77de4f525f55"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "e3a05118971bba51e50e07e0fc426a8b321f0ecf823ded7dfec6b3539b630bfd"
)
SEEDS = (7, 29, 53)
REGIMES = ("composition", "extrapolation")
GROUP_ARMS = ("amplitude", "orientation", "offset", "composed")


@dataclass(frozen=True)
class LocalMetricFieldSpecificityConfig:
    source_campaign: str = (
        "data/experiments/tinyllm_local_metric_field_transport/"
        "20260807_d6_fresh_cohort"
    )
    seeds: tuple[int, ...] = SEEDS
    phase_count: int = 16
    nuisance_replicates: int = 4
    phase_shift_bins: int = 5
    median_kernel_cosine_floor: float = 0.95
    p10_kernel_cosine_floor: float = 0.90
    p95_projector_distance_ceiling: float = math.sqrt(1.0 - 0.90**2)
    nuisance_equivalence_cosine_floor: float = 0.95
    nuisance_equivalence_difference_ceiling: float = 0.02
    phase_specificity_margin: float = 0.10
    random_specificity_margin: float = 0.20
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != SEEDS and not self.allow_underpowered:
            raise ValueError("primary metric-field specificity seeds are fixed")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.phase_count != 16 or self.nuisance_replicates != 4:
            raise ValueError("the stored cohort fixes 16 phases and four replicates")
        if self.phase_shift_bins != 5:
            raise ValueError("the corrective semantic-phase shift is fixed to five bins")
        if min(
            self.median_kernel_cosine_floor,
            self.p10_kernel_cosine_floor,
            self.p95_projector_distance_ceiling,
            self.nuisance_equivalence_cosine_floor,
            self.nuisance_equivalence_difference_ceiling,
            self.phase_specificity_margin,
            self.random_specificity_margin,
        ) <= 0.0:
            raise ValueError("all thresholds must be positive")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _implementation_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _evidence_role(config: LocalMetricFieldSpecificityConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_post_outcome_corrective_frozen_artifact_diagnostic"
    )


def nuisance_equivalence_indices(phase_count: int, replicates: int) -> np.ndarray:
    values = np.arange(phase_count * replicates).reshape(phase_count, replicates)
    return np.roll(values, -1, axis=1).reshape(-1)


def semantic_phase_shift_indices(
    phase_count: int, replicates: int, shift_bins: int
) -> np.ndarray:
    values = np.arange(phase_count * replicates).reshape(phase_count, replicates)
    return np.roll(values, -shift_bins, axis=0).reshape(-1)


def _kernel_lines(projectors: np.ndarray) -> np.ndarray:
    projectors = 0.5 * (projectors + np.swapaxes(projectors, -1, -2))
    _, vectors = np.linalg.eigh(projectors)
    return vectors[..., 0]


def line_cosines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(np.sum(_kernel_lines(left) * _kernel_lines(right), axis=1))


def projector_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.linalg.norm(left - right, axis=(1, 2)) / math.sqrt(2.0)


def random_rank_two_projectors(count: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    kernel = generator.normal(size=(count, 3))
    kernel /= np.linalg.norm(kernel, axis=1, keepdims=True)
    identity = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    return identity - kernel[:, :, None] * kernel[:, None, :]


def _control_seed(seed: int, regime: str, arm: str) -> int:
    payload = f"{HYPOTHESIS_ID}:{seed}:{regime}:{arm}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def corrected_cell_geometry(
    reference: np.ndarray,
    transformed: np.ndarray,
    nuisance_indices: np.ndarray,
    phase_indices: np.ndarray,
    random_projectors: np.ndarray,
) -> dict[str, float]:
    paired = line_cosines(reference, transformed)
    nuisance = line_cosines(reference[nuisance_indices], transformed)
    phase = line_cosines(reference[phase_indices], transformed)
    random = line_cosines(random_projectors, transformed)
    distance = projector_distances(reference, transformed)
    return {
        "paired_median_kernel_cosine": float(np.median(paired)),
        "paired_p10_kernel_cosine": float(np.quantile(paired, 0.10)),
        "paired_p95_projector_distance": float(np.quantile(distance, 0.95)),
        "nuisance_equivalence_median_kernel_cosine": float(np.median(nuisance)),
        "paired_nuisance_median_absolute_difference": float(
            abs(np.median(paired) - np.median(nuisance))
        ),
        "semantic_phase_control_median_kernel_cosine": float(np.median(phase)),
        "random_control_median_kernel_cosine": float(np.median(random)),
        "paired_over_semantic_phase_margin": float(np.median(paired) - np.median(phase)),
        "paired_over_random_margin": float(np.median(paired) - np.median(random)),
    }


def classify_checkpoint(
    *, valid: bool, base_geometry: bool, nuisance_equivalence: bool, specificity: bool
) -> str:
    if not valid:
        return "invalid"
    if base_geometry and nuisance_equivalence and specificity:
        return "nuisance_invariant_phase_specific_field"
    if base_geometry and nuisance_equivalence:
        return "phase_nonspecific_field"
    if not base_geometry:
        return "nuisance_variant_field"
    return "mixed_metric_field_geometry"


def _load_source(
    config: LocalMetricFieldSpecificityConfig,
) -> tuple[dict[str, Any], Path, dict[int, tuple[dict[str, Any], Path, Path]]]:
    campaign_path = Path(config.source_campaign) / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    expected_classes = {str(seed): "nuisance_specific_metric_field" for seed in SEEDS}
    required_gates = {
        "input_and_pair_contract": 3,
        "numerical_contract": 3,
        "full_control_pass": 3,
        "local_tangent_fresh_cohort_pass": 3,
        "geometric_transport_pass": 0,
        "causal_transport_pass": 0,
        "primary_checkpoint_pass": 0,
    }
    if (
        _sha256(campaign_path) != SOURCE_CAMPAIGN_SHA256
        or campaign.get("schema_version") != SOURCE_SCHEMA
        or campaign.get("hypothesis_id") != SOURCE_HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256") != SOURCE_IMPLEMENTATION_SHA256
        or campaign.get("aggregates", {}).get("classification_by_seed")
        != expected_classes
        or campaign.get("aggregates", {}).get("gate_counts") != required_gates
        or int(campaign.get("summary", {}).get("trained_models", -1)) != 0
        or int(campaign.get("summary", {}).get("fitted_writers", -1)) != 0
    ):
        raise ValueError(f"invalid metric-field source campaign {campaign_path}")
    details: dict[int, tuple[dict[str, Any], Path, Path]] = {}
    for entry in campaign.get("results", []):
        result_path = Path(entry["path"])
        arrays_path = Path(entry["arrays"])
        detail = json.loads(result_path.read_text(encoding="utf-8"))
        seed = int(detail.get("seed", -1))
        if (
            _sha256(result_path) != entry.get("result_sha256")
            or _sha256(arrays_path) != entry.get("arrays_sha256")
            or detail.get("classification") != "nuisance_specific_metric_field"
            or detail.get("scientific_fingerprint") != entry.get("scientific_fingerprint")
            or not all(
                detail.get("gates", {}).get(name) is True
                for name in (
                    "input_and_pair_contract",
                    "numerical_contract",
                    "full_control_pass",
                    "local_tangent_fresh_cohort_pass",
                )
            )
        ):
            raise ValueError(f"invalid metric-field source result {result_path}")
        details[seed] = (detail, result_path, arrays_path)
    if set(details) != set(SEEDS) or not set(config.seeds).issubset(details):
        raise ValueError("metric-field source seed index is incomplete")
    return campaign, campaign_path, details


def _fingerprint(
    config: LocalMetricFieldSpecificityConfig,
    seed: int,
    result_sha256: str,
    arrays_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "seed": seed,
        "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
        "source_result_sha256": result_sha256,
        "source_arrays_sha256": arrays_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _campaign_is_reusable(
    value: Mapping[str, Any],
    config: LocalMetricFieldSpecificityConfig,
    implementation: str,
) -> bool:
    if not (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("configuration") == _strict(asdict(config))
        and value.get("implementation_sha256") == implementation
        and int(value.get("summary", {}).get("completed", -1)) == len(config.seeds)
    ):
        return False
    return all(
        Path(entry.get("path", "")).is_file()
        and _sha256(Path(entry["path"])) == entry.get("result_sha256")
        for entry in value.get("results", [])
    ) and len(value.get("results", [])) == len(config.seeds)


def run_campaign(
    config: LocalMetricFieldSpecificityConfig, output: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    implementation = _implementation_digest()
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if _campaign_is_reusable(existing, config, implementation):
            print("aggregate already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed aggregate {campaign_path}")

    source, source_path, details = _load_source(config)
    nuisance_indices = nuisance_equivalence_indices(
        config.phase_count, config.nuisance_replicates
    )
    phase_indices = semantic_phase_shift_indices(
        config.phase_count, config.nuisance_replicates, config.phase_shift_bins
    )
    results = []
    for seed in config.seeds:
        seed_started = time.perf_counter()
        source_detail, source_result_path, arrays_path = details[seed]
        fingerprint = _fingerprint(
            config, seed, _sha256(source_result_path), _sha256(arrays_path)
        )
        cells = []
        with np.load(arrays_path) as arrays:
            for regime in REGIMES:
                reference = arrays[f"{regime}__reference__projector"]
                for arm in GROUP_ARMS:
                    transformed = arrays[f"{regime}__{arm}__projector"]
                    if reference.shape != (64, 3, 3) or transformed.shape != (64, 3, 3):
                        raise ValueError(f"invalid projector shape in {arrays_path}")
                    geometry = corrected_cell_geometry(
                        reference,
                        transformed,
                        nuisance_indices,
                        phase_indices,
                        random_rank_two_projectors(
                            64, _control_seed(seed, regime, arm)
                        ),
                    )
                    base = bool(
                        geometry["paired_median_kernel_cosine"]
                        >= config.median_kernel_cosine_floor
                        and geometry["paired_p10_kernel_cosine"]
                        >= config.p10_kernel_cosine_floor
                        and geometry["paired_p95_projector_distance"]
                        <= config.p95_projector_distance_ceiling
                    )
                    nuisance = bool(
                        geometry["nuisance_equivalence_median_kernel_cosine"]
                        >= config.nuisance_equivalence_cosine_floor
                        and geometry["paired_nuisance_median_absolute_difference"]
                        <= config.nuisance_equivalence_difference_ceiling
                    )
                    phase_specific = bool(
                        geometry["paired_over_semantic_phase_margin"]
                        >= config.phase_specificity_margin
                    )
                    random_specific = bool(
                        geometry["paired_over_random_margin"]
                        >= config.random_specificity_margin
                    )
                    cells.append(
                        {
                            "regime": regime,
                            "arm": arm,
                            "geometry": geometry,
                            "gates": {
                                "paired_geometry": base,
                                "nuisance_equivalence": nuisance,
                                "semantic_phase_specificity": phase_specific,
                                "random_plane_specificity": random_specific,
                                "corrected_cell_gate": bool(
                                    base and nuisance and phase_specific and random_specific
                                ),
                            },
                        }
                    )
        base = all(cell["gates"]["paired_geometry"] for cell in cells)
        nuisance = all(cell["gates"]["nuisance_equivalence"] for cell in cells)
        phase_specific = all(
            cell["gates"]["semantic_phase_specificity"] for cell in cells
        )
        random_specific = all(
            cell["gates"]["random_plane_specificity"] for cell in cells
        )
        specificity = bool(phase_specific and random_specific)
        valid = bool(
            source_detail["gates"]["input_and_pair_contract"]
            and source_detail["gates"]["numerical_contract"]
            and source_detail["gates"]["full_control_pass"]
            and source_detail["gates"]["local_tangent_fresh_cohort_pass"]
        )
        classification = classify_checkpoint(
            valid=valid,
            base_geometry=base,
            nuisance_equivalence=nuisance,
            specificity=specificity,
        )
        result_path = output / "runs" / f"seed_{seed}" / "result.json"
        result = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_id": f"tinyllm-c2-local-metric-field-specificity-seed{seed}",
            "status": "completed",
            "evidence_role": _evidence_role(config),
            "completed_at": _utc_now(),
            "seed": seed,
            "configuration": asdict(config),
            "scientific_fingerprint": fingerprint,
            "implementation_sha256": implementation,
            "provenance": {
                "source_campaign": str(source_path),
                "source_campaign_sha256": _sha256(source_path),
                "source_result": str(source_result_path),
                "source_result_sha256": _sha256(source_result_path),
                "source_arrays": str(arrays_path),
                "source_arrays_sha256": _sha256(arrays_path),
                "checkpoint_sha256": source_detail["provenance"]["checkpoint_sha256"],
            },
            "cells": cells,
            "gates": {
                "source_valid": valid,
                "paired_geometry_all_cells": base,
                "nuisance_equivalence_all_cells": nuisance,
                "semantic_phase_specificity_all_cells": phase_specific,
                "random_plane_specificity_all_cells": random_specific,
                "corrected_checkpoint_gate": bool(
                    classification == "nuisance_invariant_phase_specific_field"
                ),
            },
            "classification": classification,
            "primary_metric": float(
                classification == "nuisance_invariant_phase_specific_field"
            ),
            "analysis_seconds": time.perf_counter() - seed_started,
            "artifacts": {"result": str(result_path)},
        }
        _write_json(result_path, result)
        results.append(result)
        print(f"seed {seed}: {classification}", flush=True)
        if _implementation_digest() != implementation:
            raise RuntimeError("corrective implementation changed during run")

    classifications = [result["classification"] for result in results]
    support_count = classifications.count("nuisance_invariant_phase_specific_field")
    if any(name == "invalid" for name in classifications):
        conclusion = "invalid_campaign"
    elif len(config.seeds) == 3 and support_count == 3:
        conclusion = "supported_nuisance_invariant_phase_specific_field_three_of_three"
    elif len(set(classifications)) == 1:
        conclusion = classifications[0]
    else:
        conclusion = "checkpoint_stratified_metric_field_specificity"
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "implementation_sha256": implementation,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "device": "cpu",
        },
        "provenance": {
            "source_campaign": str(source_path),
            "source_campaign_sha256": _sha256(source_path),
            "source_implementation_sha256": source["implementation_sha256"],
        },
        "summary": {
            "requested": len(config.seeds),
            "completed": len(results),
            "failed": 0,
            "excluded": 0,
            "trained_models": 0,
            "fitted_writers": 0,
            "fitted_predictive_observers": 0,
            "stored_projector_cells": len(results) * len(REGIMES) * len(GROUP_ARMS),
        },
        "aggregates": {
            "conclusion": conclusion,
            "corrected_checkpoint_pass_count": support_count,
            "required_checkpoint_pass_count": 3,
            "classification_by_seed": {
                str(result["seed"]): result["classification"] for result in results
            },
            "classification_counts": {
                name: classifications.count(name) for name in sorted(set(classifications))
            },
            "gate_counts": {
                name: sum(bool(result["gates"][name]) for result in results)
                for name in (
                    "source_valid",
                    "paired_geometry_all_cells",
                    "nuisance_equivalence_all_cells",
                    "semantic_phase_specificity_all_cells",
                    "random_plane_specificity_all_cells",
                    "corrected_checkpoint_gate",
                )
            },
        },
        "results": [
            {
                "experiment_id": result["experiment_id"],
                "seed": result["seed"],
                "scientific_fingerprint": result["scientific_fingerprint"],
                "classification": result["classification"],
                "gates": result["gates"],
                "path": result["artifacts"]["result"],
                "result_sha256": _sha256(Path(result["artifacts"]["result"])),
            }
            for result in results
        ],
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "This corrective reuses selected checkpoints and stored fresh-cohort Jacobian fields.",
            "The five-bin semantic shift is one locked negative control, not a full orbit scan.",
            "Geometry does not establish causal phase-shifted tangent specificity.",
            "The acquisition group covers scale, planar orientation, and constant offset only.",
            "Three selected checkpoints do not establish population prevalence.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "seed_*" / "result.json"),
        },
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
            "data/experiments/tinyllm_local_metric_field_specificity/"
            "20260807_corrective_v1"
        ),
    )
    parser.add_argument("--seeds", type=_ints, default=SEEDS)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = LocalMetricFieldSpecificityConfig(
        seeds=args.seeds, allow_underpowered=args.allow_underpowered
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

