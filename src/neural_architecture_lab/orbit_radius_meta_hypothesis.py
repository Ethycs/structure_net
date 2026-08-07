"""Build the TinyLLM causal orbit-radius evidence record."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-orbit-radius-titration.v1"
HYPOTHESIS_ID = "tinyllm-causal-orbit-radius-threshold-v1"
SEEDS = (7, 17, 29, 41, 53)
DEGREES = (2, 3)
REGIMES = ("composition", "extrapolation")
RELATED_C2_SCHEMA = "nal.tinyllm-c2-character-fusion-radius.v1"
RELATED_C2_HYPOTHESIS = "tinyllm-c2-character-fusion-radius-v1"
RELATED_C2_RESULTS = Path(
    "data/experiments/tinyllm_c2_fusion_radius/20260806_d6_preregistered/campaign_results.json"
)
RELATED_C2_REPORT = Path(
    "docs/08 - Analysis/2026-08-06_tinyllm-c2-fusion-radius.md"
)


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or int(value.get("summary", {}).get("completed", -1)) != 10
        or int(value.get("summary", {}).get("failed", -1)) != 0
        or int(value.get("summary", {}).get("trained_models", -1)) != 0
    ):
        raise ValueError(f"invalid orbit-radius campaign {path}")
    return value


def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    output = []
    fingerprints = set()
    for k in DEGREES:
        for seed in SEEDS:
            detail_path = path.parent / "runs" / f"k{k}" / f"seed_{seed}" / "result.json"
            detail = json.loads(detail_path.read_text())
            if (
                detail.get("schema_version") != EXPERIMENT_SCHEMA
                or detail.get("hypothesis_id") != HYPOTHESIS_ID
                or detail.get("status") != "completed"
                or detail.get("implementation_sha256") != campaign["implementation_sha256"]
                or int(detail.get("k", -1)) != k
                or int(detail.get("seed", -1)) != seed
                or not detail.get("scientific_fingerprint")
            ):
                raise ValueError(f"invalid orbit-radius cell {detail_path}")
            if detail["scientific_fingerprint"] in fingerprints:
                raise ValueError(f"duplicate orbit-radius fingerprint {detail_path}")
            fingerprints.add(detail["scientific_fingerprint"])
            output.append(detail)
    return output


def _related_c2(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    campaign = json.loads(path.read_text())
    if (
        campaign.get("schema_version") != RELATED_C2_SCHEMA
        or campaign.get("hypothesis_id") != RELATED_C2_HYPOTHESIS
        or campaign.get("status") != "completed"
        or campaign.get("summary", {}).get("completed") != 5
        or campaign.get("summary", {}).get("failed") != 0
    ):
        raise ValueError(f"invalid related C2 campaign {path}")
    details = []
    for seed in SEEDS:
        detail_path = path.parent / "runs" / f"seed_{seed}" / "result.json"
        detail = json.loads(detail_path.read_text())
        if (
            detail.get("schema_version") != RELATED_C2_SCHEMA
            or detail.get("hypothesis_id") != RELATED_C2_HYPOTHESIS
            or detail.get("status") != "completed"
            or int(detail.get("seed", -1)) != seed
            or detail.get("implementation_sha256") != campaign["implementation_sha256"]
        ):
            raise ValueError(f"invalid related C2 cell {detail_path}")
        details.append(detail)
    return campaign, details


def build_orbit_radius_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-orbit-radius-titration.md"
    ),
    related_c2_path: Path = RELATED_C2_RESULTS,
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    related_campaign, related_details = _related_c2(related_c2_path)
    evidence = [
        {
            "experiment_id": detail["experiment_id"],
            "k": int(detail["k"]),
            "seed": int(detail["seed"]),
            "checkpoint": detail["provenance"]["checkpoint"],
            "checkpoint_sha256": detail["provenance"]["checkpoint_sha256"],
            "scientific_fingerprint": detail["scientific_fingerprint"],
            "gates": detail["gates"],
            "critical_radii": {
                regime: detail["regimes"][regime]["curves"]["exact_orbit"]["crossing"]["critical_radius"]
                for regime in REGIMES
            },
        }
        for detail in details
    ]
    related_evidence = [
        {
            "hypothesis_id": RELATED_C2_HYPOTHESIS,
            "experiment_id": detail["experiment_id"],
            "seed": int(detail["seed"]),
            "checkpoint_sha256": detail["provenance"]["checkpoint_sha256"],
            "onset_radii": {
                regime: detail["regimes"][regime]["onset_radius"]
                for regime in REGIMES
            },
            "gates": detail["gates"],
        }
        for detail in related_details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM causal orbit-radius threshold",
            "category": "representation",
            "description": "Frozen-checkpoint causal titration of exact deck-orbit amplitude at previously localized Reynolds synthesis fronts.",
            "question": "Does degree-two quotient synthesis reduce to a single shift-stable causal threshold along observed group-orbit amplitude?",
            "prediction": "Endpoint replication, one radial crossing, and shift-stable critical radius each pass in at least four of five degree-two seeds.",
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "primary_gate_passed_but_not_independently_cohort_replicated",
            "evidence_count": 15,
            "direct_experiment_count": 10,
            "related_experiment_count": 5,
            "tested_scope": "ten retained d6 degree-2/3 direct cells plus an independent five-cell C2 cohort; five seeds; 64 fresh exact orbits per shift; frozen synthesis sublayer",
            "subclaims": {
                "k2_endpoint_replication_primary_cohort": "supported_five_of_five",
                "k2_single_radial_crossing_primary_cohort": "supported_five_of_five",
                "k2_shift_stable_threshold_primary_cohort": "supported_four_of_five",
                "k2_primary_campaign_gate": "passed",
                "k2_cross_cohort_shift_stability": "not_supported_three_of_five",
                "k2_early_block0_cross_cohort_stability": "supported_three_of_three",
                "k2_later_front_cross_cohort_stability": "not_supported_zero_of_two",
                "k2_linear_chord_tracks_exact_crossing": "supported_ten_of_ten_within_one_grid_step",
                "k2_universal_quadratic_scaling": "not_supported",
                "k2_direction_control_specificity": "supported_five_of_five_in_related_campaign",
                "k3_endpoint_replication": "not_supported_two_of_five",
                "k3_shift_stable_threshold": "not_supported_zero_of_five",
                "k3_global_radial_account": "not_supported",
                "meta_level_stable_threshold": "not_confirmed",
            },
            "tags": [
                "tinyllm",
                "deck-group",
                "symmetry",
                "causal-intervention",
                "orbit-radius",
                "quotient-front",
                "reynolds-defect",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": "The direct campaign passed its degree-two gates, but an independent fresh C2 cohort found shift-stable onsets in only three of five seeds and endpoint instability at the two later fronts.",
            "confidence": 0.0,
            "confidence_assessment": "primary_campaign_pass_with_independent_cross_cohort_nonreplication",
            "independent_seed_count": 5,
            "num_direct_experiments": 10,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "The direct degree-two cohort crosses the frozen causal task gate exactly once in all five seeds under both shifts.",
                "The degree-two defect direction is nearly aligned with its full-radius direction by radius 0.5, while task sufficiency appears as a magnitude threshold.",
                "Exact and linear-chord critical radii agree within one grid step in all ten degree-two seed/shift cells.",
                "An independent controlled cohort supports direction specificity but rejects checkpoint-global shift stability.",
                "The three block-zero-attention fronts are stable across both cohorts; the two later fronts are cohort-sensitive.",
                "Degree-three endpoint and threshold instability on fresh orbits rules out one global radial mechanism at the frozen cut.",
            ],
            "suggested_hypotheses": [
                "A small subset of block-zero attention heads generates the cross-cohort-stable degree-two defect direction.",
                "Later-front instability is caused by cohort-sensitive routing before the synthesis sublayer rather than by the exact character direction itself.",
                "Degree-three support dependence arises from repeated destruction and resynthesis across distinct character-coupling channels.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {
            "direct_tests": evidence,
            "related_independent_cohort_tests": related_evidence,
        },
        "source_artifacts": [
            str(results_path),
            str(report_path),
            str(related_c2_path),
            str(RELATED_C2_REPORT),
        ],
    }


def build_orbit_radius_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        critical = {
            regime: detail["regimes"][regime]["curves"]["exact_orbit"]["crossing"]["critical_radius"]
            for regime in REGIMES
        }
        anomalies = []
        if int(detail["k"]) == 3:
            anomalies.append("Degree three is a secondary diagnostic with support-relative endpoints.")
        if not detail["gates"]["shift_stable_threshold"]:
            anomalies.append("Composition and extrapolation critical radii exceed the frozen tolerance or are undefined.")
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics={
                    **{gate: float(value) for gate, value in detail["gates"].items()},
                    "composition_critical_radius": -1.0 if critical["composition"] is None else float(critical["composition"]),
                    "extrapolation_critical_radius": -1.0 if critical["extrapolation"] is None else float(critical["extrapolation"]),
                },
                primary_metric=float(detail["all_primary_gates_passed"]),
                model_architecture=[6, int(detail["k"])],
                model_parameters=29_956_224,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["checkpoint"],
                observations=[
                    "Frozen checkpoint; no training; 64 fresh exact deck orbits per shift; nine causal radii."
                ],
                anomalies=anomalies,
                timestamp=completed,
            )
        )
    return output


def store_orbit_radius_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_orbit_radius_meta_hypothesis(results_path)
    experiments = build_orbit_radius_experiment_results(record, results_path)
    storage: dict[str, Any] = {
        "aggregate_path": str(output_path),
        "experiment_ids": [item.experiment_id for item in experiments],
    }
    if chromadb_path is not None:
        from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger

        logger = StandardizedLogger(
            LoggingConfig(
                project_name="structure_net_meta_hypotheses",
                queue_dir="data/experiment_queue",
                sent_dir="data/experiment_sent",
                rejected_dir="data/experiment_rejected",
                enable_wandb=False,
                auto_upload=False,
                enable_chromadb=True,
                chromadb_path=str(chromadb_path),
            )
        )
        logger.log_hypothesis(record["hypothesis"])
        storage["result_hashes"] = [
            logger.log_experiment_result(item) for item in experiments
        ]
        storage["chromadb_path"] = str(chromadb_path)
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output_path)
    return record


__all__ = [
    "build_orbit_radius_meta_hypothesis",
    "build_orbit_radius_experiment_results",
    "store_orbit_radius_meta_hypothesis",
]
