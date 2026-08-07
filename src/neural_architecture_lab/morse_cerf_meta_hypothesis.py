"""Build the TinyLLM Morse--Cerf meta-hypothesis evidence record."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-morse-cerf.v1"
META_HYPOTHESIS_ID = "tinyllm-equivariant-morse-cerf-quotient-front-v1"


def _read_campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("status") != "completed"
        or value.get("hypothesis_id") != META_HYPOTHESIS_ID
        or int(value.get("summary", {}).get("completed", -1)) != 10
    ):
        raise ValueError(f"invalid Morse--Cerf campaign {path}")
    return value


def _details(results_path: Path) -> list[dict[str, Any]]:
    campaign = _read_campaign(results_path)
    output = []
    root = results_path.parent
    for k in (2, 3):
        for seed in (7, 17, 29, 41, 53):
            detail_path = root / "runs" / f"k{k}" / f"seed_{seed}" / "result.json"
            detail = json.loads(detail_path.read_text())
            arrays_path = Path(detail["landscapes_path"])
            if (
                detail.get("schema_version") != EXPERIMENT_SCHEMA
                or detail.get("status") != "completed"
                or detail.get("implementation_sha256") != campaign["implementation_sha256"]
                or int(detail.get("k", -1)) != k
                or int(detail.get("seed", -1)) != seed
                or not arrays_path.is_file()
                or detail.get("landscapes_sha256")
                != hashlib.sha256(arrays_path.read_bytes()).hexdigest()
            ):
                raise ValueError(f"invalid Morse--Cerf cell {detail_path}")
            output.append(detail)
    return output


def build_morse_cerf_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path("docs/08 - Analysis/2026-08-06_tinyllm-morse-cerf.md"),
) -> dict[str, Any]:
    campaign = _read_campaign(results_path)
    details = _details(results_path)
    evidence = []
    for detail in details:
        evidence.append(
            {
                "experiment_id": detail["experiment_id"],
                "k": int(detail["k"]),
                "seed": int(detail["seed"]),
                "checkpoint": detail["provenance"]["checkpoint"],
                "checkpoint_sha256": detail["provenance"]["checkpoint_sha256"],
                "regimes": {
                    regime: {
                        "causal_front": detail["regimes"][regime]["summary"]["causal_front"],
                        "task_morse_front": detail["regimes"][regime]["summary"]["task_morse_front"],
                        "commutator_front": detail["regimes"][regime]["summary"]["commutator_front"],
                        "gates": detail["regimes"][regime]["summary"]["gates"],
                    }
                    for regime in ("composition", "extrapolation")
                },
            }
        )
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": META_HYPOTHESIS_ID,
            "name": "TinyLLM equivariant Morse--Cerf quotient front",
            "category": "representation",
            "description": "Finite-grid Morse and merge-tree analysis of exact deck-orbit mixture simplices under frozen downstream continuation.",
            "question": "Is exact orbit-averaging quotient formation a stable symmetry-constrained Morse transition rather than a task-loss threshold?",
            "prediction": "All five preregistered gates pass in at least four of five seeds separately for both degrees and both shifts.",
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "not_confirmed_near_front_events_without_stable_morse_normal_form",
            "evidence_count": len(evidence),
            "direct_experiment_count": len(evidence),
            "tested_scope": "retained d6 degree-2/3 ladder; five seeds; 32 exact orbits; composition and extrapolation; six residual sublayers",
            "subclaims": {
                "early_cover_structure_k2": "supported_both_shifts_five_of_five",
                "near_front_events_k2": "supported_both_shifts_five_of_five",
                "mature_basin_k2": "not_supported_composition_two_of_five",
                "early_cover_structure_k3": "supported_composition_four_of_five_extrapolation_five_of_five",
                "near_front_events_k3": "supported_composition_four_of_five_extrapolation_five_of_five",
                "mature_basin_k3": "not_supported_two_of_five_and_three_of_five",
                "control_specificity": "not_supported_across_both_degrees_and_shifts",
                "commutator_morse_front": "not_supported_structurally_miscalibrated_index_gate",
                "interval_certification": "not_attempted_no_stable_seed7_candidate",
                "full_hypothesis": "not_confirmed",
            },
            "tags": ["tinyllm", "morse-theory", "cerf-theory", "deck-group", "causal-intervention", "multi-seed"],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": "Near-front events replicated, but mature basins and control specificity were shift- and degree-dependent; the commutator Morse-index gate produced no valid fronts.",
            "confidence": 0.0,
            "confidence_assessment": "complete_five_seed_preregistered_failure_with_supported_local_event_subclaim",
            "independent_seed_count": 5,
            "num_direct_experiments": len(evidence),
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "Degree-two task landscapes show early cover structure and a near-causal event in every seed and shift.",
                "Mature task basins do not replicate on degree-two composition or either degree-three shift.",
                "Random and phase-shuffled controls often acquire comparable mature fronts.",
                "Degree-three landscapes contain many more discrete events and do not support one universal handle attachment.",
                "A Reynolds commutator approaching global closure becomes flat zero, so a nondegenerate center-minimum gate is inappropriate.",
            ],
            "suggested_hypotheses": [
                "Near-front task-landscape restructuring is real but model-specific rather than a universal Morse normal form.",
                "The correct autonomous endpoint is a uniform simplex commutator norm, not barycenter Morse index.",
                "Control-matched excess barriers may isolate orbit-specific topology better than absolute mature-front timing.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path), str(report_path)],
    }


def build_morse_cerf_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    completed = datetime.fromisoformat(
        _read_campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        summaries = [detail["regimes"][regime]["summary"] for regime in ("composition", "extrapolation")]
        gate_names = list(summaries[0]["gates"])
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=META_HYPOTHESIS_ID,
                metrics={
                    **{
                        f"{gate}_both_shifts": float(all(item["gates"][gate] for item in summaries))
                        for gate in gate_names
                    },
                    "task_front_present_both_shifts": float(all(item["task_morse_front"] is not None for item in summaries)),
                    "event_count": float(sum(len(item["events"]) for item in summaries)),
                },
                primary_metric=float(all(item["all_gates_passed"] for item in summaries)),
                model_architecture=[6, int(detail["k"])],
                model_parameters=29_956_224,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["checkpoint"],
                observations=["Frozen checkpoint; exact task-orbit simplex; cyclically symmetrized target-posterior KL."],
                anomalies=["The nonnegative commutator is zero at vertices and approaches a degenerate flat-zero closure landscape."],
                timestamp=completed,
            )
        )
    return output


def store_morse_cerf_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_morse_cerf_meta_hypothesis(results_path)
    experiments = build_morse_cerf_experiment_results(record, results_path)
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
        storage["result_hashes"] = [logger.log_experiment_result(item) for item in experiments]
        storage["chromadb_path"] = str(chromadb_path)
        storage["hypotheses_collection"] = "hypotheses"
        storage["experiments_collection"] = "experiments"
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return record


__all__ = [
    "build_morse_cerf_meta_hypothesis",
    "build_morse_cerf_experiment_results",
    "store_morse_cerf_meta_hypothesis",
]
