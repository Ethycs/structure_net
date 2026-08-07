"""Build conservative ledger records for the August 2026 geometry suite."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"


def _read(path: Path, schema: str, hypothesis_id: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != schema:
        raise ValueError(f"unsupported schema in {path}")
    if value.get("hypothesis_id") != hypothesis_id:
        raise ValueError(f"wrong hypothesis id in {path}")
    if value.get("status") != "completed":
        raise ValueError(f"incomplete result in {path}")
    return value


def _record(
    *,
    result: Mapping[str, Any],
    name: str,
    question: str,
    prediction: str,
    status: str,
    confirmed: bool,
    scope: str,
    subclaims: Mapping[str, str],
    evidence: list[dict[str, Any]],
    reason: str,
    metrics: Mapping[str, Any],
    insights: list[str],
    boundaries: list[str],
    artifacts: list[str],
) -> dict[str, Any]:
    hypothesis_id = str(result["hypothesis_id"])
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": hypothesis_id,
            "name": name,
            "category": "representation",
            "description": question,
            "question": question,
            "prediction": prediction,
            "created_at": result["completed_at"],
            "tested": True,
            "confirmed": confirmed,
            "confirmation_status": status,
            "evidence_count": len(evidence),
            "direct_experiment_count": len(evidence),
            "tested_scope": scope,
            "subclaims": dict(subclaims),
            "tags": ["tinyllm", "geometry", "causal-test", "preregistered"],
            "explicitly_not_tested": boundaries,
        },
        "result": {
            "confirmed": confirmed,
            "confirmation_reason": reason,
            "confidence": 1.0 if confirmed else 0.0,
            "confidence_assessment": (
                "passed_declared_single_transition_gate"
                if confirmed
                else "failed_or_incomplete_preregistered_gate"
            ),
            "num_direct_experiments": len(evidence),
            "descriptive_metrics": dict(metrics),
            "key_insights": insights,
            "completed_at": result["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": artifacts,
    }


def build_relative_carrier_record(results_path: Path, report_path: Path) -> dict[str, Any]:
    campaign = _read(
        results_path,
        "nal.tinyllm-relative-carrier-factorization.v1",
        "tinyllm-relative-carrier-fixed-quotient-v1",
    )
    if campaign["summary"] != {"requested": 20, "scheduled": 20, "completed": 20, "failed": 0, "reused": 0}:
        raise ValueError("relative-carrier campaign is not the complete 20-cell run")
    evidence = []
    for path in sorted((results_path.parent / "runs").glob("*/seed_*/result.json")):
        cell = json.loads(path.read_text(encoding="utf-8"))
        carrier = cell["carrier_analysis"]
        evidence.append({
            "experiment_id": cell["experiment_id"],
            "condition": cell["condition"],
            "seed": int(cell["seed"]),
            "checkpoint": cell["training"].get("checkpoint"),
            "composition_carrier_alignment": float(carrier["composition"]["circular_alignment"]),
            "extrapolation_carrier_alignment": float(carrier["extrapolation"]["circular_alignment"]),
            "composition_carrier_degree": float(carrier["composition"]["winding_degree"]),
            "extrapolation_carrier_degree": float(carrier["extrapolation"]["winding_degree"]),
        })
    if len(evidence) != 20:
        raise ValueError("relative-carrier direct cell count is not 20")
    arms = campaign["aggregates"]["arms"]
    learned = arms["learned_vector_carrier"]
    return _record(
        result=campaign,
        name="Relative carrier followed by a fixed quotient",
        question="Does separating relative S1 carrier recovery from a fixed downstream quotient produce a stable quotient under composition and extrapolation?",
        prediction="At least four of five learned-carrier seeds pass carrier and quotient gates on both shifts.",
        status="not_confirmed_support_relative_carrier",
        confirmed=False,
        scope="d8 N3; four arms; five seeds; composition and outside-range extrapolation",
        subclaims={
            "learned_carrier_composition": "supported_four_of_five",
            "learned_carrier_extrapolation": "not_supported_zero_of_five",
            "fixed_quotient_task_control": "supported",
            "full_factorization": "not_confirmed_zero_of_five",
        },
        evidence=evidence,
        reason="The learned carrier passed composition in four seeds but extrapolation in zero, so no seed passed the joint factorization gate.",
        metrics=campaign["aggregates"],
        insights=[
            "The learned vector carrier is accurate and degree one on composition.",
            "Outside-range alignment collapses and winding becomes unstable.",
            "A fixed quotient improves composition task accuracy but cannot repair the extrapolating carrier.",
        ],
        boundaries=["other carrier architectures", "real sensor data", "formal equivariance certificates"],
        artifacts=[str(results_path), str(report_path)],
    )


def build_degree_ladder_record(results_path: Path, report_path: Path) -> dict[str, Any]:
    campaign = _read(
        results_path,
        "nal.tinyllm-degree-k-ladder.v1",
        "tinyllm-degree-k-finite-quotient-ladder-v1",
    )
    if campaign["summary"] != {"requested": 15, "completed": 15, "failed": 0}:
        raise ValueError("degree ladder is not the complete 15-cell run")
    evidence = []
    for path in sorted((results_path.parent / "runs").glob("k*/seed_*/result.json")):
        cell = json.loads(path.read_text(encoding="utf-8"))
        full_probe = (cell.get("branch_probes") or {}).get("full")
        evidence.append({
            "experiment_id": cell["experiment_id"],
            "k": int(cell["k"]),
            "seed": int(cell["seed"]),
            "composition_degree": float(cell["maps"]["composition"]["winding_degree"]),
            "extrapolation_degree": float(cell["maps"]["extrapolation"]["winding_degree"]),
            "composition_full_branch_accuracy": (
                None if full_probe is None else float(full_probe["evaluations"]["composition"]["balanced_accuracy"])
            ),
            "extrapolation_full_branch_accuracy": (
                None if full_probe is None else float(full_probe["evaluations"]["extrapolation"]["balanced_accuracy"])
            ),
            "net_defect_charge": int(cell["net_defect_charge"]),
            "checkpoint": cell["training"]["checkpoint"],
        })
    if len(evidence) != 15:
        raise ValueError("degree-ladder direct cell count is not 15")
    degrees = campaign["aggregates"]["degrees"]
    return _record(
        result=campaign,
        name="Degree-k finite quotient ladder",
        question="Does a fixed analytic degree-k carrier let TinyLLM form increasingly nontrivial finite-fiber quotients for k=1,2,3?",
        prediction="Every k passes its map, topology, conditional branch, and seed-7 defect-charge gates in at least four of five seeds.",
        status="not_confirmed_maps_preserved_fibers_not_quotiented",
        confirmed=False,
        scope="d6; analytic carrier; k=1,2,3; five seeds; two shifts",
        subclaims={
            "degree_one_ladder_rung": "confirmed_five_of_five",
            "degree_two_map": "supported_five_of_five",
            "degree_three_map": "supported_five_of_five",
            "degree_two_fiber_contraction": "not_supported_zero_of_five",
            "degree_three_fiber_contraction": "not_supported_zero_of_five",
            "seed7_charge_equals_k": "supported_k1_k2_not_k3",
            "full_ladder": "not_confirmed",
        },
        evidence=evidence,
        reason="All degree maps survived both shifts, but k=2 and k=3 retained near-perfect conditional sheet decodability at post-MLP and full depth.",
        metrics=campaign["aggregates"],
        insights=[
            "Map degree, alignment, and persistent H1 generalize through k=3.",
            "The transformer preserves finite-sheet identity instead of forming the intended quotient for k=2 and k=3.",
            "Seed-7 net defect charge is 1, 2, and 2 for target degrees 1, 2, and 3.",
        ],
        boundaries=campaign["method_boundaries"],
        artifacts=[str(results_path), str(report_path)],
    )


def build_defect_certificate_record(results_path: Path, report_path: Path) -> dict[str, Any]:
    result = _read(
        results_path,
        "nal.tinyllm-defect-certification.v1",
        "tinyllm-d6-step15-defect-certificate-v1",
    )
    numerical = result["numerical_localization"]
    evidence = [{
        "experiment_id": "tinyllm-d6-step15-defect-certificate",
        "root_phase": float(numerical["root"]["phase"]),
        "root_path_fraction": float(numerical["root"]["path_fraction"]),
        "root_residual_norm": float(numerical["root_residual_norm"]),
        "oriented_jacobian_determinant": float(numerical["oriented_jacobian_determinant_path_phase"]),
        "endpoint_degree_change": int(numerical["endpoint_degree_change"]),
    }]
    surrogate_path = Path(
        "data/experiments/tinyllm_defect_certification/"
        "20260806_d6_step15_surrogate_interval/results.json"
    )
    surrogate = json.loads(surrogate_path.read_text()) if surrogate_path.is_file() else None
    record = _record(
        result=result,
        name="Computer-assisted d6 step-15 defect certificate",
        question="Can the localized d6 degree-changing cell be upgraded to a unique positive-index actual-network root certificate?",
        prediction="Outward-rounded boundary, root-isolation, Jacobian-sign, endpoint winding, and remainder gates all pass.",
        status="not_formally_certified_numerical_regular_positive_root",
        confirmed=False,
        scope="one d6 seed-7 step-14 to step-15 interpolation cylinder",
        subclaims={
            "floating_point_root_localized": "supported",
            "positive_numerical_index": "supported",
            "endpoint_degree_change_plus_one": "supported",
            "actual_network_interval_certificate": "not_obtained",
            "stored_chebyshev_surrogate_interval_certificate": (
                "supported" if surrogate and surrogate.get("surrogate_unique_positive_root_certified") else "not_tested"
            ),
        },
        evidence=evidence,
        reason="The numerical root is stable and regular, but the actual transformer map lacks an outward-rounded enclosure and rigorous surrogate remainder bound.",
        metrics={
            "numerical_localization": numerical,
            "certification_gates": result["certification_gates"],
            "surrogate_error": result["chebyshev_surrogate"]["maximum_sampled_value_error"],
        },
        insights=[
            "The root residual is below 1e-8 and the path-phase determinant is positive.",
            "A sampled Chebyshev audit cannot substitute for an actual-network interval enclosure.",
            "The stored floating Chebyshev polynomial has a directed-rounding unique positive-root certificate, but rigorous transfer to the transformer remains open.",
        ],
        boundaries=[result["method_boundary"]],
        artifacts=[str(results_path), str(report_path), str(result["provenance"]["source_result"])] + ([str(surrogate_path)] if surrogate else []),
    )
    if surrogate:
        record["result"]["descriptive_metrics"]["surrogate_interval_certificate"] = {
            "certified": surrogate["surrogate_unique_positive_root_certified"],
            "actual_network_transfer_certified": surrogate["actual_network_transfer_certified"],
            "candidate_hull_physical": surrogate["isolation"]["candidate_hull_physical"],
            "oriented_determinant_interval": surrogate["krawczyk"]["oriented_determinant_interval"],
        }
    return record


def build_normal_kernel_record(results_path: Path, report_path: Path) -> dict[str, Any]:
    result = _read(
        results_path,
        "nal.tinyllm-normal-kernel-radius.v1",
        "tinyllm-normal-jet-kernel-structural-radius-v1",
    )
    confirmed = bool(result["confirmed_in_declared_edit_space"])
    evidence = [{
        "experiment_id": "tinyllm-d6-step15-normal-kernel-radius",
        "edit_dimension": int(result["edit_space"]["dimension"]),
        "predicted_rank": int(result["predicted_direction_rank"]),
        "predicted_radius": float(result["normal_kernel"]["predicted_radius"]),
        "direct_radius": float(result["direct_optimization"]["radius"]),
        "radius_ratio": float(result["radius_ratio_predicted_over_direct"]),
        "predicted_residual": float(result["predicted_direction_search"]["residual_norm"]),
    }]
    return _record(
        result=result,
        name="Normal jet-kernel structural radius",
        question="Does the reduced normal kernel predict the minimum intervention radius to the degree discriminant better than random directions?",
        prediction="The radius, residual, directional-rank, and resolved degree-transition gates all pass.",
        status="supported_in_declared_single_transition_edit_space" if confirmed else "not_confirmed",
        confirmed=confirmed,
        scope="one d6 seed-7 transition; block-1 MLP output bias; Euclidean metric",
        subclaims={
            "local_radius_prediction": "supported" if confirmed else "not_confirmed",
            "random_direction_superiority": "supported" if result["primary_gates"]["predicted_direction_best_ten_percent"] else "not_supported",
            "resolved_degree_transition": "supported" if result["primary_gates"]["resolved_degree_transition"] else "not_supported",
            "population_generalization": "not_tested",
        },
        evidence=evidence,
        reason="Every preregistered local gate passes on the 4,096-point refined phase bracket." if confirmed else "At least one preregistered local gate failed.",
        metrics={
            "primary_gates": result["primary_gates"],
            "radius_ratio": result["radius_ratio_predicted_over_direct"],
            "degree_bracket": result["degree_bracket"],
        },
        insights=[
            "The predicted radius is within 10.6% of the direct nonlinear radius.",
            "The predicted direction reaches a sub-1e-6 residual and ranks in the best 10% of 33 directions.",
            "The refined phase grid resolves a degree-zero to degree-one transition.",
        ],
        boundaries=result["method_boundaries"],
        artifacts=[str(results_path), str(report_path)],
    )


def build_suite_records(
    relative_path: Path,
    ladder_path: Path,
    certificate_path: Path,
    kernel_path: Path,
) -> list[dict[str, Any]]:
    report_root = Path("docs/08 - Analysis")
    return [
        build_relative_carrier_record(relative_path, report_root / "2026-08-06_tinyllm-relative-carrier-factorization.md"),
        build_degree_ladder_record(ladder_path, report_root / "2026-08-06_tinyllm-degree-k-ladder.md"),
        build_defect_certificate_record(certificate_path, report_root / "2026-08-06_tinyllm-defect-certification.md"),
        build_normal_kernel_record(kernel_path, report_root / "2026-08-06_tinyllm-normal-kernel-radius.md"),
    ]


def _experiment_results(record: Mapping[str, Any]) -> list[ExperimentResult]:
    completed = datetime.fromisoformat(str(record["result"]["completed_at"]).replace("Z", "+00:00"))
    hypothesis_id = str(record["hypothesis"]["id"])
    output = []
    for index, evidence in enumerate(record["evidence"]["direct_tests"]):
        numeric = {
            key: float(value)
            for key, value in evidence.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        primary = next(iter(numeric.values()), 0.0)
        output.append(ExperimentResult(
            experiment_id=str(evidence.get("experiment_id", f"{hypothesis_id}-{index}")),
            hypothesis_id=hypothesis_id,
            metrics=numeric or {"completed": 1.0},
            primary_metric=primary,
            model_architecture=[],
            model_parameters=0,
            training_time=0.0,
            model_checkpoint=evidence.get("checkpoint"),
            observations=[str(record["result"]["confirmation_reason"])],
            anomalies=list(record["hypothesis"]["explicitly_not_tested"]),
            timestamp=completed,
        ))
    return output


def store_suite_records(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    chromadb_path: Path | None = None,
) -> list[dict[str, Any]]:
    logger = None
    if chromadb_path is not None:
        from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger

        logger = StandardizedLogger(LoggingConfig(
            project_name="structure_net_meta_hypotheses",
            queue_dir="data/experiment_queue",
            sent_dir="data/experiment_sent",
            rejected_dir="data/experiment_rejected",
            enable_wandb=False,
            auto_upload=False,
            enable_chromadb=True,
            chromadb_path=str(chromadb_path),
        ))
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        hypothesis_id = record["hypothesis"]["id"]
        experiments = _experiment_results(record)
        storage: dict[str, Any] = {
            "aggregate_path": str(output_dir / f"{hypothesis_id}.json"),
            "experiment_ids": [item.experiment_id for item in experiments],
        }
        if logger is not None:
            logger.log_hypothesis(record["hypothesis"])
            storage["result_hashes"] = [logger.log_experiment_result(item) for item in experiments]
            storage["chromadb_path"] = str(chromadb_path)
            storage["hypotheses_collection"] = "hypotheses"
            storage["experiments_collection"] = "experiments"
        record["storage"] = storage
        Path(storage["aggregate_path"]).write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return records


__all__ = [
    "build_relative_carrier_record",
    "build_degree_ladder_record",
    "build_defect_certificate_record",
    "build_normal_kernel_record",
    "build_suite_records",
    "store_suite_records",
]
