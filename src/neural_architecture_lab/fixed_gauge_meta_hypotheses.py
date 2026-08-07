"""Build and store the TinyLLM fixed-gauge writer evidence chain."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
WRITER_SCHEMA = "nal.tinyllm-fixed-semantic-gauge-writer.v1"
WRITER_HYPOTHESIS_ID = "tinyllm-c2-fixed-semantic-gauge-writer-v1"
WRITER_EVIDENCE_ROLE = (
    "preregistered_post_outcome_underpowered_mechanistic_positive_control"
)
WRITER_IMPLEMENTATION_SHA256 = (
    "4508847ca4fa85c2220e3691d8e9922d714e768b0f4891ce5237a4a74089c808"
)
WRITER_CAMPAIGN_SHA256 = (
    "de80e30c23e06801c75d6fae899c67d0da82b86fdaff9158d94270597df8379c"
)
WRITER_GATE_COUNTS = {
    "observation_contract": 0,
    "continuous_target_control_contract": 3,
    "fixed_gauge_causal_writer": 0,
    "shuffled_specificity": 3,
}

DECOMPOSITION_SCHEMA = "nal.tinyllm-fixed-gauge-error-decomposition.v1"
DECOMPOSITION_HYPOTHESIS_ID = "tinyllm-c2-fixed-gauge-error-decomposition-v1"
DECOMPOSITION_EVIDENCE_ROLE = (
    "preregistered_post_outcome_underpowered_mechanistic_diagnostic"
)
DECOMPOSITION_IMPLEMENTATION_SHA256 = (
    "37175d78dd4768e448bea482a271309e67f00863ffa6202a799346fad008ac38"
)
DECOMPOSITION_CAMPAIGN_SHA256 = (
    "be4cf87248550c8ace7fc474efff2ce168bce3c9002932ecf6063977c91f0fa6"
)
PRIMARY_WRITER_CAMPAIGN_SHA256 = WRITER_CAMPAIGN_SHA256
DECOMPOSITION_GATE_COUNTS = {
    "primary_replay_contract": 3,
    "oracle_carrier_contract": 3,
    "continuous_target_control_contract": 3,
    "observed_fit_oracle_eval": 0,
    "oracle_fit_oracle_eval": 0,
}
CAPACITY_SCHEMA = "nal.tinyllm-fixed-gauge-writer-capacity.v1"
CAPACITY_HYPOTHESIS_ID = "tinyllm-c2-fixed-gauge-writer-capacity-v1"
CAPACITY_EVIDENCE_ROLE = (
    "preregistered_post_outcome_underpowered_mechanistic_diagnostic"
)
CAPACITY_IMPLEMENTATION_SHA256 = (
    "7c284e35b5afc225eea45309262ab83c5f6d276736a557ebf20675ed3ccbfe7b"
)
CAPACITY_CAMPAIGN_SHA256 = (
    "c5592ee53b96e2d064a992070be6c3e699b24d88dbb658283e8dc2f00c267078"
)
CAPACITY_CONTRACT_COUNTS = {
    "predecessor_replay_contract": 3,
    "oracle_and_context_contract": 3,
    "continuous_target_control_contract": 3,
}
CAPACITY_CANDIDATE_COUNTS = {
    "quotient_order2": 0,
    "quotient_order4": 0,
    "quotient_order40": 0,
    "quotient_order4_context": 0,
}
SEEDS = (7, 29, 53)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_campaign(
    path: Path,
    *,
    schema: str,
    hypothesis_id: str,
    evidence_role: str,
    implementation: str,
    campaign_sha256: str,
    gate_counts: Mapping[str, int],
) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        _sha256(path) != campaign_sha256
        or value.get("schema_version") != schema
        or value.get("hypothesis_id") != hypothesis_id
        or value.get("status") != "completed"
        or value.get("evidence_role") != evidence_role
        or value.get("implementation_sha256") != implementation
        or int(value.get("summary", {}).get("requested", -1)) != 3
        or int(value.get("summary", {}).get("completed", -1)) != 3
        or int(value.get("summary", {}).get("failed", -1)) != 0
        or int(value.get("summary", {}).get("trained_models", -1)) != 0
        or int(value.get("summary", {}).get("fitted_predictive_observers", -1))
        != 0
        or value.get("aggregates", {}).get("gate_counts") != dict(gate_counts)
    ):
        raise ValueError(f"invalid fixed-gauge campaign {path}")
    return value


def _load_details(
    path: Path,
    campaign: Mapping[str, Any],
    *,
    schema: str,
    hypothesis_id: str,
    evidence_role: str,
    implementation: str,
) -> list[dict[str, Any]]:
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if set(entries) != set(SEEDS):
        raise ValueError(f"invalid fixed-gauge seed index {path}")
    output = []
    fingerprints: set[str] = set()
    for seed in SEEDS:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        detail = json.loads(detail_path.read_text())
        if (
            detail.get("schema_version") != schema
            or detail.get("hypothesis_id") != hypothesis_id
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != evidence_role
            or detail.get("implementation_sha256") != implementation
            or int(detail.get("seed", -1)) != seed
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or _sha256(detail_path) != entry.get("result_sha256")
        ):
            raise ValueError(f"invalid fixed-gauge result {detail_path}")
        fingerprint = detail["scientific_fingerprint"]
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate fixed-gauge fingerprint {detail_path}")
        fingerprints.add(fingerprint)
        output.append(detail)
    return output


def build_fixed_semantic_gauge_writer_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-fixed-semantic-gauge-writer.md"
    ),
) -> dict[str, Any]:
    campaign = _load_campaign(
        results_path,
        schema=WRITER_SCHEMA,
        hypothesis_id=WRITER_HYPOTHESIS_ID,
        evidence_role=WRITER_EVIDENCE_ROLE,
        implementation=WRITER_IMPLEMENTATION_SHA256,
        campaign_sha256=WRITER_CAMPAIGN_SHA256,
        gate_counts=WRITER_GATE_COUNTS,
    )
    details = _load_details(
        results_path,
        campaign,
        schema=WRITER_SCHEMA,
        hypothesis_id=WRITER_HYPOTHESIS_ID,
        evidence_role=WRITER_EVIDENCE_ROLE,
        implementation=WRITER_IMPLEMENTATION_SHA256,
    )
    if campaign["aggregates"].get("confirmed") is not False:
        raise ValueError(f"fixed-gauge writer cannot be promoted {results_path}")
    for detail in details:
        gates = detail["gates"]
        if (
            gates.get("observation_contract") is not False
            or gates.get("continuous_target_control_contract") is not True
            or gates.get("fixed_gauge_causal_writer") is not False
            or gates.get("shuffled_specificity") is not True
            or len(detail.get("heldout_cells", [])) != 4
            or min(
                cell["coordinate_metrics"]["fixed_gauge"]["variance_explained"]
                for cell in detail["heldout_cells"]
            )
            < 0.97
        ):
            raise ValueError(f"invalid fixed-gauge gate split for seed {detail['seed']}")
    evidence = [
        {
            "experiment_id": detail["experiment_id"],
            "seed": detail["seed"],
            "scientific_fingerprint": detail["scientific_fingerprint"],
            "gates": detail["gates"],
            "heldout": [
                {
                    "cohort": cell["cohort"],
                    "regime": cell["regime"],
                    "coordinate_metrics": cell["coordinate_metrics"]["fixed_gauge"],
                    "continuous": cell["states"]["fixed_gauge"]["continuous"],
                }
                for cell in detail["heldout_cells"]
            ],
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": WRITER_HYPOTHESIS_ID,
            "name": "TinyLLM fixed semantic-gauge causal writer",
            "category": "representation",
            "description": (
                "An observation-derived analytic C2 neutral carrier and target-local "
                "linear writer reproduce stable frozen quotient defects."
            ),
            "question": (
                "Can one fixed three-channel semantic gauge drive all three stable "
                "frozen continuations without a learned sidecar?"
            ),
            "prediction": "All four primary gates pass in all three checkpoints.",
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_observation_and_causal_writer_failed_three_of_three"
            ),
            "evidence_count": 3,
            "direct_experiment_count": 3,
            "power_profile": "underpowered_preregistered_post_outcome_positive_control",
            "evidence_role": WRITER_EVIDENCE_ROLE,
            "tested_scope": (
                "three selected d6 C2 block-0 fronts; two held-out cohorts under "
                "composition and extrapolation"
            ),
            "subclaims": {
                "fixed_portable_semantic_gauge": "rejected_zero_of_three",
                "continuous_target_controls": "supported_three_of_three",
                "shuffled_specificity": "supported_three_of_three",
                "high_coordinate_fit": "supported_worst_r2_at_least_0_973",
                "seed53_near_boundary": "supported_three_of_four_cells",
            },
            "tags": [
                "tinyllm",
                "c2",
                "fixed-gauge",
                "causal-patching",
                "analytic-positive-control",
                "negative-result",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "Observation and continuous causal gates fail in all three selected "
                "checkpoints despite high coordinate fit and clean specificity."
            ),
            "confidence": 0.94,
            "confidence_assessment": "exact_underpowered_post_outcome_negative_control",
            "independent_seed_count": 3,
            "num_direct_experiments": 3,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "All exact/direct/zero target controls behave as required.",
                "The observed fixed carrier narrowly misses its sensor contract.",
                "Coordinate R2 above 0.97 does not imply frozen causal sufficiency.",
                "Seed 53 passes three of four writer cells but the joint gate fails.",
            ],
            "suggested_hypotheses": [
                "An exact oracle carrier isolates sensor error from writer capacity.",
                "A nonlinear neutral writer can repair task-effective residual error.",
                "Additional invariant sensor context is required beyond phase and energy.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": _sha256(results_path),
            "implementation_sha256": WRITER_IMPLEMENTATION_SHA256,
        },
    }


def build_fixed_gauge_error_decomposition_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-fixed-gauge-error-decomposition.md"
    ),
) -> dict[str, Any]:
    campaign = _load_campaign(
        results_path,
        schema=DECOMPOSITION_SCHEMA,
        hypothesis_id=DECOMPOSITION_HYPOTHESIS_ID,
        evidence_role=DECOMPOSITION_EVIDENCE_ROLE,
        implementation=DECOMPOSITION_IMPLEMENTATION_SHA256,
        campaign_sha256=DECOMPOSITION_CAMPAIGN_SHA256,
        gate_counts=DECOMPOSITION_GATE_COUNTS,
    )
    details = _load_details(
        results_path,
        campaign,
        schema=DECOMPOSITION_SCHEMA,
        hypothesis_id=DECOMPOSITION_HYPOTHESIS_ID,
        evidence_role=DECOMPOSITION_EVIDENCE_ROLE,
        implementation=DECOMPOSITION_IMPLEMENTATION_SHA256,
    )
    expected = {str(seed): "writer_or_carrier_limited" for seed in SEEDS}
    if (
        campaign["aggregates"].get("classification_by_seed") != expected
        or campaign.get("provenance", {}).get("primary_campaign_sha256")
        != PRIMARY_WRITER_CAMPAIGN_SHA256
    ):
        raise ValueError(f"invalid fixed-gauge decomposition classes {results_path}")
    for detail in details:
        gates = detail["gates"]
        if (
            detail.get("classification") != "writer_or_carrier_limited"
            or gates.get("primary_replay_contract") is not True
            or gates.get("oracle_carrier_contract") is not True
            or gates.get("continuous_target_control_contract") is not True
            or gates.get("observed_fit_oracle_eval") is not False
            or gates.get("oracle_fit_oracle_eval") is not False
            or float(gates.get("maximum_primary_replay_error", -1.0)) != 0.0
            or len(detail.get("heldout_cells", [])) != 4
            or max(
                audit["maximum_shift_bins"]
                for cohort in detail.get("oracle_audit", {}).values()
                for audit in cohort.values()
            )
            > 1e-8
        ):
            raise ValueError(f"invalid oracle decomposition seed {detail['seed']}")
    evidence = [
        {
            "experiment_id": detail["experiment_id"],
            "seed": detail["seed"],
            "scientific_fingerprint": detail["scientific_fingerprint"],
            "classification": detail["classification"],
            "gates": detail["gates"],
            "oracle_fit": detail["alignment_fit"]["oracle_coordinate_metrics"],
            "heldout": [
                {
                    "cohort": cell["cohort"],
                    "regime": cell["regime"],
                    "coordinate_metrics": cell["oracle_evaluation"][
                        "coordinate_metrics"
                    ]["oracle_fit_oracle_eval"],
                    "continuous": cell["oracle_evaluation"]["states"][
                        "oracle_fit_oracle_eval"
                    ]["continuous"],
                }
                for cell in detail["heldout_cells"]
            ],
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": DECOMPOSITION_HYPOTHESIS_ID,
            "name": "TinyLLM fixed-gauge sensor/writer error decomposition",
            "category": "representation",
            "description": (
                "An exact latent degree-two carrier isolates observation recovery "
                "error from fixed-gauge linear-writer capacity."
            ),
            "question": (
                "Does replacing the imperfect observed carrier with the exact oracle "
                "carrier rescue the fixed causal writer?"
            ),
            "prediction": (
                "Oracle substitution classifies each failed checkpoint as sensor- "
                "or writer/carrier-limited under unchanged continuous gates."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "diagnostic_writer_or_carrier_limited_three_of_three",
            "evidence_count": 3,
            "direct_experiment_count": 3,
            "power_profile": "underpowered_preregistered_post_outcome_diagnostic",
            "evidence_role": DECOMPOSITION_EVIDENCE_ROLE,
            "tested_scope": (
                "same three selected frozen fronts and four held-out cells as the "
                "fixed semantic-gauge primary control"
            ),
            "subclaims": {
                "primary_replay": "supported_three_of_three",
                "oracle_carrier_contract": "supported_three_of_three",
                "target_controls": "supported_three_of_three",
                "sensor_only_repair": "rejected_zero_of_three",
                "oracle_refit_linear_writer": "rejected_zero_of_three",
                "writer_or_carrier_limitation": "supported_three_of_three",
            },
            "tags": [
                "tinyllm",
                "c2",
                "oracle-intervention",
                "error-decomposition",
                "causal-patching",
                "negative-result",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The diagnostic is not a confirmatory hypothesis test; it validly "
                "classifies all three checkpoints as writer/carrier-limited."
            ),
            "confidence": 0.96,
            "confidence_assessment": "exact_post_outcome_causal_error_decomposition",
            "independent_seed_count": 3,
            "num_direct_experiments": 3,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "Primary metrics replay within the frozen 1e-6 contract.",
                "The exact latent carrier cannot rescue any checkpoint.",
                "Oracle refitting does not materially improve the writer outcome.",
                "Better sensing alone is insufficient; the quotient-to-residual write must be nonlinear, invariant-context-conditioned, or both.",
            ],
            "suggested_hypotheses": [
                "A nonlinear writer from the fixed carrier passes where the linear writer fails.",
                "Declared invariant sensor context is necessary beyond the degree-two phase carrier.",
                "Task-metric training of the neutral fusion is required for portable sidecars.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": _sha256(results_path),
            "implementation_sha256": DECOMPOSITION_IMPLEMENTATION_SHA256,
            "primary_campaign_sha256": PRIMARY_WRITER_CAMPAIGN_SHA256,
        },
    }


def _load_capacity_campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        _sha256(path) != CAPACITY_CAMPAIGN_SHA256
        or value.get("schema_version") != CAPACITY_SCHEMA
        or value.get("hypothesis_id") != CAPACITY_HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != CAPACITY_EVIDENCE_ROLE
        or value.get("implementation_sha256") != CAPACITY_IMPLEMENTATION_SHA256
        or value.get("summary", {}).get("requested") != 3
        or value.get("summary", {}).get("completed") != 3
        or value.get("summary", {}).get("failed") != 0
        or value.get("summary", {}).get("trained_models") != 0
        or value.get("aggregates", {}).get("contract_counts")
        != CAPACITY_CONTRACT_COUNTS
        or value.get("aggregates", {}).get("candidate_specific_pass_counts")
        != CAPACITY_CANDIDATE_COUNTS
        or value.get("aggregates", {}).get("classification_counts")
        != {
            "calibration_context_limited": 0,
            "high_order_curvature_limited": 0,
            "invalid": 0,
            "low_order_curvature_limited": 0,
            "unresolved_writer_limited": 3,
        }
        or value.get("provenance", {}).get("predecessor_campaign_sha256")
        != DECOMPOSITION_CAMPAIGN_SHA256
    ):
        raise ValueError(f"invalid fixed-gauge writer-capacity campaign {path}")
    return value


def _load_capacity_details(
    path: Path, campaign: Mapping[str, Any]
) -> list[dict[str, Any]]:
    details = _load_details(
        path,
        campaign,
        schema=CAPACITY_SCHEMA,
        hypothesis_id=CAPACITY_HYPOTHESIS_ID,
        evidence_role=CAPACITY_EVIDENCE_ROLE,
        implementation=CAPACITY_IMPLEMENTATION_SHA256,
    )
    for detail in details:
        gates = detail.get("gates", {})
        candidates = gates.get("candidates", {})
        if (
            detail.get("classification") != "unresolved_writer_limited"
            or gates.get("predecessor_replay_contract") is not True
            or float(gates.get("maximum_predecessor_replay_error", 1.0)) > 1e-6
            or gates.get("oracle_and_context_contract") is not True
            or gates.get("continuous_target_control_contract") is not True
            or set(candidates) != set(CAPACITY_CANDIDATE_COUNTS)
            or any(
                value.get("specific_causal_pass") is not False
                or value.get("shuffled_specificity") is not True
                for value in candidates.values()
            )
            or len(detail.get("heldout_cells", [])) != 4
        ):
            raise ValueError(
                f"invalid fixed-gauge writer-capacity result for seed {detail['seed']}"
            )
    return details


def build_fixed_gauge_writer_capacity_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-fixed-gauge-writer-capacity.md"
    ),
) -> dict[str, Any]:
    campaign = _load_capacity_campaign(results_path)
    details = _load_capacity_details(results_path, campaign)
    evidence = []
    for detail in details:
        evidence.append(
            {
                "experiment_id": detail["experiment_id"],
                "seed": detail["seed"],
                "scientific_fingerprint": detail["scientific_fingerprint"],
                "classification": detail["classification"],
                "contracts": {
                    name: detail["gates"][name]
                    for name in CAPACITY_CONTRACT_COUNTS
                },
                "candidate_gates": detail["gates"]["candidates"],
                "alignment_fit_coordinate_metrics": detail["alignment_fit"]
                ["coordinate_metrics"],
                "heldout": [
                    {
                        "cohort": cell["cohort"],
                        "regime": cell["regime"],
                        "states": {
                            arm: cell["states"][arm]["continuous"]
                            for arm in (
                                "quotient_order1",
                                "quotient_order2",
                                "quotient_order4",
                                "quotient_order40",
                                "quotient_order4_context",
                            )
                        },
                    }
                    for cell in detail["heldout_cells"]
                ],
            }
        )
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": CAPACITY_HYPOTHESIS_ID,
            "name": "TinyLLM fixed-gauge writer-capacity decomposition",
            "category": "mechanism",
            "description": (
                "Nested quotient Fourier and calibration-conditioned writers test "
                "whether curvature or observed context repairs the fixed causal write."
            ),
            "question": (
                "Can a support-independent nonlinear quotient writer, or an equal-width "
                "calibration-conditioned writer, pass every held-out causal cell?"
            ),
            "prediction": (
                "At least one specific candidate classifies each checkpoint as low-order "
                "curvature, high-order curvature, or calibration-context limited."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "diagnostic_unresolved_writer_limited_three_of_three",
            "evidence_count": 3,
            "direct_experiment_count": 3,
            "power_profile": "underpowered_preregistered_post_outcome_diagnostic",
            "evidence_role": CAPACITY_EVIDENCE_ROLE,
            "tested_scope": (
                "three selected frozen d6 C2 block-0 fronts; four locked held-out "
                "causal cells and nine fitted writers per checkpoint"
            ),
            "subclaims": {
                "diagnostic_contracts": "supported_three_of_three",
                "low_order_curvature_sufficiency": "rejected_zero_of_three",
                "high_order_curvature_sufficiency": "rejected_zero_of_three",
                "calibration_context_sufficiency": "rejected_zero_of_three",
                "shuffled_specificity": "supported_twelve_of_twelve_candidates",
                "low_order_descriptive_improvement": "supported_mean_error_reduction_without_gate_pass",
                "unresolved_writer_limitation": "supported_three_of_three",
            },
            "tags": [
                "tinyllm",
                "c2",
                "fourier-writer",
                "calibration-context",
                "causal-patching",
                "capacity-control",
                "negative-result",
                "no-training",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "All contracts and shuffled controls pass, but no quotient-only or "
                "calibration-conditioned candidate passes a complete checkpoint."
            ),
            "confidence": 0.97,
            "confidence_assessment": "exact_preregistered_underpowered_capacity_falsifier",
            "independent_seed_count": 3,
            "num_direct_experiments": 3,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "Order-2 and order-4 Fourier writers lower aggregate mean shift from 0.2112 to 0.1752 and 0.1733 bins but pass no checkpoint.",
                "Equal-width order-40 and calibration-context writers worsen held-out mean shift to 0.4462 and 0.3136 bins.",
                "All twelve candidate/checkpoint shuffled-specificity gates pass.",
                "All three checkpoints remain unresolved writer-limited under the fixed decision table.",
                "The next useful object is the continuation's local task tangent, not a higher-capacity coordinate map.",
            ],
            "suggested_hypotheses": [
                "Order-4 residual error concentrates in a small task-tangent direction under the frozen continuation Jacobian.",
                "The continuation task tangent changes across nuisance support, making the residual write state-conditioned at this cut.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAPACITY_CAMPAIGN_SHA256,
            "implementation_sha256": CAPACITY_IMPLEMENTATION_SHA256,
            "predecessor_campaign_sha256": DECOMPOSITION_CAMPAIGN_SHA256,
        },
    }


def _experiment_results(
    record: Mapping[str, Any],
    details: list[dict[str, Any]],
    *,
    hypothesis_id: str,
    metric_builder: Callable[[Mapping[str, Any]], dict[str, float]],
    primary_builder: Callable[[Mapping[str, Any]], float],
) -> list[ExperimentResult]:
    completed = datetime.fromisoformat(
        str(record["result"]["completed_at"]).replace("Z", "+00:00")
    )
    return [
        ExperimentResult(
            experiment_id=detail["experiment_id"],
            hypothesis_id=hypothesis_id,
            metrics=metric_builder(detail),
            primary_metric=primary_builder(detail),
            model_architecture=[6, 6, 384],
            model_parameters=29_956_608,
            training_time=float(detail["analysis_seconds"]),
            model_checkpoint=detail["provenance"]["checkpoint"],
            observations=[
                "Frozen d6 C2 block-0 attention quotient front; two untouched held-out cohorts under both shifts."
            ],
            anomalies=[],
            timestamp=completed,
        )
        for detail in details
    ]


def build_fixed_semantic_gauge_writer_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    campaign = _load_campaign(
        results_path,
        schema=WRITER_SCHEMA,
        hypothesis_id=WRITER_HYPOTHESIS_ID,
        evidence_role=WRITER_EVIDENCE_ROLE,
        implementation=WRITER_IMPLEMENTATION_SHA256,
        campaign_sha256=WRITER_CAMPAIGN_SHA256,
        gate_counts=WRITER_GATE_COUNTS,
    )
    details = _load_details(
        results_path,
        campaign,
        schema=WRITER_SCHEMA,
        hypothesis_id=WRITER_HYPOTHESIS_ID,
        evidence_role=WRITER_EVIDENCE_ROLE,
        implementation=WRITER_IMPLEMENTATION_SHA256,
    )
    return _experiment_results(
        record,
        details,
        hypothesis_id=WRITER_HYPOTHESIS_ID,
        metric_builder=lambda item: {
            "observation_contract": float(item["gates"]["observation_contract"]),
            "continuous_target_control_contract": float(
                item["gates"]["continuous_target_control_contract"]
            ),
            "fixed_gauge_causal_writer": float(
                item["gates"]["fixed_gauge_causal_writer"]
            ),
            "shuffled_specificity": float(item["gates"]["shuffled_specificity"]),
            "fixed_gauge_mean_shift_bins": float(
                item["gates"]["fixed_gauge_aggregate_mean_shift_bins"]
            ),
            "worst_coordinate_variance_explained": float(
                item["gates"]["worst_heldout_coordinate_variance_explained"]
            ),
        },
        primary_builder=lambda item: float(item["gates"]["fixed_gauge_causal_writer"]),
    )


def build_fixed_gauge_error_decomposition_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    campaign = _load_campaign(
        results_path,
        schema=DECOMPOSITION_SCHEMA,
        hypothesis_id=DECOMPOSITION_HYPOTHESIS_ID,
        evidence_role=DECOMPOSITION_EVIDENCE_ROLE,
        implementation=DECOMPOSITION_IMPLEMENTATION_SHA256,
        campaign_sha256=DECOMPOSITION_CAMPAIGN_SHA256,
        gate_counts=DECOMPOSITION_GATE_COUNTS,
    )
    details = _load_details(
        results_path,
        campaign,
        schema=DECOMPOSITION_SCHEMA,
        hypothesis_id=DECOMPOSITION_HYPOTHESIS_ID,
        evidence_role=DECOMPOSITION_EVIDENCE_ROLE,
        implementation=DECOMPOSITION_IMPLEMENTATION_SHA256,
    )
    return _experiment_results(
        record,
        details,
        hypothesis_id=DECOMPOSITION_HYPOTHESIS_ID,
        metric_builder=lambda item: {
            "primary_replay_contract": float(item["gates"]["primary_replay_contract"]),
            "oracle_carrier_contract": float(item["gates"]["oracle_carrier_contract"]),
            "continuous_target_control_contract": float(
                item["gates"]["continuous_target_control_contract"]
            ),
            "observed_fit_oracle_eval": float(
                item["gates"]["observed_fit_oracle_eval"]
            ),
            "oracle_fit_oracle_eval": float(item["gates"]["oracle_fit_oracle_eval"]),
            "oracle_fit_variance_explained": float(
                item["alignment_fit"]["oracle_coordinate_metrics"]["variance_explained"]
            ),
        },
        primary_builder=lambda item: float(item["gates"]["oracle_fit_oracle_eval"]),
    )


def build_fixed_gauge_writer_capacity_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    campaign = _load_capacity_campaign(results_path)
    details = _load_capacity_details(results_path, campaign)
    return _experiment_results(
        record,
        details,
        hypothesis_id=CAPACITY_HYPOTHESIS_ID,
        metric_builder=lambda item: {
            "predecessor_replay_contract": float(
                item["gates"]["predecessor_replay_contract"]
            ),
            "oracle_and_context_contract": float(
                item["gates"]["oracle_and_context_contract"]
            ),
            "continuous_target_control_contract": float(
                item["gates"]["continuous_target_control_contract"]
            ),
            **{
                f"{arm}_specific_causal_pass": float(
                    item["gates"]["candidates"][arm]["specific_causal_pass"]
                )
                for arm in CAPACITY_CANDIDATE_COUNTS
            },
            **{
                f"{arm}_aggregate_mean_shift_bins": float(
                    item["gates"]["candidates"][arm]["aggregate_mean_shift_bins"]
                )
                for arm in CAPACITY_CANDIDATE_COUNTS
            },
        },
        primary_builder=lambda item: float(
            any(
                value["specific_causal_pass"]
                for value in item["gates"]["candidates"].values()
            )
        ),
    )


def _store(
    record: dict[str, Any],
    experiments: list[ExperimentResult],
    output_path: Path,
    chromadb_path: Path | None,
) -> dict[str, Any]:
    hypothesis_id = record["hypothesis"]["id"]
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
        hypothesis_readback = logger.hypotheses_collection.get(
            ids=[hypothesis_id], include=["metadatas"]
        )
        experiment_readback = logger.experiments_collection.get(
            ids=storage["result_hashes"], include=["metadatas"]
        )
        if (
            hypothesis_readback.get("ids") != [hypothesis_id]
            or len(experiment_readback.get("ids", [])) != len(experiments)
            or {
                item.get("hypothesis_id")
                for item in experiment_readback.get("metadatas", [])
            }
            != {hypothesis_id}
        ):
            raise RuntimeError(f"{hypothesis_id} ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": hypothesis_id,
            "experiment_count": len(experiment_readback["ids"]),
        }
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return record


def store_fixed_semantic_gauge_writer_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_fixed_semantic_gauge_writer_meta_hypothesis(results_path)
    experiments = build_fixed_semantic_gauge_writer_experiment_results(
        record, results_path
    )
    return _store(record, experiments, output_path, chromadb_path)


def store_fixed_gauge_error_decomposition_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_fixed_gauge_error_decomposition_meta_hypothesis(results_path)
    experiments = build_fixed_gauge_error_decomposition_experiment_results(
        record, results_path
    )
    return _store(record, experiments, output_path, chromadb_path)


def store_fixed_gauge_writer_capacity_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_fixed_gauge_writer_capacity_meta_hypothesis(results_path)
    experiments = build_fixed_gauge_writer_capacity_experiment_results(
        record, results_path
    )
    return _store(record, experiments, output_path, chromadb_path)


__all__ = [
    "build_fixed_semantic_gauge_writer_meta_hypothesis",
    "build_fixed_semantic_gauge_writer_experiment_results",
    "store_fixed_semantic_gauge_writer_meta_hypothesis",
    "build_fixed_gauge_error_decomposition_meta_hypothesis",
    "build_fixed_gauge_error_decomposition_experiment_results",
    "store_fixed_gauge_error_decomposition_meta_hypothesis",
    "build_fixed_gauge_writer_capacity_meta_hypothesis",
    "build_fixed_gauge_writer_capacity_experiment_results",
    "store_fixed_gauge_writer_capacity_meta_hypothesis",
]
