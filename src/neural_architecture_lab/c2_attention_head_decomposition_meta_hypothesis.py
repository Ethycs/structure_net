"""Build the TinyLLM held-out degree-two attention-head evidence record."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c2-attention-head-decomposition.v1"
HYPOTHESIS_ID = "tinyllm-c2-sparse-attention-head-synthesis-v1"


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or int(value.get("summary", {}).get("completed", -1)) != 5
        or int(value.get("summary", {}).get("failed", -1)) != 0
    ):
        raise ValueError(f"invalid C2 attention-head campaign {path}")
    return value


def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    output = []
    for seed in (7, 17, 29, 41, 53):
        detail_path = path.parent / "runs" / f"seed_{seed}" / "result.json"
        detail = json.loads(detail_path.read_text())
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("status") != "completed"
            or detail.get("implementation_sha256") != campaign["implementation_sha256"]
            or int(detail.get("seed", -1)) != seed
        ):
            raise ValueError(f"invalid C2 attention-head cell {detail_path}")
        output.append(detail)
    return output


def build_c2_attention_head_decomposition_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path("docs/08 - Analysis/2026-08-06_tinyllm-c2-attention-head-decomposition.md"),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    evidence = [
        {
            "experiment_id": detail["experiment_id"],
            "seed": int(detail["seed"]),
            "primary_seed": bool(detail["primary_seed"]),
            "checkpoint": detail["provenance"]["checkpoint"],
            "checkpoint_sha256": detail["provenance"]["checkpoint_sha256"],
            "source_selection": detail["source_selection"],
            "heldout": detail["heldout"],
            "gates": detail["gates"],
            "maximum_decomposition_relative_error": detail["maximum_decomposition_relative_error"],
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "Sparse held-out C2 attention-head synthesis",
            "category": "representation",
            "description": "Exhaustive six-head subset selection on a frozen source cohort followed by unchanged evaluation on two disjoint held-out exact-orbit cohorts.",
            "question": "Do cross-cohort-stable degree-two fronts synthesize the quotient through a fixed dominant subset of at most two attention heads?",
            "prediction": "All exact, sparse-selection, held-out, fixed-subset, complement-dominance, and specificity gates pass in the three frozen stable early-front seeds.",
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "not_confirmed_four_to_five_head_distributed_synthesis",
            "evidence_count": 5,
            "direct_experiment_count": 5,
            "tested_scope": "retained d6 degree-two ladder; three stable block-zero primary seeds plus two later-front contrasts; source plus two held-out cohorts; both shifts",
            "subclaims": {
                "exact_decomposition": "supported_three_of_three_primary",
                "sparse_source_selection": "not_supported_zero_of_three_primary",
                "heldout_endpoint_replication": "supported_three_of_three_primary",
                "fixed_subset_sufficiency": "not_supported_two_of_three_primary",
                "complement_dominance": "not_supported_one_of_three_primary",
                "subset_specificity": "not_supported_one_of_three_primary",
                "full_preregistered_hypothesis": "not_confirmed",
            },
            "tags": [
                "tinyllm", "attention-head", "exhaustive-subsets", "held-out-causal-test",
                "distributed-circuit", "redundancy", "shapley", "equivariance",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": "The three stable primary checkpoints require source-selected subsets of four, four, and five heads; only one seed passes complement dominance and specificity.",
            "confidence": 0.0,
            "confidence_assessment": "exhaustive_subset_rejection_on_three_preregistered_stable_seeds_with_two_heldout_cohorts",
            "independent_seed_count": 3,
            "descriptive_seed_count": 2,
            "num_direct_experiments": 5,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "Every stable early-front endpoint replicates across both held-out cohorts and shifts.",
                "Stable quotient synthesis requires four or five of six output-projected head defects.",
                "Selected subsets transfer smoothly in seeds 7 and 29, but same-size alternatives or complements remain competitive.",
                "Shapley task credit is distributed across all heads and does not reduce to residual defect norm.",
                "The symmetry mechanism should be implemented across multi-head attention rather than attached to one chosen head.",
            ],
            "suggested_hypotheses": [
                "A task-weighted low-rank subspace of joint head defects transfers across cohorts despite the failure of sparse subsets.",
                "A cross-head invariant fusion pool preserves the quotient with fewer dimensions than six independent full-rank defects.",
                "Group-equivariant multi-head attention outperforms a one-head equivariant adapter under calibrated extrapolation.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path), str(report_path)],
    }


def build_c2_attention_head_decomposition_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    completed = datetime.fromisoformat(_campaign(results_path)["completed_at"].replace("Z", "+00:00"))
    return [
        ExperimentResult(
            experiment_id=detail["experiment_id"],
            hypothesis_id=HYPOTHESIS_ID,
            metrics={gate: float(value) for gate, value in detail["gates"].items()},
            primary_metric=float(detail["all_gates_passed"]),
            model_architecture=[6, 2],
            model_parameters=29_956_224,
            training_time=float(detail["analysis_seconds"]),
            model_checkpoint=detail["provenance"]["checkpoint"],
            observations=["Frozen checkpoint; exhaustive 64-subset source selection; two unchanged held-out exact-orbit cohorts."],
            anomalies=["High Fisher preservation can coexist with failure of the frozen exact-bin causal gate."],
            timestamp=completed,
        )
        for detail in _details(results_path)
    ]


def store_c2_attention_head_decomposition_meta_hypothesis(
    results_path: Path, output_path: Path, *, chromadb_path: Path | None = None
) -> dict[str, Any]:
    record = build_c2_attention_head_decomposition_meta_hypothesis(results_path)
    experiments = build_c2_attention_head_decomposition_experiment_results(record, results_path)
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
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return record


__all__ = [
    "build_c2_attention_head_decomposition_meta_hypothesis",
    "build_c2_attention_head_decomposition_experiment_results",
    "store_c2_attention_head_decomposition_meta_hypothesis",
]
