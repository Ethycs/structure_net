"""Build the TinyLLM attention-head Reynolds-defect evidence record."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-attention-head-defect.v1"
HYPOTHESIS_ID = "tinyllm-attention-head-defect-sparsity-v1"


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or int(value.get("summary", {}).get("completed", -1)) != 5
        or int(value.get("summary", {}).get("failed", -1)) != 0
    ):
        raise ValueError(f"invalid attention-head defect campaign {path}")
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
            raise ValueError(f"invalid attention-head defect cell {detail_path}")
        output.append(detail)
    return output


def build_attention_head_defect_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path("docs/08 - Analysis/2026-08-06_tinyllm-attention-head-defect.md"),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    evidence = [
        {
            "experiment_id": detail["experiment_id"],
            "seed": int(detail["seed"]),
            "checkpoint": detail["provenance"]["checkpoint"],
            "checkpoint_sha256": detail["provenance"]["checkpoint_sha256"],
            "gates": detail["gates"],
            "regimes": {
                regime: {
                    "target_cut": detail["regimes"][regime]["target_cut"],
                    "best_pair": detail["regimes"][regime]["best_pair"],
                    "necessary_heads": detail["regimes"][regime]["necessary_heads"],
                    "sufficient_sparse_subsets": detail["regimes"][regime]["sufficient_sparse_subsets"],
                }
                for regime in ("composition", "extrapolation")
            },
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "Sparse attention-head Reynolds-defect synthesis",
            "category": "representation",
            "description": "Exact additive decomposition of degree-two attention Reynolds defects into six output-projected head components with singleton, pair, and leave-one-out causal patches.",
            "question": "Does one stable one- or two-head circuit generate most of the task-effective quotient defect?",
            "prediction": "Exact endpoints, sparse sufficiency, shift-stable sparse subset, and individual necessity pass in at least four of five seeds.",
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "not_confirmed_distributed_multihead_reynolds_synthesis",
            "evidence_count": 5,
            "direct_experiment_count": 5,
            "tested_scope": "retained d6 degree-two ladder; five seeds; frozen synthesis attention transitions; same 64-orbit cohort; both shifts",
            "subclaims": {
                "exact_additive_contract_and_endpoint": "supported_five_of_five",
                "singleton_or_pair_sufficiency": "not_supported_zero_of_five",
                "shift_stable_sparse_circuit": "not_supported_zero_of_five",
                "common_individual_necessity": "not_supported_three_of_five",
                "full_preregistered_hypothesis": "not_confirmed",
            },
            "tags": [
                "tinyllm", "attention-head", "reynolds-defect", "causal-patching",
                "distributed-circuit", "sparse-circuit-falsification", "multi-head-attention",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": "No singleton or head pair meets the joint causal and 0.90 Fisher endpoint in any seed; common leave-one-out necessity reaches only three of five.",
            "confidence": 0.0,
            "confidence_assessment": "complete_five_seed_rejection_with_exact_additive_contract",
            "independent_seed_count": 5,
            "num_direct_experiments": 5,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "All-head attention and Reynolds-defect reconstruction is exact to below the preregistered tolerance.",
                "No one- or two-head subset is causally and Fisher sufficient under both shifts in any checkpoint.",
                "Several heads are individually necessary in stable early-front seeds, but no cross-seed privileged index exists.",
                "High Fisher proximity of a pair can coexist with failure of the map-level causal task gate.",
                "The quotient defect is distributed across attention heads rather than localized to one adapter-sized circuit.",
            ],
            "suggested_hypotheses": [
                "A low-rank joint head-defect subspace transfers across held-out cohorts even though no sparse head subset does.",
                "Group-equivariant typing across all heads preserves quotient synthesis better than modifying a selected head.",
                "Task-weighted head interactions, rather than individual defect norms, predict leave-one-out necessity.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path), str(report_path)],
    }


def build_attention_head_defect_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    completed = datetime.fromisoformat(_campaign(results_path)["completed_at"].replace("Z", "+00:00"))
    return [
        ExperimentResult(
            experiment_id=detail["experiment_id"],
            hypothesis_id=HYPOTHESIS_ID,
            metrics={
                gate: float(value)
                for gate, value in detail["gates"].items()
                if isinstance(value, bool)
            },
            primary_metric=float(all(
                bool(detail["gates"][gate])
                for gate in (
                    "exact_contract_and_endpoint",
                    "sparse_sufficiency",
                    "shift_stable_sparse_circuit",
                    "individual_necessity",
                )
            )),
            model_architecture=[6, 2],
            model_parameters=29_956_224,
            training_time=float(detail["analysis_seconds"]),
            model_checkpoint=detail["provenance"]["checkpoint"],
            observations=["Frozen checkpoint; exact six-head additive defect decomposition; 29 causal subsets per shift."],
            anomalies=["Pairs with high Fisher effect can still fail the frozen causal task conjunction."],
            timestamp=completed,
        )
        for detail in _details(results_path)
    ]


def store_attention_head_defect_meta_hypothesis(
    results_path: Path, output_path: Path, *, chromadb_path: Path | None = None
) -> dict[str, Any]:
    record = build_attention_head_defect_meta_hypothesis(results_path)
    experiments = build_attention_head_defect_experiment_results(record, results_path)
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
    "build_attention_head_defect_meta_hypothesis",
    "build_attention_head_defect_experiment_results",
    "store_attention_head_defect_meta_hypothesis",
]
