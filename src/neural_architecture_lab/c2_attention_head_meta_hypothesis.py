"""Persist the TinyLLM C2 attention-head decomposition evidence."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c2-attention-head-decomposition.v1"
HYPOTHESIS_ID = "tinyllm-c2-sparse-attention-head-synthesis-v1"
SEEDS = (7, 17, 29, 41, 53)
PRIMARY_SEEDS = (7, 29, 53)


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    summary = value.get("summary", {})
    if (
        value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != "preregistered_primary_evidence"
        or summary.get("requested") != 5
        or summary.get("completed") != 5
        or summary.get("failed") != 0
        or summary.get("trained_models") != 0
        or not value.get("implementation_sha256")
    ):
        raise ValueError(f"invalid C2 attention-head campaign {path}")
    return value


def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    details = []
    fingerprints = set()
    for seed in SEEDS:
        detail_path = path.parent / "runs" / f"seed_{seed}" / "result.json"
        detail = json.loads(detail_path.read_text())
        checkpoint = Path(detail.get("provenance", {}).get("checkpoint", ""))
        comparator = Path(detail.get("provenance", {}).get("character_result", ""))
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != "preregistered_primary_evidence"
            or int(detail.get("seed", -1)) != seed
            or detail.get("implementation_sha256") != campaign["implementation_sha256"]
            or not detail.get("scientific_fingerprint")
            or not checkpoint.is_file()
            or not comparator.is_file()
        ):
            raise ValueError(f"invalid C2 attention-head cell {detail_path}")
        if detail["scientific_fingerprint"] in fingerprints:
            raise ValueError(f"duplicate scientific fingerprint {detail_path}")
        fingerprints.add(detail["scientific_fingerprint"])
        details.append(detail)
    return details


def build_c2_attention_head_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-c2-attention-head-decomposition.md"
    ),
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
            "scientific_fingerprint": detail["scientific_fingerprint"],
            "selected_heads": detail["source_selection"]["selected_heads"],
            "selected_cardinality": detail["source_selection"]["cardinality"],
            "heldout": detail["heldout"],
            "gates": detail["gates"],
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM sparse C2 attention-head synthesis",
            "category": "representation",
            "description": (
                "Exact projected-head decomposition and exhaustive 64-subset "
                "causal patching at frozen degree-two attention synthesis fronts."
            ),
            "question": (
                "Do stable early degree-two quotient fronts use a fixed sufficient "
                "and dominant subset of at most two attention heads?"
            ),
            "prediction": (
                "Each of seeds 7,29,53 selects at most two heads that remains "
                "sufficient, dominant, and specific across two held-out cohorts and both shifts."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_exact_decomposition_shows_distributed_redundant_multihead_synthesis"
            ),
            "evidence_count": 5,
            "direct_experiment_count": 5,
            "tested_scope": (
                "five retained d6 degree-two checkpoints; three primary stable early "
                "fronts; three orbit cohorts; both shifts; all 64 head subsets"
            ),
            "subclaims": {
                "exact_projected_head_decomposition": "supported_three_of_three_primary",
                "heldout_aggregate_endpoint_replication": "supported_three_of_three_primary",
                "one_or_two_head_source_selection": "not_supported_zero_of_three_primary",
                "fixed_subset_heldout_sufficiency": "not_supported_two_of_three_primary",
                "selected_subset_complement_dominance": "not_supported_one_of_three_primary",
                "same_cardinality_subset_specificity": "not_supported_one_of_three_primary",
                "distributed_multihead_synthesis": "supported_three_of_three_primary",
                "full_hypothesis": "not_confirmed",
            },
            "tags": [
                "tinyllm",
                "c2",
                "attention-head",
                "reynolds-defect",
                "causal-intervention",
                "exhaustive-subsets",
                "distributed-circuit",
                "symmetry",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The exact head sum and held-out aggregate fronts replicated, but the "
                "three primary source selections required four, four, and five heads; "
                "held-out sufficiency, dominance, and specificity also missed their gates."
            ),
            "confidence": 0.0,
            "confidence_assessment": (
                "complete_three_primary_seed_falsification_with_exact_algebraic_control"
            ),
            "independent_seed_count": 5,
            "primary_seed_count": 3,
            "num_direct_experiments": 5,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "Projected attention-head defects reconstruct the full Reynolds defect to below 5.55e-7 relative error.",
                "Stable early fronts require fixed source subsets of four, four, and five of six heads.",
                "Two four-head selections transfer smoothly and causally across held-out cohorts; the five-head selection misses composition task accuracy despite high Fisher preservation.",
                "Same-cardinality alternatives are nearly as effective in two seeds, rejecting a privileged subset there.",
                "Residual defect norm and individual-head Shapley importance do not identify the selected circuit reliably.",
            ],
            "suggested_hypotheses": [
                "The distributed C2 attention defect is generated primarily by attention-pattern/value interaction rather than value transport alone.",
                "An exact C2-equivariant replacement must preserve charged carrier types across all attention heads before neutral cross-head fusion.",
                "Per-example posterior or proper-scoring endpoints yield more stable circuit sufficiency decisions than exact-bin accuracy alone.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path), str(report_path)],
    }


def build_c2_attention_head_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        selection = detail["source_selection"]
        heldout = detail["heldout"]
        cardinality = selection["cardinality"]
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics={
                    "selected_head_count": -1.0 if cardinality is None else float(cardinality),
                    "exact_decomposition": float(detail["gates"]["exact_decomposition"]),
                    "heldout_endpoint_replication": float(
                        detail["gates"]["heldout_endpoint_replication"]
                    ),
                    "fixed_subset_sufficiency": float(
                        detail["gates"]["fixed_subset_sufficiency"]
                    ),
                    "complement_dominance": float(
                        detail["gates"]["complement_dominance"]
                    ),
                    "subset_specificity": float(detail["gates"]["subset_specificity"]),
                    "selected_worst_fisher_preservation": (
                        -1.0
                        if heldout["selected_worst_preservation"] is None
                        else float(heldout["selected_worst_preservation"])
                    ),
                },
                primary_metric=float(detail["all_gates_passed"]),
                model_architecture=[6, 6, 384],
                model_parameters=29_956_224,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["checkpoint"],
                observations=[
                    "Frozen checkpoint; exact six-head decomposition; all 64 subsets; two held-out cohorts."
                ],
                anomalies=[
                    "High smooth Fisher preservation can coexist with a failed exact-bin causal task gate."
                ],
                timestamp=completed,
            )
        )
    return output


def store_c2_attention_head_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c2_attention_head_meta_hypothesis(results_path)
    experiments = build_c2_attention_head_experiment_results(record, results_path)
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
            ids=[HYPOTHESIS_ID], include=["metadatas"]
        )
        experiment_readback = logger.experiments_collection.get(
            ids=storage["result_hashes"], include=["metadatas"]
        )
        experiment_hypotheses = {
            metadata.get("hypothesis_id")
            for metadata in experiment_readback.get("metadatas", [])
        }
        if (
            hypothesis_readback.get("ids") != [HYPOTHESIS_ID]
            or len(experiment_readback.get("ids", [])) != len(experiments)
            or experiment_hypotheses != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("C2 attention-head ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": len(experiment_readback["ids"]),
        }
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return record


__all__ = [
    "build_c2_attention_head_meta_hypothesis",
    "build_c2_attention_head_experiment_results",
    "store_c2_attention_head_meta_hypothesis",
]
