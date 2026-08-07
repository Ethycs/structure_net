"""Build the TinyLLM continuous-carrier/readout corrective evidence record."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-continuous-readout-calibration.v1.1"
HYPOTHESIS_ID = "tinyllm-c2-semantic-carrier-readout-separation-v1"
EVIDENCE_ROLE = "post_outcome_corrective_replication_evidence"
IMPLEMENTATION_SHA256 = (
    "6a88480cf37e0f03819a73d03685d0bb5ab7cd0c6554b84194d2fac9b9a127d6"
)
PRIMARY_SEEDS = (7, 29, 53)
GATE_NAMES = (
    "predecessor_identity",
    "decomposition_contract",
    "source_rank_selected_without_fallback",
    "selected_rank_at_most_three",
    "heldout_continuous_sufficiency",
    "exact_calibrator_endpoint",
    "selected_calibrator_endpoint",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    counts = value.get("aggregates", {}).get("gate_counts", {})
    if (
        value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or value.get("configuration", {}).get("post_outcome_corrective_replication")
        is not True
        or int(value.get("summary", {}).get("requested", -1)) != 3
        or int(value.get("summary", {}).get("completed", -1)) != 3
        or int(value.get("summary", {}).get("failed", -1)) != 0
        or int(value.get("summary", {}).get("trained_models", -1)) != 0
        or int(value.get("summary", {}).get("fitted_predictive_observers", -1))
        != 0
        or int(value.get("summary", {}).get("fitted_scalar_calibrators", -1))
        != 6
        or value.get("aggregates", {}).get("confirmed") is not True
        or value.get("aggregates", {}).get("selected_ranks")
        != {"7": 2, "29": 3, "53": 3}
        or any(int(counts.get(name, -1)) != 3 for name in GATE_NAMES)
        or value.get("aggregates", {}).get("seed29_margin_repair") is not True
    ):
        raise ValueError(f"invalid continuous-readout corrective campaign {path}")
    return value


def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if set(entries) != set(PRIMARY_SEEDS):
        raise ValueError(f"invalid continuous-readout result index {path}")
    output = []
    fingerprints: set[str] = set()
    for seed in PRIMARY_SEEDS:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        detail = json.loads(detail_path.read_text())
        checkpoint = Path(detail.get("provenance", {}).get("checkpoint", ""))
        character = Path(
            detail.get("provenance", {}).get("character_result", "")
        )
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or int(detail.get("seed", -1)) != seed
            or detail.get("configuration", {}).get(
                "post_outcome_corrective_replication"
            )
            is not True
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or _sha256(detail_path) != entry.get("result_sha256")
            or not checkpoint.is_file()
            or _sha256(checkpoint)
            != detail.get("provenance", {}).get("checkpoint_sha256")
            or not character.is_file()
            or _sha256(character)
            != detail.get("provenance", {}).get("character_result_sha256")
            or not all(bool(detail.get("gates", {}).get(name)) for name in GATE_NAMES)
            or not detail.get("all_primary_gates_passed")
        ):
            raise ValueError(f"invalid continuous-readout corrective cell {detail_path}")
        if seed == 29 and not detail["gates"].get("seed29_margin_repair"):
            raise ValueError(f"missing seed-29 margin repair {detail_path}")
        for predecessor in detail.get("provenance", {}).get(
            "predecessors", {}
        ).values():
            predecessor_campaign = Path(predecessor["campaign"])
            predecessor_result = (
                None
                if predecessor.get("result") is None
                else Path(predecessor["result"])
            )
            if (
                not predecessor_campaign.is_file()
                or _sha256(predecessor_campaign)
                != predecessor["campaign_sha256"]
                or (
                    predecessor_result is not None
                    and (
                        not predecessor_result.is_file()
                        or _sha256(predecessor_result)
                        != predecessor["result_sha256"]
                    )
                )
            ):
                raise ValueError(
                    f"invalid predecessor identity in {detail_path}"
                )
        fingerprint = detail["scientific_fingerprint"]
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate scientific fingerprint {detail_path}")
        fingerprints.add(fingerprint)
        output.append(detail)
    return output


def build_continuous_readout_calibration_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-continuous-readout-calibration.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    evidence = []
    for detail in details:
        selected = str(detail["source_selection"]["selected_rank"])
        evidence.append(
            {
                "experiment_id": detail["experiment_id"],
                "seed": int(detail["seed"]),
                "evidence_role": detail["evidence_role"],
                "checkpoint": detail["provenance"]["checkpoint"],
                "checkpoint_sha256": detail["provenance"]["checkpoint_sha256"],
                "scientific_fingerprint": detail["scientific_fingerprint"],
                "selected_rank": int(selected),
                "selected_rotation_bins": detail["source_selection"][
                    "selected_calibration"
                ]["rotation_bins"],
                "headline_metrics": detail["headline_metrics"],
                "gates": detail["gates"],
                "heldout": [
                    {
                        "cohort": cell["cohort"],
                        "regime": cell["regime"],
                        "continuous": cell["ranks"][selected]["continuous"],
                        "readouts": cell["selected_readouts"],
                    }
                    for cell in detail["heldout_cells"]
                ],
            }
        )
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM continuous semantic carrier/readout separation",
            "category": "representation",
            "description": (
                "Source-selected frozen C2 defect ranks evaluated with a continuous "
                "circle endpoint and one source-fitted scalar bin-boundary calibration."
            ),
            "question": (
                "Is the stable C2 semantic carrier at most three-dimensional once "
                "continuous geometry is separated from discrete decoder margin?"
            ),
            "prediction": (
                "Ranks at most three pass every held-out continuous cell, and one "
                "scalar calibration passes every discrete endpoint including the "
                "seed-29 held-out-B extrapolation repair."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "supported_post_outcome_corrective_replication_not_fresh_confirmation"
            ),
            "evidence_count": 3,
            "direct_experiment_count": 3,
            "power_profile": "underpowered_post_outcome_corrective_replication",
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "three selected stable d6 C2 block-0 fronts; source plus two held-out "
                "cohorts; composition and extrapolation; frozen decoder"
            ),
            "subclaims": {
                "rank_at_most_three": "supported_three_of_three",
                "heldout_continuous_sufficiency": "supported_three_of_three",
                "exact_moment_readout_control": "supported_three_of_three",
                "selected_rank_scalar_calibration": "supported_three_of_three",
                "seed29_margin_repair": "supported",
                "selected_ranks": "seed7_rank2_seed29_rank3_seed53_rank3",
                "full_preregistered_hypothesis": (
                    "supported_but_not_freshly_confirmed_due_primary_provenance"
                ),
            },
            "tags": [
                "tinyllm",
                "continuous-carrier",
                "decoder-calibration",
                "causal-patching",
                "low-rank",
                "c2",
                "post-outcome-corrective-replication",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "Fresh confirmatory replication after the valid producer was frozen.",
                "Population prevalence beyond the selected three-checkpoint cohort.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "All preregistered numerical gates pass in a provenance-valid 3/3 "
                "corrective replay, but both earlier primary roots have invalid "
                "producer provenance and the valid replay occurred after outcome inspection."
            ),
            "confidence": 0.75,
            "confidence_assessment": (
                "exact_deterministic_three_seed_corrective_reproduction_underpowered_not_fresh"
            ),
            "independent_seed_count": 3,
            "num_direct_experiments": 3,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "Source-selected continuous ranks are 2, 3, and 3.",
                "Every selected rank passes all four held-out continuous endpoints.",
                "One checkpoint-local scalar rotation passes all twelve held-out discrete cells.",
                "Seed 29 held-out-B extrapolation improves from 0.6875 to 0.7500 without an extra activation direction.",
                "The valid result supports a three-channel architecture budget but does not establish cross-seed chart alignment.",
            ],
            "suggested_hypotheses": [
                "The rank-selected C2 carriers admit a group-constrained causal alignment across checkpoints.",
                "A three-channel C2-typed sidecar with explicit scalar readout calibration reproduces the frozen causal chart.",
                "The third carrier direction encodes gain or confidence rather than an additional circular semantic coordinate.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [
            str(results_path),
            str(report_path),
            "data/experiments/tinyllm_continuous_readout_calibration/20260806_d6_preregistered/campaign_results.json",
            "data/experiments/tinyllm_continuous_readout_calibration/20260806_d6_preregistered_v2/campaign_results.json",
        ],
        "provenance": {
            "campaign_sha256": _sha256(results_path),
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "excluded_roots": [
                "20260806_d6_preregistered",
                "20260806_d6_preregistered_v2",
            ],
        },
    }


def build_continuous_readout_calibration_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        metrics = {name: float(detail["gates"][name]) for name in GATE_NAMES}
        metrics.update(
            {
                "selected_rank": float(detail["source_selection"]["selected_rank"]),
                "maximum_heldout_mean_moment_shift_bins": float(
                    detail["headline_metrics"][
                        "maximum_heldout_mean_moment_shift_bins"
                    ]
                ),
                "maximum_heldout_p95_moment_shift_bins": float(
                    detail["headline_metrics"][
                        "maximum_heldout_p95_moment_shift_bins"
                    ]
                ),
                "worst_selected_calibrated_accuracy_loss": float(
                    detail["headline_metrics"][
                        "worst_selected_calibrated_accuracy_loss"
                    ]
                ),
            }
        )
        if int(detail["seed"]) == 29:
            metrics["seed29_margin_repair"] = float(
                detail["gates"]["seed29_margin_repair"]
            )
        architecture = detail["architecture_provenance"]
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(detail["all_primary_gates_passed"]),
                model_architecture=[6, 6, 384],
                model_parameters=int(architecture["model_parameter_count"])
                + int(architecture["frontend_parameter_count"]),
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["checkpoint"],
                observations=[
                    "Frozen checkpoint; source-selected rank and scalar calibration; two untouched held-out cohorts under both shifts.",
                    "Evidence role is post-outcome corrective replication, not fresh confirmation.",
                ],
                anomalies=[
                    "The original and V2 campaign roots are excluded for producer-provenance failures despite identical scientific payloads."
                ],
                timestamp=completed,
            )
        )
    return output


def store_continuous_readout_calibration_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_continuous_readout_calibration_meta_hypothesis(results_path)
    experiments = build_continuous_readout_calibration_experiment_results(
        record, results_path
    )
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
        if (
            hypothesis_readback.get("ids") != [HYPOTHESIS_ID]
            or len(experiment_readback.get("ids", [])) != len(experiments)
            or {
                item.get("hypothesis_id")
                for item in experiment_readback.get("metadatas", [])
            }
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("continuous-readout calibration ChromaDB read-back failed")
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
    "build_continuous_readout_calibration_meta_hypothesis",
    "build_continuous_readout_calibration_experiment_results",
    "store_continuous_readout_calibration_meta_hypothesis",
]
