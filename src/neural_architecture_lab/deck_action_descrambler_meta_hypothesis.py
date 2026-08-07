"""Persist the TinyLLM deck-action descrambler evidence conservatively."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_HYPOTHESIS_ID = "tinyllm-deck-action-carrier-cover-v1"
META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-deck-action-descrambler.v1"
SEEDS = (7, 17, 29, 41, 53)
DEGREES = (1, 2, 3)


def _read_campaign(path: Path) -> dict[str, Any]:
    campaign = json.loads(path.read_text())
    if (
        campaign.get("schema_version") != EXPERIMENT_SCHEMA
        or campaign.get("hypothesis_id") != META_HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("summary") != {"requested": 15, "completed": 15, "failed": 0}
    ):
        raise ValueError("invalid or incomplete deck-action campaign")
    return campaign


def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _read_campaign(path)
    implementation = campaign["implementation_sha256"]
    result = []
    for k in DEGREES:
        for seed in SEEDS:
            detail_path = path.parent / "runs" / f"k{k}" / f"seed_{seed}" / "result.json"
            detail = json.loads(detail_path.read_text())
            if (
                detail.get("schema_version") != EXPERIMENT_SCHEMA
                or detail.get("implementation_sha256") != implementation
                or detail.get("status") != "completed"
                or int(detail.get("k", -1)) != k
                or int(detail.get("seed", -1)) != seed
                or not all(item["passed"] for item in detail["baseline_replay_integrity"].values())
            ):
                raise ValueError(f"invalid deck-action cell {detail_path}")
            result.append(detail)
    return result


def build_deck_action_descrambler_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path("docs/08 - Analysis/2026-08-06_tinyllm-deck-action-descrambler.md"),
) -> dict[str, Any]:
    campaign = _read_campaign(results_path)
    aggregate = campaign["aggregates"]
    details = _details(results_path)
    evidence = []
    for detail in details:
        k = int(detail["k"])
        evidence.append({
            "experiment_id": detail["experiment_id"],
            "k": k,
            "seed": int(detail["seed"]),
            "source_checkpoint": detail["provenance"]["checkpoint"],
            "source_checkpoint_sha256": detail["provenance"]["checkpoint_sha256"],
            "baseline_replay_passed": True,
            "full_composition_orbit_accuracy": float(detail["cuts"]["full"]["causal"]["composition"]["orbit_average"]["exact_bin_accuracy"]),
            "full_extrapolation_orbit_accuracy": float(detail["cuts"]["full"]["causal"]["extrapolation"]["orbit_average"]["exact_bin_accuracy"]),
            "full_composition_orbit_alignment": float(detail["cuts"]["full"]["causal"]["composition"]["orbit_average"]["circular_alignment"]),
            "full_extrapolation_orbit_alignment": float(detail["cuts"]["full"]["causal"]["extrapolation"]["orbit_average"]["circular_alignment"]),
            "full_deck_geometry_pass": (
                None if k == 1 else all(detail["cuts"]["full"]["gates"][regime]["deck_geometry_pass"] for regime in ("composition", "extrapolation"))
            ),
        })
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": META_HYPOTHESIS_ID,
            "name": "TinyLLM deck action and carrier-cover decomposition",
            "category": "representation",
            "description": "Orthogonal Procrustes deck actions, canonical isotypic blocks, and exact orbit-average activation patching on retained degree-k TinyLLMs.",
            "question": "Do degree-k hidden states form a clean linear Z_k representation, and is retained branch information redundant or computationally required?",
            "prediction": "At least one residual cut for k=2 and k=3 passes transport, closure, localization, and invariant-control gates in four seeds and receives a reproducible causal classification.",
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "not_confirmed_causal_quotient_front_without_linear_isotypic_split",
            "evidence_count": len(evidence),
            "direct_experiment_count": len(evidence),
            "tested_scope": "retained d6 degree-1/2/3 ladder; five seeds; nine cuts; composition and extrapolation",
            "subclaims": {
                "trivial_k1_orbit_control": "supported_all_cuts_five_of_five",
                "stable_linear_deck_action_k2": "not_supported_no_stable_residual_cut",
                "stable_linear_deck_action_k3": "not_supported_no_stable_residual_cut",
                "canonical_branch_localization": "supported_five_of_five_but_not_exclusive",
                "invariant_component_branch_control": "not_supported",
                "early_computational_cover": "supported_five_of_five_before_attention",
                "late_redundant_co_representation_k2": "supported_by_block2_and_full",
                "late_redundant_co_representation_k3": "supported_by_block2_and_full",
                "full_preregistered_hypothesis": "not_confirmed",
            },
            "tags": ["tinyllm", "deck-group", "descrambler", "procrustes", "activation-patching", "quotient-front", "multi-seed"],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": "No residual cut passed the complete linear deck/isotypic gate in four seeds for either k=2 or k=3, although exact orbit averaging identified a reproducible causal quotient front.",
            "confidence": 0.0,
            "confidence_assessment": "five_seed_failure_of_linear_gate_with_reproducible_causal_secondary_result",
            "independent_seed_count": 5,
            "num_direct_experiments": len(evidence),
            "descriptive_metrics": aggregate,
            "key_insights": [
                "Exact orbit averaging destroys k=2 and k=3 task maps before attention in all five seeds.",
                "By block 2 and full depth, orbit averaging preserves both maps while forcing patched branch accuracy to chance.",
                "The causal quotient front occurs earlier for k=2 than k=3.",
                "Schur multiplicities are consistent across seeds but the nominal invariant component remains branch-decodable.",
                "Random orbit pairing and phase-shuffled averaging destroy the mature map, so preservation is specific to exact task fibers.",
            ],
            "suggested_hypotheses": [
                "A group-constrained order-k Procrustes fit yields an idempotent spectral projector without invariant-component branch leakage.",
                "The quotient front depth grows with harmonic degree k.",
                "Attention constructs the first quotient-sufficient feature while later blocks denoise and amplify it.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path), str(report_path)],
    }


def build_deck_action_experiment_results(record: Mapping[str, Any], results_path: Path) -> list[ExperimentResult]:
    completed = datetime.fromisoformat(_read_campaign(results_path)["completed_at"].replace("Z", "+00:00"))
    output = []
    for detail in _details(results_path):
        full = detail["cuts"]["full"]["causal"]
        metrics = {
            "composition_orbit_accuracy": float(full["composition"]["orbit_average"]["exact_bin_accuracy"]),
            "extrapolation_orbit_accuracy": float(full["extrapolation"]["orbit_average"]["exact_bin_accuracy"]),
            "composition_orbit_alignment": float(full["composition"]["orbit_average"]["circular_alignment"]),
            "extrapolation_orbit_alignment": float(full["extrapolation"]["orbit_average"]["circular_alignment"]),
        }
        output.append(ExperimentResult(
            experiment_id=detail["experiment_id"],
            hypothesis_id=META_HYPOTHESIS_ID,
            metrics=metrics,
            primary_metric=min(metrics["composition_orbit_alignment"], metrics["extrapolation_orbit_alignment"]),
            model_architecture=[6, int(detail["k"])],
            model_parameters=29_956_224,
            training_time=float(detail["analysis_seconds"]),
            model_checkpoint=detail["provenance"]["checkpoint"],
            observations=["Frozen source checkpoint; exact nuisance-matched deck orbits; zero cross-cut replay spread."],
            anomalies=["The approximate group-average invariant component remains branch-decodable."],
            timestamp=completed,
        ))
    return output


def store_deck_action_descrambler_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_deck_action_descrambler_meta_hypothesis(results_path)
    experiments = build_deck_action_experiment_results(record, results_path)
    storage: dict[str, Any] = {
        "aggregate_path": str(output_path),
        "experiment_ids": [item.experiment_id for item in experiments],
    }
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
    "build_deck_action_descrambler_meta_hypothesis",
    "build_deck_action_experiment_results",
    "store_deck_action_descrambler_meta_hypothesis",
]
