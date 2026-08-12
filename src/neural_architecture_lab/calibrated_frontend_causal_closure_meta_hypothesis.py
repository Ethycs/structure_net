"""Build and store calibrated TinyLLM front-end causal-closure evidence."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-calibrated-frontend-causal-closure.v1"
HYPOTHESIS_ID = "tinyllm-calibrated-frontend-causal-closure-v1"
EVIDENCE_ROLE = "preregistered_outcome_directed_frozen_activation_intervention"
IMPLEMENTATION_SHA256 = (
    "5060b45674430351dabb6cd67af5e41a215f883d09b9702edd3d36b3d1d51260"
)
CAMPAIGN_SHA256 = (
    "1457fdcce224157c5d8b19c11317b3d3c47cc7d8575097c78e76ba8798431b14"
)
RESULT_MANIFEST_SHA256 = (
    "baed34a16dca206536b2e9cd221fd9f7556f4c063f85ee857352522e770844f4"
)
SOURCE_CAMPAIGN_SHA256 = (
    "80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501"
)
SOURCE_RESULT_MANIFEST_SHA256 = (
    "34bf25feb896abc9b9e06386b474fc6795c94566a23cd97a06795435fba64d68"
)
PREREGISTRATION_SHA256 = (
    "d5ebcf907d4fe99132779b94d942e6ae4f4465f10a4f01c67f806e1dbb0d802f"
)
EXPECTED_DATASET_HASHES = {
    "composition": "b025623e23f534ca3670f49f1bafc1c3979d1ca834aec2223f9c3166be8f52a6",
    "extrapolation": "f2f1e8c2c06b38fbecf856f0679e9861d9fab3e1c2a41358e55043f4d309a214",
}
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
ALL_CONDITIONS = ("raw_calibrated",) + CONDITIONS
SEEDS = (7, 17, 29, 41, 53)
CUTS = ("pre_block", "block0_post_attention", "block0_post_mlp", "full")
REGIMES = ("composition", "extrapolation")
TRANSITIONS = ("block0_attention", "block0_mlp")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, Iterable):
        return all(_finite(item) for item in value)
    return True


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    config = value.get("configuration", {})
    summary = value.get("summary", {})
    aggregates = value.get("aggregates", {})
    provenance = value.get("provenance", {})
    source_path = Path(provenance.get("source_campaign", ""))
    preregistration_path = Path(provenance.get("preregistration", ""))
    expected_summary = {
        "requested": 15,
        "scheduled": 15,
        "completed": 15,
        "failed": 0,
        "excluded": 0,
        "retries": 0,
        "reused": 0,
        "trained_models": 0,
        "trained_frontends": 0,
        "trained_task_heads": 0,
        "fitted_probes": 0,
        "fitted_observers": 0,
        "fitted_parameters": 0,
    }
    arms = aggregates.get("arms", {})
    structured_pass = all(
        arms.get(condition, {}).get("cut_pass_counts")
        == {cut: 5 for cut in CUTS}
        and arms.get(condition, {}).get("all_cuts_pass_count") == 5
        and arms.get(condition, {}).get("shuffle_pass_counts")
        == {cut: 0 for cut in CUTS}
        and arms.get(condition, {}).get("transition_closed_counts")
        == {transition: 5 for transition in TRANSITIONS}
        for condition in CONDITIONS
    )
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or value.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or tuple(config.get("conditions", ())) != ALL_CONDITIONS
        or tuple(config.get("seeds", ())) != SEEDS
        or int(config.get("required_seed_passes", -1)) != 4
        or float(config.get("accuracy_loss_ceiling", math.nan)) != 0.03
        or float(config.get("cross_entropy_increase_ceiling", math.nan)) != 0.10
        or config.get("allow_underpowered") is not False
        or summary != expected_summary
        or aggregates.get("valid") is not True
        or aggregates.get("controls_pass") is not True
        or aggregates.get("primary_hypothesis_pass") is not True
        or aggregates.get("classification") != "frontend_causal_quotient_closed"
        or not structured_pass
        or value.get("dataset_hashes") != EXPECTED_DATASET_HASHES
        or provenance.get("source_campaign_sha256") != SOURCE_CAMPAIGN_SHA256
        or provenance.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or provenance.get("source_preflight_completed_before_interventions")
        is not True
        or not source_path.is_file()
        or _sha256(source_path) != SOURCE_CAMPAIGN_SHA256
        or not preregistration_path.is_file()
        or _sha256(preregistration_path) != PREREGISTRATION_SHA256
        or not _finite(value)
    ):
        raise ValueError(f"invalid calibrated causal-closure campaign {path}")
    manifest = hashlib.sha256(
        json.dumps(
            value.get("results", []), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if manifest != RESULT_MANIFEST_SHA256:
        raise ValueError(f"invalid calibrated causal-closure manifest {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {
        (entry["condition"], int(entry["seed"])): entry
        for entry in campaign.get("results", [])
    }
    expected = {
        (condition, seed) for condition in ALL_CONDITIONS for seed in SEEDS
    }
    if set(entries) != expected:
        raise ValueError(f"invalid calibrated causal-closure result index {path}")
    output = []
    fingerprints: set[str] = set()
    for condition in ALL_CONDITIONS:
        for seed in SEEDS:
            entry = entries[(condition, seed)]
            result_path = Path(entry["path"])
            diagnostics_path = Path(entry["diagnostics_path"])
            detail = json.loads(result_path.read_text(encoding="utf-8"))
            source_path = Path(detail.get("provenance", {}).get("source_result", ""))
            source = json.loads(source_path.read_text(encoding="utf-8"))
            if (
                _sha256(result_path) != entry.get("result_sha256")
                or _sha256(diagnostics_path) != entry.get("diagnostics_sha256")
                or detail.get("schema_version") != EXPERIMENT_SCHEMA
                or detail.get("hypothesis_id") != HYPOTHESIS_ID
                or detail.get("status") != "completed"
                or detail.get("evidence_role") != EVIDENCE_ROLE
                or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
                or detail.get("condition") != condition
                or int(detail.get("seed", -1)) != seed
                or detail.get("scientific_fingerprint")
                != entry.get("scientific_fingerprint")
                or detail.get("dataset_hashes") != EXPECTED_DATASET_HASHES
                or detail.get("gates", {}).get("validity") is not True
                or any(
                    detail.get("gates", {}).get(gate) is not True
                    for gate in (
                        "finite",
                        "paired_state_identity",
                        "source_and_cut_replay",
                        "state_unchanged",
                    )
                )
                or not source_path.is_file()
                or _sha256(source_path)
                != detail.get("provenance", {}).get("source_result_sha256")
                or not _finite(detail)
            ):
                raise ValueError(f"invalid calibrated causal-closure result {result_path}")
            if condition in CONDITIONS and (
                detail.get("cut_seed_gates") != {cut: True for cut in CUTS}
                or detail.get("shuffle_seed_gates")
                != {cut: False for cut in CUTS}
                or detail.get("transition_seed_gates")
                != {transition: True for transition in TRANSITIONS}
            ):
                raise ValueError(f"structured causal gate changed {result_path}")
            fingerprint = detail["scientific_fingerprint"]
            if fingerprint in fingerprints:
                raise ValueError(f"duplicate causal-closure fingerprint {result_path}")
            fingerprints.add(fingerprint)
            detail["_model_parameters"] = int(source["model_parameters"])
            if condition in CONDITIONS:
                output.append(detail)
    return output


def _detail_summary(detail: Mapping[str, Any]) -> dict[str, float]:
    orbit_gains = []
    ce_reductions = []
    orbit_js = []
    shuffle_losses = []
    pair_ratios = []
    defects = []
    for regime in REGIMES:
        regime_result = detail["regimes"][regime]
        for cut in CUTS:
            cut_result = regime_result["cuts"][cut]
            orbit = cut_result["orbit_average"]
            shuffle = cut_result["fiber_shuffled"]
            orbit_gains.append(-float(orbit["task_sufficiency"]["accuracy_loss"]))
            ce_reductions.append(
                -float(orbit["task_sufficiency"]["cross_entropy_increase"])
            )
            orbit_js.append(float(orbit["posterior_js_from_baseline"]))
            shuffle_losses.append(
                float(shuffle["task_sufficiency"]["accuracy_loss"])
            )
            pair_ratios.append(
                float(
                    cut_result["activation_geometry"][
                        "mean_pair_rms_over_state_rms"
                    ]
                )
            )
        for transition in TRANSITIONS:
            defects.append(
                float(
                    regime_result["transitions"][transition][
                        "defect_relative_rms"
                    ]
                )
            )
    return {
        "all_cuts_both_shifts_pass": float(all(detail["cut_seed_gates"].values())),
        "pre_block_both_shifts_pass": float(detail["cut_seed_gates"]["pre_block"]),
        "all_shuffled_cuts_fail": float(
            not any(detail["shuffle_seed_gates"].values())
        ),
        "both_block0_transitions_closed": float(
            all(detail["transition_seed_gates"].values())
        ),
        "minimum_exact_accuracy_gain": min(orbit_gains),
        "maximum_exact_accuracy_gain": max(orbit_gains),
        "minimum_cross_entropy_reduction": min(ce_reductions),
        "maximum_orbit_posterior_js": max(orbit_js),
        "minimum_shuffle_accuracy_loss": min(shuffle_losses),
        "maximum_natural_pair_relative_rms": max(pair_ratios),
        "maximum_relative_reynolds_defect": max(defects),
        "analysis_seconds": float(detail["analysis_seconds"]),
    }


def build_calibrated_frontend_causal_closure_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/"
        "2026-08-10_tinyllm-calibrated-frontend-causal-closure.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    summaries = [
        {
            "experiment_id": detail["experiment_id"],
            "condition": detail["condition"],
            "seed": int(detail["seed"]),
            "scientific_fingerprint": detail["scientific_fingerprint"],
            "evidence_role": EVIDENCE_ROLE,
            **_detail_summary(detail),
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM calibrated front-end causal closure",
            "category": "mechanistic_interpretability",
            "description": (
                "Exact two-sheet activation barycenters test whether calibrated "
                "structured front ends expose a causally sufficient quotient "
                "before the first TinyLLM attention sublayer."
            ),
            "question": (
                "Can the exact task-orbit barycenter replace the natural "
                "activation while preserving the frozen task on composition "
                "and extrapolation?"
            ),
            "prediction": (
                "Both structured arms must pass every cut in at least four of "
                "five checkpoints on both shifts, while semantic-fiber shuffles fail."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": (
                "preregistered_frontend_causal_quotient_closed_five_of_five"
            ),
            "evidence_count": 10,
            "direct_experiment_count": 10,
            "fixed_checkpoint_count": 5,
            "stored_checkpoint_summary_count": 10,
            "evidence_role": EVIDENCE_ROLE,
            "power_profile": "two_arm_five_checkpoint_two_shift_frozen_intervention",
            "tested_scope": (
                "ten retained structured d8/N3 systems, exact C2 task fibers, "
                "composition and extrapolation, four residual cuts"
            ),
            "subclaims": {
                "pre_block_causal_sufficiency": "supported_five_of_five_both_arms",
                "all_cut_causal_sufficiency": "supported_five_of_five_both_arms",
                "semantic_shuffle_specificity": "supported_zero_of_eighty_cells",
                "block0_attention_task_closure": "supported_five_of_five_both_arms",
                "block0_mlp_task_closure": "supported_five_of_five_both_arms",
                "literal_residual_invariance": "not_supported_natural_pair_differences_persist",
                "raw_calibrated_comparator": "descriptive_low_baseline_only",
                "same_scope_representation_optimization_licensed": "no",
            },
            "tags": [
                "tinyllm",
                "causal-activation-patching",
                "reynolds-projection",
                "quotient",
                "equivariant-front-end",
                "frozen-checkpoint",
                "preregistered",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "Both structured arms pass the exact orbit-average patch at all "
                "four cuts in five of five checkpoints on both shifts; all "
                "structured semantic shuffles fail and no parameter was fit."
            ),
            "confidence": 0.99,
            "confidence_assessment": (
                "strong_preregistered_causal_support_with_retained_cohort_scope"
            ),
            "num_direct_experiments": 10,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "checkpoint_summaries": summaries,
            },
            "key_insights": [
                "The calibrated task quotient is causally usable before block-0 attention in every retained structured checkpoint.",
                "Orbit averaging improves all three task metrics in every structured checkpoint-shift-cut cell.",
                "Whole-barycenter semantic shuffles fail every structured cell, isolating correct orbit membership.",
                "Block-0 attention and MLP are already task-closed on the quotient; they do not synthesize it from cover variation.",
                "Natural residuals retain and amplify sheet differences, so causal quotient sufficiency is weaker and more precise than literal activation invariance.",
            ],
            "suggested_hypotheses": [
                "Architecturally calibrated quotient closure will extend to richer finite nuisance groups when an observed reference fixes the gauge.",
                "Oracle-free orbit assignment, rather than downstream quotient use, is the next deployment bottleneck.",
                "Task-irrelevant fiber complements can grow with depth while the Reynolds projection remains causally sufficient.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"checkpoint_summaries": summaries},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "source_result_manifest_sha256": SOURCE_RESULT_MANIFEST_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "dataset_hashes": EXPECTED_DATASET_HASHES,
        },
    }


def build_calibrated_frontend_causal_closure_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    campaign = _campaign(results_path)
    completed = datetime.fromisoformat(
        campaign["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        summary = _detail_summary(detail)
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=summary,
                primary_metric=float(summary["pre_block_both_shifts_pass"]),
                model_architecture=[8, int(detail["_model_parameters"])],
                model_parameters=int(detail["_model_parameters"]),
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["model_checkpoint"],
                observations=[
                    f"Condition: {detail['condition']}.",
                    "Exact C2 fiber barycenters were patched into a fully frozen continuation.",
                    "No model, front end, task head, probe, or observer was fit.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return output


def store_calibrated_frontend_causal_closure_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_calibrated_frontend_causal_closure_meta_hypothesis(
        results_path
    )
    experiments = build_calibrated_frontend_causal_closure_experiment_results(
        record, results_path
    )
    storage: dict[str, Any] = {
        "aggregate_path": str(output_path),
        "experiment_ids": [item.experiment_id for item in experiments],
    }
    if chromadb_path is not None:
        from structure_net.logging.standardized_logging import (
            LoggingConfig,
            StandardizedLogger,
        )

        root = chromadb_path.parent
        logger = StandardizedLogger(
            LoggingConfig(
                project_name="structure_net_meta_hypotheses",
                queue_dir=str(root / "experiment_queue"),
                sent_dir=str(root / "experiment_sent"),
                rejected_dir=str(root / "experiment_rejected"),
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
        hypothesis = logger.hypotheses_collection.get(
            ids=[HYPOTHESIS_ID], include=["metadatas"]
        )
        results = logger.experiments_collection.get(
            ids=storage["result_hashes"], include=["metadatas"]
        )
        if (
            hypothesis.get("ids") != [HYPOTHESIS_ID]
            or len(results.get("ids", [])) != len(experiments)
            or {
                item.get("hypothesis_id")
                for item in results.get("metadatas", [])
            }
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("calibrated causal-closure ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": len(results["ids"]),
        }
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return record


__all__ = [
    "build_calibrated_frontend_causal_closure_experiment_results",
    "build_calibrated_frontend_causal_closure_meta_hypothesis",
    "store_calibrated_frontend_causal_closure_meta_hypothesis",
]
