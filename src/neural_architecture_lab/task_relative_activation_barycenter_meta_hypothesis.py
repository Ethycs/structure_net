"""Build and store the TinyLLM task-relative activation barycenter result."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-task-relative-activation-barycenter.v1"
HYPOTHESIS_ID = "tinyllm-task-relative-activation-barycenter-v1"
EVIDENCE_ROLE = "preregistered_underpowered_frozen_task_activation_intervention"
CAMPAIGN_SHA256 = (
    "b5d6b613683326024eeb00944a3d0aba4dd7a251dd22f5fa7e8561b5e9d6aae4"
)
IMPLEMENTATION_SHA256 = (
    "ebdc1e9f376a0e6a274f34a172949e1d12f2e6f6674c71eafb24d9773145dc3c"
)
RUNNER_SHA256 = (
    "67e0854162cfd5ef4ff7dbc18f2cd2b7e9c6e88e10543e30bf0baf6b87b1c8ee"
)
RESULT_MANIFEST_SHA256 = (
    "0403600e910911415b76bdf61b139f9fa6ac403158fc8f0805015e7693008990"
)
PREREGISTRATION_SHA256 = (
    "15eeea68dde87d101f587ea258c9d928b55b9d9a55a9d897ccd027446203926e"
)
SOURCE_TASK_SHA256 = (
    "129a7a47757d2ecdde7138ca7729c0052f4c4653d5afc38106949f477e87652d"
)
SOURCE_ATLAS_SHA256 = (
    "da6dad8c00738e5fc364de45b970245fa0667731aeedbcb462b85aa94bf1c3d6"
)
PRESETS = ("d6", "d8")
REGIMES = ("training_support", "outside_range")
RESULT_SHA256 = {
    "d6": "8eeb07a703912dc1f090f8337097baaa84856984c2c447c394773192999f9e73",
    "d8": "79119a4fa6df16a2fe2d65c34ec1aca30072a0b53d3d8ea312c92b29f86d1476",
}
DIAGNOSTICS_SHA256 = {
    "d6": "45ec9f8dba69ad852e5eb39bdcbd08e28fd79a013ef854384953ea11805cff45",
    "d8": "ef86c31d418c04b7ffd4bec3b66135057c4ca825b3e4da8263a920cbce0de1dd",
}
MODEL_PARAMETERS = {"d6": 29_956_224, "d8": 50_964_992}
MODEL_ARCHITECTURE = {"d6": [6, 384], "d8": [8, 512]}


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


def _campaign(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    config = campaign.get("configuration", {})
    aggregate = campaign.get("aggregates", {})
    summary = campaign.get("summary", {})
    entries = {item.get("preset"): item for item in campaign.get("results", [])}
    expected_summary = {
        "requested": 2,
        "scheduled": 2,
        "completed": 2,
        "failed": 0,
        "excluded": 0,
        "retries": 0,
        "reused": 0,
        "trained_models": 0,
        "trained_probes": 0,
        "fitted_alignment_maps": 0,
        "fitted_decoders": 0,
    }
    preregistration = Path(
        "docs/07 - Status Reports/"
        "2026-08-10_tinyllm-task-relative-activation-barycenter-preregistration.md"
    )
    runner = Path(
        "experiments/structure_net/"
        "tinyllm_task_relative_activation_barycenter.py"
    )
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or campaign.get("schema_version") != EXPERIMENT_SCHEMA
        or campaign.get("hypothesis_id") != HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != EVIDENCE_ROLE
        or campaign.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or campaign.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or tuple(config.get("presets", ())) != PRESETS
        or tuple(config.get("regimes", ())) != REGIMES
        or int(config.get("fiber_pairs", -1)) != 512
        or float(config.get("baseline_accuracy_floor", -1)) != 0.15
        or float(config.get("posterior_js_ceiling", -1)) != 0.02
        or config.get("allow_underpowered") is not False
        or summary != expected_summary
        or aggregate.get("classification")
        != "observational_task_geometry_not_causally_sufficient"
        or aggregate.get("primary_hypothesis_pass") is not False
        or aggregate.get("valid") is not False
        or aggregate.get("mature_causal_fronts") != {"d6": None, "d8": None}
        or set(entries) != set(PRESETS)
        or campaign.get("source_hashes", {}).get("source_task_result")
        != SOURCE_TASK_SHA256
        or campaign.get("source_hashes", {}).get("source_atlas_result")
        != SOURCE_ATLAS_SHA256
        or campaign.get("source_digests", {}).get("runner") != RUNNER_SHA256
        or campaign.get("source_digests", {}).get("preregistration")
        != PREREGISTRATION_SHA256
        or not preregistration.is_file()
        or _sha256(preregistration) != PREREGISTRATION_SHA256
        or not runner.is_file()
        or _sha256(runner) != RUNNER_SHA256
        or not _finite(campaign)
    ):
        raise ValueError(f"invalid task-relative barycenter campaign {path}")
    manifest = hashlib.sha256(
        json.dumps(
            campaign.get("results", []),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if manifest != RESULT_MANIFEST_SHA256:
        raise ValueError(f"invalid task-relative barycenter manifest {path}")

    details: dict[str, dict[str, Any]] = {}
    for preset, entry in entries.items():
        result_path = Path(entry["path"])
        diagnostics_path = Path(entry["diagnostics_path"])
        detail = json.loads(result_path.read_text(encoding="utf-8"))
        gates = detail.get("gates", {})
        expected_valid = preset == "d8"
        if (
            _sha256(result_path) != RESULT_SHA256[preset]
            or entry.get("result_sha256") != RESULT_SHA256[preset]
            or _sha256(diagnostics_path) != DIAGNOSTICS_SHA256[preset]
            or entry.get("diagnostics_sha256") != DIAGNOSTICS_SHA256[preset]
            or detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or detail.get("preset") != preset
            or int(detail.get("seed", -1)) != 7
            or detail.get("summary", {}).get("mature_causal_front") is not None
            or detail.get("summary", {}).get("first_isolated_passing_cut")
            is not None
            or float(detail.get("summary", {}).get("maximum_replay_error", -1))
            != 0.0
            or gates.get("validity") is not expected_valid
            or gates.get("baseline_validity") is not expected_valid
            or gates.get("replay") is not True
            or gates.get("model_state_unchanged") is not True
            or gates.get("mature_causal_front_exists") is not False
            or gates.get("full_preset_gate") is not False
            or not all(item.get("pass") is True for item in detail.get("cohort_contracts", {}).values())
            or not all(item.get("unchanged") is True for item in detail.get("state_records", {}).values())
            or not _finite(detail)
        ):
            raise ValueError(f"invalid task-relative barycenter result {result_path}")
        cosine = detail["tasks"]["cosine_interval"]
        if any(
            cut["correct_barycenter"]["gates"]["task_sufficient"] is not False
            for regime in cosine["regimes"].values()
            for cut in regime["cuts"].values()
        ):
            raise ValueError(f"unexpected sufficient cut in {result_path}")
        details[preset] = detail
    d6_outside = details["d6"]["tasks"]["cosine_interval"]["regimes"][
        "outside_range"
    ]["baseline"]["exact_bin_accuracy"]
    d8_outside = details["d8"]["tasks"]["cosine_interval"]["regimes"][
        "outside_range"
    ]["baseline"]["exact_bin_accuracy"]
    if d6_outside != 0.1328125 or d8_outside != 0.15234375:
        raise ValueError("task-relative barycenter validity split changed")
    return campaign, details


def _summary(detail: Mapping[str, Any]) -> dict[str, float]:
    preset = str(detail["preset"])
    final = detail["cuts"][-1]
    cosine = detail["tasks"]["cosine_interval"]["regimes"]
    phase = detail["tasks"]["phase_circle"]["regimes"]
    reference = str(detail["reference_observational_front"])
    output: dict[str, float] = {
        "validity": float(detail["gates"]["validity"]),
        "baseline_validity": float(detail["gates"]["baseline_validity"]),
        "mature_causal_front_exists": 0.0,
        "maximum_replay_error": float(detail["summary"]["maximum_replay_error"]),
        "minimum_baseline_accuracy": min(
            float(item["baseline"]["exact_bin_accuracy"])
            for task in detail["tasks"].values()
            for item in task["regimes"].values()
        ),
    }
    for regime in REGIMES:
        reference_result = cosine[regime]["cuts"][reference]["correct_barycenter"]
        final_result = cosine[regime]["cuts"][final]["correct_barycenter"]
        phase_final = phase[regime]["cuts"][final]["correct_barycenter"]
        semantic_final = cosine[regime]["cuts"][final]["semantic_reassignment"]
        prefix = regime
        output[f"{prefix}_reference_front_posterior_js"] = float(
            reference_result["gates"]["mean_posterior_js"]
        )
        output[f"{prefix}_final_posterior_js"] = float(
            final_result["gates"]["mean_posterior_js"]
        )
        output[f"{prefix}_final_accuracy_gain"] = float(
            -final_result["gates"]["accuracy_loss"]
        )
        output[f"{prefix}_final_cross_entropy_improvement"] = float(
            -final_result["gates"]["cross_entropy_increase"]
        )
        output[f"{prefix}_phase_final_task_sufficient"] = float(
            phase_final["gates"]["task_sufficient"]
        )
        output[f"{prefix}_semantic_final_task_sufficient"] = float(
            semantic_final["gates"]["task_sufficient"]
        )
    output["model_depth"] = float(int(preset[1:]))
    return output


def build_task_relative_activation_barycenter_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/"
        "2026-08-10_tinyllm-task-relative-activation-barycenter.md"
    ),
) -> dict[str, Any]:
    campaign, details = _campaign(results_path)
    summaries = [
        {
            "experiment_id": detail["experiment_id"],
            "preset": preset,
            "seed": 7,
            "evidence_role": (
                EVIDENCE_ROLE if preset == "d8" else "validity_only_quarantined"
            ),
            **_summary(detail),
        }
        for preset, detail in details.items()
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM task-relative activation barycenter causality",
            "category": "mechanistic_interpretability",
            "description": (
                "Exact opposite-phase activation barycenters test whether the "
                "observed cosine quotient is a sufficient state for each frozen "
                "task continuation."
            ),
            "question": (
                "Does supervised task geometry identify the layer where an exact "
                "task-fiber activation barycenter becomes causally sufficient?"
            ),
            "prediction": (
                "Both d6 and d8 cosine models acquire mature causal fronts within "
                "one sublayer of their observational quotient fronts, while phase "
                "and semantic controls fail."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_d8_valid_no_causal_front_d6_baseline_invalid"
            ),
            "evidence_count": 2,
            "direct_quality_experiment_count": 1,
            "validity_only_experiment_count": 1,
            "fixed_checkpoint_count": 4,
            "evidence_role": EVIDENCE_ROLE,
            "power_profile": "one_seed_two_depth_two_task_two_shift_causal_bridge",
            "tested_scope": (
                "four retained seed-7 raw TinyLLMs, exact shared-nuisance phase "
                "fibers, full-sequence sublayer barycenters, two regimes"
            ),
            "subclaims": {
                "d8_mature_causal_barycenter_front": "not_supported_no_passing_cut",
                "d6_mature_causal_barycenter_front": "not_interpretable_baseline_invalid",
                "observational_complete_quotient_implies_causal_sufficiency": (
                    "no_in_valid_d8_scope"
                ),
                "branch_probe_collapse_implies_barycenter_closure": (
                    "no_in_valid_d8_scope"
                ),
                "query_only_barycenter_mature_front": (
                    "impossible_by_exact_final_cut_equivalence"
                ),
                "context_only_final_cut_effect": "exactly_none_by_architecture",
                "correct_barycenter_can_improve_task_without_preserving_computation": "yes",
                "same_scope_retraining_licensed": "no",
            },
            "tags": [
                "tinyllm",
                "task-activations",
                "causal-patching",
                "fiber-barycenter",
                "phase-circle",
                "cosine-interval",
                "observational-causal-gap",
                "frozen-checkpoint",
                "preregistered-null",
                "underpowered",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The valid d8 model has no isolated or mature cut whose exact "
                "cosine-fiber barycenter preserves both frozen computations; d6 "
                "misses the locked outside-range baseline floor and is quarantined."
            ),
            "confidence": 0.95,
            "confidence_assessment": (
                "strong_mechanistic_null_for_retained_d8_checkpoint_not_population_prevalence"
            ),
            "num_direct_experiments": 1,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "checkpoint_summaries": summaries,
            },
            "key_insights": [
                "The valid d8 checkpoint has no causal barycenter front despite its previously measured interval-like activation geometry.",
                "At the d8 observational front, posterior JS is 0.0276 in support and 0.2091 outside range, above the locked 0.02 ceiling.",
                "At full depth the d8 barycenter improves exact accuracy and cross-entropy while still changing the frozen posterior materially.",
                "Task-aligned activation geometry and useful projection are therefore weaker than autonomous quotient closure.",
                "At final post-MLP, full and query-only barycenters are output-identical because the continuation reads only the final query token; the failed final endpoint deductively excludes a mature query-only rescue.",
                "The d6 outside-range baseline reaches only 0.1328 accuracy against the locked 0.15 validity floor, so its candidate endpoints are not promoted.",
            ],
            "suggested_hypotheses": [
                "A task-relative subspace can be useful under projection without the complete residual barycenter being a sufficient frozen state.",
                "Restricting the existing barycenter to the query-token locus cannot create a mature causal front because it is exactly equivalent to the failed full barycenter at final post-MLP.",
                "A future within-query carrier/kernel patch must be specified from the frozen continuation and include a complementary-subspace control before any training."
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"checkpoint_summaries": summaries},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "source_task_result_sha256": SOURCE_TASK_SHA256,
            "source_atlas_result_sha256": SOURCE_ATLAS_SHA256,
            "result_sha256": RESULT_SHA256,
            "diagnostics_sha256": DIAGNOSTICS_SHA256,
            "deductive_corollaries": {
                "final_post_mlp_full_equals_query_only": True,
                "final_post_mlp_context_only_is_inert": True,
                "query_only_mature_front_possible": False,
            },
        },
    }


def build_task_relative_activation_barycenter_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    campaign, details = _campaign(results_path)
    completed = datetime.fromisoformat(
        campaign["completed_at"].replace("Z", "+00:00")
    )
    return [
        ExperimentResult(
            experiment_id=detail["experiment_id"],
            hypothesis_id=HYPOTHESIS_ID,
            metrics=_summary(detail),
            primary_metric=0.0,
            model_architecture=MODEL_ARCHITECTURE[preset],
            model_parameters=MODEL_PARAMETERS[preset],
            training_time=float(detail["analysis_seconds"]),
            model_checkpoint=None,
            observations=[
                f"Matched {preset} seed-7 phase and cosine checkpoints were frozen.",
                "The complete residual sequence was barycentered over exact shared-nuisance cosine fibers.",
                "No probe, alignment, decoder, or model parameter was fit.",
                (
                    "Quality evidence: valid causal null."
                    if preset == "d8"
                    else "Validity-only: outside-range cosine baseline missed the locked floor."
                ),
                f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
            ],
            anomalies=(
                ["Outside-range cosine baseline failed the preregistered accuracy floor."]
                if preset == "d6"
                else ["No isolated or mature barycenter-sufficient cut exists."]
            ),
            timestamp=completed,
        )
        for preset, detail in details.items()
    ]


def store_task_relative_activation_barycenter_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_task_relative_activation_barycenter_meta_hypothesis(results_path)
    experiments = build_task_relative_activation_barycenter_experiment_results(
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
            or {item.get("hypothesis_id") for item in results.get("metadatas", [])}
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("task-relative barycenter ChromaDB read-back failed")
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
    "build_task_relative_activation_barycenter_experiment_results",
    "build_task_relative_activation_barycenter_meta_hypothesis",
    "store_task_relative_activation_barycenter_meta_hypothesis",
]
