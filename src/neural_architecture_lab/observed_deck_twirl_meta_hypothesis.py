"""Build and store the preregistered TinyLLM observed-deck twirl evidence."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-observed-deck-twirl-causal-closure.v1"
HYPOTHESIS_ID = "tinyllm-observed-deck-twirl-causal-closure-v1"
EVIDENCE_ROLE = "preregistered_frozen_observed_action_intervention"
CAMPAIGN_SHA256 = (
    "79c3e27374d8b6f4611552595de5852ace940204bda825e64cf80eff6ab2050d"
)
IMPLEMENTATION_SHA256 = (
    "c970fe8801524f5248a9314e821b6783127596d05a2f206325ed85deb42f9629"
)
RESULT_MANIFEST_SHA256 = (
    "b91af38162fbf45e29348fbdf583cb676660d68cf22e5a795b438fd8cd015db3"
)
ACTION_CONTRACT_SHA256 = (
    "b20fca24168f4e9386e9afdbd3d1980ab100d44b6787bfa48ac1f6ef8de34d60"
)
SOURCE_CLOSURE_SHA256 = (
    "1457fdcce224157c5d8b19c11317b3d3c47cc7d8575097c78e76ba8798431b14"
)
PREREGISTRATION_SHA256 = (
    "9c397fe42b6bdcb1952d9ed7a5865889e0f919bc5dd0bffce1cb0ef56e484030"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
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
    arms = aggregates.get("arms", {})
    source_path = Path(provenance.get("source_closure_campaign", ""))
    preregistration_path = Path(provenance.get("preregistration", ""))
    expected_summary = {
        "requested": 10,
        "scheduled": 10,
        "completed": 10,
        "failed": 0,
        "excluded": 0,
        "retries": 0,
        "reused": 0,
        "trained_models": 0,
        "trained_frontends": 0,
        "trained_task_heads": 0,
        "fitted_probes": 0,
        "fitted_observers": 0,
        "fitted_action_parameters": 0,
    }
    expected_fives = {cut: 5 for cut in CUTS}
    expected_zeroes = {cut: 0 for cut in CUTS}
    expected_transitions = {transition: 5 for transition in TRANSITIONS}
    arm_contract = all(
        arms.get(condition, {}).get("twirl_pass_counts") == expected_fives
        and arms.get(condition, {}).get("action_pass_counts") == expected_fives
        and arms.get(condition, {}).get("control_twirl_pass_counts")
        == expected_zeroes
        and arms.get(condition, {}).get("control_action_pass_counts")
        == expected_zeroes
        and arms.get(condition, {}).get("transition_closed_counts")
        == expected_transitions
        for condition in CONDITIONS
    )
    action_contract = value.get("action_contract", {})
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or value.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or value.get("action_contract_sha256") != ACTION_CONTRACT_SHA256
        or tuple(config.get("conditions", ())) != CONDITIONS
        or tuple(config.get("seeds", ())) != SEEDS
        or int(config.get("required_seed_passes", -1)) != 4
        or int(config.get("maximum_control_seed_passes", -1)) != 1
        or config.get("allow_underpowered") is not False
        or summary != expected_summary
        or aggregates.get("valid") is not True
        or aggregates.get("controls_pass") is not True
        or aggregates.get("primary_hypothesis_pass") is not True
        or aggregates.get("classification")
        != "observable_twirl_closed_action_invariant"
        or not arm_contract
        or action_contract.get("pass") is not True
        or "fiber_id" not in action_contract.get("forbidden_inputs", [])
        or "latent_phase" not in action_contract.get("forbidden_inputs", [])
        or provenance.get("source_closure_campaign_sha256")
        != SOURCE_CLOSURE_SHA256
        or provenance.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or provenance.get("source_preflight_completed_before_interventions")
        is not True
        or not source_path.is_file()
        or _sha256(source_path) != SOURCE_CLOSURE_SHA256
        or not preregistration_path.is_file()
        or _sha256(preregistration_path) != PREREGISTRATION_SHA256
        or not _finite(value)
    ):
        raise ValueError(f"invalid observed-deck twirl campaign {path}")
    manifest = hashlib.sha256(
        json.dumps(
            value.get("results", []), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if manifest != RESULT_MANIFEST_SHA256:
        raise ValueError(f"invalid observed-deck result manifest {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {
        (entry["condition"], int(entry["seed"])): entry
        for entry in campaign.get("results", [])
    }
    expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    if set(entries) != expected:
        raise ValueError(f"invalid observed-deck result index {path}")
    output = []
    fingerprints: set[str] = set()
    for condition in CONDITIONS:
        for seed in SEEDS:
            entry = entries[(condition, seed)]
            result_path = Path(entry["path"])
            diagnostics_path = Path(entry["diagnostics_path"])
            detail = json.loads(result_path.read_text(encoding="utf-8"))
            provenance = detail.get("provenance", {})
            source_path = Path(provenance.get("source_result", ""))
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
                or detail.get("action_contract_sha256") != ACTION_CONTRACT_SHA256
                or detail.get("twirl_seed_gates") != {cut: True for cut in CUTS}
                or detail.get("action_seed_gates") != {cut: True for cut in CUTS}
                or detail.get("control_twirl_seed_gates")
                != {cut: False for cut in CUTS}
                or detail.get("control_action_seed_gates")
                != {cut: False for cut in CUTS}
                or detail.get("transition_seed_gates")
                != {transition: True for transition in TRANSITIONS}
                or detail.get("gates", {}).get("validity") is not True
                or not source_path.is_file()
                or _sha256(source_path) != provenance.get("source_result_sha256")
                or not _finite(detail)
            ):
                raise ValueError(f"invalid observed-deck result {result_path}")
            fingerprint = detail["scientific_fingerprint"]
            if fingerprint in fingerprints:
                raise ValueError(f"duplicate observed-deck fingerprint {result_path}")
            fingerprints.add(fingerprint)
            detail["_model_parameters"] = int(source["model_parameters"])
            output.append(detail)
    return output


def _detail_summary(detail: Mapping[str, Any]) -> dict[str, float]:
    action_losses = []
    twirl_gains = []
    action_js = []
    twirl_js = []
    control_losses = []
    action_geometry = []
    defects = []
    for regime in REGIMES:
        regime_result = detail["regimes"][regime]
        for cut in CUTS:
            cut_result = regime_result["cuts"][cut]
            action_losses.append(
                float(cut_result["correct_action"]["task_sufficiency"]["accuracy_loss"])
            )
            twirl_gains.append(
                -float(cut_result["correct_twirl"]["task_sufficiency"]["accuracy_loss"])
            )
            action_js.append(float(cut_result["correct_action"]["posterior_js_from_baseline"]))
            twirl_js.append(float(cut_result["correct_twirl"]["posterior_js_from_baseline"]))
            control_losses.append(
                float(cut_result["orthogonal_twirl"]["task_sufficiency"]["accuracy_loss"])
            )
            action_geometry.append(
                float(
                    cut_result["identity_vs_correct_action_geometry"][
                        "difference_relative_rms"
                    ]
                )
            )
        for transition in TRANSITIONS:
            defects.append(
                float(
                    regime_result["transitions"][transition]["defect_geometry"][
                        "difference_relative_rms"
                    ]
                )
            )
    return {
        "correct_action_all_cuts_pass": float(all(detail["action_seed_gates"].values())),
        "observed_twirl_all_cuts_pass": float(all(detail["twirl_seed_gates"].values())),
        "all_control_actions_fail": float(not any(detail["control_action_seed_gates"].values())),
        "all_control_twirls_fail": float(not any(detail["control_twirl_seed_gates"].values())),
        "both_block0_transitions_closed": float(all(detail["transition_seed_gates"].values())),
        "maximum_correct_action_accuracy_loss": max(action_losses),
        "minimum_observed_twirl_accuracy_gain": min(twirl_gains),
        "maximum_observed_twirl_accuracy_gain": max(twirl_gains),
        "maximum_correct_action_posterior_js": max(action_js),
        "maximum_observed_twirl_posterior_js": max(twirl_js),
        "minimum_control_twirl_accuracy_loss": min(control_losses),
        "maximum_action_relative_rms": max(action_geometry),
        "maximum_relative_reynolds_defect": max(defects),
        "analysis_seconds": float(detail["analysis_seconds"]),
    }


def build_observed_deck_twirl_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-10_tinyllm-observed-deck-twirl.md"
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
            "name": "TinyLLM observed-deck twirl causal closure",
            "category": "mechanistic_interpretability",
            "description": (
                "A deterministic target-preserving C2 action constructed from "
                "one decoded calibrated observation tests oracle-free quotient projection."
            ),
            "question": (
                "Can an observed within-example deck twirl replace oracle fiber "
                "membership while preserving the frozen task?"
            ),
            "prediction": (
                "Correct action twirls pass both shifts in at least four of five "
                "checkpoints per structured arm and orthogonal-axis controls fail."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": (
                "preregistered_observable_twirl_closed_action_invariant_five_of_five"
            ),
            "evidence_count": 10,
            "direct_experiment_count": 10,
            "fixed_checkpoint_count": 5,
            "stored_checkpoint_summary_count": 10,
            "evidence_role": EVIDENCE_ROLE,
            "power_profile": "two_arm_five_checkpoint_two_shift_observed_action_intervention",
            "tested_scope": (
                "ten retained calibrated structured d8/N3 systems, one observed "
                "C2 planar action, composition and extrapolation, four residual cuts"
            ),
            "subclaims": {
                "observable_pre_block_twirl": "supported_five_of_five_both_arms",
                "correct_action_task_invariance": "supported_five_of_five_both_arms",
                "orthogonal_axis_specificity": "supported_zero_of_160_cells",
                "block0_task_closure": "supported_five_of_five_both_transitions",
                "oracle_fiber_membership_required": "no_for_structured_C2_projection",
                "oracle_independent_nuisance_gain_deployable": "no_conflates_projection_and_denoising",
                "same_scope_group_training_licensed": "no",
            },
            "tags": [
                "tinyllm",
                "observed-group-action",
                "reynolds-twirl",
                "oracle-free",
                "causal-activation-patching",
                "frozen-checkpoint",
                "preregistered",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "Correct observed actions and twirls pass every checkpoint and "
                "cut on both shifts, all matched orthogonal controls fail, and "
                "the constructor uses no latent or fiber identifier."
            ),
            "confidence": 0.99,
            "confidence_assessment": (
                "strong_preregistered_causal_support_with_declared_action_scope"
            ),
            "num_direct_experiments": 10,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "action_contract": campaign["action_contract"],
                "checkpoint_summaries": summaries,
            },
            "key_insights": [
                "The calibrated structured C2 quotient has a deterministic observed projection; oracle fiber IDs are unnecessary.",
                "The learned front end is approximately action-invariant and its within-example twirl reduces the remaining posterior discrepancy.",
                "Orthogonal-axis reflections isolate the correct semantic action from generic isometric averaging.",
                "The larger accuracy gain from the prior oracle pair includes independent-nuisance denoising and is not a pure projection effect.",
                "Block-0 attention and MLP remain task-closed on the observed twirl.",
            ],
            "suggested_hypotheses": [
                "Unknown or noisy calibration axes will move the bottleneck from quotient use to action estimation.",
                "Anisotropic or biased sensor noise will break exact reflection compatibility before it breaks the frozen continuation.",
                "Richer nuisance groups without a one-observation action constructor will require explicit group-typed sensing rather than residual penalties.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"checkpoint_summaries": summaries},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "action_contract_sha256": ACTION_CONTRACT_SHA256,
            "source_closure_campaign_sha256": SOURCE_CLOSURE_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
        },
    }


def build_observed_deck_twirl_experiment_results(
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
                primary_metric=float(summary["observed_twirl_all_cuts_pass"]),
                model_architecture=[8, int(detail["_model_parameters"])],
                model_parameters=int(detail["_model_parameters"]),
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["model_checkpoint"],
                observations=[
                    f"Condition: {detail['condition']}.",
                    "The action used decoded planar history and observed calibration only.",
                    "No latent phase, target, branch, fiber ID, or fitted parameter entered the intervention.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return output


def store_observed_deck_twirl_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_observed_deck_twirl_meta_hypothesis(results_path)
    experiments = build_observed_deck_twirl_experiment_results(record, results_path)
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
            raise RuntimeError("observed-deck ChromaDB read-back failed")
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
    "build_observed_deck_twirl_experiment_results",
    "build_observed_deck_twirl_meta_hypothesis",
    "store_observed_deck_twirl_meta_hypothesis",
]
