"""Build and store the TinyLLM observed shared-bias repair result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-bias-reference-recentering.v2"
HYPOTHESIS_ID = "tinyllm-bias-reference-recentering-v2"
EVIDENCE_ROLE = (
    "preregistered_pre_model_numerical_corrective_bias_reference_intervention"
)
CLASSIFICATION = "observed_bias_reference_repair_specific"
CAMPAIGN_SHA256 = (
    "1996ac4c2534b62a25a2f52ceadfd21055a91bdadc81f38ccf01c6855da2b7d0"
)
IMPLEMENTATION_SHA256 = (
    "059d4ace65402fb296bcf35bf614aa411e085cc34abaf8513e077678a4828e15"
)
RUNNER_SHA256 = (
    "fd6ea5108ccd733e360010c83a1a4a411512cbed239e3c9356a6a6bb77a6996a"
)
PREREGISTRATION_SHA256 = (
    "e2d04b2852bffaac0ce190a4245f54a0e587a6d8323468afa91b922ef7c2c86b"
)
SOURCE_COMPONENT_RUNNER_SHA256 = (
    "eba5182082d8604fba47d65fc0f64706b00ac9f4fde6dbf45c63fca56ed44bb5"
)
RESULT_MANIFEST_SHA256 = (
    "7dbbe3a49f4e3ebac36e891ec63d5336ff3be2e176e26f1a610cbfceecaabb4e"
)
INTERVENTION_CONTRACT_SHA256 = (
    "6bed75b6cd9a15be35f21e53463efa28bcc2f775f1490f31414e005398894004"
)
SOURCE_CAMPAIGN_SHA256 = (
    "9f7fdf98e83a320d5d49d9191e6a0f0cd6f872f32f406381c5a290f517dbed4b"
)
SOURCE_COMPONENT_CONTRACT_SHA256 = (
    "26b8ad368fe8d1af811f2ff62d4874545c6d90b3aa5d376a9a59002092342b2f"
)
SOURCE_DVC_ROOT = "e3bfc6a9401916ffc7f942678044fb0a.dir"
SOURCE_LAKEFS_COMMIT = (
    "a0f6b67d7aad58dc96de58406abf7064728613e73134ba4959e18dd46c0cc92a"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
VARIANTS = (
    "source_full_plus",
    "recenter_correct",
    "recenter_wrong_sign",
    "recenter_target_changing",
)
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_bias_reference_recentering.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-bias-reference-recentering-v2-preregistration.md"
)
SOURCE_CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_bias_component_causal_decomposition/"
    "20260810_d10_preregistered/campaign_results.json"
)
SOURCE_COMPONENT_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_bias_component_causal_decomposition.py"
)
DATASET_HASHES = {
    "composition": "b025623e23f534ca3670f49f1bafc1c3979d1ca834aec2223f9c3166be8f52a6",
    "extrapolation": "f2f1e8c2c06b38fbecf856f0679e9861d9fab3e1c2a41358e55043f4d309a214",
}
EXPECTED_COUNTS = {
    "analytic_calibrated": {
        "source_full_plus": 1,
        "recenter_correct": 5,
        "recenter_wrong_sign": 1,
        "recenter_target_changing": 0,
    },
    "learned_calibrated_equivariant": {
        "source_full_plus": 3,
        "recenter_correct": 5,
        "recenter_wrong_sign": 0,
        "recenter_target_changing": 0,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=128)
def _require_source_digest(
    path: Path, expected: str, size: int, modified_ns: int
) -> None:
    del size, modified_ns
    if _sha256(path) != expected:
        raise ValueError(f"source artifact changed: {path}")


def _require_source(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ValueError(f"source artifact changed: {path}")
    stat = path.stat()
    _require_source_digest(path, expected, stat.st_size, stat.st_mtime_ns)


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


def _expected_configuration() -> dict[str, Any]:
    return {
        "action_involution_tolerance": 2e-6,
        "allow_underpowered": False,
        "batch_size": 256,
        "conditions": list(CONDITIONS),
        "construction_tolerance": 2e-7,
        "device": "cuda:2",
        "feature_equivalence_tolerance": 1e-6,
        "maximum_control_seed_passes": 1,
        "metric_replay_tolerance": 2e-6,
        "minimum_target_changing_feature_rms": 0.5,
        "natural_accuracy_loss_ceiling": 0.05,
        "natural_circular_error_increase_ceiling": math.pi / 16.0,
        "natural_cross_entropy_increase_ceiling": 0.1,
        "posterior_replay_tolerance": 2e-6,
        "required_seed_passes": 4,
        "sample_limit": None,
        "seeds": list(SEEDS),
        "selected_noise_sigma": 0.03125,
        "source_component_root": (
            "data/experiments/tinyllm_bias_component_causal_decomposition/"
            "20260810_d10_preregistered"
        ),
    }


def _expected_population() -> dict[str, Any]:
    return {
        "arms": {
            condition: {
                "variants": {
                    variant: {"natural_utility_pass_count": count}
                    for variant, count in counts.items()
                }
            }
            for condition, counts in EXPECTED_COUNTS.items()
        },
        "control_specific": {
            variant: {condition: True for condition in CONDITIONS}
            for variant in (
                "recenter_target_changing",
                "recenter_wrong_sign",
            )
        },
        "integrity_valid": True,
        "population_passes": {
            "recenter_correct": {condition: True for condition in CONDITIONS},
            "recenter_target_changing": {
                condition: False for condition in CONDITIONS
            },
            "recenter_wrong_sign": {
                condition: False for condition in CONDITIONS
            },
            "source_full_plus": {condition: False for condition in CONDITIONS},
        },
        "recenter_correct_passes_both_arms": True,
        "source_full_plus_fails_both_arms": True,
        "target_changing_specific_both_arms": True,
        "wrong_sign_specific_both_arms": True,
    }


def _campaign(path: Path) -> dict[str, Any]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or campaign.get("schema_version") != EXPERIMENT_SCHEMA
        or campaign.get("hypothesis_id") != HYPOTHESIS_ID
        or campaign.get("evidence_role") != EVIDENCE_ROLE
        or campaign.get("status") != "completed"
        or campaign.get("configuration") != _expected_configuration()
        or campaign.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or campaign.get("source_digests")
        != {
            "preregistration": PREREGISTRATION_SHA256,
            "runner": RUNNER_SHA256,
            "source_component_runner": SOURCE_COMPONENT_RUNNER_SHA256,
        }
        or campaign.get("source_campaign_sha256") != SOURCE_CAMPAIGN_SHA256
        or campaign.get("source_component_contract_sha256")
        != SOURCE_COMPONENT_CONTRACT_SHA256
        or campaign.get("source_dvc_root") != SOURCE_DVC_ROOT
        or campaign.get("source_lakefs_commit") != SOURCE_LAKEFS_COMMIT
        or campaign.get("dataset_hashes") != DATASET_HASHES
        or campaign.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or campaign.get("intervention_contract_sha256")
        != INTERVENTION_CONTRACT_SHA256
        or campaign.get("intervention_contract", {}).get("pass") is not True
        or campaign.get("population") != _expected_population()
        or campaign.get("aggregates")
        != {
            "classification": CLASSIFICATION,
            "integrity_valid": True,
            "intervention_contract_pass": True,
            "maximum_control_seed_passes": 1,
            "primary_evaluable": True,
            "primary_hypothesis_pass": True,
            "required_seed_passes": 4,
            "valid": True,
        }
        or campaign.get("summary")
        != {
            "completed": 10,
            "excluded": 0,
            "failed": 0,
            "fitted_actions": 0,
            "fitted_bias_estimators": 0,
            "fitted_denoisers": 0,
            "fitted_observers": 0,
            "fitted_probes": 0,
            "new_forward_variants_per_system_shift": 3,
            "new_random_draws": 0,
            "requested": 10,
            "reused": 0,
            "reused_source_variants_per_system_shift": 3,
            "trained_frontends": 0,
            "trained_models": 0,
            "trained_task_heads": 0,
        }
        or len(campaign.get("results", [])) != 10
        or not _finite(campaign)
    ):
        raise ValueError(f"invalid bias-reference campaign {path}")
    contract = campaign["intervention_contract"]["regimes"]
    if (
        max(
            record["repaired_vs_centered_corrected_maximum_absolute_error"]
            for record in contract.values()
        )
        > 2e-7
        or min(
            record["target_changing_analytic_feature_rms"]
            for record in contract.values()
        )
        < 0.5
    ):
        raise ValueError(f"invalid bias-reference construction in {path}")
    for source_path, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (SOURCE_CAMPAIGN_PATH, SOURCE_CAMPAIGN_SHA256),
        (SOURCE_COMPONENT_RUNNER_PATH, SOURCE_COMPONENT_RUNNER_SHA256),
    ):
        _require_source(source_path, expected)
    return campaign


def _details(results_path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(results_path)
    expected_cells = {
        (condition, seed) for condition in CONDITIONS for seed in SEEDS
    }
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in campaign["results"]:
        result_path = Path(entry["path"])
        diagnostics_path = Path(entry["diagnostics_path"])
        detail = json.loads(result_path.read_text(encoding="utf-8"))
        cell = (str(detail.get("condition")), int(detail.get("seed", -1)))
        if (
            _sha256(result_path) != entry.get("result_sha256")
            or _sha256(diagnostics_path) != entry.get("diagnostics_sha256")
            or entry.get("validity") is not True
            or detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or detail.get("status") != "completed"
            or detail.get("configuration") != _expected_configuration()
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or detail.get("intervention_contract_sha256")
            != INTERVENTION_CONTRACT_SHA256
            or detail.get("dataset_hashes") != DATASET_HASHES
            or set(detail.get("variant_seed_gates", {})) != set(VARIANTS)
            or set(detail.get("regimes", {})) != set(REGIMES)
            or not all(detail.get("gates", {}).values())
            or not _finite(detail)
        ):
            raise ValueError(f"invalid bias-reference result {result_path}")
        for regime in REGIMES:
            regime_detail = detail["regimes"][regime]
            if (
                regime_detail["clean_posterior_replay_maximum_absolute_error"]
                != 0.0
                or regime_detail["source_metric_replay_maximum_absolute_error"]
                != 0.0
                or regime_detail[
                    "repaired_vs_centered_feature_maximum_absolute_error"
                ]
                > 1e-6
                or regime_detail[
                    "repaired_vs_centered_posterior_maximum_absolute_error"
                ]
                > 2e-6
                or set(regime_detail.get("variants", {})) != set(VARIANTS)
                or regime_detail["variants"]["source_full_plus"].get(
                    "source_reused"
                )
                is not True
                or any(
                    regime_detail["variants"][variant].get("source_reused")
                    is not False
                    for variant in (
                        "recenter_correct",
                        "recenter_wrong_sign",
                        "recenter_target_changing",
                    )
                )
            ):
                raise ValueError(f"invalid bias-reference replay in {result_path}")
        provenance = detail["provenance"]
        for key in (
            "model_checkpoint",
            "frontend_checkpoint",
            "source_result",
            "source_component_result",
            "source_component_diagnostics",
        ):
            _require_source(Path(provenance[key]), provenance[f"{key}_sha256"])
        detail["_result_path"] = str(result_path)
        detail["_result_sha256"] = entry["result_sha256"]
        detail["_diagnostics_path"] = str(diagnostics_path)
        detail["_diagnostics_sha256"] = entry["diagnostics_sha256"]
        output[cell] = detail
    if set(output) != expected_cells:
        raise ValueError("bias-reference population changed")
    for condition in CONDITIONS:
        for variant in VARIANTS:
            count = sum(
                output[(condition, seed)]["variant_seed_gates"][variant]
                for seed in SEEDS
            )
            if count != EXPECTED_COUNTS[condition][variant]:
                raise ValueError("bias-reference pass counts changed")
    return [output[cell] for cell in sorted(output)]


def _detail_summary(detail: Mapping[str, Any]) -> dict[str, float]:
    metrics = {
        "validity": 1.0,
        **{
            f"{variant}_seed_pass": float(
                detail["variant_seed_gates"][variant]
            )
            for variant in VARIANTS
        },
        "specific_repair_seed_pattern": float(
            detail["variant_seed_gates"]["recenter_correct"]
            and not detail["variant_seed_gates"]["recenter_wrong_sign"]
            and not detail["variant_seed_gates"]["recenter_target_changing"]
        ),
    }
    for regime in REGIMES:
        regime_detail = detail["regimes"][regime]
        metrics[f"{regime}_feature_equivalence_error"] = float(
            regime_detail["repaired_vs_centered_feature_maximum_absolute_error"]
        )
        metrics[f"{regime}_posterior_equivalence_error"] = float(
            regime_detail[
                "repaired_vs_centered_posterior_maximum_absolute_error"
            ]
        )
        for variant in VARIANTS:
            natural = regime_detail["variants"][variant]["natural_utility"]
            metrics[f"{regime}_{variant}_accuracy_loss"] = float(
                natural["accuracy_loss"]
            )
            metrics[f"{regime}_{variant}_circular_increase"] = float(
                natural["circular_error_increase"]
            )
            metrics[f"{regime}_{variant}_cross_entropy_increase"] = float(
                natural["cross_entropy_increase"]
            )
    return metrics


def build_bias_reference_recentering_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-10_tinyllm-bias-reference-recentering.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    summaries = [
        {
            "experiment_id": detail["experiment_id"],
            "condition": detail["condition"],
            "seed": detail["seed"],
            "validity": True,
            "metrics": _detail_summary(detail),
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM observed shared-bias reference recentering",
            "category": "mechanistic_interpretability",
            "description": (
                "A no-fit zero-signal pilot updates only the observed calibration "
                "offset in ten frozen calibrated TinyLLM systems."
            ),
            "question": (
                "Can an observed reference to the persistent signed bias repair "
                "natural utility without retraining or target information?"
            ),
            "prediction": (
                "Correct recentering passes at least four of five seeds in both "
                "arms, while wrong-sign and target-changing controls pass at "
                "most one seed per arm."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": "confirmed_specific_observed_bias_repair",
            "evidence_count": 10,
            "direct_experiment_count": 10,
            "direct_quality_experiment_count": 10,
            "integrity_valid_experiment_count": 10,
            "fixed_checkpoint_seed_count": 5,
            "stored_system_count": 10,
            "evidence_role": EVIDENCE_ROLE,
            "power_profile": "two_arm_five_seed_frozen_reference_intervention",
            "tested_scope": (
                "ten retained d8/N3 calibrated systems, one exact shared-bias "
                "zero-signal pilot, two shifts, and two specificity controls"
            ),
            "subclaims": {
                "correct_recentering_utility": "supported_five_of_five_both_arms",
                "wrong_sign_specificity": (
                    "supported_analytic_one_of_five_learned_zero_of_five"
                ),
                "target_changing_specificity": "supported_zero_of_five_both_arms",
                "repaired_centered_equivalence": (
                    "supported_feature_and_posterior_contracts_all_systems"
                ),
                "exact_pilot_interface_repair": "supported",
                "finite_noisy_pilot_acquisition": "not_tested",
                "same_scope_retraining": "not_licensed",
            },
            "tags": [
                "tinyllm",
                "sensor-bias",
                "calibration-reference",
                "recentering",
                "causal-repair",
                "frozen-checkpoint",
                "no-fit",
                "preregistered-corrective",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "Correct recentering passed 5/5 seeds in both arms, wrong-sign "
                "passed only 1/5 analytic and 0/5 learned systems, and the "
                "target-changing control passed none."
            ),
            "confidence": 0.99,
            "confidence_assessment": (
                "high_confidence_specific_exact_pilot_positive_control"
            ),
            "num_direct_experiments": 10,
            "num_direct_quality_experiments": 10,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "population": campaign["population"],
                "intervention_contract": campaign["intervention_contract"],
                "system_summaries": summaries,
            },
            "key_insights": [
                "The signed deterministic sensor defect is removable by changing only the observed calibration offset.",
                "Every repaired frozen system reproduces the sealed centered-only utility profile on both shifts.",
                "Doubling rather than canceling the bias fails both population gates, establishing sign specificity.",
                "The observed target-changing action fails every system, establishing task specificity.",
                "The result is an exact-pilot positive control and does not yet establish finite noisy pilot acquisition."
            ],
            "suggested_hypotheses": [
                "A nested average of noisy zero-signal pilot measurements preserves the repair above a finite sample threshold.",
                "The required pilot count scales with the exact-bin calibration margin rather than internal quotient geometry.",
                "If the first noisy-pilot level fails, the practical recentering branch should close without retraining."
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"system_summaries": summaries},
        "source_artifacts": [
            str(results_path),
            str(report_path),
            str(PREREGISTRATION_PATH),
            str(SOURCE_CAMPAIGN_PATH),
        ],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "intervention_contract_sha256": INTERVENTION_CONTRACT_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "source_component_contract_sha256": (
                SOURCE_COMPONENT_CONTRACT_SHA256
            ),
            "source_dvc_root": SOURCE_DVC_ROOT,
            "source_lakefs_commit": SOURCE_LAKEFS_COMMIT,
        },
    }


def build_bias_reference_recentering_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    campaign = _campaign(results_path)
    completed = datetime.fromisoformat(
        campaign["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        metrics = _detail_summary(detail)
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=metrics["specific_repair_seed_pattern"],
                model_architecture=[8, 512],
                model_parameters=50_964_992,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["model_checkpoint"],
                observations=[
                    f"Condition: {detail['condition']}; seed: {detail['seed']}.",
                    "The exact pilot used no latent phase, task target, or answer label.",
                    "All construction, replay, equivalence, state, and specificity contracts passed.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[
                    "Version 1 stopped pre-model on a one-ULP float32 construction miss; v2 retained the threshold and separated algebraic from realized equivalence."
                ],
                timestamp=completed,
            )
        )
    return output


def store_bias_reference_recentering_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_bias_reference_recentering_meta_hypothesis(results_path)
    experiments = build_bias_reference_recentering_experiment_results(
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
            raise RuntimeError("bias-reference ChromaDB read-back failed")
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
    "build_bias_reference_recentering_experiment_results",
    "build_bias_reference_recentering_meta_hypothesis",
    "store_bias_reference_recentering_meta_hypothesis",
]
