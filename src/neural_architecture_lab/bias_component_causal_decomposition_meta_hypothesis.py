"""Build and store the TinyLLM frozen bias-component causal result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-bias-component-causal-decomposition.v1"
HYPOTHESIS_ID = "tinyllm-bias-component-causal-decomposition-v1"
EVIDENCE_ROLE = "preregistered_frozen_bias_component_intervention"
CLASSIFICATION = "deterministic_mean_sufficient"
SIGN_CLASSIFICATION = "positive_direction_specific"
CAMPAIGN_SHA256 = (
    "9f7fdf98e83a320d5d49d9191e6a0f0cd6f872f32f406381c5a290f517dbed4b"
)
IMPLEMENTATION_SHA256 = (
    "c1b340bf1e29e485d2c254902ac6aaab87abdd594e2f46875a5e828fff415c98"
)
RUNNER_SHA256 = (
    "eba5182082d8604fba47d65fc0f64706b00ac9f4fde6dbf45c63fca56ed44bb5"
)
PREREGISTRATION_SHA256 = (
    "a3052f55181f72fb9b53d4bc8ad7a42fe28d5762acd6ee4ffbb6a0d31e81d85e"
)
SOURCE_DOSE_RUNNER_SHA256 = (
    "39a72dd535f96f13bae644c74096b298b85fb8587d980211dc489ed463aeb725"
)
RESULT_MANIFEST_SHA256 = (
    "17d614cadfeca5e019258578ad9abe8dc269f899f7144e712e8154f7988ce07b"
)
COMPONENT_CONTRACT_SHA256 = (
    "26b8ad368fe8d1af811f2ff62d4874545c6d90b3aa5d376a9a59002092342b2f"
)
SOURCE_CAMPAIGN_SHA256 = (
    "9b05823ebdb88bd828f27699da596dc5e7dcf0c4af5e13f1664fa70e5111f9bd"
)
SOURCE_DVC_ROOT = "c07286d2b9710cd68228cd21f487e425.dir"
SOURCE_LAKEFS_COMMIT = (
    "d4fb92ef41e39d0cc672d672e55c9192ea0e9dcf01597b1a549efcf973577061"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
VARIANTS = ("centered", "mean_plus", "full_plus", "full_minus")
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_bias_component_causal_decomposition.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-bias-component-causal-decomposition-preregistration.md"
)
SOURCE_CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_noise_law_dose_localization/"
    "20260810_d10_preregistered/campaign_results.json"
)
SOURCE_DOSE_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_noise_law_dose_localization.py"
)
DATASET_HASHES = {
    "composition": "b025623e23f534ca3670f49f1bafc1c3979d1ca834aec2223f9c3166be8f52a6",
    "extrapolation": "f2f1e8c2c06b38fbecf856f0679e9861d9fab3e1c2a41358e55043f4d309a214",
}
EXPECTED_COUNTS = {
    "analytic_calibrated": {
        "centered": 5,
        "mean_plus": 2,
        "full_plus": 1,
        "full_minus": 4,
    },
    "learned_calibrated_equivariant": {
        "centered": 5,
        "mean_plus": 3,
        "full_plus": 3,
        "full_minus": 4,
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
        "allow_underpowered": False,
        "batch_size": 256,
        "calibrated_source_root": (
            "data/experiments/tinyllm_calibrated_frontend_causal/"
            "20260806_d8_preregistered"
        ),
        "component_reconstruction_tolerance": 2e-7,
        "conditions": list(CONDITIONS),
        "device": "cuda:2",
        "natural_accuracy_loss_ceiling": 0.05,
        "natural_circular_error_increase_ceiling": math.pi / 16.0,
        "natural_cross_entropy_increase_ceiling": 0.1,
        "required_seed_passes": 4,
        "sample_limit": None,
        "seeds": list(SEEDS),
        "selected_multiplier": 0.625,
        "selected_noise_sigma": 0.03125,
        "sign_energy_relative_tolerance": 0.02,
        "source_closure_root": (
            "data/experiments/tinyllm_calibrated_frontend_causal_closure/"
            "20260810_d15_preregistered"
        ),
        "source_dose_root": (
            "data/experiments/tinyllm_noise_law_dose_localization/"
            "20260810_d10_preregistered"
        ),
        "source_metric_replay_tolerance": 2e-6,
        "source_noise_root": (
            "data/experiments/tinyllm_noise_law_observed_twirl/"
            "20260810_d10_preregistered"
        ),
        "source_noise_sigma": 0.05,
        "source_observed_root": (
            "data/experiments/tinyllm_observed_deck_twirl/"
            "20260810_d10_preregistered"
        ),
        "source_posterior_replay_tolerance": 2e-6,
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
        "centered_passes_both_arms": True,
        "full_plus_fails_both_arms": True,
        "integrity_valid": True,
        "mean_plus_fails_both_arms": True,
        "population_passes": {
            "centered": {condition: True for condition in CONDITIONS},
            "full_minus": {condition: True for condition in CONDITIONS},
            "full_plus": {condition: False for condition in CONDITIONS},
            "mean_plus": {condition: False for condition in CONDITIONS},
        },
        "sign_classification": SIGN_CLASSIFICATION,
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
            "source_dose_runner": SOURCE_DOSE_RUNNER_SHA256,
        }
        or campaign.get("source_campaign_sha256") != SOURCE_CAMPAIGN_SHA256
        or campaign.get("source_dvc_root") != SOURCE_DVC_ROOT
        or campaign.get("source_lakefs_commit") != SOURCE_LAKEFS_COMMIT
        or campaign.get("dataset_hashes") != DATASET_HASHES
        or campaign.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or campaign.get("component_contract_sha256")
        != COMPONENT_CONTRACT_SHA256
        or campaign.get("component_contract", {}).get("pass") is not True
        or campaign.get("population") != _expected_population()
        or campaign.get("aggregates")
        != {
            "classification": CLASSIFICATION,
            "component_contract_pass": True,
            "integrity_valid": True,
            "primary_evaluable": True,
            "primary_hypothesis_pass": True,
            "required_seed_passes": 4,
            "sign_classification": SIGN_CLASSIFICATION,
            "valid": True,
        }
        or campaign.get("summary")
        != {
            "completed": 10,
            "excluded": 0,
            "failed": 0,
            "fitted_actions": 0,
            "fitted_denoisers": 0,
            "fitted_noise_models": 0,
            "fitted_observers": 0,
            "fitted_probes": 0,
            "new_forward_variants_per_system_shift": 3,
            "new_random_draws": 0,
            "requested": 10,
            "reused": 0,
            "reused_source_variants_per_system_shift": 2,
            "trained_frontends": 0,
            "trained_models": 0,
            "trained_task_heads": 0,
        }
        or len(campaign.get("results", [])) != 10
        or not _finite(campaign)
    ):
        raise ValueError(f"invalid bias-component campaign {path}")
    for source_path, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (SOURCE_CAMPAIGN_PATH, SOURCE_CAMPAIGN_SHA256),
        (SOURCE_DOSE_RUNNER_PATH, SOURCE_DOSE_RUNNER_SHA256),
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
            or detail.get("component_contract_sha256")
            != COMPONENT_CONTRACT_SHA256
            or detail.get("dataset_hashes") != DATASET_HASHES
            or detail.get("gates")
            != {
                "clean_posterior_replay": True,
                "finite": True,
                "source_full_plus_seed_gate_match": True,
                "source_metric_replay": True,
                "state_unchanged": True,
                "validity": True,
            }
            or set(detail.get("variant_seed_gates", {})) != set(VARIANTS)
            or set(detail.get("regimes", {})) != set(REGIMES)
            or not _finite(detail)
        ):
            raise ValueError(f"invalid bias-component result {result_path}")
        for regime in REGIMES:
            regime_detail = detail["regimes"][regime]
            if (
                regime_detail.get("clean_posterior_replay_maximum_absolute_error")
                != 0.0
                or regime_detail.get("source_metric_replay_maximum_absolute_error")
                != 0.0
                or set(regime_detail.get("variants", {})) != set(VARIANTS)
                or regime_detail["variants"]["full_plus"].get("source_reused")
                is not True
                or any(
                    regime_detail["variants"][variant].get("source_reused")
                    is not False
                    for variant in ("centered", "mean_plus", "full_minus")
                )
            ):
                raise ValueError(f"invalid component replay in {result_path}")
        provenance = detail["provenance"]
        for key in (
            "model_checkpoint",
            "frontend_checkpoint",
            "source_result",
            "source_dose_result",
            "source_dose_diagnostics",
        ):
            _require_source(Path(provenance[key]), provenance[f"{key}_sha256"])
        detail["_result_path"] = str(result_path)
        detail["_result_sha256"] = entry["result_sha256"]
        detail["_diagnostics_path"] = str(diagnostics_path)
        detail["_diagnostics_sha256"] = entry["diagnostics_sha256"]
        output[cell] = detail
    if set(output) != expected_cells:
        raise ValueError("bias-component population changed")
    for condition in CONDITIONS:
        for variant in VARIANTS:
            count = sum(
                output[(condition, seed)]["variant_seed_gates"][variant]
                for seed in SEEDS
            )
            if count != EXPECTED_COUNTS[condition][variant]:
                raise ValueError("bias-component pass counts changed")
    return [output[cell] for cell in sorted(output)]


def _detail_summary(detail: Mapping[str, Any]) -> dict[str, float]:
    metrics = {
        "validity": 1.0,
        "centered_seed_pass": float(detail["variant_seed_gates"]["centered"]),
        "mean_plus_seed_pass": float(detail["variant_seed_gates"]["mean_plus"]),
        "full_plus_seed_pass": float(detail["variant_seed_gates"]["full_plus"]),
        "full_minus_seed_pass": float(detail["variant_seed_gates"]["full_minus"]),
        "mean_sufficiency_seed_pattern": float(
            detail["variant_seed_gates"]["centered"]
            and not detail["variant_seed_gates"]["mean_plus"]
        ),
    }
    for regime in REGIMES:
        regime_detail = detail["regimes"][regime]
        metrics[f"{regime}_sign_disagreement"] = float(
            regime_detail["full_plus_vs_full_minus_predicted_bin_disagreement_fraction"]
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


def build_bias_component_causal_decomposition_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/"
        "2026-08-10_tinyllm-bias-component-causal-decomposition.md"
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
            "name": "TinyLLM deterministic sensor-bias causal sufficiency",
            "category": "mechanistic_interpretability",
            "description": (
                "An exact no-fit decomposition of one frozen selected-dose "
                "biased sensor-error draw into centered, mean-only, full, and "
                "sign-reversed components."
            ),
            "question": (
                "Is the persistent positive lab-frame mean sufficient to cause "
                "the biased-law natural-utility failure?"
            ),
            "prediction": (
                "Centered error passes at least four of five seeds in both "
                "arms, while the positive mean alone passes fewer than four "
                "of five in both arms."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": (
                "confirmed_deterministic_positive_mean_sufficient"
            ),
            "evidence_count": 10,
            "direct_experiment_count": 10,
            "direct_quality_experiment_count": 10,
            "integrity_valid_experiment_count": 10,
            "fixed_checkpoint_seed_count": 5,
            "stored_system_count": 10,
            "evidence_role": EVIDENCE_ROLE,
            "power_profile": "two_arm_five_seed_frozen_component_intervention",
            "tested_scope": (
                "ten retained d8/N3 calibrated systems, one frozen selected-dose "
                "draw, exact algebraic components, and two declared shifts"
            ),
            "subclaims": {
                "centered_stochastic_utility": "supported_five_of_five_both_arms",
                "positive_mean_sufficiency": (
                    "supported_analytic_two_of_five_learned_three_of_five_pass"
                ),
                "source_full_positive_failure": (
                    "reproduced_analytic_one_of_five_learned_three_of_five"
                ),
                "sign_reversal_recovery": "supported_four_of_five_both_arms",
                "sign_mechanism": SIGN_CLASSIFICATION,
                "measurement_robustness_equals_group_closure": "rejected",
                "same_scope_retraining": "not_licensed",
            },
            "tags": [
                "tinyllm",
                "sensor-bias",
                "causal-decomposition",
                "calibration",
                "directional-defect",
                "frozen-checkpoint",
                "no-fit",
                "preregistered",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "Centered noise passed 5/5 seeds in both arms, positive "
                "mean-only passed only 2/5 analytic and 3/5 learned systems, "
                "and the sign-reversed full law recovered 4/5 in both arms."
            ),
            "confidence": 0.99,
            "confidence_assessment": (
                "high_confidence_preregistered_frozen_causal_component_result"
            ),
            "num_direct_experiments": 10,
            "num_direct_quality_experiments": 10,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "population": campaign["population"],
                "component_contract": campaign["component_contract"],
                "system_summaries": summaries,
            },
            "key_insights": [
                "Centered stochastic error is utility-valid in every retained system at sigma=0.03125.",
                "The lower-energy deterministic positive mean is sufficient to reproduce the arm-level failure.",
                "Reversing the mean restores both population gates, identifying a signed calibration vulnerability.",
                "The failure is concentrated in composition exact-bin accuracy rather than a general loss of extrapolation utility.",
                "Observed group closure and natural measurement robustness are separate system contracts."
            ],
            "suggested_hypotheses": [
                "Observed-reference recentering can remove the signed mean defect without retraining TinyLLM.",
                "A wrong-sign recentering control will reproduce or amplify the composition exact-bin failure.",
                "The learned and analytic front ends share the same vulnerable sensor/readout coordinate despite different parameterizations."
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
            "component_contract_sha256": COMPONENT_CONTRACT_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "source_dvc_root": SOURCE_DVC_ROOT,
            "source_lakefs_commit": SOURCE_LAKEFS_COMMIT,
        },
    }


def build_bias_component_causal_decomposition_experiment_results(
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
                primary_metric=metrics["mean_sufficiency_seed_pattern"],
                model_architecture=[8, 512],
                model_parameters=50_964_992,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["model_checkpoint"],
                observations=[
                    f"Condition: {detail['condition']}; seed: {detail['seed']}.",
                    "Exact frozen components reused the selected source draw without new randomness.",
                    "All integrity, reconstruction, and source-replay contracts passed.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[
                    "Positive and negative matched-energy full laws have strongly asymmetric utility."
                ],
                timestamp=completed,
            )
        )
    return output


def store_bias_component_causal_decomposition_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_bias_component_causal_decomposition_meta_hypothesis(results_path)
    experiments = build_bias_component_causal_decomposition_experiment_results(
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
            raise RuntimeError("bias-component ChromaDB read-back failed")
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
    "build_bias_component_causal_decomposition_experiment_results",
    "build_bias_component_causal_decomposition_meta_hypothesis",
    "store_bias_component_causal_decomposition_meta_hypothesis",
]
