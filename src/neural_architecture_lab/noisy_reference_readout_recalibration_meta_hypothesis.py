"""Build and store TinyLLM noisy-reference readout-recalibration evidence."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-noisy-reference-readout-recalibration.v1"
HYPOTHESIS_ID = "tinyllm-noisy-reference-readout-recalibration-v1"
EVIDENCE_ROLE = "preregistered_mixed_pedigree_partial_analytic_exposure"
IMPLEMENTATION_SHA256 = (
    "05a6eda8d84535fdeda3f232069b64b079724fc1b5a5a549078c6cfd03b53e94"
)
CAMPAIGN_SHA256 = (
    "4f068487264e541fe65da955b7ef820047aab63455045b0631189dcf6da0331f"
)
SOURCE_ORIENTATION_SHA256 = (
    "876af062f9d0bdd365e2f1ffbb959d9ff2e5e4e277321b510bbe6a956456877f"
)
SOURCE_ORIENTATION_IMPLEMENTATION_SHA256 = (
    "990975365c7f0c468c3895029c1a87a98fa14d5a28853a1fc445070769218c70"
)
SOURCE_NOISE_SHA256 = (
    "b8d959169f2f249d635037a362c70522e514ace13d3bf656a91bdaea99ef43e7"
)
SOURCE_CALIBRATED_SHA256 = (
    "80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501"
)
SOURCE_CALIBRATED_IMPLEMENTATION_SHA256 = (
    "73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
SEEDS = (7, 17, 29, 41, 53)
PRIMARY_REGIMES = ("composition", "extrapolation")
EXPECTED_COUNTS = {
    "analytic_calibrated": {
        "frozen_noisy_complete_pass_count": 0,
        "linear_primary_pass_count": 3,
        "scalar_primary_pass_count": 3,
    },
    "learned_calibrated_equivariant": {
        "frozen_noisy_complete_pass_count": 2,
        "linear_primary_pass_count": 5,
        "scalar_primary_pass_count": 4,
    },
}
ARM_EVIDENCE_ROLES = {
    "analytic_calibrated": "post_outcome_corrective_replication_evidence",
    "learned_calibrated_equivariant": "preregistered_unseen_arm_evidence",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, Iterable):
        return all(_finite_numbers(item) for item in value)
    return True


def _utility_pass(
    evaluations: Mapping[str, Mapping[str, float]],
    clean_baseline: Mapping[str, float],
    ceiling: float,
) -> bool:
    return all(
        float(clean_baseline[regime])
        - float(evaluations[regime]["exact_bin_accuracy"])
        <= ceiling
        for regime in PRIMARY_REGIMES
    )


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    config = value.get("configuration", {})
    summary = value.get("summary", {})
    aggregates = value.get("aggregates", {})
    provenance = value.get("provenance", {})
    corruption = value.get("corruption_contracts", {})
    orientation_path = Path(provenance.get("orientation_campaign", ""))
    arms = aggregates.get("arms", {})
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or tuple(int(seed) for seed in config.get("seeds", ())) != SEEDS
        or float(config.get("noise_sigma", -1)) != 0.035
        or int(config.get("required_seed_passes", -1)) != 4
        or float(config.get("accuracy_loss_ceiling", -1)) != 0.03
        or float(config.get("shuffled_accuracy_ceiling", -1)) != 0.20
        or config.get("partial_outcome_relaunch") is not True
        or int(summary.get("requested", -1)) != 10
        or int(summary.get("completed", -1)) != 10
        or int(summary.get("failed", -1)) != 0
        or int(summary.get("trained_models", -1)) != 0
        or int(summary.get("trained_frontends", -1)) != 0
        or int(summary.get("new_representation_probes", -1)) != 0
        or int(summary.get("fitted_linear_readouts", -1)) != 30
        or int(summary.get("fitted_scalar_calibrators", -1)) != 20
        or aggregates.get("valid") is not True
        or aggregates.get("classification") != "arm_stratified_readout_repair"
        or int(aggregates.get("required_seed_passes", -1)) != 4
        or aggregates.get("arm_evidence_roles") != ARM_EVIDENCE_ROLES
        or any(
            int(arms.get(condition, {}).get(name, -1)) != expected
            for condition, counts in EXPECTED_COUNTS.items()
            for name, expected in counts.items()
        )
        or any(
            int(arms.get(condition, {}).get(name, -1)) != 5
            for condition in CONDITIONS
            for name in (
                "clean_linear_control_pass_count",
                "linear_clean_compatibility_pass_count",
                "shuffled_negative_control_pass_count",
            )
        )
        or provenance.get("orientation_campaign_sha256")
        != SOURCE_ORIENTATION_SHA256
        or provenance.get("orientation_implementation_sha256")
        != SOURCE_ORIENTATION_IMPLEMENTATION_SHA256
        or provenance.get("orientation_noise_sha256") != SOURCE_NOISE_SHA256
        or provenance.get("calibrated_campaign_sha256")
        != SOURCE_CALIBRATED_SHA256
        or provenance.get("calibrated_implementation_sha256")
        != SOURCE_CALIBRATED_IMPLEMENTATION_SHA256
        or not orientation_path.is_file()
        or _sha256(orientation_path) != SOURCE_ORIENTATION_SHA256
        or any(corruption.get(name) is not True for name in (
            "dataset_hash_replay_contract",
            "formula_replay_contract",
            "noise_array_replay_contract",
            "pair_shared_noise_contract",
            "unit_norm_contract",
        ))
        or not _finite_numbers(value)
    ):
        raise ValueError(f"invalid noisy-reference readout campaign {path}")
    return value


def _computed_gates(detail: Mapping[str, Any]) -> dict[str, bool]:
    clean_baseline = detail["clean_baseline_accuracy"]
    ceiling = float(detail["configuration"]["accuracy_loss_ceiling"])
    noisy_fit = detail["fits"]["noisy"]
    clean_fit = detail["fits"]["clean"]
    frozen = detail["frozen"]
    shuffled_ceiling = float(detail["configuration"]["shuffled_accuracy_ceiling"])
    noisy_linear_repair = _utility_pass(
        noisy_fit["evaluations"]["noisy"]["linear"], clean_baseline, ceiling
    )
    noisy_linear_clean = _utility_pass(
        noisy_fit["evaluations"]["clean"]["linear"], clean_baseline, ceiling
    )
    noisy_scalar_repair = _utility_pass(
        noisy_fit["evaluations"]["noisy"]["scalar"], clean_baseline, ceiling
    )
    noisy_scalar_clean = _utility_pass(
        noisy_fit["evaluations"]["clean"]["scalar"], clean_baseline, ceiling
    )
    return {
        "clean_linear_control": _utility_pass(
            clean_fit["evaluations"]["clean"]["linear"], clean_baseline, ceiling
        ),
        "noisy_linear_repair": noisy_linear_repair,
        "noisy_linear_clean_compatibility": noisy_linear_clean,
        "linear_primary": noisy_linear_repair and noisy_linear_clean,
        "noisy_scalar_repair": noisy_scalar_repair,
        "noisy_scalar_clean_compatibility": noisy_scalar_clean,
        "scalar_primary": noisy_scalar_repair and noisy_scalar_clean,
        "target_shuffled_negative_control": all(
            float(detail["target_shuffled"]["evaluations"][regime][
                "exact_bin_accuracy"
            ])
            <= shuffled_ceiling
            for regime in PRIMARY_REGIMES
        ),
        "frozen_noisy_complete": _utility_pass(
            frozen["noisy"], clean_baseline, ceiling
        ),
    }


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    indexed = {
        (entry["condition"], int(entry["seed"])): entry
        for entry in campaign.get("results", [])
    }
    expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    if set(indexed) != expected:
        raise ValueError(f"invalid noisy-reference readout result index {path}")

    output = []
    fingerprints: set[str] = set()
    for condition in CONDITIONS:
        for seed in SEEDS:
            entry = indexed[(condition, seed)]
            result_path = Path(entry["path"])
            readouts_path = Path(entry["readouts"])
            detail = json.loads(result_path.read_text(encoding="utf-8"))
            provenance = detail.get("provenance", {})
            orientation_path = Path(provenance.get("orientation_result", ""))
            source_path = Path(provenance.get("source_result", ""))
            model_path = Path(provenance.get("model_checkpoint", ""))
            frontend_path = Path(provenance.get("frontend_checkpoint", ""))
            source = (
                json.loads(source_path.read_text(encoding="utf-8"))
                if source_path.is_file()
                else {}
            )
            computed = _computed_gates(detail)
            required_true = (
                "clean_linear_control",
                "finite_numerical_contract",
                "inherited_representation_contract",
                "noisy_linear_clean_compatibility",
                "source_metric_replay_contract",
                "system_state_unchanged_contract",
                "target_shuffled_negative_control",
                "validity_contract",
            )
            if (
                _sha256(result_path) != entry.get("result_sha256")
                or _sha256(readouts_path) != entry.get("readouts_sha256")
                or detail.get("artifacts", {}).get("readouts") != str(readouts_path)
                or detail.get("artifacts", {}).get("readouts_sha256")
                != entry.get("readouts_sha256")
                or detail.get("schema_version") != EXPERIMENT_SCHEMA
                or detail.get("hypothesis_id") != HYPOTHESIS_ID
                or detail.get("status") != "completed"
                or detail.get("evidence_role") != EVIDENCE_ROLE
                or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
                or detail.get("experiment_id") != entry.get("experiment_id")
                or detail.get("condition") != condition
                or int(detail.get("seed", -1)) != seed
                or detail.get("scientific_fingerprint")
                != entry.get("scientific_fingerprint")
                or entry.get("gates") != detail.get("gates")
                or any(detail.get("gates", {}).get(name) is not True for name in required_true)
                or any(
                    detail.get("gates", {}).get(name) != computed[name]
                    for name in (
                        "clean_linear_control",
                        "noisy_linear_repair",
                        "noisy_linear_clean_compatibility",
                        "linear_primary",
                        "noisy_scalar_repair",
                        "noisy_scalar_clean_compatibility",
                        "scalar_primary",
                        "target_shuffled_negative_control",
                    )
                )
                or float(detail.get("maximum_source_metric_replay_error", math.inf))
                > float(detail["configuration"]["replay_tolerance"])
                or detail.get("source_noise_sha256") != SOURCE_NOISE_SHA256
                or not orientation_path.is_file()
                or _sha256(orientation_path)
                != provenance.get("orientation_result_sha256")
                or not source_path.is_file()
                or _sha256(source_path) != provenance.get("source_result_sha256")
                or source.get("condition") != condition
                or int(source.get("seed", -1)) != seed
                or not model_path.is_file()
                or _sha256(model_path) != provenance.get("model_checkpoint_sha256")
                or not frontend_path.is_file()
                or _sha256(frontend_path)
                != provenance.get("frontend_checkpoint_sha256")
                or not _finite_numbers(detail)
            ):
                raise ValueError(f"invalid noisy-reference readout result {result_path}")
            fingerprint = detail["scientific_fingerprint"]
            if fingerprint in fingerprints:
                raise ValueError(f"duplicate noisy-reference fingerprint {result_path}")
            fingerprints.add(fingerprint)
            detail["_model_parameters"] = int(source["model_parameters"])
            detail["_frozen_noisy_complete"] = computed["frozen_noisy_complete"]
            output.append(detail)

    for condition in CONDITIONS:
        rows = [detail for detail in output if detail["condition"] == condition]
        arm = campaign["aggregates"]["arms"][condition]
        computed_counts = {
            "frozen_noisy_complete_pass_count": sum(
                int(detail["_frozen_noisy_complete"]) for detail in rows
            ),
            "linear_primary_pass_count": sum(
                int(detail["gates"]["linear_primary"]) for detail in rows
            ),
            "scalar_primary_pass_count": sum(
                int(detail["gates"]["scalar_primary"]) for detail in rows
            ),
        }
        if any(int(arm[name]) != count for name, count in computed_counts.items()):
            raise ValueError(f"inconsistent noisy-reference arm counts {path}")
    return output


def _detail_summary(detail: Mapping[str, Any]) -> dict[str, Any]:
    noisy = detail["fits"]["noisy"]
    linear = noisy["evaluations"]["noisy"]["linear"]
    scalar = noisy["evaluations"]["noisy"]["scalar"]
    return {
        "maximum_source_metric_replay_error": float(
            detail["maximum_source_metric_replay_error"]
        ),
        "frozen_noisy_complete_pass": bool(detail["_frozen_noisy_complete"]),
        "linear_primary_pass": bool(detail["gates"]["linear_primary"]),
        "scalar_primary_pass": bool(detail["gates"]["scalar_primary"]),
        "linear_noisy_accuracy_drop_mean": mean(
            float(linear[regime]["accuracy_loss_from_frozen_clean"])
            for regime in PRIMARY_REGIMES
        ),
        "linear_noisy_accuracy_mean": mean(
            float(linear[regime]["exact_bin_accuracy"])
            for regime in PRIMARY_REGIMES
        ),
        "linear_noisy_correlation_minimum": min(
            float(linear[regime]["expected_cosine_correlation"])
            for regime in PRIMARY_REGIMES
        ),
        "linear_relative_weight_displacement": float(
            noisy["linear_fit"]["relative_weight_displacement"]
        ),
        "scalar_noisy_accuracy_drop_mean": mean(
            float(scalar[regime]["accuracy_loss_from_frozen_clean"])
            for regime in PRIMARY_REGIMES
        ),
        "scalar_noisy_correlation_minimum": min(
            float(scalar[regime]["expected_cosine_correlation"])
            for regime in PRIMARY_REGIMES
        ),
        "scalar_slope": float(noisy["scalar_fit"]["slope"]),
        "scalar_intercept": float(noisy["scalar_fit"]["intercept"]),
        "target_shuffled_accuracy_maximum": max(
            float(detail["target_shuffled"]["evaluations"][regime][
                "exact_bin_accuracy"
            ])
            for regime in PRIMARY_REGIMES
        ),
    }


def _campaign_summary(details: list[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition in CONDITIONS:
        summaries = [
            _detail_summary(detail)
            for detail in details
            if detail["condition"] == condition
        ]
        output[condition] = {
            name: mean(float(summary[name]) for summary in summaries)
            for name in (
                "linear_noisy_accuracy_drop_mean",
                "linear_noisy_accuracy_mean",
                "linear_noisy_correlation_minimum",
                "linear_relative_weight_displacement",
                "scalar_noisy_accuracy_drop_mean",
                "scalar_noisy_correlation_minimum",
                "scalar_slope",
                "scalar_intercept",
            )
        }
        output[condition]["maximum_target_shuffled_accuracy"] = max(
            float(summary["target_shuffled_accuracy_maximum"])
            for summary in summaries
        )
    output["maximum_source_metric_replay_error"] = max(
        float(detail["maximum_source_metric_replay_error"]) for detail in details
    )
    return output


def build_noisy_reference_readout_recalibration_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-noisy-reference-readout-recalibration.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    direct_tests = [
        {
            "experiment_id": detail["experiment_id"],
            "condition": detail["condition"],
            "seed": detail["seed"],
            "evidence_role": ARM_EVIDENCE_ROLES[detail["condition"]],
            "scientific_fingerprint": detail["scientific_fingerprint"],
            "provenance": detail["provenance"],
            **_detail_summary(detail),
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM noisy-reference readout recalibration",
            "category": "mechanistic_interpretability",
            "description": (
                "A frozen-system intervention refits only the final task interface "
                "at the first registered orientation-reference error."
            ),
            "question": (
                "Can an existing-capacity linear head or near-identity scalar "
                "calibrator recover clean and noisy utility without changing the "
                "front end or TinyLLM representation?"
            ),
            "prediction": (
                "A universal readout-calibration limit requires at least four of "
                "five seeds in both analytic and learned structured arms to pass "
                "the simultaneous clean/noisy composition/extrapolation gate."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "partially_supported_arm_stratified_readout_repair",
            "evidence_count": len(direct_tests),
            "direct_experiment_count": len(direct_tests),
            "independent_checkpoint_count": 5,
            "power_profile": "five_seed_preregistered_mixed_pedigree_frozen_readout_test",
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "ten retained d8/N3 calibrated checkpoints at 0.035-radian "
                "orientation-reference noise; composition and extrapolation"
            ),
            "subclaims": {
                "universal_both_arm_readout_repair": "rejected",
                "learned_linear_readout_repair": "supported_five_of_five_preregistered_unseen",
                "learned_scalar_interval_repair": "supported_four_of_five_preregistered_unseen",
                "analytic_linear_readout_repair": "not_supported_three_of_five_corrective",
                "analytic_scalar_interval_repair": "not_supported_three_of_five_corrective",
                "clean_compatibility": "supported_five_of_five_both_arms",
                "target_shuffle_specificity": "supported_five_of_five_both_arms",
                "full_model_retraining_licensed": "no",
            },
            "tags": [
                "tinyllm",
                "calibration-noise",
                "readout-recalibration",
                "frozen-checkpoint",
                "equivariant-front-end",
                "mixed-evidence-pedigree",
                "causal-intervention",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "A readout shared across checkpoints.",
                "A bounded nonlinear readout ceiling for the analytic arm.",
                "Repeated reference acquisition or a real-instrument noise model.",
                "Natural-language behavior or architecture-population prevalence.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The learned equivariant arm is causally repaired by a noisy-fitted "
                "linear head in five of five seeds and a near-identity scalar "
                "calibrator in four of five. The analytic arm reaches only three "
                "of five in either family, so the universal both-arm claim fails."
            ),
            "confidence": 0.96,
            "confidence_assessment": (
                "preregistered_unseen_learned_arm_with_corrective_analytic_comparison"
            ),
            "independent_checkpoint_count": 5,
            "num_direct_experiments": 10,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "checkpoint_means": _campaign_summary(details),
            },
            "key_insights": [
                "Learned-arm task loss at two degrees is an interface defect: a frozen-representation linear replacement passes every checkpoint.",
                "A near-identity scalar calibration passes four learned checkpoints, so the conclusion does not depend only on a large 6144-parameter reweighting.",
                "The analytic arm remains below the population threshold despite high nonlinear cosine decodability, separating decodability from declared-interface sufficiency.",
                "All clean-compatibility and shuffled-target controls pass, while no TinyLLM or front end is retrained.",
                "Only the learned-arm outcome is preregistered unseen evidence; the analytic arm is corrective replication after partial exposure."
            ],
            "suggested_hypotheses": [
                "Repeated independent orientation-reference observations can restore both arms through analytic circular averaging without changing their readouts.",
                "A single preregistered bounded nonlinear readout can determine whether analytic-arm information is usable beyond the failed linear and affine interfaces.",
                "Near-identity interval calibration trained on one held-out cohort can transfer to a fresh cohort in the learned arm."
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": direct_tests},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "source_orientation_campaign_sha256": SOURCE_ORIENTATION_SHA256,
            "source_orientation_implementation_sha256": (
                SOURCE_ORIENTATION_IMPLEMENTATION_SHA256
            ),
            "source_noise_sha256": SOURCE_NOISE_SHA256,
            "source_calibrated_campaign_sha256": SOURCE_CALIBRATED_SHA256,
            "source_calibrated_implementation_sha256": (
                SOURCE_CALIBRATED_IMPLEMENTATION_SHA256
            ),
        },
    }


def build_noisy_reference_readout_recalibration_experiment_results(
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
        metrics = {
            name: float(value) for name, value in summary.items()
        }
        arm_role = ARM_EVIDENCE_ROLES[detail["condition"]]
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(summary["linear_primary_pass"]),
                model_architecture=[8, int(detail["_model_parameters"])],
                model_parameters=int(detail["_model_parameters"]),
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["model_checkpoint"],
                observations=[
                    f"Condition: {detail['condition']}.",
                    f"Evidence role: {arm_role}.",
                    "TinyLLM and the structured front end remained frozen.",
                    "Only detached linear and affine-scalar task interfaces were fitted.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=(
                    [
                        "The analytic arm misses the population gate despite retained nonlinear cosine decodability."
                    ]
                    if detail["condition"] == "analytic_calibrated"
                    else []
                ),
                timestamp=completed,
            )
        )
    return output


def store_noisy_reference_readout_recalibration_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_noisy_reference_readout_recalibration_meta_hypothesis(
        results_path
    )
    experiments = build_noisy_reference_readout_recalibration_experiment_results(
        record, results_path
    )
    storage: dict[str, Any] = {
        "aggregate_path": str(output_path),
        "experiment_ids": [item.experiment_id for item in experiments],
    }
    if chromadb_path is not None:
        from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger

        storage_root = chromadb_path.parent
        logger = StandardizedLogger(
            LoggingConfig(
                project_name="structure_net_meta_hypotheses",
                queue_dir=str(storage_root / "experiment_queue"),
                sent_dir=str(storage_root / "experiment_sent"),
                rejected_dir=str(storage_root / "experiment_rejected"),
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
            raise RuntimeError("noisy-reference readout ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": len(experiment_readback["ids"]),
        }
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return record


__all__ = [
    "build_noisy_reference_readout_recalibration_experiment_results",
    "build_noisy_reference_readout_recalibration_meta_hypothesis",
    "store_noisy_reference_readout_recalibration_meta_hypothesis",
]
