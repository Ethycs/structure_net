"""Build and store TinyLLM calibration readout-decomposition evidence."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-calibration-readout-decomposition.v1"
HYPOTHESIS_ID = "tinyllm-calibration-readout-decomposition-v1"
EVIDENCE_ROLE = (
    "preregistered_existing_checkpoint_post_outcome_mechanistic_diagnostic"
)
IMPLEMENTATION_SHA256 = (
    "0649a40c17384266f360daf33b68519e8f6d068aeec03893cfe9229f4f8d222d"
)
CAMPAIGN_SHA256 = (
    "833392ad7956ddcf715a20211431586f371edee280f167bf5a4fa51437cdc6c6"
)
TREE_MANIFEST_SHA256 = (
    "ec2a7338655b3cb3e02ab3808665d3b0947318abf06b0a17e0fbedab6798cc1c"
)
SOURCE_CAMPAIGN_SHA256 = (
    "170d99553058ab544d95d6abf4d26ea1062bfe1b79fda2deefc531dfaebd0f6e"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "f273abf83231e4fc37583154c24158c81bba15a29e11ca3f96daf0b579be0f70"
)
SOURCE_MANIFEST_SHA256 = (
    "23598aaef5a0d16825ff9a928de57857e7bd58b10feab8853739448acfd983fc"
)
SOURCE_NOISE_SHA256 = (
    "1896c852644f724f5a2d214d5e69380eef3e46abb2e7b295eb0d4bace51a0c82"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
SEEDS = (7, 17, 29, 41, 53)
LEVELS = (0.0, 0.05, 0.10, 0.20)
REGIMES = ("composition", "extrapolation")
READOUTS = ("model_argmax", "model_posterior_mean", "frontend", "observer")
PATCHES = ("observer", "target_oracle", "flipped", "shuffled", "kernel")
EXPECTED_PRIMARY_COUNTS = {
    "readouts": {
        name: {condition: 0 for condition in CONDITIONS} for name in READOUTS
    },
    "patches": {
        "observer": {condition: 0 for condition in CONDITIONS},
        "target_oracle": {condition: 5 for condition in CONDITIONS},
        "flipped": {condition: 0 for condition in CONDITIONS},
        "shuffled": {condition: 0 for condition in CONDITIONS},
        "kernel": {condition: 0 for condition in CONDITIONS},
    },
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


def _noise_key(level: float) -> str:
    return f"sigma_{level:.2f}".replace(".", "p")


def _evaluation_key(regime: str, level: float) -> str:
    return f"{regime}__{_noise_key(level)}"


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    summary = value.get("summary", {})
    aggregates = value.get("aggregates", {})
    provenance = value.get("provenance", {})
    configuration = value.get("configuration", {})
    source_campaign = Path(provenance.get("source_campaign", ""))
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or provenance.get("source_campaign_sha256") != SOURCE_CAMPAIGN_SHA256
        or provenance.get("source_implementation_sha256")
        != SOURCE_IMPLEMENTATION_SHA256
        or provenance.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256
        or provenance.get("source_noise_sha256") != SOURCE_NOISE_SHA256
        or not source_campaign.is_file()
        or _sha256(source_campaign) != SOURCE_CAMPAIGN_SHA256
        or tuple(configuration.get("conditions", ())) != CONDITIONS
        or tuple(configuration.get("seeds", ())) != SEEDS
        or tuple(float(item) for item in configuration.get("levels", ())) != LEVELS
        or int(configuration.get("required_seed_passes", -1)) != 4
        or int(configuration.get("control_pass_ceiling", -1)) != 1
        or float(configuration.get("task_accuracy_drop_ceiling", -1)) != 0.03
        or int(summary.get("requested", -1)) != 10
        or int(summary.get("completed", -1)) != 10
        or int(summary.get("failed", -1)) != 0
        or int(summary.get("excluded", -1)) != 0
        or int(summary.get("trained_models", -1)) != 0
        or int(summary.get("trained_frontends", -1)) != 0
        or int(summary.get("trained_task_heads", -1)) != 0
        or int(summary.get("fitted_diagnostic_observers", -1)) != 10
        or aggregates.get("valid") is not True
        or aggregates.get("classification")
        != "reference_coordinate_precision_limited"
        or aggregates.get("primary_hypothesis_pass") is not False
        or aggregates.get("controls_specific") is not True
        or aggregates.get("pass_counts_at_sigma_0p20")
        != EXPECTED_PRIMARY_COUNTS
        or int(aggregates.get("required_seed_passes", -1)) != 4
    ):
        raise ValueError(f"invalid calibration readout-decomposition campaign {path}")
    return value


def _seed_passes(detail: Mapping[str, Any], level: float) -> dict[str, dict[str, bool]]:
    return {
        "readouts": {
            name: all(
                bool(
                    detail["evaluations"][_evaluation_key(regime, level)][
                        "readouts"
                    ][name]["utility_pass"]
                )
                for regime in REGIMES
            )
            for name in READOUTS
        },
        "patches": {
            name: all(
                bool(
                    detail["evaluations"][_evaluation_key(regime, level)][
                        "patches"
                    ][name]["utility_pass"]
                )
                for regime in REGIMES
            )
            for name in PATCHES
        },
    }


def _counts_by_level(details: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        section: {
            name: {
                condition: {
                    _noise_key(level): sum(
                        int(_seed_passes(detail, level)[section][name])
                        for detail in details
                        if detail["condition"] == condition
                    )
                    for level in LEVELS
                }
                for condition in CONDITIONS
            }
            for name in (READOUTS if section == "readouts" else PATCHES)
        }
        for section in ("readouts", "patches")
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
        raise ValueError(f"invalid readout-decomposition result index {path}")

    output = []
    fingerprints: set[str] = set()
    expected_evaluations = {
        _evaluation_key(regime, level) for regime in REGIMES for level in LEVELS
    }
    for condition in CONDITIONS:
        for seed in SEEDS:
            entry = indexed[(condition, seed)]
            result_path = Path(entry["path"])
            observer_path = Path(entry["observer"])
            detail = json.loads(result_path.read_text(encoding="utf-8"))
            source = detail.get("source", {})
            source_result_path = Path(source.get("result", ""))
            source_degradation_path = Path(
                detail.get("source_degradation_result", "")
            )
            model_path = Path(source.get("model", ""))
            frontend_path = Path(source.get("frontend", ""))
            source_result = (
                json.loads(source_result_path.read_text(encoding="utf-8"))
                if source_result_path.is_file()
                else {}
            )
            computed_primary = _seed_passes(detail, 0.20)
            if (
                _sha256(result_path) != entry.get("result_sha256")
                or not observer_path.is_file()
                or _sha256(observer_path) != entry.get("observer_sha256")
                or detail.get("artifacts", {}).get("observer")
                != str(observer_path)
                or detail.get("artifacts", {}).get("observer_sha256")
                != entry.get("observer_sha256")
                or detail.get("schema_version") != EXPERIMENT_SCHEMA
                or detail.get("hypothesis_id") != HYPOTHESIS_ID
                or detail.get("status") != "completed"
                or detail.get("evidence_role") != EVIDENCE_ROLE
                or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
                or detail.get("experiment_id") != entry.get("experiment_id")
                or detail.get("condition") != condition
                or int(detail.get("seed", -1)) != seed
                or set(detail.get("evaluations", {})) != expected_evaluations
                or detail.get("clean_replay", {}).get("pass") is not True
                or float(
                    detail.get("clean_replay", {}).get(
                        "maximum_absolute_error", math.inf
                    )
                )
                > 1e-5
                or detail.get("finite_pass") is not True
                or detail.get("source_state_pass") is not True
                or detail.get("input_target_identity_pass") is not True
                or detail.get("valid") is not True
                or detail.get("seed_passes_at_0p20") != computed_primary
                or not computed_primary["patches"]["target_oracle"]
                or any(computed_primary["readouts"].values())
                or any(
                    computed_primary["patches"][name]
                    for name in ("observer", "flipped", "shuffled", "kernel")
                )
                or not source_degradation_path.is_file()
                or _sha256(source_degradation_path)
                != detail.get("source_degradation_result_sha256")
                or not source_result_path.is_file()
                or _sha256(source_result_path) != source.get("result_sha256")
                or source_result.get("condition") != condition
                or int(source_result.get("seed", -1)) != seed
                or not model_path.is_file()
                or _sha256(model_path) != source.get("model_sha256")
                or not frontend_path.is_file()
                or _sha256(frontend_path) != source.get("frontend_sha256")
                or not _finite_numbers(detail)
            ):
                raise ValueError(
                    f"invalid calibration readout-decomposition result {result_path}"
                )
            fingerprint = detail["scientific_fingerprint"]
            if fingerprint in fingerprints:
                raise ValueError(
                    f"duplicate readout-decomposition fingerprint {result_path}"
                )
            fingerprints.add(fingerprint)
            detail["_source_model_parameters"] = int(source_result["model_parameters"])
            output.append(detail)

    if _counts_by_level(output) != campaign["aggregates"]["pass_counts_by_level"]:
        raise ValueError(f"inconsistent readout-decomposition level counts {path}")
    return output


def _level_values(
    detail: Mapping[str, Any], level: float, section: str, name: str
) -> list[Mapping[str, Any]]:
    return [
        detail["evaluations"][_evaluation_key(regime, level)][section][name]
        for regime in REGIMES
    ]


def _detail_summary(detail: Mapping[str, Any]) -> dict[str, Any]:
    observer = _level_values(detail, 0.20, "readouts", "observer")
    observer_patch = _level_values(detail, 0.20, "patches", "observer")
    oracle_patch = _level_values(detail, 0.20, "patches", "target_oracle")
    diagnostics = [
        detail["evaluations"][_evaluation_key(regime, 0.20)][
            "causal_diagnostics"
        ]["observer"]
        for regime in REGIMES
    ]
    return {
        "clean_replay_maximum_error": float(
            detail["clean_replay"]["maximum_absolute_error"]
        ),
        "observer_accuracy_mean_at_0p20": mean(
            float(value["exact_bin_accuracy"]) for value in observer
        ),
        "observer_accuracy_drop_mean_at_0p20": mean(
            float(value["accuracy_drop_from_own_clean"]) for value in observer
        ),
        "observer_cosine_correlation_mean_at_0p20": mean(
            float(value["cosine_pearson"]) for value in observer
        ),
        "observer_cosine_rmse_mean_at_0p20": mean(
            float(value["cosine_rmse"]) for value in observer
        ),
        "observer_patch_accuracy_mean_at_0p20": mean(
            float(value["exact_bin_accuracy"]) for value in observer_patch
        ),
        "oracle_patch_accuracy_mean_at_0p20": mean(
            float(value["exact_bin_accuracy"]) for value in oracle_patch
        ),
        "observer_patch_post_moment_error_mean_at_0p20": mean(
            float(value["mean_absolute_post_patch_moment_error"])
            for value in diagnostics
        ),
        "observer_patch_norm_mean_at_0p20": mean(
            float(value["mean_patch_norm"]) for value in diagnostics
        ),
        "primary_seed_passes": detail["seed_passes_at_0p20"],
    }


def _campaign_summary(details: list[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = [detail for detail in details if detail["condition"] == condition]
        summaries = [_detail_summary(detail) for detail in rows]
        output[condition] = {
            name: mean(float(summary[name]) for summary in summaries)
            for name in (
                "observer_accuracy_mean_at_0p20",
                "observer_accuracy_drop_mean_at_0p20",
                "observer_cosine_correlation_mean_at_0p20",
                "observer_cosine_rmse_mean_at_0p20",
                "observer_patch_accuracy_mean_at_0p20",
                "oracle_patch_accuracy_mean_at_0p20",
                "observer_patch_post_moment_error_mean_at_0p20",
                "observer_patch_norm_mean_at_0p20",
            )
        }
    output["maximum_clean_replay_error"] = max(
        float(detail["clean_replay"]["maximum_absolute_error"])
        for detail in details
    )
    return output


def build_calibration_readout_decomposition_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-calibration-readout-decomposition.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    direct_tests = [
        {
            "experiment_id": detail["experiment_id"],
            "condition": detail["condition"],
            "seed": detail["seed"],
            "scientific_fingerprint": detail["scientific_fingerprint"],
            "source": detail["source"],
            "observer_fit": detail["observer_fit"],
            **_detail_summary(detail),
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM calibration readout decomposition",
            "category": "mechanistic_interpretability",
            "description": (
                "A post-outcome corrective replay compares frozen posterior, "
                "front-end, observer, and task-gradient readouts under calibration "
                "noise in ten retained TinyLLM systems."
            ),
            "question": (
                "Can a no-fit or one-step observer-based readout recover task utility "
                "while the probe-defined quotient remains present?"
            ),
            "prediction": (
                "A decoder-relation limit requires the frozen observer and its causal "
                "task-gradient patch to pass both arms at sigma 0.20 with controls."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_reference_coordinate_precision_limited"
            ),
            "evidence_count": len(direct_tests),
            "direct_experiment_count": len(direct_tests),
            "independent_checkpoint_count": 5,
            "power_profile": (
                "five_seed_preregistered_design_post_outcome_corrective_replication"
            ),
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "ten retained d8/N3 calibrated checkpoints; source calibration "
                "levels 0--0.20; composition and extrapolation"
            ),
            "subclaims": {
                "posterior_shape_calibration_limited": "rejected_zero_of_five_both_arms",
                "frozen_observer_utility_at_0p20": "rejected_zero_of_five_both_arms",
                "observer_task_gradient_patch_at_0p20": "rejected_zero_of_five_both_arms",
                "target_oracle_patch_at_0p20": "supported_five_of_five_both_arms",
                "flipped_shuffled_kernel_specificity": "supported_zero_of_five_all_controls",
                "reference_coordinate_precision_limited": "supported_both_arms",
                "readout_only_training_licensed": "no",
                "measurement_or_denoising_priority": "yes",
            },
            "tags": [
                "tinyllm",
                "calibration-noise",
                "readout-decomposition",
                "frozen-checkpoint",
                "task-gradient",
                "positive-control",
                "reference-precision",
                "post-outcome-replication",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "Fresh independent confirmation of the selected classification.",
                "Training with noisy calibration or a learned reference denoiser.",
                "A real-instrument uncertainty model.",
                "Natural-language behavior or architecture-population prevalence.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "At sigma 0.20 all deployable readouts and the observer-gradient "
                "patch pass zero of five checkpoints in both arms. The true-target "
                "oracle patch passes five of five with all controls at zero, showing "
                "that the local decoder can use a precise coordinate but the observed "
                "reference-derived estimate is not precise enough."
            ),
            "confidence": 0.97,
            "confidence_assessment": (
                "post_outcome_corrective_replication_with_five_seed_causal_controls"
            ),
            "independent_checkpoint_count": 5,
            "num_direct_experiments": 10,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "checkpoint_means": _campaign_summary(details),
            },
            "key_insights": [
                "High cosine correlation is insufficient: the frozen observer remains near 0.976 at sigma 0.20 but loses roughly 23 task-accuracy points.",
                "Forcing the posterior moment to the observer estimate does not restore task utility despite sub-0.002 post-patch moment error.",
                "The true-target oracle write passes every checkpoint and control, so the frozen local task covector and logits can use a sufficiently precise scalar.",
                "The bottleneck is precision of the reference-derived coordinate, not posterior shape or a missing low-capacity decoder relation.",
                "The first campaign root is quarantined; this record uses the byte-stable corrective replay only.",
            ],
            "suggested_hypotheses": [
                "A prospective observed-reference denoiser can reduce cosine RMSE enough to restore the frozen task gate.",
                "Orientation, speed, and offset field errors contribute separably to reference-coordinate error and should be localized before any model optimization.",
                "If a label-free analytic uncertainty correction cannot approach the oracle, the measurement process rather than TinyLLM capacity is limiting."
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": direct_tests},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "tree_manifest_sha256": TREE_MANIFEST_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "source_implementation_sha256": SOURCE_IMPLEMENTATION_SHA256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_noise_sha256": SOURCE_NOISE_SHA256,
            "quarantined_v1_campaign_sha256": (
                "1dd86dca5ef02bf2f3d5fd206ee972eaac8663e786f29ea8dd57a49a20038e39"
            ),
        },
    }


def build_calibration_readout_decomposition_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        summary = _detail_summary(detail)
        metrics = {
            name: float(value)
            for name, value in summary.items()
            if name != "primary_seed_passes"
        }
        metrics.update(
            {
                f"pass_{section}_{name}": float(value)
                for section, values in summary["primary_seed_passes"].items()
                for name, value in values.items()
            }
        )
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(
                    summary["primary_seed_passes"]["patches"]["observer"]
                ),
                model_architecture=[8, int(detail["_source_model_parameters"])],
                model_parameters=int(detail["_source_model_parameters"]),
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["source"]["model"],
                observations=[
                    f"Condition: {detail['condition']}.",
                    "No TinyLLM, front end, or task head was trained.",
                    "One clean diagnostic observer was fit and frozen across perturbations.",
                    "This is post-outcome corrective replication, not fresh confirmation.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[
                    "The target-oracle task-gradient patch passes while the observer-coordinate patch fails."
                ],
                timestamp=completed,
            )
        )
    return output


def store_calibration_readout_decomposition_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_calibration_readout_decomposition_meta_hypothesis(results_path)
    experiments = build_calibration_readout_decomposition_experiment_results(
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
            raise RuntimeError("calibration readout ChromaDB read-back failed")
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
    "build_calibration_readout_decomposition_experiment_results",
    "build_calibration_readout_decomposition_meta_hypothesis",
    "store_calibration_readout_decomposition_meta_hypothesis",
]
