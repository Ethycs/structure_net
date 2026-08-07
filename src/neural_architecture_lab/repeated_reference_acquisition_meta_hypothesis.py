"""Build and store corrective TinyLLM repeated-reference evidence."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-repeated-reference-acquisition.v1"
HYPOTHESIS_ID = "tinyllm-repeated-reference-acquisition-v1"
EVIDENCE_ROLE = (
    "corrective_outcome_informed_frozen_system_acquisition_intervention"
)
IMPLEMENTATION_SHA256 = (
    "2693f5d42d460a3b43ac2951a895545b7d565a81487b69c9988a36d78da0f40c"
)
CAMPAIGN_SHA256 = (
    "e045849e2c7abb19fc682d826de912760a29c38636c151651f2e81ed363b4832"
)
RESULT_MANIFEST_SHA256 = (
    "4cae9415fb7000d35ea8f5f3e6b4ba5453746245e12194de3d13378c2db96b61"
)
NOISE_SHA256 = (
    "895d398e4ca31611f04f92921e1169704eeba6292813de5b3015415dbd746fec"
)
DENOISER_SHA256 = (
    "6e041c0111667f71f4d3c85f2f62ca29cc2d376f496945a17bd01f96daeb1657"
)
SOURCE_ORIENTATION_SHA256 = (
    "876af062f9d0bdd365e2f1ffbb959d9ff2e5e4e277321b510bbe6a956456877f"
)
SOURCE_CALIBRATED_SHA256 = (
    "80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
RULES = ("analytic_circular_mean", "learned_equivariant")
COUNTS = (1, 4, 16, 64, 256)
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(f"{_sha256(path)}  {path}\n".encode("utf-8"))
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


def _expected_pass(condition: str, seed: int, count: int) -> bool:
    if count <= 16:
        return False
    if count == 256:
        return True
    return condition == "analytic_calibrated" or seed != 53


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    config = value.get("configuration", {})
    summary = value.get("summary", {})
    aggregates = value.get("aggregates", {})
    artifacts = value.get("artifacts", {})
    provenance = value.get("provenance", {})
    noise_path = Path(artifacts.get("noise_arrays", ""))
    denoiser_path = Path(artifacts.get("denoiser", ""))
    source_path = Path(provenance.get("orientation_campaign", ""))
    arms = aggregates.get("arms", {})
    expected_counts = {
        "analytic_calibrated": {"m_001": 0, "m_004": 0, "m_016": 0,
                                "m_064": 5, "m_256": 5},
        "learned_calibrated_equivariant": {
            "m_001": 0, "m_004": 0, "m_016": 0, "m_064": 4, "m_256": 5
        },
    }
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or tuple(config.get("seeds", ())) != SEEDS
        or tuple(config.get("replicate_counts", ())) != COUNTS
        or float(config.get("measurement_sigma_radians", math.nan)) != 0.175
        or float(config.get("accuracy_loss_ceiling", math.nan)) != 0.03
        or int(config.get("required_seed_passes", -1)) != 4
        or config.get("allow_underpowered") is not False
        or summary != {
            "requested": 10, "scheduled": 10, "completed": 10,
            "failed": 0, "excluded": 0, "reused": 0, "retries": 0,
            "trained_models": 0, "trained_frontends": 0,
            "trained_task_heads": 0, "fitted_observers": 0,
            "fitted_probes": 0, "fitted_acquisition_denoisers": 1,
        }
        or aggregates.get("valid") is not True
        or aggregates.get("primary_hypothesis_pass") is not True
        or aggregates.get("classification")
        != "acquisition_variance_causally_sufficient"
        or aggregates.get("controls_pass") is not True
        or aggregates.get("rate_pass") is not True
        or aggregates.get("denoiser_state_unchanged") is not True
        or any(
            arms.get(condition, {}).get("rules", {}).get(rule, {}).get(
                "pass_counts"
            ) != expected_counts[condition]
            for condition in CONDITIONS for rule in RULES
        )
        or any(
            int(arms.get(condition, {}).get("oracle_pass_count", -1)) != 5
            or int(arms.get(condition, {}).get("shuffled_task_pass_count", -1)) != 0
            or int(arms.get(condition, {}).get("single_observation_pass_count", -1)) != 0
            for condition in CONDITIONS
        )
        or provenance.get("orientation_campaign_sha256")
        != SOURCE_ORIENTATION_SHA256
        or provenance.get("calibrated_campaign_sha256")
        != SOURCE_CALIBRATED_SHA256
        or not source_path.is_file()
        or _sha256(source_path) != SOURCE_ORIENTATION_SHA256
        or not noise_path.is_file() or _sha256(noise_path) != NOISE_SHA256
        or artifacts.get("noise_arrays_sha256") != NOISE_SHA256
        or not denoiser_path.is_file() or _sha256(denoiser_path) != DENOISER_SHA256
        or artifacts.get("denoiser_sha256") != DENOISER_SHA256
        or not _finite(value)
    ):
        raise ValueError(f"invalid corrective repeated-reference campaign {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    indexed = {
        (entry["condition"], int(entry["seed"])): entry
        for entry in campaign.get("results", [])
    }
    expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    if set(indexed) != expected:
        raise ValueError(f"invalid corrective result index {path}")
    output: list[dict[str, Any]] = []
    result_paths: list[Path] = []
    fingerprints: set[str] = set()
    for condition in CONDITIONS:
        for seed in SEEDS:
            entry = indexed[(condition, seed)]
            result_path = Path(entry["path"])
            detail = json.loads(result_path.read_text(encoding="utf-8"))
            provenance = detail.get("provenance", {})
            source_path = Path(provenance.get("source_result", ""))
            model_path = Path(provenance.get("model_checkpoint", ""))
            frontend_path = Path(provenance.get("frontend_checkpoint", ""))
            source = json.loads(source_path.read_text(encoding="utf-8"))
            if (
                _sha256(result_path) != entry.get("result_sha256")
                or detail.get("schema_version") != EXPERIMENT_SCHEMA
                or detail.get("hypothesis_id") != HYPOTHESIS_ID
                or detail.get("status") != "completed"
                or detail.get("evidence_role") != EVIDENCE_ROLE
                or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
                or detail.get("condition") != condition
                or int(detail.get("seed", -1)) != seed
                or detail.get("scientific_fingerprint")
                != entry.get("scientific_fingerprint")
                or any(detail.get("gates", {}).get(name) is not True for name in (
                    "acquisition_contract", "clean_replay", "denoiser_contract",
                    "finite", "noise_contract", "single_observation_replay",
                    "source_representation_pass_task_fail", "state_unchanged",
                    "validity",
                ))
                or detail.get("controls", {}).get("exact_reference_oracle", {}).get(
                    "seed_gate"
                ) is not True
                or detail.get("controls", {}).get(
                    "fiber_shuffled_analytic_primary", {}
                ).get("task_gate") is not False
                or any(
                    detail.get("rules", {}).get(rule, {}).get("pass_by_m", {}).get(
                        f"m_{count:03d}"
                    ) is not _expected_pass(condition, seed, count)
                    for rule in RULES for count in COUNTS
                )
                or not source_path.is_file()
                or _sha256(source_path) != provenance.get("source_result_sha256")
                or not model_path.is_file()
                or _sha256(model_path) != provenance.get("model_checkpoint_sha256")
                or not frontend_path.is_file()
                or _sha256(frontend_path) != provenance.get("frontend_checkpoint_sha256")
                or not _finite(detail)
            ):
                raise ValueError(f"invalid corrective repeated-reference result {result_path}")
            fingerprint = detail["scientific_fingerprint"]
            if fingerprint in fingerprints:
                raise ValueError(f"duplicate corrective fingerprint {result_path}")
            fingerprints.add(fingerprint)
            detail["_model_parameters"] = int(source["model_parameters"])
            output.append(detail)
            result_paths.append(result_path)
    if _manifest_sha256(result_paths) != RESULT_MANIFEST_SHA256:
        raise ValueError(f"invalid corrective result manifest {path}")
    return output


def _level(detail: Mapping[str, Any], rule: str, count: int) -> Mapping[str, Any]:
    return next(
        item for item in detail["rules"][rule]["levels"]
        if int(item["replicate_count"]) == count
    )


def _detail_summary(detail: Mapping[str, Any]) -> dict[str, Any]:
    analytic64 = _level(detail, "analytic_circular_mean", 64)
    analytic256 = _level(detail, "analytic_circular_mean", 256)
    learned64 = _level(detail, "learned_equivariant", 64)
    return {
        "analytic_m64_pass": bool(analytic64["seed_gate"]),
        "learned_m64_pass": bool(learned64["seed_gate"]),
        "analytic_m256_pass": bool(analytic256["seed_gate"]),
        "oracle_pass": bool(detail["controls"]["exact_reference_oracle"]["seed_gate"]),
        "shuffled_pass": bool(
            detail["controls"]["fiber_shuffled_analytic_primary"]["task_gate"]
        ),
        "analytic_m64_accuracy_loss_mean": mean(
            float(analytic64["task_recovery"][regime]["accuracy_loss_from_clean"])
            for regime in REGIMES
        ),
        "analytic_m256_accuracy_loss_mean": mean(
            float(analytic256["task_recovery"][regime]["accuracy_loss_from_clean"])
            for regime in REGIMES
        ),
        "analysis_seconds": float(detail["analysis_seconds"]),
    }


def build_repeated_reference_acquisition_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/"
        "2026-08-07_tinyllm-repeated-reference-acquisition-corrective.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    tests = [
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
            "name": "TinyLLM corrective repeated-reference acquisition",
            "category": "mechanistic_interpretability",
            "description": (
                "A second stored acquisition array tests whether coherent reference "
                "averaging repairs ten frozen calibrated TinyLLM systems."
            ),
            "question": (
                "Does reducing observed orientation-reference variance restore the "
                "unchanged task computation under composition and extrapolation?"
            ),
            "prediction": (
                "Analytic and equivariant aggregation must pass at least four of five "
                "checkpoints in both arms while shuffled references fail."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "corrective_outcome_informed_acquisition_repair_supported_not_confirmatory"
            ),
            "evidence_count": 10,
            "direct_experiment_count": 10,
            "independent_checkpoint_count": 5,
            "evidence_role": EVIDENCE_ROLE,
            "power_profile": "five_seed_two_arm_corrective_frozen_acquisition_test",
            "tested_scope": (
                "ten retained d8/N3 calibrated checkpoints, synthetic independent "
                "0.175-radian orientation noise, composition and extrapolation"
            ),
            "subclaims": {
                "confirmatory_status": "not_confirmatory_post_outcome_expansion",
                "population_acquisition_repair_at_m64": "supported_both_rules_both_arms",
                "all_checkpoint_repair_at_m256": "supported_ten_of_ten",
                "analytic_arm_m64": "supported_five_of_five",
                "learned_arm_m64": "supported_four_of_five",
                "learned_aggregator_advantage": "not_supported",
                "inverse_square_rate_law": "supported_both_shifts",
                "fiber_shuffle_specificity": "supported_zero_of_five_both_arms",
                "model_retraining_licensed": "no",
            },
            "tags": [
                "tinyllm", "repeated-reference", "acquisition-precision",
                "frozen-checkpoint", "so2-equivariance", "corrective-evidence",
                "causal-intervention",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The causal acquisition repair is valid and converges with the earlier "
                "preregistered subresult, but this comparison was expanded after an "
                "analytic shakedown outcome was inspected."
            ),
            "confidence": 0.97,
            "confidence_assessment": "strong_corrective_mechanism_evidence_not_confirmation",
            "independent_checkpoint_count": 5,
            "num_direct_experiments": 10,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "standard_error_law": campaign["acquisition_contracts"][
                    "standard_error_law"
                ],
                "tests": tests,
            },
            "key_insights": [
                "A coherent input-side reference correction repairs frozen task behavior without representation retraining.",
                "The dose-response and inverse-square rate law identify reference variance rather than generic extra computation.",
                "The learned set aggregator matches but does not improve on the analytic circular sufficient statistic.",
                "One learned-front-end checkpoint requires 256 repeats, making the exact 64-repeat threshold draw-sensitive.",
                "Exact-reference and shuffled-reference controls distinguish coherent acquisition from marginal orientation statistics.",
            ],
            "suggested_hypotheses": [
                "Across fresh acquisition draws, m=64 is a population threshold while m=256 is a stable per-checkpoint ceiling.",
                "The failed one-step residual ceiling lies outside the local validity radius of the trained residual manifold.",
                "Correlated or biased reference noise will require an estimator beyond the circular mean.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": tests},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "noise_sha256": NOISE_SHA256,
            "denoiser_sha256": DENOISER_SHA256,
            "source_orientation_campaign_sha256": SOURCE_ORIENTATION_SHA256,
            "source_calibrated_campaign_sha256": SOURCE_CALIBRATED_SHA256,
        },
    }


def build_repeated_reference_acquisition_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    campaign = _campaign(results_path)
    completed = datetime.fromisoformat(campaign["completed_at"].replace("Z", "+00:00"))
    output = []
    for detail in _details(results_path):
        summary = _detail_summary(detail)
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics={name: float(value) for name, value in summary.items()},
                primary_metric=float(summary["analytic_m64_pass"]),
                model_architecture=[8, int(detail["_model_parameters"])],
                model_parameters=int(detail["_model_parameters"]),
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["model_checkpoint"],
                observations=[
                    f"Condition: {detail['condition']}.",
                    "TinyLLM, front end, and task head remained frozen.",
                    "Evidence is corrective and outcome-informed, not confirmatory.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=(
                    ["This checkpoint first clears the task gate at 256 rather than 64 repeats."]
                    if not summary["analytic_m64_pass"] else []
                ),
                timestamp=completed,
            )
        )
    return output


def store_repeated_reference_acquisition_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_repeated_reference_acquisition_meta_hypothesis(results_path)
    experiments = build_repeated_reference_acquisition_experiment_results(
        record, results_path
    )
    storage: dict[str, Any] = {
        "aggregate_path": str(output_path),
        "experiment_ids": [item.experiment_id for item in experiments],
    }
    if chromadb_path is not None:
        from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger

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
        storage["result_hashes"] = [logger.log_experiment_result(item) for item in experiments]
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
            raise RuntimeError("repeated-reference ChromaDB read-back failed")
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
    "build_repeated_reference_acquisition_experiment_results",
    "build_repeated_reference_acquisition_meta_hypothesis",
    "store_repeated_reference_acquisition_meta_hypothesis",
]
