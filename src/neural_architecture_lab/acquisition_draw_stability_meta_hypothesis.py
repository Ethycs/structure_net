"""Build and store preregistered TinyLLM acquisition-draw stability evidence."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-acquisition-draw-stability.v1"
HYPOTHESIS_ID = "tinyllm-acquisition-draw-stability-v1"
EVIDENCE_ROLE = "preregistered_frozen_system_acquisition_draw_replication"
IMPLEMENTATION_SHA256 = (
    "a0eae3da0dfcf74328ff0f2fa264a8e712b61f901337608f9e23ef93657d0440"
)
CAMPAIGN_SHA256 = (
    "968f85010129d761268b4816d85ddd2ab578bbc93307e8a936e58fa891e89d93"
)
RESULT_MANIFEST_SHA256 = (
    "d13e52a07423e507cef034c78b734219b85abc8468feae6313b52148fa95b163"
)
DRAW_ARRAY_SHA256 = (
    "57eca80cccf1b916a60d79d5982bdbffe3b515cee7dfbee7645830448779aace"
)
SOURCE_ORIENTATION_SHA256 = (
    "876af062f9d0bdd365e2f1ffbb959d9ff2e5e4e277321b510bbe6a956456877f"
)
SOURCE_CALIBRATED_SHA256 = (
    "80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
SEEDS = (7, 17, 29, 41, 53)
COUNTS = (64, 256)
DRAW_COUNT = 16
EXPECTED_M64_PASSES = {
    "analytic_calibrated": {7: 14, 17: 12, 29: 15, 41: 16, 53: 16},
    "learned_calibrated_equivariant": {
        7: 14,
        17: 16,
        29: 15,
        41: 16,
        53: 10,
    },
}


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
    artifacts = value.get("artifacts", {})
    provenance = value.get("provenance", {})
    draw_path = Path(artifacts.get("draw_arrays", ""))
    source_path = Path(provenance.get("orientation_campaign", ""))
    counts = aggregates.get("counts", {})
    m64 = counts.get("m64", {})
    m256 = counts.get("m256", {})
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
        "fitted_acquisition_parameters": 0,
    }
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or value.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or tuple(config.get("arms", ())) != CONDITIONS
        or tuple(config.get("seeds", ())) != SEEDS
        or tuple(config.get("replicate_counts", ())) != COUNTS
        or int(config.get("draw_count", -1)) != DRAW_COUNT
        or int(config.get("draw_seed_root", -1)) != 81_027_026
        or float(config.get("measurement_sigma_radians", math.nan)) != 0.175
        or float(config.get("accuracy_loss_ceiling", math.nan)) != 0.03
        or int(config.get("required_seed_passes", -1)) != 4
        or int(config.get("required_primary_draw_passes", -1)) != 15
        or config.get("allow_underpowered") is not False
        or summary != expected_summary
        or aggregates.get("valid") is not True
        or aggregates.get("controls_pass") is not True
        or aggregates.get("primary_hypothesis_pass") is not True
        or aggregates.get("classification")
        != "m64_broadly_stable_checkpoint_variable"
        or int(m64.get("complete_draw_passes", -1)) != 14
        or int(m256.get("complete_draw_passes", -1)) != 16
        or m64.get("arm_population_draw_passes")
        != {
            "analytic_calibrated": 14,
            "learned_calibrated_equivariant": 16,
        }
        or m256.get("arm_population_draw_passes")
        != {
            "analytic_calibrated": 16,
            "learned_calibrated_equivariant": 16,
        }
        or value.get("noise_contracts", {}).get("pass") is not True
        or value.get("acquisition_contracts", {}).get("pass") is not True
        or provenance.get("orientation_campaign_sha256")
        != SOURCE_ORIENTATION_SHA256
        or provenance.get("calibrated_campaign_sha256")
        != SOURCE_CALIBRATED_SHA256
        or not source_path.is_file()
        or _sha256(source_path) != SOURCE_ORIENTATION_SHA256
        or not draw_path.is_file()
        or _sha256(draw_path) != DRAW_ARRAY_SHA256
        or artifacts.get("draw_arrays_sha256") != DRAW_ARRAY_SHA256
        or not _finite(value)
    ):
        raise ValueError(f"invalid acquisition-draw stability campaign {path}")
    manifest = hashlib.sha256(
        json.dumps(
            value.get("results", []),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if manifest != RESULT_MANIFEST_SHA256:
        raise ValueError(f"invalid acquisition-draw result manifest {path}")
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
        raise ValueError(f"invalid acquisition-draw result index {path}")
    output: list[dict[str, Any]] = []
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
            draws = detail.get("draws", {})
            m64_passes = sum(
                int(draws.get(f"draw_{draw:02d}", {}).get("m64", {}).get("seed_gate") is True)
                for draw in range(DRAW_COUNT)
            )
            m256_passes = sum(
                int(draws.get(f"draw_{draw:02d}", {}).get("m256", {}).get("seed_gate") is True)
                for draw in range(DRAW_COUNT)
            )
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
                or set(draws) != {f"draw_{draw:02d}" for draw in range(DRAW_COUNT)}
                or any(
                    set(draws[f"draw_{draw:02d}"]) != {"m64", "m256"}
                    for draw in range(DRAW_COUNT)
                )
                or m64_passes != EXPECTED_M64_PASSES[condition][seed]
                or m256_passes != DRAW_COUNT
                or any(
                    detail.get("gates", {}).get(name) is not True
                    for name in (
                        "acquisition_contract",
                        "clean_replay",
                        "finite",
                        "noise_contract",
                        "state_unchanged",
                        "validity",
                    )
                )
                or detail.get("controls", {})
                .get("exact_reference_oracle", {})
                .get("seed_gate")
                is not True
                or detail.get("controls", {})
                .get("fiber_shuffled_draw0_m256", {})
                .get("seed_gate")
                is not False
                or detail.get("inherited_single_observation", {}).get("seed_gate")
                is not False
                or not source_path.is_file()
                or _sha256(source_path) != provenance.get("source_result_sha256")
                or not model_path.is_file()
                or _sha256(model_path) != provenance.get("model_checkpoint_sha256")
                or not frontend_path.is_file()
                or _sha256(frontend_path)
                != provenance.get("frontend_checkpoint_sha256")
                or not _finite(detail)
            ):
                raise ValueError(f"invalid acquisition-draw result {result_path}")
            fingerprint = detail["scientific_fingerprint"]
            if fingerprint in fingerprints:
                raise ValueError(f"duplicate acquisition-draw fingerprint {result_path}")
            fingerprints.add(fingerprint)
            detail["_model_parameters"] = int(source["model_parameters"])
            output.append(detail)
    return output


def _detail_summary(detail: Mapping[str, Any]) -> dict[str, Any]:
    pass_counts = {}
    accuracy_losses = {64: [], 256: []}
    for count in COUNTS:
        passes = 0
        for draw in range(DRAW_COUNT):
            level = detail["draws"][f"draw_{draw:02d}"][f"m{count}"]
            passes += int(level["seed_gate"])
            accuracy_losses[count].extend(
                float(item["accuracy_loss_from_clean"])
                for item in level["task_recovery"].values()
            )
        pass_counts[count] = passes
    return {
        "m64_draw_passes": pass_counts[64],
        "m64_draw_pass_frequency": pass_counts[64] / DRAW_COUNT,
        "m256_draw_passes": pass_counts[256],
        "m256_draw_pass_frequency": pass_counts[256] / DRAW_COUNT,
        "m64_mean_accuracy_loss": mean(accuracy_losses[64]),
        "m64_max_accuracy_loss": max(accuracy_losses[64]),
        "m256_mean_accuracy_loss": mean(accuracy_losses[256]),
        "m256_max_accuracy_loss": max(accuracy_losses[256]),
        "shuffled_pass": bool(
            detail["controls"]["fiber_shuffled_draw0_m256"]["seed_gate"]
        ),
        "analysis_seconds": float(detail["analysis_seconds"]),
    }


def build_acquisition_draw_stability_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-10_tinyllm-acquisition-draw-stability.md"
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
            "name": "TinyLLM acquisition-draw stability",
            "category": "mechanistic_interpretability",
            "description": (
                "Sixteen fresh independent acquisition arrays test the stability "
                "of analytic repeated-reference recovery in ten frozen TinyLLM systems."
            ),
            "question": (
                "Is m=256 a stable two-arm population ceiling, and how often does "
                "the earlier m=64 threshold survive a fresh acquisition draw?"
            ),
            "prediction": (
                "At least fifteen of sixteen complete m=256 draws must pass both "
                "five-checkpoint populations on composition and extrapolation."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": (
                "preregistered_m256_stability_confirmed_m64_checkpoint_variable"
            ),
            "evidence_count": 16,
            "direct_experiment_count": 16,
            "independent_acquisition_draw_count": 16,
            "fixed_checkpoint_count": 5,
            "stored_checkpoint_summary_count": 10,
            "evidence_role": EVIDENCE_ROLE,
            "power_profile": "sixteen_draw_two_arm_five_checkpoint_frozen_replication",
            "tested_scope": (
                "sixteen independent Gaussian orientation-acquisition draws, "
                "m=64 and m=256, ten retained d8/N3 structured systems, "
                "composition and extrapolation"
            ),
            "subclaims": {
                "m256_complete_draw_stability": "supported_sixteen_of_sixteen",
                "m256_checkpoint_cells": "supported_160_of_160",
                "m64_complete_draw_stability": "broad_not_stable_fourteen_of_sixteen",
                "m64_analytic_population": "fourteen_of_sixteen",
                "m64_learned_population": "sixteen_of_sixteen",
                "m64_checkpoint_variability": "supported_sixteen_failures_of_160_cells",
                "fiber_shuffle_specificity": "supported_zero_of_five_both_arms",
                "independent_gaussian_sample_count_branch": "closed",
                "model_retraining_licensed": "no",
            },
            "tags": [
                "tinyllm",
                "acquisition-draw",
                "repeated-reference",
                "sample-complexity",
                "frozen-checkpoint",
                "causal-intervention",
                "preregistered",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "All sixteen fresh complete m=256 draws pass both frozen "
                "five-checkpoint populations; controls and exact resume pass. "
                "The separately locked m=64 endpoint passes fourteen of sixteen."
            ),
            "confidence": 0.98,
            "confidence_assessment": (
                "strong_preregistered_support_with_declared_binomial_scope"
            ),
            "independent_acquisition_draw_count": 16,
            "num_direct_experiments": 16,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "noise_contracts": campaign["noise_contracts"],
                "acquisition_contracts": campaign["acquisition_contracts"],
                "checkpoint_summaries": tests,
            },
            "key_insights": [
                "m=256 repairs every frozen checkpoint on all sixteen fresh draws without fitting any parameter.",
                "m=64 is a broad population threshold but fails two complete draws and sixteen checkpoint-draw cells.",
                "Checkpoint redundancy lets the learned-front-end population pass m=64 even though seed 53 passes only ten of sixteen draws.",
                "Exact-reference, inherited single-observation, and fiber-shuffled controls isolate coherent acquisition precision.",
                "The independent-Gaussian sample-count branch is closed; more optimization under the same noise model is not licensed.",
            ],
            "suggested_hypotheses": [
                "Correlated orientation errors will break the inverse-square recovery observed under independent draws.",
                "Systematic orientation bias will create a non-vanishing task-error floor that repetition alone cannot remove.",
                "Acquisition-aware deployment should target a precision criterion rather than a universal fixed repeat count."
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"checkpoint_summaries": tests},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "draw_array_sha256": DRAW_ARRAY_SHA256,
            "source_orientation_campaign_sha256": SOURCE_ORIENTATION_SHA256,
            "source_calibrated_campaign_sha256": SOURCE_CALIBRATED_SHA256,
        },
    }


def build_acquisition_draw_stability_experiment_results(
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
                metrics={name: float(value) for name, value in summary.items()},
                primary_metric=float(summary["m256_draw_pass_frequency"]),
                model_architecture=[8, int(detail["_model_parameters"])],
                model_parameters=int(detail["_model_parameters"]),
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["model_checkpoint"],
                observations=[
                    f"Condition: {detail['condition']}.",
                    "The acquisition draw is the replication unit; this row summarizes one fixed checkpoint.",
                    "TinyLLM, front end, task head, and analytic circular mean remained frozen.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=(
                    [
                        f"This checkpoint passes only {summary['m64_draw_passes']}/16 m=64 acquisition draws."
                    ]
                    if int(summary["m64_draw_passes"]) < 15
                    else []
                ),
                timestamp=completed,
            )
        )
    return output


def store_acquisition_draw_stability_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_acquisition_draw_stability_meta_hypothesis(results_path)
    experiments = build_acquisition_draw_stability_experiment_results(
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
            raise RuntimeError("acquisition-draw ChromaDB read-back failed")
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
    "build_acquisition_draw_stability_experiment_results",
    "build_acquisition_draw_stability_meta_hypothesis",
    "store_acquisition_draw_stability_meta_hypothesis",
]
