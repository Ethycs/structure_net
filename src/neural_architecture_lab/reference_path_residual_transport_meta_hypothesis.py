"""Build and store TinyLLM reference-path residual-transport evidence."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-reference-path-residual-transport.v1.4"
HYPOTHESIS_ID = "tinyllm-reference-path-residual-transport-v1"
EVIDENCE_ROLE = "corrective_outcome_exposed_frozen_residual_transport_audit"
IMPLEMENTATION_SHA256 = (
    "5cb1944e57db0515dbc7ad5e3956a328be733cdda7f5dda34631d21e8cf3a81c"
)
CAMPAIGN_SHA256 = (
    "6b232f523cd570f10ebfcc07c47abae6a724b568d4cfc2852e4b91ccff01321f"
)
RESULT_MANIFEST_SHA256 = (
    "5d765445082346512092dfa0acb00e3e39db369bd71829cbe4d96c8871593b4b"
)
SOURCE_CAMPAIGN_SHA256 = (
    "269fd948f0d6fee8916bbe3cb94c1d87f76572e43c103b52fc8775fa9653031e"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "6c3cc4463b2c515280c778461e4606dca6ffb19f08d11cb5b7a852a942c7df77"
)
SOURCE_RESULT_MANIFEST_SHA256 = (
    "8181d48f99850d5e487b5209d648373410bea3b46b2eafcb58f57118ed898c1c"
)
SOURCE_REPEAT_ARRAY_SHA256 = (
    "8886c1f6ad0fd307720748e44b1741edb656fdb9e29e913c7ad059b72397eef4"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
STEP_COUNTS = (1, 2, 4, 8, 16)
SCHEDULES = ("true_cosine", "path_moment", "shuffled_path_moment")
EXPECTED_TRUE_COSINE = {
    "analytic_calibrated": {seed: True for seed in SEEDS},
    "learned_calibrated_equivariant": {
        7: True,
        17: False,
        29: True,
        41: True,
        53: True,
    },
}
EXPECTED_SOURCE_ONE_STEP = {
    "analytic_calibrated": {seed: False for seed in SEEDS},
    "learned_calibrated_equivariant": {
        7: True,
        17: False,
        29: False,
        41: True,
        53: False,
    },
}
EXPECTED_ROLLOUT_COUNTS = {
    "true_cosine": {
        "1": {"analytic_calibrated": 0, "learned_calibrated_equivariant": 2},
        "2": {"analytic_calibrated": 5, "learned_calibrated_equivariant": 3},
        "4": {"analytic_calibrated": 5, "learned_calibrated_equivariant": 5},
        "8": {"analytic_calibrated": 5, "learned_calibrated_equivariant": 4},
        "16": {"analytic_calibrated": 5, "learned_calibrated_equivariant": 4},
    },
    "path_moment": {
        "1": {"analytic_calibrated": 0, "learned_calibrated_equivariant": 0},
        "2": {"analytic_calibrated": 0, "learned_calibrated_equivariant": 0},
        "4": {"analytic_calibrated": 0, "learned_calibrated_equivariant": 1},
        "8": {"analytic_calibrated": 0, "learned_calibrated_equivariant": 1},
        "16": {"analytic_calibrated": 0, "learned_calibrated_equivariant": 0},
    },
    "shuffled_path_moment": {
        str(step): {
            "analytic_calibrated": 0,
            "learned_calibrated_equivariant": 0,
        }
        for step in STEP_COUNTS
    },
}


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


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    config = value.get("configuration", {})
    summary = value.get("summary", {})
    aggregates = value.get("aggregates", {})
    contracts = value.get("path_contracts", {})
    provenance = value.get("provenance", {})
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or value.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or tuple(config.get("seeds", ())) != SEEDS
        or tuple(config.get("step_counts", ())) != STEP_COUNTS
        or int(config.get("fine_steps", -1)) != 16
        or float(config.get("accuracy_loss_ceiling", math.nan)) != 0.03
        or float(config.get("replay_tolerance", math.nan)) != 2e-6
        or int(config.get("required_seed_passes", -1)) != 4
        or int(config.get("control_pass_ceiling", -1)) != 1
        or config.get("allow_underpowered") is not False
        or summary
        != {
            "requested": 10,
            "completed": 10,
            "failed": 0,
            "reused": 0,
            "trained_models": 0,
            "trained_frontends": 0,
            "trained_task_heads": 0,
            "trained_probes": 0,
            "fitted_transport_parameters": 0,
        }
        or aggregates.get("valid") is not True
        or aggregates.get("systems_valid") is not True
        or aggregates.get("scientific_controls_valid") is not True
        or aggregates.get("classification") != "semantic_schedule_only"
        or aggregates.get("rollout_pass_counts") != EXPECTED_ROLLOUT_COUNTS
        or aggregates.get("actual_m64_reference_pass_counts")
        != {condition: 5 for condition in CONDITIONS}
        or aggregates.get("exact_endpoint_residual_pass_counts")
        != {condition: 5 for condition in CONDITIONS}
        or aggregates.get("source_one_step_true_cosine_pass_counts")
        != {"analytic_calibrated": 0, "learned_calibrated_equivariant": 2}
        or aggregates.get("positive_control_contract") is not True
        or aggregates.get("shuffled_control_contract") is not True
        or aggregates.get("source_one_step_below_population_contract") is not True
        or contracts.get("exact_fiber_path_sharing_contract") is not True
        or contracts.get("reference_path_construction_contract") is not True
        or contracts.get("nested_step_grid_contract") is not True
        or max(contracts.get("maximum_exact_fiber_path_sharing_error", {}).values())
        > 1e-12
        or max(contracts.get("maximum_reference_path_endpoint_error", {}).values())
        != 0.0
        or provenance.get("source_campaign_sha256") != SOURCE_CAMPAIGN_SHA256
        or provenance.get("source_implementation_sha256")
        != SOURCE_IMPLEMENTATION_SHA256
        or provenance.get("source_result_manifest_sha256")
        != SOURCE_RESULT_MANIFEST_SHA256
        or provenance.get("source_repeat_arrays_sha256")
        != SOURCE_REPEAT_ARRAY_SHA256
        or not _finite_numbers(value)
    ):
        raise ValueError(f"invalid reference-path residual-transport campaign {path}")
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
        raise ValueError(f"invalid reference-path result index {path}")

    output: list[dict[str, Any]] = []
    result_paths: list[Path] = []
    fingerprints: set[str] = set()
    for condition in CONDITIONS:
        for seed in SEEDS:
            entry = indexed[(condition, seed)]
            result_path = Path(entry["path"])
            arrays_path = Path(entry["arrays_path"])
            detail = json.loads(result_path.read_text(encoding="utf-8"))
            gates = detail.get("gates", {})
            transport = detail.get("transport_gates", {})
            rollouts = transport.get("rollouts", {})
            provenance = detail.get("provenance", {})
            source_result_path = Path(provenance.get("source_result", ""))
            model_path = Path(provenance.get("model_checkpoint", ""))
            frontend_path = Path(provenance.get("frontend_checkpoint", ""))
            k16_true = EXPECTED_TRUE_COSINE[condition][seed]
            k1_true = EXPECTED_SOURCE_ONE_STEP[condition][seed]
            if (
                not result_path.is_file()
                or _sha256(result_path) != entry.get("result_sha256")
                or not arrays_path.is_file()
                or _sha256(arrays_path) != entry.get("arrays_sha256")
                or detail.get("artifacts", {}).get("diagnostic_arrays_sha256")
                != entry.get("arrays_sha256")
                or detail.get("schema_version") != EXPERIMENT_SCHEMA
                or detail.get("hypothesis_id") != HYPOTHESIS_ID
                or detail.get("status") != "completed"
                or detail.get("evidence_role") != EVIDENCE_ROLE
                or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
                or detail.get("condition") != condition
                or int(detail.get("seed", -1)) != seed
                or detail.get("experiment_id") != entry.get("experiment_id")
                or detail.get("scientific_fingerprint")
                != entry.get("scientific_fingerprint")
                or any(
                    gates.get(name) is not True
                    for name in (
                        "exact_fiber_path_sharing_contract",
                        "finite_numerical_contract",
                        "nested_step_grid_contract",
                        "reference_path_construction_contract",
                        "source_metric_replay_contract",
                        "system_state_unchanged_contract",
                        "validity_contract",
                    )
                )
                or float(detail.get("maximum_source_metric_replay_error", math.inf))
                != 0.0
                or float(detail.get("minimum_task_gradient_norm", 0.0)) <= 0.0
                or transport.get("actual_m64_reference", {}).get("task_gate")
                is not True
                or transport.get("exact_endpoint_residual", {}).get("task_gate")
                is not True
                or rollouts.get("true_cosine", {}).get("1", {}).get("task_gate")
                is not k1_true
                or rollouts.get("true_cosine", {}).get("16", {}).get("task_gate")
                is not k16_true
                or rollouts.get("path_moment", {}).get("16", {}).get("task_gate")
                is not False
                or any(
                    rollouts.get("shuffled_path_moment", {})
                    .get(str(step), {})
                    .get("task_gate")
                    is not False
                    for step in STEP_COUNTS
                )
                or set(detail.get("regimes", {})) != set(REGIMES)
                or not source_result_path.is_file()
                or _sha256(source_result_path)
                != provenance.get("source_result_sha256")
                or not model_path.is_file()
                or _sha256(model_path) != provenance.get("model_checkpoint_sha256")
                or not frontend_path.is_file()
                or _sha256(frontend_path)
                != provenance.get("frontend_checkpoint_sha256")
                or not _finite_numbers(detail)
            ):
                raise ValueError(
                    f"invalid reference-path residual-transport result {result_path}"
                )
            fingerprint = detail["scientific_fingerprint"]
            if fingerprint in fingerprints:
                raise ValueError(f"duplicate transport fingerprint {result_path}")
            fingerprints.add(fingerprint)
            output.append(detail)
            result_paths.append(result_path)
    if _manifest_sha256(result_paths) != RESULT_MANIFEST_SHA256:
        raise ValueError(f"invalid reference-path result manifest {path}")
    return output


def _detail_summary(detail: Mapping[str, Any]) -> dict[str, Any]:
    transport = detail["transport_gates"]
    regimes = detail["regimes"]
    rollouts = transport["rollouts"]
    return {
        "complete_finite_radius_gate": bool(
            transport["actual_m64_reference"]["task_gate"]
            and transport["exact_endpoint_residual"]["task_gate"]
            and rollouts["true_cosine"]["16"]["task_gate"]
            and rollouts["path_moment"]["16"]["task_gate"]
            and not rollouts["shuffled_path_moment"]["16"]["task_gate"]
        ),
        "actual_m64_reference_pass": bool(
            transport["actual_m64_reference"]["task_gate"]
        ),
        "exact_endpoint_residual_pass": bool(
            transport["exact_endpoint_residual"]["task_gate"]
        ),
        "source_one_step_true_cosine_pass": bool(
            rollouts["true_cosine"]["1"]["task_gate"]
        ),
        "k16_true_cosine_pass": bool(
            rollouts["true_cosine"]["16"]["task_gate"]
        ),
        "k16_path_moment_pass": bool(
            rollouts["path_moment"]["16"]["task_gate"]
        ),
        "k16_shuffled_path_moment_pass": bool(
            rollouts["shuffled_path_moment"]["16"]["task_gate"]
        ),
        "k16_accuracy_loss": {
            schedule: {
                regime: float(
                    rollouts[schedule]["16"]["accuracy_loss_from_clean"][regime]
                )
                for regime in REGIMES
            }
            for schedule in SCHEDULES
        },
        "geometry": {
            regime: {
                **{
                    name: float(
                        regimes[regime]["actual_reference_path"]["geometry"][name]
                    )
                    for name in (
                        "arc_chord_ratio",
                        "maximum_relative_chord_deviation",
                    )
                },
                **{
                    name: float(
                        regimes[regime]["actual_reference_path"]["local_transport"][
                            name
                        ]
                    )
                    for name in (
                        "mean_parallel_increment_fraction",
                        "mean_orthogonal_increment_fraction",
                        "relative_first_order_moment_error",
                        "mean_local_posterior_js",
                        "mean_local_relative_residual_error",
                    )
                },
            }
            for regime in REGIMES
        },
        "k16_rollouts": {
            regime: {
                schedule: {
                    name: float(regimes[regime]["rollouts"][schedule]["16"][name])
                    for name in (
                        "mean_absolute_actual_endpoint_moment_error",
                        "mean_absolute_target_cosine_error",
                        "mean_posterior_js_from_actual_endpoint",
                        "mean_relative_residual_endpoint_error",
                    )
                }
                for schedule in ("true_cosine", "path_moment")
            }
            for regime in REGIMES
        },
        "minimum_task_gradient_norm": float(detail["minimum_task_gradient_norm"]),
        "maximum_source_metric_replay_error": float(
            detail["maximum_source_metric_replay_error"]
        ),
    }


def build_reference_path_residual_transport_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-reference-path-residual-transport.md"
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
            "provenance": detail["provenance"],
            "gates": detail["gates"],
            **_detail_summary(detail),
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM reference-path residual transport",
            "category": "mechanistic_interpretability",
            "description": (
                "A frozen-checkpoint audit integrates minimum-norm local task "
                "updates along semantic, model-derived, and fiber-shuffled "
                "reference schedules."
            ),
            "question": (
                "Did the earlier one-step true-coordinate write fail primarily "
                "because it exceeded the local validity radius?"
            ),
            "prediction": (
                "At K=16 both true-cosine and observed path-moment schedules "
                "pass at least four of five checkpoints in each structured arm."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "semantic_schedule_only_finite_radius_hypothesis_rejected"
            ),
            "evidence_count": 10,
            "direct_experiment_count": 10,
            "independent_checkpoint_count": 5,
            "power_profile": (
                "five_seed_corrective_outcome_exposed_frozen_transport_audit"
            ),
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "ten retained d8/N3 calibrated checkpoints; 1,024 composition "
                "and 1,024 extrapolation examples; final-query residual; "
                "nested K=1,2,4,8,16"
            ),
            "subclaims": {
                "systems_replay_and_controls": "supported_ten_of_ten",
                "actual_reference_and_exact_endpoint": (
                    "supported_five_of_five_both_arms"
                ),
                "k16_true_cosine_transport": (
                    "supported_five_of_five_analytic_four_of_five_learned"
                ),
                "k16_path_moment_transport": "rejected_zero_of_five_both_arms",
                "k16_shuffled_specificity": "supported_zero_of_five_both_arms",
                "finite_radius_only_explanation": "rejected",
                "scalar_model_moment_as_sufficient_transport_chart": "rejected",
                "model_or_frontend_retraining_licensed": "no",
                "link_cobordism_triggered": "no_codimension_two_locus_measured",
            },
            "tags": [
                "tinyllm",
                "reference-path",
                "residual-transport",
                "task-gradient",
                "frozen-checkpoint",
                "causal-intervention",
                "scalar-chart",
                "negative-result",
                "outcome-exposed-corrective",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "A full answer-simplex or centered-logit transport chart.",
                "Projection back to the actual residual curve.",
                "Natural-language behavior or architecture-population prevalence.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "K=16 semantic true-cosine transport reaches the population gate "
                "in both arms, but the preregistered model-derived path-moment "
                "transport passes zero checkpoints in both arms."
            ),
            "confidence": 0.99,
            "confidence_assessment": (
                "exact_five_seed_corrective_component_falsification"
            ),
            "independent_checkpoint_count": 5,
            "num_direct_experiments": 10,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "path_contracts": campaign["path_contracts"],
                "direct_tests": direct_tests,
            },
            "key_insights": [
                (
                    "Repeated local relinearization can follow an externally "
                    "supplied semantic cosine coordinate."
                ),
                (
                    "The frozen path's ordered posterior moment is not a "
                    "sufficient scalar transport coordinate."
                ),
                (
                    "Local scalar moment predictions are accurate, yet 78--81% "
                    "of actual residual increments are orthogonal to the scalar "
                    "task gradient."
                ),
                (
                    "Path-moment rollouts track the requested endpoint moment and "
                    "remain closer to the actual residual, but still miss semantic "
                    "cosine and exact-bin task utility."
                ),
                (
                    "The finite-radius-only explanation is falsified without "
                    "retraining any component."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "A no-fit centered-logit or answer-simplex tangent chart "
                    "recovers task transport along the observed reference path."
                ),
                (
                    "If multi-coordinate transport fails with accurate local "
                    "replay, continuation-relevant residual directions lie "
                    "outside the answer-state Jacobian and the residual-writer "
                    "branch should terminate."
                ),
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": direct_tests},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "source_implementation_sha256": SOURCE_IMPLEMENTATION_SHA256,
            "source_result_manifest_sha256": SOURCE_RESULT_MANIFEST_SHA256,
            "source_repeat_array_sha256": SOURCE_REPEAT_ARRAY_SHA256,
        },
    }


def build_reference_path_residual_transport_experiment_results(
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
            "campaign_valid": 1.0,
            "complete_finite_radius_gate": float(
                summary["complete_finite_radius_gate"]
            ),
            "actual_m64_reference_pass": float(
                summary["actual_m64_reference_pass"]
            ),
            "exact_endpoint_residual_pass": float(
                summary["exact_endpoint_residual_pass"]
            ),
            "source_one_step_true_cosine_pass": float(
                summary["source_one_step_true_cosine_pass"]
            ),
            "k16_true_cosine_pass": float(summary["k16_true_cosine_pass"]),
            "k16_path_moment_pass": float(summary["k16_path_moment_pass"]),
            "k16_shuffled_path_moment_pass": float(
                summary["k16_shuffled_path_moment_pass"]
            ),
            "minimum_task_gradient_norm": summary["minimum_task_gradient_norm"],
            "maximum_source_metric_replay_error": summary[
                "maximum_source_metric_replay_error"
            ],
        }
        for schedule in SCHEDULES:
            for regime in REGIMES:
                metrics[f"k16_{schedule}_{regime}_accuracy_loss"] = summary[
                    "k16_accuracy_loss"
                ][schedule][regime]
        for regime in REGIMES:
            for name, value in summary["geometry"][regime].items():
                metrics[f"{regime}_{name}"] = value
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(summary["complete_finite_radius_gate"]),
                model_architecture=[8, 8, 384],
                model_parameters=50_965_504,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["model_checkpoint"],
                observations=[
                    f"Condition: {detail['condition']}.",
                    "All model, front-end, and answer parameters remained frozen.",
                    (
                        "K=16 true-cosine and path-moment schedules were gated "
                        "separately on both shifts."
                    ),
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[
                    "Semantic true-cosine transport passes while model-derived "
                    "path-moment transport fails."
                ],
                timestamp=completed,
            )
        )
    return output


def store_reference_path_residual_transport_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_reference_path_residual_transport_meta_hypothesis(results_path)
    experiments = build_reference_path_residual_transport_experiment_results(
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
            raise RuntimeError("reference-path transport ChromaDB read-back failed")
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
    "build_reference_path_residual_transport_experiment_results",
    "build_reference_path_residual_transport_meta_hypothesis",
    "store_reference_path_residual_transport_meta_hypothesis",
]
