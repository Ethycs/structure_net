"""Build and store TinyLLM posterior-coordinate transport evidence."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-posterior-coordinate-transport.v1"
HYPOTHESIS_ID = "tinyllm-posterior-coordinate-transport-v1"
EVIDENCE_ROLE = "preregistered_outcome_directed_posterior_coordinate_transport"
IMPLEMENTATION_SHA256 = (
    "a91cae656663ebb6ab68b9fdacf82806bf417c5bffb5cc7b70cab32ebfaf5fe2"
)
CAMPAIGN_SHA256 = (
    "0733c3069d2c012f760a112307cfcb1c743c8ac19c64f288d19db6f97a08c57d"
)
RESULT_MANIFEST_SHA256 = (
    "317cc0840c8808dc82632f19acb45c6a255853d3c13d1a2ef225b356b2ce33ef"
)
SOURCE_CAMPAIGN_SHA256 = (
    "6b232f523cd570f10ebfcc07c47abae6a724b568d4cfc2852e4b91ccff01321f"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "5cb1944e57db0515dbc7ad5e3956a328be733cdda7f5dda34631d21e8cf3a81c"
)
SOURCE_RESULT_MANIFEST_SHA256 = (
    "5d765445082346512092dfa0acb00e3e39db369bd71829cbe4d96c8871593b4b"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
RANK_LABELS = ("1", "2", "4", "8", "full")
STEP_KEYS = ("4", "16")
EXPECTED_K16_ACTUAL = {
    "analytic_calibrated": {
        7: ("4", "8", "full"),
        17: ("full",),
        29: ("4", "8", "full"),
        41: ("4", "8", "full"),
        53: ("full",),
    },
    "learned_calibrated_equivariant": {
        7: ("4", "8", "full"),
        17: ("8", "full"),
        29: ("8", "full"),
        41: ("8", "full"),
        53: ("full",),
    },
}
EXPECTED_RANK_GATES = {
    "1": False,
    "2": False,
    "4": False,
    "8": False,
    "full": True,
}
CLASSIFICATION = "high_rank_answer_chart"


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
    aggregates = value.get("aggregates", {})
    provenance = value.get("provenance", {})
    coordinate = value.get("coordinate_contracts", {})
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or value.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or aggregates.get("valid") is not True
        or aggregates.get("systems_valid") is not True
        or aggregates.get("scientific_controls_valid") is not True
        or aggregates.get("shuffled_control_contract") is not True
        or aggregates.get("positive_control_contract") is not True
        or aggregates.get("classification") != CLASSIFICATION
        or aggregates.get("smallest_passing_rank") != "full"
        or aggregates.get("rank_gates") != EXPECTED_RANK_GATES
        or aggregates.get("actual_m64_reference_pass_counts")
        != {condition: 5 for condition in CONDITIONS}
        or aggregates.get("exact_endpoint_residual_pass_counts")
        != {condition: 5 for condition in CONDITIONS}
        or coordinate.get("coordinate_basis_orthonormal_contract") is not True
        or coordinate.get("coordinate_basis_nested_contract") is not True
        or provenance.get("source_campaign_sha256") != SOURCE_CAMPAIGN_SHA256
        or provenance.get("source_implementation_sha256")
        != SOURCE_IMPLEMENTATION_SHA256
        or provenance.get("source_result_manifest_sha256")
        != SOURCE_RESULT_MANIFEST_SHA256
        or int(value.get("summary", {}).get("completed", -1)) != 10
        or int(value.get("summary", {}).get("failed", -1)) != 0
        or not _finite_numbers(value)
    ):
        raise ValueError(f"invalid posterior-coordinate transport campaign {path}")
    counts = aggregates.get("rollout_pass_counts", {})
    for rank in RANK_LABELS:
        for condition in CONDITIONS:
            expected = sum(
                1
                for seed in SEEDS
                if rank in EXPECTED_K16_ACTUAL[condition][seed]
            )
            if counts["actual_coordinate"][rank]["16"][condition] != expected:
                raise ValueError(
                    f"unexpected actual-coordinate pass count {rank} {condition}"
                )
            if counts["shuffled_coordinate"][rank]["16"][condition] != 0:
                raise ValueError(
                    f"unexpected shuffled pass count {rank} {condition}"
                )
    return value


def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    indexed = {
        (entry["condition"], int(entry["seed"])): entry
        for entry in campaign.get("results", [])
    }
    expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    if set(indexed) != expected:
        raise ValueError(f"invalid posterior-coordinate result index {path}")
    output: list[dict[str, Any]] = []
    result_paths: list[Path] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            entry = indexed[(condition, seed)]
            result_path = Path(entry["path"])
            arrays_path = Path(entry["arrays_path"])
            detail = json.loads(result_path.read_text(encoding="utf-8"))
            gates = detail.get("gates", {})
            transport = detail.get("transport_gates", {})
            rollouts = transport.get("rollouts", {})
            expected_ranks = EXPECTED_K16_ACTUAL[condition][seed]
            if (
                not result_path.is_file()
                or _sha256(result_path) != entry.get("result_sha256")
                or not arrays_path.is_file()
                or _sha256(arrays_path) != entry.get("arrays_sha256")
                or detail.get("schema_version") != EXPERIMENT_SCHEMA
                or detail.get("hypothesis_id") != HYPOTHESIS_ID
                or detail.get("status") != "completed"
                or detail.get("evidence_role") != EVIDENCE_ROLE
                or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
                or detail.get("condition") != condition
                or int(detail.get("seed", -1)) != seed
                or any(not bool(item) for item in gates.values())
                or float(
                    detail.get("maximum_source_metric_replay_error", math.inf)
                )
                != 0.0
                or transport.get("actual_m64_reference", {}).get("task_gate")
                is not True
                or transport.get("exact_endpoint_residual", {}).get("task_gate")
                is not True
                or transport.get("scalar_comparator", {})
                .get("16", {})
                .get("task_gate")
                is not False
                or any(
                    rollouts["actual_coordinate"][rank]["16"]["task_gate"]
                    is not (rank in expected_ranks)
                    for rank in RANK_LABELS
                )
                or any(
                    rollouts["shuffled_coordinate"][rank][step]["task_gate"]
                    is not False
                    for rank in RANK_LABELS
                    for step in STEP_KEYS
                )
                or not _finite_numbers(detail)
            ):
                raise ValueError(
                    f"invalid posterior-coordinate transport result {result_path}"
                )
            output.append(detail)
            result_paths.append(result_path)
    if _manifest_sha256(result_paths) != RESULT_MANIFEST_SHA256:
        raise ValueError(f"invalid posterior-coordinate result manifest {path}")
    return output


def _detail_summary(detail: Mapping[str, Any]) -> dict[str, Any]:
    transport = detail["transport_gates"]
    regimes = detail["regimes"]
    return {
        "actual_m64_reference_pass": bool(
            transport["actual_m64_reference"]["task_gate"]
        ),
        "exact_endpoint_residual_pass": bool(
            transport["exact_endpoint_residual"]["task_gate"]
        ),
        "k16_rank_gates": {
            rank: bool(
                transport["rollouts"]["actual_coordinate"][rank]["16"]["task_gate"]
            )
            for rank in RANK_LABELS
        },
        "k4_rank_gates": {
            rank: bool(
                transport["rollouts"]["actual_coordinate"][rank]["4"]["task_gate"]
            )
            for rank in RANK_LABELS
        },
        "k16_full_accuracy_loss": {
            regime: float(
                transport["rollouts"]["actual_coordinate"]["full"]["16"][
                    "accuracy_loss_from_clean"
                ][regime]
            )
            for regime in REGIMES
        },
        "k16_full_diagnostics": {
            regime: {
                name: float(
                    regimes[regime]["rollouts"]["actual_coordinate"]["full"]["16"][
                        name
                    ]
                )
                for name in (
                    "mean_relative_residual_endpoint_error",
                    "mean_posterior_js_from_actual_endpoint",
                    "mean_endpoint_coordinate_error",
                    "maximum_condition_number",
                    "maximum_step_norm",
                )
            }
            for regime in REGIMES
        },
        "maximum_source_metric_replay_error": float(
            detail["maximum_source_metric_replay_error"]
        ),
    }


def build_posterior_coordinate_transport_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-posterior-coordinate-transport.md"
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
            "name": "TinyLLM posterior-coordinate transport rank ladder",
            "category": "mechanistic_interpretability",
            "description": (
                "A frozen-checkpoint rank ladder integrates minimum-norm "
                "pseudoinverse updates of nested DCT answer-logit coordinates "
                "along the stored reference-path schedule."
            ),
            "question": (
                "Does the scalar path-moment rollout fail because it discards "
                "answer-relevant posterior shape, or because no answer-output "
                "coordinate system integrates the trained residual path?"
            ),
            "prediction": (
                "A small fixed vector of ordered answer coordinates transports "
                "the frozen task where one ordered moment fails; the strong "
                "compact-chart prediction is rank at most four."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "high_rank_answer_chart_compact_prediction_rejected"
            ),
            "evidence_count": 10,
            "direct_experiment_count": 10,
            "independent_checkpoint_count": 5,
            "power_profile": (
                "five_seed_preregistered_outcome_directed_frozen_transport_ladder"
            ),
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "ten retained d8/N3 calibrated checkpoints; 1,024 composition "
                "and 1,024 extrapolation examples; final-query residual; "
                "DCT ranks 1,2,4,8,full over 16 answer bins; nested K=4,16"
            ),
            "subclaims": {
                "systems_replay_and_controls": "supported_ten_of_ten_exact",
                "actual_reference_and_exact_endpoint": (
                    "supported_five_of_five_both_arms"
                ),
                "full_rank_k16_coordinate_transport": (
                    "supported_five_of_five_both_arms"
                ),
                "full_rank_k4_coordinate_transport": (
                    "supported_five_of_five_both_arms"
                ),
                "compact_rank_le4_transport": "rejected_population_gate",
                "rank8_transport": "rejected_three_and_four_of_five",
                "shuffled_specificity": "supported_zero_passes_all_ranks",
                "scalar_comparator_k16": "replayed_zero_of_five_both_arms",
                "answer_coordinates_nonintegrable": "rejected",
                "model_or_frontend_retraining_licensed": "no",
                "link_cobordism_triggered": "no_codimension_two_locus_measured",
            },
            "tags": [
                "tinyllm",
                "reference-path",
                "posterior-coordinate",
                "rank-ladder",
                "dct-basis",
                "frozen-checkpoint",
                "causal-intervention",
                "preregistered",
            ],
            "explicitly_not_tested": list(campaign["method_boundaries"])
            + [
                "Deployable estimation of the coordinate schedule without the "
                "stored reference path.",
                "Direct projection onto the observed residual curve.",
                "Natural-language behavior or architecture-population prevalence.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The complete centered answer-logit chart transports the frozen "
                "task in five of five checkpoints in both arms at K=4 and K=16, "
                "but every compact rank (1, 2, 4, 8) fails the population gate, "
                "so the locked classification is high_rank_answer_chart and the "
                "compact-chart prediction is rejected."
            ),
            "confidence": 0.99,
            "confidence_assessment": (
                "exact_five_seed_preregistered_rank_ladder_with_controls"
            ),
            "independent_checkpoint_count": 5,
            "num_direct_experiments": 10,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "path_contracts": campaign["path_contracts"],
                "coordinate_contracts": campaign["coordinate_contracts"],
                "direct_tests": direct_tests,
            },
            "key_insights": [
                (
                    "Posterior shape is a sufficient transport target: the full "
                    "centered answer-logit schedule integrates the trained "
                    "residual path where the scalar ordered moment fails."
                ),
                (
                    "Sufficiency is not compact: ranks one through eight all "
                    "fail the four-of-five population gate in at least one arm."
                ),
                (
                    "Task recovery does not require reaching the actual "
                    "endpoint residual: full-rank rollouts stop 31--40% of the "
                    "chord away from it while matching its answer posterior to "
                    "posterior JS near zero."
                ),
                (
                    "Coordinate Jacobians remained full row rank with condition "
                    "numbers at most 8.1, so the compact-rank failures are "
                    "chart-insufficiency, not pseudoinverse pathology."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "The answer-relevant residual subspace at the final query "
                    "has effective dimension between nine and fifteen in these "
                    "checkpoints."
                ),
                (
                    "A deployable transport interface must estimate a "
                    "high-dimensional posterior-shape schedule, not a scalar "
                    "moment; no writer, observer, or retraining branch is "
                    "licensed by this outcome."
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
        },
    }


def build_posterior_coordinate_transport_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        summary = _detail_summary(detail)
        metrics: dict[str, float] = {
            "campaign_valid": 1.0,
            "actual_m64_reference_pass": float(
                summary["actual_m64_reference_pass"]
            ),
            "exact_endpoint_residual_pass": float(
                summary["exact_endpoint_residual_pass"]
            ),
            "maximum_source_metric_replay_error": summary[
                "maximum_source_metric_replay_error"
            ],
        }
        for rank in RANK_LABELS:
            metrics[f"k16_rank_{rank}_pass"] = float(
                summary["k16_rank_gates"][rank]
            )
            metrics[f"k4_rank_{rank}_pass"] = float(
                summary["k4_rank_gates"][rank]
            )
        for regime in REGIMES:
            metrics[f"k16_full_{regime}_accuracy_loss"] = summary[
                "k16_full_accuracy_loss"
            ][regime]
            for name, value in summary["k16_full_diagnostics"][regime].items():
                metrics[f"k16_full_{regime}_{name}"] = value
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(summary["k16_rank_gates"]["full"]),
                model_architecture=[8, 8, 384],
                model_parameters=50_965_504,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["model_checkpoint"],
                observations=[
                    f"Condition: {detail['condition']}.",
                    "All model, front-end, and answer parameters remained frozen.",
                    (
                        "Nested DCT ranks 1,2,4,8,full were gated separately at "
                        "K=4 and K=16 with fiber-block-shuffled controls."
                    ),
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[
                    "Full-rank coordinate transport passes while every compact "
                    "rank fails the population gate."
                ],
                timestamp=completed,
            )
        )
    return output


def store_posterior_coordinate_transport_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_posterior_coordinate_transport_meta_hypothesis(results_path)
    experiments = build_posterior_coordinate_transport_experiment_results(
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
            raise RuntimeError(
                "posterior-coordinate transport ChromaDB read-back failed"
            )
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
    "build_posterior_coordinate_transport_experiment_results",
    "build_posterior_coordinate_transport_meta_hypothesis",
    "store_posterior_coordinate_transport_meta_hypothesis",
]
