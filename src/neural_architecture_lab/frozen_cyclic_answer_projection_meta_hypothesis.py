"""Build and store the TinyLLM frozen cyclic answer-projection result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-frozen-cyclic-answer-projection.v1"
HYPOTHESIS_ID = "tinyllm-frozen-cyclic-answer-projection-v1"
EVIDENCE_ROLE = "registered_outcome_informed_artifact_only_answer_head_diagnostic"
CLASSIFICATION = "partial_cyclic_answer_projection_repair"
RESULT_SHA256 = (
    "e76cbc3747211ececa209a16202c43b672f41a00c72ea473ed43202c97b374e6"
)
DIAGNOSTICS_SHA256 = (
    "fe47fb505aa3ee64542afa5a3bc7da051c3d9ba23c800f863b0d9dfb83e0e54e"
)
RUNNER_SHA256 = (
    "0fbb88ade2cef3a8439983519c766e56ddb1ba354ce6078ee7aab75c3c610fd0"
)
PREREGISTRATION_SHA256 = (
    "40a24744e956664b1c67950067021f45c54738f7d7dc0ae27e945e48f01ea05d"
)
IMPLEMENTATION_SHA256 = (
    "f8fccad4898322aadcf542b69fae7a5cefbfd7bf01b11ba718610ee70f01d8b5"
)
FINGERPRINT = (
    "080693343e1d7c3c8fcd420cf4131a66aca93a4cedba67cf12cc1508309ff237"
)
SOURCE_RESULT_SHA256 = (
    "e0bbb6120272627de59acf639e886d434ab53ec7ed0fc11874d77864a4dd1312"
)
SOURCE_DIAGNOSTICS_SHA256 = (
    "95334b9475bd73111f20f07747ec716fe0bf3c5ff44c14c2a91b78c825218d69"
)
INTERFACE_RESULT_SHA256 = (
    "16deb857f4449c66a22eac347c2c7e2b2cf23fbd5a8a1ad0daaf39660bf910f0"
)
INTERFACE_DIAGNOSTICS_SHA256 = (
    "3bfbf496e1cb85d79548849b533b4139c0737e7079a53e4e5bb2f974de7f9476"
)
REGIMES = ("composition", "extrapolation")
STRICT_ORDERS = (1, 2, 4)
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_frozen_cyclic_answer_projection.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-frozen-cyclic-answer-projection-preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/2026-08-11_tinyllm-frozen-cyclic-answer-projection.md"
)


@lru_cache(maxsize=32)
def _sha256_cached(path: Path, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256_cached(path, stat.st_size, stat.st_mtime_ns)


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


def _result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    artifacts = result.get("artifacts", {})
    source = result.get("source", {})
    expected_bins = {
        1: {
            "radius_1": list(range(4, 13)),
            "radius_8": list(range(3, 13)),
        },
        2: {
            "radius_1": list(range(2, 14)),
            "radius_8": list(range(2, 14)),
        },
        4: {
            "radius_1": [1, *range(5, 15)],
            "radius_8": [1, *range(5, 15)],
        },
    }
    if (
        _sha256(path) != RESULT_SHA256
        or result.get("schema_version") != EXPERIMENT_SCHEMA
        or result.get("hypothesis_id") != HYPOTHESIS_ID
        or result.get("status") != "completed"
        or result.get("evidence_role") != EVIDENCE_ROLE
        or result.get("classification") != CLASSIFICATION
        or result.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or result.get("scientific_fingerprint") != FINGERPRINT
        or result.get("gates")
        != {
            "finite": True,
            "identity_replay": True,
            "natural_index_replay": True,
            "source_bin_replay": True,
            "uniform_control": True,
            "validity": True,
        }
        or result.get("summary")
        != {
            "checkpoint_load_count": 0,
            "complete_order_count": 0,
            "fitted_parameters": 0,
            "maximum_identity_replay_error": 1.7881393432617188e-07,
            "maximum_uniform_control_error": 0.0,
            "smallest_complete_order": None,
            "trained_frontends": 0,
            "trained_heads": 0,
            "trained_models": 0,
            "trained_probes": 0,
        }
        or not Path(artifacts.get("diagnostics", "")).is_file()
        or _sha256(Path(artifacts["diagnostics"])) != DIAGNOSTICS_SHA256
        or not Path(source.get("result", "")).is_file()
        or _sha256(Path(source["result"])) != SOURCE_RESULT_SHA256
        or not Path(source.get("diagnostics", "")).is_file()
        or _sha256(Path(source["diagnostics"])) != SOURCE_DIAGNOSTICS_SHA256
        or not Path(source.get("interface_result", "")).is_file()
        or _sha256(Path(source["interface_result"])) != INTERFACE_RESULT_SHA256
        or not Path(source.get("interface_diagnostics", "")).is_file()
        or _sha256(Path(source["interface_diagnostics"]))
        != INTERFACE_DIAGNOSTICS_SHA256
        or not _finite(result)
    ):
        raise ValueError(f"invalid frozen cyclic answer-projection result {path}")
    if _sha256(RUNNER_PATH) != RUNNER_SHA256:
        raise ValueError(f"source artifact changed: {RUNNER_PATH}")
    if _sha256(PREREGISTRATION_PATH) != PREREGISTRATION_SHA256:
        raise ValueError(f"source artifact changed: {PREREGISTRATION_PATH}")
    for order in STRICT_ORDERS:
        record = result["orders"][str(order)]
        for regime in REGIMES:
            if (
                record[regime]["radius_1"]["reachable_bins"]
                != expected_bins[order]["radius_1"]
                or record[regime]["radius_8"]["reachable_bins"]
                != expected_bins[order]["radius_8"]
                or record[regime]["natural_metrics_pass"] is not False
                or record[regime]["shifted_natural_metrics_pass"] is not False
                or float(record[regime]["shifted_natural_metrics"]["exact_bin_accuracy"])
                != 0.0
            ):
                raise ValueError(f"cyclic answer-projection order {order} changed")
    return result


def _direct_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    orders = {}
    for order in STRICT_ORDERS:
        orders[str(order)] = {}
        for regime in REGIMES:
            record = result["orders"][str(order)][regime]
            orders[str(order)][regime] = {
                "radius_1_reachable_bins": record["radius_1"]["reachable_bins"],
                "radius_8_reachable_bins": record["radius_8"]["reachable_bins"],
                "radius_1_reachability": record["radius_1"][
                    "exact_bin_reachability"
                ],
                "radius_1_oracle_accuracy": record["radius_1"][
                    "minimum_cross_entropy_selection_metrics"
                ]["exact_bin_accuracy"],
                "radius_1_oracle_pass": record["radius_1"][
                    "minimum_cross_entropy_selection_pass"
                ],
                "natural_accuracy": record["natural_metrics"]["exact_bin_accuracy"],
                "natural_pass": record["natural_metrics_pass"],
                "shifted_accuracy": record["shifted_natural_metrics"][
                    "exact_bin_accuracy"
                ],
            }
    return {
        "experiment_id": result["experiment_id"],
        "classification": result["classification"],
        "scientific_fingerprint": result["scientific_fingerprint"],
        "complete_order_count": result["summary"]["complete_order_count"],
        "orders": orders,
        "identity_replay_error": result["summary"][
            "maximum_identity_replay_error"
        ],
        "uniform_control_error": result["summary"][
            "maximum_uniform_control_error"
        ],
    }


def build_frozen_cyclic_answer_projection_meta_hypothesis(
    results_path: Path,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    result = _result(results_path)
    direct = _direct_summary(result)
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "Fixed cyclic answer-row geometry repairs the TinyLLM chart hole",
            "category": "artifact-only answer-head causal decomposition",
            "description": (
                "Fixed C16 Fourier projections at orders one, two, and four "
                "intervene only on sealed d10 learned seed-29 answer logits."
            ),
            "question": (
                "Can answer-group structure alone restore complete reachable-bin "
                "coverage and natural task adequacy?"
            ),
            "prediction": (
                "At least one strict Fourier order gives full radius-one coverage, "
                "passes both oracle and natural task floors, and fails shifted controls."
            ),
            "created_at": result["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "not_confirmed_no_strict_order_completes_or_naturally_repairs_chart",
            "evidence_count": 1,
            "direct_experiment_count": 1,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "one outcome-selected d10 learned seed-29 stored answer curve; "
                "fixed C16 orders 1, 2, and 4; composition and extrapolation"
            ),
            "success_criteria": {
                "complete_radius_1_coverage": 16,
                "oracle_and_natural_task_floors": "both shifts",
                "strict_orders": list(STRICT_ORDERS),
                "semantic_shift_bins": 4,
            },
            "subclaims": {
                "complete_deployable_chart_repair": "contradicted_zero_of_three_orders",
                "complete_coverage_at_radius_8": "contradicted_zero_of_three_orders",
                "natural_task_repair": "contradicted_zero_of_six_order_shift_cells",
                "partial_answer_head_sensitivity": "supported_orders_one_two_and_four",
                "order_two_target_oracle": "supported_both_shifts_but_not_natural",
                "semantic_specificity": "supported_shifted_accuracy_zero_all_strict_cells",
                "population_frequency": "not_tested_single_outcome_selected_checkpoint",
            },
            "tags": [
                "tinyllm",
                "registered-post-outcome",
                "artifact-only",
                "cyclic-fourier",
                "answer-head",
                "negative-result",
            ],
            "explicitly_not_tested": result["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "Every strict order exchanges some winning regions for others; "
                "none reaches all sixteen bins or passes natural task adequacy."
            ),
            "confidence": 0.99,
            "confidence_assessment": "high_within_registered_single_checkpoint_projection_family",
            "num_direct_experiments": 1,
            "successful_direct_experiments": 1,
            "failed_direct_experiments": 0,
            "descriptive_metrics": direct,
            "key_insights": [
                "The answer head causally affects which rows can win, but fixed cyclic smoothing is insufficient.",
                "Order two improves the target-using oracle while leaving natural calibration failed.",
                "The scalar coordinate, continuation, and answer chart are co-adapted.",
            ],
            "suggested_hypotheses": [
                "A prospectively typed end-to-end scalar/continuation/readout interface outperforms a frozen-backbone readout-only arm across seeds."
            ],
            "completed_at": result["completed_at"],
        },
        "evidence": {"direct_tests": [direct]},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "result_sha256": RESULT_SHA256,
            "diagnostics_sha256": DIAGNOSTICS_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "scientific_fingerprint": FINGERPRINT,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "source_diagnostics_sha256": SOURCE_DIAGNOSTICS_SHA256,
            "interface_result_sha256": INTERFACE_RESULT_SHA256,
            "interface_diagnostics_sha256": INTERFACE_DIAGNOSTICS_SHA256,
            "artifact_validation": "all source artifacts, controls, strict-order bin sets, and natural/shifted gates revalidated",
        },
    }


def build_frozen_cyclic_answer_projection_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    result = _result(results_path)
    interface = json.loads(
        Path(result["source"]["interface_result"]).read_text(encoding="utf-8")
    )
    architecture = json.loads(
        Path(interface["source"]["result"]).read_text(encoding="utf-8")
    )
    completed = datetime.fromisoformat(
        result["completed_at"].replace("Z", "+00:00")
    )
    metrics = {
        "validity": 1.0,
        "complete_order_count": 0.0,
        "order_2_composition_oracle_accuracy": float(
            result["orders"]["2"]["composition"]["radius_1"][
                "minimum_cross_entropy_selection_metrics"
            ]["exact_bin_accuracy"]
        ),
        "order_2_extrapolation_oracle_accuracy": float(
            result["orders"]["2"]["extrapolation"]["radius_1"][
                "minimum_cross_entropy_selection_metrics"
            ]["exact_bin_accuracy"]
        ),
        "order_2_composition_natural_accuracy": float(
            result["orders"]["2"]["composition"]["natural_metrics"][
                "exact_bin_accuracy"
            ]
        ),
        "order_2_extrapolation_natural_accuracy": float(
            result["orders"]["2"]["extrapolation"]["natural_metrics"][
                "exact_bin_accuracy"
            ]
        ),
    }
    return [
        ExperimentResult(
            experiment_id=result["experiment_id"],
            hypothesis_id=HYPOTHESIS_ID,
            metrics=metrics,
            primary_metric=0.0,
            model_architecture=[10, 640],
            model_parameters=int(architecture["model_parameters"]),
            training_time=float(result["wall_seconds"]),
            model_checkpoint=interface["source"]["model_checkpoint"],
            observations=[
                f"Classification: {CLASSIFICATION}.",
                "Primary metric is the number of strict orders passing the complete chart gate.",
                "Artifact-only computation; zero checkpoint loads and fitted parameters.",
                f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
            ],
            anomalies=[],
            timestamp=completed,
        )
    ]


def store_frozen_cyclic_answer_projection_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_frozen_cyclic_answer_projection_meta_hypothesis(results_path)
    experiments = build_frozen_cyclic_answer_projection_experiment_results(
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
            or len(results.get("ids", [])) != 1
            or {item.get("hypothesis_id") for item in results.get("metadatas", [])}
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("frozen cyclic answer-projection ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": 1,
        }
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return record


__all__ = [
    "build_frozen_cyclic_answer_projection_experiment_results",
    "build_frozen_cyclic_answer_projection_meta_hypothesis",
    "store_frozen_cyclic_answer_projection_meta_hypothesis",
]
