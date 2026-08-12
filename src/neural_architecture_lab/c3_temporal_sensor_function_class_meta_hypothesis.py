"""Build and store the C3 temporal sensor function-class preflight evidence."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c3-temporal-sensor-function-class.v1"
HYPOTHESIS_ID = "tinyllm-c3-temporal-sensor-function-class-v1"
EVIDENCE_ROLE = "prospective_no_training_c3_sensor_function_class_preflight"
CLASSIFICATION = "existing_c3_sensor_contains_analytic_carrier_and_task_gradient"
RESULT_SHA256 = "6a01db25ebc2ed15d202884c39f16db685d5218647b0bb209e2e5a737696a383"
RUNNER_SHA256 = "95331fa823f193449af1c46f961f22f4aaf902300695f179912b75d8ed4bcade"
PREREGISTRATION_SHA256 = (
    "83f20896853cb7612618c93d3302a5bfa0b2091b6c957625eecbe3451401701f"
)
REPORT_SHA256 = "f6b7910bc37249ce236f67e7e817150cb2df72a935f889d70013ad2879684605"
RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_sensor_function_class/"
    "20260811_preregistered/result.json"
)
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_temporal_sensor_function_class.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-temporal-sensor-function-class-"
    "preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-temporal-sensor-function-class.md"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
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


def _result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    gates = result.get("gates", {})
    accounting = result.get("accounting", {})
    witness = result.get("witness", {})
    gradient = result.get("gradient_route", {})
    if (
        _sha256(path) != RESULT_SHA256
        or result.get("schema_version") != EXPERIMENT_SCHEMA
        or result.get("hypothesis_id") != HYPOTHESIS_ID
        or result.get("evidence_role") != EVIDENCE_ROLE
        or result.get("status") != "completed"
        or result.get("classification") != CLASSIFICATION
        or result.get("source_hashes", {}).get("runner") != RUNNER_SHA256
        or result.get("source_contract_pass") is not True
        or gates
        != {
            "analytic_function_class_witness": True,
            "campaign_licensed": True,
            "no_checkpoint_loaded": True,
            "no_tinyllm_instantiated": True,
            "source_contract": True,
            "target_gradient_route": True,
            "zero_optimizer_steps": True,
        }
        or accounting
        != {
            "checkpoints_loaded": 0,
            "closed_form_nonzero_witness_parameters": 5,
            "optimizer_steps": 0,
            "sensor_family_parameters": 184,
            "tinyllm_models_instantiated": 0,
            "trained_parameters": 0,
        }
        or witness.get("pass") is not True
        or witness.get("parameter_count") != 184
        or witness.get("nonzero_parameter_count") != 5
        or float(witness.get("scalar_reconstruction_maximum_absolute_error", 1.0))
        > 5e-7
        or gradient.get("pass") is not True
        or gradient.get("finite") is not True
        or float(gradient.get("total_gradient_norm", 0.0)) < 1e-6
        or float(gradient.get("nonzero_gradient_fraction", 0.0)) < 0.90
        or float(gradient.get("deck_loss_absolute_error", 1.0)) > 1e-6
        or float(gradient.get("deck_gradient_maximum_absolute_error", 1.0))
        > 2e-5
        or float(gradient.get("diagnostic_loss_decrease", 0.0)) < 1e-4
        or gradient.get("state_sha256_before")
        != gradient.get("state_sha256_after_restore")
        or not _finite(result)
    ):
        raise ValueError(f"invalid C3 sensor function-class preflight {path}")
    for regime in ("composition", "extrapolation"):
        measured = witness.get("regimes", {}).get(regime, {})
        if (
            measured.get("pass") is not True
            or measured.get("fixed_interval_task_pass") is not True
            or measured.get("quantizer_saturation_count") != 0
            or measured.get("sample_count") != 4096
            or float(measured.get("carrier_maximum_absolute_error", 1.0)) > 2e-6
            or float(measured.get("maximum_nonidentity_action_error", 1.0)) > 2e-6
            or float(measured.get("temporal_target_correlation", 0.0)) < 0.99
            or float(measured.get("temporal_target_rmse", 1.0)) > 0.08
            or abs(float(measured.get("shuffled_target_correlation", 1.0))) > 0.10
            or float(measured.get("shuffled_target_rmse", 0.0)) < 0.80
        ):
            raise ValueError(
                f"invalid C3 sensor function-class regime {regime} in {path}"
            )
    for source, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (REPORT_PATH, REPORT_SHA256),
    ):
        if not source.is_file() or _sha256(source) != expected:
            raise ValueError(f"source artifact changed: {source}")
    return result


def build_c3_temporal_sensor_function_class_meta_hypothesis(
    results_path: Path = RESULT_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    result = _result(results_path)
    witness = result["witness"]
    gradient = result["gradient_route"]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": (
                "The existing exact-C3 sensor contains the analytic temporal "
                "carrier and a usable task-gradient path"
            ),
            "category": "prospective no-training function-class preflight",
            "description": (
                "A five-nonzero-parameter GELU-difference witness and a local "
                "derivative audit test the 184-parameter sensor before training."
            ),
            "question": (
                "Can the existing learned sensor represent the known analytic "
                "carrier, and can the fixed temporal task reach its parameters?"
            ),
            "prediction": (
                "The closed-form witness passes both registered shifts and the "
                "true fixed-interface loss has a finite nonzero sensor gradient."
            ),
            "created_at": result["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": (
                "confirmed_function_class_capacity_and_local_gradient_only"
            ),
            "evidence_count": 1,
            "direct_experiment_count": 1,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "existing 184-parameter exact-C3 sensor; registered 4096-example "
                "composition and extrapolation cohorts; fixed temporal operator "
                "and sixteen-bin decoder; CPU no-training derivative audit"
            ),
            "success_criteria": {
                "carrier_error_maximum": 2e-6,
                "deck_action_error_maximum": 2e-6,
                "temporal_correlation_minimum": 0.99,
                "temporal_rmse_maximum": 0.08,
                "gradient_norm_minimum": 1e-6,
                "gradient_nonzero_fraction_minimum": 0.90,
                "local_loss_decrease_minimum": 1e-4,
            },
            "subclaims": {
                "analytic_carrier_in_function_class": (
                    "supported_five_nonzero_parameter_witness"
                ),
                "composition_fixed_pipeline": "supported_complete_gate",
                "extrapolation_fixed_pipeline": "supported_complete_gate",
                "exact_c3_action": "supported_both_nonidentity_elements",
                "target_roll_specificity": "supported_both_shifts",
                "local_true_task_gradient_route": "supported",
                "global_trainability": "not_tested",
                "tinyllm_utility": "not_tested",
                "conditional_sensor_only_campaign": "licensed_not_yet_tested",
            },
            "tags": [
                "tinyllm",
                "c3",
                "function-class",
                "analytic-witness",
                "gradient-audit",
                "no-training",
                "sensor-only",
            ],
            "explicitly_not_tested": result["method_boundaries"],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "A closed-form five-parameter state matches the analytic carrier "
                "below 7e-7 on both shifts, the fixed task passes, and the true "
                "task reaches the random sensor with a finite downhill gradient."
            ),
            "confidence": 1.0,
            "confidence_assessment": (
                "constructive_witness_and_preregistered_local_derivative_contract"
            ),
            "num_direct_experiments": 1,
            "successful_direct_experiments": 1,
            "failed_direct_experiments": 0,
            "primary_seed_passes": 1,
            "descriptive_metrics": {
                "parameter_count": witness["parameter_count"],
                "nonzero_witness_parameters": witness[
                    "nonzero_parameter_count"
                ],
                "scalar_reconstruction_error": witness[
                    "scalar_reconstruction_maximum_absolute_error"
                ],
                "composition": witness["regimes"]["composition"],
                "extrapolation": witness["regimes"]["extrapolation"],
                "gradient_route": gradient,
            },
            "key_insights": [
                (
                    "The predecessor sensor family is not missing the analytic "
                    "carrier; two GELU units reconstruct the scalar exactly."
                ),
                (
                    "The known temporal law and metric decoder provide a direct "
                    "task-gradient route without a transformer continuation."
                ),
                (
                    "The remaining scientific uncertainty is optimization and "
                    "acquisition, not function-class capacity."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "Across five random seeds, task-only training of the existing "
                    "sensor against the frozen temporal operator and decoder "
                    "recovers a support-stable analytic carrier specifically."
                )
            ],
            "completed_at": result["completed_at"],
        },
        "evidence": {
            "direct_tests": [
                {
                    "experiment_id": (
                        "tinyllm-c3-temporal-sensor-function-class-preflight"
                    ),
                    "classification": result["classification"],
                    "gates": result["gates"],
                    "witness": witness,
                    "gradient_route": gradient,
                    "accounting": result["accounting"],
                }
            ]
        },
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "result_sha256": RESULT_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "report_sha256": REPORT_SHA256,
            "artifact_validation": (
                "result, source hashes, gates, both shift cells, gradient route, "
                "zero-training accounting, state restoration, and report revalidated"
            ),
        },
    }


def build_c3_temporal_sensor_function_class_experiment_results(
    record: Mapping[str, Any],
    results_path: Path = RESULT_PATH,
) -> list[ExperimentResult]:
    del record
    result = _result(results_path)
    witness = result["witness"]
    gradient = result["gradient_route"]
    metrics: dict[str, float] = {
        "primary_contract_pass": 1.0,
        "source_contract_pass": 1.0,
        "witness_pass": 1.0,
        "gradient_route_pass": 1.0,
        "optimizer_steps": 0.0,
        "tinyllm_models_instantiated": 0.0,
        "sensor_parameter_count": 184.0,
        "nonzero_witness_parameter_count": 5.0,
        "gradient_norm": float(gradient["total_gradient_norm"]),
        "gradient_nonzero_fraction": float(
            gradient["nonzero_gradient_fraction"]
        ),
        "local_loss_decrease": float(gradient["diagnostic_loss_decrease"]),
    }
    for regime in ("composition", "extrapolation"):
        item = witness["regimes"][regime]
        metrics[f"{regime}_carrier_error"] = float(
            item["carrier_maximum_absolute_error"]
        )
        metrics[f"{regime}_temporal_correlation"] = float(
            item["temporal_target_correlation"]
        )
        metrics[f"{regime}_temporal_rmse"] = float(
            item["temporal_target_rmse"]
        )
        metrics[f"{regime}_fixed_accuracy"] = float(
            item["fixed_interval_task"]["exact_bin_accuracy"]
        )
    completed = datetime.fromisoformat(
        result["completed_at"].replace("Z", "+00:00")
    )
    return [
        ExperimentResult(
            experiment_id=(
                "tinyllm-c3-temporal-sensor-function-class-preflight"
            ),
            hypothesis_id=HYPOTHESIS_ID,
            metrics=metrics,
            primary_metric=1.0,
            model_architecture=[3, 16, 8, 2],
            model_parameters=184,
            training_time=0.0,
            model_checkpoint="",
            observations=[
                "Closed-form five-nonzero-parameter analytic witness.",
                "Zero optimizer steps, zero TinyLLM instances, and zero checkpoints.",
                "Primary metric is the preregistered function-class and gradient-route contract.",
            ],
            anomalies=[
                "One of 184 scalar parameters has exactly zero numerical gradient in the registered random audit."
            ],
            timestamp=completed,
        )
    ]


def store_c3_temporal_sensor_function_class_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c3_temporal_sensor_function_class_meta_hypothesis(results_path)
    experiments = build_c3_temporal_sensor_function_class_experiment_results(
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
        stored = logger.experiments_collection.get(
            ids=storage["result_hashes"], include=["metadatas"]
        )
        if (
            hypothesis.get("ids") != [HYPOTHESIS_ID]
            or len(stored.get("ids", [])) != 1
            or {item.get("hypothesis_id") for item in stored.get("metadatas", [])}
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("C3 sensor function-class ChromaDB read-back failed")
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
    "build_c3_temporal_sensor_function_class_experiment_results",
    "build_c3_temporal_sensor_function_class_meta_hypothesis",
    "store_c3_temporal_sensor_function_class_meta_hypothesis",
]
