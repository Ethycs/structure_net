"""Build and store C3 relational connection function-class evidence."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c3-relational-connection-function-class.v1"
HYPOTHESIS_ID = "tinyllm-c3-relational-connection-function-class-v1"
EVIDENCE_ROLE = "no_training_exact_function_class_gradient_and_lifecycle_preflight"
CLASSIFICATION = (
    "connection_invariant_function_class_contains_transport_and_task_gradient"
)
RESULT_SHA256 = "2292e971bb655db246565675fece8dfd9e1546692b9b782e940a8bbef49de82c"
RUNNER_SHA256 = "7010794c3be5fda05a035e5a0b4a178aacd40c934dc1f5f7b36eb2eb03ea96b1"
PREREGISTRATION_SHA256 = (
    "421573a12a09a6782bd44b16e0f57e5e13a158f5bc740d19b8d4447964c5ae86"
)
REPORT_SHA256 = "66641c7f4b90328f775d525ec2fbe2166cdbc6dd33ed3832e45ef3fd74bf638a"
RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_relational_connection_function_class/"
    "20260811_preregistered/result.json"
)
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_relational_connection_function_class.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-relational-connection-function-class-"
    "preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-relational-connection-function-class.md"
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


def _expected_aggregates() -> dict[str, Any]:
    return {
        "analytic_witness_pass": True,
        "classification": CLASSIFICATION,
        "cpu_lifecycle_pass": True,
        "cuda_lifecycle_pass": True,
        "gradient_route_pass": True,
        "matched_sensor_readout_campaign_licensed": True,
        "source_contract_pass": True,
        "structural_invariance_pass": True,
        "unrestricted_tinyllm_training_licensed": False,
        "valid": True,
    }


def _expected_accounting() -> dict[str, int]:
    return {
        "closed_form_nonzero_witness_parameters": 6,
        "fresh_diagnostic_examples": 512,
        "historical_checkpoints_loaded": 0,
        "optimizer_steps": 0,
        "source_examples_reused": 40_960,
        "temporary_lifecycle_checkpoint_loads": 2,
        "temporary_lifecycle_checkpoints_created": 1,
        "tinyllm_models_instantiated": 0,
        "trained_parameters": 0,
    }


def _result(path: Path = RESULT_PATH) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    witness = result.get("analytic_witness", {})
    gradient = result.get("gradient_and_invariance", {})
    lifecycle = result.get("checkpoint_device_lifecycle", {})
    architecture = result.get("architecture_contract", {})
    if (
        _sha256(path) != RESULT_SHA256
        or result.get("schema_version") != EXPERIMENT_SCHEMA
        or result.get("hypothesis_id") != HYPOTHESIS_ID
        or result.get("evidence_role") != EVIDENCE_ROLE
        or result.get("status") != "completed"
        or result.get("aggregates") != _expected_aggregates()
        or result.get("accounting") != _expected_accounting()
        or result.get("source_hashes", {}).get("runner") != RUNNER_SHA256
        or architecture.get("pass") is not True
        or architecture.get("total_parameters") != 187
        or witness.get("pass") is not True
        or witness.get("nonzero_parameter_count") != 6
        or gradient.get("pass") is not True
        or gradient.get("finite") is not True
        or float(gradient.get("total_gradient_norm", 0.0)) < 1e-6
        or float(gradient.get("nonzero_gradient_fraction", 0.0)) < 0.90
        or float(gradient.get("action_loss_absolute_error", 1.0)) > 5e-6
        or float(gradient.get("action_gradient_maximum_error", 1.0)) > 2e-5
        or float(gradient.get("diagnostic_loss_decrease", 0.0)) < 1e-4
        or gradient.get("state_sha256_before")
        != gradient.get("state_sha256_after_restore")
        or lifecycle.get("pass") is not True
        or lifecycle.get("cpu_output_tensor_exact") is not True
        or lifecycle.get("cuda", {}).get("pass") is not True
        or not _finite(result)
    ):
        raise ValueError(f"invalid C3 relational function-class result {path}")
    cells = witness.get("cells", [])
    if len(cells) != 10:
        raise ValueError("relational function-class witness population changed")
    for cell in cells:
        if (
            cell.get("pass") is not True
            or cell.get("dataset_hash_match") is not True
            or float(cell.get("charged_character_maximum_error", 1.0)) > 2e-6
            or float(cell.get("scalar", {}).get("rmse", 1.0)) > 0.01
            or float(cell.get("task", {}).get("exact_bin_accuracy", 0.0)) < 0.98
            or float(cell.get("local_action_prediction_maximum_error", 1.0))
            > 2e-5
        ):
            raise ValueError("relational function-class witness cell changed")
    states = gradient.get("state_invariance", [])
    if len(states) != 3 or not all(item.get("pass") is True for item in states):
        raise ValueError("all-parameter implementation audit changed")
    for control in ("connection_shuffled_gradient", "target_shuffled_gradient"):
        if gradient.get(control, {}).get("pass") is not True:
            raise ValueError("gradient specificity control changed")
    for source, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (REPORT_PATH, REPORT_SHA256),
    ):
        if not source.is_file() or _sha256(source) != expected:
            raise ValueError(f"source artifact changed: {source}")
    return result


def build_c3_relational_connection_function_class_meta_hypothesis(
    results_path: Path = RESULT_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    result = _result(results_path)
    witness = result["analytic_witness"]
    gradient = result["gradient_and_invariance"]
    lifecycle = result["checkpoint_device_lifecycle"]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": (
                "An exact connection-invariant function class contains C3 "
                "transport and a specific task-gradient route"
            ),
            "category": "prospective no-training function-class and lifecycle preflight",
            "description": (
                "A 187-parameter charged encoder and neutral connection product "
                "are tested by a six-weight analytic witness, fresh gradients, "
                "action controls, checkpoint replay, and real CUDA execution."
            ),
            "question": (
                "Can a compact architecture exclude gauge-violating functions "
                "while containing and differentiably reaching the known observed-"
                "connection endpoint relation?"
            ),
            "prediction": (
                "The analytic witness passes all ten sealed cells, every sampled "
                "parameter state is action invariant, true gradients differ from "
                "connection/target shuffles, and CPU/CUDA lifecycle passes."
            ),
            "created_at": result["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": (
                "confirmed_exact_function_class_gradient_and_cuda_lifecycle_only"
            ),
            "evidence_count": 1,
            "direct_experiment_count": 1,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "187-parameter exact-C3 charged endpoint module; ten sealed "
                "composition/extrapolation cells; one fresh 512-example gradient "
                "cohort; CPU weights-only replay and RTX 3060 CUDA lifecycle"
            ),
            "success_criteria": {
                "character_error_maximum": 2e-6,
                "output_invariance_error_maximum": 2e-5,
                "gradient_norm_minimum": 1e-6,
                "gradient_nonzero_fraction_minimum": 0.90,
                "control_gradient_cosine_maximum": 0.95,
                "control_relative_difference_minimum": 0.20,
                "cpu_cuda_output_error_maximum": 5e-5,
            },
            "subclaims": {
                "analytic_transport_in_function_class": (
                    "supported_six_nonzero_parameter_witness"
                ),
                "all_parameter_action_invariance": (
                    "supported_by_architecture_and_three_state_audit"
                ),
                "specific_local_true_task_gradient": "supported",
                "cpu_checkpoint_replay": "supported_tensor_exact",
                "cuda_lifecycle": "supported_rtx3060",
                "global_trainability": "not_tested",
                "matched_sensor_readout_campaign": "licensed_not_yet_tested",
                "tinyllm_utility": "not_tested",
                "unrestricted_tinyllm_training": "not_licensed",
            },
            "tags": [
                "tinyllm",
                "c3",
                "connection",
                "function-class",
                "exact-equivariance",
                "analytic-witness",
                "gradient-specificity",
                "cuda-lifecycle",
                "no-training",
            ],
            "explicitly_not_tested": result["method_boundaries"],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "A six-weight state reproduces analytic transport on all ten "
                "sealed cells; the exact neutral architecture, fresh specific "
                "gradient, bit-exact CPU replay, and CUDA lifecycle all pass."
            ),
            "confidence": 1.0,
            "confidence_assessment": (
                "constructive_witness_architectural_identity_fresh_gradient_and_cuda"
            ),
            "num_direct_experiments": 1,
            "successful_direct_experiments": 1,
            "failed_direct_experiments": 0,
            "primary_seed_passes": 1,
            "descriptive_metrics": {
                "aggregates": result["aggregates"],
                "architecture_contract": result["architecture_contract"],
                "witness": witness,
                "gradient_and_invariance": gradient,
                "checkpoint_device_lifecycle": lifecycle,
            },
            "key_insights": [
                (
                    "Connection invariance can be removed from the optimization "
                    "problem by exposing only a transported neutral endpoint product."
                ),
                (
                    "The known analytic solution occupies six of 187 parameters, "
                    "so capacity is not the remaining uncertainty."
                ),
                (
                    "True gradients are strongly distinct from both connection- "
                    "and target-shuffled controls before any optimizer step."
                ),
                (
                    "The next uncertainty is global acquisition, not symmetry, "
                    "function-class containment, gradient access, or device lifecycle."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "Across five matched seeds, optimization of the 187-parameter "
                    "exact connection-invariant module recovers the endpoint relation "
                    "on composition and extrapolation, unlike connection/target shuffles."
                )
            ],
            "completed_at": result["completed_at"],
        },
        "evidence": {
            "direct_tests": [
                {
                    "experiment_id": (
                        "tinyllm-c3-relational-connection-function-class-preflight"
                    ),
                    "classification": result["aggregates"]["classification"],
                    "aggregates": result["aggregates"],
                    "architecture_contract": result["architecture_contract"],
                    "analytic_witness": witness,
                    "gradient_and_invariance": gradient,
                    "checkpoint_device_lifecycle": lifecycle,
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
            "source_hashes": result["source_hashes"],
            "artifact_validation": (
                "sealed result/source hashes, architecture parameter/path contract, "
                "ten-cell witness, three-state action audit, fresh gradient and "
                "controls, state restoration, CPU/CUDA lifecycle, and zero-training "
                "accounting revalidated"
            ),
        },
    }


def build_c3_relational_connection_function_class_experiment_results(
    record: Mapping[str, Any],
    results_path: Path = RESULT_PATH,
) -> list[ExperimentResult]:
    del record
    result = _result(results_path)
    witness = result["analytic_witness"]
    gradient = result["gradient_and_invariance"]
    lifecycle = result["checkpoint_device_lifecycle"]
    metrics: dict[str, float] = {
        "primary_contract_pass": 1.0,
        "analytic_witness_pass": 1.0,
        "structural_invariance_pass": 1.0,
        "gradient_route_pass": 1.0,
        "cpu_lifecycle_pass": 1.0,
        "cuda_lifecycle_pass": 1.0,
        "optimizer_steps": 0.0,
        "tinyllm_models_instantiated": 0.0,
        "module_parameter_count": 187.0,
        "nonzero_witness_parameter_count": 6.0,
        "gradient_norm": float(gradient["total_gradient_norm"]),
        "gradient_nonzero_fraction": float(
            gradient["nonzero_gradient_fraction"]
        ),
        "local_loss_decrease": float(gradient["diagnostic_loss_decrease"]),
        "connection_control_gradient_cosine": float(
            gradient["connection_shuffled_gradient"][
                "cosine_with_true_gradient"
            ]
        ),
        "target_control_gradient_cosine": float(
            gradient["target_shuffled_gradient"]["cosine_with_true_gradient"]
        ),
        "cpu_cuda_output_error": float(
            lifecycle["cuda"]["cpu_cuda_output_maximum_error"]
        ),
        "maximum_witness_character_error": max(
            float(cell["charged_character_maximum_error"])
            for cell in witness["cells"]
        ),
        "maximum_witness_rmse": max(
            float(cell["scalar"]["rmse"]) for cell in witness["cells"]
        ),
    }
    completed = datetime.fromisoformat(
        result["completed_at"].replace("Z", "+00:00")
    )
    return [
        ExperimentResult(
            experiment_id=(
                "tinyllm-c3-relational-connection-function-class-preflight"
            ),
            hypothesis_id=HYPOTHESIS_ID,
            metrics=metrics,
            primary_metric=1.0,
            model_architecture=[3, 16, 8, 2, 1],
            model_parameters=187,
            training_time=0.0,
            model_checkpoint="",
            observations=[
                "Closed-form six-nonzero-parameter analytic transport witness.",
                "Exact connection-neutral architecture and three-state action audit.",
                "Fresh control-specific gradient plus CPU and RTX 3060 CUDA lifecycle.",
                "Zero optimizer steps and zero TinyLLM instances.",
            ],
            anomalies=[],
            timestamp=completed,
        )
    ]


def store_c3_relational_connection_function_class_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c3_relational_connection_function_class_meta_hypothesis(
        results_path
    )
    experiments = build_c3_relational_connection_function_class_experiment_results(
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
            or {
                item.get("hypothesis_id")
                for item in stored.get("metadatas", [])
            }
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("C3 relational function-class ChromaDB read-back failed")
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
    "build_c3_relational_connection_function_class_experiment_results",
    "build_c3_relational_connection_function_class_meta_hypothesis",
    "store_c3_relational_connection_function_class_meta_hypothesis",
]
