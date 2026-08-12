"""Build and store the C3 temporal sensor mechanism decomposition."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-c3-temporal-sensor-mechanism.v1"
HYPOTHESIS_ID = "tinyllm-c3-temporal-sensor-mechanism-v1"
EVIDENCE_ROLE = "registered_post_outcome_artifact_only_sensor_mechanism"
CLASSIFICATION = "affine_identity_character_carries_learned_solution"
RESULT_SHA256 = "c3dbfecd7a6381c2129e4d99f135557f003ad8a225fa2b4d3f4fa0cb429f669b"
RUNNER_SHA256 = "8ca2143d8f7262b192c23eeb83aa56a6cc7a55d9cf6670c32a68c765122c14d3"
PREREGISTRATION_SHA256 = (
    "b32dbd5fce221b7eec50a7845f41d2dd036275397d69837e293db25bfecb72d5"
)
REPORT_SHA256 = "f5620f802f93dce2125781d53d55c68ecbb8a1e8949018d0a6d8de3ef8b1b404"
SOURCE_CAMPAIGN_SHA256 = (
    "4d023d9f9d77f64b1fe75970acd63461cfd5a64aa1de60c7954a2c671c427012"
)
SOURCE_CHECKPOINT_MANIFEST_SHA256 = (
    "e83832872f29d072d710859e022f4e17d1b6da6a9e16b63049f41d4ea2eb01a0"
)
SEEDS = (7, 17, 29, 41, 53)
SOURCE_ARMS = ("learned_true", "learned_target_shuffled")
REGIMES = ("composition", "extrapolation")
RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_sensor_mechanism/"
    "20260811_preregistered/campaign_results.json"
)
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_temporal_sensor_mechanism.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-temporal-sensor-mechanism-preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/2026-08-11_tinyllm-c3-temporal-sensor-mechanism.md"
)
SOURCE_CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_sensor_only/"
    "20260811_preregistered/campaign_results.json"
)


@lru_cache(maxsize=64)
def _sha256_cached(path: Path, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
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
    aggregates = result.get("aggregates", {})
    accounting = result.get("accounting", {})
    source_hashes = result.get("source_hashes", {})
    source = result.get("source", {})
    if (
        _sha256(path) != RESULT_SHA256
        or result.get("schema_version") != EXPERIMENT_SCHEMA
        or result.get("hypothesis_id") != HYPOTHESIS_ID
        or result.get("status") != "completed"
        or result.get("evidence_role") != EVIDENCE_ROLE
        or source_hashes.get("runner") != RUNNER_SHA256
        or source_hashes.get("preregistration") != PREREGISTRATION_SHA256
        or source_hashes.get("source_campaign") != SOURCE_CAMPAIGN_SHA256
        or source.get("checkpoint_manifest_sha256")
        != SOURCE_CHECKPOINT_MANIFEST_SHA256
        or aggregates.get("classification") != CLASSIFICATION
        or aggregates.get("primary_hypothesis_pass") is not True
        or aggregates.get("valid") is not True
        or aggregates.get("full_replay_pass_count") != 10
        or aggregates.get("true_affine_only_pass_count") != 5
        or aggregates.get("true_nonlinear_residual_only_pass_count") != 0
        or aggregates.get("shuffled_affine_only_pass_count") != 0
        or accounting
        != {
            "checkpoints_loaded": 10,
            "optimizer_steps": 0,
            "parameters_changed": 0,
            "target_free_complex_affine_fits": 10,
            "target_using_fits": 0,
            "tinyllm_models_instantiated": 0,
        }
        or result.get("environment", {}).get("device") != "cuda:0"
        or not _finite(result)
    ):
        raise ValueError(f"invalid C3 temporal sensor mechanism result {path}")

    for source_path, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (REPORT_PATH, REPORT_SHA256),
        (SOURCE_CAMPAIGN_PATH, SOURCE_CAMPAIGN_SHA256),
    ):
        if not source_path.is_file() or _sha256(source_path) != expected:
            raise ValueError(f"source artifact changed: {source_path}")

    cells = result.get("cells", [])
    identities = {(cell.get("seed"), cell.get("source_arm")) for cell in cells}
    expected_identities = {
        (seed, arm) for seed in SEEDS for arm in SOURCE_ARMS
    }
    if len(cells) != 10 or identities != expected_identities:
        raise ValueError("C3 sensor mechanism checkpoint population changed")
    for cell in cells:
        arm = cell["source_arm"]
        expected = arm == "learned_true"
        gates = cell.get("response_seed_gates", {})
        if (
            cell.get("valid") is not True
            or cell.get("state_unchanged") is not True
            or cell.get("state_sha256_before") != cell.get("state_sha256_after")
            or cell.get("full_replay_pass") is not True
            or cell.get("coefficient_reconstruction_pass") is not True
            or float(cell.get("affine_fit", {}).get("slope_magnitude", 0.0))
            < 1e-6
            or gates.get("full_replay") is not expected
            or gates.get("affine_only") is not expected
            or gates.get("nonlinear_residual_only") is not False
        ):
            raise ValueError(
                f"invalid C3 sensor mechanism cell {cell.get('seed')}/{arm}"
            )
        for regime in REGIMES:
            measured = cell.get("regimes", {}).get(regime, {})
            responses = measured.get("responses", {})
            if (
                float(measured.get("coefficient_reconstruction_maximum_error", 1.0))
                > 1e-6
                or float(measured.get("direct_encoder_posterior_maximum_error", 1.0))
                > 2e-6
                or measured.get("stored_metric_replay", {}).get("pass") is not True
                or responses.get("full_replay", {}).get("joint_pass") is not expected
                or responses.get("affine_only", {}).get("joint_pass") is not expected
                or responses.get("nonlinear_residual_only", {}).get("joint_pass")
                is not False
            ):
                raise ValueError(
                    f"invalid C3 sensor response {cell['seed']}/{arm}/{regime}"
                )
    return result


def _cells_by_seed(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    indexed = {
        (int(cell["seed"]), cell["source_arm"]): cell
        for cell in result["cells"]
    }
    return [
        {
            "seed": seed,
            "true": indexed[(seed, "learned_true")],
            "target_shuffled": indexed[(seed, "learned_target_shuffled")],
        }
        for seed in SEEDS
    ]


def build_c3_temporal_sensor_mechanism_meta_hypothesis(
    results_path: Path = RESULT_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    result = _result(results_path)
    direct = _cells_by_seed(result)
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": (
                "The learned exact-C3 sensors solve the temporal task through "
                "their affine identity-character response"
            ),
            "category": "registered post-outcome artifact-only causal mechanism",
            "description": (
                "Ten sealed sensor checkpoints are decomposed into a source-fitted "
                "complex affine response and its exact nonlinear complement."
            ),
            "question": (
                "Did successful task training rediscover the analytic affine "
                "identity mechanism, or require nonlinear shared-response harmonics?"
            ),
            "prediction": (
                "True affine patches pass at least four seeds, true residual patches "
                "and shuffled affine patches pass at most one, and all ten full "
                "checkpoints replay."
            ),
            "created_at": result["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": (
                "confirmed_affine_identity_character_causal_sufficiency_five_of_five"
            ),
            "evidence_count": 5,
            "direct_experiment_count": 5,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "five learned-true and five matched target-shuffled 184-parameter "
                "exact-C3 sensor checkpoints; source-fitted target-free affine "
                "responses; composition and outside-speed extrapolation"
            ),
            "success_criteria": {
                "true_affine_seed_passes_minimum": 4,
                "true_nonlinear_residual_seed_passes_maximum": 1,
                "target_shuffled_affine_seed_passes_maximum": 1,
                "full_checkpoint_replays_required": 10,
                "complete_carrier_and_task_gate_required_both_shifts": True,
            },
            "subclaims": {
                "affine_identity_causal_sufficiency": "supported_five_of_five",
                "nonlinear_response_necessity": "rejected_zero_of_five",
                "target_shuffle_specificity": "supported_zero_of_five",
                "full_checkpoint_replay": "supported_ten_of_ten",
                "functional_five_parameter_compression": (
                    "supported_in_registered_noiseless_task_scope"
                ),
                "parameter_equality_to_analytic_witness": "not_tested",
                "tinyllm_utility": "not_tested_tinyllm_absent",
                "noisy_or_new_group_transfer": "not_tested",
            },
            "tags": [
                "tinyllm",
                "c3",
                "sensor-mechanism",
                "affine-identity",
                "character-projection",
                "causal-patch",
                "artifact-only",
                "no-training",
            ],
            "explicitly_not_tested": result["method_boundaries"],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "Affine identity patches preserve the complete two-shift result in "
                "all five true seeds; nonlinear complements and shuffled affine "
                "controls preserve it in zero seeds."
            ),
            "confidence": 1.0,
            "confidence_assessment": (
                "registered_artifact_only_five_of_five_with_zero_of_five_controls"
            ),
            "num_direct_experiments": 5,
            "successful_direct_experiments": 5,
            "failed_direct_experiments": 0,
            "primary_seed_passes": 5,
            "descriptive_metrics": result["aggregates"],
            "key_insights": [
                (
                    "The successful 184-parameter sensors are functionally carried "
                    "by the same affine identity character as the analytic witness."
                ),
                (
                    "Task training selects the affine carrier orientation required "
                    "by the frozen temporal operator and physical decoder."
                ),
                (
                    "The isolated nonlinear response is neither sufficient nor "
                    "necessary under either registered shift."
                ),
                (
                    "The noiseless sensor-capacity branch is closed; extra sensor "
                    "capacity and harmonic scans are not licensed."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "A typed sensor remains useful when declared noise or missing "
                    "calibration makes the analytic affine rule insufficient."
                ),
                (
                    "A separately preregistered typed continuation can preserve the "
                    "carrier orientation without reintroducing the failed interface."
                ),
            ],
            "completed_at": result["completed_at"],
        },
        "evidence": {"direct_tests": direct},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "result_sha256": RESULT_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "report_sha256": REPORT_SHA256,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "source_checkpoint_manifest_sha256": (
                SOURCE_CHECKPOINT_MANIFEST_SHA256
            ),
            "artifact_validation": (
                "primary result, source campaign, twenty-checkpoint manifest, "
                "runner, preregistration, report, ten state identities, ten direct "
                "replays, affine/residual reconstruction, both shifts, and locked "
                "population classification revalidated"
            ),
        },
    }


def build_c3_temporal_sensor_mechanism_experiment_results(
    record: Mapping[str, Any],
    results_path: Path = RESULT_PATH,
) -> list[ExperimentResult]:
    del record
    result = _result(results_path)
    experiments = []
    completed = datetime.fromisoformat(
        result["completed_at"].replace("Z", "+00:00")
    )
    for pair in _cells_by_seed(result):
        true = pair["true"]
        shuffled = pair["target_shuffled"]
        metrics: dict[str, float] = {
            "validity": 1.0,
            "true_affine_joint_pass": 1.0,
            "true_nonlinear_residual_joint_pass": 0.0,
            "shuffled_affine_joint_pass": 0.0,
            "full_replay_pass_true": 1.0,
            "full_replay_pass_shuffled": 1.0,
            "optimizer_steps": 0.0,
            "parameters_changed": 0.0,
            "tinyllm_models_instantiated": 0.0,
            "source_complex_affine_r2": float(
                true["affine_fit"]["source_complex_r2"]
            ),
        }
        for regime in REGIMES:
            affine = true["regimes"][regime]["responses"]["affine_only"]
            residual = true["regimes"][regime]["responses"][
                "nonlinear_residual_only"
            ]
            control = shuffled["regimes"][regime]["responses"]["affine_only"]
            metrics[f"true_affine_{regime}_accuracy"] = float(
                affine["task"]["exact_bin_accuracy"]
            )
            metrics[f"true_affine_{regime}_correlation"] = float(
                affine["task"]["posterior_mean_correlation"]
            )
            metrics[f"true_residual_{regime}_accuracy"] = float(
                residual["task"]["exact_bin_accuracy"]
            )
            metrics[f"shuffled_affine_{regime}_accuracy"] = float(
                control["task"]["exact_bin_accuracy"]
            )
        experiments.append(
            ExperimentResult(
                experiment_id=(
                    f"tinyllm-c3-temporal-sensor-mechanism-seed{pair['seed']}"
                ),
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=1.0,
                model_architecture=[3, 16, 8, 2],
                model_parameters=184,
                training_time=0.0,
                model_checkpoint=true["checkpoint"]["path"],
                observations=[
                    f"Artifact-only affine/residual decomposition seed={pair['seed']}.",
                    "Target-free affine fit; frozen character, temporal operator, and decoder.",
                    "TinyLLM absent; no optimization or parameter change.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return experiments


def store_c3_temporal_sensor_mechanism_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c3_temporal_sensor_mechanism_meta_hypothesis(results_path)
    experiments = build_c3_temporal_sensor_mechanism_experiment_results(
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
            or len(stored.get("ids", [])) != 5
            or {
                item.get("hypothesis_id")
                for item in stored.get("metadatas", [])
            }
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("C3 sensor mechanism ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": 5,
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
    "build_c3_temporal_sensor_mechanism_experiment_results",
    "build_c3_temporal_sensor_mechanism_meta_hypothesis",
    "store_c3_temporal_sensor_mechanism_meta_hypothesis",
]
