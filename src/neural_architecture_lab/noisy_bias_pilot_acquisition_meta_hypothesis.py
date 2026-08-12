"""Build and store the TinyLLM finite noisy-bias pilot result."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

import numpy as np

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-noisy-bias-pilot-acquisition.v1"
HYPOTHESIS_ID = "tinyllm-noisy-bias-pilot-acquisition-v1"
EVIDENCE_ROLE = "preregistered_frozen_reused_draw_bias_pilot_titration"
CLASSIFICATION = "finite_noisy_pilot_repair_reliable"
IMPLEMENTATION_SHA256 = (
    "e034f234c3d0406cc62920de0b4a5efe591e1ca30f2d8d1f06c4fa6a29ab50b3"
)
CAMPAIGN_SHA256 = (
    "29d34222764215edb654d2741a85394223999a591cd7e2b44272612e1db8ccdf"
)
RESULT_MANIFEST_SHA256 = (
    "648f53acac88b625c357447bb930e55ae3312ff2555c1ef33ea72341ae527130"
)
RUNNER_SHA256 = (
    "827ff5b6d083b1e872954b5eb9a95a848632c555ecece1a51af54e1945573ed2"
)
PREREGISTRATION_SHA256 = (
    "2e19e9c7bf3908c97e29fd614a1801ef9e3eaec568a3babf3a1fe24adfa9830b"
)
PILOT_CONTRACT_SHA256 = (
    "303c19e3ef8d35476f479f267226b0b6d4a2bcf7fd7cfc7e628e5a3f1c883583"
)
PILOT_ARRAY_SHA256 = (
    "bfab87e36131fcfda8acdc33d7b9cce6e59c592dcca49c75e6d37b41410dcbc7"
)
PILOT_ARRAY_CONTENT_SHA256 = (
    "81cea5d30b15c90b4101c015880afc98111ac03b23704c1ff4b22b45567a054c"
)
SOURCE_REPAIR_CAMPAIGN_SHA256 = (
    "1996ac4c2534b62a25a2f52ceadfd21055a91bdadc81f38ccf01c6855da2b7d0"
)
SOURCE_ACQUISITION_CAMPAIGN_SHA256 = (
    "968f85010129d761268b4816d85ddd2ab578bbc93307e8a936e58fa891e89d93"
)
SOURCE_ACQUISITION_ARRAY_SHA256 = (
    "57eca80cccf1b916a60d79d5982bdbffe3b515cee7dfbee7645830448779aace"
)
SOURCE_REPAIR_RUNNER_SHA256 = (
    "fd6ea5108ccd733e360010c83a1a4a411512cbed239e3c9356a6a6bb77a6996a"
)
SOURCE_ACQUISITION_RUNNER_SHA256 = (
    "54c293d94582e4aa826772ac9c9a3791b5ed66c01aa9635fef75f433f7fe4e0d"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
SEEDS = (7, 17, 29, 41, 53)
COUNTS = (1, 4, 16, 64, 256)
REGIMES = ("composition", "extrapolation")
DRAW_COUNT = 16
EXPECTED_DRAW_PASSES = {
    "analytic_calibrated": {
        7: (15, 16, 16, 16, 16),
        17: (12, 16, 16, 16, 16),
        29: (15, 16, 16, 16, 16),
        41: (15, 16, 16, 16, 16),
        53: (9, 12, 16, 16, 16),
    },
    "learned_calibrated_equivariant": {
        7: (14, 16, 16, 15, 16),
        17: (11, 15, 16, 16, 16),
        29: (15, 16, 16, 16, 16),
        41: (14, 16, 16, 16, 16),
        53: (16, 16, 16, 16, 16),
    },
}
EXPECTED_SOURCE_FULL = {
    "analytic_calibrated": {7: False, 17: False, 29: False, 41: False, 53: True},
    "learned_calibrated_equivariant": {
        7: False,
        17: True,
        29: True,
        41: False,
        53: True,
    },
}
EXPECTED_WRONG_SIGN = {
    "analytic_calibrated": {7: False, 17: False, 29: False, 41: False, 53: True},
    "learned_calibrated_equivariant": {seed: False for seed in SEEDS},
}
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_noisy_bias_pilot_acquisition.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-noisy-bias-pilot-acquisition-preregistration.md"
)
SOURCE_REPAIR_CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_bias_reference_recentering/"
    "20260810_d10_preregistered_v2/campaign_results.json"
)
SOURCE_ACQUISITION_CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_acquisition_draw_stability/"
    "20260810_d16_preregistered/campaign_results.json"
)
SOURCE_ACQUISITION_ARRAY_PATH = Path(
    "data/experiments/tinyllm_acquisition_draw_stability/"
    "20260810_d16_preregistered/acquisition_draw_errors.npz"
)
SOURCE_REPAIR_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_bias_reference_recentering.py"
)
SOURCE_ACQUISITION_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_acquisition_draw_stability.py"
)
DATASET_HASHES = {
    "composition": "b025623e23f534ca3670f49f1bafc1c3979d1ca834aec2223f9c3166be8f52a6",
    "extrapolation": "f2f1e8c2c06b38fbecf856f0679e9861d9fab3e1c2a41358e55043f4d309a214",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _array_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=128)
def _require_digest(path: Path, expected: str, size: int, modified_ns: int) -> None:
    del size, modified_ns
    if _sha256(path) != expected:
        raise ValueError(f"source artifact changed: {path}")


def _require_source(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ValueError(f"source artifact changed: {path}")
    stat = path.stat()
    _require_digest(path, expected, stat.st_size, stat.st_mtime_ns)


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


def _expected_configuration() -> dict[str, Any]:
    return {
        "acquisition_channels": [0, 1],
        "acquisition_draw_seed_root": 81_027_026,
        "acquisition_split": "composition",
        "allow_underpowered": False,
        "array_correlation_ceiling": 0.05,
        "array_mean_tolerance": 0.03,
        "array_std_maximum": 1.05,
        "array_std_minimum": 0.95,
        "batch_size": 256,
        "conditions": list(CONDITIONS),
        "counts": list(COUNTS),
        "devices": ["cuda:0", "cuda:1", "cuda:2"],
        "draw_count": DRAW_COUNT,
        "maximum_control_seed_passes": 1,
        "natural_accuracy_loss_ceiling": 0.05,
        "natural_circular_error_increase_ceiling": math.pi / 16.0,
        "natural_cross_entropy_increase_ceiling": 0.1,
        "pilot_noise_sigma": 0.03125 / math.sqrt(2.0),
        "required_draw_passes": 15,
        "required_seed_passes": 4,
        "sample_limit": None,
        "seeds": list(SEEDS),
        "selected_noise_sigma": 0.03125,
        "source_acquisition_root": (
            "data/experiments/tinyllm_acquisition_draw_stability/"
            "20260810_d16_preregistered"
        ),
        "source_repair_root": (
            "data/experiments/tinyllm_bias_reference_recentering/"
            "20260810_d10_preregistered_v2"
        ),
        "source_replay_tolerance": 2e-6,
    }


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    aggregates = value.get("aggregates", {})
    population = value.get("population", {})
    counts = population.get("counts", {})
    expected_counts = {
        "1": ({"analytic_calibrated": 13, "learned_calibrated_equivariant": 14}, 12),
        "4": ({condition: 16 for condition in CONDITIONS}, 16),
        "16": ({condition: 16 for condition in CONDITIONS}, 16),
        "64": ({condition: 16 for condition in CONDITIONS}, 16),
        "256": ({condition: 16 for condition in CONDITIONS}, 16),
    }
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("status") != "completed"
        or value.get("configuration") != _expected_configuration()
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or value.get("source_digests")
        != {
            "preregistration": PREREGISTRATION_SHA256,
            "runner": RUNNER_SHA256,
            "source_acquisition_runner": SOURCE_ACQUISITION_RUNNER_SHA256,
            "source_repair_runner": SOURCE_REPAIR_RUNNER_SHA256,
        }
        or value.get("source_repair_campaign_sha256")
        != SOURCE_REPAIR_CAMPAIGN_SHA256
        or value.get("source_acquisition_campaign_sha256")
        != SOURCE_ACQUISITION_CAMPAIGN_SHA256
        or value.get("source_acquisition_arrays_sha256")
        != SOURCE_ACQUISITION_ARRAY_SHA256
        or value.get("dataset_hashes") != DATASET_HASHES
        or value.get("pilot_contract_sha256") != PILOT_CONTRACT_SHA256
        or value.get("pilot_arrays_sha256") != PILOT_ARRAY_SHA256
        or value.get("pilot_array_content_sha256")
        != PILOT_ARRAY_CONTENT_SHA256
        or value.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or _json_hash(value.get("results", [])) != RESULT_MANIFEST_SHA256
        or value.get("pilot_contract", {}).get("pass") is not True
        or aggregates
        != {
            "classification": CLASSIFICATION,
            "integrity_valid": True,
            "maximum_control_seed_passes": 1,
            "pilot_contract_pass": True,
            "primary_evaluable": True,
            "primary_hypothesis_pass": True,
            "required_draw_passes": 15,
            "required_seed_passes": 4,
            "shakedown_outcome_classification": None,
            "smallest_reliable_count": 4,
            "source_controls_pass": True,
            "valid": True,
        }
        or population.get("smallest_reliable_count") != 4
        or population.get("source_controls_pass") is not True
        or population.get("integrity_valid") is not True
        or len(value.get("results", [])) != 10
        or not _finite(value)
    ):
        raise ValueError(f"invalid noisy-bias pilot campaign {path}")
    for count, (arm_passes, complete_passes) in expected_counts.items():
        level = counts.get(count, {})
        if (
            level.get("arm_population_draw_passes") != arm_passes
            or level.get("complete_draw_passes") != complete_passes
            or level.get("draws") != DRAW_COUNT
        ):
            raise ValueError(f"invalid noisy-bias count {count} in {path}")
    for source_path, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (SOURCE_REPAIR_CAMPAIGN_PATH, SOURCE_REPAIR_CAMPAIGN_SHA256),
        (SOURCE_ACQUISITION_CAMPAIGN_PATH, SOURCE_ACQUISITION_CAMPAIGN_SHA256),
        (SOURCE_ACQUISITION_ARRAY_PATH, SOURCE_ACQUISITION_ARRAY_SHA256),
        (SOURCE_REPAIR_RUNNER_PATH, SOURCE_REPAIR_RUNNER_SHA256),
        (SOURCE_ACQUISITION_RUNNER_PATH, SOURCE_ACQUISITION_RUNNER_SHA256),
    ):
        _require_source(source_path, expected)
    pilot_path = Path(value["pilot_arrays"])
    _require_source(pilot_path, PILOT_ARRAY_SHA256)
    with np.load(pilot_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    if _array_digest(arrays) != PILOT_ARRAY_CONTENT_SHA256:
        raise ValueError(f"invalid noisy-bias pilot array content {pilot_path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {
        (entry["condition"], int(entry["seed"])): entry
        for entry in campaign["results"]
    }
    expected_cells = {
        (condition, seed) for condition in CONDITIONS for seed in SEEDS
    }
    if set(entries) != expected_cells:
        raise ValueError(f"invalid noisy-bias result population {path}")
    output = []
    fingerprints: set[str] = set()
    for condition in CONDITIONS:
        for seed in SEEDS:
            entry = entries[(condition, seed)]
            result_path = Path(entry["path"])
            diagnostics_path = Path(entry["diagnostics_path"])
            detail = json.loads(result_path.read_text(encoding="utf-8"))
            provenance = detail.get("provenance", {})
            observed_passes = tuple(
                sum(
                    int(
                        detail["draw_count_seed_gates"][f"draw_{draw:02d}"][
                            str(count)
                        ]
                    )
                    for draw in range(DRAW_COUNT)
                )
                for count in COUNTS
            )
            if (
                _sha256(result_path) != entry.get("result_sha256")
                or _sha256(diagnostics_path) != entry.get("diagnostics_sha256")
                or entry.get("validity") is not True
                or detail.get("schema_version") != EXPERIMENT_SCHEMA
                or detail.get("hypothesis_id") != HYPOTHESIS_ID
                or detail.get("evidence_role") != EVIDENCE_ROLE
                or detail.get("status") != "completed"
                or detail.get("condition") != condition
                or int(detail.get("seed", -1)) != seed
                or detail.get("configuration") != _expected_configuration()
                or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
                or detail.get("scientific_fingerprint")
                != entry.get("scientific_fingerprint")
                or detail.get("pilot_contract_sha256") != PILOT_CONTRACT_SHA256
                or detail.get("pilot_arrays_sha256") != PILOT_ARRAY_SHA256
                or observed_passes != EXPECTED_DRAW_PASSES[condition][seed]
                or detail.get("source_exact_pilot_seed_gate") is not True
                or detail.get("source_full_plus_seed_gate")
                is not EXPECTED_SOURCE_FULL[condition][seed]
                or detail.get("wrong_sign_draw0_m256_seed_gate")
                is not EXPECTED_WRONG_SIGN[condition][seed]
                or any(
                    detail.get("gates", {}).get(name) is not True
                    for name in (
                        "clean_posterior_replay",
                        "finite",
                        "source_metric_replay",
                        "state_unchanged",
                        "validity",
                    )
                )
                or set(detail.get("draw_count_seed_gates", {}))
                != {f"draw_{draw:02d}" for draw in range(DRAW_COUNT)}
                or any(
                    set(detail["draw_count_seed_gates"][f"draw_{draw:02d}"])
                    != {str(count) for count in COUNTS}
                    for draw in range(DRAW_COUNT)
                )
                or not _finite(detail)
            ):
                raise ValueError(f"invalid noisy-bias pilot result {result_path}")
            for artifact_key, digest_key in (
                ("source_repair_result", "source_repair_result_sha256"),
                ("source_repair_diagnostics", "source_repair_diagnostics_sha256"),
                ("source_result", "source_result_sha256"),
                ("model_checkpoint", "model_checkpoint_sha256"),
                ("frontend_checkpoint", "frontend_checkpoint_sha256"),
            ):
                _require_source(
                    Path(provenance[artifact_key]), provenance[digest_key]
                )
            fingerprint = detail["scientific_fingerprint"]
            if fingerprint in fingerprints:
                raise ValueError(f"duplicate noisy-bias fingerprint {result_path}")
            fingerprints.add(fingerprint)
            source = json.loads(
                Path(provenance["source_result"]).read_text(encoding="utf-8")
            )
            detail["_model_parameters"] = int(source["model_parameters"])
            output.append(detail)
    return output


def _detail_summary(detail: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "validity": 1.0,
        "source_exact_pilot_seed_pass": float(
            detail["source_exact_pilot_seed_gate"]
        ),
        "source_full_plus_seed_pass": float(detail["source_full_plus_seed_gate"]),
        "wrong_sign_seed_pass": float(detail["wrong_sign_draw0_m256_seed_gate"]),
    }
    for count in COUNTS:
        pass_count = sum(
            int(
                detail["draw_count_seed_gates"][f"draw_{draw:02d}"][str(count)]
            )
            for draw in range(DRAW_COUNT)
        )
        accuracy_losses = []
        circular_increases = []
        cross_entropy_increases = []
        for regime in REGIMES:
            for draw in range(DRAW_COUNT):
                gate = detail["regimes"][regime]["draws"][
                    f"draw_{draw:02d}"
                ]["counts"][str(count)]["natural_utility"]
                accuracy_losses.append(float(gate["accuracy_loss"]))
                circular_increases.append(float(gate["circular_error_increase"]))
                cross_entropy_increases.append(
                    float(gate["cross_entropy_increase"])
                )
        summary[f"m{count}_draw_passes"] = float(pass_count)
        summary[f"m{count}_draw_pass_frequency"] = pass_count / DRAW_COUNT
        summary[f"m{count}_mean_accuracy_loss"] = mean(accuracy_losses)
        summary[f"m{count}_max_accuracy_loss"] = max(accuracy_losses)
        summary[f"m{count}_max_circular_error_increase"] = max(
            circular_increases
        )
        summary[f"m{count}_max_cross_entropy_increase"] = max(
            cross_entropy_increases
        )
    summary["analysis_seconds"] = float(detail["analysis_seconds"])
    return summary


def build_noisy_bias_pilot_acquisition_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-10_tinyllm-noisy-bias-pilot-acquisition.md"
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
            "name": "TinyLLM finite noisy shared-bias pilot acquisition",
            "category": "mechanistic_interpretability",
            "description": (
                "A frozen nested count ladder tests whether noisy zero-signal "
                "pilot averaging preserves the exact observed shared-bias repair."
            ),
            "question": (
                "Can a finite target-free noisy pilot recover both frozen "
                "structured TinyLLM populations under composition and extrapolation?"
            ),
            "prediction": (
                "At least fifteen of sixteen complete m=256 draws must pass, "
                "with exact-source and wrong-sign controls preserved."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": (
                "preregistered_finite_noisy_pilot_m4_population_repair_confirmed"
            ),
            "evidence_count": 16,
            "direct_experiment_count": 16,
            "stored_checkpoint_summary_count": 10,
            "fixed_checkpoint_count": 10,
            "evidence_role": EVIDENCE_ROLE,
            "power_profile": "sixteen_draw_two_arm_five_checkpoint_frozen_titration",
            "tested_scope": (
                "sixteen reused independent Gaussian pilot draws, five nested "
                "counts, ten retained d8/N3 systems, composition and extrapolation"
            ),
            "subclaims": {
                "m256_complete_draw_repair": "supported_sixteen_of_sixteen",
                "m256_checkpoint_cells": "supported_160_of_160",
                "smallest_reliable_population_count": "four",
                "m4_complete_draw_repair": "supported_sixteen_of_sixteen",
                "m4_system_level_ceiling": "not_supported_two_checkpoint_failures",
                "m1_complete_draw_repair": "insufficient_twelve_of_sixteen",
                "wrong_sign_specificity": (
                    "supported_analytic_one_of_five_learned_zero_of_five"
                ),
                "exact_pilot_source": "supported_five_of_five_both_arms",
                "new_randomness_or_fitting": "none",
                "exact_gaussian_pilot_count_branch": "closed",
                "same_scope_retraining": "not_licensed",
            },
            "tags": [
                "tinyllm",
                "shared-bias",
                "zero-signal-pilot",
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
                "Both five-checkpoint populations pass all sixteen complete "
                "draws at m=4 and m=256, while m=1 passes only twelve and the "
                "wrong-sign control remains below its ceiling."
            ),
            "confidence": 0.98,
            "confidence_assessment": (
                "strong_preregistered_support_with_declared_population_and_noise_scope"
            ),
            "num_direct_experiments": 16,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "population": campaign["population"],
                "pilot_contract": campaign["pilot_contract"],
                "checkpoint_summaries": tests,
            },
            "key_insights": [
                "Four target-free noisy pilot observations recover both frozen populations on all sixteen draws.",
                "One observation is insufficient, passing only twelve complete draws.",
                "Population redundancy makes m=4 reliable even though two individual checkpoints retain draw failures.",
                "The wrong-sign and inherited target-changing controls preserve signed and task specificity.",
                "The exact independent-Gaussian pilot-count branch is closed without retraining or fitting.",
            ],
            "suggested_hypotheses": [
                "Persistent between-pilot drift will introduce a non-vanishing repair floor.",
                "Temporally correlated pilot errors will require more than four observations at the same marginal variance.",
                "Per-system acquisition budgets should use a precision guarantee rather than the population threshold.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"checkpoint_summaries": tests},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "pilot_contract_sha256": PILOT_CONTRACT_SHA256,
            "pilot_array_sha256": PILOT_ARRAY_SHA256,
            "pilot_array_content_sha256": PILOT_ARRAY_CONTENT_SHA256,
            "source_repair_campaign_sha256": SOURCE_REPAIR_CAMPAIGN_SHA256,
            "source_acquisition_campaign_sha256": (
                SOURCE_ACQUISITION_CAMPAIGN_SHA256
            ),
            "source_acquisition_array_sha256": SOURCE_ACQUISITION_ARRAY_SHA256,
        },
    }


def build_noisy_bias_pilot_acquisition_experiment_results(
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
                primary_metric=float(summary["m4_draw_pass_frequency"]),
                model_architecture=[8, int(detail["_model_parameters"])],
                model_parameters=int(detail["_model_parameters"]),
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["model_checkpoint"],
                observations=[
                    f"Condition: {detail['condition']}.",
                    "The acquisition draw is the replication unit; this row summarizes one fixed checkpoint.",
                    "Pilot arrays, TinyLLM, front end, task head, and bias estimator remained frozen.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=(
                    [
                        f"This checkpoint passes only {int(summary['m4_draw_passes'])}/16 m=4 draws."
                    ]
                    if int(summary["m4_draw_passes"]) < DRAW_COUNT
                    else []
                ),
                timestamp=completed,
            )
        )
    return output


def store_noisy_bias_pilot_acquisition_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_noisy_bias_pilot_acquisition_meta_hypothesis(results_path)
    experiments = build_noisy_bias_pilot_acquisition_experiment_results(
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
            raise RuntimeError("noisy-bias pilot ChromaDB read-back failed")
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
    "build_noisy_bias_pilot_acquisition_experiment_results",
    "build_noisy_bias_pilot_acquisition_meta_hypothesis",
    "store_noisy_bias_pilot_acquisition_meta_hypothesis",
]
