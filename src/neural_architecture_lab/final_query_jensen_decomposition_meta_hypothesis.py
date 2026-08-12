"""Build and store the TinyLLM final-query Jensen decomposition result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-final-query-jensen-decomposition.v1"
HYPOTHESIS_ID = "tinyllm-final-query-jensen-decomposition-v1"
EVIDENCE_ROLE = "registered_post_outcome_frozen_final_query_diagnostic"
CLASSIFICATION = "generic_jensen_sufficient_with_layernorm_modulation"
CAMPAIGN_SHA256 = (
    "ab47fb645001e84d2b4e198f5bd269011bdc0438d056ea840f183156b98fc88a"
)
DIAGNOSTICS_SHA256 = (
    "cd8695fa9e0040657a661e7aae4324408a63ad5f3fd2cf858ba8a163ed4a8459"
)
IMPLEMENTATION_SHA256 = (
    "a9d06810ef198e6f58654cdde309c0225db75f82b5bd6a65268f7e2baf960353"
)
RUNNER_SHA256 = (
    "a53eaad8509b1bd1b08ad12e03caff396215591f24a2c9dab6a2bf78b9af0f33"
)
REGISTRATION_SHA256 = (
    "0c8e4d871637007405e8e57868a24a02962f52b04a2e1c87af72de0f71b5df69"
)
POSTERIOR_RUNNER_SHA256 = (
    "52b59f96e53e76c1794e9d2d251760a76eb2300f8e64e70dbe165090a5899895"
)
POSTERIOR_CAMPAIGN_SHA256 = (
    "93d9e22d766aa56943f0bd0c41b31ed25dc592e97ae0667863f3963588c38cde"
)
POSTERIOR_DIAGNOSTICS_SHA256 = (
    "ad1798b960375503381cb59725dfb84db9d823f488ab8073fea178d0399e0974"
)
PARENT_CAMPAIGN_SHA256 = (
    "b5d6b613683326024eeb00944a3d0aba4dd7a251dd22f5fa7e8561b5e9d6aae4"
)
PARENT_RESULT_SHA256 = (
    "79119a4fa6df16a2fe2d65c34ec1aca30072a0b53d3d8ea312c92b29f86d1476"
)
PARENT_DIAGNOSTICS_SHA256 = (
    "ef86c31d418c04b7ffd4bec3b66135057c4ca825b3e4da8263a920cbce0de1dd"
)
CHECKPOINT_SHA256 = (
    "8170856da5e6b1f8b7a7f1c2c2121ffd4c53edaaf2ea5aabe01fa14d2f063f3f"
)
SCIENTIFIC_FINGERPRINT = (
    "6f2a37301dc208b209473eae540541ba9145274bfa3d2fb0ee0780ddf3dfea1c"
)
REGIMES = ("training_support", "outside_range")
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_final_query_jensen_decomposition.py"
)
REGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-final-query-jensen-decomposition-preregistration.md"
)
POSTERIOR_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_final_query_task_kernel.py"
)
POSTERIOR_CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_final_query_task_kernel/"
    "20260810_d8_seed7_preregistered/campaign_results.json"
)
POSTERIOR_DIAGNOSTICS_PATH = Path(
    "data/experiments/tinyllm_final_query_task_kernel/"
    "20260810_d8_seed7_preregistered/diagnostics.npz"
)
PARENT_CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_task_relative_activation_barycenter/"
    "20260810_seed7_preregistered/campaign_results.json"
)
PARENT_RESULT_PATH = Path(
    "data/experiments/tinyllm_task_relative_activation_barycenter/"
    "20260810_seed7_preregistered/runs/d8/result.json"
)
PARENT_DIAGNOSTICS_PATH = Path(
    "data/experiments/tinyllm_task_relative_activation_barycenter/"
    "20260810_seed7_preregistered/runs/d8/diagnostics.npz"
)
CHECKPOINT_PATH = Path(
    "data/experiments/tinyllm_task_quotient_contrast/20260805_d6_d8/"
    "checkpoints/d8_cosine_interval_trained_seed7.pt"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=24)
def _require_source_digest(
    path: Path, expected: str, size: int, modified_ns: int
) -> None:
    del size, modified_ns
    if _sha256(path) != expected:
        raise ValueError(f"source artifact changed: {path}")


def _require_source(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ValueError(f"source artifact changed: {path}")
    stat = path.stat()
    _require_source_digest(path, expected, stat.st_size, stat.st_mtime_ns)


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
        "accounting_tolerance": 1e-10,
        "allow_underpowered": False,
        "batch_size": 64,
        "device": "cuda:1",
        "fiber_pairs": 512,
        "jensen_coverage_floor": 0.9,
        "jensen_tolerance": 1e-10,
        "material_gain_floor": 1e-4,
        "near_complete_remainder_ratio": 0.1,
        "regimes": list(REGIMES),
        "replay_tolerance": 2e-6,
        "target_pair_tolerance": 1e-12,
    }


def _campaign(path: Path) -> dict[str, Any]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    diagnostics_path = Path(campaign.get("artifacts", {}).get("diagnostics", ""))
    expected_summary = {
        "analyzed_pairs": 1024,
        "completed": 1,
        "excluded": 0,
        "failed": 0,
        "fitted_carriers": 0,
        "fitted_decoders": 0,
        "requested": 1,
        "retries": 0,
        "reused": 0,
        "scheduled": 1,
        "trained_models": 0,
        "trained_probes": 0,
    }
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or campaign.get("schema_version") != EXPERIMENT_SCHEMA
        or campaign.get("hypothesis_id") != HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != EVIDENCE_ROLE
        or campaign.get("classification") != CLASSIFICATION
        or campaign.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or campaign.get("scientific_fingerprint") != SCIENTIFIC_FINGERPRINT
        or campaign.get("configuration") != _expected_configuration()
        or campaign.get("summary") != expected_summary
        or campaign.get("gates")
        != {
            "primary_hypothesis_pass": True,
            "state_unchanged": True,
            "validity": True,
        }
        or set(campaign.get("regimes", {})) != set(REGIMES)
        or campaign.get("artifacts", {}).get("diagnostics_sha256")
        != DIAGNOSTICS_SHA256
        or not diagnostics_path.is_file()
        or _sha256(diagnostics_path) != DIAGNOSTICS_SHA256
        or campaign.get("source_digests", {}).get("runner") != RUNNER_SHA256
        or campaign.get("source_digests", {}).get("preregistration")
        != REGISTRATION_SHA256
        or campaign.get("source_hashes", {}).get("posterior_runner")
        != POSTERIOR_RUNNER_SHA256
        or campaign.get("source_hashes", {}).get("posterior_campaign")
        != POSTERIOR_CAMPAIGN_SHA256
        or campaign.get("source_hashes", {}).get("posterior_diagnostics")
        != POSTERIOR_DIAGNOSTICS_SHA256
        or campaign.get("source_hashes", {}).get("parent_campaign")
        != PARENT_CAMPAIGN_SHA256
        or campaign.get("source_hashes", {}).get("parent_d8_result")
        != PARENT_RESULT_SHA256
        or campaign.get("source_hashes", {}).get("parent_d8_diagnostics")
        != PARENT_DIAGNOSTICS_SHA256
        or campaign.get("source_hashes", {}).get("checkpoint_d8_cosine_interval")
        != CHECKPOINT_SHA256
        or campaign.get("state_record", {}).get("unchanged") is not True
        or campaign["state_record"].get("initial_model_state_sha256")
        != campaign["state_record"].get("final_model_state_sha256")
        or campaign["state_record"].get("initial_system_state_sha256")
        != campaign["state_record"].get("final_system_state_sha256")
        or not _finite(campaign)
    ):
        raise ValueError(f"invalid final-query Jensen campaign {path}")

    for source_path, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (REGISTRATION_PATH, REGISTRATION_SHA256),
        (POSTERIOR_RUNNER_PATH, POSTERIOR_RUNNER_SHA256),
        (POSTERIOR_CAMPAIGN_PATH, POSTERIOR_CAMPAIGN_SHA256),
        (POSTERIOR_DIAGNOSTICS_PATH, POSTERIOR_DIAGNOSTICS_SHA256),
        (PARENT_CAMPAIGN_PATH, PARENT_CAMPAIGN_SHA256),
        (PARENT_RESULT_PATH, PARENT_RESULT_SHA256),
        (PARENT_DIAGNOSTICS_PATH, PARENT_DIAGNOSTICS_SHA256),
        (CHECKPOINT_PATH, CHECKPOINT_SHA256),
    ):
        _require_source(source_path, expected)

    expected_contracts = {
        "accounting_identity": True,
        "control_pairwise_logit_jensen": True,
        "finite": True,
        "maximum_target_pair_error": 0.0,
        "pairwise_logit_jensen": True,
        "replay": True,
        "target_pair_identity": True,
    }
    for regime in REGIMES:
        detail = campaign["regimes"][regime]
        natural = detail.get("natural", {})
        control = detail.get("semantic_reassignment_control", {})
        replay = detail.get("replay", {})
        expected_near = regime == "training_support"
        if (
            detail.get("regime") != regime
            or detail.get("contracts") != expected_contracts
            or detail.get("gates")
            != {
                "generic_jensen_sufficient": True,
                "material_activation_gain": True,
                "near_complete_jensen": expected_near,
                "primary_regime_gate": True,
                "validity": True,
            }
            or detail.get("cohort_contract", {}).get("pass") is not True
            or float(natural.get("minimum_pair_jensen_gain", -1.0)) < -1e-10
            or float(natural.get("observed_activation_gain", 0.0)) < 1e-4
            or float(natural.get("jensen_gain", 0.0))
            < 0.9 * float(natural.get("observed_activation_gain", 0.0))
            or float(natural.get("maximum_absolute_accounting_residual", 1.0))
            > 1e-10
            or float(control.get("minimum_pair_jensen_gain", -1.0)) < -1e-10
            or replay.get("parent_replay_required") is not True
            or any(
                float(replay.get(name, 1.0)) != 0.0
                for name in (
                    "maximum_capture_error",
                    "maximum_baseline_replay_error",
                    "maximum_capture_posterior_replay_error",
                    "maximum_correct_barycenter_replay_error",
                    "maximum_semantic_reassignment_replay_error",
                    "maximum_replay_error",
                )
            )
        ):
            raise ValueError(
                f"invalid final-query Jensen regime {regime} in {path}"
            )
    return campaign


def _summary(campaign: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {
        "validity": 1.0,
        "primary_hypothesis_pass": 1.0,
        "state_unchanged": 1.0,
        "fiber_pairs_per_regime": 512.0,
    }
    for regime in REGIMES:
        detail = campaign["regimes"][regime]
        natural = detail["natural"]
        control = detail["semantic_reassignment_control"]
        for name in (
            "endpoint_cross_entropy",
            "logit_midpoint_cross_entropy",
            "activation_midpoint_cross_entropy",
            "jensen_gain",
            "nonlinear_remainder",
            "observed_activation_gain",
            "jensen_over_observed_gain",
            "nonlinear_over_jensen",
            "positive_observed_gain_pair_fraction",
            "mean_midpoint_posterior_js",
            "p95_midpoint_posterior_js",
            "mean_logit_remainder_over_endpoint_chord",
            "p95_logit_remainder_over_endpoint_chord",
        ):
            metrics[f"{regime}_{name}"] = float(natural[name])
        for name in (
            "jensen_gain",
            "nonlinear_remainder",
            "observed_activation_gain",
            "positive_observed_gain_pair_fraction",
        ):
            metrics[f"{regime}_control_{name}"] = float(control[name])
    return metrics


def build_final_query_jensen_decomposition_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/"
        "2026-08-10_tinyllm-final-query-jensen-decomposition.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    metrics = _summary(campaign)
    evidence = {
        "experiment_id": "tinyllm-final-query-jensen-d8-seed7",
        "preset": "d8",
        "seed": 7,
        "evidence_role": EVIDENCE_ROLE,
        "classification": campaign["classification"],
        "scientific_fingerprint": campaign["scientific_fingerprint"],
        "metrics": metrics,
    }
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM final-query Jensen decomposition",
            "category": "mechanistic_interpretability",
            "description": (
                "A no-fit diagnostic separates generic logit-space Jensen gain "
                "from final-layer-normalization geometry at the raw d8 final query."
            ),
            "question": (
                "Does the observed same-target activation-barycenter CE gain "
                "require favorable final-LN geometry?"
            ),
            "prediction": (
                "Generic logit Jensen gain supplies at least 90% of the observed "
                "activation gain on support and outside range."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": (
                "supported_registered_post_outcome_diagnostic_one_checkpoint"
            ),
            "evidence_count": 1,
            "direct_quality_experiment_count": 0,
            "diagnostic_experiment_count": 1,
            "fixed_checkpoint_count": 1,
            "evidence_role": EVIDENCE_ROLE,
            "power_profile": (
                "one_seed_one_depth_two_shift_1024_pair_post_outcome_diagnostic"
            ),
            "tested_scope": (
                "one retained raw d8 cosine seed-7 checkpoint, 512 exact fibers "
                "per regime, final post-MLP query, and native final-LN/head"
            ),
            "subclaims": {
                "generic_jensen_sufficient_support": "yes",
                "generic_jensen_sufficient_outside_range": "yes",
                "near_complete_jensen_support": "yes",
                "near_complete_jensen_outside_range": "no_adverse_ln_modulation",
                "jensen_is_quotient_specific": "no",
                "parent_posterior_preservation_rehabilitated": "no",
                "same_scope_retraining": "no",
            },
            "tags": [
                "tinyllm",
                "jensen-gap",
                "activation-barycenter",
                "final-layer-norm",
                "causal-interpretability",
                "frozen-checkpoint",
                "post-outcome-diagnostic",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "The generic logit midpoint supplies 100.35% and 136.10% of "
                "the observed CE gain; final LN is negligible on support and "
                "adverse outside range."
            ),
            "confidence": 0.99,
            "confidence_assessment": (
                "exact_frozen_checkpoint_decomposition_not_population_confirmation"
            ),
            "num_direct_experiments": 1,
            "descriptive_metrics": metrics,
            "key_insights": [
                "The raw final-query CE improvement is available from generic logit convexity in both shifts.",
                "Final layer normalization cancels 26.53% of the Jensen gain outside range rather than creating it.",
                "The target-changing control also obeys Jensen, proving the inequality is not quotient-specific.",
                "Lower average task loss cannot rescue a failed frozen-posterior preservation gate.",
            ],
            "suggested_hypotheses": [
                "Future activation-averaging studies should include a logit-midpoint Jensen comparator.",
                "Only causal continuation and semantic controls, not average-loss improvement, should license quotient interpretation.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": [evidence]},
        "source_artifacts": [
            str(results_path),
            str(Path(campaign["artifacts"]["diagnostics"])),
            str(report_path),
            str(REGISTRATION_PATH),
        ],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "diagnostics_sha256": DIAGNOSTICS_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "registration_sha256": REGISTRATION_SHA256,
            "posterior_campaign_sha256": POSTERIOR_CAMPAIGN_SHA256,
            "parent_campaign_sha256": PARENT_CAMPAIGN_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "scientific_fingerprint": SCIENTIFIC_FINGERPRINT,
        },
    }


def build_final_query_jensen_decomposition_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    campaign = _campaign(results_path)
    metrics = _summary(campaign)
    completed = datetime.fromisoformat(
        campaign["completed_at"].replace("Z", "+00:00")
    )
    return [
        ExperimentResult(
            experiment_id="tinyllm-final-query-jensen-d8-seed7",
            hypothesis_id=HYPOTHESIS_ID,
            metrics=metrics,
            primary_metric=min(
                metrics[f"{regime}_jensen_over_observed_gain"]
                for regime in REGIMES
            ),
            model_architecture=[8, 512],
            model_parameters=50_964_992,
            training_time=float(campaign["analysis_seconds"]),
            model_checkpoint=str(CHECKPOINT_PATH),
            observations=[
                "One retained raw d8 seed-7 checkpoint was frozen; no quantity was fit or trained.",
                "Generic logit Jensen gain was sufficient on support and outside range.",
                "The exact accounting and all parent posterior replays passed.",
                f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
            ],
            anomalies=[
                "Outside range, final layer normalization cancels 26.53% of the available Jensen gain."
            ],
            timestamp=completed,
        )
    ]


def store_final_query_jensen_decomposition_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_final_query_jensen_decomposition_meta_hypothesis(results_path)
    experiments = build_final_query_jensen_decomposition_experiment_results(
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
            or {item.get("hypothesis_id") for item in results.get("metadatas", [])}
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("final-query Jensen ChromaDB read-back failed")
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
    "build_final_query_jensen_decomposition_experiment_results",
    "build_final_query_jensen_decomposition_meta_hypothesis",
    "store_final_query_jensen_decomposition_meta_hypothesis",
]

