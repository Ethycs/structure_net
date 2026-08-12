"""Build and store the TinyLLM calibrated architecture-family result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-calibrated-architecture-replication.v1"
HYPOTHESIS_ID = "tinyllm-calibrated-architecture-replication-v1"
EVIDENCE_ROLE = "prospective_architecture_family_replication"
CLASSIFICATION = "structured_closure_not_architecture_stable"
CAMPAIGN_SHA256 = (
    "656d9814a032d1899810e81d398adf935cea3e1116712460e2062da188a0c9e2"
)
RESULT_MANIFEST_SHA256 = (
    "903b61792fd624c520f2e563caf7742a40717993d129fcf489f618975bf41052"
)
ARTIFACT_MANIFEST_SHA256 = (
    "97550245f9f52a93c84dc6faf486d9837e1460832316ece51d23e23d15c37a4b"
)
IMPLEMENTATION_SHA256 = (
    "5ac391fd2d719a219cff9673c92004c4ae3e7401b242d8561844e26cd6bf21ab"
)
CAMPAIGN_FINGERPRINT = (
    "b8dc7ce7c717fdf166e83468813fd6771fb729d4d6bc94e620e8b4e7b271ff5d"
)
PREFLIGHT_SHA256 = (
    "ea8b12c006dc860d41715095c4a111bb1fcdd844a859c83427122ad1d02bbe6a"
)
PREFLIGHT_ARTIFACT_SHA256 = (
    "3ff8a376a0dea2b0bc1e89c65afd92292bd9e3241e9a328a758548d675582c63"
)
RUNNER_SHA256 = (
    "661384de2eac23d95dbc550e0ecf49b14ebfeb01ef47fc1a8164e7e3b2b0ca90"
)
PREREGISTRATION_SHA256 = (
    "36c1a8c35823fda3076b6a73648facd7fc18513c3c969bfa297ca9c0b34c4c77"
)
PRESETS = ("d6", "d10")
CONDITIONS = (
    "raw_calibrated",
    "analytic_calibrated",
    "learned_calibrated_equivariant",
)
STRUCTURED = CONDITIONS[1:]
SEEDS = (7, 17, 29, 41, 53)
CUTS = ("pre_block", "block0_post_attention", "block0_post_mlp", "full")
REGIMES = ("composition", "extrapolation")
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_calibrated_architecture_replication.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-calibrated-architecture-replication-preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-10_tinyllm-calibrated-architecture-replication.md"
)


@lru_cache(maxsize=256)
def _sha256_cached(path: Path, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
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


def _expected_aggregates() -> dict[str, Any]:
    counts = {
        ("d6", "raw_calibrated"): (0, 5, 3, 0, False),
        ("d6", "analytic_calibrated"): (5, 5, 5, 5, True),
        ("d6", "learned_calibrated_equivariant"): (5, 1, 5, 1, False),
        ("d10", "raw_calibrated"): (0, 5, 4, 0, False),
        ("d10", "analytic_calibrated"): (5, 3, 5, 3, False),
        ("d10", "learned_calibrated_equivariant"): (5, 1, 5, 1, False),
    }
    joint_seeds = {
        ("d6", "raw_calibrated"): (),
        ("d6", "analytic_calibrated"): SEEDS,
        ("d6", "learned_calibrated_equivariant"): (29,),
        ("d10", "raw_calibrated"): (),
        ("d10", "analytic_calibrated"): (7, 41, 53),
        ("d10", "learned_calibrated_equivariant"): (17,),
    }
    arms: dict[str, Any] = {}
    for preset in PRESETS:
        arms[preset] = {}
        for condition in CONDITIONS:
            representation, task, causal, joint, success = counts[(preset, condition)]
            passing = set(joint_seeds[(preset, condition)])
            arms[preset][condition] = {
                "causal_all_cuts_pass_count": causal,
                "joint_pass_by_seed": {
                    str(seed): seed in passing for seed in SEEDS
                },
                "joint_pass_count": joint,
                "population_control_pass": True,
                "representation_pass_count": representation,
                "shuffle_preblock_pass_count": 0,
                "success": success,
                "task_adequacy_pass_count": task,
                "valid_count": 5,
            }
    return {
        "arms": arms,
        "classification": CLASSIFICATION,
        "controls_pass": True,
        "primary_hypothesis_pass": False,
        "required_seed_passes": 4,
        "valid": True,
    }


def _campaign(path: Path) -> dict[str, Any]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    expected_sources = {
        "calibrated_source": (
            "73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77"
        ),
        "causal_source": (
            "0169a1653695ebe7e55b7a3f49b12401f73439f5a784c2ddd5584a4b685761c6"
        ),
        "preflight": (
            "12b8c635a1a119f2cbf2933fbb27b15a93eb499e2ddf7289503139d44b92d972"
        ),
        "preregistration": PREREGISTRATION_SHA256,
        "runner": RUNNER_SHA256,
    }
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or campaign.get("schema_version") != EXPERIMENT_SCHEMA
        or campaign.get("hypothesis_id") != HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != EVIDENCE_ROLE
        or campaign.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or campaign.get("campaign_fingerprint") != CAMPAIGN_FINGERPRINT
        or campaign.get("preflight_sha256") != PREFLIGHT_SHA256
        or campaign.get("implementation_sources") != expected_sources
        or campaign.get("summary")
        != {
            "completed": 30,
            "failed": 0,
            "requested": 30,
            "reused": 0,
            "scheduled": 30,
        }
        or campaign.get("aggregates") != _expected_aggregates()
        or not _finite(campaign)
    ):
        raise ValueError(f"invalid calibrated architecture campaign {path}")
    preflight = Path(campaign.get("preflight", ""))
    if not preflight.is_file() or _sha256(preflight) != PREFLIGHT_ARTIFACT_SHA256:
        raise ValueError(f"invalid calibrated architecture preflight {preflight}")
    if _sha256(RUNNER_PATH) != RUNNER_SHA256:
        raise ValueError(f"source artifact changed: {RUNNER_PATH}")
    if _sha256(PREREGISTRATION_PATH) != PREREGISTRATION_SHA256:
        raise ValueError(f"source artifact changed: {PREREGISTRATION_PATH}")
    return campaign


def _details(results_path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(results_path)
    details = []
    for preset in PRESETS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                path = (
                    results_path.parent
                    / "runs"
                    / preset
                    / condition
                    / f"seed_{seed}"
                    / "result.json"
                )
                detail = json.loads(path.read_text(encoding="utf-8"))
                artifacts = detail.get("artifacts", {})
                artifact_contract = (
                    ("checkpoint", "checkpoint_sha256"),
                    ("frontend_checkpoint", "frontend_checkpoint_sha256"),
                    ("diagnostics", "diagnostics_sha256"),
                )
                valid_artifacts = all(
                    Path(artifacts.get(name, "")).is_file()
                    and _sha256(Path(artifacts[name])) == artifacts.get(digest)
                    for name, digest in artifact_contract
                )
                if (
                    detail.get("schema_version") != EXPERIMENT_SCHEMA
                    or detail.get("hypothesis_id") != HYPOTHESIS_ID
                    or detail.get("status") != "completed"
                    or detail.get("evidence_role")
                    != (
                        "prospective_matched_comparator"
                        if condition == "raw_calibrated"
                        else "prospective_primary"
                    )
                    or detail.get("preset") != preset
                    or detail.get("condition") != condition
                    or int(detail.get("seed", -1)) != seed
                    or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
                    or not detail.get("scientific_fingerprint")
                    or detail.get("preflight_sha256") != PREFLIGHT_SHA256
                    or detail.get("gates", {}).get("validity") is not True
                    or not valid_artifacts
                    or not _finite(detail)
                ):
                    raise ValueError(f"invalid calibrated architecture cell {path}")
                details.append(detail)
    if len(details) != 30:
        raise ValueError("calibrated architecture campaign must contain 30 cells")
    return details


def _detail_summary(detail: Mapping[str, Any]) -> dict[str, float]:
    probes = []
    for cut in ("frontend", "full"):
        evaluations = detail["representation"]["cuts"][cut]["probe"]["evaluations"]
        probes.extend(evaluations[regime] for regime in REGIMES)
    task = detail["representation"]["task_metrics"]
    return {
        "representation_pass": float(detail["gates"]["representation_pass"]),
        "task_adequacy_pass": float(detail["gates"]["task_adequacy_pass"]),
        "causal_all_cuts_pass": float(detail["gates"]["causal_all_cuts_pass"]),
        "joint_seed_pass": float(detail["gates"]["joint_seed_pass"]),
        "minimum_cosine_correlation": min(float(p["cosine_pearson"]) for p in probes),
        "maximum_conditional_branch_accuracy": max(
            float(p["balanced_accuracy"]) for p in probes
        ),
        "maximum_conditional_log_loss_gain": max(
            float(p["conditional_log_loss_gain_over_cosine_only"]) for p in probes
        ),
        "composition_exact_bin_accuracy": float(task["composition"]["exact_bin_accuracy"]),
        "extrapolation_exact_bin_accuracy": float(task["extrapolation"]["exact_bin_accuracy"]),
        "causal_cut_pass_count": float(sum(detail["causal"]["cut_seed_gates"].values())),
    }


def build_calibrated_architecture_replication_meta_hypothesis(
    results_path: Path,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    summaries = [
        {
            "experiment_id": detail["experiment_id"],
            "preset": detail["preset"],
            "condition": detail["condition"],
            "seed": detail["seed"],
            "scientific_fingerprint": detail["scientific_fingerprint"],
            "checkpoint": detail["artifacts"]["checkpoint"],
            "checkpoint_sha256": detail["artifacts"]["checkpoint_sha256"],
            **_detail_summary(detail),
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "Calibrated quotient closure is stable across TinyLLM presets",
            "category": "architecture-family replication",
            "description": (
                "Matched d6 and d10 raw, analytic, and learned-equivariant models "
                "test representation geometry, natural task adequacy, and exact "
                "four-cut orbit-barycenter causal closure."
            ),
            "question": (
                "Does the complete calibrated structured result generalize from "
                "the retained d8 anchor to fresh d6 and d10 presets?"
            ),
            "prediction": (
                "Both structured arms pass the joint representation, task-adequacy, "
                "causal, and control gate in at least four of five seeds in both presets."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "not_confirmed_task_adequacy_not_architecture_stable",
            "evidence_count": 30,
            "direct_experiment_count": 30,
            "tested_scope": (
                "fresh d6/d10 TinyLLM preset family; calibrated cosine task; "
                "three matched arms; five seeds; retained d8 same-seed anchor"
            ),
            "success_criteria": {
                "required_seed_passes": 4,
                "structured_arms": list(STRUCTURED),
                "fresh_presets": list(PRESETS),
                "joint_seedwise_gate": (
                    "representation plus d8-relative task adequacy plus four-cut "
                    "causal closure plus validity controls on both shifts"
                ),
            },
            "subclaims": {
                "structured_representation_family_stable": "supported_all_twenty_seeds",
                "structured_causal_closure_family_stable": "supported_all_twenty_seeds_all_cuts",
                "natural_task_adequacy_family_stable": "contradicted",
                "d6_analytic_complete": "supported_five_of_five",
                "d10_analytic_complete": "not_supported_three_of_five",
                "d6_learned_complete": "not_supported_one_of_five",
                "d10_learned_complete": "not_supported_one_of_five",
                "raw_representation_specificity": "supported_zero_of_ten",
                "full_preregistered_hypothesis": "not_confirmed",
            },
            "tags": [
                "tinyllm",
                "prospective-replication",
                "architecture-family",
                "equivariant-front-end",
                "causal-activation-patching",
                "negative-result",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "All structured representation and causal populations pass, but "
                "task adequacy reaches only 3/5 d10 analytic and 1/5 in each "
                "learned population, below the required 4/5."
            ),
            "confidence": 0.99,
            "confidence_assessment": (
                "high_confidence_preregistered_negative_with_positive_causal_subclaim"
            ),
            "num_direct_experiments": 30,
            "successful_direct_experiments": 30,
            "failed_direct_experiments": 0,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "checkpoint_summaries": summaries,
            },
            "key_insights": [
                "Structured quotient geometry generalizes across both fresh presets.",
                "Exact orbit-barycenter causal closure passes every structured cut and seed.",
                "Natural task usefulness is a separate architecture- and optimization-sensitive conjunct.",
                "The learned d10 extrapolation failures are too large to treat as threshold noise.",
            ],
            "suggested_hypotheses": [
                "Exact cosine substitution repairs failures caused by sensor calibration.",
                "A one-dimensional reachability oracle separates coordinate calibration from continuation failure.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": summaries},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "checkpoint_frontend_diagnostics_manifest_sha256": ARTIFACT_MANIFEST_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "campaign_fingerprint": CAMPAIGN_FINGERPRINT,
            "preflight_sha256": PREFLIGHT_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "artifact_validation": "all 30 declared checkpoint, front-end, and diagnostics digests revalidated",
        },
    }


def build_calibrated_architecture_replication_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    campaign = _campaign(results_path)
    completed = datetime.fromisoformat(
        campaign["completed_at"].replace("Z", "+00:00")
    )
    output = []
    architectures = {
        "d6": (6, 384),
        "d10": (10, 640),
    }
    for detail in _details(results_path):
        summary = _detail_summary(detail)
        layers, width = architectures[detail["preset"]]
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=summary,
                primary_metric=float(summary["joint_seed_pass"]),
                model_architecture=[layers, width],
                model_parameters=int(detail["model_parameters"]),
                training_time=float(detail["wall_seconds"]),
                model_checkpoint=detail["artifacts"]["checkpoint"],
                observations=[
                    f"Preset: {detail['preset']}.",
                    f"Condition: {detail['condition']}.",
                    "Primary metric is the locked complete joint seed gate.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return output


def store_calibrated_architecture_replication_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_calibrated_architecture_replication_meta_hypothesis(results_path)
    experiments = build_calibrated_architecture_replication_experiment_results(
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
            raise RuntimeError("calibrated architecture ChromaDB read-back failed")
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
    "build_calibrated_architecture_replication_experiment_results",
    "build_calibrated_architecture_replication_meta_hypothesis",
    "store_calibrated_architecture_replication_meta_hypothesis",
]
