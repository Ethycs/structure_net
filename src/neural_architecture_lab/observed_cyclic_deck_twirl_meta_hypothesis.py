"""Build and store the preregistered TinyLLM observed cyclic-deck evidence."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-observed-cyclic-deck-twirl-front.v1"
HYPOTHESIS_ID = "tinyllm-observed-cyclic-deck-twirl-front-v1"
EVIDENCE_ROLE = "preregistered_frozen_observed_action_replication"
CAMPAIGN_SHA256 = (
    "7a1b099495f7ecb6c3eeea7c9b836411a5baee709eb6b51ab8103f88927e8a86"
)
IMPLEMENTATION_SHA256 = (
    "0c377e7a928e1926e916a981fec7464f0d4575e7f0e229bbbf1b7fc5298a0e56"
)
RESULT_MANIFEST_SHA256 = (
    "52648e21fc8c3c47d62d588449fcc9cd1d36ac63302d836c5785eeb130f20184"
)
SOURCE_LADDER_SHA256 = (
    "cf12b76691da41b7bc15e47570bce324f6aaefc7c9f670ef68db1fa4d9421046"
)
SOURCE_DECK_SHA256 = (
    "a3c14ce7022b7301344beaca876e0d454445c972a57de69c9cd4cd89098036b3"
)
PREREGISTRATION_SHA256 = (
    "4ea38b10c77eede7a67d07c9d253a80ad058cd879ac49995cf762ad680f1dd59"
)
DEGREES = (2, 3)
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
CUTS = (
    "frontend",
    "block_0_pre_attention",
    "block_0_post_attention",
    "block_0_post_mlp",
    "block_1_post_attention",
    "block_1_post_mlp",
    "block_2_post_attention",
    "block_2_post_mlp",
    "full",
)
MODEL_PARAMETERS = 29_957_760


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
    provenance = value.get("provenance", {})
    degree_results = aggregates.get("degrees", {})
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
        "fitted_action_parameters": 0,
    }
    expected_counts = {
        "2": {
            "mature_observed_twirl_pass_count": 5,
            "computational_cover_pass_count": 5,
            "front_replication_pass_count": 4,
            "control_validity_pass_count": 5,
            "semantic_control_preserved_count": 0,
            "all_primary_gates_pass": True,
        },
        "3": {
            "mature_observed_twirl_pass_count": 5,
            "computational_cover_pass_count": 5,
            "front_replication_pass_count": 2,
            "control_validity_pass_count": 5,
            "semantic_control_preserved_count": 0,
            "all_primary_gates_pass": False,
        },
    }
    source_ladder = Path(provenance.get("source_ladder_campaign", ""))
    source_deck = Path(provenance.get("source_deck_campaign", ""))
    preregistration = Path(provenance.get("preregistration", ""))
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or value.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or tuple(config.get("degrees", ())) != DEGREES
        or tuple(config.get("seeds", ())) != SEEDS
        or int(config.get("evaluation_orbits", -1)) != 256
        or int(config.get("map_points", -1)) != 192
        or int(config.get("required_seed_passes", -1)) != 4
        or int(config.get("maximum_control_seed_passes", -1)) != 1
        or int(config.get("front_cut_tolerance", -1)) != 1
        or config.get("allow_underpowered") is not False
        or summary != expected_summary
        or aggregates.get("valid") is not True
        or aggregates.get("primary_hypothesis_pass") is not False
        or aggregates.get("classification") != "c2_only_observed_front"
        or any(
            any(degree_results.get(k, {}).get(name) != expected for name, expected in fields.items())
            for k, fields in expected_counts.items()
        )
        or len(value.get("action_contracts", {})) != 10
        or not all(
            item.get("pass") is True
            for item in value.get("action_contracts", {}).values()
        )
        or provenance.get("source_ladder_campaign_sha256") != SOURCE_LADDER_SHA256
        or provenance.get("source_deck_campaign_sha256") != SOURCE_DECK_SHA256
        or provenance.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or not source_ladder.is_file()
        or _sha256(source_ladder) != SOURCE_LADDER_SHA256
        or not source_deck.is_file()
        or _sha256(source_deck) != SOURCE_DECK_SHA256
        or not preregistration.is_file()
        or _sha256(preregistration) != PREREGISTRATION_SHA256
        or not _finite(value)
    ):
        raise ValueError(f"invalid observed cyclic-deck campaign {path}")
    manifest = hashlib.sha256(
        json.dumps(
            value.get("results", []), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if manifest != RESULT_MANIFEST_SHA256:
        raise ValueError(f"invalid cyclic-deck result manifest {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {
        (int(entry["k"]), int(entry["seed"])): entry
        for entry in campaign.get("results", [])
    }
    expected = {(k, seed) for k in DEGREES for seed in SEEDS}
    if set(entries) != expected:
        raise ValueError(f"invalid cyclic-deck result index {path}")
    expected_front_seeds = {2: {7, 29, 41, 53}, 3: {17, 41}}
    output = []
    fingerprints: set[str] = set()
    for k in DEGREES:
        for seed in SEEDS:
            entry = entries[(k, seed)]
            result_path = Path(entry["path"])
            diagnostics_path = Path(entry["diagnostics_path"])
            detail = json.loads(result_path.read_text(encoding="utf-8"))
            provenance = detail.get("provenance", {})
            source_result = Path(provenance.get("source_result", ""))
            reference_result = Path(provenance.get("reference_result", ""))
            seed_gates = detail.get("seed_gates", {})
            if (
                _sha256(result_path) != entry.get("result_sha256")
                or _sha256(diagnostics_path) != entry.get("diagnostics_sha256")
                or detail.get("schema_version") != EXPERIMENT_SCHEMA
                or detail.get("hypothesis_id") != HYPOTHESIS_ID
                or detail.get("status") != "completed"
                or detail.get("evidence_role") != EVIDENCE_ROLE
                or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
                or int(detail.get("k", -1)) != k
                or int(detail.get("seed", -1)) != seed
                or detail.get("scientific_fingerprint")
                != entry.get("scientific_fingerprint")
                or detail.get("action_contract", {}).get("pass") is not True
                or detail.get("gates", {}).get("validity") is not True
                or seed_gates.get("mature_observed_twirl") is not True
                or seed_gates.get("computational_cover") is not True
                or seed_gates.get("control_validity") is not True
                or seed_gates.get("semantic_control_preserved") is not False
                or seed_gates.get("front_replication")
                != (seed in expected_front_seeds[k])
                or not source_result.is_file()
                or _sha256(source_result) != provenance.get("source_result_sha256")
                or not reference_result.is_file()
                or _sha256(reference_result)
                != provenance.get("reference_result_sha256")
                or not _finite(detail)
            ):
                raise ValueError(f"invalid observed cyclic-deck result {result_path}")
            fingerprint = detail["scientific_fingerprint"]
            if fingerprint in fingerprints:
                raise ValueError(f"duplicate cyclic-deck fingerprint {result_path}")
            fingerprints.add(fingerprint)
            output.append(detail)
    return output


def _detail_summary(detail: Mapping[str, Any]) -> dict[str, float]:
    correct_losses = []
    control_own_losses = []
    control_source_losses = []
    full_js = []
    full_state = []
    front_distances = []
    replay_errors = []
    for regime in REGIMES:
        regime_result = detail["regimes"][regime]
        replay_errors.append(float(regime_result["maximum_replay_error"]))
        front_distances.append(float(regime_result["front_cut_distance"]))
        full = regime_result["cuts"]["full"]
        correct_losses.append(
            float(full["baseline_correct"]["exact_bin_accuracy"])
            - float(full["correct_twirl"]["exact_bin_accuracy"])
        )
        control_own_losses.append(
            float(full["baseline_control"]["exact_bin_accuracy"])
            - float(full["control_twirl_own_target"]["exact_bin_accuracy"])
        )
        control_source_losses.append(
            float(full["baseline_correct"]["exact_bin_accuracy"])
            - float(full["control_twirl_source_target"]["exact_bin_accuracy"])
        )
        full_js.append(
            float(full["correct_action_posterior_dispersion"]["mean_js_from_anchor"])
        )
        full_state.append(
            float(full["correct_action_state_geometry"]["action_difference_relative_rms"])
        )
    gates = detail["seed_gates"]
    full_seed_gate = bool(
        gates["mature_observed_twirl"]
        and gates["computational_cover"]
        and gates["front_replication"]
        and gates["control_validity"]
        and not gates["semantic_control_preserved"]
    )
    return {
        "full_preregistered_seed_gate": float(full_seed_gate),
        "mature_observed_twirl": float(gates["mature_observed_twirl"]),
        "computational_cover": float(gates["computational_cover"]),
        "front_replication": float(gates["front_replication"]),
        "shifted_target_control_valid": float(gates["control_validity"]),
        "semantic_control_preserved": float(gates["semantic_control_preserved"]),
        "maximum_front_cut_distance": max(front_distances),
        "maximum_correct_twirl_accuracy_loss": max(correct_losses),
        "minimum_correct_twirl_accuracy_gain": min(-value for value in correct_losses),
        "maximum_correct_twirl_accuracy_gain": max(-value for value in correct_losses),
        "maximum_control_own_target_accuracy_loss": max(control_own_losses),
        "minimum_control_source_target_accuracy_loss": min(control_source_losses),
        "maximum_full_action_posterior_js": max(full_js),
        "maximum_full_action_state_relative_rms": max(full_state),
        "maximum_replay_error": max(replay_errors),
        "analysis_seconds": float(detail["analysis_seconds"]),
    }


def build_observed_cyclic_deck_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-10_tinyllm-observed-cyclic-deck-twirl.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    summaries = [
        {
            "experiment_id": detail["experiment_id"],
            "k": int(detail["k"]),
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
            "name": "TinyLLM observed cyclic-deck causal-front replication",
            "category": "mechanistic_interpretability",
            "description": (
                "One-observation calibrated C2/C3 actions test whether mature "
                "Reynolds sufficiency and the earlier generator-defined front transfer."
            ),
            "question": (
                "Can a cyclic deck orbit constructed from one observation reproduce "
                "both mature quotient sufficiency and the prior causal-front cut?"
            ),
            "prediction": (
                "Every population gate, including front agreement within one cut, "
                "passes in at least four of five checkpoints for both C2 and C3."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "preregistered_full_front_not_confirmed_c2_4of5_c3_2of5_"
                "mature_both_5of5"
            ),
            "evidence_count": 10,
            "direct_experiment_count": 10,
            "fixed_checkpoint_count": 10,
            "stored_checkpoint_summary_count": 10,
            "evidence_role": EVIDENCE_ROLE,
            "power_profile": "two_degree_five_checkpoint_two_shift_causal_replication",
            "tested_scope": (
                "ten retained d6 analytic-carrier C2/C3 systems, one-observation "
                "continuous cyclic actions, two shifts, nine activation cuts"
            ),
            "subclaims": {
                "mature_observed_C2_twirl": "supported_five_of_five",
                "mature_observed_C3_twirl": "supported_five_of_five",
                "early_computational_cover_C2": "supported_five_of_five",
                "early_computational_cover_C3": "supported_five_of_five",
                "C2_front_replication_within_one_cut": "supported_four_of_five",
                "C3_front_replication_within_one_cut": "not_supported_two_of_five",
                "half_turn_semantic_specificity": "supported_zero_of_ten_false_positives",
                "generator_orbit_membership_required_for_mature_twirl": "no_in_tested_C2_C3_scope",
                "front_location_discretization_invariant": "no",
                "same_scope_retraining_licensed": "no",
            },
            "tags": [
                "tinyllm",
                "observed-cyclic-action",
                "C2",
                "C3",
                "reynolds-twirl",
                "causal-front",
                "quantization-sensitivity",
                "frozen-checkpoint",
                "preregistered-null",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "C3 front agreement reaches only two of five checkpoints, below "
                "the locked four-seed gate, although mature C2/C3 twirls and all "
                "cover/control gates pass five of five."
            ),
            "confidence": 0.99,
            "confidence_assessment": (
                "strong_preregistered_null_for_universal_front_with_positive_mature_closure"
            ),
            "num_direct_experiments": 10,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "action_contracts": campaign["action_contracts"],
                "checkpoint_summaries": summaries,
            },
            "key_insights": [
                "Mature C2 and C3 Reynolds sufficiency is constructible from one observed calibrated input in every retained checkpoint.",
                "The observed action preserves the exact analytic character but differs materially from separately quantizing every generator sheet.",
                "C2 causal-front locations remain stable in four of five checkpoints; C3 locations do so in only two of five and shift later under extrapolation.",
                "Full-depth quotient sufficiency is more robust than the first-preserved activation cut.",
                "A matched half-task-turn orbit works for its shifted target and fails for the source target in every checkpoint.",
            ],
            "suggested_hypotheses": [
                "Quantization-aware and continuous actions share mature quotient closure but induce different early fronts for higher-order groups.",
                "Architecture-population variation will dominate exact causal-front location while preserving mature cyclic quotient sufficiency.",
                "Unknown group frames will move the next bottleneck from quotient use to action estimation.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"checkpoint_summaries": summaries},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "source_ladder_campaign_sha256": SOURCE_LADDER_SHA256,
            "source_deck_campaign_sha256": SOURCE_DECK_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
        },
    }


def build_observed_cyclic_deck_experiment_results(
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
                metrics=summary,
                primary_metric=float(summary["full_preregistered_seed_gate"]),
                model_architecture=[6, 384],
                model_parameters=MODEL_PARAMETERS,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["checkpoint"],
                observations=[
                    f"Degree: C{detail['k']}.",
                    "The cyclic orbit was constructed from one decoded calibrated observation.",
                    "No latent, target, branch, fiber ID, fitted action, or new training entered the intervention.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=(
                    []
                    if detail["seed_gates"]["front_replication"]
                    else ["Observed first-preserved cut differs from the locked generator reference by more than one cut."]
                ),
                timestamp=completed,
            )
        )
    return output


def store_observed_cyclic_deck_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_observed_cyclic_deck_meta_hypothesis(results_path)
    experiments = build_observed_cyclic_deck_experiment_results(record, results_path)
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
            raise RuntimeError("observed cyclic-deck ChromaDB read-back failed")
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
    "build_observed_cyclic_deck_experiment_results",
    "build_observed_cyclic_deck_meta_hypothesis",
    "store_observed_cyclic_deck_meta_hypothesis",
]
