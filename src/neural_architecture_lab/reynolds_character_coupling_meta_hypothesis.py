"""Build the TinyLLM Reynolds character-coupling evidence record."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-reynolds-character-coupling.v1"
HYPOTHESIS_ID = "tinyllm-reynolds-character-coupling-synthesis-v1"


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or int(value.get("summary", {}).get("completed", -1)) != 10
    ):
        raise ValueError(f"invalid character-coupling campaign {path}")
    return value


def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    output = []
    for k in (2, 3):
        for seed in (7, 17, 29, 41, 53):
            detail_path = path.parent / "runs" / f"k{k}" / f"seed_{seed}" / "result.json"
            detail = json.loads(detail_path.read_text())
            if (
                detail.get("schema_version") != EXPERIMENT_SCHEMA
                or detail.get("status") != "completed"
                or detail.get("implementation_sha256") != campaign["implementation_sha256"]
                or int(detail.get("k", -1)) != k
                or int(detail.get("seed", -1)) != seed
            ):
                raise ValueError(f"invalid character-coupling cell {detail_path}")
            output.append(detail)
    return output


def build_reynolds_character_coupling_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path("docs/08 - Analysis/2026-08-06_tinyllm-reynolds-character-coupling.md"),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    evidence = [
        {
            "experiment_id": detail["experiment_id"],
            "k": int(detail["k"]),
            "seed": int(detail["seed"]),
            "checkpoint": detail["provenance"]["checkpoint"],
            "checkpoint_sha256": detail["provenance"]["checkpoint_sha256"],
            "gates": detail["gates"],
            "synthesis_fronts": {
                regime: detail["regimes"][regime]["synthesis_target_cut"]
                for regime in ("composition", "extrapolation")
            },
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM Reynolds character-coupling synthesis",
            "category": "representation",
            "description": "Causal propagated-versus-actual barycenter patches and symmetry-neutral Taylor decomposition at every attention/MLP residual sublayer.",
            "question": "Does the causal quotient front mark invariant synthesis from populated deck-character modes, with most task effect explained quadratically?",
            "prediction": "All causal localization, shift stability, neutral quadratic, and control gates pass in four of five seeds for both degrees.",
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "partially_supported_exact_k2_synthesis_without_universal_quadratic_sufficiency",
            "evidence_count": 10,
            "direct_experiment_count": 10,
            "tested_scope": "retained d6 degree-2/3 ladder; five seeds; 64 exact orbits; twelve residual sublayers; both shifts",
            "subclaims": {
                "k2_shift_stable_regime": "supported_five_of_five",
                "k2_exact_synthesis_localization": "supported_five_of_five_zero_distance",
                "k2_neutral_quadratic_sufficiency": "not_supported_three_of_five",
                "k3_shift_stable_regime": "not_supported_two_of_five",
                "k3_synthesis_localization": "not_supported_three_of_five",
                "k3_neutral_quadratic_sufficiency": "not_supported_zero_of_five",
                "control_specificity": "supported_five_of_five_both_degrees",
                "cubic_rescue": "not_supported",
                "full_hypothesis": "not_confirmed",
            },
            "tags": ["tinyllm", "reynolds-defect", "character-theory", "deck-group", "causal-intervention", "invariant-synthesis"],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": "Exact Reynolds-defect synthesis localizes every degree-two causal front, but quadratic sufficiency and degree-three shift stability fail the preregistered thresholds.",
            "confidence": 0.0,
            "confidence_assessment": "complete_five_seed_partial_support_with_primary_gate_failure",
            "independent_seed_count": 5,
            "num_direct_experiments": 10,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "Every degree-two synthesis front exactly matches the frozen causal quotient front under both shifts.",
                "The complete degree-two causal regime sequence is shift-stable in all five seeds.",
                "Neutral quadratic task effect is strong only in the three early block-zero-attention fronts.",
                "Degree-three carriers contain both conjugate characters, but their finite Reynolds defects are not captured by local quadratic or cubic truncations.",
                "Shuffled or norm-matched generic directions do not reproduce exact character-dependent synthesis.",
            ],
            "suggested_hypotheses": [
                "Causal ablation of populated irreducible modes and their neutral fusion channels identifies the exact degree-two synthesis pathway.",
                "Later synthesis fronts require nonlocal or higher-order character coupling rather than a local Taylor truncation.",
                "Degree-three quotient sufficiency is support-relative because invariant synthesis can be destroyed and re-created across depth.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path), str(report_path)],
    }


def build_reynolds_character_coupling_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    completed = datetime.fromisoformat(_campaign(results_path)["completed_at"].replace("Z", "+00:00"))
    return [
        ExperimentResult(
            experiment_id=detail["experiment_id"],
            hypothesis_id=HYPOTHESIS_ID,
            metrics={**{gate: float(value) for gate, value in detail["gates"].items()}},
            primary_metric=float(detail["all_gates_passed"]),
            model_architecture=[6, int(detail["k"])],
            model_parameters=29_956_224,
            training_time=float(detail["analysis_seconds"]),
            model_checkpoint=detail["provenance"]["checkpoint"],
            observations=["Frozen checkpoint; exact deck orbits; all twelve residual sublayers."],
            anomalies=["Residual Taylor accuracy and downstream task-effect accuracy differ sharply."],
            timestamp=completed,
        )
        for detail in _details(results_path)
    ]


def store_reynolds_character_coupling_meta_hypothesis(
    results_path: Path, output_path: Path, *, chromadb_path: Path | None = None
) -> dict[str, Any]:
    record = build_reynolds_character_coupling_meta_hypothesis(results_path)
    experiments = build_reynolds_character_coupling_experiment_results(record, results_path)
    storage: dict[str, Any] = {
        "aggregate_path": str(output_path),
        "experiment_ids": [item.experiment_id for item in experiments],
    }
    if chromadb_path is not None:
        from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger

        logger = StandardizedLogger(
            LoggingConfig(
                project_name="structure_net_meta_hypotheses",
                queue_dir="data/experiment_queue",
                sent_dir="data/experiment_sent",
                rejected_dir="data/experiment_rejected",
                enable_wandb=False,
                auto_upload=False,
                enable_chromadb=True,
                chromadb_path=str(chromadb_path),
            )
        )
        logger.log_hypothesis(record["hypothesis"])
        storage["result_hashes"] = [logger.log_experiment_result(item) for item in experiments]
        storage["chromadb_path"] = str(chromadb_path)
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return record


__all__ = [
    "build_reynolds_character_coupling_meta_hypothesis",
    "build_reynolds_character_coupling_experiment_results",
    "store_reynolds_character_coupling_meta_hypothesis",
]
