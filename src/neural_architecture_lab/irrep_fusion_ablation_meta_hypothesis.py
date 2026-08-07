"""Build the TinyLLM deck-irrep fusion-ablation evidence record."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-deck-irrep-fusion-ablation.v1"
HYPOTHESIS_ID = "tinyllm-deck-irrep-fusion-ablation-v1"


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or int(value.get("summary", {}).get("completed", -1)) != 10
        or int(value.get("summary", {}).get("failed", -1)) != 0
    ):
        raise ValueError(f"invalid irrep-fusion campaign {path}")
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
                raise ValueError(f"invalid irrep-fusion cell {detail_path}")
            output.append(detail)
    return output


def build_irrep_fusion_ablation_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path("docs/08 - Analysis/2026-08-06_tinyllm-irrep-fusion-ablation.md"),
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
            "frozen_character_result_sha256": detail["provenance"]["frozen_character_result_sha256"],
            "gates": detail["gates"],
            "regimes": {
                regime: {
                    "target_cut": detail["regimes"][regime]["target_cut"],
                    "first_passing_alpha": detail["regimes"][regime]["first_passing_alpha"],
                    "phase_sensitivity": detail["regimes"][regime]["phase_intervention"]["median_normalized_continuous_phase_sensitivity"],
                    "phase_phenotype": detail["regimes"][regime]["phase_intervention"]["phenotype"],
                }
                for regime in ("composition", "extrapolation")
            },
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM deck-irrep fusion ablation",
            "category": "representation",
            "description": "Checkpoint-only charged-mode removal, orbit-matched carrier substitution, and continuous C3 carrier-phase intervention at frozen causal synthesis fronts.",
            "question": "Are deck-character carriers causally necessary and orbit-specific, and is C3 synthesis universally controlled by finite-group phase invariants beyond the radial quadratic invariant?",
            "prediction": "Group contract, charged necessity, carrier specificity, finite-C3 phase sensitivity, and shift-stable phase phenotype pass in four of five seeds.",
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "partially_supported_causal_irreps_without_universal_c3_phase",
            "evidence_count": 10,
            "direct_experiment_count": 10,
            "tested_scope": "retained d6 degree-2/3 ladder; five seeds; frozen synthesis transitions; 64 exact same-cohort orbits; both shifts",
            "subclaims": {
                "exact_group_contract": "supported_five_of_five_both_degrees",
                "charged_mode_necessity": "supported_five_of_five_both_degrees",
                "orbit_specific_carrier": "supported_five_of_five_both_degrees_under_joint_reproduction_gate",
                "finite_c3_phase_mechanism": "not_supported_three_of_five",
                "shift_stable_c3_phase_phenotype": "supported_four_of_five",
                "pure_o2_radial_surrogate": "falsified_in_three_of_five_c3_seeds",
                "full_hypothesis": "not_confirmed",
            },
            "tags": [
                "tinyllm", "deck-group", "irreducible-representation", "character-carrier",
                "causal-ablation", "c3-phase", "equivariance", "invariant-fusion",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": "Charged carriers are necessary and the matched control does not reproduce the joint effect in every seed, but finite-C3 phase sensitivity reaches only three of five seeds.",
            "confidence": 0.0,
            "confidence_assessment": "complete_five_seed_partial_support_with_primary_c3_gate_failure",
            "independent_seed_count": 5,
            "num_direct_experiments": 10,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "Removing nontrivial deck characters destroys quotient sufficiency at every frozen C2 and C3 synthesis front.",
                "No norm-matched cross-orbit carrier meets the joint causal and 70-percent Fisher reproduction criterion.",
                "Three C3 checkpoints use information changed by continuous carrier phase but preserved by exact C3 rotations.",
                "Two C3 checkpoints are mixed, so no universal finite-group phase normal form is supported.",
                "Exact Ck-equivariant architectures should retain charged channels and permit both radial and finite-phase neutral fusions.",
            ],
            "suggested_hypotheses": [
                "An exact Ck-equivariant TinyLLM with typed charged channels outperforms an O2-radial surrogate under the calibrated extrapolation task.",
                "The C3 phase-sensitive checkpoints expose dominant third- or sixth-character harmonics in a preregistered phase-response Fourier decomposition.",
                "Checkpoint-to-checkpoint phase phenotype is predicted by synthesis cut and charged-carrier radius only after conditioning on shift support.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path), str(report_path)],
    }


def build_irrep_fusion_ablation_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    completed = datetime.fromisoformat(_campaign(results_path)["completed_at"].replace("Z", "+00:00"))
    return [
        ExperimentResult(
            experiment_id=detail["experiment_id"],
            hypothesis_id=HYPOTHESIS_ID,
            metrics={gate: float(value) for gate, value in detail["gates"].items()},
            primary_metric=float(detail["all_gates_passed"]),
            model_architecture=[6, int(detail["k"])],
            model_parameters=29_956_224,
            training_time=float(detail["analysis_seconds"]),
            model_checkpoint=detail["provenance"]["checkpoint"],
            observations=["Frozen checkpoint and synthesis front; exact deck-irrep causal interventions; no fitted observer."],
            anomalies=["Finite-C3 phase sensitivity is strong in three seeds and mixed or shift-dependent in two."],
            timestamp=completed,
        )
        for detail in _details(results_path)
    ]


def store_irrep_fusion_ablation_meta_hypothesis(
    results_path: Path, output_path: Path, *, chromadb_path: Path | None = None
) -> dict[str, Any]:
    record = build_irrep_fusion_ablation_meta_hypothesis(results_path)
    experiments = build_irrep_fusion_ablation_experiment_results(record, results_path)
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
    "build_irrep_fusion_ablation_meta_hypothesis",
    "build_irrep_fusion_ablation_experiment_results",
    "store_irrep_fusion_ablation_meta_hypothesis",
]
