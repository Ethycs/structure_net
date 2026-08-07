"""Build and store the TinyLLM carrier-Jacobian axis-audit record."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-carrier-jacobian-axis-audit.v1"
HYPOTHESIS_ID = "tinyllm-c2-carrier-jacobian-axis-audit-v1"
EVIDENCE_ROLE = "preregistered_post_outcome_underpowered_mechanistic_diagnostic"
IMPLEMENTATION_SHA256 = (
    "3e1b25510ecf2bbac9ab0424e283f38ac8ec660c2036e2ca77dd0c0671d1cff7"
)
CAMPAIGN_SHA256 = (
    "d953889c7103bdbececf16b9b2f450d14122509bc8305fb5235c86ad0446a6ee"
)
PREDECESSOR_SHA256 = (
    "44707fd4bcd810e63614671aa491095fae735ee52359d464ab25abb10a2bc228"
)
DIRECTED_PAIRS = ((7, 29), (7, 53), (29, 7), (29, 53), (53, 7), (53, 29))
EXPECTED_CLASSIFICATIONS = {
    (7, 29): "mixed",
    (7, 53): "axes_1_2_dominant",
    (29, 7): "mixed",
    (29, 53): "mixed",
    (53, 7): "mixed",
    (53, 29): "axes_1_2_dominant",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    summary = value.get("summary", {})
    aggregates = value.get("aggregates", {})
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or value.get("provenance", {}).get("source_campaign_sha256")
        != PREDECESSOR_SHA256
        or int(summary.get("requested", -1)) != 6
        or int(summary.get("completed", -1)) != 6
        or int(summary.get("failed", -1)) != 0
        or int(summary.get("excluded", -1)) != 0
        or int(summary.get("trained_models", -1)) != 0
        or int(summary.get("fitted_predictive_observers", -1)) != 0
        or int(summary.get("fitted_coordinate_maps", -1)) != 0
        or int(summary.get("reproduced_predecessor_maps", -1)) != 6
        or int(summary.get("finite_difference_continuations", -1)) != 288
        or aggregates.get("adequate_pair_count") != 6
        or aggregates.get("axis_3_dominant_count") != 0
        or aggregates.get("axes_1_2_dominant_count") != 2
        or aggregates.get("mixed_count") != 4
        or aggregates.get("nonlinear_or_unresolved_count") != 0
        or aggregates.get("universal_2d_base_plus_local_scalar_supported")
        is not False
        or aggregates.get("shared_axes_causally_misoriented_supported") is not False
        or aggregates.get("conclusion")
        != "mixed_checkpoint_stratified_carrier_atlas"
    ):
        raise ValueError(f"invalid carrier-Jacobian axis-audit campaign {path}")
    return value


def _verify_hashed_artifact(provenance: Mapping[str, Any], field: str) -> None:
    artifact = Path(provenance.get(field, ""))
    if not artifact.is_file() or _sha256(artifact) != provenance.get(
        f"{field}_sha256"
    ):
        raise ValueError(f"invalid provenance artifact {field}")


def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {
        (int(item["source_seed"]), int(item["target_seed"])): item
        for item in campaign.get("results", [])
    }
    if set(entries) != set(DIRECTED_PAIRS):
        raise ValueError(f"invalid carrier-Jacobian directed-pair index {path}")
    details: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for pair in DIRECTED_PAIRS:
        entry = entries[pair]
        detail_path = Path(entry["path"])
        detail = json.loads(detail_path.read_text())
        local = detail.get("local_linearization", {})
        attribution = detail.get("axis_attribution", {})
        frame_errors = detail.get("canonical_frame", {}).get(
            "predecessor_reproduction_max_abs", {}
        )
        if (
            _sha256(detail_path) != entry.get("result_sha256")
            or detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or (int(detail.get("source_seed", -1)), int(detail.get("target_seed", -1)))
            != pair
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or attribution.get("classification") != EXPECTED_CLASSIFICATIONS[pair]
            or entry.get("classification") != EXPECTED_CLASSIFICATIONS[pair]
            or local.get("adequate") is not True
            or not all(local.get("gates", {}).values())
            or len(local.get("gates", {})) != 5
            or float(local.get("signed_error_zero_referenced_r2", -1.0)) < 0.983
            or float(local.get("sign_agreement", -1.0)) != 1.0
            or max(float(value) for value in frame_errors.values()) > 1e-10
            or len(detail.get("heldout_cells", [])) != 4
            or max(
                float(cell["axis_reconstruction_relative_error"])
                for cell in detail.get("heldout_cells", [])
            )
            > 1e-8
            or detail.get("provenance", {}).get("source_campaign_sha256")
            != PREDECESSOR_SHA256
        ):
            raise ValueError(f"invalid carrier-Jacobian result {detail_path}")
        provenance = detail["provenance"]
        _verify_hashed_artifact(provenance, "source_campaign")
        _verify_hashed_artifact(provenance, "source_pair_result")
        for role in ("source", "target"):
            role_provenance = provenance[role]
            for field in (
                "checkpoint",
                "frontend_checkpoint",
                "source_result",
                "character_result",
            ):
                _verify_hashed_artifact(role_provenance, field)
        fingerprint = detail["scientific_fingerprint"]
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate scientific fingerprint {detail_path}")
        fingerprints.add(fingerprint)
        details.append(detail)
    return details


def build_carrier_jacobian_axis_audit_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-carrier-jacobian-axis-audit.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    direct_tests = []
    for detail in details:
        local = detail["local_linearization"]
        attribution = detail["axis_attribution"]
        direct_tests.append(
            {
                "experiment_id": detail["experiment_id"],
                "evidence_role": EVIDENCE_ROLE,
                "source_seed": int(detail["source_seed"]),
                "target_seed": int(detail["target_seed"]),
                "scientific_fingerprint": detail["scientific_fingerprint"],
                "local_linearization_adequate": local["adequate"],
                "signed_error_zero_referenced_r2": local[
                    "signed_error_zero_referenced_r2"
                ],
                "prediction_residual_mae_fraction": local[
                    "prediction_residual_mae_fraction"
                ],
                "sign_agreement": local["sign_agreement"],
                "classification": attribution["classification"],
                "shared_predicted_rms_fraction": attribution[
                    "shared_predicted_rms_fraction"
                ],
                "third_predicted_rms_fraction": attribution[
                    "third_predicted_rms_fraction"
                ],
                "shared_only_error_retention": attribution[
                    "shared_only_error_retention"
                ],
                "third_only_error_retention": attribution[
                    "third_only_error_retention"
                ],
            }
        )
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM carrier-Jacobian axis audit",
            "category": "representation",
            "description": (
                "Frozen target-continuation Jacobians and canonical-axis retain-only "
                "patches localize failed cross-seed C2 carrier transport."
            ),
            "question": (
                "Is cross-seed causal error a universal weak-third-axis correction, "
                "or are shared canonical axes also causally misoriented?"
            ),
            "prediction": (
                "All six local models are adequate and axis 3 is dominant in at "
                "least five of six directed pairs."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_mixed_checkpoint_stratified_causal_atlas"
            ),
            "evidence_count": 6,
            "direct_experiment_count": 6,
            "power_profile": "underpowered_preregistered_post_outcome_diagnostic",
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "six directed maps among three selected stable d6 C2 block-0 "
                "fronts; 24 reused held-out cells under composition and extrapolation"
            ),
            "subclaims": {
                "local_task_linearization": "supported_six_of_six",
                "universal_weak_third_axis_correction": "rejected_zero_of_six_dominant",
                "universal_shared_axis_misorientation": "not_supported_two_of_six_dominant",
                "checkpoint_stratified_causal_atlas": "supported_mixed_four_of_six",
                "group_anchored_task_metric_is_appropriate_next_test": "supported",
                "portable_two_channel_sidecar": "not_yet_supported",
            },
            "tags": [
                "tinyllm",
                "cross-seed",
                "causal-patching",
                "jacobian",
                "canonical-correlation",
                "c2",
                "negative-result",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "A group-anchored Jacobian- or Fisher-weighted transport map.",
                "A trained typed sidecar or equivariant encoder.",
                "Population prevalence beyond the selected three checkpoints.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "All six local Jacobians are adequate, but axis 3 dominates zero "
                "directions; two pairs are shared-axis dominant and four are mixed."
            ),
            "confidence": 0.9,
            "confidence_assessment": (
                "exact_preregistered_post_outcome_underpowered_mechanistic_rejection"
            ),
            "independent_seed_count": 3,
            "num_direct_experiments": 6,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "Centered derivatives at 0.025 and 0.05 target standard deviations agree in all six directions.",
                "Local Jacobians explain 0.9835--0.9941 of zero-referenced signed transport error with perfect sign agreement.",
                "The weak third canonical axis is not dominant in any directed pair.",
                "Shared-axis mismatch retains 0.710--0.892 of paired error and is substantial in every direction.",
                "The causal atlas is checkpoint-stratified even though its statistical carrier geometry is shared.",
            ],
            "suggested_hypotheses": [
                "A group-anchored target-Jacobian metric improves causal transport over Euclidean and affine-ridge maps.",
                "Portable post-hoc carrier coordinates require a task-sensitive metric, not canonical correlation alone.",
                "If task-metric transport fails, checkpoint-local causal charts should replace a shared sidecar interface.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": direct_tests},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "predecessor_campaign_sha256": PREDECESSOR_SHA256,
        },
    }


def build_carrier_jacobian_axis_audit_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output: list[ExperimentResult] = []
    for detail in _details(results_path):
        local = detail["local_linearization"]
        attribution = detail["axis_attribution"]
        classification = attribution["classification"]
        metrics = {
            "local_linearization_adequate": float(local["adequate"]),
            "signed_error_zero_referenced_r2": float(
                local["signed_error_zero_referenced_r2"]
            ),
            "prediction_residual_mae_fraction": float(
                local["prediction_residual_mae_fraction"]
            ),
            "sign_agreement": float(local["sign_agreement"]),
            "axis_3_dominant": float(classification == "axis_3_dominant"),
            "axes_1_2_dominant": float(classification == "axes_1_2_dominant"),
            "mixed": float(classification == "mixed"),
            "shared_predicted_rms_fraction": float(
                attribution["shared_predicted_rms_fraction"]
            ),
            "third_predicted_rms_fraction": float(
                attribution["third_predicted_rms_fraction"]
            ),
            "shared_only_error_retention": float(
                attribution["shared_only_error_retention"]
            ),
            "third_only_error_retention": float(
                attribution["third_only_error_retention"]
            ),
        }
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(classification == "axis_3_dominant"),
                model_architecture=[6, 6, 384],
                model_parameters=29_956_608,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["target"]["checkpoint"],
                observations=[
                    "Frozen target continuation; no model, observer, or map fitted.",
                    "The local task Jacobian is adequate for this directed pair.",
                    f"Preregistered axis classification: {classification}.",
                ],
                anomalies=(
                    []
                    if classification == "axis_3_dominant"
                    else ["The predicted universal weak-third-axis correction is absent."]
                ),
                timestamp=completed,
            )
        )
    return output


def store_carrier_jacobian_axis_audit_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_carrier_jacobian_axis_audit_meta_hypothesis(results_path)
    experiments = build_carrier_jacobian_axis_audit_experiment_results(
        record, results_path
    )
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
        storage["result_hashes"] = [
            logger.log_experiment_result(item) for item in experiments
        ]
        storage["chromadb_path"] = str(chromadb_path)
        hypothesis_readback = logger.hypotheses_collection.get(
            ids=[HYPOTHESIS_ID], include=["metadatas"]
        )
        experiment_readback = logger.experiments_collection.get(
            ids=storage["result_hashes"], include=["metadatas"]
        )
        if (
            hypothesis_readback.get("ids") != [HYPOTHESIS_ID]
            or len(experiment_readback.get("ids", [])) != len(experiments)
            or {
                item.get("hypothesis_id")
                for item in experiment_readback.get("metadatas", [])
            }
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("carrier-Jacobian axis-audit ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": len(experiment_readback["ids"]),
        }
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return record


__all__ = [
    "build_carrier_jacobian_axis_audit_meta_hypothesis",
    "build_carrier_jacobian_axis_audit_experiment_results",
    "store_carrier_jacobian_axis_audit_meta_hypothesis",
]
