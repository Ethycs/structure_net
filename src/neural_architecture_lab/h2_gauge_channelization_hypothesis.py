"""NAL registration for the preregistered A8 H2 gauge study."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import Hypothesis, HypothesisCategory


REGISTRY_SCHEMA = "nal.pending-hypothesis-registry.v1"
REGISTERED_AT = datetime(2026, 8, 30, tzinfo=timezone.utc)
HYPOTHESIS_ID = "tinyllm-h2-gauge-channelization-v1"
PARENT_HYPOTHESIS_ID = "tinyllm-causal-h2-attention-v1"
PREREGISTRATION = (
    "docs/07 - Status Reports/"
    "2026-08-30_tinyllm-h2-gauge-channelization-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "59451a7cc0e61de73335f5c455954b355ebb9611c5c5cb4133ec1d27acd4a53b"
)
A7_CAMPAIGN = (
    "data/experiments/tinyllm_causal_h2_attention/"
    "20260830_registered/campaign_results.json"
)
A7_CAMPAIGN_SHA256 = (
    "4885696dd746a52cb015b51d34733901c2acd50baccfc59f2f76cfe176eeb9b2"
)
A7_IMPLEMENTATION_SHA256 = (
    "9e61430463ecc11ff5d99560625a9f85a6e5b189d39a43fdbcf26bb06e804dbc"
)
A6_CAMPAIGN_SHA256 = (
    "6f42a59b3a723eb4b80742e8fab8278be9b21d1db35131cc4ea81bd702e03c01"
)
CHECKPOINT_SHA256 = (
    "5a7491b4231c30feaaabda7babf0a831dd4d72fbb4e7f3e757114cadf098df09"
)
TOKEN_STREAM_SHA256 = (
    "f339655453b970ae6cc1cbdc7c78f8a5234c42437bc7bba8fb31a1dee5c9d765"
)
EVALUATION_SEEDS = [101, 211, 307, 401, 503]


def run_a8_pending(_config: dict[str, Any]) -> None:
    """Prevent accidental execution outside the frozen campaign runner."""

    raise RuntimeError(
        "A8 is registered but not finalized; execute only with the frozen "
        "tinyllm_h2_gauge_channelization campaign runner."
    )


def build_hypothesis() -> Hypothesis:
    """Build the frozen A8 pending hypothesis."""

    controls = {
        "menu_id": "A8",
        "planned_schema": "nal.tinyllm-h2-gauge-channelization.v1",
        "status": "registered_not_run",
        "parent_hypothesis_id": PARENT_HYPOTHESIS_ID,
        "parent_aggregate_path": A7_CAMPAIGN,
        "parent_aggregate_sha256": A7_CAMPAIGN_SHA256,
        "parent_implementation_sha256": A7_IMPLEMENTATION_SHA256,
        "a6_aggregate_sha256": A6_CAMPAIGN_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "token_stream_sha256": TOKEN_STREAM_SHA256,
        "preregistration": PREREGISTRATION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "evaluation_seed_semantics": "input-selection replicates, not model seeds",
        "evaluation_seeds": EVALUATION_SEEDS,
        "layers": list(range(8)),
        "heads": [0, 3, 7],
        "lengths": [64, 128, 256],
        "primary_length": 256,
        "block_widths": [1, 2, 4],
        "optimizer": {
            "parameterization": "cayley_skew_symmetric",
            "dtype": "float64",
            "updates": 96,
            "learning_rate": 0.03,
            "restarts": ["identity", "spectral_covariance"],
            "objective": "normalized_offblock_frobenius_energy",
        },
        "compression_gates": {
            "storage_ratio_median_max": 0.75,
            "storage_ratio_p90_max": 1.0,
            "operation_ratio_median_max": 0.75,
            "operation_ratio_p90_max": 1.0,
        },
        "decision_order": [
            "invalid_parent_or_gauge_contract",
            "gauge_channelization_compression_pass",
            "gauge_channelization_representation_only",
            "gauge_channelization_sparsity_accuracy_tradeoff",
            "gauge_channelization_no_structural_gain",
        ],
    }
    return Hypothesis(
        id=HYPOTHESIS_ID,
        name="TinyLLM H2 internal-gauge channelization",
        description=(
            "Jointly rotate fixed A7 nested H2 bases and test deterministic "
            "diagonal or small-block pruning of transfers and couplings."
        ),
        category=HypothesisCategory.SPARSITY,
        question=(
            "Does removable coordinate mixing, rather than rank alone, explain "
            "the finite-size overhead of the accurate A7 H2 representation?"
        ),
        prediction=(
            "At least one optimized fixed-width block arm will preserve the A7 "
            "attention certificate while materially reducing factor accounting."
        ),
        test_function=run_a8_pending,
        parameter_space={"evaluation_seed": EVALUATION_SEEDS},
        control_parameters=controls,
        success_metrics={
            "valid_gauge_contract": 1.0,
            "primary_length_pass_fraction": 0.80,
            "minimum_layer_pass_fraction": 0.50,
            "compression_pass": 1.0,
        },
        created_at=REGISTERED_AT,
        tags=[
            "A8",
            "TinyLLM",
            "H2",
            "gauge",
            "channelization",
            "block-sparsity",
            "pending",
        ],
        references=[A7_CAMPAIGN, PREREGISTRATION],
        tested=False,
    )


def _registry_record(hypothesis: Hypothesis) -> dict[str, Any]:
    return {
        "id": hypothesis.id,
        "menu_id": "A8",
        "name": hypothesis.name,
        "description": hypothesis.description,
        "category": hypothesis.category.value,
        "question": hypothesis.question,
        "prediction": hypothesis.prediction,
        "test_function": (
            f"{hypothesis.test_function.__module__}."
            f"{hypothesis.test_function.__name__}"
        ),
        "parameter_space": hypothesis.parameter_space,
        "control_parameters": hypothesis.control_parameters,
        "success_metrics": hypothesis.success_metrics,
        "statistical_significance": hypothesis.statistical_significance,
        "created_at": hypothesis.created_at.isoformat(),
        "tags": hypothesis.tags,
        "references": hypothesis.references,
        "preregistration": PREREGISTRATION,
        "status": "registered_not_run",
        "tested": False,
        "result_count": 0,
    }


def register_hypothesis(lab: Any, output_path: Path) -> dict[str, Any]:
    """Register A8, verify Chroma readback, and persist the central ledger."""

    hypothesis = build_hypothesis()
    lab.register_hypothesis(hypothesis)
    if hypothesis.id not in lab.hypotheses or hypothesis.id not in lab.pending_hypotheses:
        raise RuntimeError("A8 in-memory pending registration readback failed")

    collection = getattr(lab.logger, "hypotheses_collection", None)
    if collection is None:
        raise RuntimeError("A8 registration requires ChromaDB persistence")
    readback = collection.get(ids=[hypothesis.id], include=["metadatas"])
    metadata = readback.get("metadatas", [])
    if readback.get("ids") != [hypothesis.id] or not metadata:
        raise RuntimeError("A8 ChromaDB registration readback failed")
    if metadata[0].get("tested") is not False:
        raise RuntimeError("A8 was not persisted as pending")

    record = {
        "schema": REGISTRY_SCHEMA,
        "registered_at": REGISTERED_AT.isoformat(),
        "status": "registered_not_run",
        "parent_hypothesis_id": PARENT_HYPOTHESIS_ID,
        "hypotheses": [_registry_record(hypothesis)],
        "storage": {
            "registry_path": str(output_path),
            "chromadb_path": str(lab.results_dir / "chroma_db"),
            "collection": "hypotheses",
            "readback": {
                "verified": True,
                "hypothesis_ids": [hypothesis.id],
                "tested": False,
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return record


__all__ = [
    "HYPOTHESIS_ID",
    "PREREGISTRATION_SHA256",
    "build_hypothesis",
    "register_hypothesis",
]
