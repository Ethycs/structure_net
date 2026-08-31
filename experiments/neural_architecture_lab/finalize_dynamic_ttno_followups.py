#!/usr/bin/env python3
"""Finalize the A5--A7 NAL registry from hash-verified campaign results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger


REGISTRY = Path(
    "data/hypothesis_registry/tinyllm-dynamic-ttno-followups-v1.json"
)
CHROMADB = Path("data/chroma_db")
CAMPAIGNS = {
    "tinyllm-ttno-cut-localization-parity-v1": {
        "path": Path(
            "data/experiments/tinyllm_ttno_cut_localization_parity/"
            "20260830_registered/campaign_results.json"
        ),
        "sha256": "d9585b4d4d833e632052d2277ecdd11eb2949899aa1073e00d0f006c234f04ef",
        "classification": "intrinsic_operator_rank_growth",
    },
    "tinyllm-hss-shared-basis-nesting-v1": {
        "path": Path(
            "data/experiments/tinyllm_hss_shared_basis_nesting/"
            "20260830_registered/campaign_results.json"
        ),
        "sha256": "6f42a59b3a723eb4b80742e8fab8278be9b21d1db35131cc4ea81bd702e03c01",
        "classification": "shared_and_nested_hierarchy_supported",
    },
    "tinyllm-causal-h2-attention-v1": {
        "path": Path(
            "data/experiments/tinyllm_causal_h2_attention/"
            "20260830_registered/campaign_results.json"
        ),
        "sha256": "4885696dd746a52cb015b51d34733901c2acd50baccfc59f2f76cfe176eeb9b2",
        "classification": "h2_representation_pass_no_finite_size_compression",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    registry: dict[str, Any] = json.loads(REGISTRY.read_text(encoding="utf-8"))
    items = {item["id"]: item for item in registry["hypotheses"]}
    if set(items) != set(CAMPAIGNS):
        raise RuntimeError("registry IDs do not match A5--A7")

    completed_at = []
    for hypothesis_id, expected in CAMPAIGNS.items():
        path = expected["path"]
        if _sha256(path) != expected["sha256"]:
            raise RuntimeError(f"campaign hash mismatch for {hypothesis_id}")
        campaign = json.loads(path.read_text(encoding="utf-8"))
        classification = campaign["aggregates"]["classification"]
        if (
            campaign.get("status") != "completed"
            or campaign.get("hypothesis_id") != hypothesis_id
            or classification != expected["classification"]
            or campaign["summary"]["completed"] != 5
            or campaign["summary"]["failed"] != 0
        ):
            raise RuntimeError(f"campaign contract mismatch for {hypothesis_id}")
        completed_at.append(campaign["completed_at"])
        item = items[hypothesis_id]
        item.update(
            {
                "status": "completed",
                "tested": True,
                "result_count": campaign["summary"]["completed"],
                "classification": classification,
                "campaign": {
                    "path": str(path),
                    "sha256": expected["sha256"],
                    "completed_at": campaign["completed_at"],
                    "evaluation_seeds_completed": campaign["summary"]["completed"],
                    "failed": campaign["summary"]["failed"],
                },
            }
        )

    logger = StandardizedLogger(
        LoggingConfig(
            project_name="structure_net_completed_hypotheses",
            queue_dir="data/experiment_queue",
            sent_dir="data/experiment_sent",
            rejected_dir="data/experiment_rejected",
            enable_wandb=False,
            auto_upload=False,
            enable_chromadb=True,
            chromadb_path=str(CHROMADB),
        )
    )
    for hypothesis_id, item in items.items():
        logger.log_hypothesis(
            {
                "id": hypothesis_id,
                "name": item["name"],
                "description": item["description"],
                "category": item["category"],
                "question": item["question"],
                "prediction": item["prediction"],
                "created_at": item["created_at"],
                "tested": True,
                "confirmation_status": item["classification"],
                "tested_scope": "one_checkpoint_five_input_prefixes",
                "evidence_count": item["result_count"],
                "direct_experiment_count": item["result_count"],
            }
        )
    expected_ids = list(CAMPAIGNS)
    readback = logger.hypotheses_collection.get(
        ids=expected_ids, include=["metadatas"]
    )
    metadata = {item["id"]: item for item in readback["metadatas"]}
    if set(readback["ids"]) != set(expected_ids) or any(
        metadata[hypothesis_id]["tested"] is not True
        or metadata[hypothesis_id]["confirmation_status"]
        != CAMPAIGNS[hypothesis_id]["classification"]
        for hypothesis_id in expected_ids
    ):
        raise RuntimeError("completed hypothesis ChromaDB read-back failed")

    registry["status"] = "completed"
    registry["completed_at"] = max(completed_at)
    registry["storage"]["readback"] = {
        "verified": True,
        "hypothesis_ids": expected_ids,
        "tested": True,
        "classifications": {
            hypothesis_id: CAMPAIGNS[hypothesis_id]["classification"]
            for hypothesis_id in expected_ids
        },
    }
    temporary = REGISTRY.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(registry, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(REGISTRY)
    print(
        json.dumps(
            {
                "status": registry["status"],
                "hypotheses": {
                    hypothesis_id: items[hypothesis_id]["classification"]
                    for hypothesis_id in expected_ids
                },
                "readback": registry["storage"]["readback"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
