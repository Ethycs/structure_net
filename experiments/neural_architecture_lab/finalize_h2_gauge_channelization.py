#!/usr/bin/env python3
"""Finalize the A8 NAL registry from the hash-verified campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger


HYPOTHESIS_ID = "tinyllm-h2-gauge-channelization-v1"
CLASSIFICATION = "gauge_channelization_sparsity_accuracy_tradeoff"
REGISTRY = Path(
    "data/hypothesis_registry/tinyllm-h2-gauge-channelization-v1.json"
)
CAMPAIGN = Path(
    "data/experiments/tinyllm_h2_gauge_channelization/"
    "20260830_registered/campaign_results.json"
)
CAMPAIGN_SHA256 = (
    "cbd5f01d218c10ae22e8193a06ae34f058b011d3f41dffdb2cae1fd3d78dfa32"
)
CHROMADB = Path("data/chroma_db")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if _sha256(CAMPAIGN) != CAMPAIGN_SHA256:
        raise RuntimeError("A8 campaign hash mismatch")
    campaign: dict[str, Any] = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
    if (
        campaign.get("status") != "completed"
        or campaign.get("hypothesis_id") != HYPOTHESIS_ID
        or campaign.get("aggregates", {}).get("classification") != CLASSIFICATION
        or campaign.get("summary", {}).get("completed") != 5
        or campaign.get("summary", {}).get("failed") != 0
        or campaign.get("aggregates", {}).get("validity", {}).get("pass") is not True
    ):
        raise RuntimeError("A8 campaign contract mismatch")

    registry: dict[str, Any] = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if len(registry.get("hypotheses", [])) != 1:
        raise RuntimeError("A8 registry must contain exactly one hypothesis")
    item = registry["hypotheses"][0]
    if item.get("id") != HYPOTHESIS_ID:
        raise RuntimeError("A8 registry ID mismatch")
    item.update(
        {
            "status": "completed",
            "tested": True,
            "result_count": 5,
            "classification": CLASSIFICATION,
            "campaign": {
                "path": str(CAMPAIGN),
                "sha256": CAMPAIGN_SHA256,
                "completed_at": campaign["completed_at"],
                "evaluation_seeds_completed": 5,
                "failed": 0,
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
    logger.log_hypothesis(
        {
            "id": HYPOTHESIS_ID,
            "name": item["name"],
            "description": item["description"],
            "category": item["category"],
            "question": item["question"],
            "prediction": item["prediction"],
            "created_at": item["created_at"],
            "tested": True,
            "confirmation_status": CLASSIFICATION,
            "tested_scope": "one_checkpoint_five_input_prefixes",
            "evidence_count": 5,
            "direct_experiment_count": 5,
        }
    )
    readback = logger.hypotheses_collection.get(
        ids=[HYPOTHESIS_ID], include=["metadatas"]
    )
    if readback.get("ids") != [HYPOTHESIS_ID] or not readback.get("metadatas"):
        raise RuntimeError("A8 completed ChromaDB readback failed")
    metadata = readback["metadatas"][0]
    if (
        metadata.get("tested") is not True
        or metadata.get("confirmation_status") != CLASSIFICATION
    ):
        raise RuntimeError("A8 completed ChromaDB metadata mismatch")

    registry["status"] = "completed"
    registry["completed_at"] = campaign["completed_at"]
    registry["storage"]["readback"] = {
        "verified": True,
        "hypothesis_ids": [HYPOTHESIS_ID],
        "tested": True,
        "classifications": {HYPOTHESIS_ID: CLASSIFICATION},
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
                "hypothesis_id": HYPOTHESIS_ID,
                "classification": CLASSIFICATION,
                "readback": registry["storage"]["readback"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
