#!/usr/bin/env python3
"""Store the TinyLLM layer-by-task geometry atlas in the NAL ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_architecture_lab.task_geometry_atlas_meta_hypothesis import (
    store_task_geometry_atlas_meta_hypothesis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_task_geometry_atlas/20260805_d6_d8_seed7/results.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/meta_hypotheses/tinyllm-layer-task-carrier-and-quotient-atlas-v1.json"
        ),
    )
    parser.add_argument("--chromadb-path", type=Path, default=Path("data/chroma_db"))
    parser.add_argument("--no-chromadb", action="store_true")
    args = parser.parse_args()
    record = store_task_geometry_atlas_meta_hypothesis(
        args.results,
        args.output,
        chromadb_path=None if args.no_chromadb else args.chromadb_path,
    )
    print(
        json.dumps(
            {
                "hypothesis_id": record["hypothesis"]["id"],
                "confirmation_status": record["hypothesis"]["confirmation_status"],
                "aggregate_path": record["storage"]["aggregate_path"],
                "experiment_count": len(record["storage"]["experiment_ids"]),
                "result_hashes": record["storage"].get("result_hashes", []),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
