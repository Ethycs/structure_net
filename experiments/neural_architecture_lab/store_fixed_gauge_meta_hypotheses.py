#!/usr/bin/env python3
"""Store the TinyLLM fixed-gauge writer and oracle decomposition evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_architecture_lab.fixed_gauge_meta_hypotheses import (
    store_fixed_gauge_error_decomposition_meta_hypothesis,
    store_fixed_gauge_writer_capacity_meta_hypothesis,
    store_fixed_semantic_gauge_writer_meta_hypothesis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--writer-results",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_fixed_semantic_gauge_writer/"
            "20260806_d6_preregistered_diagnostic/campaign_results.json"
        ),
    )
    parser.add_argument(
        "--decomposition-results",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_fixed_gauge_error_decomposition/"
            "20260806_d6_preregistered_diagnostic/campaign_results.json"
        ),
    )
    parser.add_argument(
        "--capacity-results",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_fixed_gauge_writer_capacity/"
            "20260806_d6_preregistered_diagnostic/campaign_results.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/meta_hypotheses")
    )
    parser.add_argument("--chromadb-path", type=Path, default=Path("data/chroma_db"))
    parser.add_argument("--no-chromadb", action="store_true")
    args = parser.parse_args()
    chromadb = None if args.no_chromadb else args.chromadb_path
    writer = store_fixed_semantic_gauge_writer_meta_hypothesis(
        args.writer_results,
        args.output_dir / "tinyllm-c2-fixed-semantic-gauge-writer-v1.json",
        chromadb_path=chromadb,
    )
    decomposition = store_fixed_gauge_error_decomposition_meta_hypothesis(
        args.decomposition_results,
        args.output_dir / "tinyllm-c2-fixed-gauge-error-decomposition-v1.json",
        chromadb_path=chromadb,
    )
    capacity = store_fixed_gauge_writer_capacity_meta_hypothesis(
        args.capacity_results,
        args.output_dir / "tinyllm-c2-fixed-gauge-writer-capacity-v1.json",
        chromadb_path=chromadb,
    )
    print(
        json.dumps(
            {
                "writer": {
                    "hypothesis_id": writer["hypothesis"]["id"],
                    "status": writer["hypothesis"]["confirmation_status"],
                    "readback": writer["storage"].get("readback"),
                },
                "decomposition": {
                    "hypothesis_id": decomposition["hypothesis"]["id"],
                    "status": decomposition["hypothesis"]["confirmation_status"],
                    "readback": decomposition["storage"].get("readback"),
                },
                "capacity": {
                    "hypothesis_id": capacity["hypothesis"]["id"],
                    "status": capacity["hypothesis"]["confirmation_status"],
                    "readback": capacity["storage"].get("readback"),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
