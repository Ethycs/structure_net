#!/usr/bin/env python3
"""Store the preregistered TinyLLM finite noisy-bias pilot result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_architecture_lab.noisy_bias_pilot_acquisition_meta_hypothesis import (
    store_noisy_bias_pilot_acquisition_meta_hypothesis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_noisy_bias_pilot_acquisition/"
            "20260810_d16_preregistered/campaign_results.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/meta_hypotheses/"
            "tinyllm-noisy-bias-pilot-acquisition-v1.json"
        ),
    )
    parser.add_argument("--chromadb-path", type=Path, default=Path("data/chroma_db"))
    parser.add_argument("--no-chromadb", action="store_true")
    args = parser.parse_args()
    record = store_noisy_bias_pilot_acquisition_meta_hypothesis(
        args.results,
        args.output,
        chromadb_path=None if args.no_chromadb else args.chromadb_path,
    )
    print(
        json.dumps(
            {
                "hypothesis_id": record["hypothesis"]["id"],
                "confirmed": record["hypothesis"]["confirmed"],
                "confirmation_status": record["hypothesis"][
                    "confirmation_status"
                ],
                "checkpoint_summary_count": len(
                    record["storage"]["experiment_ids"]
                ),
                "readback": record["storage"].get("readback"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
