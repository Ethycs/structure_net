#!/usr/bin/env python3
"""Store the TinyLLM continuous-carrier/readout corrective result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_architecture_lab.continuous_readout_calibration_meta_hypothesis import (
    store_continuous_readout_calibration_meta_hypothesis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_continuous_readout_calibration/"
            "20260806_d6_corrective_v3_6a88480c/campaign_results.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/meta_hypotheses/"
            "tinyllm-c2-semantic-carrier-readout-separation-v1.json"
        ),
    )
    parser.add_argument(
        "--chromadb-path", type=Path, default=Path("data/chroma_db")
    )
    parser.add_argument("--no-chromadb", action="store_true")
    args = parser.parse_args()
    record = store_continuous_readout_calibration_meta_hypothesis(
        args.results,
        args.output,
        chromadb_path=None if args.no_chromadb else args.chromadb_path,
    )
    print(
        json.dumps(
            {
                "hypothesis_id": record["hypothesis"]["id"],
                "confirmation_status": record["hypothesis"][
                    "confirmation_status"
                ],
                "experiment_count": len(record["storage"]["experiment_ids"]),
                "readback": record["storage"].get("readback"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
