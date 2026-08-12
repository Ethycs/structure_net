#!/usr/bin/env python3
"""Store the TinyLLM frozen scalar-domain extension result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_architecture_lab.frozen_scalar_domain_extension_meta_hypothesis import (
    store_frozen_scalar_domain_extension_meta_hypothesis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_frozen_scalar_domain_extension/"
            "20260811_d10_learned_seed29_registered/result.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/meta_hypotheses/tinyllm-frozen-scalar-domain-extension-v1.json"
        ),
    )
    parser.add_argument("--chromadb-path", type=Path, default=Path("data/chroma_db"))
    parser.add_argument("--no-chromadb", action="store_true")
    args = parser.parse_args()
    record = store_frozen_scalar_domain_extension_meta_hypothesis(
        args.results,
        args.output,
        chromadb_path=None if args.no_chromadb else args.chromadb_path,
    )
    print(
        json.dumps(
            {
                "hypothesis_id": record["hypothesis"]["id"],
                "confirmed": record["hypothesis"]["confirmed"],
                "confirmation_status": record["hypothesis"]["confirmation_status"],
                "experiment_count": len(record["storage"]["experiment_ids"]),
                "readback": record["storage"].get("readback"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
