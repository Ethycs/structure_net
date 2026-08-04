#!/usr/bin/env python3
"""Store the MNIST/Fashion replication evidence in the NAL hypothesis ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_architecture_lab.meta_hypothesis import store_meta_hypothesis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replication-results",
        type=Path,
        default=Path(
            "data/experiments/patched_density_replication/20260804_seed4/results.json"
        ),
    )
    parser.add_argument(
        "--checkpoint-audit",
        type=Path,
        default=Path("data/experiments/mnist_checkpoint_audit/20260804/results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/meta_hypotheses/meta-extrema-sidecar-transfer-mnist-fashion-v1.json"
        ),
    )
    parser.add_argument(
        "--chromadb-path",
        type=Path,
        default=Path("data/chroma_db"),
    )
    parser.add_argument(
        "--no-chromadb",
        action="store_true",
        help="Write only the aggregate JSON record.",
    )
    args = parser.parse_args()

    record = store_meta_hypothesis(
        replication_path=args.replication_results,
        checkpoint_audit_path=args.checkpoint_audit,
        output_path=args.output,
        chromadb_path=None if args.no_chromadb else args.chromadb_path,
    )
    print(
        json.dumps(
            {
                "hypothesis_id": record["hypothesis"]["id"],
                "confirmation_status": record["hypothesis"][
                    "confirmation_status"
                ],
                "aggregate_path": record["storage"]["aggregate_path"],
                "experiment_ids": record["storage"]["experiment_ids"],
                "result_hashes": record["storage"].get("result_hashes", []),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
