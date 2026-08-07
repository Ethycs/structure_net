#!/usr/bin/env python3
"""Store the four TinyLLM geometry-suite results in the NAL ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_architecture_lab.geometry_experiment_suite_meta_hypothesis import (
    build_suite_records,
    store_suite_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relative", type=Path, default=Path("data/experiments/tinyllm_relative_carrier_factorization/20260806_d8_preregistered/campaign_results.json"))
    parser.add_argument("--ladder", type=Path, default=Path("data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered/campaign_results.json"))
    parser.add_argument("--certificate", type=Path, default=Path("data/experiments/tinyllm_defect_certification/20260806_d6_step15/results.json"))
    parser.add_argument("--kernel", type=Path, default=Path("data/experiments/tinyllm_normal_kernel_radius/20260806_d6_step15_refined/results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/meta_hypotheses"))
    parser.add_argument("--chromadb-path", type=Path, default=Path("data/chroma_db"))
    parser.add_argument("--no-chromadb", action="store_true")
    args = parser.parse_args()
    records = build_suite_records(args.relative, args.ladder, args.certificate, args.kernel)
    store_suite_records(records, args.output_dir, chromadb_path=None if args.no_chromadb else args.chromadb_path)
    print(json.dumps([
        {
            "hypothesis_id": record["hypothesis"]["id"],
            "confirmation_status": record["hypothesis"]["confirmation_status"],
            "confirmed": record["hypothesis"]["confirmed"],
            "direct_experiments": len(record["storage"]["experiment_ids"]),
        }
        for record in records
    ], indent=2))


if __name__ == "__main__":
    main()
