#!/usr/bin/env python3
"""Register the preregistered A5--A7 TinyLLM studies in NAL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_architecture_lab.core import LabConfig
from neural_architecture_lab.dynamic_ttno_followup_hypotheses import (
    register_dynamic_ttno_followups,
)
from neural_architecture_lab.lab import NeuralArchitectureLab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/hypothesis_registry/tinyllm-dynamic-ttno-followups-v1.json"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data"),
        help="NAL storage root; its chroma_db collection receives the records.",
    )
    args = parser.parse_args()

    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "message": "A5--A7 are already finalized; registration left unchanged.",
                    },
                    indent=2,
                )
            )
            return

    lab = NeuralArchitectureLab(
        LabConfig(
            project_name="structure_net_pending_hypotheses",
            results_dir=str(args.results_dir),
            device_ids=[-1],
            max_parallel_experiments=1,
            enable_wandb=False,
            enable_adaptive_hypotheses=False,
            auto_balance=False,
            verbose=False,
        )
    )
    record = register_dynamic_ttno_followups(lab, args.output)
    print(
        json.dumps(
            {
                "status": record["status"],
                "hypothesis_ids": [item["id"] for item in record["hypotheses"]],
                "readback": record["storage"]["readback"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
