#!/usr/bin/env python3
"""Register the preregistered A8 H2 gauge-channelization study in NAL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_architecture_lab.core import LabConfig
from neural_architecture_lab.h2_gauge_channelization_hypothesis import (
    register_hypothesis,
)
from neural_architecture_lab.lab import NeuralArchitectureLab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/hypothesis_registry/tinyllm-h2-gauge-channelization-v1.json"
        ),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "message": "A8 is already finalized; registration left unchanged.",
                    },
                    indent=2,
                )
            )
            return

    lab = NeuralArchitectureLab(
        LabConfig(
            project_name="structure_net_h2_gauge_channelization",
            results_dir=str(args.results_dir),
            device_ids=[-1],
            max_parallel_experiments=1,
            enable_wandb=False,
            enable_adaptive_hypotheses=False,
            auto_balance=False,
            verbose=False,
        )
    )
    record = register_hypothesis(lab, args.output)
    print(
        json.dumps(
            {
                "status": record["status"],
                "hypothesis_id": record["hypotheses"][0]["id"],
                "readback": record["storage"]["readback"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
