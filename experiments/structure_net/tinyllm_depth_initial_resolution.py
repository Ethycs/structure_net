#!/usr/bin/env python3
"""Resolve near-zero circular maps at selected depths of initialized TinyLLM."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

try:
    from experiments.structure_net.tinyllm_depth_graded_quotient import (
        resolve_circle_depth_record,
    )
    from experiments.structure_net.tinyllm_predictive_circle import (
        CircleTaskConfig,
        _resolve_device,
        _tinyllm_config,
    )
except ModuleNotFoundError:
    from tinyllm_depth_graded_quotient import (  # type: ignore[no-redef]
        resolve_circle_depth_record,
    )
    from tinyllm_predictive_circle import (  # type: ignore[no-redef]
        CircleTaskConfig,
        _resolve_device,
        _tinyllm_config,
    )
from structure_net.components.models import TinyLLMModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="d8")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--depths", default="3.75,4,4.5,6,6.25")
    parser.add_argument("--initial-phase-points", type=int, default=1024)
    parser.add_argument("--maximum-phase-points", type=int, default=65536)
    parser.add_argument("--device", default="cuda:auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_depth_graded_quotient/"
            "20260805_d8_seed7/initial_resolution.json"
        ),
    )
    args = parser.parse_args()
    task = CircleTaskConfig()
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = TinyLLMModel(_tinyllm_config(args.preset, task, args.seed)).to(device)
    depths = [float(item) for item in args.depths.split(",") if item.strip()]
    records = [
        resolve_circle_depth_record(
            model,
            task,
            depth,
            device,
            initial_phase_points=args.initial_phase_points,
            maximum_phase_points=args.maximum_phase_points,
        )
        for depth in depths
    ]
    result = {
        "schema_version": "nal.tinyllm-depth-initial-resolution.v1",
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preset": args.preset,
        "seed": args.seed,
        "depths": depths,
        "records": records,
        "all_rows_resolved": all(item["sampling_resolved"] for item in records),
        "interpretation_boundary": (
            "Adaptive point sampling resolves winding numerically but does not "
            "certify nonvanishing with intervals."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
