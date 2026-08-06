#!/usr/bin/env python3
"""Refine final TinyLLM task fronts on a denser continuous-depth grid."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

try:
    from experiments.structure_net.tinyllm_depth_graded_quotient import (
        DepthGradedCampaignConfig,
        analyze_depth_diagram,
    )
    from experiments.structure_net.tinyllm_predictive_circle import (
        CircleTaskConfig,
        _resolve_device,
    )
except ModuleNotFoundError:
    from tinyllm_depth_graded_quotient import (  # type: ignore[no-redef]
        DepthGradedCampaignConfig,
        analyze_depth_diagram,
    )
    from tinyllm_predictive_circle import (  # type: ignore[no-redef]
        CircleTaskConfig,
        _resolve_device,
    )
from structure_net.components.models import TinyLLMModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_depth_graded_quotient/"
            "20260805_d8_seed7/results.json"
        ),
    )
    parser.add_argument("--depth-step", type=float, default=0.05)
    parser.add_argument("--maximum-depth", type=float, default=None)
    parser.add_argument("--phase-points", type=int, default=128)
    parser.add_argument("--device", default="cuda:auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_depth_graded_quotient/"
            "20260805_d8_seed7/front_refinement.json"
        ),
    )
    args = parser.parse_args()
    source = json.loads(args.results.read_text(encoding="utf-8"))
    task = CircleTaskConfig(**source["task_config"])
    layer_count = int(source["runs"][0]["model_config"]["n_layer"])
    maximum_depth = (
        float(layer_count) if args.maximum_depth is None else args.maximum_depth
    )
    if not 0.0 < maximum_depth <= layer_count:
        raise ValueError("maximum depth must be inside the model")
    intervals = int(round(maximum_depth / args.depth_step))
    depths = np.linspace(0.0, maximum_depth, intervals + 1)
    device = _resolve_device(args.device)
    evaluation_config = DepthGradedCampaignConfig(
        preset=source["campaign_config"]["preset"],
        seed=source["campaign_config"]["seed"],
        training_steps=1,
        checkpoints=(0, 1),
        depth_step=args.depth_step,
        phase_points=args.phase_points,
        max_phase_points=1_024,
        max_row_phase_points=8_192,
        device=args.device,
        save_checkpoints=False,
    )
    records = []
    root = args.results.parent
    for run in source["runs"]:
        checkpoint = root / run["model_checkpoint"]
        model = TinyLLMModel.from_checkpoint(
            checkpoint, map_location=device
        ).eval()
        diagram, _ = analyze_depth_diagram(
            model,
            task,
            depths,
            run["quotient"],
            evaluation_config,
            device,
            final_checkpoint=False,
        )
        records.append(
            {
                "experiment_id": run["experiment_id"],
                "training_arm": run["training_arm"],
                "quotient": run["quotient"],
                "coarse_front_depth": run["front_trajectory"][-1]["front_depth"],
                "refined_front_depth": diagram["front_depth"],
                "phase_grid_points": diagram["phase_grid_points"],
                "depth_cells": diagram["depth_cells"],
                **(
                    {"depth_defect_charge": diagram["depth_defect_charge"]}
                    if run["quotient"] == "phase_circle"
                    else {}
                ),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            run["experiment_id"],
            json.dumps(
                {
                    "coarse": records[-1]["coarse_front_depth"],
                    "refined": records[-1]["refined_front_depth"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    result = {
        "schema_version": "nal.tinyllm-depth-front-refinement.v1",
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_results": str(args.results),
        "depth_step": args.depth_step,
        "depth_grid": depths.tolist(),
        "evaluation_config": asdict(evaluation_config),
        "records": records,
        "interpretation_boundary": (
            "Threshold front localization is grid-resolved to the reported depth "
            "step; branch geometry remains an operational quotient proxy."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
