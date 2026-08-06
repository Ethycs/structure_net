#!/usr/bin/env python3
"""Build a layer-by-task geometry atlas for frozen TinyLLM checkpoints.

The atlas uses paired task-reference geometry, nonlinear local decoders,
matched-fiber collapse, and independently discovered persistent-cohomology
coordinates at the query embedding and after every attention and MLP sublayer.
It reports carrier/rank proxies explicitly; it does not claim a validated
chain-level induced map.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

try:
    from experiments.structure_net.tinyllm_internal_quotient_probe import (
        FiberDataset,
        ProbeCampaignConfig,
        _inputs_for_family,
        fit_probe_suite,
        generate_matched_fiber_dataset,
    )
    from experiments.structure_net.tinyllm_predictive_circle import (
        CircleTaskConfig,
        _resolve_device,
    )
except ModuleNotFoundError:  # Direct ``python path/to/script.py`` execution.
    from tinyllm_internal_quotient_probe import (  # type: ignore[no-redef]
        FiberDataset,
        ProbeCampaignConfig,
        _inputs_for_family,
        fit_probe_suite,
        generate_matched_fiber_dataset,
    )
    from tinyllm_predictive_circle import (  # type: ignore[no-redef]
        CircleTaskConfig,
        _resolve_device,
    )
from structure_net.components.analyzers import (
    circular_phase_alignment,
    circular_winding_degree,
    nuisance_collapse_ratio,
    paired_geometry_alignment,
    persistent_cohomology_circle_coordinate,
    representation_distance_matrix,
    summarize_persistence,
)
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-task-geometry-atlas.v1"
HYPOTHESIS_ID = "tinyllm-layer-task-carrier-and-quotient-atlas-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def stage_labels(layer_count: int) -> List[str]:
    labels = ["query_embedding"]
    for block in range(1, layer_count + 1):
        labels.extend((f"block_{block}_post_attention", f"block_{block}_post_mlp"))
    return labels


@torch.no_grad()
def extract_sublayer_representations(
    model: TinyLLMModel,
    inputs: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
) -> List[np.ndarray]:
    """Capture the fixed-query stream after each attention and MLP update."""
    if model.feedback_connections or model.refinement_steps:
        raise ValueError("sublayer atlas currently requires a feed-forward TinyLLM graph")
    model.eval()
    captured: Optional[List[List[torch.Tensor]]] = None
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size].to(device)
        positions = torch.arange(batch.shape[1], device=device, dtype=torch.long)
        value = model.transformer["wte"](batch) + model.transformer["wpe"](positions)
        stages = [value]
        for block in model.transformer["h"]:
            value = value + block.attn(block.ln_1(value))
            stages.append(value)
            value = value + block.mlp(block.ln_2(value))
            stages.append(value)
        if captured is None:
            captured = [[] for _ in stages]
        for index, state in enumerate(stages):
            captured[index].append(state[:, -1, :].detach().cpu())
    assert captured is not None
    return [torch.cat(values).float().numpy() for values in captured]


def _reference_grid(
    task: CircleTaskConfig,
    points: int,
    *,
    family: str,
    seed: int,
) -> tuple[torch.Tensor, np.ndarray]:
    future = np.linspace(0.0, 2.0 * math.pi, points, endpoint=False)
    directions = np.ones(points)
    current = (future - task.future_delta) % (2.0 * math.pi)
    inputs = _inputs_for_family(
        task, current, directions, family=family, seed=seed
    )
    return inputs, future


def _reference_distances(target_phase: np.ndarray, quotient: str) -> np.ndarray:
    if quotient == "phase_circle":
        target = np.column_stack((np.cos(target_phase), np.sin(target_phase)))
    elif quotient == "cosine_interval":
        target = np.cos(target_phase)[:, None]
    else:
        raise ValueError(f"unknown quotient: {quotient}")
    return representation_distance_matrix(target, metric="euclidean")


def _fiber_record(values: np.ndarray, dataset: FiberDataset) -> Dict[str, float]:
    distances = representation_distance_matrix(values, metric="euclidean")
    result = nuisance_collapse_ratio(distances, dataset.fiber_id.numpy())
    if not np.isfinite(result["nuisance_collapse_ratio"]):
        result["nuisance_collapse_ratio"] = 1.0
        result["defined"] = 0.0
    else:
        result["defined"] = 1.0
    return result


def _geometry_record(
    values: np.ndarray,
    target_phase: np.ndarray,
    quotient: str,
    *,
    neighbors: int,
) -> Dict[str, Any]:
    representation = representation_distance_matrix(values, metric="euclidean")
    reference = _reference_distances(target_phase, quotient)
    persistence = summarize_persistence(representation)
    coordinate = persistent_cohomology_circle_coordinate(representation)
    alignment = circular_phase_alignment(coordinate["angles"], target_phase)
    degree = circular_winding_degree(coordinate["angles"])
    result = {
        "paired_geometry": paired_geometry_alignment(
            reference,
            representation,
            neighbors=min(neighbors, len(values) - 1),
        ),
        "normalized_h1_lifetime": persistence.normalized_h1_lifetime,
        "persistent_coordinate_found": coordinate["found"],
        "persistent_coordinate_phase_alignment": alignment["alignment"],
        "persistent_coordinate_winding_degree": degree,
        "persistent_coordinate_lifetime": coordinate["lifetime"],
    }
    robust_phase_generator = (
        quotient == "phase_circle"
        and persistence.normalized_h1_lifetime >= 0.3
        and alignment["alignment"] >= 0.8
        and abs(abs(degree) - 1.0) <= 0.15
    )
    result["target_h1_carrier_rank_proxy"] = 1 if robust_phase_generator else 0
    return result


def _first_stage(
    stages: Sequence[Mapping[str, Any]], field: str, *, skip_embedding: bool = False
) -> Optional[str]:
    for stage in stages[1:] if skip_embedding else stages:
        if stage[field]:
            return str(stage["stage"])
    return None


def analyze_checkpoint(
    checkpoint: Path,
    task: CircleTaskConfig,
    campaign: ProbeCampaignConfig,
    device: torch.device,
    datasets: Mapping[str, FiberDataset],
) -> Dict[str, Any]:
    started = time.perf_counter()
    model = TinyLLMModel.from_checkpoint(checkpoint, map_location="cpu").to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    quotient = str(metadata["quotient"])
    preset = model.config.preset or f"d{model.config.n_layer}"
    labels = stage_labels(model.config.n_layer)

    activations = {
        key: extract_sublayer_representations(
            model,
            dataset.input_ids,
            device,
            batch_size=campaign.activation_batch_size,
        )
        for key, dataset in datasets.items()
    }
    grids: Dict[str, Any] = {}
    for family_index, family in enumerate(
        ("training_support", "disjoint_third_harmonic_drift")
    ):
        inputs, target_phase = _reference_grid(
            task,
            campaign.phase_grid_points,
            family=family,
            seed=campaign.seed + 80_003 + family_index * 1_009,
        )
        grids[family] = {
            "target_phase": target_phase,
            "activations": extract_sublayer_representations(
                model,
                inputs,
                device,
                batch_size=campaign.activation_batch_size,
            ),
        }

    atlas = []
    for index, label in enumerate(labels):
        probe = fit_probe_suite(
            activations["train"][index],
            activations["validation"][index],
            {
                "in_distribution": activations["in_distribution"][index],
                "disjoint_nuisance": activations["disjoint_nuisance"][index],
            },
            datasets["train"],
            datasets["validation"],
            {
                "in_distribution": datasets["in_distribution"],
                "disjoint_nuisance": datasets["disjoint_nuisance"],
            },
            campaign,
            device,
            seed=campaign.seed + 2_003 * index + 19 * model.config.n_layer,
        )
        geometry = {
            family: _geometry_record(
                grid["activations"][index],
                grid["target_phase"],
                quotient,
                neighbors=8,
            )
            for family, grid in grids.items()
        }
        fibers = {
            "in_distribution": _fiber_record(
                activations["in_distribution"][index], datasets["in_distribution"]
            ),
            "disjoint_nuisance": _fiber_record(
                activations["disjoint_nuisance"][index], datasets["disjoint_nuisance"]
            ),
        }
        in_domain = probe["evaluations"]["in_distribution"]
        disjoint = probe["evaluations"]["disjoint_nuisance"]
        base_geometry = geometry["training_support"]
        phase_carrier = (
            quotient == "phase_circle"
            and in_domain["phase_alignment"] >= 0.9
            and base_geometry["target_h1_carrier_rank_proxy"] == 1
        )
        cosine_quotient = (
            quotient == "cosine_interval"
            and in_domain["cosine_pearson"] >= 0.9
            and in_domain["conditional_branch_accuracy"] <= 0.6
            and fibers["in_distribution"]["nuisance_collapse_ratio"] <= 0.5
            and base_geometry["paired_geometry"]["distance_spearman"] >= 0.6
        )
        atlas.append(
            {
                "stage_index": index,
                "stage": label,
                "probe": probe,
                "geometry": geometry,
                "fiber_geometry": fibers,
                "approximate_retract_probe_score": (
                    in_domain["phase_alignment"]
                    if quotient == "phase_circle"
                    else in_domain["cosine_pearson"]
                ),
                "fiber_branch_component_proxy": (
                    2
                    if in_domain["conditional_branch_accuracy"] >= 0.8
                    else 1
                    if in_domain["conditional_branch_accuracy"] <= 0.6
                    else None
                ),
                "phase_task_carrier": phase_carrier,
                "cosine_internal_quotient": cosine_quotient,
                "held_out_branch_criterion": (
                    disjoint["conditional_branch_accuracy"] >= 0.7
                    if quotient == "phase_circle"
                    else disjoint["conditional_branch_accuracy"] <= 0.6
                ),
            }
        )

    result = {
        "experiment_id": f"tinyllm-geometry-atlas-{preset}-{quotient}-seed{metadata['seed']}",
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "checkpoint": str(checkpoint),
        "preset": preset,
        "quotient": quotient,
        "seed": int(metadata["seed"]),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "stagewise_atlas": atlas,
        "localization": {
            "first_phase_task_carrier": _first_stage(atlas, "phase_task_carrier"),
            "first_cosine_internal_quotient": _first_stage(
                atlas, "cosine_internal_quotient"
            ),
            "first_held_out_branch_criterion": _first_stage(
                atlas, "held_out_branch_criterion", skip_embedding=True
            ),
        },
        "analysis_seconds": time.perf_counter() - started,
        "scope_warning": (
            "Carrier rank is a paired coordinate/degree proxy, not a validated induced chain map. "
            "The retract score is held-out probe agreement, not a homotopy proof."
        ),
    }
    del model, activations, grids
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def aggregate_atlas(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for preset in sorted({str(run["preset"]) for run in runs}):
        cells = {
            str(run["quotient"]): run
            for run in runs
            if run["preset"] == preset
        }
        phase = cells.get("phase_circle")
        cosine = cells.get("cosine_interval")
        if not phase or not cosine:
            continue
        phase_final = phase["stagewise_atlas"][-1]
        cosine_final = cosine["stagewise_atlas"][-1]
        criteria = {
            "phase_task_carrier_localized": phase["localization"][
                "first_phase_task_carrier"
            ]
            is not None,
            "cosine_internal_quotient_localized": cosine["localization"][
                "first_cosine_internal_quotient"
            ]
            is not None,
            "phase_final_carrier_survives": phase_final["phase_task_carrier"],
            "cosine_final_quotient_survives": cosine_final[
                "cosine_internal_quotient"
            ],
            "phase_final_held_out_branch_recoverable": phase_final[
                "held_out_branch_criterion"
            ],
            "cosine_final_held_out_branch_collapsed": cosine_final[
                "held_out_branch_criterion"
            ],
        }
        result[preset] = {
            "localization": {
                "phase_task_carrier": phase["localization"][
                    "first_phase_task_carrier"
                ],
                "cosine_internal_quotient": cosine["localization"][
                    "first_cosine_internal_quotient"
                ],
            },
            "pre_registered_criteria": criteria,
            "criteria_passed": all(criteria.values()),
        }
    return result


def run_atlas_campaign(
    checkpoints: Sequence[Path],
    task: CircleTaskConfig,
    campaign: ProbeCampaignConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(campaign.device)
    torch.use_deterministic_algorithms(True)
    datasets = {
        "train": generate_matched_fiber_dataset(
            task,
            sample_count=campaign.train_samples,
            seed=campaign.seed + 101,
            family="training_support",
        ),
        "validation": generate_matched_fiber_dataset(
            task,
            sample_count=campaign.validation_samples,
            seed=campaign.seed + 211,
            family="training_support",
        ),
        "in_distribution": generate_matched_fiber_dataset(
            task,
            sample_count=campaign.test_samples,
            seed=campaign.seed + 307,
            family="training_support",
        ),
        "disjoint_nuisance": generate_matched_fiber_dataset(
            task,
            sample_count=campaign.test_samples,
            seed=campaign.seed + 401,
            family="disjoint_third_harmonic_drift",
        ),
    }
    bundle: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "running",
        "started_at": _utc_now(),
        "task_config": asdict(task),
        "campaign_config": asdict(campaign),
        "checkpoint_paths": [str(path) for path in checkpoints],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
        },
        "pre_registered_interpretation": (
            "A phase task carrier is localized when a held-out phase decoder and an "
            "independently discovered degree-one persistent coordinate agree. A cosine "
            "internal quotient is localized when cosine decodes, branch is at chance, "
            "matched fibers contract, and paired interval geometry agrees."
        ),
        "method_boundaries": [
            "target_h1_carrier_rank_proxy is not a chain-level induced-map rank",
            "approximate_retract_probe_score is not a homotopy retraction proof",
            "fiber_branch_component_proxy is not a computed Reeb graph or cosheaf",
            "only retained seed-7 checkpoints are analyzed",
        ],
        "runs": [],
    }
    partial = output_dir / "results.partial.json"
    _write_json(partial, bundle)
    for checkpoint in checkpoints:
        run = analyze_checkpoint(checkpoint, task, campaign, device, datasets)
        bundle["runs"].append(run)
        _write_json(partial, bundle)
        print(run["experiment_id"], json.dumps(run["localization"], sort_keys=True), flush=True)
    bundle["aggregates"] = aggregate_atlas(bundle["runs"])
    bundle["status"] = "completed"
    bundle["completed_at"] = _utc_now()
    observed = bool(bundle["aggregates"]) and all(
        value["criteria_passed"] for value in bundle["aggregates"].values()
    )
    bundle["claim_status"] = {
        "confirmed": False,
        "atlas_prediction_observed": observed,
        "interpretation": (
            "The task-carrier and internal-quotient localizations passed in both retained model classes."
            if observed
            else "At least one frozen task-carrier or internal-quotient localization criterion failed."
        ),
        "confirmation_limit": "One retained training seed and proxy rather than chain-level induced maps.",
    }
    _write_json(output_dir / "results.json", bundle)
    _write_json(partial, bundle)
    return bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_task_quotient_contrast/20260805_d6_d8/checkpoints"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/tinyllm_task_geometry_atlas/20260805_d6_d8_seed7"),
    )
    parser.add_argument("--device", default="cuda:auto")
    parser.add_argument("--probe-steps", type=int, default=300)
    parser.add_argument("--train-samples", type=int, default=4_096)
    parser.add_argument("--validation-samples", type=int, default=1_024)
    parser.add_argument("--test-samples", type=int, default=1_024)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoints = sorted(args.checkpoint_dir.glob("*_trained_seed7.pt"))
    if len(checkpoints) != 4:
        raise FileNotFoundError(
            f"expected four matched checkpoints in {args.checkpoint_dir}, found {len(checkpoints)}"
        )
    campaign = ProbeCampaignConfig(
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        probe_steps=args.probe_steps,
        device=args.device,
    )
    bundle = run_atlas_campaign(
        checkpoints, CircleTaskConfig(), campaign, args.output
    )
    print(json.dumps(bundle["claim_status"], indent=2, sort_keys=True))
    print(args.output / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
