#!/usr/bin/env python3
"""Launch reproducible TinyLLM feedback seeds through canonical local NAL.

Each NAL experiment owns one seed and executes the matched baseline,
recompute-only, and random-feedback arms sequentially. Independent seeds may
share a GPU through calibrated device slots. This remains systems evidence;
the arithmetic-token task is not a language-model quality benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

try:
    from experiments.structure_net.tinyllm_feedback_shakedown import (
        ShakedownConfig,
        run_shakedown,
    )
except ModuleNotFoundError:  # Direct ``python path/to/script.py`` execution.
    from tinyllm_feedback_shakedown import (  # type: ignore[no-redef]
        ShakedownConfig,
        run_shakedown,
    )
from neural_architecture_lab.core import (
    Experiment,
    ExperimentResult,
    ExperimentStatus,
    LabConfig,
)
from neural_architecture_lab.runners import AsyncExperimentRunner


SCHEMA_VERSION = "nal.tinyllm-feedback-local-campaign.v1"
HYPOTHESIS_ID = "tinyllm-delayed-feedback-lifecycle-v1"
DEFAULT_SEEDS = (7, 17, 29, 41, 53)
DEFAULT_MEMORY_GB = {
    "tiny": 0.25,
    "d6": 1.5,
    "d8": 2.25,
    "d10": 3.25,
    "d11": 4.0,
    "d12": 4.75,
}


@dataclass(frozen=True)
class TinyLLMNALCampaignConfig:
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    preset: str = "tiny"
    training_steps: int = 8
    batch_size: int = 8
    sequence_length: int = 12
    vocab_size: int = 64
    learning_rate: float = 0.01
    device_ids: tuple[int, ...] = (-1,)
    gpu_slots_per_device: int = 1
    gpu_memory_per_experiment_gb: Optional[float] = None
    max_gpu_slots_per_device: int = 3
    max_parallel_experiments: int = 8
    max_retries: int = 1
    resume: bool = True
    isolated_timing: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Campaign seeds must be non-empty and distinct")
        if self.training_steps < 0:
            raise ValueError("training_steps cannot be negative")
        if self.isolated_timing and self.max_parallel_experiments != 1:
            raise ValueError("isolated timing requires max_parallel_experiments=1")


def _result_for_arm(bundle: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    return next(result for result in bundle["results"] if result["arm"] == arm)


def tinyllm_feedback_campaign_worker(
    experiment: Experiment,
    device_id: int,
) -> ExperimentResult:
    """Canonical, picklable NAL worker for one complete matched seed."""
    parameters = experiment.parameters
    device = "cpu" if device_id < 0 else f"cuda:{device_id}"
    seed = int(experiment.seed if experiment.seed is not None else parameters["seed"])
    output_dir = Path(parameters["output_dir"]) / f"seed_{seed}"
    config = ShakedownConfig(
        preset=str(parameters["preset"]),
        vocab_size=int(parameters["vocab_size"]),
        sequence_length=int(parameters["sequence_length"]),
        batch_size=int(parameters["batch_size"]),
        training_steps=int(parameters["training_steps"]),
        learning_rate=float(parameters["learning_rate"]),
        seed=seed,
        device=device,
    )

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    bundle = run_shakedown(config, output_dir)
    if device_id >= 0:
        torch.cuda.synchronize()
    duration = time.perf_counter() - started
    peak_allocated_gb = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if device_id >= 0
        else 0.0
    )
    peak_reserved_gb = (
        torch.cuda.max_memory_reserved() / (1024**3)
        if device_id >= 0
        else 0.0
    )

    baseline = _result_for_arm(bundle, "baseline")
    recompute = _result_for_arm(bundle, "recompute_control")
    feedback = _result_for_arm(bundle, "random_feedback")
    metrics = {
        "seed": float(seed),
        "logical_device_id": float(device_id),
        "baseline_final_loss": float(baseline["metrics"]["final_loss"]),
        "recompute_final_loss": float(recompute["metrics"]["final_loss"]),
        "feedback_final_loss": float(feedback["metrics"]["final_loss"]),
        "feedback_final_loss_delta": float(
            bundle["comparisons"]["feedback_final_loss_delta"]
        ),
        "matched_initialization": float(
            bundle["comparisons"]["all_arms_share_pre_intervention_loss"]
        ),
        "checkpoint_round_trips_exact": float(
            bundle["comparisons"]["all_checkpoint_round_trips_exact"]
        ),
        "total_parameters": float(feedback["metrics"]["total_parameters"]),
        "feedback_parameters": float(feedback["metrics"]["feedback_parameters"]),
        "active_feedback_connections": float(
            feedback["metrics"]["active_feedback_connections"]
        ),
        "peak_cuda_allocated_gb": peak_allocated_gb,
        "peak_cuda_reserved_gb": peak_reserved_gb,
        "timing_isolated": float(bool(parameters["isolated_timing"])),
    }
    model = bundle["model_config"]
    architecture = (
        [int(model["vocab_size"])]
        + [int(model["n_embd"])] * int(model["n_layer"])
        + [int(model["vocab_size"])]
    )
    return ExperimentResult(
        experiment_id=experiment.id,
        hypothesis_id=experiment.hypothesis_id,
        metrics=metrics,
        primary_metric=-metrics["feedback_final_loss"],
        model_architecture=architecture,
        model_parameters=int(feedback["metrics"]["total_parameters"]),
        training_time=duration,
        training_history=list(feedback["training_history"]),
        model_checkpoint=str(output_dir / feedback["model_checkpoint"]),
        observations=[
            "systems_lifecycle_only_not_quality_evidence",
            f"device={device}",
            (
                "isolated_timing"
                if parameters["isolated_timing"]
                else "shared_run_timing_is_not_a_benchmark"
            ),
        ],
    )


def _experiments(
    config: TinyLLMNALCampaignConfig,
    output_dir: Path,
) -> List[Experiment]:
    common = {
        "preset": config.preset,
        "training_steps": config.training_steps,
        "batch_size": config.batch_size,
        "sequence_length": config.sequence_length,
        "vocab_size": config.vocab_size,
        "learning_rate": config.learning_rate,
        "output_dir": str(output_dir / "runs"),
        "isolated_timing": config.isolated_timing,
        "worker_schema_version": SCHEMA_VERSION,
    }
    return [
        Experiment(
            id=f"tinyllm-feedback-seed-{seed}",
            hypothesis_id=HYPOTHESIS_ID,
            name=f"TinyLLM feedback matched controls, seed {seed}",
            parameters={**common, "seed": seed},
            seed=seed,
        )
        for seed in config.seeds
    ]


def _result_record(
    result: ExperimentResult, *, isolated_timing: bool
) -> Dict[str, Any]:
    return {
        "experiment_id": result.experiment_id,
        "status": result.status.value,
        "metrics": result.metrics,
        "primary_metric": result.primary_metric,
        "model_parameters": result.model_parameters,
        "training_time": result.training_time if isolated_timing else None,
        "shared_execution_wall_time": (
            None if isolated_timing else result.training_time
        ),
        "model_checkpoint": result.model_checkpoint,
        "observations": result.observations,
        "error": result.error,
    }


async def run_campaign(
    config: TinyLLMNALCampaignConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    """Execute the campaign and persist a resumable aggregate."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lab_config = LabConfig(
        project_name="tinyllm_feedback_local_campaign",
        results_dir=str(output_dir),
        device_ids=list(config.device_ids),
        max_parallel_experiments=config.max_parallel_experiments,
        gpu_slots_per_device=config.gpu_slots_per_device,
        gpu_memory_per_experiment_gb=config.gpu_memory_per_experiment_gb,
        max_gpu_slots_per_device=config.max_gpu_slots_per_device,
        max_experiment_retries=config.max_retries,
        resume_completed_experiments=config.resume,
        auto_balance=False,
        enable_wandb=False,
        verbose=True,
    )
    runner = AsyncExperimentRunner(lab_config, tinyllm_feedback_campaign_worker)
    results = await runner.run_experiments(_experiments(config, output_dir))
    successful = [result for result in results if result.error is None]
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "claim_scope": "systems_lifecycle_only_not_quality_evidence",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": asdict(config),
        "scheduler": {
            "backend": "nal_local_spawn",
            "logical_device_slots": list(runner.slot_plan.slots),
            "slots_by_device": {
                str(key): value
                for key, value in runner.slot_plan.slots_by_device.items()
            },
            "free_memory_gb": {
                str(key): value
                for key, value in runner.slot_plan.free_memory_gb.items()
            },
            "calibration": runner.slot_plan.calibration,
            "isolated_timing": config.isolated_timing,
        },
        "summary": {
            "requested": len(results),
            "completed": len(successful),
            "failed": len(results) - len(successful),
            "all_matched": (
                len(successful) == len(results)
                and bool(successful)
                and all(
                    result.metrics.get("matched_initialization") == 1.0
                    for result in successful
                )
            ),
            "all_checkpoints_exact": (
                len(successful) == len(results)
                and bool(successful)
                and all(
                    result.metrics.get("checkpoint_round_trips_exact") == 1.0
                    for result in successful
                )
            ),
        },
        "results": [
            _result_record(result, isolated_timing=config.isolated_timing)
            for result in results
        ],
    }
    destination = output_dir / "campaign_results.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return bundle


def _parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_devices(value: str) -> tuple[int, ...]:
    normalized = value.strip().lower()
    if normalized == "cpu":
        return (-1,)
    if normalized == "auto":
        if not torch.cuda.is_available():
            return (-1,)
        return tuple(range(torch.cuda.device_count()))
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument(
        "--preset", default="tiny", choices=tuple(DEFAULT_MEMORY_GB)
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", default="auto", help="Logical CUDA IDs, auto, or cpu")
    parser.add_argument(
        "--slots-per-gpu",
        type=int,
        default=0,
        help="Fixed slots per GPU; 0 calibrates from currently free memory",
    )
    parser.add_argument("--memory-per-seed-gb", type=float)
    parser.add_argument("--max-slots-per-gpu", type=int, default=3)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--isolated-timing", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/tinyllm_feedback_nal_campaign"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    devices = _parse_devices(args.gpus)
    max_parallel = 1 if args.isolated_timing else args.max_parallel
    memory_gb = (
        args.memory_per_seed_gb
        if args.memory_per_seed_gb is not None
        else DEFAULT_MEMORY_GB[args.preset]
    )
    config = TinyLLMNALCampaignConfig(
        seeds=_parse_seeds(args.seeds),
        preset=args.preset,
        training_steps=args.steps,
        batch_size=args.batch_size,
        device_ids=devices,
        gpu_slots_per_device=args.slots_per_gpu,
        gpu_memory_per_experiment_gb=memory_gb,
        max_gpu_slots_per_device=args.max_slots_per_gpu,
        max_parallel_experiments=max_parallel,
        max_retries=args.retries,
        resume=args.resume,
        isolated_timing=args.isolated_timing,
    )
    bundle = asyncio.run(run_campaign(config, args.output))
    print(json.dumps({"scheduler": bundle["scheduler"], "summary": bundle["summary"]}, indent=2))
    print(args.output / "campaign_results.json")
    return 0 if bundle["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
