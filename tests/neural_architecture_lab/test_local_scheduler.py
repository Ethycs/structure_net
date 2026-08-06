"""Tests for NAL's local device slots, retries, and resume ledger."""

from __future__ import annotations

import pytest

from neural_architecture_lab.core import (
    Experiment,
    ExperimentResult,
    LabConfig,
)
from neural_architecture_lab.local_scheduler import build_device_slot_plan
from neural_architecture_lab.runners import AsyncExperimentRunner, ParallelExperimentRunner


def _experiment() -> Experiment:
    return Experiment(
        id="scheduler-test-seed-7",
        hypothesis_id="scheduler-test",
        name="scheduler test",
        parameters={"architecture": [2, 1], "seed": 7},
        seed=7,
    )


def _success(experiment: Experiment, device_id: int) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment.id,
        hypothesis_id=experiment.hypothesis_id,
        metrics={"device_id": float(device_id), "score": 1.0},
        primary_metric=1.0,
        model_architecture=[2, 1],
        model_parameters=3,
        training_time=0.01,
    )


def test_fixed_slots_use_logical_ordinals_and_repeat_each_capacity():
    plan = build_device_slot_plan(
        [0, 1],
        slots_per_device=3,
        memory_per_experiment_gb=None,
        max_slots_per_device=4,
        usable_memory_fraction=0.8,
        cuda_available=True,
        cuda_device_count=2,
    )

    assert plan.slots == (0, 0, 0, 1, 1, 1)
    assert plan.slots_by_device == {0: 3, 1: 3}
    assert plan.calibration == "fixed"


def test_memory_calibration_respects_free_memory_safety_and_cap():
    gib = 1024**3
    plan = build_device_slot_plan(
        [0, 1],
        slots_per_device=0,
        memory_per_experiment_gb=1.5,
        max_slots_per_device=4,
        usable_memory_fraction=0.8,
        cuda_available=True,
        cuda_device_count=2,
        free_memory_bytes={0: 8 * gib, 1: 4 * gib},
    )

    assert plan.slots_by_device == {0: 4, 1: 2}
    assert plan.slots == (0, 0, 0, 0, 1, 1)
    assert plan.calibration == "free-memory"


def test_physical_ids_are_rejected_when_not_visible_logical_ordinals():
    with pytest.raises(ValueError, match="Logical CUDA device"):
        build_device_slot_plan(
            [2],
            slots_per_device=1,
            memory_per_experiment_gb=None,
            max_slots_per_device=4,
            usable_memory_fraction=0.8,
            cuda_available=True,
            cuda_device_count=1,
        )


@pytest.mark.asyncio
async def test_runner_retries_failure_and_persists_completion(tmp_path):
    attempts = []

    def flaky_worker(experiment: Experiment, device_id: int) -> ExperimentResult:
        attempts.append(device_id)
        if len(attempts) == 1:
            return ExperimentResult(
                experiment_id=experiment.id,
                hypothesis_id=experiment.hypothesis_id,
                metrics={},
                primary_metric=0.0,
                model_architecture=[2, 1],
                model_parameters=0,
                training_time=0.0,
                error="transient",
            )
        return _success(experiment, device_id)

    config = LabConfig(
        results_dir=str(tmp_path),
        device_ids=[-1],
        max_parallel_experiments=1,
        max_experiment_retries=1,
        resume_completed_experiments=True,
        auto_balance=False,
        enable_wandb=False,
        verbose=False,
    )
    runner = AsyncExperimentRunner(config, flaky_worker)
    experiment = _experiment()

    result = await runner.run_experiment(experiment)

    assert result.error is None
    assert attempts == [-1, -1]
    assert runner.result_ledger.load(experiment).metrics["score"] == 1.0


@pytest.mark.asyncio
async def test_runner_resumes_without_calling_worker(tmp_path):
    config = LabConfig(
        results_dir=str(tmp_path),
        device_ids=[-1],
        max_parallel_experiments=1,
        resume_completed_experiments=True,
        auto_balance=False,
        enable_wandb=False,
        verbose=False,
    )
    experiment = _experiment()
    first = AsyncExperimentRunner(config, _success)
    expected = await first.run_experiment(experiment)

    def must_not_run(experiment: Experiment, device_id: int) -> ExperimentResult:
        raise AssertionError("resumed experiment executed again")

    resumed_experiment = _experiment()
    second = AsyncExperimentRunner(config, must_not_run)
    actual = await second.run_experiment(resumed_experiment)

    assert actual.metrics == expected.metrics
    assert resumed_experiment.result is actual


def test_parallel_runner_name_uses_canonical_slot_backend(tmp_path):
    config = LabConfig(
        results_dir=str(tmp_path),
        device_ids=[-1],
        max_parallel_experiments=1,
        auto_balance=False,
        enable_wandb=False,
        verbose=False,
    )

    runner = ParallelExperimentRunner(config, _success)

    assert isinstance(runner, AsyncExperimentRunner)
    assert runner.slot_plan.slots == (-1,)
