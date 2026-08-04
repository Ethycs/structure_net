"""Focused tests for the canonical NAL runner and experiment lifecycle."""

from datetime import datetime
from unittest.mock import patch

import pytest

from neural_architecture_lab.core import (
    Experiment,
    ExperimentResult,
    ExperimentStatus,
    LabConfig,
)
from neural_architecture_lab.runners import (
    AsyncExperimentRunner,
    ExperimentRunner,
    run_test_function,
)
from neural_architecture_lab.lab import NumpyJSONEncoder
import json


def canonical_worker(experiment: Experiment, device_id: int) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment.id,
        hypothesis_id=experiment.hypothesis_id,
        metrics={"accuracy": 0.75, "device_id": float(device_id)},
        primary_metric=0.75,
        model_architecture=[2, 1],
        model_parameters=3,
        training_time=0.01,
    )


def legacy_test_function(parameters):
    return object(), {
        "accuracy": 0.5,
        "parameters": 7,
        "training_time": 0.02,
        "architecture": [4, 1],
    }


def make_experiment() -> Experiment:
    return Experiment(
        id="experiment-1",
        hypothesis_id="hypothesis-1",
        name="tiny experiment",
        parameters={"architecture": [2, 1]},
    )


def test_worker_adapter_preserves_canonical_result():
    experiment = make_experiment()

    result = run_test_function(experiment, -1, canonical_worker)

    assert result.experiment_id == experiment.id
    assert result.status is ExperimentStatus.COMPLETED
    assert result.duration == result.training_time


def test_worker_adapter_normalizes_legacy_hypothesis_function():
    experiment = make_experiment()

    result = run_test_function(experiment, -1, legacy_test_function)

    assert result.metrics["accuracy"] == 0.5
    assert result.primary_metric == 0.5
    assert result.model_architecture == [4, 1]
    assert result.model_parameters == 7


def test_result_failure_status_and_legacy_timestamp_coercion():
    result = ExperimentResult(
        experiment_id="experiment-1",
        hypothesis_id="hypothesis-1",
        metrics={},
        primary_metric=0.0,
        model_architecture=[],
        model_parameters=0,
        training_time=0.0,
        timestamp=0.0,
        error="boom",
    )

    assert result.status is ExperimentStatus.FAILED
    assert isinstance(result.timestamp, datetime)
    assert result.timestamp.tzinfo is not None

    payload = json.loads(json.dumps(result.__dict__, cls=NumpyJSONEncoder))
    assert payload["status"] == "failed"
    assert payload["timestamp"].startswith("1970-01-01T00:00:00")


@pytest.mark.asyncio
async def test_async_runner_updates_experiment_lifecycle():
    config = LabConfig(
        auto_balance=False,
        device_ids=[-1],
        max_parallel_experiments=1,
        verbose=False,
        enable_wandb=False,
    )
    runner = AsyncExperimentRunner(config, canonical_worker)
    experiment = make_experiment()

    assert runner.current_workers == 0

    result = await runner.run_experiment(experiment)

    assert result.status is ExperimentStatus.COMPLETED
    assert experiment.status is ExperimentStatus.COMPLETED
    assert experiment.result is result
    assert isinstance(experiment.started_at, datetime)
    assert isinstance(experiment.completed_at, datetime)
    assert experiment.started_at.tzinfo is not None
    assert experiment.started_at <= experiment.completed_at


@pytest.mark.asyncio
async def test_runner_facade_implements_single_experiment_contract():
    config = LabConfig(
        auto_balance=False,
        device_ids=[-1],
        max_parallel_experiments=1,
        verbose=False,
        enable_wandb=False,
    )
    runner = ExperimentRunner(config, canonical_worker)

    result = await runner.run_experiment(make_experiment())

    assert result.primary_metric == 0.75


@pytest.mark.asyncio
async def test_cpu_batch_runner_propagates_safe_loader_settings():
    config = LabConfig(
        auto_balance=False,
        device_ids=[-1],
        max_parallel_experiments=1,
        verbose=False,
        enable_wandb=False,
    )
    runner = AsyncExperimentRunner(config, canonical_worker)
    experiment = make_experiment()

    await runner.run_experiments([experiment])

    assert experiment.parameters['num_workers'] == 0


@pytest.mark.asyncio
async def test_cpu_runner_does_not_dispatch_pytorch_work_to_thread_pool():
    config = LabConfig(
        auto_balance=False,
        device_ids=[-1],
        max_parallel_experiments=1,
        verbose=False,
        enable_wandb=False,
    )
    runner = AsyncExperimentRunner(config, canonical_worker)

    with patch(
        'neural_architecture_lab.runners.ThreadPoolExecutor',
        side_effect=AssertionError('CPU workers must execute inline'),
    ):
        result = await runner.run_experiment(make_experiment())

    assert result.status is ExperimentStatus.COMPLETED
