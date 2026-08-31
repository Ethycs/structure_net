import asyncio
import importlib.util

import pytest

from neural_architecture_lab.core import (
    Experiment,
    ExperimentResult,
    LabConfig,
)
import neural_architecture_lab.ray_runner as ray_runner


RAY_AVAILABLE = importlib.util.find_spec("ray") is not None


def _config(**overrides) -> LabConfig:
    values = dict(
        results_dir="nal_results_test_ray",
        device_ids=[0, 1],
        gpu_slots_per_device=2,
    )
    values.update(overrides)
    return LabConfig(**values)


def test_lab_config_declares_the_backend_fields() -> None:
    config = LabConfig()
    assert config.execution_backend == "local"
    assert config.ray_address is None


def test_gpus_per_experiment_mirrors_slot_math() -> None:
    assert ray_runner.gpus_per_experiment(_config(gpu_slots_per_device=2)) == 0.5
    assert ray_runner.gpus_per_experiment(_config(gpu_slots_per_device=1)) == 1.0
    assert ray_runner.gpus_per_experiment(_config(device_ids=[])) == 0.0
    assert ray_runner.gpus_per_experiment(_config(device_ids=[-1])) == 0.0


def test_select_runner_rejects_unknown_backends() -> None:
    config = _config()
    config.execution_backend = "slurm"
    with pytest.raises(ValueError, match="unknown execution backend"):
        ray_runner.select_runner(config)


def test_select_runner_returns_local_backend_by_default() -> None:
    # Assert on the selected class by name rather than identity: other suites
    # stub or reload ``neural_architecture_lab.runners``, which rebinds the
    # class object and makes an isinstance check order-dependent.
    runner = ray_runner.select_runner(_config())
    assert type(runner).__name__ == "AsyncExperimentRunner"
    assert not isinstance(runner, ray_runner.RayExperimentRunner)


@pytest.mark.skipif(RAY_AVAILABLE, reason="ray installed; hint path inactive")
def test_missing_ray_raises_an_actionable_error() -> None:
    with pytest.raises(ImportError, match="pinned"):
        ray_runner.RayExperimentRunner(_config())
    config = _config()
    config.execution_backend = "ray"
    with pytest.raises(ImportError, match="central-dev"):
        ray_runner.select_runner(config)


def _cpu_worker(experiment: Experiment, device_id: int) -> ExperimentResult:
    assert device_id == -1
    return ExperimentResult(
        experiment_id=experiment.id,
        hypothesis_id=experiment.hypothesis_id,
        metrics={"echo": float(experiment.parameters["value"])},
        primary_metric=float(experiment.parameters["value"]),
        model_architecture=[],
        model_parameters=0,
        training_time=0.0,
    )


@pytest.mark.skipif(not RAY_AVAILABLE, reason="ray not installed (awaiting pin)")
def test_ray_backend_runs_a_cpu_experiment(tmp_path) -> None:
    config = _config(
        results_dir=str(tmp_path),
        device_ids=[-1],
        resume_completed_experiments=True,
    )
    runner = ray_runner.RayExperimentRunner(config, worker=_cpu_worker)
    experiment = Experiment(
        id="ray_smoke_1",
        hypothesis_id="ray-backend-smoke",
        name="ray smoke",
        parameters={"value": 41.0},
    )
    try:
        result = asyncio.run(runner.run_experiment(experiment))
        assert result.error is None
        assert result.primary_metric == 41.0
        # Ledger resume returns the persisted result without re-executing.
        resumed = asyncio.run(runner.run_experiment(experiment))
        assert resumed.primary_metric == 41.0
    finally:
        runner.shutdown()
