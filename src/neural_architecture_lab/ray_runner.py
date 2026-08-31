"""Ray execution backend for the Neural Architecture Lab.

Submits NAL experiments as Ray tasks so research campaigns share one
cluster-wide resource ledger with serving workloads (the network Ray Serve +
Triton integration), instead of privately claiming GPUs through the local
device-slot scheduler.

Design notes:

- The worker contract is unchanged: ``run_test_function(experiment,
  device_id, worker)`` executes inside a Ray task exactly as it executes
  inside the local spawn-context process pool, so results are byte-compatible
  with the local backend and the fingerprint ledger resumes either.
- Under Ray, GPU placement belongs to Ray: each task requests
  ``num_gpus = 1 / gpu_slots_per_device`` and sees only its assigned GPUs via
  ``CUDA_VISIBLE_DEVICES``, so workers always address ``cuda:0``. This
  removes the host's physical-vs-logical ordinal hazard for scheduled work.
- ``ray`` is imported lazily and is not a package dependency; constructing
  the runner without Ray installed raises an actionable error. The cluster's
  Ray version is pinned network-wide by the infrastructure owner — install to
  match that pin, not ad hoc.
- Fractional GPUs are scheduler bookkeeping, not VRAM isolation. Preemption
  is cooperative: campaigns tolerate eviction because completed experiments
  resume from the fingerprint ledger.
"""

from __future__ import annotations

import asyncio
import importlib.util
import time
import traceback
from typing import Any, List, Optional

from .core import (
    Experiment,
    ExperimentResult,
    ExperimentRunnerBase,
    ExperimentStatus,
    ExperimentWorker,
    LabConfig,
    utc_now,
)
from .local_scheduler import ExperimentResultLedger


RAY_INSTALL_HINT = (
    "RayExperimentRunner requires the 'ray' package, which is intentionally "
    "not a project dependency: the shared cluster pins one exact Ray version "
    "network-wide, and every node must install that same pin. Obtain the "
    "pinned version from the cluster owner (central-dev) and install it into "
    "this environment before selecting execution_backend='ray'."
)


def gpus_per_experiment(config: LabConfig) -> float:
    """Fractional GPU request per experiment, mirroring local slot math."""
    if not config.device_ids or all(device < 0 for device in config.device_ids):
        return 0.0
    return 1.0 / max(1, int(config.gpu_slots_per_device))


class RayExperimentRunner(ExperimentRunnerBase):
    """Run NAL experiments as tasks on a shared Ray cluster."""

    def __init__(
        self,
        config: LabConfig,
        worker: Optional[ExperimentWorker] = None,
        *,
        address: Optional[str] = None,
    ) -> None:
        if importlib.util.find_spec("ray") is None:
            raise ImportError(RAY_INSTALL_HINT)
        super().__init__(config, worker)
        self.address = address
        self.max_retries = max(0, int(config.max_experiment_retries))
        self.resume_completed = bool(config.resume_completed_experiments)
        self.result_ledger = ExperimentResultLedger(config.results_dir)
        self.gpus_per_experiment = gpus_per_experiment(config)
        self._remote_worker = None
        self._started_ray = False

    # ------------------------------------------------------------------
    # Cluster lifecycle
    # ------------------------------------------------------------------

    def _ensure_ray(self):
        import ray

        if not ray.is_initialized():
            ray.init(
                address=self.address,
                namespace="neural-architecture-lab",
                ignore_reinit_error=True,
            )
            self._started_ray = self.address is None
        if self._remote_worker is None:
            from .runners import run_test_function

            self._remote_worker = ray.remote(
                num_gpus=self.gpus_per_experiment,
                max_retries=0,  # NAL owns retries so attempts are ledger-visible
            )(run_test_function)
        return ray

    def shutdown(self) -> None:
        """Detach from the cluster (stops Ray only if this runner started it)."""
        import ray

        if self._started_ray and ray.is_initialized():
            ray.shutdown()
        self._remote_worker = None
        self._started_ray = False

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _failure_result(
        self, experiment: Experiment, error: BaseException, started: float
    ) -> ExperimentResult:
        return ExperimentResult(
            experiment_id=experiment.id,
            hypothesis_id=experiment.hypothesis_id,
            metrics={},
            primary_metric=0.0,
            model_architecture=list(experiment.parameters.get("architecture", [])),
            model_parameters=0,
            training_time=time.time() - started,
            error=f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
        )

    async def run_experiment(self, experiment: Experiment) -> ExperimentResult:
        if self.worker is None:
            raise RuntimeError("No experiment worker has been configured")
        if self.resume_completed:
            resumed = self.result_ledger.load(experiment)
            if resumed is not None:
                experiment.status = ExperimentStatus.COMPLETED
                experiment.result = resumed
                experiment.completed_at = resumed.timestamp
                return resumed

        ray = self._ensure_ray()
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = utc_now()
        # Ray masks GPU visibility per task, so GPU workers always see their
        # assignment as logical device 0; CPU-only plans use -1 unchanged.
        device_id = 0 if self.gpus_per_experiment > 0 else -1

        result: ExperimentResult
        attempts = 0
        started = time.time()
        while True:
            attempts += 1
            try:
                reference = self._remote_worker.remote(
                    experiment, device_id, self.worker
                )
                result = await asyncio.wrap_future(reference.future())
            except Exception as error:  # noqa: BLE001 - converted to result
                result = self._failure_result(experiment, error, started)
            if result.error is None or attempts > self.max_retries:
                break

        self.result_ledger.save(
            experiment,
            result,
            attempts=attempts,
            device_id=device_id,
        )
        experiment.status = result.status
        experiment.completed_at = utc_now()
        experiment.result = result
        return result

    async def run_experiments(
        self, experiments: List[Experiment]
    ) -> List[ExperimentResult]:
        # Ray's scheduler is the concurrency limiter (tasks queue on GPU
        # resources); a semaphore only bounds driver-side submission bursts.
        semaphore = asyncio.Semaphore(
            max(1, int(self.config.max_parallel_experiments))
        )

        async def bounded(experiment: Experiment) -> ExperimentResult:
            async with semaphore:
                return await self.run_experiment(experiment)

        return await asyncio.gather(
            *(bounded(experiment) for experiment in experiments)
        )


def select_runner(config: LabConfig, worker: Optional[ExperimentWorker] = None) -> Any:
    """Choose the execution backend declared by ``config.execution_backend``."""
    backend = getattr(config, "execution_backend", "local") or "local"
    if backend == "ray":
        return RayExperimentRunner(
            config, worker, address=getattr(config, "ray_address", None)
        )
    if backend == "local":
        from .runners import AsyncExperimentRunner

        return AsyncExperimentRunner(config, worker)
    raise ValueError(f"unknown execution backend: {backend!r}")
