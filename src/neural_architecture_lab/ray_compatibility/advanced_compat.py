"""
Advanced Ray compatibility for complex NAL runners.

Provides Ray-based replacements for advanced features like:
- Async GPU streaming
- Worker pools with queues
- Mixed precision training
- Advanced profiling
"""

import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.util.queue import Queue as RayQueue
from ray.util.actor_pool import ActorPool
import torch
from torch.cuda.amp import autocast, GradScaler
import asyncio
from typing import List, Dict, Any, Optional, Callable, Union
import time
import gc
from concurrent.futures import Future
from dataclasses import dataclass

from ..core import (
    Experiment, ExperimentResult, ExperimentStatus,
    LabConfig, ExperimentRunnerBase
)
from ..advanced_runners import GPUMemoryManager


@ray.remote
class RayWorkerActor:
    """
    Ray actor that maintains state and can process multiple experiments.
    Replaces the worker pool pattern from the original implementation.
    """
    
    def __init__(self, device_id: int, worker_fn: Callable):
        self.device_id = device_id
        self.worker_fn = worker_fn
        self.device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
        
        # Set GPU device
        if torch.cuda.is_available() and device_id >= 0:
            torch.cuda.set_device(device_id)
    
    def run_experiment(self, experiment: Experiment) -> ExperimentResult:
        """Run a single experiment on this worker."""
        return self.worker_fn(experiment, self.device_id)
    
    def cleanup(self):
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


class RayWorkerPool:
    """
    Ray-based worker pool that replaces multiprocessing.Queue pattern.
    """
    
    def __init__(self, num_workers: int, worker_fn: Callable, devices: Optional[List[int]] = None):
        self.num_workers = num_workers
        self.worker_fn = worker_fn
        
        # Create Ray actors
        if devices is None:
            devices = list(range(num_workers))
        
        self.workers = [
            RayWorkerActor.remote(device_id=devices[i % len(devices)], worker_fn=worker_fn)
            for i in range(num_workers)
        ]
        
        # Create actor pool for load balancing
        self.pool = ActorPool(self.workers)
    
    def submit_experiments(self, experiments: List[Experiment]) -> List[ExperimentResult]:
        """Submit experiments to the worker pool."""
        # Submit all experiments
        futures = []
        for exp in experiments:
            future = self.pool.submit(lambda actor, e: actor.run_experiment.remote(e), exp)
            futures.append(future)
        
        # Gather results
        results = []
        while futures:
            ready, futures = self.pool.get_next_unordered()
            results.append(ready)
        
        return results
    
    def cleanup(self):
        """Clean up all workers."""
        for worker in self.workers:
            ray.get(worker.cleanup.remote())


class RayAsyncGPURunner:
    """
    Ray-based async GPU runner with streaming capabilities.
    """
    
    def __init__(self, num_gpus: int, max_concurrent: int = 4):
        self.num_gpus = num_gpus
        self.max_concurrent = max_concurrent
        
        # Initialize Ray if needed
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
    
    async def run_experiments_async(self, 
                                  experiments: List[Experiment],
                                  worker_fn: Callable) -> List[ExperimentResult]:
        """Run experiments asynchronously with GPU streaming."""
        # Create a queue for results
        result_queue = RayQueue(maxsize=len(experiments))
        
        # Create async worker function
        @ray.remote(num_gpus=1)
        def async_worker(exp: Experiment) -> ExperimentResult:
            return worker_fn(exp, device_id=0)  # Ray handles GPU assignment
        
        # Submit all tasks
        futures = [async_worker.remote(exp) for exp in experiments]
        
        # Gather results asynchronously
        results = []
        for future in futures:
            result = await asyncio.wrap_future(
                asyncio.ensure_future(ray.get(future))
            )
            results.append(result)
            
        return results


class RayAdvancedRunner(ExperimentRunnerBase):
    """
    Advanced Ray-based runner with all features from advanced_runners.py.
    """
    
    def __init__(self, lab_config: Optional[LabConfig] = None):
        super().__init__(lab_config)
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                num_cpus=lab_config.max_cpus if lab_config else None,
                num_gpus=lab_config.num_gpus if lab_config else None,
                ignore_reinit_error=True
            )
        
        # Create worker pool if using multiple GPUs
        self.worker_pool = None
        if lab_config and lab_config.num_gpus > 1:
            self.worker_pool = RayWorkerPool(
                num_workers=lab_config.num_gpus,
                worker_fn=ray_mixed_precision_worker,
                devices=list(range(lab_config.num_gpus))
            )
    
    def run_experiments_with_mixed_precision(self, 
                                           experiments: List[Experiment]) -> List[ExperimentResult]:
        """Run experiments with automatic mixed precision."""
        if self.worker_pool:
            return self.worker_pool.submit_experiments(experiments)
        
        # Single GPU or CPU execution
        @ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)
        def run_with_amp(exp):
            return ray_mixed_precision_worker(exp, 0)
        
        futures = [run_with_amp.remote(exp) for exp in experiments]
        return ray.get(futures)
    
    def run_with_profiling(self, 
                          experiments: List[Experiment],
                          profile_dir: str = "./profiles") -> List[ExperimentResult]:
        """Run experiments with comprehensive profiling."""
        @ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)
        def run_with_profile(exp):
            return ray_profiled_worker(exp, 0, profile_dir)
        
        futures = [run_with_profile.remote(exp) for exp in experiments]
        return ray.get(futures)
    
    def cleanup(self):
        """Clean up resources."""
        if self.worker_pool:
            self.worker_pool.cleanup()


def ray_mixed_precision_worker(experiment: Experiment, device_id: int) -> ExperimentResult:
    """
    Worker function with automatic mixed precision training.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_time = time.time()
    
    try:
        from structure_net.core.network_factory import create_standard_network
        from data_factory import create_dataset
        
        # Create model
        model = create_standard_network(
            architecture=experiment.parameters['architecture'],
            sparsity=experiment.parameters.get('sparsity', 0.02),
            device=device
        )
        
        # Setup mixed precision
        scaler = GradScaler() if device == 'cuda' else None
        
        # Create dataset
        dataset = create_dataset(
            experiment.parameters.get('dataset', 'cifar10'),
            batch_size=experiment.parameters.get('batch_size', 128)
        )
        
        # Training setup
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=experiment.parameters.get('learning_rate', 0.001)
        )
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop with mixed precision
        model.train()
        for epoch in range(experiment.parameters.get('epochs', 10)):
            for data, target in dataset['train_loader']:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                if scaler:
                    with autocast():
                        output = model(data)
                        loss = criterion(output, target)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
        
        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in dataset['test_loader']:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        
        return ExperimentResult(
            experiment_id=experiment.id,
            status=ExperimentStatus.COMPLETED,
            metrics={
                'accuracy': accuracy,
                'parameters': sum(p.numel() for p in model.parameters()),
                'training_time': time.time() - start_time,
                'used_amp': scaler is not None
            },
            duration=time.time() - start_time,
            error=None
        )
        
    except Exception as e:
        import traceback
        return ExperimentResult(
            experiment_id=experiment.id,
            status=ExperimentStatus.FAILED,
            metrics={},
            duration=time.time() - start_time,
            error=str(e) + "\n" + traceback.format_exc()
        )


def ray_profiled_worker(experiment: Experiment, device_id: int, profile_dir: str) -> ExperimentResult:
    """
    Worker with profiling enabled.
    """
    import torch.profiler as profiler
    import os
    
    profile_path = os.path.join(profile_dir, f"{experiment.id}_profile")
    
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=profiler.tensorboard_trace_handler(profile_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        result = ray_mixed_precision_worker(experiment, device_id)
        prof.step()
    
    # Add profiling info to metrics
    result.metrics['profile_path'] = profile_path
    return result


def migrate_existing_runner(runner_class, lab_config: Optional[LabConfig] = None):
    """
    Factory function to migrate existing runner classes to Ray.
    
    Usage:
        # Instead of:
        runner = AdvancedExperimentRunner(lab_config)
        
        # Use:
        runner = migrate_existing_runner(AdvancedExperimentRunner, lab_config)
    """
    # Check if it's already a Ray runner
    if hasattr(runner_class, '_ray_compatible'):
        return runner_class(lab_config)
    
    # Return Ray-based replacement
    return RayAdvancedRunner(lab_config)