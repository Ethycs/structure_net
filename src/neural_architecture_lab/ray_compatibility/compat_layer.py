"""
Ray compatibility layer for NAL - maintains existing API while using Ray backend.

This module provides drop-in replacements for the existing experiment runners
and workers, using Ray for distributed execution while preserving the original
interfaces.
"""

import ray
from ray import train, tune
from ray.train import Checkpoint, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import traceback
from typing import List, Dict, Any, Optional, Callable, Union
import numpy as np
from dataclasses import asdict

# Import existing NAL types
from ..core import (
    Experiment, ExperimentResult, ExperimentStatus,
    LabConfig, ExperimentRunnerBase
)


class RayExperimentRunner(ExperimentRunnerBase):
    """
    Drop-in replacement for existing experiment runners using Ray.
    
    Maintains the same interface as the original runners but uses Ray
    for distributed execution, resource management, and fault tolerance.
    """
    
    def __init__(self, lab_config: Optional[LabConfig] = None):
        super().__init__(lab_config)
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(
                num_cpus=lab_config.max_cpus if lab_config else None,
                num_gpus=lab_config.num_gpus if lab_config else None,
                ignore_reinit_error=True
            )
    
    def run_experiments(self, 
                       experiments: List[Experiment], 
                       runner_fn: Optional[Callable] = None) -> List[ExperimentResult]:
        """
        Run experiments using Ray with compatibility for existing runner functions.
        """
        if runner_fn is None:
            runner_fn = ray_experiment_worker
        
        # Convert experiments to Ray tasks
        futures = []
        for exp in experiments:
            # Determine GPU allocation
            num_gpus = 1 if self.lab_config and self.lab_config.num_gpus > 0 else 0
            
            # Create Ray remote function that wraps the existing runner
            ray_runner = ray.remote(num_gpus=num_gpus)(runner_fn)
            
            # Submit task
            future = ray_runner.remote(exp, device_id=0)  # Ray handles device assignment
            futures.append(future)
        
        # Gather results
        results = ray.get(futures)
        return results
    
    def run_experiments_async(self, 
                             experiments: List[Experiment],
                             callback: Optional[Callable] = None) -> None:
        """
        Run experiments asynchronously with optional callbacks.
        """
        # This can be implemented with Ray's async patterns
        # For now, delegate to synchronous version
        results = self.run_experiments(experiments)
        if callback:
            for result in results:
                callback(result)


class RayGPUMemoryManager:
    """
    Compatibility wrapper for GPU memory management using Ray's resource system.
    """
    
    def __init__(self, max_memory_fraction: float = 0.8):
        self.max_memory_fraction = max_memory_fraction
    
    def get_available_memory(self, device_id: int) -> int:
        """Ray automatically manages GPU memory, return a reasonable default."""
        if not torch.cuda.is_available():
            return 0
        
        # Ray handles memory management, return total memory as "available"
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        return int(total_memory * self.max_memory_fraction)
    
    def optimize_batch_size(self, device_id: int, base_batch_size: int, model_size: int) -> int:
        """Let Ray handle optimization, return base batch size."""
        return base_batch_size


def ray_experiment_worker(experiment: Experiment, device_id: int = 0) -> ExperimentResult:
    """
    Ray-compatible worker function that maintains compatibility with existing workers.
    
    This function can wrap existing worker functions or provide a new implementation.
    """
    # Set device (Ray handles GPU assignment automatically)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set random seed
    if experiment.seed is not None:
        torch.manual_seed(experiment.seed)
        np.random.seed(experiment.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(experiment.seed)
    
    start_time = time.time()
    params = experiment.parameters
    
    try:
        # Import structure_net components (deferred to avoid circular imports)
        from structure_net.core.network_factory import create_standard_network
        from data_factory import create_dataset
        
        # Create model
        model = create_standard_network(
            architecture=params['architecture'],
            sparsity=params.get('sparsity', 0.02),
            device=device
        )
        
        # Create dataset
        dataset = create_dataset(
            params.get('dataset', 'cifar10'),
            batch_size=params.get('batch_size', 128)
        )
        train_loader = dataset['train_loader']
        test_loader = dataset['test_loader']
        
        # Training setup
        optimizer = optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001))
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        epochs = params.get('epochs', 10)
        for epoch in range(epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # Handle potential shape mismatches
                if data.dim() > 2 and hasattr(model, 'layers') and model.layers:
                    if isinstance(model.layers[0], nn.Linear):
                        data = data.view(data.size(0), -1)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Report to Ray periodically
                if batch_idx % 100 == 0:
                    session.report({"loss": loss.item(), "epoch": epoch})
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                if data.dim() > 2 and hasattr(model, 'layers') and model.layers:
                    if isinstance(model.layers[0], nn.Linear):
                        data = data.view(data.size(0), -1)
                
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total if total > 0 else 0
        
        # Prepare result
        metrics = {
            'accuracy': accuracy,
            'parameters': sum(p.numel() for p in model.parameters()),
            'training_time': time.time() - start_time
        }
        
        return ExperimentResult(
            experiment_id=experiment.id,
            status=ExperimentStatus.COMPLETED,
            metrics=metrics,
            duration=time.time() - start_time,
            error=None
        )
        
    except Exception as e:
        return ExperimentResult(
            experiment_id=experiment.id,
            status=ExperimentStatus.FAILED,
            metrics={},
            duration=time.time() - start_time,
            error=str(e) + "\n" + traceback.format_exc()
        )


class RayTuneExperimentRunner(RayExperimentRunner):
    """
    Advanced runner using Ray Tune for hyperparameter optimization.
    """
    
    def run_hyperparameter_search(self, 
                                 base_experiment: Experiment,
                                 param_space: Dict[str, Any],
                                 num_samples: int = 10) -> List[ExperimentResult]:
        """
        Run hyperparameter search using Ray Tune.
        """
        def tune_objective(config):
            # Create experiment with tune config
            exp = Experiment(
                id=f"{base_experiment.id}_tune_{time.time()}",
                name=f"{base_experiment.name}_tune",
                parameters={**base_experiment.parameters, **config},
                seed=base_experiment.seed
            )
            
            # Run experiment
            result = ray_experiment_worker(exp, device_id=0)
            
            # Report metrics to Ray Tune
            session.report({
                "accuracy": result.metrics.get('accuracy', 0),
                "loss": result.metrics.get('loss', float('inf'))
            })
        
        # Configure Tune
        tune_config = tune.TuneConfig(
            metric="accuracy",
            mode="max",
            num_samples=num_samples,
        )
        
        # Run tuning
        analysis = tune.run(
            tune_objective,
            config=param_space,
            tune_config=tune_config,
            resources_per_trial={"gpu": 1 if torch.cuda.is_available() else 0}
        )
        
        # Convert results
        results = []
        for trial in analysis.trials:
            result = ExperimentResult(
                experiment_id=f"{base_experiment.id}_tune_{trial.trial_id}",
                status=ExperimentStatus.COMPLETED if trial.status == "TERMINATED" else ExperimentStatus.FAILED,
                metrics=trial.last_result,
                duration=trial.last_update_time - trial.start_time if trial.start_time else 0,
                error=trial.error_msg
            )
            results.append(result)
        
        return results


# Compatibility functions to replace imports in existing code
def create_experiment_runner(lab_config: Optional[LabConfig] = None, 
                           use_tune: bool = False) -> ExperimentRunnerBase:
    """
    Factory function to create appropriate runner.
    """
    if use_tune:
        return RayTuneExperimentRunner(lab_config)
    return RayExperimentRunner(lab_config)


# Worker function wrappers for compatibility
def run_structure_net_experiment(experiment: Experiment, device_id: int = 0) -> ExperimentResult:
    """Compatibility wrapper for existing run_structure_net_experiment calls."""
    return ray_experiment_worker(experiment, device_id)


def run_seed_search_task(experiment: Experiment, device_id: int = 0) -> ExperimentResult:
    """Compatibility wrapper for seed search tasks."""
    # Can import and wrap the original seed search worker if needed
    return ray_experiment_worker(experiment, device_id)


def run_tournament_task(experiment: Experiment, device_id: int = 0) -> ExperimentResult:
    """Compatibility wrapper for tournament tasks."""
    # Can import and wrap the original tournament worker if needed
    return ray_experiment_worker(experiment, device_id)