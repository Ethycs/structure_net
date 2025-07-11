"""
Worker utilities for NAL experiments.

Provides utilities for experiment workers to register themselves and update their status.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from src.structure_net.logging.standardized_logging import StandardizedLogger, LoggingConfig


class WorkerRegistry:
    """Registry for experiment workers to track their status."""
    
    def __init__(self, experiment_id: str, hypothesis_id: str, chromadb_path: str = "data/chroma_db"):
        """Initialize worker registry."""
        self.experiment_id = experiment_id
        self.hypothesis_id = hypothesis_id
        self.pid = os.getpid()
        self.start_time = time.time()
        
        # Initialize logger with ChromaDB only (no file queue needed)
        logging_config = LoggingConfig(
            enable_chromadb=True,
            chromadb_path=chromadb_path,
            enable_wandb=False,
            auto_upload=False
        )
        self.logger = StandardizedLogger(logging_config)
        
    def register_start(self, epochs: int, batch_size: int, 
                      architecture: list, device: str, **kwargs) -> None:
        """Register experiment start with detailed metadata."""
        # Estimate duration based on architecture and epochs
        n_params = self._estimate_parameters(architecture)
        seconds_per_epoch = self._estimate_seconds_per_epoch(n_params, batch_size, device)
        estimated_duration = epochs * seconds_per_epoch
        
        # Register with all metadata
        self.logger.register_experiment_start(
            self.experiment_id,
            self.hypothesis_id,
            estimated_duration=estimated_duration,
            pid=self.pid,
            architecture=str(architecture),
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            n_parameters=n_params,
            **kwargs
        )
        
        # Print status
        completion_time = datetime.now() + timedelta(seconds=estimated_duration)
        print(f"📝 [{self.experiment_id.split('_')[-1]}] PID: {self.pid} | "
              f"Est. completion: {completion_time.strftime('%H:%M:%S')} "
              f"({estimated_duration:.0f}s)", flush=True)
    
    def update_progress(self, current_epoch: int, total_epochs: int, 
                       current_accuracy: float = None) -> None:
        """Update experiment progress."""
        elapsed = time.time() - self.start_time
        progress = current_epoch / total_epochs
        
        # Estimate remaining time based on actual speed
        if progress > 0:
            total_estimated = elapsed / progress
            remaining = total_estimated - elapsed
            new_completion = datetime.now() + timedelta(seconds=remaining)
            
            # Update in ChromaDB
            self.logger.update_experiment_status(
                self.experiment_id,
                'running',
                progress=progress * 100,
                current_epoch=current_epoch,
                elapsed_time=elapsed,
                estimated_completion=new_completion.isoformat(),
                current_accuracy=current_accuracy
            )
    
    def register_completion(self, metrics: Dict[str, Any]) -> None:
        """Register experiment completion."""
        duration = time.time() - self.start_time
        
        # Update final status
        self.logger.update_experiment_status(
            self.experiment_id,
            'completed',
            training_time=duration,
            completed_at=datetime.now().isoformat(),
            **metrics
        )
    
    def register_failure(self, error: str) -> None:
        """Register experiment failure."""
        duration = time.time() - self.start_time
        
        # Update failure status
        self.logger.update_experiment_status(
            self.experiment_id,
            'failed',
            training_time=duration,
            completed_at=datetime.now().isoformat(),
            error=error[:500]  # Limit error message length
        )
    
    def _estimate_parameters(self, architecture: list) -> int:
        """Estimate number of parameters from architecture."""
        if not architecture or len(architecture) < 2:
            return 0
        
        n_params = 0
        for i in range(len(architecture) - 1):
            n_params += architecture[i] * architecture[i+1]  # Weights
            n_params += architecture[i+1]  # Biases
        return n_params
    
    def _estimate_seconds_per_epoch(self, n_params: int, batch_size: int, device: str) -> float:
        """Estimate seconds per epoch based on model size and hardware."""
        # Base estimates (these can be tuned based on actual measurements)
        if 'cuda' in device:
            # GPU estimates
            base_time = 1.0  # Base overhead
            param_factor = n_params / 1e6 * 0.5  # 0.5s per million params
            batch_factor = 128 / batch_size * 0.2  # Smaller batches are less efficient
        else:
            # CPU estimates
            base_time = 2.0
            param_factor = n_params / 1e6 * 5.0  # Much slower on CPU
            batch_factor = 128 / batch_size * 1.0
        
        return base_time + param_factor + batch_factor


class WorkerWrapper:
    """Picklable wrapper for test functions with worker registration."""
    
    def __init__(self, test_function: Callable):
        self.test_function = test_function
    
    def __call__(self, experiment, device_id):
        # Extract key parameters
        config = experiment.parameters
        if 'params' in config and isinstance(config['params'], dict):
            actual_config = {**config}
            actual_config.update(config['params'])
            config = actual_config
        
        # Create worker registry
        registry = WorkerRegistry(
            experiment.id,
            experiment.hypothesis_id
        )
        
        # Register start
        registry.register_start(
            epochs=config.get('epochs', 10),
            batch_size=config.get('batch_size', 128),
            architecture=config.get('architecture', []),
            device=f'cuda:{device_id}' if device_id >= 0 else 'cpu',
            lr_strategy=config.get('lr_strategy', 'default'),
            sparsity=config.get('sparsity', 0.0)
        )
        
        try:
            # Run the actual test function
            result = self.test_function(experiment, device_id)
            
            # Register completion
            if hasattr(result, 'metrics') and result.metrics:
                registry.register_completion(result.metrics)
            
            return result
            
        except Exception as e:
            # Register failure
            registry.register_failure(str(e))
            raise


def create_worker_wrapper(test_function: Callable) -> Callable:
    """
    Create a wrapped version of a test function that includes worker registration.
    
    Args:
        test_function: The original test function
        
    Returns:
        Wrapped function with automatic worker registration
    """
    return WorkerWrapper(test_function)