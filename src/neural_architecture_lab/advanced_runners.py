"""
Advanced experiment runners for NAL with full feature support.

This module provides runners that support:
- GPU memory management and optimization
- Async GPU streaming
- Advanced multiprocessing with work queues
- Mixed precision training
- Comprehensive profiling
- Memory cleanup and optimization
"""

import os
import gc
import time
import traceback
import asyncio
import queue
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass

from .core import (
    Experiment, ExperimentResult, ExperimentStatus,
    LabConfig, ExperimentRunnerBase
)

# Import structure_net components
from structure_net.core.network_factory import create_standard_network
from structure_net.core.io_operations import save_model_seed, load_model_seed
from structure_net.core.network_analysis import get_network_stats
from structure_net.evolution.adaptive_learning_rates.unified_manager import AdaptiveLearningRateManager
from structure_net.evolution.components import create_standard_evolution_system, NetworkContext
from structure_net.evolution.metrics import CompleteMetricsSystem
from structure_net.evolution.residual_blocks import create_residual_network
from structure_net.profiling.factory import create_comprehensive_profiler
from structure_net.logging.standardized_logging import StandardizedLogger, LoggingConfig
from data_factory import create_dataset, get_dataset_config


class GPUMemoryManager:
    """Manages GPU memory allocation and cleanup."""
    
    def __init__(self, max_memory_fraction: float = 0.8):
        self.max_memory_fraction = max_memory_fraction
        self.allocated_memory = {}
        
    def get_available_memory(self, device_id: int) -> int:
        """Get available memory on GPU device."""
        if not torch.cuda.is_available():
            return 0
        
        torch.cuda.set_device(device_id)
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(device_id)
        cached_memory = torch.cuda.memory_reserved(device_id)
        
        available = total_memory - max(allocated_memory, cached_memory)
        max_usable = int(total_memory * self.max_memory_fraction)
        
        return min(available, max_usable)
    
    def optimize_batch_size(self, device_id: int, base_batch_size: int, model_size: int) -> int:
        """Optimize batch size based on available memory."""
        available_memory = self.get_available_memory(device_id)
        
        # Estimate memory per sample (rough heuristic)
        memory_per_sample = model_size * 4 + 3072 * 4  # Model params + CIFAR-10 input
        max_batch_size = available_memory // (memory_per_sample * 10)  # Safety factor
        
        return min(base_batch_size, max(1, max_batch_size))
    
    def cleanup_gpu_memory(self, device_id: int):
        """Clean up GPU memory."""
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
        gc.collect()


def advanced_experiment_worker(
    work_queue: mp.Queue,
    results_queue: mp.Queue,
    device_id: int,
    process_id: int,
    memory_manager: GPUMemoryManager,
    enable_profiling: bool = False,
    enable_mixed_precision: bool = True
):
    """Worker function for advanced parallel experiment execution."""
    torch.cuda.set_device(device_id)
    device = f'cuda:{device_id}'
    
    # Initialize profiler if enabled
    profiler = None
    if enable_profiling:
        profiler = create_comprehensive_profiler(
            output_dir=f"nal_profiling_{device_id}_{process_id}",
            enable_wandb=False
        )
    
    while True:
        try:
            item = work_queue.get(timeout=5)
            if item is None:  # Sentinel to stop
                break
            
            experiment, idx = item
            
            # Run experiment with full features
            result = run_advanced_experiment(
                experiment, device_id, memory_manager,
                profiler, enable_mixed_precision
            )
            
            results_queue.put((idx, result))
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Worker GPU {device_id} Process {process_id} error: {e}")
            traceback.print_exc()
            # Put error result
            error_result = ExperimentResult(
                experiment_id=experiment.id if 'experiment' in locals() else 'unknown',
                hypothesis_id='unknown',
                metrics={},
                primary_metric=0.0,
                model_architecture=[],
                model_parameters=0,
                training_time=0.0,
                error=str(e)
            )
            results_queue.put((idx if 'idx' in locals() else -1, error_result))


def run_advanced_experiment(
    experiment: Experiment,
    device_id: int,
    memory_manager: GPUMemoryManager,
    profiler: Optional[Any] = None,
    enable_mixed_precision: bool = True
) -> ExperimentResult:
    """
    Run a single experiment with advanced features.
    
    Includes:
    - Dynamic batch size optimization
    - Mixed precision training
    - Memory management
    - Profiling support
    """
    # Set device
    torch.cuda.set_device(device_id)
    device = f'cuda:{device_id}'
    
    # Set random seed
    if experiment.seed is not None:
        torch.manual_seed(experiment.seed)
        np.random.seed(experiment.seed)
        torch.cuda.manual_seed(experiment.seed)
    
    # Extract parameters
    params = experiment.parameters
    
    # Start timing
    start_time = time.time()
    
    # Start profiling session
    if profiler:
        profiler.start_session(f"experiment_{experiment.id}")
    
    try:
        # Create model
        if params.get('use_residual', False):
            model = create_residual_network(
                architecture=params['architecture'],
                sparsity=params.get('sparsity', 0.02),
                residual_positions=params.get('residual_positions', [2, 4]),
                device=device
            )
        else:
            model = create_standard_network(
                architecture=params['architecture'],
                sparsity=params.get('sparsity', 0.02),
                device=device
            )
        
        # Get model stats for memory optimization
        stats = get_network_stats(model)
        model_size = stats['total_parameters']
        
        # Optimize batch size based on available memory
        base_batch_size = params.get('batch_size', 128)
        optimized_batch_size = memory_manager.optimize_batch_size(
            device_id, base_batch_size, model_size
        )
        
        if optimized_batch_size != base_batch_size:
            print(f"Optimized batch size: {base_batch_size} -> {optimized_batch_size}")
        
        # Setup data loaders with optimized batch size
        dataset_name = params.get('dataset', 'cifar10').lower()
        dataset = create_dataset(
            dataset_name,
            batch_size=optimized_batch_size,
            subset_fraction=0.1 if params.get('quick_test', False) else None
        )
        train_loader = dataset['train_loader']
        test_loader = dataset['test_loader']
        
        # Validate architecture matches dataset
        dataset_config = get_dataset_config(dataset_name)
        if params['architecture'][0] != dataset_config.input_size:
            # Fix the architecture to match the dataset
            params['architecture'][0] = dataset_config.input_size
            print(f"Adjusted architecture input dimension to {dataset_config.input_size} for {dataset_name}")
            
            # Recreate model with corrected architecture
            model = create_standard_network(
                architecture=params['architecture'],
                sparsity=params.get('sparsity', 0.02),
                device=device
            )
            stats = get_network_stats(model)
            model_size = stats['total_parameters']
        
        # Setup optimizer with adaptive learning rate if specified
        lr_manager = None
        if params.get('lr_strategy'):
            lr_manager = AdaptiveLearningRateManager(
                network=model,
                base_lr=params.get('base_lr', 0.001),
                strategy=params['lr_strategy'],
                enable_extrema_phase=True,
                enable_layer_age_aware=True,
                enable_multi_scale=True,
                enable_unified_system=True
            )
            optimizer = lr_manager.create_adaptive_optimizer()
        else:
            # Default optimizer if no lr_strategy specified
            optimizer = optim.Adam(model.parameters(), lr=params.get('base_lr', 0.001))
        
        # Setup growth system if specified
        growth_system = None
        evolution_system = None
        if params.get('enable_growth', False):
            # Use the new composable evolution system
            evolution_system = create_standard_evolution_system()
            # Configure growth parameters
            for component in evolution_system.get_components():
                if hasattr(component, 'configure'):
                    component.configure({
                        'growth_interval': params.get('growth_interval', 10),
                        'neurons_per_growth': params.get('neurons_per_growth', 32),
                        'max_neurons': params.get('max_neurons', 2048)
                    })
        
        # Setup metrics system if specified
        metrics_system = None
        if params.get('enable_metrics', False):
            from ..structure_net.evolution.metrics.base import ThresholdConfig, MetricsConfig
            threshold_config = ThresholdConfig()
            metrics_config = MetricsConfig(
                track_layer_metrics=True,
                track_connection_metrics=True,
                enable_topological_analysis=True,
                enable_autocorrelation=True
            )
            metrics_system = CompleteMetricsSystem(model, threshold_config, metrics_config)
        
        # Setup loss function
        criterion = nn.CrossEntropyLoss()
        
        # Setup mixed precision training (disable if using adaptive learning rates)
        # AdaptiveOptimizerWrapper doesn't work well with GradScaler
        if lr_manager and enable_mixed_precision:
            print("Note: Disabling mixed precision due to adaptive learning rate usage")
            enable_mixed_precision = False
        scaler = GradScaler() if enable_mixed_precision else None
        
        # Training history
        training_history = []
        best_accuracy = 0.0
        growth_events = 0
        
        # Training loop with advanced features
        epochs = params.get('epochs', 50)
        memory_cleanup_frequency = params.get('memory_cleanup_frequency', 10)
        
        for epoch in range(epochs):
            # Memory cleanup
            if epoch > 0 and epoch % memory_cleanup_frequency == 0:
                memory_manager.cleanup_gpu_memory(device_id)
            
            # Train one epoch
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            # Async data loading with prefetching
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Non-blocking transfer to GPU
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # Flatten inputs if needed (for networks expecting flattened input)
                if len(params['architecture']) > 0 and params['architecture'][0] in [784, 3072]:
                    inputs = inputs.view(inputs.size(0), -1)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                if enable_mixed_precision:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    # Scaled backward pass
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_accuracy = correct / total
            
            # Validation with no_grad and inference mode
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            
            with torch.inference_mode():  # More efficient than no_grad
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    if len(params['architecture']) > 0 and params['architecture'][0] in [784, 3072]:
                        inputs = inputs.view(inputs.size(0), -1)
                    
                    if enable_mixed_precision:
                        with autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            test_accuracy = correct / total
            best_accuracy = max(best_accuracy, test_accuracy)
            
            # Update learning rate
            if lr_manager:
                context = {
                    'loss': train_loss / len(train_loader),
                    'accuracy': train_accuracy,
                    'val_loss': test_loss / len(test_loader),
                    'val_accuracy': test_accuracy
                }
                lr_manager.update_learning_rates(optimizer, epoch, **context)
            
            # Check for growth using new composable system
            if evolution_system and epoch > 0 and epoch % params.get('growth_interval', 10) == 0:
                # Create network context for evolution
                device_obj = torch.device(device)
                ctx = NetworkContext(
                    network=model,
                    data_loader=train_loader,
                    device=device_obj
                )
                # Add current metrics to context
                ctx.current_metrics = {
                    'accuracy': test_accuracy,
                    'loss': test_loss / len(test_loader)
                }
                
                # Run one evolution iteration
                old_params = sum(p.numel() for p in model.parameters())
                evolved_ctx = evolution_system.evolve_network(ctx, num_iterations=1)
                
                # Check if growth occurred
                if evolved_ctx.network is not ctx.network:
                    model = evolved_ctx.network
                    new_params = sum(p.numel() for p in model.parameters())
                    growth_events += 1
                    
                    # Recreate optimizer for new model
                    if lr_manager:
                        optimizer = lr_manager.create_adaptive_optimizer()
                    else:
                        optimizer = optim.Adam(model.parameters(), lr=params.get('base_lr', 0.001))
                    
                    # Update batch size after growth
                    optimized_batch_size = memory_manager.optimize_batch_size(
                        device_id, base_batch_size, new_params
                    )
                    
                    # Recreate data loaders if batch size changed
                    if optimized_batch_size != train_loader.batch_size:
                        train_loader = DataLoader(
                            train_dataset,
                            batch_size=optimized_batch_size,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True,
                            persistent_workers=True
                        )
                        test_loader = DataLoader(
                            test_dataset,
                            batch_size=optimized_batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            persistent_workers=True
                        )
            
            # Collect metrics
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_loss / len(train_loader),
                'train_accuracy': train_accuracy,
                'test_loss': test_loss / len(test_loader),
                'test_accuracy': test_accuracy,
                'lr': lr_manager.get_current_rates_summary()['base_lr'] if lr_manager else optimizer.param_groups[0]['lr']
            }
            
            if metrics_system:
                with torch.no_grad():
                    metrics_data = metrics_system.analyze(next(iter(test_loader))[0][:1].to(device))
                    epoch_metrics.update(metrics_data)
            
            training_history.append(epoch_metrics)
        
        # Final evaluation
        model.eval()
        final_stats = get_network_stats(model)
        
        # Calculate primary metric
        primary_metric = calculate_primary_metric(
            params.get('primary_metric_type', 'accuracy'),
            best_accuracy,
            final_stats,
            training_history
        )
        
        # Compile final metrics
        metrics = {
            'accuracy': best_accuracy,
            'final_train_accuracy': training_history[-1]['train_accuracy'],
            'convergence_epochs': len(training_history),
            'growth_events': growth_events,
            'final_parameters': final_stats['total_parameters'],
            'final_layers': len(final_stats['layers']),
            'sparsity': final_stats.get('overall_sparsity', 0.0),
            'training_time': time.time() - start_time,
            'optimized_batch_size': optimized_batch_size,
            'memory_used': torch.cuda.max_memory_allocated(device_id) / 1e9  # GB
        }
        
        # Save model if specified
        model_checkpoint = None
        if params.get('save_model', False):
            model_path = f"nal_models/{experiment.id}_best.pt"
            os.makedirs("nal_models", exist_ok=True)
            save_model_seed(
                model, params['architecture'],
                accuracy=best_accuracy,
                sparsity=params.get('sparsity', 0.02),
                filepath=model_path
            )
            model_checkpoint = model_path
        
        result = ExperimentResult(
            experiment_id=experiment.id,
            hypothesis_id=experiment.hypothesis_id,
            metrics=metrics,
            primary_metric=primary_metric,
            model_architecture=params['architecture'],
            model_parameters=final_stats['total_parameters'],
            training_time=time.time() - start_time,
            training_history=training_history,
            model_checkpoint=model_checkpoint
        )
        
        # Log the result
        logger = StandardizedLogger()
        logger.log_experiment_result(result)
        
        return result
        
    except Exception as e:
        result = ExperimentResult(
            experiment_id=experiment.id,
            hypothesis_id=experiment.hypothesis_id,
            metrics={},
            primary_metric=0.0,
            model_architecture=params.get('architecture', []),
            model_parameters=0,
            training_time=time.time() - start_time,
            error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        )
        
        # Log the error result
        logger = StandardizedLogger()
        logger.log_experiment_result(result)
        
        return result
    
    finally:
        # Clean up
        if profiler:
            profiler.end_session()
        
        # Clean up GPU memory
        memory_manager.cleanup_gpu_memory(device_id)


def calculate_primary_metric(
    metric_type: str,
    best_accuracy: float,
    final_stats: Dict[str, Any],
    training_history: List[Dict[str, Any]]
) -> float:
    """Calculate primary metric based on type."""
    if metric_type == 'accuracy':
        return best_accuracy
    elif metric_type == 'efficiency':
        # Accuracy per parameter (in millions)
        return best_accuracy / (final_stats['total_parameters'] / 1e6)
    elif metric_type == 'convergence_speed':
        # Epochs to reach 90% of best accuracy
        target_acc = best_accuracy * 0.9
        convergence_epoch = next(
            (i for i, h in enumerate(training_history) if h['test_accuracy'] >= target_acc),
            len(training_history)
        )
        return 1.0 / (convergence_epoch + 1)  # Higher is better
    elif metric_type == 'fitness':
        # Tournament-style fitness combining multiple factors
        efficiency = best_accuracy / (final_stats['total_parameters'] / 1e6)
        return best_accuracy * 0.7 + efficiency * 0.3
    else:
        return best_accuracy


class AdvancedExperimentRunner(ExperimentRunnerBase):
    """Advanced experiment runner with full feature support."""
    
    def __init__(self, config: LabConfig, logger: StandardizedLogger):
        self.config = config
        self.logger = logger
        self.device_ids = config.device_ids
        self.max_parallel = config.max_parallel_experiments
        self.memory_managers = {
            device_id: GPUMemoryManager(config.max_memory_per_experiment)
            for device_id in self.device_ids
        }

    
    async def run_experiment(self, experiment: Experiment) -> ExperimentResult:
        """Run a single experiment with advanced features."""
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = time.time()
        
        # Assign device
        device_id = experiment.device_id
        if device_id is None:
            device_id = self.device_ids[hash(experiment.id) % len(self.device_ids)]
        
        # Get memory manager for device
        memory_manager = self.memory_managers[device_id]
        
        # Run experiment in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  # Use default thread pool
            run_advanced_experiment,
            experiment, device_id, memory_manager, None, True
        )
        
        # Update experiment status
        experiment.status = ExperimentStatus.COMPLETED if result.error is None else ExperimentStatus.FAILED
        experiment.completed_at = time.time()
        experiment.result = result
        
        # Log result using proper schema
        try:
            from ..structure_net.logging.schemas import (
                TrainingExperiment, ExperimentConfig, NetworkArchitecture,
                PerformanceMetrics, TrainingEpoch
            )
            
            # Create experiment config
            exp_config = ExperimentConfig(
                dataset="cifar10",
                batch_size=experiment.parameters.get('batch_size', 128),
                learning_rate=experiment.parameters.get('base_lr', 0.001),
                max_epochs=experiment.parameters.get('epochs', 50),
                device=f"cuda:{device_id}",
                random_seed=experiment.seed
            )
            
            # Create network architecture
            arch = NetworkArchitecture(
                layers=result.model_architecture,
                total_parameters=result.model_parameters,
                total_connections=result.model_parameters,  # Approximation
                sparsity=result.metrics.get('sparsity', 0.02),
                depth=len(result.model_architecture)
            )
            
            # Create performance metrics
            perf = PerformanceMetrics(
                accuracy=result.metrics.get('accuracy', 0.0),
                loss=result.metrics.get('final_loss', 1.0)
            )
            
            # Create training experiment
            training_exp = TrainingExperiment(
                experiment_id=experiment.id,
                experiment_type="training_experiment",
                config=exp_config,
                architecture=arch,
                training_history=[],  # Could populate from result.training_history
                final_performance=perf,
                total_epochs=result.metrics.get('convergence_epochs', 0)
            )
            
            # Log the experiment
            self.logger.log_experiment_result(training_exp)
            
        except Exception as e:
            # Logging is optional, don't fail the experiment
            print(f"Warning: Failed to log experiment: {e}")
        
        return result
    
    async def run_experiments(self, experiments: List[Experiment]) -> List[ExperimentResult]:
        """Run multiple experiments with advanced parallel processing."""
        results = []
        
        # Create work queue
        work_queue = mp.Queue()
        for i, exp in enumerate(experiments):
            work_queue.put((exp, i))
        
        # Add sentinels
        num_workers = min(self.max_parallel, len(self.device_ids) * 2)
        for _ in range(num_workers):
            work_queue.put(None)
        
        # Results queue
        results_queue = mp.Queue()
        
        # Start workers
        processes = []
        for i in range(num_workers):
            device_id = self.device_ids[i % len(self.device_ids)]
            memory_manager = self.memory_managers[device_id]
            
            p = mp.Process(
                target=advanced_experiment_worker,
                args=(work_queue, results_queue, device_id, i,
                      memory_manager, self.config.verbose, True)
            )
            p.start()
            processes.append(p)
        
        # Collect results
        collected_results = {}
        expected_count = len(experiments)
        
        while len(collected_results) < expected_count:
            try:
                idx, result = results_queue.get(timeout=30)
                collected_results[idx] = result
                
                if self.config.verbose:
                    print(f"Progress: {len(collected_results)}/{expected_count}")
                
            except queue.Empty:
                print("Warning: Timeout waiting for results")
                break
        
        # Wait for processes
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        
        # Sort results
        results = [collected_results.get(i, self._create_error_result(experiments[i]))
                  for i in range(len(experiments))]
        
        return results
    
    def _create_error_result(self, experiment: Experiment) -> ExperimentResult:
        """Create error result for failed experiment."""
        return ExperimentResult(
            experiment_id=experiment.id,
            hypothesis_id=experiment.hypothesis_id,
            metrics={},
            primary_metric=0.0,
            model_architecture=[],
            model_parameters=0,
            training_time=0.0,
            error="Experiment timed out or failed"
        )