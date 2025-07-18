"""
Experiment runners for the Neural Architecture Lab.
"""

import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os

from .core import (
    Experiment, ExperimentResult, ExperimentStatus,
    LabConfig, ExperimentRunnerBase
)
from .resource_monitor import get_auto_balancer, ResourceLimits

# Import structure_net components
from structure_net.core.network_factory import create_standard_network
from structure_net.core.io_operations import save_model_seed, load_model_seed
from structure_net.core.network_analysis import get_network_stats
from structure_net.evolution.adaptive_learning_rates.unified_manager import AdaptiveLearningRateManager
from structure_net.evolution.components import create_standard_evolution_system
from structure_net.evolution.metrics.integrated_system import CompleteMetricsSystem
from structure_net.evolution.residual_blocks import ResidualGrowthStrategy
from structure_net.profiling.factory import create_comprehensive_profiler
from structure_net.logging.standardized_logging import StandardizedLogger, LoggingConfig

# Import data factory components
from data_factory import create_dataset


def run_structure_net_experiment(experiment: Experiment, device_id: int = 0) -> ExperimentResult:
    """
    Run a single structure_net experiment with full system integration.
    
    This function can be pickled and run in separate processes.
    """
    # Set device
    if torch.cuda.is_available() and device_id >= 0:
        torch.cuda.set_device(device_id)
        device = f'cuda:{device_id}'
    else:
        device = 'cpu'
    
    # Set random seed
    if experiment.seed is not None:
        torch.manual_seed(experiment.seed)
        np.random.seed(experiment.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(experiment.seed)
    
    # Extract parameters
    params = experiment.parameters
    
    # Start timing
    start_time = time.time()
    
    try:
        # Create model based on experiment type
        if params.get('use_residual', False):
            from src.structure_net.evolution.residual_blocks import create_residual_network
            model = create_residual_network(
                architecture=params['architecture'],
                skip_frequency=params.get('skip_frequency', 2)
            ).to(device)
        else:
            model = create_standard_network(
                architecture=params['architecture'],
                sparsity=params.get('sparsity', 0.02),
                device=device
            )
        
        # Get initial stats
        stats = get_network_stats(model)
        
        # Setup data loaders using data factory
        dataset_name = params.get('dataset', 'cifar10')
        subset_fraction = 0.1 if params.get('quick_test', False) else None
        
        dataset_dict = create_dataset(
            dataset_name=dataset_name,
            batch_size=params.get('batch_size', 128),
            subset_fraction=subset_fraction,
            num_workers=2,
            pin_memory=True,
            experiment_id=experiment.id
        )
        
        train_loader = dataset_dict['train_loader']
        test_loader = dataset_dict['test_loader']
        dataset_config = dataset_dict['config']
        
        # Setup optimizer
        optimizer = optim.Adam(model.parameters(), lr=params.get('base_lr', 0.001))
        
        # Setup adaptive learning rate if specified
        lr_manager = None
        if params.get('lr_strategy'):
            lr_manager = AdaptiveLearningRateManager(
                network=model,
                base_lr=params.get('base_lr', 0.001),
                strategy=params['lr_strategy']
            )
            optimizer = lr_manager.create_adaptive_optimizer()
        
        # Setup growth system if specified
        growth_system = None
        if params.get('enable_growth', False):
            from structure_net.evolution.integrated_growth_system_v2 import IntegratedGrowthSystem
            growth_system = IntegratedGrowthSystem(
                network=model,
                threshold_config=None,
                metrics_config=None
            )
        
        # Setup metrics system if specified
        metrics_system = None
        if params.get('enable_metrics', False):
            metrics_system = CompleteMetricsSystem()
        
        # Setup loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        training_history = []
        best_accuracy = 0.0
        growth_events = 0
        
        # Training loop
        epochs = params.get('epochs', 50)
        for epoch in range(epochs):
            # Train one epoch
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Flatten inputs if needed (for fully connected architectures)
                if len(params['architecture']) > 0 and params['architecture'][0] == dataset_config.input_size:
                    inputs = inputs.view(inputs.size(0), -1)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_accuracy = correct / total
            
            # Validation
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    if len(params['architecture']) > 0 and params['architecture'][0] == dataset_config.input_size:
                        inputs = inputs.view(inputs.size(0), -1)
                    
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
                    'epoch': epoch,
                    'loss': train_loss / len(train_loader),
                    'accuracy': train_accuracy
                }
                lr_manager.update(context)
            
            # Check for growth
            if growth_system and epoch > 0 and epoch % params.get('growth_interval', 10) == 0:
                should_grow = growth_system.should_grow(
                    model, {'accuracy': test_accuracy, 'loss': test_loss / len(test_loader)}
                )
                if should_grow:
                    old_params = sum(p.numel() for p in model.parameters())
                    model = growth_system.grow_network(model, optimizer)
                    new_params = sum(p.numel() for p in model.parameters())
                    growth_events += 1
                    
                    print(f"Growth event {growth_events}: {old_params} -> {new_params} parameters")
            
            # Collect metrics
            if metrics_system:
                metrics_data = metrics_system.analyze(model)
            
            # Record history
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss / len(train_loader),
                'train_accuracy': train_accuracy,
                'test_loss': test_loss / len(test_loader),
                'test_accuracy': test_accuracy,
                'lr': optimizer.param_groups[0]['lr'] if not lr_manager else lr_manager.get_current_lr()
            })
        
        # Final evaluation
        model.eval()
        final_stats = get_network_stats(model)
        
        # Calculate primary metric based on hypothesis
        if params.get('primary_metric_type') == 'accuracy':
            primary_metric = best_accuracy
        elif params.get('primary_metric_type') == 'efficiency':
            # Accuracy per parameter (in millions)
            primary_metric = best_accuracy / (final_stats['total_parameters'] / 1e6)
        elif params.get('primary_metric_type') == 'convergence_speed':
            # Epochs to reach 90% of best accuracy
            target_acc = best_accuracy * 0.9
            convergence_epoch = next(
                (i for i, h in enumerate(training_history) if h['test_accuracy'] >= target_acc),
                epochs
            )
            primary_metric = 1.0 / (convergence_epoch + 1)  # Higher is better
        else:
            primary_metric = best_accuracy
        
        # Compile metrics
        metrics = {
            'accuracy': best_accuracy,
            'final_train_accuracy': training_history[-1]['train_accuracy'],
            'convergence_epochs': len(training_history),
            'growth_events': growth_events,
            'final_parameters': final_stats['total_parameters'],
            'final_layers': final_stats['num_layers'],
            'sparsity': final_stats.get('sparsity', 0.0),
            'training_time': time.time() - start_time
        }
        
        # Save model if specified
        model_checkpoint = None
        if params.get('save_model', False):
            model_path = f"nal_models/{experiment.id}_best.pt"
            os.makedirs("nal_models", exist_ok=True)
            save_model_seed(
                model, model_path,
                accuracy=best_accuracy,
                architecture=params['architecture'],
                sparsity=params.get('sparsity', 0.02)
            )
            model_checkpoint = model_path
        
        return ExperimentResult(
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
        
    except Exception as e:
        # Return error result
        return ExperimentResult(
            experiment_id=experiment.id,
            hypothesis_id=experiment.hypothesis_id,
            metrics={},
            primary_metric=0.0,
            model_architecture=params.get('architecture', []),
            model_parameters=0,
            training_time=time.time() - start_time,
            error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        )


class AsyncExperimentRunner(ExperimentRunnerBase):
    """Asynchronous experiment runner using multiprocessing with auto-balancing."""
    
    def __init__(self, config: LabConfig):
        self.config = config
        self.device_ids = config.device_ids
        self.max_parallel = config.max_parallel_experiments
        
        # Initialize auto-balancer if enabled
        self.auto_balance = getattr(config, 'auto_balance', True)
        if self.auto_balance:
            resource_limits = ResourceLimits(
                target_cpu_percent=getattr(config, 'target_cpu_percent', 75.0),
                max_cpu_percent=getattr(config, 'max_cpu_percent', 90.0),
                target_gpu_percent=getattr(config, 'target_gpu_percent', 85.0),
                max_gpu_percent=getattr(config, 'max_gpu_percent', 95.0),
                target_memory_percent=getattr(config, 'target_memory_percent', 80.0),
                max_memory_percent=getattr(config, 'max_memory_percent', 90.0)
            )
            self.balancer = get_auto_balancer(resource_limits)
            
            # Get optimal initial settings
            initial_settings = self.balancer.get_optimal_initial_settings()
            self.max_parallel = initial_settings['parallel_experiments']
            self.current_batch_size = initial_settings['batch_size']
            self.current_workers = initial_settings['num_workers']
            
            if config.verbose:
                print(f"🤖 Auto-balancer initialized:")
                print(f"   Parallel experiments: {self.max_parallel}")
                print(f"   Initial batch size: {self.current_batch_size}")
                print(f"   Data workers: {self.current_workers}")
        else:
            self.balancer = None
            self.current_batch_size = 128
            self.current_workers = 2
    
    async def run_experiment(self, experiment: Experiment, test_function: Callable) -> ExperimentResult:
        """Run a single experiment asynchronously."""
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = time.time()
        
        # Register experiment start in ChromaDB if logger available
        if hasattr(self, 'logger') and self.logger:
            estimated_duration = experiment.parameters.get('epochs', 10) * 5.0  # Rough estimate
            self.logger.register_experiment_start(
                experiment.id,
                experiment.hypothesis_id,
                estimated_duration=estimated_duration,
                architecture=str(experiment.parameters.get('architecture', [])),
                device_id=experiment.device_id
            )
        
        device_id = experiment.device_id or self.device_ids[hash(experiment.id) % len(self.device_ids)]
        
        # Print one-line experiment status
        exp_id = experiment.id.split('_')[-1]  # Get just the experiment number
        arch = experiment.parameters.get('architecture', [])
        n_layers = len(arch) - 1 if isinstance(arch, list) else 0
        epochs = experiment.parameters.get('epochs', 10)
        lr_strategy = experiment.parameters.get('lr_strategy', 'default')
        device_str = f'cuda:{device_id}' if device_id >= 0 else 'cpu'
        
        print(f"🚀 [{exp_id}] Starting on {device_str} | "
              f"Arch: {n_layers}L | "
              f"Strategy: {lr_strategy} | "
              f"Epochs: {epochs}", flush=True)
        
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor,
                test_function,
                experiment,
                device_id
            )
        
        experiment.status = ExperimentStatus.COMPLETED if result.error is None else ExperimentStatus.FAILED
        experiment.completed_at = time.time()
        experiment.result = result
        
        # Print completion status
        duration = experiment.completed_at - experiment.started_at
        if result.error is None:
            accuracy = result.metrics.get('accuracy', 0.0) if hasattr(result, 'metrics') else 0.0
            primary_metric = result.primary_metric if hasattr(result, 'primary_metric') else 0.0
            print(f"✅ [{exp_id}] Completed in {duration:.1f}s | "
                  f"Accuracy: {accuracy:.2%} | "
                  f"Primary metric: {primary_metric:.3f}", flush=True)
        else:
            error_msg = str(result.error).split('\n')[0][:50] if hasattr(result, 'error') else 'Unknown error'
            print(f"❌ [{exp_id}] Failed after {duration:.1f}s | Error: {error_msg}...", flush=True)
        
        # Update ChromaDB status if available
        if hasattr(self, 'logger') and self.logger and result.error is None:
            self.logger.update_experiment_status(
                experiment.id,
                'completed',
                accuracy=accuracy,
                primary_metric=primary_metric,
                training_time=duration
            )
        
        return result
    
    async def run_experiments(self, experiments: List[Experiment], test_function: Callable) -> List[ExperimentResult]:
        """Run multiple experiments with controlled parallelism and auto-balancing."""
        results = []
        
        # Pass runtime parameters to experiments
        for exp in experiments:
            if self.auto_balance:
                exp.parameters['batch_size'] = self.current_batch_size
                exp.parameters['num_workers'] = self.current_workers
        
        i = 0
        while i < len(experiments):
            # Check and apply auto-balancing recommendations
            if self.balancer:
                recommendations = self.balancer.get_recommendations(
                    self.max_parallel,
                    self.current_batch_size,
                    self.current_workers
                )
                
                # Apply recommendations
                if recommendations['parallel_experiments'] != self.max_parallel:
                    self.max_parallel = recommendations['parallel_experiments']
                    if self.config.verbose:
                        print(f"🔄 Adjusted parallel experiments to {self.max_parallel}")
                
                if recommendations['batch_size'] != self.current_batch_size:
                    self.current_batch_size = recommendations['batch_size']
                    # Update remaining experiments
                    for exp in experiments[i:]:
                        exp.parameters['batch_size'] = self.current_batch_size
                    if self.config.verbose:
                        print(f"🔄 Adjusted batch size to {self.current_batch_size}")
                
                if recommendations['num_workers'] != self.current_workers:
                    self.current_workers = recommendations['num_workers']
                    # Update remaining experiments
                    for exp in experiments[i:]:
                        exp.parameters['num_workers'] = self.current_workers
                    if self.config.verbose:
                        print(f"🔄 Adjusted data workers to {self.current_workers}")
            
            # Run batch with current settings
            batch_size = min(self.max_parallel, len(experiments) - i)
            batch = experiments[i:i + batch_size]
            
            for j, exp in enumerate(batch):
                if exp.device_id is None:
                    exp.device_id = self.device_ids[j % len(self.device_ids)]
            
            batch_tasks = [self.run_experiment(exp, test_function) for exp in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            
            i += batch_size
            
            # Small delay between batches
            if i < len(experiments):
                await asyncio.sleep(1)
        
        return results


class ParallelExperimentRunner(ExperimentRunnerBase):
    """Parallel experiment runner using multiprocessing.Pool."""
    
    def __init__(self, config: LabConfig):
        self.config = config
        self.device_ids = config.device_ids
        self.max_parallel = config.max_parallel_experiments
    
    async def run_experiment(self, experiment: Experiment) -> ExperimentResult:
        """Run a single experiment."""
        device_id = experiment.device_id or self.device_ids[0]
        return run_structure_net_experiment(experiment, device_id)
    
    async def run_experiments(self, experiments: List[Experiment]) -> List[ExperimentResult]:
        """Run multiple experiments in parallel."""
        # Prepare arguments
        args = []
        for i, exp in enumerate(experiments):
            device_id = exp.device_id or self.device_ids[i % len(self.device_ids)]
            args.append((exp, device_id))
        
        # Run in parallel
        with mp.Pool(processes=self.max_parallel) as pool:
            results = pool.starmap(run_structure_net_experiment, args)
        
        return results


class ExperimentRunner:
    """Main experiment runner that selects appropriate backend."""
    
    def __init__(self, config: LabConfig):
        self.config = config
        self.async_runner = AsyncExperimentRunner(config)
    
    async def run_experiments(self, experiments: List[Experiment]) -> List[ExperimentResult]:
        """Run experiments using the most appropriate method."""
        return await self.async_runner.run_experiments(experiments)
