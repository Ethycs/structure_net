#!/usr/bin/env python3
"""
Example usage of the Scheduler Strategy Selector component.

This demonstrates how to use the strategy selector to replace the old factory pattern.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from .scheduler_strategy_selector import SchedulerStrategySelector, SchedulingStrategy


def example_basic_usage():
    """Basic usage example."""
    # Create a simple network
    network = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Create strategy selector
    selector = SchedulerStrategySelector()
    
    # Define context
    context = {
        'network': network,
        'strategy': SchedulingStrategy.BASIC,
        'base_lr': 0.001,
        'dataset_size': 'medium',
        'task_type': 'classification'
    }
    
    # Select strategy
    strategy_config = selector.select_strategy(context)
    print(f"Selected strategy: {strategy_config}")
    
    # Create orchestrator
    orchestrator = selector.create_orchestrator(network, strategy_config)
    
    # Create optimizer
    optimizer = selector.create_adaptive_optimizer(network, orchestrator)
    
    return orchestrator, optimizer


def example_transfer_learning():
    """Transfer learning example."""
    # Pretrained network
    network = nn.Sequential(
        nn.Linear(784, 256),  # Pretrained layers
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),   # New layers
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    selector = SchedulerStrategySelector()
    
    context = {
        'network': network,
        'strategy': SchedulingStrategy.TRANSFER_LEARNING,
        'base_lr': 0.0001,  # Lower for fine-tuning
        'is_pretrained': True,
        'dataset_size': 'small',
        'task_type': 'classification'
    }
    
    strategy_config = selector.select_strategy(context)
    orchestrator = selector.create_orchestrator(network, strategy_config)
    optimizer = selector.create_adaptive_optimizer(network, orchestrator)
    
    return orchestrator, optimizer


def example_custom_strategy():
    """Custom strategy example."""
    network = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    selector = SchedulerStrategySelector()
    
    # Custom configuration
    context = {
        'network': network,
        'strategy': SchedulingStrategy.CUSTOM,
        'base_lr': 0.005,
        'custom_schedulers': ['extrema_phase', 'multi_scale'],
        'scheduler_configs': {
            'extrema_phase': {
                'explosive_threshold': 0.15,
                'steady_threshold': 0.02
            },
            'multi_scale': {
                'temporal_decay': 0.02,
                'coarse_multiplier': 0.05
            }
        }
    }
    
    strategy_config = selector.select_strategy(context)
    orchestrator = selector.create_orchestrator(network, strategy_config)
    optimizer = selector.create_adaptive_optimizer(
        network, 
        orchestrator,
        optimizer_class=optim.SGD,
        optimizer_kwargs={'momentum': 0.9}
    )
    
    return orchestrator, optimizer


def example_automatic_recommendation():
    """Automatic strategy recommendation example."""
    network = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    selector = SchedulerStrategySelector()
    
    # Let the selector recommend a strategy
    context = {
        'network': network,
        'dataset_size': 'large',
        'is_pretrained': False,
        'task_type': 'classification'
    }
    
    recommended_strategy = selector.recommend_strategy(context)
    print(f"Recommended strategy: {recommended_strategy.value}")
    
    # Use the recommendation
    context['strategy'] = recommended_strategy
    context['base_lr'] = 0.001
    
    strategy_config = selector.select_strategy(context)
    orchestrator = selector.create_orchestrator(network, strategy_config)
    optimizer = selector.create_adaptive_optimizer(network, orchestrator)
    
    return orchestrator, optimizer


def example_training_loop():
    """Complete training loop example."""
    # Setup
    network = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    selector = SchedulerStrategySelector()
    
    context = {
        'network': network,
        'strategy': SchedulingStrategy.ADVANCED,
        'base_lr': 0.001,
        'dataset_size': 'medium'
    }
    
    strategy_config = selector.select_strategy(context)
    orchestrator = selector.create_orchestrator(network, strategy_config)
    optimizer = selector.create_adaptive_optimizer(network, orchestrator)
    
    # Training loop (simplified)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10):
        # Orchestrate learning rates
        lr_context = {
            'epoch': epoch,
            'global_step': epoch * 100,  # Approximate
            'data_loader': None  # Would be actual data loader
        }
        
        orchestration_result = orchestrator.orchestrate(
            {'optimizer': optimizer, 'network': network},
            lr_context
        )
        
        adapted_optimizer = orchestration_result['adapted_optimizer']
        lr_info = orchestration_result['lr_info']
        
        print(f"Epoch {epoch}: Average LR = {sum(lr_info.values())/len(lr_info):.6f}")
        
        # Training steps would go here...


def migration_from_factory():
    """Show migration from old factory pattern."""
    network = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    
    # OLD WAY (deprecated):
    # from structure_net.evolution.adaptive_learning_rates.factory import create_advanced_manager
    # manager = create_advanced_manager(network, base_lr=0.001)
    
    # NEW WAY:
    selector = SchedulerStrategySelector()
    context = {
        'network': network,
        'strategy': SchedulingStrategy.ADVANCED,
        'base_lr': 0.001
    }
    strategy_config = selector.select_strategy(context)
    orchestrator = selector.create_orchestrator(network, strategy_config)
    optimizer = selector.create_adaptive_optimizer(network, orchestrator)
    
    print("Successfully migrated from factory to strategy pattern!")
    
    return orchestrator, optimizer


if __name__ == "__main__":
    print("=== Scheduler Strategy Selector Examples ===\n")
    
    print("1. Basic Usage:")
    example_basic_usage()
    
    print("\n2. Transfer Learning:")
    example_transfer_learning()
    
    print("\n3. Custom Strategy:")
    example_custom_strategy()
    
    print("\n4. Automatic Recommendation:")
    example_automatic_recommendation()
    
    print("\n5. Training Loop:")
    example_training_loop()
    
    print("\n6. Migration from Factory:")
    migration_from_factory()