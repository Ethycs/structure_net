#!/usr/bin/env python3
"""
Standardized Logging Example

This example demonstrates how to use the new standardized logging system
with Pydantic validation and WandB artifacts.

Run this example to see:
- Schema validation in action
- Real-time WandB logging
- Artifact creation and queuing
- Error handling and debugging
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.structure_net.logging import (
    create_growth_logger,
    create_training_logger,
    create_tournament_logger,
    get_queue_status
)
from src.structure_net import create_standard_network


def example_1_basic_growth_logging():
    """Example 1: Basic growth experiment logging."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Growth Experiment Logging")
    print("="*60)
    
    # Create logger with validation
    logger = create_growth_logger(
        project_name="standardized_logging_demo",
        experiment_name=f"basic_growth_{datetime.now().strftime('%H%M%S')}",
        config={
            'dataset': 'mnist',
            'batch_size': 64,
            'learning_rate': 0.001,
            'device': 'cpu',
            'target_accuracy': 0.90
        },
        tags=['demo', 'basic_growth', 'example']
    )
    
    # Create a simple network
    network = create_standard_network(
        architecture=[784, 128, 64, 10],
        sparsity=0.02,
        device='cpu'
    )
    
    print("📊 Starting experiment...")
    logger.log_experiment_start(
        network=network,
        target_accuracy=0.90,
        seed_architecture=[784, 128, 64, 10]
    )
    
    # Simulate growth iterations
    for iteration in range(3):
        print(f"🔄 Growth iteration {iteration}")
        
        # Simulate training progress
        accuracy = 0.70 + (iteration * 0.08)
        loss = 0.8 - (iteration * 0.15)
        
        # Simulate extrema analysis
        extrema_analysis = {
            'total_extrema': 15 + (iteration * 5),
            'extrema_ratio': 0.12 + (iteration * 0.03),
            'dead_neurons': {
                '0': [1, 5, 12] if iteration > 0 else [],
                '1': [3, 8] if iteration > 1 else []
            },
            'saturated_neurons': {
                '0': [45, 67],
                '1': [23, 34, 56] if iteration > 0 else []
            },
            'layer_health': {
                '0': 0.85 - (iteration * 0.05),
                '1': 0.90 - (iteration * 0.03),
                '2': 0.88
            }
        }
        
        # Simulate growth actions
        growth_actions = []
        growth_occurred = False
        
        if iteration > 0:  # Growth after first iteration
            growth_actions.append({
                'action': 'add_patch',
                'position': 1,
                'size': 5,
                'reason': f"Dead neurons detected in layer 1: {len(extrema_analysis['dead_neurons']['1'])}",
                'success': True
            })
            growth_occurred = True
        
        # Log iteration with validation
        logger.log_growth_iteration(
            iteration=iteration,
            network=network,
            accuracy=accuracy,
            loss=loss,
            extrema_analysis=extrema_analysis,
            growth_actions=growth_actions,
            growth_occurred=growth_occurred,
            credits=100.0 - (iteration * 10)  # Simulate credit system
        )
    
    # Finish experiment
    print("✅ Finishing experiment...")
    artifact_hash = logger.finish_experiment(final_accuracy=0.86)
    
    print(f"🎯 Experiment completed!")
    print(f"📦 Artifact hash: {artifact_hash}")
    print(f"🔗 WandB URL: {logger.wandb_logger.run.url}")
    
    return artifact_hash


def example_2_training_experiment():
    """Example 2: Standard training experiment."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Standard Training Experiment")
    print("="*60)
    
    # Create training logger
    logger = create_training_logger(
        project_name="standardized_logging_demo",
        experiment_name=f"training_{datetime.now().strftime('%H%M%S')}",
        config={
            'dataset': 'cifar10',
            'batch_size': 32,
            'learning_rate': 0.001,
            'max_epochs': 5,
            'device': 'cpu',
            'optimizer': 'adam'
        },
        tags=['demo', 'training', 'cifar10']
    )
    
    # Create simple network for demonstration
    network = nn.Sequential(
        nn.Linear(3072, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    print("📊 Starting training experiment...")
    logger.log_experiment_start(network=network)
    
    # Simulate training epochs
    for epoch in range(5):
        print(f"🔄 Training epoch {epoch}")
        
        # Simulate training metrics
        train_loss = 2.0 - (epoch * 0.3)
        train_acc = 0.20 + (epoch * 0.15)
        val_loss = 2.2 - (epoch * 0.25)
        val_acc = 0.18 + (epoch * 0.12)
        lr = 0.001 * (0.9 ** epoch)  # Decay learning rate
        
        # Log epoch with validation
        logger.log_training_epoch(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            learning_rate=lr,
            duration=45.0 + (epoch * 2)  # Simulate epoch duration
        )
    
    # Finish experiment
    print("✅ Finishing training...")
    artifact_hash = logger.finish_experiment(final_accuracy=val_acc)
    
    print(f"🎯 Training completed!")
    print(f"📦 Artifact hash: {artifact_hash}")
    
    return artifact_hash


def example_3_tournament_experiment():
    """Example 3: Tournament-based growth experiment."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Tournament Growth Experiment")
    print("="*60)
    
    # Create tournament logger
    logger = create_tournament_logger(
        project_name="standardized_logging_demo",
        experiment_name=f"tournament_{datetime.now().strftime('%H%M%S')}",
        config={
            'dataset': 'mnist',
            'tournament_strategies': ['extrema_growth', 'random_growth', 'layer_addition'],
            'tournament_epochs': 3,
            'device': 'cpu'
        },
        tags=['demo', 'tournament', 'strategy_comparison']
    )
    
    network = create_standard_network([784, 64, 10], sparsity=0.05, device='cpu')
    
    print("📊 Starting tournament experiment...")
    logger.log_experiment_start(network=network)
    
    # Simulate tournament iterations
    for iteration in range(2):
        print(f"🏆 Tournament iteration {iteration}")
        
        # Simulate tournament results
        tournament_results = {
            'winner': {
                'strategy': 'extrema_growth',
                'improvement': 0.15 + (iteration * 0.05),
                'final_accuracy': 0.85 + (iteration * 0.03),
                'execution_time': 120.5,
                'success': True
            },
            'all_results': [
                {
                    'strategy': 'extrema_growth',
                    'improvement': 0.15 + (iteration * 0.05),
                    'final_accuracy': 0.85 + (iteration * 0.03),
                    'execution_time': 120.5,
                    'success': True
                },
                {
                    'strategy': 'random_growth',
                    'improvement': 0.08 + (iteration * 0.02),
                    'final_accuracy': 0.78 + (iteration * 0.02),
                    'execution_time': 95.2,
                    'success': True
                },
                {
                    'strategy': 'layer_addition',
                    'improvement': 0.12 + (iteration * 0.03),
                    'final_accuracy': 0.82 + (iteration * 0.025),
                    'execution_time': 150.8,
                    'success': True
                }
            ]
        }
        
        # Log tournament results with validation
        logger.log_tournament_results(tournament_results, iteration)
        
        # Also log as growth iteration
        logger.log_growth_iteration(
            iteration=iteration,
            network=network,
            accuracy=tournament_results['winner']['final_accuracy'],
            growth_occurred=True,
            growth_actions=[{
                'action': 'tournament_winner',
                'reason': f"Tournament selected: {tournament_results['winner']['strategy']}",
                'success': True
            }]
        )
    
    # Finish experiment
    print("✅ Finishing tournament...")
    final_accuracy = tournament_results['winner']['final_accuracy']
    artifact_hash = logger.finish_experiment(final_accuracy=final_accuracy)
    
    print(f"🎯 Tournament completed!")
    print(f"🏆 Winner: {tournament_results['winner']['strategy']}")
    print(f"📦 Artifact hash: {artifact_hash}")
    
    return artifact_hash


def example_4_validation_errors():
    """Example 4: Demonstrate validation error handling."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Validation Error Handling")
    print("="*60)
    
    logger = create_growth_logger(
        project_name="standardized_logging_demo",
        experiment_name=f"validation_demo_{datetime.now().strftime('%H%M%S')}",
        tags=['demo', 'validation', 'error_handling']
    )
    
    network = create_standard_network([784, 32, 10], device='cpu')
    logger.log_experiment_start(network=network)
    
    print("🧪 Testing validation errors...")
    
    # Test 1: Invalid accuracy (> 1.0)
    print("\n1. Testing invalid accuracy...")
    try:
        logger.log_growth_iteration(
            iteration=0,
            network=network,
            accuracy=1.5,  # ❌ Invalid: > 1.0
            growth_occurred=False
        )
        print("❌ Should have failed!")
    except Exception as e:
        print(f"✅ Caught validation error: {type(e).__name__}")
        print(f"   Message: {str(e)[:100]}...")
    
    # Test 2: Negative iteration
    print("\n2. Testing negative iteration...")
    try:
        logger.log_growth_iteration(
            iteration=-1,  # ❌ Invalid: negative
            network=network,
            accuracy=0.85,
            growth_occurred=False
        )
        print("❌ Should have failed!")
    except Exception as e:
        print(f"✅ Caught validation error: {type(e).__name__}")
        print(f"   Message: {str(e)[:100]}...")
    
    # Test 3: Invalid growth action
    print("\n3. Testing invalid growth action...")
    try:
        logger.log_growth_iteration(
            iteration=0,
            network=network,
            accuracy=0.85,
            growth_occurred=True,
            growth_actions=[{
                'action': 'invalid_action',  # ❌ Not in allowed types
                'reason': 'Testing validation'
            }]
        )
        print("❌ Should have failed!")
    except Exception as e:
        print(f"✅ Caught validation error: {type(e).__name__}")
        print(f"   Message: {str(e)[:100]}...")
    
    # Test 4: Valid iteration (should work)
    print("\n4. Testing valid iteration...")
    try:
        logger.log_growth_iteration(
            iteration=0,
            network=network,
            accuracy=0.85,  # ✅ Valid
            growth_occurred=True,
            growth_actions=[{
                'action_type': 'add_layer',  # ✅ Valid
                'reason': 'Testing successful validation',
                'success': True
            }]
        )
        print("✅ Valid iteration logged successfully!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Finish experiment
    logger.finish_experiment(final_accuracy=0.85, save_artifact=False)
    print("\n✅ Validation testing completed!")


def example_5_queue_management():
    """Example 5: Demonstrate queue management."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Queue Management")
    print("="*60)
    
    # Check initial queue status
    print("📊 Initial queue status:")
    status = get_queue_status()
    for key, value in status.items():
        if key != 'directories':
            print(f"   {key}: {value}")
    
    # Create a simple experiment that will be queued
    logger = create_growth_logger(
        project_name="standardized_logging_demo",
        experiment_name=f"queue_demo_{datetime.now().strftime('%H%M%S')}",
        tags=['demo', 'queue_management']
    )
    
    network = create_standard_network([784, 16, 10], device='cpu')
    logger.log_experiment_start(network=network)
    
    # Log a simple iteration
    logger.log_growth_iteration(
        iteration=0,
        network=network,
        accuracy=0.75,
        growth_occurred=False
    )
    
    # Save artifact (will be queued)
    print("\n📦 Saving experiment artifact...")
    artifact_hash = logger.save_experiment_artifact()
    
    # Check queue status after adding experiment
    print("\n📊 Queue status after adding experiment:")
    status = get_queue_status()
    for key, value in status.items():
        if key != 'directories':
            print(f"   {key}: {value}")
    
    # Finish experiment
    logger.finish_experiment(save_artifact=False)  # Don't save again
    
    print(f"\n✅ Queue management demo completed!")
    print(f"📦 Artifact {artifact_hash} is queued for upload")
    print("\n💡 To process the queue, run:")
    print("   python -m structure_net.logging.cli process")


def main():
    """Run all examples."""
    print("🚀 STANDARDIZED LOGGING SYSTEM EXAMPLES")
    print("="*60)
    print("This demo shows the new logging system with:")
    print("✅ Pydantic validation")
    print("✅ WandB artifact creation")
    print("✅ Local-first queue system")
    print("✅ Real-time monitoring")
    print("✅ Error handling")
    
    try:
        # Run examples
        example_1_basic_growth_logging()
        example_2_training_experiment()
        example_3_tournament_experiment()
        example_4_validation_errors()
        example_5_queue_management()
        
        print("\n" + "="*60)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Final queue status
        print("\n📊 Final queue status:")
        status = get_queue_status()
        for key, value in status.items():
            if key != 'directories':
                print(f"   {key}: {value}")
        
        print("\n💡 Next steps:")
        print("1. Check your WandB project for real-time metrics")
        print("2. Process the artifact queue:")
        print("   python -m structure_net.logging.cli process")
        print("3. View artifacts in WandB for persistent data")
        print("4. Try the CLI tools:")
        print("   python -m structure_net.logging.cli status")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("💡 This might be due to missing dependencies or WandB setup")
        print("   Try: wandb login")
        raise


if __name__ == "__main__":
    main()
