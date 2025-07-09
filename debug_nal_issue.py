#!/usr/bin/env python3
"""Debug script to investigate NAL experiment issue."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import torch
from src.neural_architecture_lab import (
    NeuralArchitectureLab,
    LabConfig,
    Experiment,
    AdvancedExperimentRunner
)

async def test_single_experiment():
    """Test a single NAL experiment to see what happens."""
    
    # Configure NAL
    config = LabConfig(
        max_parallel_experiments=1,
        experiment_timeout=300,
        device_ids=[0],
        min_experiments_per_hypothesis=1,
        require_statistical_significance=False,
        results_dir="debug_nal_test",
        verbose=True
    )
    
    # Create lab with advanced runner
    lab = NeuralArchitectureLab(config)
    lab.runner = AdvancedExperimentRunner(config)
    
    # Create a simple experiment
    experiment = Experiment(
        id='debug_test_001',
        hypothesis_id='debug_hypothesis',
        name='Debug Test',
        parameters={
            'architecture': [784, 256, 128, 10],
            'sparsity': 0.02,
            'batch_size': 128,
            'epochs': 5,  # Quick test
            'base_lr': 0.001,
            'lr_strategy': 'basic',
            'enable_growth': False,
            'enable_residual_blocks': False,
            'dataset': 'cifar10',
            'primary_metric_type': 'accuracy',
            'quick_test': True  # Use subset of data
        },
        seed=42
    )
    
    print("Running experiment...")
    print(f"Parameters: {experiment.parameters}")
    
    # Run experiment
    result = await lab.runner.run_experiment(experiment)
    
    print(f"\nResult:")
    print(f"  Experiment ID: {result.experiment_id}")
    print(f"  Primary metric: {result.primary_metric}")
    print(f"  Metrics: {result.metrics}")
    print(f"  Model parameters: {result.model_parameters}")
    print(f"  Training time: {result.training_time:.2f}s")
    print(f"  Error: {result.error}")
    
    if result.error:
        print(f"\nFull error:\n{result.error}")
    
    return result

def main():
    """Run the debug test."""
    print("NAL Debugging Script")
    print("=" * 50)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU available - this might be the issue!")
        return
    
    # Run async test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(test_single_experiment())
        if result.metrics.get('accuracy', 0) == 0:
            print("\n⚠️  WARNING: Accuracy is 0! This confirms the issue.")
            
            # Let's also test the run_advanced_experiment function directly
            print("\nTesting run_advanced_experiment directly...")
            from src.neural_architecture_lab.advanced_runners import run_advanced_experiment, GPUMemoryManager
            
            memory_manager = GPUMemoryManager()
            direct_result = run_advanced_experiment(
                experiment=Experiment(
                    id='direct_test',
                    hypothesis_id='direct_hypothesis',
                    name='Direct Test',
                    parameters={
                        'architecture': [784, 256, 128, 10],
                        'sparsity': 0.02,
                        'batch_size': 128,
                        'epochs': 2,
                        'base_lr': 0.001,
                        'lr_strategy': 'basic',
                        'enable_growth': False,
                        'enable_residual_blocks': False,
                        'dataset': 'cifar10',
                        'primary_metric_type': 'accuracy',
                        'quick_test': True
                    },
                    seed=42
                ),
                device_id=0,
                memory_manager=memory_manager,
                enable_mixed_precision=False
            )
            
            print(f"\nDirect result:")
            print(f"  Primary metric: {direct_result.primary_metric}")
            print(f"  Metrics: {direct_result.metrics}")
            print(f"  Error: {direct_result.error}")
            
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()