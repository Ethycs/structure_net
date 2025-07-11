#!/usr/bin/env python3
"""
Test script to demonstrate NAL's auto-balancing capabilities.

This script monitors resource usage and shows how the system automatically
adjusts parallelism, batch sizes, and worker counts.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_architecture_lab import (
    NeuralArchitectureLab,
    LabConfig,
    Hypothesis,
    HypothesisCategory,
    Experiment,
    ExperimentResult
)
import torch
import time
import psutil


def dummy_experiment(experiment: Experiment, device_id: int) -> ExperimentResult:
    """A dummy experiment that simulates resource usage."""
    config = experiment.parameters
    
    # Simulate work
    batch_size = config.get('batch_size', 128)
    duration = config.get('duration', 5.0)
    
    # Create dummy tensors to use memory
    if torch.cuda.is_available() and device_id >= 0:
        device = f'cuda:{device_id}'
        # Allocate GPU memory based on batch size
        dummy_data = torch.randn(batch_size, 3, 224, 224, device=device)
        dummy_model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(64 * 222 * 222, 10)
        ).to(device)
    else:
        device = 'cpu'
        dummy_data = torch.randn(batch_size, 100)
    
    # Simulate computation
    start = time.time()
    while time.time() - start < duration:
        if device != 'cpu':
            # GPU computation
            _ = dummy_model(dummy_data)
        else:
            # CPU computation
            _ = torch.matmul(dummy_data, dummy_data.T)
        time.sleep(0.1)
    
    # Return mock results
    return ExperimentResult(
        experiment_id=experiment.id,
        hypothesis_id=experiment.hypothesis_id,
        metrics={'resource_test': 1.0, 'batch_size': batch_size},
        primary_metric=1.0,
        model_architecture=[100, 50, 10],
        model_parameters=5000,
        training_time=duration
    )


async def main():
    """Run auto-balance test."""
    print("🧪 Testing NAL Auto-Balancer")
    print(f"System: {psutil.cpu_count()} CPUs, {torch.cuda.device_count()} GPUs")
    print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print()
    
    # Create lab config with auto-balancing
    config = LabConfig(
        project_name="autobalance_test",
        results_dir="data/autobalance_test",
        verbose=True,
        auto_balance=True,
        target_cpu_percent=70.0,
        max_cpu_percent=85.0,
        target_gpu_percent=80.0,
        max_gpu_percent=90.0
    )
    
    # Create lab
    lab = NeuralArchitectureLab(config)
    
    # Create a hypothesis with many experiments
    hypothesis = Hypothesis(
        id="resource_test",
        name="Resource Usage Test",
        description="Test auto-balancing under different loads",
        category=HypothesisCategory.TRAINING,
        question="Can the system auto-balance resources?",
        prediction="System will adjust to maintain target utilization",
        test_function=dummy_experiment,
        parameter_space={
            'duration': [2.0, 5.0, 10.0],
            'batch_size': [64, 128, 256, 512]
        },
        control_parameters={},
        success_metrics={'resource_test': 0.5}
    )
    
    # Register and run
    lab.register_hypothesis(hypothesis)
    
    print("Running experiments with auto-balancing...")
    print("Watch for 🔄 adjustment messages!\n")
    
    result = await lab.test_hypothesis(hypothesis.id)
    
    print(f"\n✅ Completed {result.successful_experiments} experiments")
    print(f"❌ Failed {result.failed_experiments} experiments")
    
    # Show final resource state
    print("\nFinal resource utilization:")
    print(f"CPU: {psutil.cpu_percent(interval=1):.1f}%")
    print(f"Memory: {psutil.virtual_memory().percent:.1f}%")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            used = torch.cuda.memory_allocated(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {used:.1f}/{total:.1f} GB ({used/total*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())