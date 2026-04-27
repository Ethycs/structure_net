#!/usr/bin/env python3
"""
NAL-Powered Seed Search Experiment

This script defines and runs a systematic, GPU-accelerated search for
optimal seed networks using the Neural Architecture Lab (NAL).
"""

import asyncio
from datetime import datetime
import argparse

from src.neural_architecture_lab import (
    NeuralArchitectureLab,
    LabConfig,
    LabConfigFactory,
    Hypothesis,
    HypothesisCategory,
)
from src.seed_search.architecture_generator import ArchitectureGenerator
from src.neural_architecture_lab.workers.seed_search_worker import evaluate_seed_task

def get_default_lab_config() -> LabConfig:
    """Returns a default configuration for the NAL lab for this experiment."""
    return LabConfig(
        project_name="seed_search",
        max_parallel_experiments=4,
        log_level="INFO",
        enable_wandb=False,
        save_best_models=True,
        device_ids=[0]
    )

def create_seed_search_hypothesis(dataset_name: str, num_architectures: int) -> Hypothesis:
    """Creates the hypothesis for the seed search experiment."""
    arch_generator = ArchitectureGenerator.from_dataset(dataset_name)
    architectures = arch_generator.generate_systematic_batch(num_architectures)

    parameter_space = {
        "architecture": architectures,
        "sparsity": [0.01, 0.02, 0.05, 0.1],
        "seed": list(range(5)),
    }

    return Hypothesis(
        id="seed_search_optimal",
        name="Optimal Seed Search",
        description="Systematically search for optimal seed networks based on patchability and accuracy.",
        category=HypothesisCategory.ARCHITECTURE,
        question="Which combination of architecture, sparsity, and random seed produces the most promising starting network?",
        prediction="Certain simple, sparse architectures will exhibit high patchability and provide a good foundation for growth.",
        test_function=evaluate_seed_task,
        parameter_space=parameter_space,
        control_parameters={"dataset": dataset_name, "epochs": 10, "batch_size": 128},
        success_metrics={"patchability": 0.8, "accuracy": 0.2},
    )

async def run_seed_search(lab_config: LabConfig, hypothesis: Hypothesis):
    """Runs the seed search experiment."""
    lab = NeuralArchitectureLab(lab_config)
    lab.register_hypothesis(hypothesis)
    
    results = await lab.run_all_hypotheses()
    
    result = results.get(hypothesis.id)
    if result:
        print("\n" + "="*60)
        print("Seed Search Results")
        print("="*60)
        print(f"Hypothesis Confirmed: {result.confirmed}")
        
        best_params = result.best_parameters
        best_metrics = result.best_metrics
        
        print("\n🏆 Best Seed Found (by patchability):")
        if best_params:
            print(f"  - Architecture: {best_params.get('architecture')}")
            print(f"  - Sparsity: {best_params.get('sparsity'):.3f}")
            print(f"  - Seed: {best_params.get('seed')}")
        if best_metrics:
            print(f"  - Patchability: {best_metrics.get('patchability'):.4f}")
            print(f"  - Accuracy: {best_metrics.get('accuracy'):.2%}")

    else:
        print("Seed search did not produce a result.")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="NAL-Powered Seed Search")
    parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset to use for the search.')
    parser.add_argument('--num-arch', type=int, default=50, help='Number of architectures to generate.')
    
    LabConfigFactory.add_arguments(parser)
    args = parser.parse_args()

    lab_config = LabConfigFactory.from_args(args, base_config=get_default_lab_config())
    hypothesis = create_seed_search_hypothesis(args.dataset, args.num_arch)
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_seed_search(lab_config, hypothesis))

if __name__ == "__main__":
    main()