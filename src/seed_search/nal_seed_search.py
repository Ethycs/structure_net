#!/usr/bin/env python3
"""
NAL-Powered GPU Seed Hunter

This script uses the Neural Architecture Lab (NAL) framework to perform a
systematic, GPU-accelerated search for optimal seed networks.
"""

import asyncio
from datetime import datetime

from src.neural_architecture_lab import (
    NeuralArchitectureLab,
    LabConfig,
    Hypothesis
)
from src.neural_architecture_lab.hypothesis_library import SeedSearchHypotheses

def main():
    """Main entry point for the NAL-powered seed search."""
    print("🚀 NAL-Powered Seed Search 🚀")

    # 1. Define the Seed Search Hypothesis
    seed_hypothesis = SeedSearchHypotheses.find_optimal_seeds()

    # 2. Configure and run the Neural Architecture Lab
    lab_config = LabConfig(
        # Adjust parallel experiments based on available GPUs
        max_parallel_experiments=4, 
        results_dir=f"nal_seed_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        save_best_models=True,
        min_experiments_per_hypothesis=100 # Run a large number of experiments
    )
    
    lab = NeuralArchitectureLab(lab_config)
    lab.register_hypothesis(seed_hypothesis)

    # 3. Run the experiment
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(lab.run_all_hypotheses())
    
    # 4. Print the results
    result = results.get("seed_search_optimal")
    if result:
        print("\n" + "="*60)
        print("Seed Search Results")
        print("="*60)
        print(f"Hypothesis Confirmed: {result.confirmed}")
        
        best_params = result.best_parameters
        best_metrics = result.best_metrics
        
        print("\n🏆 Best Seed Found (by patchability):")
        print(f"  - Architecture: {best_params.get('architecture')}")
        print(f"  - Sparsity: {best_params.get('sparsity'):.3f}")
        print(f"  - Seed: {best_params.get('seed')}")
        print(f"  - Patchability: {best_metrics.get('patchability'):.4f}")
        print(f"  - Accuracy: {best_metrics.get('accuracy'):.2%}")

    else:
        print("Seed search did not produce a result.")

if __name__ == "__main__":
    main()
