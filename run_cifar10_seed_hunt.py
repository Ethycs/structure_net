#!/usr/bin/env python3
"""
CIFAR-10 Seed Hunt - High Accuracy, Low Sparsity

Hunt for CIFAR-10 seeds with high accuracy and low sparsity using the canonical system.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
from src.structure_net import GPUSeedHunter, ArchitectureGenerator

def main():
    print("🔍 CIFAR-10 SEED HUNT - HIGH ACCURACY, LOW SPARSITY")
    print("=" * 60)
    
    # Configuration for high accuracy, low sparsity
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create architecture generator for CIFAR-10
    arch_gen = ArchitectureGenerator(
        input_size=3072,  # CIFAR-10 flattened
        num_classes=10    # 10 classes
    )
    
    # Generate architectures optimized for accuracy
    architectures = []
    
    # Add some proven good architectures for CIFAR-10
    proven_archs = [
        [3072, 512, 10],      # Simple but effective
        [3072, 1024, 10],     # Wider for better capacity
        [3072, 2048, 10],     # Even wider
        [3072, 512, 128, 10], # Two layers
        [3072, 1024, 256, 10], # Two layers, wider
        [3072, 2048, 512, 10], # Two layers, very wide
        [3072, 1024, 512, 128, 10], # Three layers
        [3072, 2048, 1024, 256, 10], # Three layers, wide
    ]
    
    architectures.extend(proven_archs)
    
    # Add some generated architectures
    generated = arch_gen.generate_systematic_batch(10)
    architectures.extend(generated)
    
    print(f"Testing {len(architectures)} architectures")
    
    # Create GPU seed hunter with settings optimized for high accuracy
    hunter = GPUSeedHunter(
        num_gpus=1,
        device=device.type,
        save_promising=True,
        dataset='cifar10',
        save_threshold=0.40,  # High accuracy threshold
        keep_top_k=10
    )
    
    # Set sorting configuration
    hunter._disable_sorting = False  # Enable neuron sorting
    hunter._sort_frequency = 3       # Sort frequently
    
    # Hunt for seeds using GPU saturated search
    print("\n🚀 Starting seed hunt...")
    results = hunter.gpu_saturated_search(
        num_architectures=len(architectures),
        seeds_per_arch=20,  # More seeds per config
        sparsity=0.01       # Low sparsity for high accuracy
    )
    
    # Print results
    print(f"\n📊 SEED HUNT RESULTS")
    print("=" * 40)
    print(f"Total configurations tested: {results['total_configs']}")
    print(f"Total seeds tested: {results['total_seeds']}")
    print(f"Models saved: {results['models_saved']}")
    print(f"Best accuracy: {results['best_accuracy']:.2%}")
    print(f"Best efficiency: {results['best_efficiency']:.4f}")
    
    # Show top models
    if 'top_models' in results:
        print(f"\n🏆 TOP MODELS:")
        for i, model in enumerate(results['top_models'][:5]):
            print(f"  {i+1}. Acc: {model['accuracy']:.2%}, "
                  f"Arch: {model['architecture']}, "
                  f"Sparsity: {model['sparsity']:.1%}, "
                  f"Seed: {model['seed']}")
    
    print(f"\n✅ Seed hunt complete! Check data/promising_models/ for results.")

if __name__ == "__main__":
    main()
