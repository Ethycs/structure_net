#!/usr/bin/env python3
"""
Command Line Interface for GPU Seed Hunter

Provides a clean CLI for running seed hunting experiments using the canonical standard.
"""

import argparse
import os
import torch
import json
import numpy as np
from typing import Dict, Any

from .gpu_seed_hunter import GPUSeedHunter, SparsitySweepConfig


def main():
    """Main entry point for the seed hunter CLI"""
    parser = argparse.ArgumentParser(description='GPU Seed Hunter with Canonical Standard')
    parser.add_argument('--mode', type=str, default='single', 
                       choices=['single', 'sweep', 'range'],
                       help='Search mode: single sparsity, predefined sweep, or custom range (default: single)')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                       help='Dataset to use (default: mnist)')
    parser.add_argument('--num-architectures', type=int, default=30,
                       help='Number of architectures to test (default: 30)')
    parser.add_argument('--seeds-per-arch', type=int, default=10,
                       help='Number of seeds per architecture (default: 10)')
    parser.add_argument('--sparsity', type=float, default=0.02,
                       help='Sparsity level for single mode (default: 0.02)')
    parser.add_argument('--sparsity-min', type=float, default=0.001,
                       help='Minimum sparsity for range mode (default: 0.001)')
    parser.add_argument('--sparsity-max', type=float, default=0.1,
                       help='Maximum sparsity for range mode (default: 0.1)')
    parser.add_argument('--sparsity-step', type=float, default=0.01,
                       help='Sparsity increment for range mode (default: 0.01)')
    parser.add_argument('--sparsity-list', type=str, default=None,
                       help='Comma-separated list of sparsities (e.g., "0.01,0.02,0.05,0.1")')
    parser.add_argument('--thresh', type=float, default=0.25,
                       help='Accuracy threshold for saving models (default: 0.25 = 25%%)')
    parser.add_argument('--disable-sorting', action='store_true',
                       help='Disable neuron sorting during training')
    parser.add_argument('--sort-frequency', type=int, default=5,
                       help='Sort neurons every N epochs (default: 5)')
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='Number of GPUs to use for parallel processing (default: 1)')
    parser.add_argument('--gpu-ids', type=str, default=None,
                       help='Comma-separated GPU IDs to use (e.g., "0,1,2"). If not specified, uses first N GPUs.')
    
    args = parser.parse_args()
    
    # Configure GPU usage
    if args.gpu_ids:
        gpu_list = [int(gpu.strip()) for gpu in args.gpu_ids.split(',')]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_list))
        effective_num_gpus = len(gpu_list)
        print(f"🎮 Using specified GPUs: {gpu_list}")
    else:
        effective_num_gpus = min(args.num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        if torch.cuda.is_available():
            gpu_list = list(range(effective_num_gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_list))
            print(f"🎮 Using first {effective_num_gpus} GPUs: {gpu_list}")
        else:
            print("🎮 CUDA not available, using CPU")
    
    # Determine sparsity values to test
    if args.mode == 'single':
        sparsity_values = [args.sparsity]
    elif args.mode == 'sweep':
        # Predefined comprehensive sweep
        sparsity_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    elif args.mode == 'range':
        if args.sparsity_list:
            # Parse comma-separated list
            sparsity_values = [float(s.strip()) for s in args.sparsity_list.split(',')]
        else:
            # Generate range
            sparsity_values = []
            current = args.sparsity_min
            while current <= args.sparsity_max:
                sparsity_values.append(current)
                current += args.sparsity_step
    
    print("🔍 GPU Seed Hunter with Canonical Standard")
    print("="*80)
    print(f"Mode: {args.mode.upper()}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Architectures: {args.num_architectures}")
    print(f"Seeds per arch: {args.seeds_per_arch}")
    print(f"Neuron sorting: {'Disabled' if args.disable_sorting else f'Every {args.sort_frequency} epochs'}")
    
    if args.mode == 'single':
        print(f"Sparsity: {args.sparsity:.1%}")
    else:
        print(f"Sparsity values: {[f'{s:.1%}' for s in sparsity_values]}")
        print(f"Total sparsity levels: {len(sparsity_values)}")
        print(f"Total experiments: {args.num_architectures * args.seeds_per_arch * len(sparsity_values)}")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hunter = GPUSeedHunter(
        num_gpus=effective_num_gpus, 
        device=device, 
        dataset=args.dataset, 
        save_threshold=args.thresh
    )
    
    # Store run args for checkpointer
    hunter._run_args = args
    
    # Store all results across sparsity levels
    all_sparsity_results = {}
    
    for sparsity in sparsity_values:
        print(f"\n🎯 Testing Sparsity Level: {sparsity:.1%}")
        print("-" * 50)
        
        # Update hunter's test implementation to use sorting settings
        hunter._disable_sorting = args.disable_sorting
        hunter._sort_frequency = args.sort_frequency
        
        results = hunter.gpu_saturated_search(
            num_architectures=args.num_architectures,
            seeds_per_arch=args.seeds_per_arch,
            sparsity=sparsity
        )
        
        all_sparsity_results[sparsity] = results
        
        # Print summary for this sparsity level
        best_acc = results['best_accuracy']
        print(f"\n📊 Sparsity {sparsity:.1%} Summary:")
        print(f"   Best accuracy: {best_acc['accuracy']:.2%} (arch: {best_acc['architecture']}, seed: {best_acc['seed']})")
    
    # Cross-sparsity analysis
    if len(sparsity_values) > 1:
        print(f"\n🔬 CROSS-SPARSITY ANALYSIS")
        print("="*50)
        
        # Find overall best across all sparsity levels
        all_results = []
        for sparsity, results in all_sparsity_results.items():
            for result in results['all_results']:
                result['tested_sparsity'] = sparsity
                all_results.append(result)
        
        # Overall best performers
        overall_best_acc = max(all_results, key=lambda x: x['accuracy'])
        overall_best_patch = max(all_results, key=lambda x: x['patchability'])
        overall_best_eff = max(all_results, key=lambda x: x['accuracy']/x['parameters'])
        
        print(f"\n🏆 Overall Best Accuracy: {overall_best_acc['accuracy']:.2%}")
        print(f"   Architecture: {overall_best_acc['architecture']}")
        print(f"   Seed: {overall_best_acc['seed']}")
        print(f"   Sparsity: {overall_best_acc['tested_sparsity']:.1%}")
        
        print(f"\n🎯 Overall Best Patchability: {overall_best_patch['patchability']:.3f}")
        print(f"   Architecture: {overall_best_patch['architecture']}")
        print(f"   Accuracy: {overall_best_patch['accuracy']:.2%}")
        print(f"   Sparsity: {overall_best_patch['tested_sparsity']:.1%}")
        
        print(f"\n⚡ Overall Best Efficiency: {overall_best_eff['accuracy']/overall_best_eff['parameters']*1000:.3f} acc/kparam")
        print(f"   Architecture: {overall_best_eff['architecture']}")
        print(f"   Accuracy: {overall_best_eff['accuracy']:.2%}")
        print(f"   Sparsity: {overall_best_eff['tested_sparsity']:.1%}")
        
        # Sparsity trend analysis
        print(f"\n📈 Sparsity Trend Analysis:")
        sparsity_summary = {}
        for sparsity in sorted(sparsity_values):
            results = all_sparsity_results[sparsity]
            avg_acc = np.mean([r['accuracy'] for r in results['all_results']])
            avg_patch = np.mean([r['patchability'] for r in results['all_results']])
            best_acc = results['best_accuracy']['accuracy']
            sparsity_summary[sparsity] = {
                'avg_accuracy': avg_acc,
                'best_accuracy': best_acc,
                'avg_patchability': avg_patch
            }
            print(f"   {sparsity:.1%}: avg_acc={avg_acc:.2%}, best_acc={best_acc:.2%}, avg_patch={avg_patch:.3f}")
        
        # Save comprehensive results
        results_dir = f'data/seed_hunt_results_{args.dataset}'
        os.makedirs(results_dir, exist_ok=True)
        
        comprehensive_results = {
            'mode': args.mode,
            'dataset': args.dataset,
            'sparsity_values': sparsity_values,
            'neuron_sorting': not args.disable_sorting,
            'sort_frequency': args.sort_frequency,
            'overall_best_accuracy': overall_best_acc,
            'overall_best_patchability': overall_best_patch,
            'overall_best_efficiency': overall_best_eff,
            'sparsity_summary': sparsity_summary,
            'detailed_results': all_sparsity_results,
            'canonical_standard_version': '1.0.0'
        }
        
        with open(f'{results_dir}/comprehensive_sparsity_sweep.json', 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        print(f"\n💾 Comprehensive results saved to {results_dir}/comprehensive_sparsity_sweep.json")
    
    else:
        # Single sparsity mode - use original result handling
        results = all_sparsity_results[sparsity_values[0]]
    
    print("\n🎊 BEST SEEDS SUMMARY")
    print("="*50)
    
    best = results['best_patchable']
    print(f"🎯 Best for patching:")
    print(f"   Architecture: {best['architecture']}")
    print(f"   Seed: {best['seed']}")
    print(f"   Initial accuracy: {best['accuracy']:.2%}")
    print(f"   Patchability score: {best['patchability']:.3f}")
    print(f"   Parameters: {best['parameters']:,}")
    
    best_acc = results['best_accuracy']
    print(f"\n🏆 Best accuracy:")
    print(f"   Architecture: {best_acc['architecture']}")
    print(f"   Seed: {best_acc['seed']}")
    print(f"   Accuracy: {best_acc['accuracy']:.2%}")
    
    best_eff = results['best_efficient']
    print(f"\n⚡ Most efficient:")
    print(f"   Architecture: {best_eff['architecture']}")
    print(f"   Seed: {best_eff['seed']}")
    print(f"   Efficiency: {best_eff['accuracy']/best_eff['parameters']*1000:.3f} acc/kparam")
    
    # Save results
    results_dir = f'data/seed_hunt_results_{args.dataset}'
    os.makedirs(results_dir, exist_ok=True)
    with open(f'{results_dir}/gpu_saturated_results.json', 'w') as f:
        json.dump({
            'mode': args.mode,
            'dataset': args.dataset,
            'sparsity': args.sparsity,
            'best_accuracy': results['best_accuracy'],
            'best_patchable': results['best_patchable'],
            'best_efficient': results['best_efficient'],
            'summary_stats': {
                'total_experiments': len(results['all_results']),
                'avg_accuracy': np.mean([r['accuracy'] for r in results['all_results']]),
                'avg_patchability': np.mean([r['patchability'] for r in results['all_results']]),
            },
            'canonical_standard_version': '1.0.0'
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {results_dir}/gpu_saturated_results.json")
    print("\n🎉 Seed hunt completed using canonical standard!")
    print("✅ All models saved in canonical format for perfect compatibility.")


if __name__ == "__main__":
    main()
