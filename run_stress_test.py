#!/usr/bin/env python3
"""
Simple runner for the Ultimate Structure Net Stress Test

This script provides an easy way to run the comprehensive stress test
with different configurations and safety checks.
"""

import sys
import os

# IMPORTANT: Set spawn method BEFORE importing torch
import multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

import torch
import argparse
from datetime import datetime

def check_system_requirements():
    """Check if system meets requirements for stress test."""
    print("🔍 Checking system requirements...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available, will run on CPU (very slow)")
        return False
    
    # Check GPU memory
    total_memory = 0
    gpu_count = torch.cuda.device_count()
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        total_memory += memory_gb
        print(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")
    
    print(f"   Total GPU memory: {total_memory:.1f}GB")
    
    if total_memory < 4:
        print("⚠️  Warning: Low GPU memory, consider reducing batch size")
        return False
    
    print("✅ System requirements check passed")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run Ultimate Structure Net Stress Test")
    parser.add_argument("--quick", action="store_true", help="Run quick test (smaller scale)")
    parser.add_argument("--full", action="store_true", help="Run full stress test")
    parser.add_argument("--tournament-size", type=int, default=None, help="Override tournament size")
    parser.add_argument("--generations", type=int, default=None, help="Override number of generations")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per generation")
    parser.add_argument("--no-growth", action="store_true", help="Disable network growth")
    parser.add_argument("--no-residual", action="store_true", help="Disable residual blocks")
    parser.add_argument("--no-metrics", action="store_true", help="Disable comprehensive metrics")
    parser.add_argument("--no-profiling", action="store_true", help="Disable profiling")
    
    args = parser.parse_args()
    
    print("🚀 Ultimate Structure Net Stress Test Runner")
    print("=" * 60)
    
    # Check system requirements
    if not check_system_requirements():
        response = input("\nSystem requirements not optimal. Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Import the stress test (after system check)
    try:
        from experiments.ultimate_stress_test import StressTestConfig, ArchitectureTournament, calculate_optimal_memory_usage
    except ImportError as e:
        print(f"❌ Failed to import stress test: {e}")
        print("Make sure you're running from the structure_net directory")
        return
    
    # Calculate optimal configuration
    memory_info = calculate_optimal_memory_usage()
    
    # Create configuration based on arguments
    if args.quick:
        print("🏃 Running QUICK stress test")
        config = StressTestConfig(
            tournament_size=args.tournament_size or 16,
            generations=args.generations or 3,
            epochs_per_generation=args.epochs or 10,
            enable_growth=not args.no_growth,
            enable_residual_blocks=not args.no_residual,
            enable_comprehensive_metrics=not args.no_metrics,
            enable_profiling=not args.no_profiling
        )
    elif args.full:
        print("🔥 Running FULL stress test")
        config = StressTestConfig(
            tournament_size=args.tournament_size or memory_info.get('recommended_tournament_size', 64),
            generations=args.generations or 10,
            epochs_per_generation=args.epochs or 20,
            enable_growth=not args.no_growth,
            enable_residual_blocks=not args.no_residual,
            enable_comprehensive_metrics=not args.no_metrics,
            enable_profiling=not args.no_profiling
        )
    else:
        print("⚖️  Running BALANCED stress test")
        config = StressTestConfig(
            tournament_size=args.tournament_size or min(32, memory_info.get('recommended_tournament_size', 32)),
            generations=args.generations or 5,
            epochs_per_generation=args.epochs or 15,
            enable_growth=not args.no_growth,
            enable_residual_blocks=not args.no_residual,
            enable_comprehensive_metrics=not args.no_metrics,
            enable_profiling=not args.no_profiling
        )
    
    print(f"\n⚙️  Configuration:")
    print(f"   Tournament size: {config.tournament_size}")
    print(f"   Generations: {config.generations}")
    print(f"   Epochs per generation: {config.epochs_per_generation}")
    print(f"   Growth enabled: {config.enable_growth}")
    print(f"   Residual blocks: {config.enable_residual_blocks}")
    print(f"   Comprehensive metrics: {config.enable_comprehensive_metrics}")
    print(f"   Profiling: {config.enable_profiling}")
    
    # Estimate runtime
    estimated_experiments = config.tournament_size * config.generations
    estimated_minutes = estimated_experiments * config.epochs_per_generation / 60  # Rough estimate
    
    print(f"\n⏱️  Estimated runtime: {estimated_minutes:.1f} minutes")
    print(f"   Total experiments: {estimated_experiments}")
    
    # Confirm before starting
    response = input(f"\nStart stress test? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Run the stress test
    print(f"\n🚀 Starting stress test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        tournament = ArchitectureTournament(config)
        results = tournament.run_tournament()
        
        if results:
            print("\n🎉 Stress test completed successfully!")
            print(f"   Results saved to: stress_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        else:
            print("\n💥 Stress test failed!")
            
    except KeyboardInterrupt:
        print("\n⚠️  Stress test interrupted by user")
    except Exception as e:
        print(f"\n❌ Stress test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
