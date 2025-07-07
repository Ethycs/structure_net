#!/usr/bin/env python3
"""
Model Validation and Cleanup Script

Uses the canonical model_io functions to validate and clean up models in the data directory.
"""

import argparse
from src.structure_net.core.model_io import cleanup_data_directory

def main():
    parser = argparse.ArgumentParser(description='Validate and cleanup models using canonical standard')
    parser.add_argument('--data-dir', default='data', help='Directory containing models to validate')
    parser.add_argument('--min-accuracy', type=float, default=0.30, help='Minimum acceptable accuracy')
    parser.add_argument('--max-dead-ratio', type=float, default=0.5, help='Maximum acceptable dead neuron ratio')
    parser.add_argument('--live', action='store_true', help='Actually delete files (default is dry run)')
    parser.add_argument('--device', default='cpu', help='Device to use for model loading')
    
    args = parser.parse_args()
    
    print("🔬 MODEL VALIDATION AND CLEANUP")
    print("=" * 50)
    print(f"Using canonical model_io functions")
    print(f"Data directory: {args.data_dir}")
    print(f"Minimum accuracy: {args.min_accuracy:.1%}")
    print(f"Maximum dead ratio: {args.max_dead_ratio:.1%}")
    print(f"Mode: {'LIVE DELETION' if args.live else 'DRY RUN'}")
    print(f"Device: {args.device}")
    
    # Run the canonical cleanup function
    results = cleanup_data_directory(
        data_dir=args.data_dir,
        min_accuracy=args.min_accuracy,
        max_dead_ratio=args.max_dead_ratio,
        dry_run=not args.live,
        device=args.device
    )
    
    print(f"\n🎯 FINAL SUMMARY")
    print("=" * 30)
    validation = results['validation_results']
    deletion = results['deletion_results']
    
    print(f"Total models found: {validation['total_files']}")
    print(f"Good models: {validation['good_models']}")
    print(f"Bad models: {validation['bad_models']}")
    print(f"Models {'deleted' if args.live else 'would be deleted'}: {deletion['deleted_count']}")
    print(f"Space {'freed' if args.live else 'would be freed'}: {deletion['freed_space_mb']:.1f} MB")
    
    if not args.live and deletion['deleted_count'] > 0:
        print(f"\n💡 To actually delete the files, run with --live flag")
    
    return results

if __name__ == "__main__":
    main()
