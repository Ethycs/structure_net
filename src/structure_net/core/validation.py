#!/usr/bin/env python3
"""
Model Validation and Quality Assessment

This module provides functions for validating model quality,
cleaning up invalid models, and maintaining data directory hygiene.
"""

import torch
import torch.nn as nn
import os
import glob
from datetime import datetime
from typing import Dict, Any, List
from .io_operations import load_model_seed
from .network_analysis import get_network_stats


def validate_model_quality(model: nn.Sequential, 
                          checkpoint_data: Dict[str, Any],
                          min_accuracy: float = 0.30,
                          max_dead_ratio: float = 0.5) -> Dict[str, Any]:
    """
    Validate model quality using canonical standards.
    
    Args:
        model: Network loaded with load_model_seed()
        checkpoint_data: Checkpoint data from load_model_seed()
        min_accuracy: Minimum acceptable accuracy
        max_dead_ratio: Maximum acceptable ratio of dead neurons
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'is_valid': True,
        'score': 0.0,
        'issues': [],
        'warnings': [],
        'metrics': {}
    }
    
    # Extract basic metrics
    accuracy = checkpoint_data.get('accuracy', 0.0)
    architecture = checkpoint_data.get('architecture', [])
    network_stats = get_network_stats(model)
    
    validation['metrics'] = {
        'accuracy': accuracy,
        'architecture': architecture,
        'total_parameters': network_stats['total_parameters'],
        'total_connections': network_stats['total_connections'],
        'sparsity': network_stats['overall_sparsity']
    }
    
    # Quality checks
    score = 0.0
    
    # 1. Accuracy check
    if accuracy < min_accuracy:
        validation['issues'].append(f"Low accuracy: {accuracy:.2%} < {min_accuracy:.2%}")
        validation['is_valid'] = False
    else:
        score += accuracy * 50  # Up to 50 points for accuracy
    
    # 2. Architecture sanity
    if len(architecture) < 2:
        validation['issues'].append(f"Invalid architecture: {architecture}")
        validation['is_valid'] = False
    elif architecture[0] not in [784, 3072]:
        validation['warnings'].append(f"Unusual input size: {architecture[0]}")
    else:
        score += 10  # 10 points for valid architecture
    
    # 3. Network efficiency
    if network_stats['total_parameters'] > 0:
        efficiency = accuracy / (network_stats['total_parameters'] / 1000)
        validation['metrics']['efficiency'] = efficiency
        score += min(efficiency * 20, 20)  # Up to 20 points for efficiency
    
    # 4. Sparsity check
    sparsity = network_stats['overall_sparsity']
    if 0.01 <= sparsity <= 0.1:  # Good sparsity range
        score += 10
    elif sparsity > 0.5:
        validation['warnings'].append(f"Very high sparsity: {sparsity:.1%}")
    
    # 5. Dead neuron analysis (simplified)
    dead_neurons = checkpoint_data.get('dead_neurons', 0)
    if isinstance(dead_neurons, (int, float)) and dead_neurons > 0:
        dead_ratio = dead_neurons / network_stats['total_parameters']
        validation['metrics']['dead_ratio'] = dead_ratio
        if dead_ratio > max_dead_ratio:
            validation['issues'].append(f"Too many dead neurons: {dead_ratio:.1%}")
            validation['is_valid'] = False
        else:
            score += 10
    else:
        score += 5  # Partial credit if no dead neuron data
    
    validation['score'] = min(score, 100.0)
    
    return validation


def validate_models_in_directory(directory: str,
                                min_accuracy: float = 0.30,
                                max_dead_ratio: float = 0.5,
                                device: str = 'cpu') -> Dict[str, Any]:
    """
    Validate all models in a directory using canonical standards.
    
    Args:
        directory: Directory containing .pt model files
        min_accuracy: Minimum acceptable accuracy
        max_dead_ratio: Maximum acceptable ratio of dead neurons
        device: Device to load models on
        
    Returns:
        Dictionary with validation results for all models
    """
    print(f"🔍 Validating models in: {directory}")
    
    # Find all .pt files
    pattern = os.path.join(directory, "**", "*.pt")
    model_files = glob.glob(pattern, recursive=True)
    
    if not model_files:
        print(f"❌ No .pt files found in {directory}")
        return {'total_files': 0, 'results': []}
    
    print(f"📁 Found {len(model_files)} model files")
    
    results = []
    good_models = []
    bad_models = []
    
    for i, filepath in enumerate(model_files):
        print(f"\n📋 Validating {i+1}/{len(model_files)}: {os.path.basename(filepath)}")
        
        result = {
            'filepath': filepath,
            'filename': os.path.basename(filepath),
            'can_load': False,
            'validation': None,
            'decision': 'UNKNOWN'
        }
        
        try:
            # Try to load with canonical function
            model, checkpoint = load_model_seed(filepath, device)
            result['can_load'] = True
            
            # Validate quality
            validation = validate_model_quality(model, checkpoint, min_accuracy, max_dead_ratio)
            result['validation'] = validation
            
            if validation['is_valid']:
                result['decision'] = 'KEEP'
                good_models.append(result)
                print(f"   ✅ KEEP - Score: {validation['score']:.1f}, Acc: {validation['metrics']['accuracy']:.2%}")
            else:
                result['decision'] = 'DELETE'
                bad_models.append(result)
                issues = '; '.join(validation['issues'])
                print(f"   ❌ DELETE - {issues}")
            
        except Exception as e:
            result['validation'] = {'is_valid': False, 'issues': [f"Load failed: {str(e)}"]}
            result['decision'] = 'DELETE'
            bad_models.append(result)
            print(f"   ❌ DELETE - Cannot load: {str(e)}")
        
        results.append(result)
    
    summary = {
        'total_files': len(model_files),
        'good_models': len(good_models),
        'bad_models': len(bad_models),
        'results': results,
        'validation_criteria': {
            'min_accuracy': min_accuracy,
            'max_dead_ratio': max_dead_ratio
        }
    }
    
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 30)
    print(f"Total files: {summary['total_files']}")
    print(f"Good models: {summary['good_models']} ({summary['good_models']/summary['total_files']:.1%})")
    print(f"Bad models: {summary['bad_models']} ({summary['bad_models']/summary['total_files']:.1%})")
    
    return summary


def delete_invalid_models(validation_results: Dict[str, Any],
                         dry_run: bool = True) -> Dict[str, Any]:
    """
    Delete models that failed validation using canonical standards.
    
    Args:
        validation_results: Results from validate_models_in_directory()
        dry_run: If True, only show what would be deleted
        
    Returns:
        Dictionary with deletion results
    """
    bad_models = [r for r in validation_results['results'] if r['decision'] == 'DELETE']
    
    if not bad_models:
        print("✅ No bad models to delete!")
        return {'deleted_count': 0, 'freed_space_mb': 0}
    
    print(f"\n🗑️  {'DRY RUN - ' if dry_run else ''}DELETING INVALID MODELS")
    print("=" * 50)
    
    deleted_count = 0
    freed_space_mb = 0
    
    for result in bad_models:
        filepath = result['filepath']
        filename = result['filename']
        
        try:
            file_size = os.path.getsize(filepath)
            file_size_mb = file_size / 1_000_000
            
            if result['validation']:
                issues = '; '.join(result['validation']['issues'])
            else:
                issues = "Cannot load model"
            
            print(f"{'[DRY RUN] ' if dry_run else ''}🗑️  {filename}")
            print(f"   Size: {file_size_mb:.2f} MB")
            print(f"   Reason: {issues}")
            
            if not dry_run:
                os.remove(filepath)
                print(f"   ✅ Deleted")
            
            deleted_count += 1
            freed_space_mb += file_size_mb
            
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
    
    summary = {
        'deleted_count': deleted_count,
        'freed_space_mb': freed_space_mb,
        'dry_run': dry_run
    }
    
    print(f"\n📊 {'DRY RUN ' if dry_run else ''}DELETION SUMMARY")
    print("=" * 30)
    print(f"Files {'would be ' if dry_run else ''}deleted: {deleted_count}")
    print(f"Space {'would be ' if dry_run else ''}freed: {freed_space_mb:.1f} MB")
    
    return summary


def cleanup_data_directory(data_dir: str = "data",
                          min_accuracy: float = 0.30,
                          max_dead_ratio: float = 0.5,
                          dry_run: bool = True,
                          device: str = 'cpu') -> Dict[str, Any]:
    """
    THE canonical data cleanup function.
    
    Validates all models in the data directory and removes invalid ones.
    
    Args:
        data_dir: Directory to clean up
        min_accuracy: Minimum acceptable accuracy
        max_dead_ratio: Maximum acceptable ratio of dead neurons
        dry_run: If True, only show what would be deleted
        device: Device to load models on
        
    Returns:
        Dictionary with cleanup results
    """
    print(f"🧹 CANONICAL DATA DIRECTORY CLEANUP")
    print("=" * 50)
    print(f"Directory: {data_dir}")
    print(f"Min accuracy: {min_accuracy:.1%}")
    print(f"Max dead ratio: {max_dead_ratio:.1%}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE DELETION'}")
    
    # Validate all models
    validation_results = validate_models_in_directory(
        data_dir, min_accuracy, max_dead_ratio, device
    )
    
    # Delete invalid models
    deletion_results = delete_invalid_models(validation_results, dry_run)
    
    # Combined summary
    cleanup_summary = {
        'validation_results': validation_results,
        'deletion_results': deletion_results,
        'cleanup_timestamp': datetime.now().isoformat(),
        'parameters': {
            'data_dir': data_dir,
            'min_accuracy': min_accuracy,
            'max_dead_ratio': max_dead_ratio,
            'dry_run': dry_run
        }
    }
    
    print(f"\n🎯 CLEANUP COMPLETE")
    print(f"   Validated: {validation_results['total_files']} files")
    print(f"   Keeping: {validation_results['good_models']} models")
    print(f"   {'Would delete' if dry_run else 'Deleted'}: {deletion_results['deleted_count']} models")
    print(f"   {'Would free' if dry_run else 'Freed'}: {deletion_results['freed_space_mb']:.1f} MB")
    
    return cleanup_summary


# Export validation functions
__all__ = [
    'validate_model_quality',
    'validate_models_in_directory',
    'delete_invalid_models',
    'cleanup_data_directory'
]
