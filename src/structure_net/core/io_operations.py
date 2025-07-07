#!/usr/bin/env python3
"""
I/O Operations for Model Persistence

This module handles saving and loading of models using the canonical standard.
All I/O operations ensure perfect compatibility across different systems.
"""

import torch
import torch.nn as nn
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from .layers import StandardSparseLayer
from .network_factory import create_standard_network
from .network_analysis import get_network_stats


def save_model_seed(model: nn.Sequential,
                   architecture: List[int],
                   seed: int,
                   metrics: Dict[str, Any],
                   filepath: str,
                   optimizer: Optional[torch.optim.Optimizer] = None) -> str:
    """
    THE canonical save function - enforces standard format.
    
    This function only accepts models built with create_standard_network()
    and always saves in the standard format with guaranteed key structure.
    
    Args:
        model: Network created with create_standard_network()
        architecture: Original architecture specification
        seed: Random seed used to create the model
        metrics: Performance and training metrics
        filepath: Path to save the checkpoint
        optimizer: Optional optimizer state to save
        
    Returns:
        Actual filepath where model was saved
    """
    # Validate model structure
    if not isinstance(model, nn.Sequential):
        raise ValueError("Model must be nn.Sequential created with create_standard_network()")
    
    # Verify all sparse layers are StandardSparseLayer
    sparse_layers = [layer for layer in model if isinstance(layer, StandardSparseLayer)]
    if len(sparse_layers) != len(architecture) - 1:
        raise ValueError(f"Model structure mismatch: expected {len(architecture)-1} sparse layers, got {len(sparse_layers)}")
    
    # Get network statistics
    network_stats = get_network_stats(model)
    
    # Create comprehensive checkpoint
    checkpoint = {
        # Model state (THE standard format)
        'model_state_dict': model.state_dict(),
        'architecture': architecture,
        'seed': seed,
        'sparsity': metrics.get('sparsity', network_stats['overall_sparsity']),
        
        # Training state
        'epoch': metrics.get('epoch', 0),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        
        # Performance metrics
        'accuracy': metrics.get('accuracy', 0.0),
        'patchability_score': metrics.get('patchability', 0.0),
        'extrema_counts': metrics.get('extrema_score', 0.0),
        'efficiency': metrics.get('accuracy', 0.0) / network_stats['total_parameters'],
        
        # Network statistics
        'network_stats': network_stats,
        
        # Training settings
        'neuron_sorting_enabled': metrics.get('sorted', True),
        'sort_frequency': metrics.get('sort_frequency', 5),
        'training_epochs': metrics.get('epochs', 15),
        
        # Neuron analysis
        'dead_neurons': metrics.get('dead_neurons', 0),
        'saturated_neurons': metrics.get('saturated_neurons', 0),
        'activation_patterns': metrics.get('activation_patterns'),
        
        # Reproducibility
        'torch_version': torch.__version__,
        'random_state': torch.get_rng_state(),
        'cuda_random_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        
        # Metadata
        'save_timestamp': datetime.now().isoformat(),
        'model_io_version': '1.0.0'
    }
    
    # Ensure directory exists (only if filepath has a directory component)
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    
    print(f"✅ Saved standard model: {filepath}")
    print(f"   Architecture: {architecture}")
    print(f"   Connections: {network_stats['total_connections']:,}")
    print(f"   Sparsity: {network_stats['overall_sparsity']:.1%}")
    print(f"   Accuracy: {metrics.get('accuracy', 0.0):.2%}")
    
    return filepath


def load_model_seed(filepath: str, 
                   device: str = 'cpu') -> Tuple[nn.Sequential, Dict[str, Any]]:
    """
    THE canonical load function - guaranteed compatibility.
    
    This function always rebuilds models using create_standard_network()
    to ensure perfect compatibility regardless of how they were originally saved.
    
    Args:
        filepath: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (model, checkpoint_data)
    """
    print(f"🔄 Loading standard model: {filepath}")
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    # Extract essential information
    architecture = checkpoint['architecture']
    sparsity = checkpoint.get('sparsity', 0.02)
    seed = checkpoint.get('seed', None)
    
    print(f"   Architecture: {architecture}")
    print(f"   Sparsity: {sparsity:.1%}")
    print(f"   Seed: {seed}")
    
    # Rebuild model using THE standard factory
    model = create_standard_network(
        architecture=architecture,
        sparsity=sparsity,
        seed=seed,
        device=device
    )
    
    # Load state dict (guaranteed to match because we used standard factory)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✅ State dict loaded successfully")
    except Exception as e:
        print(f"   ❌ State dict loading failed: {e}")
        print(f"   🔧 Attempting key mapping for legacy compatibility...")
        
        # Legacy compatibility: try to map old keys to standard keys
        state_dict = checkpoint['model_state_dict']
        mapped_state_dict = {}
        
        # Map keys from various legacy formats to standard format
        for old_key, value in state_dict.items():
            # Standard format: already correct
            if old_key in model.state_dict():
                mapped_state_dict[old_key] = value
            # Legacy format mapping
            elif '.linear.' in old_key:
                # Already in correct format
                mapped_state_dict[old_key] = value
            else:
                # Try to map other formats
                if old_key.endswith('.weight') and not '.linear.' in old_key:
                    new_key = old_key.replace('.weight', '.linear.weight')
                    if new_key in model.state_dict():
                        mapped_state_dict[new_key] = value
                elif old_key.endswith('.bias') and not '.linear.' in old_key:
                    new_key = old_key.replace('.bias', '.linear.bias')
                    if new_key in model.state_dict():
                        mapped_state_dict[new_key] = value
                else:
                    mapped_state_dict[old_key] = value
        
        model.load_state_dict(mapped_state_dict)
        print(f"   ✅ Legacy state dict mapped and loaded")
    
    # Verify network statistics match
    network_stats = get_network_stats(model)
    expected_connections = checkpoint.get('network_stats', {}).get('total_connections')
    if expected_connections and network_stats['total_connections'] != expected_connections:
        print(f"   ⚠️  Connection count mismatch: expected {expected_connections}, got {network_stats['total_connections']}")
    else:
        print(f"   ✅ Network statistics verified")
    
    return model, checkpoint


def test_save_load_compatibility(test_architectures: Optional[List[List[int]]] = None,
                               test_sparsities: Optional[List[float]] = None,
                               device: str = 'cpu') -> bool:
    """
    THE canonical compatibility test.
    
    This function verifies that save/load operations preserve model behavior
    exactly, ensuring round-trip fidelity.
    
    Args:
        test_architectures: List of architectures to test
        test_sparsities: List of sparsity levels to test
        device: Device to run tests on
        
    Returns:
        True if all tests pass, False otherwise
    """
    print("🧪 Testing save/load compatibility...")
    
    # Default test cases
    if test_architectures is None:
        test_architectures = [
            [784, 10],           # Simple direct connection
            [784, 128, 10],      # Single hidden layer
            [3072, 256, 128, 10] # Multi-layer (CIFAR-10 style)
        ]
    
    if test_sparsities is None:
        test_sparsities = [0.01, 0.02, 0.05]
    
    all_tests_passed = True
    test_count = 0
    
    for arch in test_architectures:
        for sparsity in test_sparsities:
            test_count += 1
            print(f"\n  Test {test_count}: {arch} @ {sparsity:.1%} sparsity")
            
            try:
                # Create original model
                seed = 42 + test_count  # Unique seed per test
                original_model = create_standard_network(arch, sparsity, seed, device)
                
                # Generate test input
                batch_size = 10
                test_input = torch.randn(batch_size, arch[0], device=device)
                
                # Get original output
                original_model.eval()
                with torch.no_grad():
                    original_output = original_model(test_input)
                
                # Save model
                test_dir = "test_checkpoints"
                os.makedirs(test_dir, exist_ok=True)
                test_filepath = os.path.join(test_dir, f"test_model_{test_count}.pt")
                
                test_metrics = {
                    'accuracy': 0.85,
                    'sparsity': sparsity,
                    'epoch': 10
                }
                
                save_model_seed(original_model, arch, seed, test_metrics, test_filepath)
                
                # Load model
                loaded_model, checkpoint_data = load_model_seed(test_filepath, device)
                
                # Get loaded output
                loaded_model.eval()
                with torch.no_grad():
                    loaded_output = loaded_model(test_input)
                
                # Compare outputs
                max_diff = torch.max(torch.abs(original_output - loaded_output)).item()
                
                if max_diff < 1e-6:
                    print(f"    ✅ PASS: Max difference {max_diff:.2e}")
                else:
                    print(f"    ❌ FAIL: Max difference {max_diff:.2e} (threshold: 1e-6)")
                    all_tests_passed = False
                
                # Verify metadata
                if checkpoint_data['architecture'] != arch:
                    print(f"    ❌ FAIL: Architecture mismatch")
                    all_tests_passed = False
                
                if abs(checkpoint_data['sparsity'] - sparsity) > 1e-6:
                    print(f"    ❌ FAIL: Sparsity mismatch")
                    all_tests_passed = False
                
                # Clean up
                os.remove(test_filepath)
                
            except Exception as e:
                print(f"    ❌ FAIL: Exception {e}")
                all_tests_passed = False
    
    # Clean up test directory
    try:
        os.rmdir("test_checkpoints")
    except:
        pass
    
    print(f"\n🎯 Compatibility Test Results:")
    print(f"   Tests run: {test_count}")
    print(f"   Status: {'✅ ALL PASSED' if all_tests_passed else '❌ SOME FAILED'}")
    
    return all_tests_passed


# Export I/O functions
__all__ = [
    'save_model_seed',
    'load_model_seed',
    'test_save_load_compatibility'
]
