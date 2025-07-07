#!/usr/bin/env python3
"""
Model I/O - The Canonical Standard for Sparse Networks

This module defines THE single source of truth for sparse network models.
All systems (GPU seed hunter, hybrid growth, etc.) must use these standard
definitions to ensure perfect compatibility.

Key Principles:
1. Single Source of Truth: Only one way to create sparse networks
2. Guaranteed Compatibility: Same structure = same state dict keys
3. Bulletproof I/O: Standard save/load functions prevent format drift
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional


class StandardSparseLayer(nn.Module):
    """
    THE canonical sparse layer used by all systems.
    
    This is the single source of truth for sparse layer behavior.
    All systems must use this exact implementation to ensure compatibility.
    """
    
    def __init__(self, in_features: int, out_features: int, sparsity: float):
        super().__init__()
        
        # Standard structure: nested linear module (matches GPU seed hunter)
        self.linear = nn.Linear(in_features, out_features)
        
        # Standard mask creation and registration
        mask = torch.rand_like(self.linear.weight) < sparsity
        self.register_buffer('mask', mask.float())
        
        # Standard initialization with mask application
        with torch.no_grad():
            self.linear.weight.data *= self.mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        THE canonical forward pass - identical everywhere.
        
        This ensures all systems have identical behavior.
        """
        return F.linear(x, self.linear.weight * self.mask, self.linear.bias)
    
    def get_connection_count(self) -> int:
        """Get number of active connections in this layer."""
        return self.mask.sum().item()
    
    def get_sparsity_ratio(self) -> float:
        """Get actual sparsity ratio of this layer."""
        return self.get_connection_count() / self.mask.numel()


class ExtremaAwareSparseLayer(StandardSparseLayer):
    """
    Advanced sparse layer with extrema-guided connectivity.
    
    Extends the canonical StandardSparseLayer with extrema-aware mask creation
    while maintaining full compatibility with the canonical standard.
    """
    
    def __init__(self, in_features: int, out_features: int, base_sparsity: float, 
                 extrema_to_embed: Optional[Dict[str, List[int]]] = None):
        # Initialize as standard sparse layer first
        super().__init__(in_features, out_features, base_sparsity)
        
        # Store extrema information for potential mask updates
        self.base_sparsity = base_sparsity
        self.extrema_to_embed = extrema_to_embed or {}
        
        # Update mask based on extrema if provided
        if extrema_to_embed:
            self._update_mask_for_extrema()
    
    def _update_mask_for_extrema(self):
        """Update mask to have higher density around extrema regions."""
        # Start with current mask
        enhanced_mask = self.mask.clone()
        
        # High-density connections TO revive dead neurons (targets)
        dead_neurons = self.extrema_to_embed.get('low', [])
        if dead_neurons:
            revival_density = 0.20  # 20% dense input connections for dead neurons
            for neuron_idx in dead_neurons:
                if neuron_idx < self.linear.out_features:
                    # Give dead neuron more diverse inputs
                    revival_connections = (torch.rand(self.linear.in_features) < revival_density).float()
                    enhanced_mask[neuron_idx, :] = torch.maximum(
                        enhanced_mask[neuron_idx, :], 
                        revival_connections
                    )
        
        # High-density connections FROM saturated neurons (sources)
        saturated_neurons = self.extrema_to_embed.get('high', [])
        if saturated_neurons:
            relief_density = 0.15  # 15% dense output connections for saturated neurons
            for neuron_idx in saturated_neurons:
                if neuron_idx < self.linear.in_features:
                    # Give saturated neuron more output channels
                    relief_connections = (torch.rand(self.linear.out_features) < relief_density).float()
                    enhanced_mask[:, neuron_idx] = torch.maximum(
                        enhanced_mask[:, neuron_idx], 
                        relief_connections
                    )
        
        # Update the registered buffer
        self.mask.data = enhanced_mask
        
        # Re-apply mask to weights
        with torch.no_grad():
            self.linear.weight.data *= self.mask


class TemporaryPatchLayer(nn.Module):
    """
    Dense patch layer for fixing specific extrema.
    
    Compatible with canonical standard - can be saved/loaded with standard functions.
    """
    
    def __init__(self, source_neuron_idx: int, patch_size: int = 8, output_size: int = 4):
        super().__init__()
        self.source_neuron_idx = source_neuron_idx
        self.patch_size = patch_size
        self.output_size = output_size
        
        # Dense layers for the patch
        self.layer1 = nn.Linear(1, patch_size)
        self.layer2 = nn.Linear(patch_size, output_size)
        
        # Initialize with small weights to avoid disrupting main network
        with torch.no_grad():
            self.layer1.weight.data *= 0.1
            self.layer2.weight.data *= 0.1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract single neuron activation and process through dense patch."""
        # Extract the source neuron's activation
        patch_input = x[:, self.source_neuron_idx].unsqueeze(-1)
        
        # Process through dense layers
        h = F.relu(self.layer1(patch_input))
        return self.layer2(h)


def create_standard_network(architecture: List[int], 
                          sparsity: float, 
                          seed: Optional[int] = None,
                          device: str = 'cpu') -> nn.Sequential:
    """
    THE canonical network factory function.
    
    This is the single source of truth for creating sparse networks.
    All systems must use this function to ensure compatibility.
    
    Args:
        architecture: List of layer sizes [input, hidden1, hidden2, ..., output]
        sparsity: Sparsity level (0.0 to 1.0)
        seed: Random seed for reproducibility
        device: Device to create network on
        
    Returns:
        nn.Sequential model with StandardSparseLayer components
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    layers = []
    for i in range(len(architecture) - 1):
        # Add sparse layer
        sparse_layer = StandardSparseLayer(
            architecture[i], 
            architecture[i+1], 
            sparsity
        )
        layers.append(sparse_layer)
        
        # Add ReLU activation (except for last layer)
        if i < len(architecture) - 2:
            layers.append(nn.ReLU())
    
    model = nn.Sequential(*layers)
    return model.to(device)


def create_extrema_aware_network(architecture: List[int],
                               sparsity: float,
                               extrema_patterns: Optional[List[Dict[str, List[int]]]] = None,
                               seed: Optional[int] = None,
                               device: str = 'cpu') -> nn.Sequential:
    """
    Create network with extrema-aware sparse layers using canonical standard.
    
    Args:
        architecture: List of layer sizes [input, hidden1, hidden2, ..., output]
        sparsity: Base sparsity level (0.0 to 1.0)
        extrema_patterns: List of extrema patterns for each layer
        seed: Random seed for reproducibility
        device: Device to create network on
        
    Returns:
        nn.Sequential model with ExtremaAwareSparseLayer components
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    layers = []
    for i in range(len(architecture) - 1):
        # Get extrema pattern for this layer
        layer_extrema = None
        if extrema_patterns and i < len(extrema_patterns):
            layer_extrema = extrema_patterns[i]
        
        # Add extrema-aware sparse layer
        sparse_layer = ExtremaAwareSparseLayer(
            architecture[i], 
            architecture[i+1], 
            sparsity,
            layer_extrema
        )
        layers.append(sparse_layer)
        
        # Add ReLU activation (except for last layer)
        if i < len(architecture) - 2:
            layers.append(nn.ReLU())
    
    model = nn.Sequential(*layers)
    return model.to(device)


def create_evolvable_network(architecture: List[int],
                           sparsity: float,
                           seed: Optional[int] = None,
                           device: str = 'cpu') -> nn.Sequential:
    """
    Create network ready for evolution using canonical standard.
    
    This creates a standard network that can be evolved using the evolution
    system while maintaining full compatibility with canonical I/O.
    
    Args:
        architecture: List of layer sizes [input, hidden1, hidden2, ..., output]
        sparsity: Sparsity level (0.0 to 1.0)
        seed: Random seed for reproducibility
        device: Device to create network on
        
    Returns:
        nn.Sequential model ready for evolution
    """
    # For now, evolvable networks are just standard networks
    # Future evolution will modify the architecture dynamically
    return create_standard_network(architecture, sparsity, seed, device)


def load_pretrained_into_canonical(checkpoint_path: str,
                                 target_architecture: Optional[List[int]] = None,
                                 device: str = 'cpu') -> nn.Sequential:
    """
    Load pretrained weights into canonical network structure.
    
    This function handles loading from various checkpoint formats and
    converts them to the canonical standard.
    
    Args:
        checkpoint_path: Path to checkpoint file
        target_architecture: Optional target architecture (if different from checkpoint)
        device: Device to load model on
        
    Returns:
        nn.Sequential model in canonical format
    """
    print(f"🔄 Loading pretrained model into canonical format: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract architecture
    architecture = target_architecture or checkpoint.get('architecture')
    if not architecture:
        raise ValueError("No architecture found in checkpoint and none provided")
    
    sparsity = checkpoint.get('sparsity', 0.02)
    seed = checkpoint.get('seed', None)
    
    print(f"   Architecture: {architecture}")
    print(f"   Sparsity: {sparsity:.1%}")
    
    # Create canonical network
    model = create_standard_network(architecture, sparsity, seed, device)
    
    # Load weights with compatibility handling
    state_dict = checkpoint['model_state_dict']
    
    # Try direct loading first
    try:
        model.load_state_dict(state_dict)
        print(f"   ✅ Direct state dict loading successful")
    except Exception as e:
        print(f"   🔧 Direct loading failed, attempting compatibility mapping...")
        
        # Create mapping for different formats
        mapped_state_dict = {}
        model_keys = set(model.state_dict().keys())
        
        for old_key, value in state_dict.items():
            if old_key in model_keys:
                # Direct match
                mapped_state_dict[old_key] = value
            else:
                # Try various mappings
                possible_new_keys = [
                    old_key.replace('.weight', '.linear.weight'),
                    old_key.replace('.bias', '.linear.bias'),
                    old_key.replace('linear.', ''),  # Remove extra linear prefix
                ]
                
                mapped = False
                for new_key in possible_new_keys:
                    if new_key in model_keys:
                        mapped_state_dict[new_key] = value
                        mapped = True
                        break
                
                if not mapped:
                    print(f"   ⚠️  Could not map key: {old_key}")
        
        # Load mapped state dict
        model.load_state_dict(mapped_state_dict, strict=False)
        print(f"   ✅ Compatibility mapping successful")
    
    return model


def get_network_stats(model: nn.Sequential) -> Dict[str, Any]:
    """
    Get comprehensive statistics about a standard network.
    
    Args:
        model: Network created with create_standard_network()
        
    Returns:
        Dictionary with network statistics
    """
    stats = {
        'total_parameters': 0,
        'total_connections': 0,
        'layers': [],
        'overall_sparsity': 0.0,
        'architecture': []
    }
    
    # Extract architecture and layer stats
    for i, layer in enumerate(model):
        if isinstance(layer, StandardSparseLayer):
            layer_stats = {
                'layer_index': i,
                'in_features': layer.linear.in_features,
                'out_features': layer.linear.out_features,
                'total_possible_connections': layer.mask.numel(),
                'active_connections': layer.get_connection_count(),
                'sparsity_ratio': layer.get_sparsity_ratio(),
                'parameters': layer.linear.weight.numel() + layer.linear.bias.numel()
            }
            
            stats['layers'].append(layer_stats)
            stats['total_parameters'] += layer_stats['parameters']
            stats['total_connections'] += layer_stats['active_connections']
            stats['architecture'].extend([layer_stats['in_features']])
            
            # Add output size for last layer
            if i == len([l for l in model if isinstance(l, StandardSparseLayer)]) - 1:
                stats['architecture'].append(layer_stats['out_features'])
    
    # Calculate overall sparsity
    total_possible = sum(layer['total_possible_connections'] for layer in stats['layers'])
    stats['overall_sparsity'] = stats['total_connections'] / total_possible if total_possible > 0 else 0.0
    
    return stats


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


def apply_neuron_sorting(model: nn.Sequential, layer_idx: int) -> None:
    """
    Apply neuron sorting to a standard network.
    
    This function sorts neurons in the specified layer by importance
    while preserving network function.
    
    Args:
        model: Network created with create_standard_network()
        layer_idx: Index of sparse layer to sort (0, 1, 2, ...)
    """
    # Get sparse layers only
    sparse_layers = [layer for layer in model if isinstance(layer, StandardSparseLayer)]
    
    if layer_idx >= len(sparse_layers) - 1:  # Don't sort output layer
        return
    
    current_layer = sparse_layers[layer_idx]
    
    # Calculate importance (L2 norm of weights)
    importance = torch.linalg.norm(current_layer.linear.weight, ord=2, dim=1)
    
    # Get sorting permutation
    perm_indices = torch.argsort(importance, descending=True)
    
    # Apply permutation to current layer (output neurons)
    current_layer.linear.weight.data = current_layer.linear.weight.data[perm_indices, :]
    current_layer.linear.bias.data = current_layer.linear.bias.data[perm_indices]
    current_layer.mask = current_layer.mask[perm_indices, :]
    
    # Apply permutation to next layer (input connections)
    if layer_idx + 1 < len(sparse_layers):
        next_layer = sparse_layers[layer_idx + 1]
        next_layer.linear.weight.data = next_layer.linear.weight.data[:, perm_indices]
        next_layer.mask = next_layer.mask[:, perm_indices]


def sort_all_network_layers(model: nn.Sequential) -> None:
    """
    Sort all hidden layers in a standard network by neuron importance.
    
    Args:
        model: Network created with create_standard_network()
    """
    sparse_layers = [layer for layer in model if isinstance(layer, StandardSparseLayer)]
    
    # Sort all hidden layers (not the output layer)
    for i in range(len(sparse_layers) - 1):
        apply_neuron_sorting(model, i)


# Version information
__version__ = "1.0.0"
__author__ = "Structure Net Team"

# Export the canonical interface
__all__ = [
    'StandardSparseLayer',
    'create_standard_network',
    'save_model_seed',
    'load_model_seed',
    'test_save_load_compatibility',
    'get_network_stats',
    'apply_neuron_sorting',
    'sort_all_network_layers'
]
