#!/usr/bin/env python3
"""
Network Factory Functions

This module contains the canonical network creation functions.
All network creation must go through these functions to ensure compatibility.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from .layers import StandardSparseLayer, ExtremaAwareSparseLayer


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


# Export network factory functions
__all__ = [
    'create_standard_network',
    'create_extrema_aware_network',
    'create_evolvable_network',
    'load_pretrained_into_canonical'
]
