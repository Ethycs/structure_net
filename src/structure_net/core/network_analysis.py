#!/usr/bin/env python3
"""
Network Analysis and Statistics

This module provides functions for analyzing network structure,
performance, and health metrics.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from .layers import StandardSparseLayer


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
            
            # Build architecture properly: add input size for first layer only
            if len(stats['architecture']) == 0:
                stats['architecture'].append(layer_stats['in_features'])
            
            # Always add output size for each layer
            stats['architecture'].append(layer_stats['out_features'])
    
    # Calculate overall sparsity
    total_possible = sum(layer['total_possible_connections'] for layer in stats['layers'])
    stats['overall_sparsity'] = stats['total_connections'] / total_possible if total_possible > 0 else 0.0
    
    return stats


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


# Export analysis functions
__all__ = [
    'get_network_stats',
    'apply_neuron_sorting',
    'sort_all_network_layers'
]
