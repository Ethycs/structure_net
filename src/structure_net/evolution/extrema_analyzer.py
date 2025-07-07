#!/usr/bin/env python3
"""
Extrema Analyzer - Using Canonical Standard

This module provides extrema detection and analysis capabilities
for networks created with the canonical model standard.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from torch.utils.data import DataLoader

from ..core.model_io import StandardSparseLayer, get_network_stats


@torch.no_grad()
def analyze_layer_extrema(activations: torch.Tensor, 
                         dead_threshold: float = 0.01, 
                         saturated_threshold: float = 0.99) -> Dict[str, List[int]]:
    """
    Identifies dead and saturated neurons from their activations.
    
    Args:
        activations: Tensor of activations [batch_size, num_neurons]
        dead_threshold: Threshold below which neurons are considered dead
        saturated_threshold: Threshold above which neurons are considered saturated
        
    Returns:
        Dictionary with 'low' (dead) and 'high' (saturated) neuron indices
    """
    mean_activations = activations.mean(dim=0)
    
    dead_neurons = torch.where(mean_activations < dead_threshold)[0].tolist()
    saturated_neurons = torch.where(mean_activations > saturated_threshold)[0].tolist()
    
    return {'low': dead_neurons, 'high': saturated_neurons}


@torch.no_grad()
def detect_network_extrema(model: nn.Sequential, 
                          data_loader: DataLoader,
                          device: str = 'cpu',
                          max_batches: int = 10) -> List[Dict[str, List[int]]]:
    """
    Detect extrema patterns across all layers of a canonical network.
    
    Args:
        model: Network created with canonical standard
        data_loader: DataLoader for getting activations
        device: Device to run analysis on
        max_batches: Maximum number of batches to analyze
        
    Returns:
        List of extrema patterns for each layer
    """
    model.eval()
    model = model.to(device)
    
    # Collect activations from all layers
    layer_activations = []
    
    def create_hook(layer_activations, layer_idx):
        def hook(module, input, output):
            if isinstance(module, StandardSparseLayer):
                # Store post-activation values (after ReLU if not last layer)
                layer_activations.append((layer_idx, output.detach().cpu()))
        return hook
    
    # Register hooks for all StandardSparseLayer modules
    hooks = []
    layer_idx = 0
    for module in model.modules():
        if isinstance(module, StandardSparseLayer):
            hook = create_hook(layer_activations, layer_idx)
            hooks.append(module.register_forward_hook(hook))
            layer_idx += 1
    
    # Run forward passes to collect activations
    all_layer_activations = [[] for _ in range(layer_idx)]
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
                
            data = data.to(device)
            if data.dim() > 2:
                data = data.view(data.size(0), -1)  # Flatten
            
            # Clear previous activations
            layer_activations.clear()
            
            # Forward pass
            _ = model(data)
            
            # Store activations by layer
            for layer_idx, activation in layer_activations:
                all_layer_activations[layer_idx].append(activation)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Analyze extrema for each layer
    extrema_patterns = []
    for layer_idx in range(len(all_layer_activations)):
        if all_layer_activations[layer_idx]:
            # Concatenate all batches for this layer
            layer_acts = torch.cat(all_layer_activations[layer_idx], dim=0)
            
            # Apply ReLU if not the last layer (to match training behavior)
            if layer_idx < len(all_layer_activations) - 1:
                layer_acts = F.relu(layer_acts)
            
            # Analyze extrema
            extrema = analyze_layer_extrema(layer_acts)
            extrema_patterns.append(extrema)
        else:
            extrema_patterns.append({'low': [], 'high': []})
    
    return extrema_patterns


def get_extrema_statistics(extrema_patterns: List[Dict[str, List[int]]]) -> Dict[str, Any]:
    """
    Get summary statistics about extrema patterns across the network.
    
    Args:
        extrema_patterns: List of extrema patterns from detect_network_extrema
        
    Returns:
        Dictionary with extrema statistics
    """
    stats = {
        'total_dead_neurons': 0,
        'total_saturated_neurons': 0,
        'layers_with_dead': 0,
        'layers_with_saturated': 0,
        'worst_layer_dead': -1,
        'worst_layer_saturated': -1,
        'max_dead_in_layer': 0,
        'max_saturated_in_layer': 0,
        'layer_stats': []
    }
    
    for i, extrema in enumerate(extrema_patterns):
        dead_count = len(extrema['low'])
        saturated_count = len(extrema['high'])
        
        stats['total_dead_neurons'] += dead_count
        stats['total_saturated_neurons'] += saturated_count
        
        if dead_count > 0:
            stats['layers_with_dead'] += 1
            if dead_count > stats['max_dead_in_layer']:
                stats['max_dead_in_layer'] = dead_count
                stats['worst_layer_dead'] = i
        
        if saturated_count > 0:
            stats['layers_with_saturated'] += 1
            if saturated_count > stats['max_saturated_in_layer']:
                stats['max_saturated_in_layer'] = saturated_count
                stats['worst_layer_saturated'] = i
        
        stats['layer_stats'].append({
            'layer_index': i,
            'dead_neurons': dead_count,
            'saturated_neurons': saturated_count,
            'total_extrema': dead_count + saturated_count
        })
    
    return stats


def print_extrema_analysis(extrema_patterns: List[Dict[str, List[int]]], 
                          network_stats: Optional[Dict[str, Any]] = None):
    """
    Print a human-readable analysis of extrema patterns.
    
    Args:
        extrema_patterns: List of extrema patterns from detect_network_extrema
        network_stats: Optional network statistics from get_network_stats
    """
    stats = get_extrema_statistics(extrema_patterns)
    
    print("\n🔍 EXTREMA ANALYSIS")
    print("=" * 50)
    
    print(f"📊 Overall Statistics:")
    print(f"   Total dead neurons: {stats['total_dead_neurons']}")
    print(f"   Total saturated neurons: {stats['total_saturated_neurons']}")
    print(f"   Layers with dead neurons: {stats['layers_with_dead']}")
    print(f"   Layers with saturated neurons: {stats['layers_with_saturated']}")
    
    if stats['worst_layer_dead'] >= 0:
        print(f"   Worst layer (dead): Layer {stats['worst_layer_dead']} ({stats['max_dead_in_layer']} dead)")
    
    if stats['worst_layer_saturated'] >= 0:
        print(f"   Worst layer (saturated): Layer {stats['worst_layer_saturated']} ({stats['max_saturated_in_layer']} saturated)")
    
    print(f"\n📋 Layer-by-Layer Analysis:")
    for layer_stat in stats['layer_stats']:
        layer_idx = layer_stat['layer_index']
        dead = layer_stat['dead_neurons']
        saturated = layer_stat['saturated_neurons']
        total = layer_stat['total_extrema']
        
        # Get layer size if network stats available
        layer_size = "unknown"
        if network_stats and layer_idx < len(network_stats['layers']):
            layer_size = network_stats['layers'][layer_idx]['out_features']
        
        print(f"   Layer {layer_idx} (size: {layer_size}): {dead} dead, {saturated} saturated, {total} total extrema")
        
        # Show specific neuron indices for small numbers
        if dead > 0 and dead <= 10:
            dead_indices = extrema_patterns[layer_idx]['low']
            print(f"      Dead neurons: {dead_indices}")
        
        if saturated > 0 and saturated <= 10:
            saturated_indices = extrema_patterns[layer_idx]['high']
            print(f"      Saturated neurons: {saturated_indices}")
