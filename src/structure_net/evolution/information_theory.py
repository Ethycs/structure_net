#!/usr/bin/env python3
"""
Information Theory - Using Canonical Standard

This module provides information theory analysis capabilities
for networks created with the canonical model standard.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple
from torch.utils.data import DataLoader

from ..core.model_io import StandardSparseLayer


def estimate_mi_proxy(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    A fast proxy for Mutual Information based on correlation.
    
    Args:
        x: Input tensor [batch_size, features]
        y: Output tensor [batch_size, features]
        
    Returns:
        MI proxy estimate (float)
    """
    if x.numel() == 0 or y.numel() == 0: 
        return 0.0
    
    # Normalize tensors
    x_norm = F.normalize(x, dim=1)
    y_norm = F.normalize(y, dim=1)
    
    # Calculate correlation on minimum dimensions
    min_dim = min(x_norm.shape[1], y_norm.shape[1])
    correlation = (x_norm[:, :min_dim] * y_norm[:, :min_dim]).sum(dim=1).mean()
    
    # Convert correlation to MI approximation
    mi_approx = -0.5 * torch.log(1 - correlation**2 + 1e-8)
    return mi_approx.item()


@torch.no_grad()
def analyze_information_flow(model: nn.Sequential, 
                           data_loader: DataLoader,
                           device: str = 'cpu',
                           max_batches: int = 10) -> List[float]:
    """
    Analyze information flow through a canonical network using MI proxy.
    
    Args:
        model: Network created with canonical standard
        data_loader: DataLoader for getting activations
        device: Device to run analysis on
        max_batches: Maximum number of batches to analyze
        
    Returns:
        List of MI estimates between consecutive layers
    """
    model.eval()
    model = model.to(device)
    
    # Collect activations from all layers
    all_activations = []
    
    def create_hook(all_activations, layer_idx):
        def hook(module, input, output):
            if isinstance(module, StandardSparseLayer):
                all_activations.append((layer_idx, output.detach().cpu()))
        return hook
    
    # Register hooks for all StandardSparseLayer modules
    hooks = []
    layer_idx = 0
    for module in model.modules():
        if isinstance(module, StandardSparseLayer):
            hook = create_hook(all_activations, layer_idx)
            hooks.append(module.register_forward_hook(hook))
            layer_idx += 1
    
    # Collect activations across batches
    layer_activations = [[] for _ in range(layer_idx)]
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
                
            data = data.to(device)
            if data.dim() > 2:
                data = data.view(data.size(0), -1)  # Flatten
            
            # Store input activations
            input_activations = data.detach().cpu()
            
            # Clear previous activations
            all_activations.clear()
            
            # Forward pass
            _ = model(data)
            
            # Store activations by layer (including input)
            if batch_idx == 0:
                layer_activations[0].append(input_activations)  # Input layer
            else:
                layer_activations[0].append(input_activations)
            
            for layer_idx, activation in all_activations:
                if layer_idx + 1 < len(layer_activations):
                    layer_activations[layer_idx + 1].append(activation)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate MI between consecutive layers
    mi_flow = []
    for i in range(len(layer_activations) - 1):
        if layer_activations[i] and layer_activations[i + 1]:
            # Concatenate activations across batches
            x = torch.cat(layer_activations[i], dim=0)
            y = torch.cat(layer_activations[i + 1], dim=0)
            
            # Apply ReLU to intermediate layers
            if i > 0:  # Not input layer
                x = F.relu(x)
            if i + 1 < len(layer_activations) - 1:  # Not output layer
                y = F.relu(y)
            
            # Estimate MI
            mi = estimate_mi_proxy(x, y)
            mi_flow.append(mi)
        else:
            mi_flow.append(0.0)
    
    return mi_flow


def detect_information_bottlenecks(mi_flow: List[float], 
                                 threshold: float = 0.05) -> List[Dict[str, Any]]:
    """
    Detect information bottlenecks from MI flow analysis.
    
    Args:
        mi_flow: List of MI estimates from analyze_information_flow
        threshold: Minimum information loss to consider as bottleneck
        
    Returns:
        List of bottleneck information dictionaries
    """
    bottlenecks = []
    
    # Calculate information loss at each step
    for i in range(len(mi_flow) - 1):
        info_loss = mi_flow[i] - mi_flow[i + 1]
        
        if info_loss > threshold:
            severity = info_loss / (mi_flow[0] + 1e-6) if mi_flow[0] > 0 else 0
            
            bottlenecks.append({
                'position': i + 1,  # Bottleneck is AT layer i+1
                'info_loss': info_loss,
                'severity': severity,
                'input_mi': mi_flow[i],
                'output_mi': mi_flow[i + 1]
            })
    
    # Sort by severity (highest first)
    bottlenecks.sort(key=lambda x: x['severity'], reverse=True)
    
    return bottlenecks


def calculate_optimal_intervention(bottleneck: Dict[str, Any], 
                                 base_sparsity: float = 0.02) -> Dict[str, Any]:
    """
    Calculate optimal intervention for a detected bottleneck.
    
    Args:
        bottleneck: Bottleneck information from detect_information_bottlenecks
        base_sparsity: Base sparsity level of the network
        
    Returns:
        Dictionary with intervention recommendation
    """
    info_loss = bottleneck['info_loss']
    severity = bottleneck['severity']
    
    # Capacity formula: I_max = -s * log(s) * width
    capacity_per_neuron = -base_sparsity * np.log(base_sparsity) if 0 < base_sparsity < 1 else 0.1
    
    if capacity_per_neuron > 0:
        neurons_needed = int(np.ceil(info_loss / capacity_per_neuron))
    else:
        neurons_needed = 64  # Fallback
    
    # Clamp to reasonable range
    neurons_needed = min(max(neurons_needed, 16), 1024)
    
    # Differentiated strategy based on severity
    if severity > 0.5:  # Severe loss
        return {
            'type': 'insert_layer',
            'width': neurons_needed,
            'position': bottleneck['position'],
            'reason': f'Severe information loss ({info_loss:.3f} bits)',
            'severity': severity
        }
    elif severity > 0.2:  # Moderate loss
        return {
            'type': 'add_skip_connection',
            'position': bottleneck['position'],
            'reason': f'Moderate information loss ({info_loss:.3f} bits)',
            'severity': severity
        }
    else:  # Mild loss
        return {
            'type': 'increase_density',
            'position': bottleneck['position'],
            'density_increase': min(0.1, severity),
            'reason': f'Mild information loss ({info_loss:.3f} bits)',
            'severity': severity
        }


def print_information_analysis(mi_flow: List[float], 
                             bottlenecks: List[Dict[str, Any]]):
    """
    Print human-readable information flow analysis.
    
    Args:
        mi_flow: List of MI estimates from analyze_information_flow
        bottlenecks: List of bottlenecks from detect_information_bottlenecks
    """
    print("\n🌊 INFORMATION FLOW ANALYSIS")
    print("=" * 50)
    
    print(f"📊 MI Flow: {[f'{mi:.3f}' for mi in mi_flow]}")
    
    if bottlenecks:
        print(f"\n🔥 Detected {len(bottlenecks)} Information Bottlenecks:")
        for i, bottleneck in enumerate(bottlenecks):
            pos = bottleneck['position']
            loss = bottleneck['info_loss']
            severity = bottleneck['severity']
            
            print(f"   {i+1}. Layer {pos}: {loss:.3f} bits lost (severity: {severity:.1%})")
            
            # Calculate intervention
            intervention = calculate_optimal_intervention(bottleneck)
            print(f"      Recommended: {intervention['type']} - {intervention['reason']}")
    else:
        print("\n✅ No significant information bottlenecks detected")
    
    # Overall flow health
    total_loss = mi_flow[0] - mi_flow[-1] if len(mi_flow) > 1 else 0
    efficiency = mi_flow[-1] / mi_flow[0] if mi_flow[0] > 0 else 0
    
    print(f"\n📈 Overall Information Health:")
    print(f"   Total information loss: {total_loss:.3f} bits")
    print(f"   Information efficiency: {efficiency:.1%}")
    
    if efficiency < 0.5:
        print("   ⚠️  Low information efficiency - network may benefit from evolution")
    elif efficiency > 0.8:
        print("   ✅ Good information efficiency")
    else:
        print("   📊 Moderate information efficiency")
