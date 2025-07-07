#!/usr/bin/env python3
"""
LSUV (Layer-Sequential Unit-Variance) Initialization for Sparse Networks

This module provides LSUV initialization optimized for sparse networks.
LSUV is particularly effective for sparse networks because it:
1. Ensures stable gradients through proper variance scaling
2. Works layer-by-layer to handle sparse connectivity patterns
3. Adapts to the actual sparsity of each layer

Key Features:
- Sparse-aware LSUV that only considers active connections
- Efficient implementation leveraging sparsity patterns
- Integration with StandardSparseLayer
- Proper handling of pretrained vs new layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Union, Dict
from .layers import StandardSparseLayer


def lsuv_init_layer(layer: StandardSparseLayer, 
                   sample_input: torch.Tensor,
                   target_variance: float = 1.0,
                   max_iterations: int = 10,
                   tolerance: float = 0.01,
                   verbose: bool = False) -> torch.Tensor:
    """
    Apply LSUV initialization to a single sparse layer.
    
    This function is optimized for sparse networks - it only considers
    active connections when computing variance, making it much more
    efficient than standard LSUV.
    
    Args:
        layer: StandardSparseLayer to initialize
        sample_input: Sample input tensor for forward pass
        target_variance: Target output variance (default: 1.0)
        max_iterations: Maximum LSUV iterations (default: 10)
        tolerance: Convergence tolerance (default: 0.01)
        verbose: Print iteration details
        
    Returns:
        Output tensor after LSUV initialization
    """
    if verbose:
        print(f"🔧 LSUV initializing layer: {layer.linear.weight.shape}")
    
    with torch.no_grad():
        for iteration in range(max_iterations):
            # Forward pass through sparse layer
            output = layer(sample_input)
            
            # Calculate output variance
            output_var = output.var().item()
            
            if verbose:
                print(f"   Iteration {iteration}: variance = {output_var:.4f}")
            
            # Check convergence
            if abs(output_var - target_variance) < tolerance:
                if verbose:
                    print(f"   ✅ Converged in {iteration} iterations")
                break
            
            # Scale weights to achieve target variance
            if output_var > 0:
                scale_factor = torch.sqrt(torch.tensor(target_variance / output_var))
                
                # Apply scaling only to active connections (sparse-aware)
                layer.linear.weight.data *= scale_factor
                
                # Re-apply mask to ensure sparsity is preserved
                layer.linear.weight.data *= layer.mask
            else:
                if verbose:
                    print(f"   ⚠️  Zero variance detected, skipping scaling")
                break
        
        # Final forward pass to get the initialized output
        output = layer(sample_input)
        
        if verbose:
            final_var = output.var().item()
            print(f"   🎯 Final variance: {final_var:.4f}")
        
        return output


def lsuv_init_network(network: nn.Sequential,
                     sample_batch: torch.Tensor,
                     target_variance: float = 1.0,
                     max_iterations: int = 10,
                     tolerance: float = 0.01,
                     skip_pretrained: bool = True,
                     verbose: bool = True) -> None:
    """
    Apply LSUV initialization to an entire sparse network.
    
    This function applies LSUV layer-by-layer, using the output of each
    layer as input to the next. This ensures proper variance propagation
    through the entire network.
    
    Args:
        network: nn.Sequential network with StandardSparseLayer components
        sample_batch: Sample input batch for initialization
        target_variance: Target variance for each layer
        max_iterations: Maximum iterations per layer
        tolerance: Convergence tolerance
        skip_pretrained: Skip layers that appear to be pretrained
        verbose: Print detailed progress
    """
    if verbose:
        print(f"🚀 LSUV NETWORK INITIALIZATION")
        print("=" * 50)
        print(f"   Target variance: {target_variance}")
        print(f"   Max iterations per layer: {max_iterations}")
        print(f"   Tolerance: {tolerance}")
        print(f"   Skip pretrained: {skip_pretrained}")
    
    network.eval()  # Set to eval mode for initialization
    
    # Track activations through the network
    current_input = sample_batch
    layer_count = 0
    
    with torch.no_grad():
        for i, layer in enumerate(network):
            if isinstance(layer, StandardSparseLayer):
                layer_count += 1
                
                if verbose:
                    print(f"\n🔧 Layer {layer_count}: {layer.linear.weight.shape}")
                
                # Check if layer appears to be pretrained
                if skip_pretrained and _is_layer_pretrained(layer):
                    if verbose:
                        print(f"   ⏭️  Skipping pretrained layer")
                    # Still need to compute output for next layer
                    current_input = layer(current_input)
                    continue
                
                # Apply LSUV to this layer
                current_input = lsuv_init_layer(
                    layer=layer,
                    sample_input=current_input,
                    target_variance=target_variance,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    verbose=verbose
                )
                
            elif isinstance(layer, nn.ReLU):
                # Apply ReLU and continue
                current_input = layer(current_input)
                if verbose:
                    print(f"   📊 After ReLU: mean={current_input.mean():.4f}, std={current_input.std():.4f}")
            
            else:
                # Other layer types - just pass through
                current_input = layer(current_input)
    
    if verbose:
        print(f"\n✅ LSUV initialization complete!")
        print(f"   Initialized {layer_count} sparse layers")
        final_output = network(sample_batch)
        print(f"   Final output: mean={final_output.mean():.4f}, std={final_output.std():.4f}")


def _is_layer_pretrained(layer: StandardSparseLayer, 
                        variance_threshold: float = 0.1) -> bool:
    """
    Heuristic to detect if a layer appears to be pretrained.
    
    Pretrained layers typically have:
    - Non-random weight distributions
    - Specific variance patterns
    - Non-zero biases
    
    Args:
        layer: Layer to check
        variance_threshold: Threshold for detecting random initialization
        
    Returns:
        True if layer appears pretrained
    """
    with torch.no_grad():
        # Check weight variance (random init usually has higher variance)
        weight_var = layer.linear.weight.var().item()
        
        # Check bias patterns (pretrained often has non-zero biases)
        bias_nonzero = (layer.linear.bias != 0).sum().item()
        bias_ratio = bias_nonzero / layer.linear.bias.numel()
        
        # Check weight magnitude distribution
        weight_mean_abs = layer.linear.weight.abs().mean().item()
        
        # Heuristic: pretrained if low variance OR significant bias pattern OR specific magnitude
        is_pretrained = (
            weight_var < variance_threshold or 
            bias_ratio > 0.5 or 
            weight_mean_abs > 0.01
        )
        
        return is_pretrained


def lsuv_init_new_layers_only(network: nn.Sequential,
                             sample_batch: torch.Tensor,
                             pretrained_layer_indices: Optional[List[int]] = None,
                             target_variance: float = 1.0,
                             verbose: bool = True) -> None:
    """
    Apply LSUV only to new (non-pretrained) layers in a network.
    
    This is the recommended approach when working with networks that have
    some pretrained components and some newly added layers.
    
    Args:
        network: Network with mix of pretrained and new layers
        sample_batch: Sample input for initialization
        pretrained_layer_indices: Indices of layers to skip (if None, auto-detect)
        target_variance: Target variance for new layers
        verbose: Print progress
    """
    if verbose:
        print(f"🎯 LSUV FOR NEW LAYERS ONLY")
        print("=" * 40)
    
    # Auto-detect pretrained layers if not specified
    if pretrained_layer_indices is None:
        pretrained_layer_indices = []
        sparse_layer_idx = 0
        
        for layer in network:
            if isinstance(layer, StandardSparseLayer):
                if _is_layer_pretrained(layer):
                    pretrained_layer_indices.append(sparse_layer_idx)
                sparse_layer_idx += 1
        
        if verbose:
            print(f"   🔍 Auto-detected pretrained layers: {pretrained_layer_indices}")
    
    # Apply LSUV with selective skipping
    network.eval()
    current_input = sample_batch
    sparse_layer_idx = 0
    
    with torch.no_grad():
        for layer in network:
            if isinstance(layer, StandardSparseLayer):
                if sparse_layer_idx in pretrained_layer_indices:
                    if verbose:
                        print(f"   ⏭️  Skipping pretrained layer {sparse_layer_idx}")
                    current_input = layer(current_input)
                else:
                    if verbose:
                        print(f"   🔧 LSUV initializing new layer {sparse_layer_idx}")
                    current_input = lsuv_init_layer(
                        layer=layer,
                        sample_input=current_input,
                        target_variance=target_variance,
                        verbose=verbose
                    )
                sparse_layer_idx += 1
                
            elif isinstance(layer, nn.ReLU):
                current_input = layer(current_input)
            else:
                current_input = layer(current_input)
    
    if verbose:
        print(f"✅ Selective LSUV complete!")


def create_lsuv_initialized_network(architecture: List[int],
                                  sparsity: float,
                                  sample_batch: torch.Tensor,
                                  target_variance: float = 1.0,
                                  seed: Optional[int] = None,
                                  device: str = 'cpu',
                                  verbose: bool = True) -> nn.Sequential:
    """
    Create a sparse network with LSUV initialization.
    
    This is a convenience function that creates a standard sparse network
    and immediately applies LSUV initialization.
    
    Args:
        architecture: Network architecture [input, hidden1, ..., output]
        sparsity: Sparsity level
        sample_batch: Sample batch for LSUV initialization
        target_variance: Target variance for LSUV
        seed: Random seed
        device: Device to create network on
        verbose: Print progress
        
    Returns:
        LSUV-initialized sparse network
    """
    from .network_factory import create_standard_network
    
    if verbose:
        print(f"🏗️  Creating LSUV-initialized network")
        print(f"   Architecture: {architecture}")
        print(f"   Sparsity: {sparsity:.1%}")
        print(f"   Target variance: {target_variance}")
    
    # Create standard network
    network = create_standard_network(
        architecture=architecture,
        sparsity=sparsity,
        seed=seed,
        device=device
    )
    
    # Move sample batch to same device
    sample_batch = sample_batch.to(device)
    
    # Apply LSUV initialization
    lsuv_init_network(
        network=network,
        sample_batch=sample_batch,
        target_variance=target_variance,
        verbose=verbose
    )
    
    return network


def analyze_network_variance_flow(network: nn.Sequential,
                                sample_batch: torch.Tensor,
                                verbose: bool = True) -> List[Dict[str, float]]:
    """
    Analyze variance flow through a network.
    
    This function helps diagnose variance issues and determine if
    LSUV initialization is needed.
    
    Args:
        network: Network to analyze
        sample_batch: Sample input batch
        verbose: Print analysis results
        
    Returns:
        List of variance statistics for each layer
    """
    if verbose:
        print(f"📊 NETWORK VARIANCE FLOW ANALYSIS")
        print("=" * 40)
    
    network.eval()
    variance_stats = []
    current_input = sample_batch
    layer_idx = 0
    
    with torch.no_grad():
        for i, layer in enumerate(network):
            if isinstance(layer, StandardSparseLayer):
                # Compute output
                output = layer(current_input)
                
                # Calculate statistics
                stats = {
                    'layer_index': layer_idx,
                    'input_mean': current_input.mean().item(),
                    'input_std': current_input.std().item(),
                    'input_var': current_input.var().item(),
                    'output_mean': output.mean().item(),
                    'output_std': output.std().item(),
                    'output_var': output.var().item(),
                    'weight_mean': layer.linear.weight.mean().item(),
                    'weight_std': layer.linear.weight.std().item(),
                    'bias_mean': layer.linear.bias.mean().item(),
                    'bias_std': layer.linear.bias.std().item(),
                    'sparsity': (layer.mask == 0).float().mean().item(),
                    'active_connections': layer.mask.sum().item()
                }
                
                variance_stats.append(stats)
                
                if verbose:
                    print(f"\n🔍 Layer {layer_idx}:")
                    print(f"   Input:  mean={stats['input_mean']:.4f}, std={stats['input_std']:.4f}, var={stats['input_var']:.4f}")
                    print(f"   Output: mean={stats['output_mean']:.4f}, std={stats['output_std']:.4f}, var={stats['output_var']:.4f}")
                    print(f"   Weight: mean={stats['weight_mean']:.4f}, std={stats['weight_std']:.4f}")
                    print(f"   Sparsity: {stats['sparsity']:.1%}, Active: {stats['active_connections']}")
                    
                    # Variance change analysis
                    var_ratio = stats['output_var'] / max(stats['input_var'], 1e-8)
                    if var_ratio > 2.0:
                        print(f"   ⚠️  Variance explosion: {var_ratio:.2f}x")
                    elif var_ratio < 0.5:
                        print(f"   ⚠️  Variance collapse: {var_ratio:.2f}x")
                    else:
                        print(f"   ✅ Variance stable: {var_ratio:.2f}x")
                
                current_input = output
                layer_idx += 1
                
            elif isinstance(layer, nn.ReLU):
                current_input = layer(current_input)
                if verbose:
                    print(f"   📊 After ReLU: mean={current_input.mean():.4f}, std={current_input.std():.4f}")
            else:
                current_input = layer(current_input)
    
    if verbose:
        print(f"\n📈 VARIANCE FLOW SUMMARY:")
        if variance_stats:
            input_var = variance_stats[0]['input_var']
            output_var = variance_stats[-1]['output_var']
            total_ratio = output_var / max(input_var, 1e-8)
            print(f"   Total variance change: {total_ratio:.2f}x")
            
            # Check for problematic layers
            problem_layers = []
            for i, stats in enumerate(variance_stats):
                if i > 0:
                    prev_var = variance_stats[i-1]['output_var']
                    curr_var = stats['output_var']
                    ratio = curr_var / max(prev_var, 1e-8)
                    if ratio > 2.0 or ratio < 0.5:
                        problem_layers.append(i)
            
            if problem_layers:
                print(f"   ⚠️  Problematic layers: {problem_layers}")
                print(f"   💡 Consider LSUV initialization")
            else:
                print(f"   ✅ Variance flow looks healthy")
    
    return variance_stats


# Export the LSUV interface
__all__ = [
    'lsuv_init_layer',
    'lsuv_init_network', 
    'lsuv_init_new_layers_only',
    'create_lsuv_initialized_network',
    'analyze_network_variance_flow'
]
