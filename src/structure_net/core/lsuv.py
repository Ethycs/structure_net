#!/usr/bin/env python3
"""
LSUV (Layer-Sequential Unit-Variance) Initialization - Compatibility Module

This module provides backward compatibility for LSUV initialization functions.
The implementation has been refactored into a proper component at
structure_net.components.trainers.LSUVTrainer.

DEPRECATED: Please use structure_net.components.trainers.LSUVTrainer instead.
"""

import warnings
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union, Dict, Any

from ..components.layers import StandardSparseLayer
from ..components.trainers import LSUVTrainer

# Issue deprecation warning
warnings.warn(
    "Importing from structure_net.core.lsuv is deprecated. "
    "Please use structure_net.components.trainers.LSUVTrainer instead.",
    DeprecationWarning,
    stacklevel=2
)


# Compatibility wrapper functions
def lsuv_init_layer(layer: StandardSparseLayer, 
                   sample_input: torch.Tensor,
                   target_variance: float = 1.0,
                   max_iterations: int = 10,
                   tolerance: float = 0.01,
                   verbose: bool = False) -> torch.Tensor:
    """
    DEPRECATED: Use LSUVTrainer instead.
    
    Apply LSUV initialization to a single sparse layer.
    """
    trainer = LSUVTrainer(
        target_variance=target_variance,
        max_iterations=max_iterations,
        tolerance=tolerance,
        verbose=verbose
    )
    
    # Create a dummy network with just this layer
    dummy_net = nn.Sequential(layer)
    trainer.initialize_model(dummy_net, sample_input, skip_pretrained=False)
    
    return layer(sample_input)


def lsuv_init_network(network: nn.Sequential,
                     sample_input: torch.Tensor,
                     target_variance: float = 1.0,
                     max_iterations: int = 10,
                     tolerance: float = 0.01,
                     verbose: bool = False,
                     skip_pretrained: bool = True) -> nn.Sequential:
    """
    DEPRECATED: Use LSUVTrainer instead.
    
    Apply LSUV initialization to all sparse layers in a network.
    """
    trainer = LSUVTrainer(
        target_variance=target_variance,
        max_iterations=max_iterations,
        tolerance=tolerance,
        verbose=verbose
    )
    
    return trainer.initialize_model(network, sample_input, skip_pretrained=skip_pretrained)


def _is_layer_pretrained(layer: StandardSparseLayer, 
                        variance_threshold: float = 0.1,
                        weight_threshold: float = 0.01) -> bool:
    """
    DEPRECATED: This is now a method in LSUVTrainer.
    
    Heuristic to detect if a layer has been pretrained.
    """
    trainer = LSUVTrainer()
    return trainer._is_layer_pretrained(layer, variance_threshold, weight_threshold)


def lsuv_init_new_layers_only(network: nn.Sequential,
                             sample_input: torch.Tensor,
                             target_variance: float = 1.0,
                             max_iterations: int = 10,
                             tolerance: float = 0.01,
                             verbose: bool = False) -> nn.Sequential:
    """
    DEPRECATED: Use LSUVTrainer with skip_pretrained=True.
    
    Apply LSUV only to layers that appear to be newly initialized.
    """
    return lsuv_init_network(
        network, 
        sample_input, 
        target_variance=target_variance,
        max_iterations=max_iterations,
        tolerance=tolerance,
        verbose=verbose,
        skip_pretrained=True
    )


def create_lsuv_initialized_network(architecture: List[int],
                                  sparsity: float,
                                  sample_batch_size: int = 32,
                                  seed: Optional[int] = None,
                                  device: str = 'cpu',
                                  verbose: bool = False) -> nn.Sequential:
    """
    DEPRECATED: Create network using network_factory then initialize with LSUVTrainer.
    
    Create and initialize a sparse network with LSUV in one step.
    """
    from .network_factory import create_standard_network
    
    # Create network
    network = create_standard_network(architecture, sparsity, seed=seed, device=device)
    
    # Create sample input
    sample_input = torch.randn(sample_batch_size, architecture[0], device=device)
    
    # Apply LSUV
    trainer = LSUVTrainer(verbose=verbose)
    trainer.initialize_model(network, sample_input, skip_pretrained=False)
    
    return network


def analyze_network_variance_flow(network: nn.Sequential,
                                sample_input: torch.Tensor,
                                verbose: bool = False) -> Dict[str, Any]:
    """
    DEPRECATED: Use LSUVTrainer.analyze_variance_flow instead.
    
    Analyze how variance flows through the network.
    """
    trainer = LSUVTrainer(verbose=verbose)
    return trainer.analyze_variance_flow(network, sample_input)


# Export functions for compatibility
__all__ = [
    'lsuv_init_layer',
    'lsuv_init_network',
    'lsuv_init_new_layers_only',
    'create_lsuv_initialized_network',
    'analyze_network_variance_flow'
]