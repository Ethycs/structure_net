#!/usr/bin/env python3
"""
LSUV Trainer Component

Layer-Sequential Unit-Variance (LSUV) initialization trainer optimized for sparse networks.
This component provides initialization strategies that ensure stable gradients in sparse networks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple, Union, Dict, Any

from ...core.base_components import BaseTrainer
from ...core.interfaces import (
    ComponentContract, ComponentVersion, Maturity, ResourceRequirements, ResourceLevel
)
from ..layers import StandardSparseLayer


class LSUVTrainer(BaseTrainer):
    """
    LSUV (Layer-Sequential Unit-Variance) Initialization Trainer.
    
    LSUV is particularly effective for sparse networks because it:
    1. Ensures stable gradients through proper variance scaling
    2. Works layer-by-layer to handle sparse connectivity patterns
    3. Adapts to the actual sparsity of each layer
    """
    
    def get_contract(self) -> ComponentContract:
        return ComponentContract(
            component_name="LSUVTrainer",
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"model", "sample_input"},
            provided_outputs={"initialized_model", "variance_stats"},
            optional_inputs={"target_variance", "max_iterations", "tolerance", "skip_pretrained"},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False,
                parallel_safe=False,
                estimated_runtime_seconds=10.0
            )
        )
    
    def __init__(self, 
                 target_variance: float = 1.0,
                 max_iterations: int = 10,
                 tolerance: float = 0.01,
                 verbose: bool = False):
        """
        Initialize LSUV trainer.
        
        Args:
            target_variance: Target output variance for each layer
            max_iterations: Maximum iterations per layer
            tolerance: Convergence tolerance
            verbose: Print detailed progress
        """
        super().__init__("LSUVTrainer")
        self.target_variance = target_variance
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        
        # Statistics tracking
        self.layer_stats = []
        self.initialization_history = []
    
    def train_epoch(self, model: nn.Module, data_loader: Any) -> Dict[str, Any]:
        """
        Not used for LSUV - initialization happens in initialize_model.
        """
        raise NotImplementedError("LSUV is an initialization method, use initialize_model instead")
    
    def evaluate(self, model: nn.Module, data_loader: Any) -> Dict[str, Any]:
        """
        Evaluate variance flow through the network.
        """
        return self.analyze_variance_flow(model, next(iter(data_loader))[0])
    
    def initialize_model(self, 
                        network: nn.Sequential,
                        sample_input: torch.Tensor,
                        skip_pretrained: bool = True) -> nn.Sequential:
        """
        Apply LSUV initialization to all sparse layers in the network.
        
        Args:
            network: Sequential network with StandardSparseLayer components
            sample_input: Sample input batch for variance calculation
            skip_pretrained: Skip layers that appear to be pretrained
            
        Returns:
            The initialized network (modified in-place)
        """
        if self.verbose:
            print("🚀 Starting LSUV initialization for sparse network")
            print(f"   Target variance: {self.target_variance}")
            print(f"   Sample input shape: {sample_input.shape}")
        
        # Forward through network layer by layer
        current_input = sample_input
        
        for idx, layer in enumerate(network):
            if isinstance(layer, StandardSparseLayer):
                # Check if layer should be skipped
                if skip_pretrained and self._is_layer_pretrained(layer):
                    if self.verbose:
                        print(f"⏭️  Skipping pretrained layer {idx}")
                    current_input = layer(current_input)
                    continue
                
                # Initialize this layer
                current_input = self._init_single_layer(
                    layer, current_input, idx
                )
            else:
                # Pass through non-sparse layers (e.g., ReLU)
                current_input = layer(current_input)
        
        if self.verbose:
            print("✅ LSUV initialization complete!")
        
        return network
    
    def _init_single_layer(self,
                          layer: StandardSparseLayer,
                          sample_input: torch.Tensor,
                          layer_idx: int) -> torch.Tensor:
        """
        Apply LSUV initialization to a single sparse layer.
        """
        if self.verbose:
            print(f"🔧 LSUV initializing layer {layer_idx}: {layer.linear.weight.shape}")
        
        layer_stat = {
            'layer_idx': layer_idx,
            'shape': tuple(layer.linear.weight.shape),
            'sparsity': 1 - layer.get_sparsity_ratio(),
            'iterations': 0,
            'initial_variance': None,
            'final_variance': None
        }
        
        with torch.no_grad():
            for iteration in range(self.max_iterations):
                # Forward pass through sparse layer
                output = layer(sample_input)
                
                # Calculate output variance
                output_var = output.var().item()
                
                if iteration == 0:
                    layer_stat['initial_variance'] = output_var
                
                if self.verbose:
                    print(f"   Iteration {iteration}: variance = {output_var:.4f}")
                
                # Check convergence
                if abs(output_var - self.target_variance) < self.tolerance:
                    if self.verbose:
                        print(f"   ✅ Converged in {iteration} iterations")
                    layer_stat['iterations'] = iteration
                    layer_stat['final_variance'] = output_var
                    break
                
                # Scale weights to achieve target variance
                if output_var > 0:
                    scale_factor = torch.sqrt(torch.tensor(self.target_variance / output_var))
                    
                    # Apply scaling only to active connections (sparse-aware)
                    layer.linear.weight.data *= scale_factor
                    
                    # Re-apply mask to ensure sparsity is preserved
                    layer.linear.weight.data *= layer.mask
                else:
                    if self.verbose:
                        print("   ⚠️  Warning: Zero variance detected, reinitializing weights")
                    # Reinitialize with small random values
                    nn.init.normal_(layer.linear.weight, mean=0, std=0.01)
                    layer.linear.weight.data *= layer.mask
            
            layer_stat['final_variance'] = output.var().item()
            layer_stat['iterations'] = self.max_iterations
        
        self.layer_stats.append(layer_stat)
        return output
    
    def _is_layer_pretrained(self, 
                           layer: StandardSparseLayer,
                           variance_threshold: float = 0.1,
                           weight_threshold: float = 0.01) -> bool:
        """
        Heuristic to detect if a layer has been pretrained.
        """
        with torch.no_grad():
            # Get active weights (non-masked)
            active_weights = layer.linear.weight[layer.mask.bool()]
            
            if len(active_weights) == 0:
                return False
            
            # Check weight statistics
            weight_var = active_weights.var().item()
            weight_mean_abs = active_weights.abs().mean().item()
            
            # Pretrained layers typically have:
            # 1. Non-zero variance (not freshly initialized)
            # 2. Meaningful weight magnitudes
            is_pretrained = (weight_var > variance_threshold and 
                           weight_mean_abs > weight_threshold)
            
            return is_pretrained
    
    def analyze_variance_flow(self,
                            network: nn.Sequential,
                            sample_input: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze how variance flows through the network.
        """
        variance_flow = {
            'input_variance': sample_input.var().item(),
            'layer_variances': [],
            'activation_variances': []
        }
        
        current_input = sample_input
        
        with torch.no_grad():
            for idx, layer in enumerate(network):
                output = layer(current_input)
                
                if isinstance(layer, StandardSparseLayer):
                    variance_flow['layer_variances'].append({
                        'layer_idx': idx,
                        'output_variance': output.var().item(),
                        'output_mean': output.mean().item(),
                        'output_std': output.std().item()
                    })
                elif isinstance(layer, nn.ReLU):
                    variance_flow['activation_variances'].append({
                        'layer_idx': idx,
                        'output_variance': output.var().item(),
                        'sparsity': (output == 0).float().mean().item()
                    })
                
                current_input = output
        
        variance_flow['output_variance'] = current_input.var().item()
        return variance_flow
    
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state including statistics."""
        return {
            'layer_stats': self.layer_stats,
            'initialization_history': self.initialization_history,
            'config': {
                'target_variance': self.target_variance,
                'max_iterations': self.max_iterations,
                'tolerance': self.tolerance
            }
        }
    
    def load_training_state(self, state: Dict[str, Any]):
        """Load training state."""
        self.layer_stats = state.get('layer_stats', [])
        self.initialization_history = state.get('initialization_history', [])
        
        config = state.get('config', {})
        self.target_variance = config.get('target_variance', self.target_variance)
        self.max_iterations = config.get('max_iterations', self.max_iterations)
        self.tolerance = config.get('tolerance', self.tolerance)