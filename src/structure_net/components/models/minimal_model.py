"""
Minimal Model Component.

A simple feedforward neural network with sparse connections,
reimplemented as a self-aware component.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

from ...core import (
    BaseModel, ComponentContract, ComponentVersion,
    Maturity, ResourceRequirements, ResourceLevel,
    ILayer
)
from ..layers.sparse_layer import SparseLayer


class MinimalModel(BaseModel):
    """
    A minimal sparse neural network model.
    
    This component provides a simple feedforward architecture with
    configurable sparsity, following the component architecture.
    """
    
    def __init__(self,
                 layer_sizes: List[int],
                 sparsity: float = 0.9,
                 activation: str = 'relu',
                 bias: bool = True,
                 name: str = None):
        """
        Initialize minimal model.
        
        Args:
            layer_sizes: List of layer dimensions [input, hidden1, ..., output]
            sparsity: Target sparsity level (0 to 1)
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            bias: Whether to use bias terms
            name: Optional custom name
        """
        super().__init__(name or "MinimalModel")
        
        self.layer_sizes = layer_sizes
        self.sparsity = sparsity
        self.activation_fn = activation
        self.use_bias = bias
        
        # Build the network
        self._build_network()
        
        # Define contract
        self._contract = ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={'input'},
            provided_outputs={
                'output',
                'model.architecture',
                'model.sparsity_info',
                'model.layer_activations'
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    @property
    def contract(self) -> ComponentContract:
        """Return the component contract."""
        return self._contract
    
    def _build_network(self):
        """Build the network layers."""
        self.layers = nn.ModuleList()
        self._layers = []  # For IModel interface
        
        # Create layers
        for i in range(len(self.layer_sizes) - 1):
            in_features = self.layer_sizes[i]
            out_features = self.layer_sizes[i + 1]
            
            # Create sparse layer
            sparse_layer = SparseLayer(
                in_features=in_features,
                out_features=out_features,
                sparsity=self.sparsity,
                bias=self.use_bias,
                name=f"{self.name}_layer_{i}"
            )
            
            self.layers.append(sparse_layer)
            self._layers.append(sparse_layer)
            
            # Add activation (except for last layer)
            if i < len(self.layer_sizes) - 2:
                activation = self._get_activation()
                if activation:
                    self.layers.append(activation)
    
    def _get_activation(self) -> Optional[nn.Module]:
        """Get activation module based on string name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }
        return activations.get(self.activation_fn.lower())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Store layer activations for analysis
        self.layer_activations = []
        
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Forward through layers
        for layer in self.layers:
            x = layer(x)
            
            # Store activations from sparse layers
            if isinstance(layer, SparseLayer):
                self.layer_activations.append(x.detach())
        
        return x
    
    def get_layers(self) -> List[ILayer]:
        """Get all sparse layers in the model."""
        return self._layers
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get high-level architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        
        # Count active connections
        active_connections = 0
        layer_info = []
        
        for i, layer in enumerate(self._layers):
            if hasattr(layer, 'get_analysis_properties'):
                props = layer.get_analysis_properties()
                if 'mask' in props:
                    active = props['mask'].sum().item()
                    total = props['mask'].numel()
                    active_connections += active
                    
                    layer_info.append({
                        'name': f'layer_{i}',
                        'active_connections': active,
                        'total_connections': total,
                        'sparsity': 1.0 - (active / total if total > 0 else 0)
                    })
        
        return {
            'architecture': self.layer_sizes,
            'total_parameters': total_params,
            'active_connections': active_connections,
            'target_sparsity': self.sparsity,
            'actual_sparsity': 1.0 - (active_connections / total_params if total_params > 0 else 0),
            'activation': self.activation_fn,
            'num_layers': len(self._layers),
            'layer_details': layer_info
        }
    
    def get_connectivity_stats(self) -> Dict[str, Any]:
        """Get detailed connectivity statistics."""
        stats = {
            'layers': [],
            'total_connections': 0,
            'active_connections': 0,
            'dead_neurons': 0
        }
        
        for i, layer in enumerate(self._layers):
            if hasattr(layer, 'get_analysis_properties'):
                props = layer.get_analysis_properties()
                
                if 'weight' in props and 'mask' in props:
                    weight = props['weight']
                    mask = props['mask']
                    
                    # Active connections
                    active = mask.sum().item()
                    total = mask.numel()
                    
                    # Dead neurons (outputs with no connections)
                    output_connections = mask.sum(dim=1)
                    dead_outputs = (output_connections == 0).sum().item()
                    
                    # Input importance (how many outputs each input connects to)
                    input_connections = mask.sum(dim=0)
                    
                    layer_stats = {
                        'layer_index': i,
                        'shape': list(weight.shape),
                        'total_connections': total,
                        'active_connections': active,
                        'sparsity': 1.0 - (active / total if total > 0 else 0),
                        'dead_neurons': dead_outputs,
                        'avg_connections_per_neuron': active / weight.shape[0] if weight.shape[0] > 0 else 0,
                        'max_input_connections': input_connections.max().item(),
                        'min_input_connections': input_connections.min().item()
                    }
                    
                    stats['layers'].append(layer_stats)
                    stats['total_connections'] += total
                    stats['active_connections'] += active
                    stats['dead_neurons'] += dead_outputs
        
        return stats
    
    def add_connections(self, layer_idx: int, num_connections: int):
        """
        Add connections to a specific layer.
        
        Args:
            layer_idx: Index of the layer to modify
            num_connections: Number of connections to add
        """
        if 0 <= layer_idx < len(self._layers):
            layer = self._layers[layer_idx]
            if hasattr(layer, 'add_connections'):
                layer.add_connections(num_connections)
                self.log(logging.INFO, 
                        f"Added {num_connections} connections to layer {layer_idx}")
    
    def prune_connections(self, layer_idx: int, num_connections: int):
        """
        Remove connections from a specific layer.
        
        Args:
            layer_idx: Index of the layer to modify
            num_connections: Number of connections to remove
        """
        if 0 <= layer_idx < len(self._layers):
            layer = self._layers[layer_idx]
            if hasattr(layer, 'prune_connections'):
                layer.prune_connections(num_connections)
                self.log(logging.INFO,
                        f"Pruned {num_connections} connections from layer {layer_idx}")
    
    def get_gradient_stats(self) -> Dict[str, float]:
        """Get gradient statistics across the model."""
        stats = {
            'gradient_norm': 0.0,
            'max_gradient': 0.0,
            'min_gradient': float('inf'),
            'num_zero_gradients': 0
        }
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                stats['gradient_norm'] += grad_norm ** 2
                
                grad_max = param.grad.abs().max().item()
                grad_min = param.grad.abs().min().item()
                
                stats['max_gradient'] = max(stats['max_gradient'], grad_max)
                stats['min_gradient'] = min(stats['min_gradient'], grad_min)
                
                # Count zero gradients
                stats['num_zero_gradients'] += (param.grad == 0).sum().item()
        
        stats['gradient_norm'] = stats['gradient_norm'] ** 0.5
        return stats
    
    def reset_parameters(self):
        """Reset all parameters to initial values."""
        for layer in self._layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def to_dense(self) -> nn.Module:
        """Convert to a dense (non-sparse) version."""
        dense_layers = []
        
        for i, module in enumerate(self.layers):
            if isinstance(module, SparseLayer):
                # Create dense linear layer
                dense = nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None
                )
                
                # Copy weights
                props = module.get_analysis_properties()
                if 'weight' in props and 'mask' in props:
                    dense.weight.data = props['weight'] * props['mask']
                
                if module.bias is not None:
                    dense.bias.data = module.bias.data
                
                dense_layers.append(dense)
            else:
                # Keep activation layers as is
                dense_layers.append(module)
        
        return nn.Sequential(*dense_layers)