"""
Minimal Network Implementation for Multi-Scale Snapshots Experiment

This module implements the base minimal network that starts with 0.01% connections
and supports dynamic growth based on extrema detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class MinimalNetwork(nn.Module):
    """
    A neural network that starts with minimal connectivity (0.01%) and grows dynamically.
    
    Key features:
    - Sparse initialization with only 0.01% of possible connections
    - Dynamic growth capability
    - Extrema tracking for growth decisions
    - Multi-scale snapshot support
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        sparsity: float = 0.0001,  # 0.01% connectivity
        activation: str = 'tanh',
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.layer_sizes = layer_sizes
        self.sparsity = sparsity
        self.activation_name = activation
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize layers with sparse connectivity
        self.layers = nn.ModuleList()
        self.connection_masks = []  # Track which connections are active
        self.activation_history = []  # Track activations for extrema detection
        
        # Initialize sparse layers
        self._initialize_sparse_layers()
        
        # Set activation function
        if activation == 'tanh':
            self.activation_fn = torch.tanh
        elif activation == 'sigmoid':
            self.activation_fn = torch.sigmoid
        elif activation == 'relu':
            self.activation_fn = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Move to device
        self.to(self.device)
        
        # Growth tracking
        self.growth_history = []
        self.extrema_history = []
        
    def _initialize_sparse_layers(self):
        """Initialize layers with sparse connectivity."""
        for i in range(len(self.layer_sizes) - 1):
            in_features = self.layer_sizes[i]
            out_features = self.layer_sizes[i + 1]
            
            # Create full linear layer
            layer = nn.Linear(in_features, out_features, bias=True)
            
            # Initialize weights with Xavier/He initialization
            if self.activation_name in ['tanh', 'sigmoid']:
                nn.init.xavier_uniform_(layer.weight)
            else:  # ReLU
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            
            # Create sparse mask
            total_connections = in_features * out_features
            num_active = max(1, int(total_connections * self.sparsity))
            
            # Create mask with random sparse connections
            mask = torch.zeros(out_features, in_features, dtype=torch.bool)
            flat_mask = mask.view(-1)
            active_indices = torch.randperm(total_connections)[:num_active]
            flat_mask[active_indices] = True
            mask = flat_mask.view(out_features, in_features)
            
            # Apply mask to weights
            with torch.no_grad():
                layer.weight.data *= mask.float()
            
            self.layers.append(layer)
            self.connection_masks.append(mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sparse connectivity and activation tracking."""
        activations = []
        
        for i, (layer, mask) in enumerate(zip(self.layers, self.connection_masks)):
            # Apply sparse connectivity
            with torch.no_grad():
                layer.weight.data *= mask.float()
            
            # Linear transformation
            x = layer(x)
            
            # Apply activation (except for output layer)
            if i < len(self.layers) - 1:
                x = self.activation_fn(x)
            
            # Store activations for extrema detection
            activations.append(x.detach().clone())
        
        # Update activation history
        self.activation_history = activations
        
        return x
    
    def detect_extrema(self, threshold_high: float = 0.95, threshold_low: float = 0.05) -> Dict[int, Dict[str, List[int]]]:
        """
        Detect high and low extrema in network activations.
        
        Args:
            threshold_high: Threshold for high extrema
            threshold_low: Threshold for low extrema
            
        Returns:
            Dictionary mapping layer index to extrema indices
        """
        if not self.activation_history:
            return {}
        
        extrema = {}
        
        for layer_idx, activations in enumerate(self.activation_history[:-1]):  # Skip output layer
            layer_extrema = {'high': [], 'low': []}
            
            # Get mean activation across batch
            mean_activations = activations.mean(dim=0)
            
            if self.activation_name == 'relu':
                # For ReLU, detect dead neurons (always zero)
                dead_neurons = (mean_activations == 0).nonzero(as_tuple=True)[0].tolist()
                layer_extrema['low'] = dead_neurons
                
                # High extrema for ReLU could be very high activations
                high_neurons = (mean_activations > mean_activations.quantile(0.95)).nonzero(as_tuple=True)[0].tolist()
                layer_extrema['high'] = high_neurons
                
            else:  # tanh, sigmoid
                # High extrema
                high_neurons = (mean_activations > threshold_high).nonzero(as_tuple=True)[0].tolist()
                layer_extrema['high'] = high_neurons
                
                # Low extrema
                low_neurons = (mean_activations < threshold_low).nonzero(as_tuple=True)[0].tolist()
                layer_extrema['low'] = low_neurons
            
            extrema[layer_idx] = layer_extrema
        
        # Store extrema history
        self.extrema_history.append(extrema)
        
        return extrema
    
    def add_connections(self, connections: List[Tuple[int, int, int, float]]):
        """
        Add new connections to the network.
        
        Args:
            connections: List of (source_layer, source_neuron, target_layer, target_neuron, weight)
        """
        for source_layer, source_neuron, target_layer, target_neuron in connections:
            if target_layer == source_layer + 1:  # Only adjacent layers for now
                layer_idx = source_layer
                
                # Update mask
                self.connection_masks[layer_idx][target_neuron, source_neuron] = True
                
                # Initialize new connection weight
                with torch.no_grad():
                    if self.activation_name in ['tanh', 'sigmoid']:
                        init_weight = torch.randn(1) * math.sqrt(2.0 / self.layer_sizes[source_layer])
                    else:  # ReLU
                        init_weight = torch.randn(1) * math.sqrt(2.0 / self.layer_sizes[source_layer])
                    
                    self.layers[layer_idx].weight.data[target_neuron, source_neuron] = init_weight * 0.1
        
        # Record growth event
        self.growth_history.append({
            'connections_added': len(connections),
            'total_connections': sum(mask.sum().item() for mask in self.connection_masks)
        })
    
    def add_neurons(self, clones: List[Dict]):
        """
        Add cloned neurons to adjacent layers.
        
        Args:
            clones: List of clone specifications
        """
        # For now, this is a placeholder - full implementation would require
        # dynamically resizing layers, which is complex in PyTorch
        # In practice, we might pre-allocate extra neurons and activate them
        pass
    
    def get_connectivity_stats(self) -> Dict:
        """Get current connectivity statistics."""
        total_possible = sum(
            self.layer_sizes[i] * self.layer_sizes[i + 1] 
            for i in range(len(self.layer_sizes) - 1)
        )
        
        total_active = sum(mask.sum().item() for mask in self.connection_masks)
        
        return {
            'total_possible_connections': total_possible,
            'total_active_connections': total_active,
            'connectivity_ratio': total_active / total_possible,
            'sparsity': 1.0 - (total_active / total_possible)
        }
    
    def get_gradient_norm(self) -> float:
        """Compute gradient norm for growth detection."""
        total_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def state_dict_sparse(self) -> Dict:
        """Get state dict including sparse connectivity information."""
        state = super().state_dict()
        state['connection_masks'] = self.connection_masks
        state['layer_sizes'] = self.layer_sizes
        state['sparsity'] = self.sparsity
        state['activation_name'] = self.activation_name
        state['growth_history'] = self.growth_history
        return state
    
    def load_state_dict_sparse(self, state_dict: Dict):
        """Load state dict including sparse connectivity information."""
        # Extract sparse-specific information
        self.connection_masks = state_dict.pop('connection_masks')
        self.layer_sizes = state_dict.pop('layer_sizes')
        self.sparsity = state_dict.pop('sparsity')
        self.activation_name = state_dict.pop('activation_name')
        self.growth_history = state_dict.pop('growth_history', [])
        
        # Load standard state dict
        super().load_state_dict(state_dict)


def create_minimal_network(
    input_size: int,
    hidden_sizes: List[int],
    output_size: int,
    sparsity: float = 0.0001,
    activation: str = 'tanh'
) -> MinimalNetwork:
    """
    Factory function to create a minimal network.
    
    Args:
        input_size: Size of input layer
        hidden_sizes: List of hidden layer sizes
        output_size: Size of output layer
        sparsity: Initial connectivity ratio (default 0.01%)
        activation: Activation function name
        
    Returns:
        MinimalNetwork instance
    """
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    return MinimalNetwork(layer_sizes, sparsity, activation)


# Example usage and testing
if __name__ == "__main__":
    # Create a minimal network for MNIST
    network = create_minimal_network(
        input_size=784,
        hidden_sizes=[256, 128],
        output_size=10,
        sparsity=0.0001,
        activation='tanh'
    )
    
    print("Network created:")
    print(f"Architecture: {network.layer_sizes}")
    print(f"Connectivity stats: {network.get_connectivity_stats()}")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 784)
    output = network(x)
    print(f"Output shape: {output.shape}")
    
    # Test extrema detection
    extrema = network.detect_extrema()
    print(f"Detected extrema: {extrema}")
