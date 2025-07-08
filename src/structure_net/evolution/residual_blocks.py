#!/usr/bin/env python3
"""
Residual Block Implementation for Structure Net

This module provides residual block capabilities that integrate seamlessly
with the sparse network architecture and canonical standard.

Key features:
- Sparse residual blocks with skip connections
- Adaptive residual insertion during growth
- Compatible with extrema-driven growth
- Maintains canonical standard compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from ..core.layers import StandardSparseLayer
from ..core.network_factory import create_standard_network


class SparseResidualBlock(nn.Module):
    """
    Sparse Residual Block with skip connections.
    
    Implements residual learning: output = F(x) + x
    where F(x) is a sequence of sparse layers.
    """
    
    def __init__(self, 
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 sparsity: float = 0.02,
                 num_layers: int = 2,
                 device: str = 'cuda'):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.sparsity = sparsity
        self.num_layers = num_layers
        
        # Create the residual path (F(x))
        self.residual_layers = nn.ModuleList()
        
        # First layer: in_features -> hidden_features
        self.residual_layers.append(
            StandardSparseLayer(in_features, hidden_features, sparsity)
        )
        
        # Hidden layers: hidden_features -> hidden_features
        for _ in range(num_layers - 2):
            self.residual_layers.append(
                StandardSparseLayer(hidden_features, hidden_features, sparsity)
            )
        
        # Final layer: hidden_features -> out_features
        if num_layers > 1:
            self.residual_layers.append(
                StandardSparseLayer(hidden_features, out_features, sparsity)
            )
        
        # Skip connection projection (if dimensions don't match)
        self.skip_projection = None
        if in_features != out_features:
            self.skip_projection = StandardSparseLayer(
                in_features, out_features, sparsity * 0.5  # Sparser skip connection
            )
        
        # Move to device
        self.to(device)
        
        print(f"🔗 Created SparseResidualBlock: {in_features} → {hidden_features} → {out_features}")
        print(f"   Layers: {num_layers}, Sparsity: {sparsity:.1%}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection"""
        # Store input for skip connection
        identity = x
        
        # Apply residual path F(x)
        residual = x
        for i, layer in enumerate(self.residual_layers):
            residual = layer(residual)
            if i < len(self.residual_layers) - 1:  # Apply ReLU except for last layer
                residual = F.relu(residual)
        
        # Apply skip connection projection if needed
        if self.skip_projection is not None:
            identity = self.skip_projection(identity)
        
        # Residual connection: F(x) + x
        output = residual + identity
        
        return F.relu(output)  # Final activation
    
    def get_sparsity_stats(self) -> Dict[str, float]:
        """Get sparsity statistics for the residual block"""
        stats = {}
        
        for i, layer in enumerate(self.residual_layers):
            if hasattr(layer, 'mask'):
                density = layer.mask.float().mean().item()
                stats[f'residual_layer_{i}'] = 1.0 - density
        
        if self.skip_projection and hasattr(self.skip_projection, 'mask'):
            density = self.skip_projection.mask.float().mean().item()
            stats['skip_projection'] = 1.0 - density
        
        return stats


class AdaptiveResidualInsertion:
    """
    Adaptive insertion of residual blocks during network growth.
    
    Analyzes network bottlenecks and inserts residual blocks where
    they would be most beneficial.
    """
    
    def __init__(self, 
                 min_block_size: int = 32,
                 max_block_size: int = 256,
                 sparsity_factor: float = 0.02):
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.sparsity_factor = sparsity_factor
    
    def analyze_insertion_points(self, 
                                network: nn.Module,
                                data_loader,
                                device: str = 'cuda') -> List[Dict[str, Any]]:
        """
        Analyze where residual blocks would be most beneficial.
        
        Returns list of insertion points with metadata.
        """
        insertion_points = []
        
        # Get sparse layers
        sparse_layers = [layer for layer in network if isinstance(layer, StandardSparseLayer)]
        
        if len(sparse_layers) < 2:
            return insertion_points
        
        # Analyze gradient flow and activations
        network.eval()
        gradient_norms = []
        activation_variances = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                if batch_idx >= 3:  # Analyze a few batches
                    break
                
                data = data.to(device).view(data.size(0), -1)
                
                # Forward pass to collect activations
                x = data
                batch_activations = []
                
                for layer in sparse_layers:
                    x = F.relu(layer(x))
                    batch_activations.append(x.detach())
                
                activation_variances.append([act.var().item() for act in batch_activations])
        
        # Calculate average activation variances
        avg_variances = np.mean(activation_variances, axis=0)
        
        # Identify bottlenecks (low variance = potential bottleneck)
        for i, variance in enumerate(avg_variances[:-1]):  # Don't insert after last layer
            # Criteria for residual block insertion:
            # 1. Low activation variance (bottleneck)
            # 2. Sufficient layer size
            # 3. Not too close to input/output
            
            layer = sparse_layers[i]
            layer_size = layer.linear.out_features
            
            if (variance < 0.1 and  # Low variance bottleneck
                layer_size >= self.min_block_size and  # Sufficient size
                i > 0 and i < len(sparse_layers) - 2):  # Not at edges
                
                insertion_points.append({
                    'position': i + 1,  # Insert after this layer
                    'layer_size': layer_size,
                    'bottleneck_severity': 1.0 - variance,
                    'recommended_hidden_size': min(layer_size * 2, self.max_block_size)
                })
        
        # Sort by bottleneck severity (most severe first)
        insertion_points.sort(key=lambda x: x['bottleneck_severity'], reverse=True)
        
        return insertion_points
    
    def insert_residual_block(self,
                             network: nn.Module,
                             insertion_point: Dict[str, Any],
                             device: str = 'cuda') -> nn.Module:
        """
        Insert a residual block at the specified position.
        
        Returns new network with residual block inserted.
        """
        position = insertion_point['position']
        hidden_size = insertion_point['recommended_hidden_size']
        
        # Get current network architecture
        sparse_layers = [layer for layer in network if isinstance(layer, StandardSparseLayer)]
        
        if position >= len(sparse_layers):
            print(f"⚠️  Invalid insertion position {position}")
            return network
        
        # Determine residual block dimensions
        if position == 0:
            in_features = sparse_layers[0].linear.in_features
        else:
            in_features = sparse_layers[position - 1].linear.out_features
        
        out_features = sparse_layers[position].linear.in_features
        
        print(f"🔗 Inserting residual block at position {position}")
        print(f"   Dimensions: {in_features} → {hidden_size} → {out_features}")
        
        # Create residual block
        residual_block = SparseResidualBlock(
            in_features=in_features,
            hidden_features=hidden_size,
            out_features=out_features,
            sparsity=self.sparsity_factor,
            num_layers=3,  # 3-layer residual block
            device=device
        )
        
        # Create new network with residual block inserted
        new_network = self._create_network_with_residual_block(
            network, residual_block, position, device
        )
        
        return new_network
    
    def _create_network_with_residual_block(self,
                                          original_network: nn.Module,
                                          residual_block: SparseResidualBlock,
                                          position: int,
                                          device: str) -> nn.Module:
        """Create new network with residual block inserted at position"""
        
        # Get original sparse layers
        original_layers = [layer for layer in original_network if isinstance(layer, StandardSparseLayer)]
        
        # Create new network as ModuleList
        new_layers = nn.ModuleList()
        
        # Add layers before insertion point
        for i in range(position):
            new_layers.append(original_layers[i])
        
        # Add residual block
        new_layers.append(residual_block)
        
        # Add layers after insertion point
        for i in range(position, len(original_layers)):
            new_layers.append(original_layers[i])
        
        # Create new network container
        class ResidualNetwork(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = layers
            
            def forward(self, x):
                for layer in self.layers:
                    if isinstance(layer, SparseResidualBlock):
                        x = layer(x)
                    else:
                        x = F.relu(layer(x))
                return x
            
            def __iter__(self):
                return iter(self.layers)
        
        new_network = ResidualNetwork(new_layers).to(device)
        
        print(f"✅ Created new network with residual block")
        print(f"   Original layers: {len(original_layers)}")
        print(f"   New layers: {len(new_layers)}")
        
        return new_network


class ResidualGrowthStrategy:
    """
    Complete residual growth strategy for structure_net.
    
    Integrates residual block insertion with extrema-driven growth.
    """
    
    def __init__(self, 
                 sparsity: float = 0.02,
                 min_layers_for_residual: int = 3,
                 residual_threshold: float = 0.3):
        self.sparsity = sparsity
        self.min_layers_for_residual = min_layers_for_residual
        self.residual_threshold = residual_threshold
        self.inserter = AdaptiveResidualInsertion(sparsity_factor=sparsity)
    
    def should_add_residual_block(self, 
                                 network: nn.Module,
                                 extrema_analysis: Dict[str, Any]) -> bool:
        """Determine if network would benefit from residual blocks"""
        
        # Get network size
        sparse_layers = [layer for layer in network if isinstance(layer, StandardSparseLayer)]
        
        # Need minimum layers for residual blocks to be beneficial
        if len(sparse_layers) < self.min_layers_for_residual:
            return False
        
        # Check if extrema ratio suggests deep network issues
        extrema_ratio = extrema_analysis.get('extrema_ratio', 0.0)
        
        # High extrema ratio in deep networks suggests gradient flow issues
        # that residual blocks can help with
        if len(sparse_layers) >= 5 and extrema_ratio > self.residual_threshold:
            return True
        
        return False
    
    def add_residual_blocks(self,
                           network: nn.Module,
                           data_loader,
                           device: str = 'cuda',
                           max_blocks: int = 2) -> Tuple[nn.Module, List[Dict[str, Any]]]:
        """
        Add residual blocks to network where beneficial.
        
        Returns:
            Updated network and list of insertion actions
        """
        actions = []
        current_network = network
        
        # Analyze insertion points
        insertion_points = self.inserter.analyze_insertion_points(
            current_network, data_loader, device
        )
        
        if not insertion_points:
            print("🔗 No beneficial residual block insertion points found")
            return current_network, actions
        
        # Insert residual blocks (limit to max_blocks)
        blocks_added = 0
        for point in insertion_points[:max_blocks]:
            if blocks_added >= max_blocks:
                break
            
            print(f"🔗 Adding residual block {blocks_added + 1}/{max_blocks}")
            
            current_network = self.inserter.insert_residual_block(
                current_network, point, device
            )
            
            actions.append({
                'action': 'add_residual_block',
                'position': point['position'],
                'hidden_size': point['recommended_hidden_size'],
                'bottleneck_severity': point['bottleneck_severity']
            })
            
            blocks_added += 1
        
        print(f"✅ Added {blocks_added} residual blocks to network")
        
        return current_network, actions


def create_residual_network(architecture: List[int],
                           sparsity: float = 0.02,
                           residual_positions: List[int] = None,
                           device: str = 'cuda') -> nn.Module:
    """
    Create a network with residual blocks at specified positions.
    
    Args:
        architecture: Base network architecture
        sparsity: Sparsity level for all layers
        residual_positions: Positions where to insert residual blocks
        device: Device to create network on
    
    Returns:
        Network with residual blocks
    """
    if residual_positions is None:
        residual_positions = []
    
    # Create base network
    base_network = create_standard_network(architecture, sparsity, device)
    
    if not residual_positions:
        return base_network
    
    # Insert residual blocks at specified positions
    inserter = AdaptiveResidualInsertion(sparsity_factor=sparsity)
    current_network = base_network
    
    # Sort positions in reverse order to maintain correct indices
    for position in sorted(residual_positions, reverse=True):
        if position < len(architecture) - 1:  # Valid position
            insertion_point = {
                'position': position,
                'recommended_hidden_size': min(architecture[position] * 2, 256)
            }
            
            current_network = inserter.insert_residual_block(
                current_network, insertion_point, device
            )
    
    return current_network


# Export all components
__all__ = [
    'SparseResidualBlock',
    'AdaptiveResidualInsertion',
    'ResidualGrowthStrategy',
    'create_residual_network'
]
