"""
Multi-Scale Model Component.

A neural network that supports multi-scale growth and evolution,
with dynamic architecture modification capabilities.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple

from ...core import (
    BaseModel, ComponentContract, ComponentVersion,
    Maturity, ResourceRequirements, ResourceLevel,
    ILayer, EvolutionContext
)
from ..layers.sparse_layer import SparseLayer


class MultiScaleBlock(nn.Module):
    """
    A block that can process features at multiple scales.
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 scales: List[int], sparsity: float = 0.9):
        """
        Initialize multi-scale block.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension per scale
            scales: List of scale factors (e.g., [1, 2, 4])
            sparsity: Sparsity level for connections
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.scales = scales
        
        # Create sparse layers for each scale
        self.scale_layers = nn.ModuleList()
        for scale in scales:
            # Adjust features based on scale
            scale_out = out_features // scale
            self.scale_layers.append(
                SparseLayer(in_features, scale_out, sparsity)
            )
        
        # Fusion layer to combine scales
        total_scale_features = sum(out_features // s for s in scales)
        self.fusion = nn.Linear(total_scale_features, out_features)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass processing multiple scales.
        
        Returns:
            output: Fused multi-scale features
            scale_outputs: Individual scale outputs for analysis
        """
        scale_outputs = {}
        scale_features = []
        
        for i, (scale, layer) in enumerate(zip(self.scales, self.scale_layers)):
            # Process at this scale
            scale_out = layer(x)
            scale_outputs[f'scale_{scale}'] = scale_out
            scale_features.append(scale_out)
        
        # Concatenate and fuse
        combined = torch.cat(scale_features, dim=-1)
        output = self.fusion(combined)
        
        return output, scale_outputs


class MultiScaleModel(BaseModel):
    """
    A neural network supporting multi-scale processing and dynamic growth.
    
    Features:
    - Multi-scale feature extraction
    - Dynamic architecture modification
    - Scale-aware growth strategies
    - Hierarchical feature fusion
    """
    
    def __init__(self,
                 initial_architecture: List[int],
                 scales: List[int] = [1, 2, 4],
                 initial_sparsity: float = 0.9,
                 use_multi_scale_blocks: bool = True,
                 name: str = None):
        """
        Initialize multi-scale model.
        
        Args:
            initial_architecture: Initial layer sizes
            scales: Scale factors for multi-scale processing
            initial_sparsity: Initial sparsity level
            use_multi_scale_blocks: Whether to use multi-scale blocks
            name: Optional custom name
        """
        super().__init__(name or "MultiScaleModel")
        
        self.architecture = initial_architecture
        self.scales = scales
        self.sparsity = initial_sparsity
        self.use_multi_scale_blocks = use_multi_scale_blocks
        
        # Build initial network
        self._build_network()
        
        # Track growth history
        self.growth_history: List[Dict[str, Any]] = []
        self.scale_importance: Dict[int, float] = {s: 1.0/len(scales) for s in scales}
        
        # Define contract
        self._contract = ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={'input'},
            provided_outputs={
                'output',
                'model.architecture',
                'model.scale_features',
                'model.growth_potential'
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    @property
    def contract(self) -> ComponentContract:
        """Return component contract."""
        return self._contract
    
    def _build_network(self):
        """Build the network architecture."""
        self.blocks = nn.ModuleList()
        self._layers = []  # For ILayer interface
        
        for i in range(len(self.architecture) - 1):
            in_features = self.architecture[i]
            out_features = self.architecture[i + 1]
            
            if self.use_multi_scale_blocks and i < len(self.architecture) - 2:
                # Use multi-scale block for hidden layers
                block = MultiScaleBlock(
                    in_features, out_features,
                    self.scales, self.sparsity
                )
                self.blocks.append(block)
                
                # Add scale layers to layer list
                for scale_layer in block.scale_layers:
                    if isinstance(scale_layer, ILayer):
                        self._layers.append(scale_layer)
            else:
                # Use standard sparse layer
                layer = SparseLayer(
                    in_features, out_features,
                    self.sparsity, name=f"{self.name}_layer_{i}"
                )
                self.blocks.append(layer)
                self._layers.append(layer)
            
            # Add activation (except last layer)
            if i < len(self.architecture) - 2:
                self.blocks.append(nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale processing.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Store scale features for analysis
        self.scale_features = {}
        
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Process through blocks
        for i, block in enumerate(self.blocks):
            if isinstance(block, MultiScaleBlock):
                x, scale_outputs = block(x)
                self.scale_features[f'block_{i}'] = scale_outputs
            else:
                x = block(x)
        
        return x
    
    def get_layers(self) -> List[ILayer]:
        """Get all layers in the model."""
        return self._layers
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        
        # Count multi-scale blocks
        num_ms_blocks = sum(1 for b in self.blocks if isinstance(b, MultiScaleBlock))
        
        # Get scale utilization
        scale_stats = {}
        for scale in self.scales:
            scale_params = sum(
                p.numel() for block in self.blocks 
                if isinstance(block, MultiScaleBlock)
                for i, s in enumerate(block.scales)
                if s == scale
                for p in block.scale_layers[i].parameters()
            )
            scale_stats[f'scale_{scale}_params'] = scale_params
        
        return {
            'architecture': self.architecture,
            'total_parameters': total_params,
            'num_layers': len(self._layers),
            'num_multi_scale_blocks': num_ms_blocks,
            'scales': self.scales,
            'scale_statistics': scale_stats,
            'sparsity': self.sparsity,
            'depth': len(self.architecture) - 1
        }
    
    def get_growth_potential(self) -> Dict[str, Any]:
        """Analyze potential for network growth."""
        potential = {
            'layer_growth': [],
            'scale_growth': {},
            'recommended_growth': []
        }
        
        # Analyze each layer's capacity for growth
        for i, layer in enumerate(self._layers):
            if hasattr(layer, 'get_analysis_properties'):
                props = layer.get_analysis_properties()
                
                # Check utilization
                if 'mask' in props:
                    utilization = props['mask'].sum().item() / props['mask'].numel()
                    
                    layer_potential = {
                        'layer_index': i,
                        'current_utilization': utilization,
                        'growth_capacity': 1.0 - utilization,
                        'recommended_connections': int((1.0 - utilization) * props['mask'].numel() * 0.1)
                    }
                    potential['layer_growth'].append(layer_potential)
        
        # Analyze scale importance for growth
        for scale, importance in self.scale_importance.items():
            potential['scale_growth'][f'scale_{scale}'] = {
                'importance': importance,
                'should_grow': importance > 1.0 / len(self.scales)
            }
        
        # Generate growth recommendations
        if potential['layer_growth']:
            # Find layers with low utilization
            for layer_info in potential['layer_growth']:
                if layer_info['growth_capacity'] > 0.5:
                    potential['recommended_growth'].append({
                        'type': 'add_connections',
                        'layer': layer_info['layer_index'],
                        'amount': layer_info['recommended_connections']
                    })
        
        return potential
    
    def add_layer(self, position: int, width: int, 
                  use_multi_scale: Optional[bool] = None):
        """
        Add a new layer at the specified position.
        
        Args:
            position: Position in architecture (0 to len-1)
            width: Width of new layer
            use_multi_scale: Whether to use multi-scale block
        """
        if position < 0 or position > len(self.architecture) - 1:
            raise ValueError(f"Invalid position {position}")
        
        # Update architecture
        self.architecture.insert(position, width)
        
        # Rebuild network (simple approach)
        # In practice, you'd want to preserve existing weights
        old_state = self.state_dict()
        self._build_network()
        
        # Try to restore weights where possible
        new_state = self.state_dict()
        for key in old_state:
            if key in new_state and old_state[key].shape == new_state[key].shape:
                new_state[key] = old_state[key]
        self.load_state_dict(new_state)
        
        # Log growth
        self.growth_history.append({
            'type': 'add_layer',
            'position': position,
            'width': width,
            'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        })
        
        self.log(logging.INFO, f"Added layer at position {position} with width {width}")
    
    def grow_scale(self, block_idx: int, scale: int, growth_factor: float = 1.5):
        """
        Grow a specific scale in a multi-scale block.
        
        Args:
            block_idx: Index of the block to grow
            scale: Scale to grow
            growth_factor: How much to grow (multiplier on connections)
        """
        if block_idx >= len(self.blocks):
            return
        
        block = self.blocks[block_idx]
        if not isinstance(block, MultiScaleBlock):
            return
        
        # Find the scale layer
        for i, s in enumerate(block.scales):
            if s == scale:
                scale_layer = block.scale_layers[i]
                if hasattr(scale_layer, 'add_connections'):
                    current_connections = scale_layer.mask.sum().item()
                    new_connections = int(current_connections * (growth_factor - 1))
                    scale_layer.add_connections(new_connections)
                    
                    self.log(logging.INFO,
                            f"Grew scale {scale} in block {block_idx} by {new_connections} connections")
                break
    
    def update_scale_importance(self, scale_gradients: Dict[int, float]):
        """
        Update importance weights for different scales based on gradients.
        
        Args:
            scale_gradients: Gradient magnitudes for each scale
        """
        # Normalize gradients
        total_grad = sum(scale_gradients.values())
        if total_grad > 0:
            for scale in self.scales:
                if scale in scale_gradients:
                    # Update with momentum
                    new_importance = scale_gradients[scale] / total_grad
                    self.scale_importance[scale] = (
                        0.9 * self.scale_importance[scale] + 
                        0.1 * new_importance
                    )
    
    def get_scale_features(self) -> Dict[str, torch.Tensor]:
        """Get the most recent scale features from forward pass."""
        return self.scale_features
    
    def prune_scale(self, scale: int, prune_ratio: float = 0.1):
        """
        Prune connections from a specific scale across all blocks.
        
        Args:
            scale: Scale to prune
            prune_ratio: Fraction of connections to remove
        """
        pruned_total = 0
        
        for block in self.blocks:
            if isinstance(block, MultiScaleBlock):
                for i, s in enumerate(block.scales):
                    if s == scale:
                        scale_layer = block.scale_layers[i]
                        if hasattr(scale_layer, 'prune_connections'):
                            active = scale_layer.mask.sum().item()
                            to_prune = int(active * prune_ratio)
                            scale_layer.prune_connections(to_prune)
                            pruned_total += to_prune
        
        self.log(logging.INFO, f"Pruned {pruned_total} connections from scale {scale}")
        
        return pruned_total