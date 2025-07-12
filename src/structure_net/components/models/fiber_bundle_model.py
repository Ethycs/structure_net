"""
Fiber Bundle Model Component.

Neural network with explicit fiber bundle geometry for:
- Gauge-invariant optimization
- Catastrophe-aware growth
- Multi-class neuron tracking
- Geometric regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import logging

from ...core import (
    BaseModel, ComponentContract, ComponentVersion,
    Maturity, ResourceRequirements, ResourceLevel,
    ILayer, EvolutionContext
)


@dataclass
class FiberBundleConfig:
    """Configuration for fiber bundle model."""
    base_dim: int  # Number of layers (base space dimension)
    fiber_dim: int  # Width of each layer (fiber dimension)
    initial_sparsity: float = 0.98
    
    # Geometric constraints
    max_curvature: float = 1.0
    max_holonomy: float = 0.1
    gauge_regularization: float = 0.01
    
    # Multi-class neuron constraints
    max_classes_per_neuron: int = 3
    specialization_pressure: float = 0.1
    
    # Growth strategy
    growth_strategy: str = "curvature_guided"
    
    # Integration flags
    use_homological_metrics: bool = True
    use_compactification: bool = False


class StructuredLayer(nn.Module, ILayer):
    """
    Layer with gauge-preserving structured connections.
    
    Implements ILayer interface for component compatibility.
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 sparsity: float = 0.98, name: str = None):
        """Initialize structured layer."""
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.target_sparsity = sparsity
        self._name = name or f"StructuredLayer_{in_features}x{out_features}"
        
        # Parameters
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Binary mask for sparsity
        self.register_buffer('mask', torch.zeros_like(self.weight))
        
        # Gauge field for connection strengths
        self.register_buffer('gauge_field', torch.ones_like(self.weight))
        
        # Class assignments for multi-class neurons
        self.register_buffer('neuron_classes', torch.zeros(out_features, dtype=torch.long))
        
        # Initialize with structured sparsity
        self._initialize_structured_sparsity()
        
        # Component contract
        self._contract = ComponentContract(
            component_name=self._name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={'input'},
            provided_outputs={'output', 'analysis_properties'},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False
            )
        )
    
    @property
    def name(self) -> str:
        """Component name."""
        return self._name
    
    @property
    def contract(self) -> ComponentContract:
        """Component contract."""
        return self._contract
    
    def _initialize_structured_sparsity(self):
        """Initialize with gauge-respecting sparsity pattern."""
        num_connections = int((1 - self.target_sparsity) * self.in_features * self.out_features)
        
        # Create structured connections preserving permutation symmetry
        connections_per_neuron = max(1, num_connections // self.out_features)
        
        for i in range(self.out_features):
            # Each output connects to a structured set of inputs
            start_idx = (i * connections_per_neuron) % self.in_features
            for j in range(connections_per_neuron):
                in_idx = (start_idx + j) % self.in_features
                self.mask[i, in_idx] = 1
                
                # Initialize weight with proper scaling
                self.weight.data[i, in_idx] = torch.randn(1) * np.sqrt(2.0 / connections_per_neuron)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gauge field modulation."""
        # Apply mask and gauge field
        effective_weight = self.weight * self.mask * self.gauge_field
        return F.linear(x, effective_weight, self.bias)
    
    def get_analysis_properties(self) -> Dict[str, torch.Tensor]:
        """Get properties for analysis."""
        return {
            'weight': self.weight,
            'mask': self.mask,
            'gauge_field': self.gauge_field,
            'effective_weight': self.weight * self.mask * self.gauge_field,
            'bias': self.bias,
            'neuron_classes': self.neuron_classes,
            'sparsity': torch.tensor(1.0 - self.mask.mean().item())
        }
    
    def supports_modification(self) -> bool:
        """Layer supports structural modifications."""
        return True
    
    def add_connections(self, num_new: int, preserve_gauge: bool = True):
        """Add new connections while preserving gauge structure."""
        current_connections = self.mask.sum().item()
        
        # Find unconnected pairs
        unconnected = (self.mask == 0).nonzero(as_tuple=False)
        
        if len(unconnected) == 0:
            return
        
        # Select new connections preserving gauge symmetry
        if preserve_gauge:
            # Prefer connections that maintain local gauge invariance
            scores = []
            for idx in range(len(unconnected)):
                i, j = unconnected[idx]
                # Score based on existing connection pattern
                row_density = self.mask[i].sum() / self.in_features
                col_density = self.mask[:, j].sum() / self.out_features
                score = 1.0 / (1.0 + row_density + col_density)
                scores.append(score)
            
            scores = torch.tensor(scores)
            _, indices = torch.topk(scores, min(num_new, len(scores)))
        else:
            indices = torch.randperm(len(unconnected))[:num_new]
        
        for idx in indices:
            i, j = unconnected[idx]
            self.mask[i, j] = 1
            
            # Initialize new weight
            fan_in = self.mask[i].sum().item()
            self.weight.data[i, j] = torch.randn(1) * np.sqrt(2.0 / fan_in)
            
            # Initialize gauge field value
            self.gauge_field[i, j] = 1.0
    
    def compute_curvature(self) -> torch.Tensor:
        """Compute connection curvature (gauge field strength)."""
        # Simplified curvature: variation in gauge field
        gauge_grad_x = self.gauge_field[:, 1:] - self.gauge_field[:, :-1]
        gauge_grad_y = self.gauge_field[1:, :] - self.gauge_field[:-1, :]
        
        # Pad to original size
        gauge_grad_x = F.pad(gauge_grad_x, (0, 1))
        gauge_grad_y = F.pad(gauge_grad_y, (1, 0))
        
        # Curvature magnitude
        curvature = torch.sqrt(gauge_grad_x**2 + gauge_grad_y**2)
        return curvature
    
    def compute_holonomy(self, path_length: int = 4) -> float:
        """Compute holonomy (parallel transport around loops)."""
        # Simplified: product of gauge field values around small loops
        holonomy = 0.0
        count = 0
        
        h, w = self.gauge_field.shape
        for i in range(h - path_length):
            for j in range(w - path_length):
                # Compute product around square loop
                loop_product = (
                    self.gauge_field[i, j] *
                    self.gauge_field[i, j + path_length - 1] *
                    self.gauge_field[i + path_length - 1, j + path_length - 1] *
                    self.gauge_field[i + path_length - 1, j]
                )
                holonomy += abs(loop_product - 1.0)
                count += 1
        
        return holonomy / max(count, 1)


class FiberBundleModel(BaseModel):
    """
    Neural network with explicit fiber bundle structure.
    
    Key concepts:
    - Base space: Layer indices [0, 1, ..., L]
    - Fiber: Activation space at each layer
    - Connection: Weight matrices with gauge symmetry
    - Parallel transport: Information flow through layers
    """
    
    def __init__(self, config: FiberBundleConfig, name: str = None):
        """
        Initialize fiber bundle model.
        
        Args:
            config: Fiber bundle configuration
            name: Optional custom name
        """
        super().__init__(name or "FiberBundleModel")
        
        self.config = config
        
        # Build fiber bundle structure
        self._build_bundle()
        
        # Geometric tracking
        self.curvature_history: List[float] = []
        self.holonomy_history: List[float] = []
        
        # Multi-class neuron tracking
        self.class_evolution: Dict[int, List[int]] = {}
        
        # Define contract
        self._contract = ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={'input'},
            provided_outputs={
                'output',
                'model.architecture',
                'model.geometric_properties',
                'model.class_distribution'
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
    
    def _build_bundle(self):
        """Build the fiber bundle structure."""
        self.fibers = nn.ModuleList()
        self._layers = []  # For ILayer interface
        
        # Create structured connections between fibers
        for i in range(self.config.base_dim - 1):
            layer = StructuredLayer(
                self.config.fiber_dim,
                self.config.fiber_dim,
                self.config.initial_sparsity,
                name=f"{self.name}_fiber_{i}"
            )
            
            self.fibers.append(layer)
            self._layers.append(layer)
            
            # Add activation
            if i < self.config.base_dim - 2:
                self.fibers.append(nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with geometric tracking.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Track activations for geometric analysis
        self.fiber_activations = []
        
        # Ensure correct input dimension
        if x.shape[-1] != self.config.fiber_dim:
            raise ValueError(f"Input dimension {x.shape[-1]} doesn't match fiber dimension {self.config.fiber_dim}")
        
        # Parallel transport through fibers
        for i, module in enumerate(self.fibers):
            x = module(x)
            
            # Store fiber activations
            if isinstance(module, StructuredLayer):
                self.fiber_activations.append(x.detach())
                
                # Update geometric quantities
                curvature = module.compute_curvature()
                self.current_curvature = curvature.mean().item()
                
                holonomy = module.compute_holonomy()
                self.current_holonomy = holonomy
        
        return x
    
    def get_layers(self) -> List[ILayer]:
        """Get all layers in the model."""
        return self._layers
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        active_connections = sum(
            layer.mask.sum().item() for layer in self._layers
        )
        
        return {
            'base_dimension': self.config.base_dim,
            'fiber_dimension': self.config.fiber_dim,
            'total_parameters': total_params,
            'active_connections': active_connections,
            'sparsity': 1.0 - (active_connections / total_params if total_params > 0 else 0),
            'num_fibers': len(self._layers),
            'growth_strategy': self.config.growth_strategy,
            'gauge_regularization': self.config.gauge_regularization
        }
    
    def get_geometric_properties(self) -> Dict[str, Any]:
        """Get current geometric properties."""
        properties = {
            'curvatures': [],
            'holonomies': [],
            'gauge_fields': [],
            'total_curvature': 0.0,
            'total_holonomy': 0.0
        }
        
        for layer in self._layers:
            curvature = layer.compute_curvature()
            holonomy = layer.compute_holonomy()
            
            properties['curvatures'].append(curvature.mean().item())
            properties['holonomies'].append(holonomy)
            properties['gauge_fields'].append(layer.gauge_field.clone())
            
            properties['total_curvature'] += curvature.sum().item()
            properties['total_holonomy'] += holonomy
        
        # Check geometric constraints
        properties['curvature_violation'] = max(0, properties['total_curvature'] - self.config.max_curvature)
        properties['holonomy_violation'] = max(0, properties['total_holonomy'] - self.config.max_holonomy)
        
        return properties
    
    def get_class_distribution(self) -> Dict[str, Any]:
        """Get multi-class neuron distribution."""
        distribution = {
            'neurons_per_class': {},
            'class_specialization': [],
            'entropy': 0.0
        }
        
        for layer_idx, layer in enumerate(self._layers):
            class_counts = torch.bincount(layer.neuron_classes)
            
            for class_id in range(len(class_counts)):
                if class_id not in distribution['neurons_per_class']:
                    distribution['neurons_per_class'][class_id] = 0
                distribution['neurons_per_class'][class_id] += class_counts[class_id].item()
            
            # Compute specialization (how concentrated classes are)
            if len(class_counts) > 0:
                probs = class_counts.float() / class_counts.sum()
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                distribution['class_specialization'].append({
                    'layer': layer_idx,
                    'entropy': entropy,
                    'dominant_class': class_counts.argmax().item()
                })
        
        return distribution
    
    def enforce_gauge_constraint(self, max_change: float = 0.1):
        """Enforce gauge constraints on the network."""
        for layer in self._layers:
            # Smooth gauge field to reduce curvature
            gauge_field = layer.gauge_field
            
            # Apply Gaussian smoothing
            kernel = torch.tensor([[0.25, 0.5, 0.25]]).view(1, 1, 1, 3)
            smoothed = F.conv1d(
                gauge_field.unsqueeze(0).unsqueeze(0),
                kernel,
                padding=1
            ).squeeze()
            
            # Limit change to preserve stability
            change = smoothed - gauge_field
            change = torch.clamp(change, -max_change, max_change)
            
            layer.gauge_field.data += change
    
    def evolve_class_assignments(self, activations: torch.Tensor, 
                               labels: Optional[torch.Tensor] = None):
        """
        Evolve neuron class assignments based on activations.
        
        Args:
            activations: Current layer activations
            labels: Optional class labels for supervised evolution
        """
        if labels is None:
            # Unsupervised: cluster based on activation patterns
            # This is a simplified version
            for layer_idx, layer in enumerate(self._layers):
                if layer_idx < len(self.fiber_activations):
                    acts = self.fiber_activations[layer_idx]
                    
                    # Simple k-means style assignment
                    # In practice, use more sophisticated clustering
                    mean_acts = acts.mean(dim=0)
                    for i in range(layer.out_features):
                        # Assign to class based on activation level
                        if mean_acts[i] > mean_acts.mean():
                            layer.neuron_classes[i] = 1
                        else:
                            layer.neuron_classes[i] = 0
        else:
            # Supervised: assign based on contribution to correct class
            # This would require gradient analysis
            pass
    
    def add_fiber_connection(self, fiber_idx: int, num_connections: int):
        """Add connections to specific fiber."""
        if 0 <= fiber_idx < len(self._layers):
            self._layers[fiber_idx].add_connections(
                num_connections, 
                preserve_gauge=True
            )
            
            # Log geometric change
            new_curvature = self._layers[fiber_idx].compute_curvature().mean().item()
            self.log(logging.INFO,
                    f"Added {num_connections} connections to fiber {fiber_idx}, "
                    f"new curvature: {new_curvature:.4f}")