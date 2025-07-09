"""
Fiber Bundle Neural Network Implementation

Implements neural networks with explicit fiber bundle geometry for:
- Gauge-invariant optimization
- Catastrophe-aware growth
- Multi-class neuron tracking
- Geometric regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

# Assuming these are the correct relative imports for your project structure.
# If these are incorrect, they may need to be adjusted.
from ..core.layers import SparseLinear
from ..evolution.metrics.base import BaseMetricAnalyzer
from ..evolution.metrics.homological_analysis import HomologicalAnalyzer
from ..evolution.metrics.sensitivity_analysis import SensitivityAnalyzer
# from ..evolution.components.evolution_system import EvolutionSystem
# from ..logging.wandb_integration import WandBLogger

logger = logging.getLogger(__name__)


@dataclass
class FiberBundleConfig:
    """Configuration for fiber bundle network"""
    base_dim: int  # Number of layers (base space dimension)
    fiber_dim: int  # Width of each layer (fiber dimension)
    initial_sparsity: float = 0.02
    growth_rate: float = 0.1
    
    # Geometric constraints
    max_curvature: float = 1.0
    max_holonomy: float = 0.1
    gauge_regularization: float = 0.01
    
    # Multi-class neuron constraints
    max_classes_per_neuron: int = 3
    specialization_pressure: float = 0.1
    
    # Growth strategy
    growth_strategy: str = "curvature_guided"  # or "holonomy_minimal", "catastrophe_avoiding"
    
    # Integration with existing system
    use_homological_metrics: bool = True
    use_compactification: bool = False


class StructuredConnection(nn.Module):
    """
    Connection between fibers with structure preservation
    
    Maintains gauge invariance and allows controlled growth
    """
    
    def __init__(self, in_features: int, out_features: int, sparsity: float = 0.98):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.target_sparsity = sparsity
        
        # Initialize sparse weight matrix
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Binary mask for sparsity
        self.register_buffer('mask', torch.zeros_like(self.weight))
        
        # Initialize with structured sparsity
        self._initialize_structured_sparsity()
    
    def _initialize_structured_sparsity(self):
        """Initialize with gauge-respecting sparsity pattern"""
        num_connections = int((1 - self.target_sparsity) * self.in_features * self.out_features)
        
        # Create structured connections (not random)
        # Connect in a way that preserves permutation symmetry
        connections_per_neuron = max(1, num_connections // self.out_features)
        
        for i in range(self.out_features):
            # Each output connects to a small, structured set of inputs
            start_idx = (i * connections_per_neuron) % self.in_features
            for j in range(connections_per_neuron):
                in_idx = (start_idx + j) % self.in_features
                self.mask[i, in_idx] = 1
                
                # Initialize weight
                self.weight.data[i, in_idx] = torch.randn(1) * np.sqrt(2.0 / connections_per_neuron)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse matrix multiplication with bias"""
        masked_weight = self.weight * self.mask
        return F.linear(x, masked_weight, self.bias)
    
    def weight_matrix(self) -> torch.Tensor:
        """Get the effective weight matrix"""
        return self.weight * self.mask
    
    def add_connections(self, num_new: int):
        """Add new connections while preserving structure"""
        current_connections = self.mask.sum().item()
        
        # Find unconnected pairs
        unconnected = (self.mask == 0).nonzero(as_tuple=False)
        
        if len(unconnected) == 0:
            return
        
        # Select new connections to add
        # Prefer connections that maintain gauge symmetry
        indices = torch.randperm(len(unconnected))[:num_new]
        
        for idx in indices:
            i, j = unconnected[idx]
            self.mask[i, j] = 1
            
            # Initialize new weight
            fan_in = self.mask[i].sum().item()
            self.weight.data[i, j] = torch.randn(1) * np.sqrt(2.0 / fan_in)
    
    def get_sparsity_info(self) -> Dict[str, int]:
        """Get sparsity information"""
        total = self.in_features * self.out_features
        active = int(self.mask.sum().item())
        return {
            'total': total,
            'active': active,
            'sparsity': 1 - (active / total)
        }


class FiberBundle(nn.Module):
    """
    Neural network with explicit fiber bundle structure
    
    Key concepts:
    - Base space: Layer indices [0, 1, ..., L]
    - Fiber: Activation space at each layer (R^n)
    - Connection: Weight matrices with gauge symmetry
    - Parallel transport: Information flow through layers
    """
    
    def __init__(self, config: FiberBundleConfig):
        super().__init__()
        self.config = config
        
        # Initialize bundle structure
        self.fibers = nn.ModuleList()
        self.connections = nn.ModuleList()
        
        # Tracking structures
        self.curvature_history = []
        self.holonomy_measurements = []
        self.catastrophe_locations = []
        self.neuron_class_associations = {}
        
        # Initialize with sparse seed
        self._initialize_minimal_bundle()
        
        # Integration with existing analysis tools
        self.homological_analyzer = HomologicalAnalyzer() if config.use_homological_metrics else None
        self.sensitivity_analyzer = SensitivityAnalyzer(self)
        
    def _initialize_minimal_bundle(self):
        """Initialize network with minimal viable connections"""
        # This assumes a fixed fiber_dim for all layers for simplicity.
        # A more advanced version could have variable fiber dimensions.
        for layer_idx in range(self.config.base_dim):
            # Create fiber (layer)
            # Using SparseLinear for the fibers themselves, as they have internal structure
            fiber = SparseLinear(
                in_features=self.config.fiber_dim,
                out_features=self.config.fiber_dim,
                sparsity=self.config.initial_sparsity
            )
            self.fibers.append(fiber)
            
            # Create connection to next layer
            if layer_idx < self.config.base_dim - 1:
                connection = StructuredConnection(
                    self.config.fiber_dim,
                    self.config.fiber_dim,
                    sparsity=self.config.initial_sparsity
                )
                self.connections.append(connection)
    
    def forward(self, x: torch.Tensor, return_activations: bool = False) -> torch.Tensor:
        """
        Forward pass with optional activation tracking
        
        Implements parallel transport through the fiber bundle
        """
        activations = [x] if return_activations else None
        
        h = x
        for idx, (fiber, connection) in enumerate(zip(self.fibers[:-1], self.connections)):
            # Apply fiber transformation
            h = fiber(h)
            
            # Parallel transport to next fiber
            h = connection(h)
            
            # Nonlinearity (creates curvature)
            h = F.relu(h)
            
            if return_activations:
                activations.append(h.clone())
        
        # Final fiber
        h = self.fibers[-1](h)
        
        if return_activations:
            activations.append(h)
            return h, activations
        return h
    
    def compute_connection_curvature(self, layer_idx: int) -> torch.Tensor:
        """
        Compute curvature of connection at given layer
        
        Uses commutator of adjacent connections as curvature measure
        """
        if layer_idx >= len(self.connections) - 1:
            return torch.tensor(0.0)
        
        W1 = self.connections[layer_idx].weight_matrix()
        W2 = self.connections[layer_idx + 1].weight_matrix()
        
        # Approximate curvature via commutator
        # This is a simplification. True curvature is more complex.
        commutator = torch.matmul(W2, W1) - torch.matmul(W1.T, W2.T)
        curvature = torch.norm(commutator, 'fro')
        
        return curvature
    
    def measure_holonomy(self, test_vectors: torch.Tensor) -> float:
        """
        Measure holonomy by transporting vectors through network and back
        """
        device = next(self.parameters()).device
        test_vectors = test_vectors.to(device)
        
        # Forward transport
        h_forward, _ = self.forward(test_vectors, return_activations=True)
        
        # Backward transport (approximate inverse)
        h_back = h_forward
        for idx in reversed(range(len(self.connections))):
            # Approximate inverse transport
            W = self.connections[idx].weight_matrix()
            W_pinv = torch.pinverse(W)
            h_back = torch.matmul(h_back, W_pinv.T)
        
        # Measure deviation
        holonomy = torch.norm(h_back - test_vectors) / torch.norm(test_vectors)
        return holonomy.item()
    
    def grow_network(self, growth_data: Dict[str, Any]):
        """
        Grow network based on geometric principles
        """
        strategy = self.config.growth_strategy
        
        if strategy == "curvature_guided":
            self._grow_curvature_guided(growth_data)
        elif strategy == "holonomy_minimal":
            self._grow_holonomy_minimal(growth_data)
        elif strategy == "catastrophe_avoiding":
            self._grow_catastrophe_avoiding(growth_data)
        else:
            raise ValueError(f"Unknown growth strategy: {strategy}")
    
    def _grow_curvature_guided(self, growth_data: Dict[str, Any]):
        """Add connections where curvature is highest"""
        curvatures = []
        for idx in range(len(self.connections)):
            curv = self.compute_connection_curvature(idx)
            curvatures.append((idx, curv))
        
        curvatures.sort(key=lambda x: x[1], reverse=True)
        
        num_to_grow = int(self.config.growth_rate * len(self.connections))
        for idx, _ in curvatures[:num_to_grow]:
            self.connections[idx].add_connections(
                num_new=int(self.config.fiber_dim * 0.01)
            )
    
    def _grow_holonomy_minimal(self, growth_data: Dict[str, Any]):
        """Add connections to minimize holonomy"""
        test_vectors = torch.randn(100, self.config.fiber_dim, device=next(self.parameters()).device)
        
        transport_losses = []
        h = test_vectors
        for idx, connection in enumerate(self.connections):
            h_next = connection(F.relu(h))
            
            W = connection.weight_matrix()
            W_pinv = torch.pinverse(W)
            h_reconstructed = torch.matmul(h_next, W_pinv.T)
            
            loss = torch.norm(h_reconstructed - h)
            transport_losses.append((idx, loss))
            
            h = h_next
        
        transport_losses.sort(key=lambda x: x[1], reverse=True)
        num_to_grow = int(self.config.growth_rate * len(self.connections))
        
        for idx, _ in transport_losses[:num_to_grow]:
            self.connections[idx].add_connections(
                num_new=int(self.config.fiber_dim * 0.01)
            )

    def _grow_catastrophe_avoiding(self, growth_data: Dict[str, Any]):
        """Add connections to bypass regions prone to catastrophic events."""
        test_inputs = growth_data.get('test_inputs')
        if test_inputs is None:
            logger.warning("Catastrophe-avoiding growth requires 'test_inputs' in growth_data.")
            return

        catastrophes = self.detect_catastrophe_points(test_inputs)
        
        # This is a simplified representation of a "density map".
        # A real implementation would be more sophisticated.
        cat_density = torch.zeros(self.config.base_dim - 1)
        for idx in catastrophes:
            # For simplicity, assume catastrophes are linked to specific layers.
            # A more complex mapping would be needed in a real scenario.
            layer_idx = idx % (self.config.base_dim - 1)
            cat_density[layer_idx] += 1

        if cat_density.sum() == 0:
            return # No catastrophes detected

        # Grow connections to bypass the most catastrophic regions
        hotspots = torch.topk(cat_density, k=int(self.config.growth_rate * len(self.connections))).indices
        for layer_idx in hotspots:
            # Add connections in the layer *before* the hotspot to reroute info
            if layer_idx > 0:
                self.connections[layer_idx - 1].add_connections(
                    num_new=int(self.config.fiber_dim * 0.01)
                )

    def _grow_class_aware(self, dataloader):
        """Grow new fibers to relieve pressure on overloaded neurons."""
        analysis = self.analyze_multiclass_neurons(dataloader)
        overloaded_mask = analysis['classes_per_neuron'] > self.config.max_classes_per_neuron
        
        if overloaded_mask.any():
            # This is a simplified growth mechanism. A real implementation would be more complex.
            # For now, we just add connections to the corresponding connection module.
            layer_idx = -2 # Hardcoded for simplicity, corresponds to the layer analyzed
            connection_idx = len(self.connections) + layer_idx
            
            num_new_connections = int(overloaded_mask.sum() * 0.05 * self.config.fiber_dim)
            self.connections[connection_idx].add_connections(num_new=num_new_connections)
    
    def detect_catastrophe_points(self, test_inputs: torch.Tensor, epsilon: float = 0.01) -> List[int]:
        """
        Detect catastrophe points in the network
        """
        device = next(self.parameters()).device
        test_inputs = test_inputs.to(device)
        
        clean_outputs = self.forward(test_inputs)
        clean_preds = clean_outputs.argmax(dim=1)
        
        catastrophic_indices = []
        for i in range(test_inputs.shape[0]):
            x = test_inputs[i:i+1]
            
            perturbed = x + epsilon * torch.randn_like(x)
            perturbed_output = self.forward(perturbed)
            perturbed_pred = perturbed_output.argmax(dim=1)
            
            if perturbed_pred != clean_preds[i]:
                catastrophic_indices.append(i)
        
        return catastrophic_indices
    
    def analyze_multiclass_neurons(self, dataloader, layer_idx: int = -2) -> Dict[str, Any]:
        """
        Analyze which neurons respond to multiple classes
        """
        self.eval()
        device = next(self.parameters()).device
        
        num_classes = 10 # Assuming 10 classes for now
        class_activations = {i: [] for i in range(num_classes)}
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                _, activations = self.forward(inputs, return_activations=True)
                layer_acts = activations[layer_idx]
                
                for i in range(inputs.shape[0]):
                    class_idx = labels[i].item()
                    if class_idx < num_classes:
                        class_activations[class_idx].append(layer_acts[i].cpu())
        
        analysis = self._compute_multiclass_statistics(class_activations)
        self.neuron_class_associations[layer_idx] = analysis
        return analysis
    
    def _compute_multiclass_statistics(self, class_activations: Dict[int, List[torch.Tensor]]) -> Dict[str, Any]:
        """Compute statistics about multi-class neuron responses"""
        mean_activations = {}
        for class_idx, acts in class_activations.items():
            if acts:
                stacked = torch.stack(acts)
                mean_activations[class_idx] = stacked.mean(dim=0)
        
        if not mean_activations:
            return {}
        
        num_neurons = list(mean_activations.values())[0].shape[0]
        num_classes = len(mean_activations)
        
        neuron_class_matrix = torch.zeros(num_neurons, num_classes)
        for class_idx, mean_act in mean_activations.items():
            neuron_class_matrix[:, class_idx] = mean_act
        
        threshold = 0.3
        classes_per_neuron = (neuron_class_matrix > threshold).sum(dim=1)
        
        return {
            'neuron_class_matrix': neuron_class_matrix,
            'classes_per_neuron': classes_per_neuron,
            'multi_class_count': int((classes_per_neuron >= 2).sum().item()),
            'highly_selective_count': int((classes_per_neuron == 1).sum().item()),
            'dead_neurons': int((classes_per_neuron == 0).sum().item()),
            'promiscuous_neurons': int((classes_per_neuron >= 5).sum().item()),
            'mean_classes_per_neuron': float(classes_per_neuron.float().mean().item())
        }
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get metrics compatible with existing logging system
        """
        metrics = {}
        
        total_curvature = 0
        for idx in range(len(self.connections)):
            curv = self.compute_connection_curvature(idx)
            total_curvature += curv.item()
            metrics[f'curvature/layer_{idx}'] = curv.item()
        
        metrics['curvature/total'] = total_curvature
        if self.connections:
            metrics['curvature/mean'] = total_curvature / len(self.connections)
        
        if self.holonomy_measurements:
            metrics['holonomy/latest'] = self.holonomy_measurements[-1]
            metrics['holonomy/mean'] = np.mean(self.holonomy_measurements)
        
        for layer_idx, analysis in self.neuron_class_associations.items():
            metrics[f'multiclass/layer_{layer_idx}/count'] = analysis['multi_class_count']
            metrics[f'multiclass/layer_{layer_idx}/mean_classes'] = analysis['mean_classes_per_neuron']
        
        total_params = 0
        active_params = 0
        for connection in self.connections:
            info = connection.get_sparsity_info()
            total_params += info['total']
            active_params += info['active']
        
        if total_params > 0:
            metrics['sparsity/global'] = 1 - (active_params / total_params)
        
        return metrics


class FiberBundleBuilder:
    """
    Builder class for creating fiber bundle networks
    """
    
    @staticmethod
    def create_mnist_bundle() -> FiberBundle:
        """Create fiber bundle network for MNIST"""
        config = FiberBundleConfig(
            base_dim=5,
            fiber_dim=256,
            initial_sparsity=0.98,
            growth_rate=0.05,
            max_classes_per_neuron=2,
            growth_strategy="holonomy_minimal"
        )
        return FiberBundle(config)
    
    @staticmethod
    def create_cifar10_bundle() -> FiberBundle:
        """Create fiber bundle network for CIFAR-10"""
        config = FiberBundleConfig(
            base_dim=8,
            fiber_dim=512,
            initial_sparsity=0.95,
            growth_rate=0.1,
            max_classes_per_neuron=5,
            growth_strategy="curvature_guided",
            gauge_regularization=0.1
        )
        return FiberBundle(config)
    
    @staticmethod
    def create_from_config(config_dict: Dict[str, Any]) -> FiberBundle:
        """Create from configuration dictionary"""
        config = FiberBundleConfig(**config_dict)
        return FiberBundle(config)