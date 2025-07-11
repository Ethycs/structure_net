"""
Test fixtures for component testing.

Provides reusable test data, models, and utilities for testing
metrics and analyzers in the component architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import numpy as np

from src.structure_net.core import (
    ILayer, IModel, EvolutionContext, AnalysisReport,
    ComponentContract, ComponentVersion, Maturity,
    ComponentStatus, ResourceRequirements, ResourceLevel
)


class DummyLayer(ILayer):
    """Dummy layer for testing metrics."""
    
    def __init__(self, name: str, in_features: int, out_features: int,
                 add_bias: bool = True, sparsity: float = 0.0):
        self._name = name
        self.linear = nn.Linear(in_features, out_features, bias=add_bias)
        self._status = ComponentStatus.ACTIVE
        
        # Apply sparsity if requested
        if sparsity > 0:
            mask = torch.rand_like(self.linear.weight) > sparsity
            self.linear.weight.data *= mask.float()
            self.mask = mask
        else:
            self.mask = None
            
        self._contract = ComponentContract(
            component_name=name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"x"},
            provided_outputs={"y"},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def contract(self) -> ComponentContract:
        return self._contract
    
    @property
    def status(self) -> ComponentStatus:
        return self._status
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    def get_weight(self) -> torch.Tensor:
        """Get effective weight (considering mask if sparse)."""
        if self.mask is not None:
            return self.linear.weight * self.mask
        return self.linear.weight


class DummyModel(IModel):
    """Dummy model for testing analyzers."""
    
    def __init__(self, name: str, layers: Optional[List[int]] = None,
                 add_nonlinearity: bool = True):
        self._name = name
        self._status = ComponentStatus.ACTIVE
        
        # Default architecture
        if layers is None:
            layers = [10, 20, 10, 5]
        
        # Build layers
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(
                nn.Linear(layers[i], layers[i + 1])
            )
        
        self.add_nonlinearity = add_nonlinearity
        self.activation = nn.ReLU()
        
        self._contract = ComponentContract(
            component_name=name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"x"},
            provided_outputs={"y"},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def contract(self) -> ComponentContract:
        return self._contract
    
    @property
    def status(self) -> ComponentStatus:
        return self._status
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.add_nonlinearity and i < len(self.layers) - 1:
                x = self.activation(x)
        return x
    
    def get_layers(self) -> List[nn.Module]:
        """Get list of layers."""
        return list(self.layers)


def create_test_layer(in_features: int = 10, out_features: int = 5,
                     sparsity: float = 0.0, name: str = "test_layer") -> DummyLayer:
    """Create a test layer with specified properties."""
    return DummyLayer(name, in_features, out_features, sparsity=sparsity)


def create_test_model(layers: Optional[List[int]] = None,
                     name: str = "test_model") -> DummyModel:
    """Create a test model with specified architecture."""
    return DummyModel(name, layers)


def create_test_activations(batch_size: int = 32, features: int = 10,
                          sparsity: float = 0.0, seed: Optional[int] = 42) -> torch.Tensor:
    """Create test activation tensor."""
    if seed is not None:
        torch.manual_seed(seed)
    
    activations = torch.randn(batch_size, features)
    
    # Apply sparsity if requested
    if sparsity > 0:
        mask = torch.rand_like(activations) > sparsity
        activations *= mask.float()
    
    return activations


def create_test_gradients(shape: tuple, scale: float = 1.0,
                         seed: Optional[int] = 42) -> torch.Tensor:
    """Create test gradient tensor."""
    if seed is not None:
        torch.manual_seed(seed)
    
    gradients = torch.randn(shape) * scale
    return gradients


def create_test_context(data: Optional[Dict[str, Any]] = None) -> EvolutionContext:
    """Create test evolution context."""
    if data is None:
        data = {}
    return EvolutionContext(data)


def create_test_report() -> AnalysisReport:
    """Create empty analysis report."""
    return AnalysisReport()


def create_layer_activations_data(model: DummyModel, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Generate layer activations for a model."""
    activations = {}
    x = input_data
    
    for i, layer in enumerate(model.layers):
        x = layer(x)
        activations[f'layer_{i}'] = x.clone()
        
        if model.add_nonlinearity and i < len(model.layers) - 1:
            x = model.activation(x)
            activations[f'layer_{i}_activated'] = x.clone()
    
    return activations


def create_compact_data(original_size: int = 10000,
                       compression_ratio: float = 0.2,
                       num_patches: int = 10) -> Dict[str, Any]:
    """Create test compactification data."""
    compressed_size = int(original_size * compression_ratio)
    
    patches = []
    patch_size = compressed_size // num_patches
    
    for i in range(num_patches):
        patch_data = torch.randn(patch_size) * (1 - i * 0.05)  # Decreasing magnitude
        patches.append({
            'data': patch_data,
            'position': i * patch_size,
            'size': patch_size
        })
    
    skeleton = torch.randn(100) * 0.1  # Small skeleton
    
    return {
        'patches': patches,
        'skeleton': skeleton,
        'original_size': original_size,
        'metadata': {
            'compression_method': 'test',
            'timestamp': '2024-01-01'
        },
        'reconstruction_error': 0.05
    }


def create_trajectory_data(num_trajectories: int = 5,
                         trajectory_length: int = 10,
                         features: int = 20) -> List[List[torch.Tensor]]:
    """Create test trajectory data for dynamics analysis."""
    trajectories = []
    
    for _ in range(num_trajectories):
        trajectory = []
        # Start with random initial state
        state = torch.randn(features)
        
        for t in range(trajectory_length):
            # Evolve with some dynamics
            noise = torch.randn_like(state) * 0.1
            state = state * 0.95 + noise  # Decay with noise
            trajectory.append(state.clone())
        
        trajectories.append(trajectory)
    
    return trajectories


def create_graph_data(num_nodes: int = 50,
                     edge_probability: float = 0.1) -> Dict[str, Any]:
    """Create test graph data."""
    # Create random adjacency matrix
    adj_matrix = torch.rand(num_nodes, num_nodes) < edge_probability
    adj_matrix = adj_matrix.float()
    
    # Make symmetric
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    adj_matrix = (adj_matrix > 0.5).float()
    
    # Remove self-loops
    adj_matrix.fill_diagonal_(0)
    
    # Create node features
    node_features = torch.randn(num_nodes, 10)
    
    return {
        'adjacency': adj_matrix,
        'features': node_features,
        'num_nodes': num_nodes
    }


class TestMetricsMixin:
    """Mixin providing common test assertions for metrics."""
    
    def assert_metric_output_valid(self, output: Dict[str, Any],
                                  expected_keys: List[str]):
        """Assert metric output has expected structure."""
        assert isinstance(output, dict), "Metric output must be dictionary"
        
        for key in expected_keys:
            assert key in output, f"Missing expected key: {key}"
            
        # Check all values are numeric
        for key, value in output.items():
            assert isinstance(value, (int, float, np.integer, np.floating)), \
                f"Value for {key} must be numeric, got {type(value)}"
    
    def assert_metric_range(self, value: float, min_val: float = 0.0,
                          max_val: float = 1.0, name: str = "metric"):
        """Assert metric value is in expected range."""
        assert min_val <= value <= max_val, \
            f"{name} = {value} outside range [{min_val}, {max_val}]"
    
    def assert_probability(self, value: float, name: str = "probability"):
        """Assert value is valid probability."""
        self.assert_metric_range(value, 0.0, 1.0, name)
    
    def assert_non_negative(self, value: float, name: str = "value"):
        """Assert value is non-negative."""
        assert value >= 0, f"{name} = {value} is negative"


class TestAnalyzersMixin:
    """Mixin providing common test assertions for analyzers."""
    
    def assert_analysis_output_valid(self, output: Dict[str, Any],
                                   expected_sections: List[str]):
        """Assert analyzer output has expected structure."""
        assert isinstance(output, dict), "Analyzer output must be dictionary"
        
        for section in expected_sections:
            assert section in output, f"Missing expected section: {section}"
    
    def assert_recommendations_valid(self, recommendations: List[str]):
        """Assert recommendations are valid."""
        assert isinstance(recommendations, list), "Recommendations must be list"
        
        for rec in recommendations:
            assert isinstance(rec, str), "Each recommendation must be string"
            assert len(rec) > 0, "Recommendation cannot be empty"
    
    def assert_scores_valid(self, scores: Dict[str, float]):
        """Assert scores dictionary is valid."""
        assert isinstance(scores, dict), "Scores must be dictionary"
        
        for name, score in scores.items():
            assert isinstance(score, (int, float)), \
                f"Score {name} must be numeric"
            assert 0 <= score <= 1, \
                f"Score {name} = {score} outside [0, 1]"