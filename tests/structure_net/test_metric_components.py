"""
Tests for metric components.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

from src.structure_net.core import (
    ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentStatus
)
from src.structure_net.components.metrics import (
    SparsityMetric, DeadNeuronMetric, EntropyMetric,
    LayerMIMetric, GradientMetric, ActivationMetric
)


# Dummy implementations for testing
class DummyLayer(ILayer):
    """Dummy layer for testing."""
    
    def __init__(self, name: str, in_features: int, out_features: int):
        self._name = name
        self.linear = nn.Linear(in_features, out_features)
        self._contract = ComponentContract(
            component_name=name,
            version=(1, 0, 0),
            maturity="stable",
            required_inputs={"x"},
            provided_outputs={"y"}
        )
        self._status = ComponentStatus.ACTIVE
    
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
    
    def get_params(self) -> Dict[str, Any]:
        return {"in_features": self.linear.in_features, 
                "out_features": self.linear.out_features}
    
    def set_params(self, params: Dict[str, Any]) -> None:
        pass
    
    def named_parameters(self):
        return self.linear.named_parameters()


class DummyModel(nn.Module, IModel):
    """Dummy model for testing."""
    
    def __init__(self, name: str = "test_model"):
        super().__init__()
        self._name = name
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 10)
        self._contract = ComponentContract(
            component_name=name,
            version=(1, 0, 0),
            maturity="stable",
            required_inputs={"x"},
            provided_outputs={"y"}
        )
        self._status = ComponentStatus.ACTIVE
    
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
        x = torch.relu(self.layer1(x))
        return self.layer2(x)
    
    def get_params(self) -> Dict[str, Any]:
        return {"layers": 2}
    
    def set_params(self, params: Dict[str, Any]) -> None:
        pass


class TestSparsityMetric:
    """Test SparsityMetric component."""
    
    def test_layer_sparsity(self):
        """Test sparsity computation for a single layer."""
        layer = DummyLayer("test_layer", 10, 20)
        
        # Make weights sparse
        with torch.no_grad():
            layer.linear.weight.data *= 0.1
            layer.linear.weight.data[::2, ::2] = 0  # Set some weights to zero
        
        metric = SparsityMetric()
        context = EvolutionContext()
        
        result = metric.analyze(layer, context)
        
        assert "sparsity_ratio" in result
        assert "num_zeros" in result
        assert "num_total" in result
        assert 0 <= result["sparsity_ratio"] <= 1
        assert result["num_zeros"] > 0
    
    def test_model_sparsity(self):
        """Test sparsity computation for entire model."""
        model = DummyModel()
        
        # Make some weights sparse
        with torch.no_grad():
            model.layer1.weight.data *= 0.01
            model.layer2.weight.data[::2] = 0
        
        metric = SparsityMetric()
        context = EvolutionContext()
        
        result = metric.analyze(model, context)
        
        assert "sparsity_ratio" in result
        assert result["sparsity_ratio"] > 0


class TestDeadNeuronMetric:
    """Test DeadNeuronMetric component."""
    
    def test_dead_neuron_detection(self):
        """Test dead neuron detection."""
        layer = DummyLayer("test_layer", 10, 20)
        
        # Create activations with some dead neurons
        batch_size = 32
        activations = torch.randn(batch_size, 20)
        activations[:, :5] = 0  # First 5 neurons are dead
        
        metric = DeadNeuronMetric()
        context = EvolutionContext()
        context['activations'] = activations
        
        result = metric.analyze(layer, context)
        
        assert "dead_neuron_ratio" in result
        assert "num_dead" in result
        assert result["num_dead"] >= 5  # At least the 5 we set to zero
        assert result["dead_neuron_ratio"] >= 0.25  # At least 5/20


class TestEntropyMetric:
    """Test EntropyMetric component."""
    
    def test_entropy_computation(self):
        """Test entropy computation."""
        layer = DummyLayer("test_layer", 10, 20)
        
        # Create activations with varying entropy
        batch_size = 32
        # Low entropy (all similar values)
        low_entropy_acts = torch.ones(batch_size, 20) * 0.5
        # High entropy (uniform distribution)
        high_entropy_acts = torch.rand(batch_size, 20)
        
        metric = EntropyMetric()
        
        # Test low entropy
        context = EvolutionContext()
        context['activations'] = low_entropy_acts
        low_result = metric.analyze(layer, context)
        
        # Test high entropy
        context['activations'] = high_entropy_acts
        high_result = metric.analyze(layer, context)
        
        assert low_result["entropy"] < high_result["entropy"]
        assert "normalized_entropy" in low_result
        assert "effective_bits" in low_result


class TestLayerMIMetric:
    """Test LayerMIMetric component."""
    
    def test_mutual_information(self):
        """Test MI computation between layers."""
        model = DummyModel()
        
        # Create correlated activations
        batch_size = 32
        layer1_acts = torch.randn(batch_size, 20)
        layer2_acts = layer1_acts + torch.randn(batch_size, 20) * 0.1  # Highly correlated
        
        layer_activations = {
            "layer1": layer1_acts,
            "layer2": layer2_acts
        }
        
        metric = LayerMIMetric()
        context = EvolutionContext()
        context['layer_activations'] = layer_activations
        
        result = metric.analyze(model, context)
        
        assert "mutual_information" in result
        assert "normalized_mi" in result
        assert result["mutual_information"] > 0  # Should have positive MI
        assert "layer_pairs" in result


class TestGradientMetric:
    """Test GradientMetric component."""
    
    def test_gradient_statistics(self):
        """Test gradient statistics computation."""
        layer = DummyLayer("test_layer", 10, 20)
        
        # Create mock gradients
        gradients = torch.randn(20, 10) * 0.01  # Small gradients
        
        metric = GradientMetric()
        context = EvolutionContext()
        context['gradients'] = gradients
        
        result = metric.analyze(layer, context)
        
        assert "gradient_norm" in result
        assert "gradient_variance" in result
        assert "vanishing_ratio" in result
        assert "exploding_ratio" in result
        assert result["gradient_norm"] > 0


class TestActivationMetric:
    """Test ActivationMetric component."""
    
    def test_activation_statistics(self):
        """Test activation statistics computation."""
        layer = DummyLayer("test_layer", 10, 20)
        
        # Create activations with known properties
        batch_size = 32
        activations = torch.randn(batch_size, 20)
        
        metric = ActivationMetric()
        context = EvolutionContext()
        context['activations'] = activations
        
        result = metric.analyze(layer, context)
        
        assert "mean" in result
        assert "std" in result
        assert "sparsity" in result
        assert "percentiles" in result
        assert "activation_patterns" in result
        
        # Check percentiles
        assert "p50" in result["percentiles"]  # Median
    
    def test_pattern_detection(self):
        """Test activation pattern detection."""
        layer = DummyLayer("test_layer", 10, 20)
        
        # Create activations with patterns
        batch_size = 32
        activations = torch.zeros(batch_size, 20)
        activations[:, :5] = 0.95  # Saturated high
        activations[:, 5:10] = 0.05  # Saturated low
        activations[:, 10:15] = torch.randn(batch_size, 5)  # Normal
        
        metric = ActivationMetric(pattern_detection=True)
        context = EvolutionContext()
        context['activations'] = activations
        
        result = metric.analyze(layer, context)
        
        assert "activation_patterns" in result
        patterns = result["activation_patterns"]
        
        # Should detect saturation patterns
        assert any("saturation" in key for key in patterns.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])