"""
Tests for model components.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

from src.structure_net.core import EvolutionContext
from src.structure_net.components.models import (
    MinimalModel, FiberBundleModel, FiberBundleConfig, MultiScaleModel
)
from src.structure_net.components.layers import SparseLayer


class TestMinimalModel:
    """Test MinimalModel component."""
    
    @pytest.fixture
    def model(self):
        return MinimalModel(
            layer_sizes=[784, 256, 128, 10],
            sparsity=0.9,
            activation='relu'
        )
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.layer_sizes == [784, 256, 128, 10]
        assert model.sparsity == 0.9
        assert model.activation_fn == 'relu'
        assert len(model.get_layers()) == 3  # 3 sparse layers
    
    def test_forward_pass(self, model):
        """Test forward pass."""
        x = torch.randn(32, 784)
        output = model(x)
        
        assert output.shape == (32, 10)
        assert hasattr(model, 'layer_activations')
        assert len(model.layer_activations) == 3
    
    def test_architecture_summary(self, model):
        """Test architecture summary."""
        summary = model.get_architecture_summary()
        
        assert 'architecture' in summary
        assert summary['architecture'] == [784, 256, 128, 10]
        assert 'total_parameters' in summary
        assert 'active_connections' in summary
        assert 'actual_sparsity' in summary
        assert summary['num_layers'] == 3
    
    def test_connectivity_stats(self, model):
        """Test connectivity statistics."""
        stats = model.get_connectivity_stats()
        
        assert 'layers' in stats
        assert len(stats['layers']) == 3
        assert 'total_connections' in stats
        assert 'active_connections' in stats
        
        # Check layer stats
        for layer_stat in stats['layers']:
            assert 'sparsity' in layer_stat
            assert layer_stat['sparsity'] >= 0.85  # Should be close to target
    
    def test_add_connections(self, model):
        """Test adding connections."""
        # Get initial stats
        initial_stats = model.get_connectivity_stats()
        initial_active = initial_stats['active_connections']
        
        # Add connections to first layer
        model.add_connections(0, 100)
        
        # Check increased connections
        new_stats = model.get_connectivity_stats()
        assert new_stats['active_connections'] > initial_active
    
    def test_gradient_stats(self, model):
        """Test gradient statistics."""
        # Create dummy loss and backward
        x = torch.randn(32, 784)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        stats = model.get_gradient_stats()
        assert 'gradient_norm' in stats
        assert stats['gradient_norm'] > 0
        assert 'max_gradient' in stats
        assert 'num_zero_gradients' in stats
    
    def test_contract(self, model):
        """Test component contract."""
        contract = model.contract
        
        assert contract.component_name == "MinimalModel"
        assert contract.maturity.value == "stable"
        assert 'input' in contract.required_inputs
        assert 'output' in contract.provided_outputs


class TestFiberBundleModel:
    """Test FiberBundleModel component."""
    
    @pytest.fixture
    def config(self):
        return FiberBundleConfig(
            base_dim=4,  # 4 layers
            fiber_dim=128,  # 128 neurons per layer
            initial_sparsity=0.95,
            max_curvature=1.0,
            gauge_regularization=0.01
        )
    
    @pytest.fixture
    def model(self, config):
        return FiberBundleModel(config)
    
    def test_initialization(self, model, config):
        """Test model initialization."""
        assert model.config.base_dim == 4
        assert model.config.fiber_dim == 128
        assert len(model.get_layers()) == 3  # base_dim - 1
    
    def test_forward_pass(self, model):
        """Test forward pass with geometric tracking."""
        x = torch.randn(32, 128)  # Batch x fiber_dim
        output = model(x)
        
        assert output.shape == (32, 128)
        assert hasattr(model, 'fiber_activations')
        assert hasattr(model, 'current_curvature')
        assert hasattr(model, 'current_holonomy')
    
    def test_geometric_properties(self, model):
        """Test geometric property computation."""
        # Run forward pass to initialize
        x = torch.randn(32, 128)
        model(x)
        
        props = model.get_geometric_properties()
        
        assert 'curvatures' in props
        assert 'holonomies' in props
        assert 'gauge_fields' in props
        assert 'total_curvature' in props
        assert 'curvature_violation' in props
        
        # Check dimensions
        assert len(props['curvatures']) == 3
        assert len(props['gauge_fields']) == 3
    
    def test_gauge_constraint(self, model):
        """Test gauge constraint enforcement."""
        # Get initial curvature
        initial_props = model.get_geometric_properties()
        initial_curvature = initial_props['total_curvature']
        
        # Enforce constraints
        model.enforce_gauge_constraint(max_change=0.05)
        
        # Check if curvature changed
        new_props = model.get_geometric_properties()
        # Curvature should be smoothed (usually reduced)
        assert new_props['total_curvature'] != initial_curvature
    
    def test_class_distribution(self, model):
        """Test multi-class neuron tracking."""
        dist = model.get_class_distribution()
        
        assert 'neurons_per_class' in dist
        assert 'class_specialization' in dist
        
        # Initially all neurons should be class 0
        assert 0 in dist['neurons_per_class']
    
    def test_fiber_connection_growth(self, model):
        """Test adding connections to specific fiber."""
        # Get initial connections
        layer = model.get_layers()[0]
        initial_active = layer.mask.sum().item()
        
        # Add connections
        model.add_fiber_connection(0, 50)
        
        # Check increased connections
        new_active = layer.mask.sum().item()
        assert new_active > initial_active
    
    def test_structured_layer_properties(self, model):
        """Test structured layer functionality."""
        layer = model.get_layers()[0]
        
        # Check ILayer interface
        assert layer.supports_modification()
        
        props = layer.get_analysis_properties()
        assert 'weight' in props
        assert 'mask' in props
        assert 'gauge_field' in props
        assert 'neuron_classes' in props


class TestMultiScaleModel:
    """Test MultiScaleModel component."""
    
    @pytest.fixture
    def model(self):
        return MultiScaleModel(
            initial_architecture=[784, 256, 128, 10],
            scales=[1, 2, 4],
            initial_sparsity=0.9,
            use_multi_scale_blocks=True
        )
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.architecture == [784, 256, 128, 10]
        assert model.scales == [1, 2, 4]
        assert model.use_multi_scale_blocks
    
    def test_forward_pass(self, model):
        """Test forward pass with multi-scale processing."""
        x = torch.randn(32, 784)
        output = model(x)
        
        assert output.shape == (32, 10)
        assert hasattr(model, 'scale_features')
        
        # Check scale features were captured
        for key in model.scale_features:
            assert 'scale_1' in model.scale_features[key]
            assert 'scale_2' in model.scale_features[key]
            assert 'scale_4' in model.scale_features[key]
    
    def test_architecture_summary(self, model):
        """Test architecture summary with scale information."""
        summary = model.get_architecture_summary()
        
        assert 'scales' in summary
        assert summary['scales'] == [1, 2, 4]
        assert 'num_multi_scale_blocks' in summary
        assert 'scale_statistics' in summary
    
    def test_growth_potential(self, model):
        """Test growth potential analysis."""
        potential = model.get_growth_potential()
        
        assert 'layer_growth' in potential
        assert 'scale_growth' in potential
        assert 'recommended_growth' in potential
        
        # Check scale growth recommendations
        for scale in model.scales:
            assert f'scale_{scale}' in potential['scale_growth']
    
    def test_add_layer(self, model):
        """Test dynamic layer addition."""
        initial_arch = model.architecture.copy()
        
        # Add layer at position 2
        model.add_layer(2, 192)
        
        # Check architecture updated
        assert len(model.architecture) == len(initial_arch) + 1
        assert model.architecture[2] == 192
        
        # Check growth history
        assert len(model.growth_history) == 1
        assert model.growth_history[0]['type'] == 'add_layer'
    
    def test_scale_growth(self, model):
        """Test growing specific scale."""
        # Assuming first block is multi-scale
        model.grow_scale(0, scale=2, growth_factor=1.5)
        
        # This should not error
        # In practice, we'd check the actual connection count increased
    
    def test_scale_importance_update(self, model):
        """Test updating scale importance."""
        initial_importance = model.scale_importance.copy()
        
        # Update with fake gradients
        scale_gradients = {1: 0.5, 2: 0.3, 4: 0.2}
        model.update_scale_importance(scale_gradients)
        
        # Check importance updated
        for scale in model.scales:
            assert model.scale_importance[scale] != initial_importance[scale]
    
    def test_scale_pruning(self, model):
        """Test pruning specific scale."""
        pruned = model.prune_scale(scale=4, prune_ratio=0.1)
        
        # Should return number of pruned connections
        assert isinstance(pruned, int)
    
    def test_multi_scale_block(self):
        """Test MultiScaleBlock functionality."""
        from src.structure_net.components.models.multi_scale_model import MultiScaleBlock
        
        block = MultiScaleBlock(
            in_features=256,
            out_features=128,
            scales=[1, 2, 4],
            sparsity=0.9
        )
        
        x = torch.randn(32, 256)
        output, scale_outputs = block(x)
        
        assert output.shape == (32, 128)
        assert 'scale_1' in scale_outputs
        assert 'scale_2' in scale_outputs
        assert 'scale_4' in scale_outputs


class TestSparseLayer:
    """Test SparseLayer component."""
    
    @pytest.fixture
    def layer(self):
        return SparseLayer(
            in_features=256,
            out_features=128,
            sparsity=0.9,
            bias=True
        )
    
    def test_initialization(self, layer):
        """Test layer initialization."""
        assert layer.in_features == 256
        assert layer.out_features == 128
        assert layer.target_sparsity == 0.9
        assert layer.bias is not None
    
    def test_sparsity_mask(self, layer):
        """Test sparsity mask initialization."""
        active = layer.mask.sum().item()
        total = layer.mask.numel()
        actual_sparsity = 1.0 - (active / total)
        
        # Should be close to target
        assert abs(actual_sparsity - 0.9) < 0.05
    
    def test_forward_pass(self, layer):
        """Test forward pass."""
        x = torch.randn(32, 256)
        output = layer(x)
        
        assert output.shape == (32, 128)
    
    def test_add_remove_connections(self, layer):
        """Test connection management."""
        # Get initial count
        initial_active = layer.get_connection_count()
        
        # Add connections
        layer.add_connections(50)
        after_add = layer.get_connection_count()
        assert after_add > initial_active
        
        # Remove connections
        layer.prune_connections(25)
        after_prune = layer.get_connection_count()
        assert after_prune < after_add
        assert after_prune > initial_active  # Net increase
    
    def test_importance_tracking(self, layer):
        """Test importance score tracking."""
        # Run forward and backward
        x = torch.randn(32, 256)
        layer.train()
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        # Importance scores should update
        assert layer.importance_scores.sum() > 0
    
    def test_sparsity_stats(self, layer):
        """Test sparsity statistics."""
        stats = layer.get_sparsity_stats()
        
        assert 'total_possible' in stats
        assert 'active' in stats
        assert 'sparsity' in stats
        assert 'output_stats' in stats
        assert 'input_stats' in stats
        
        # Check detailed stats
        assert 'dead_outputs' in stats['output_stats']
        assert 'unused_inputs' in stats['input_stats']