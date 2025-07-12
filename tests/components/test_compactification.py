"""
Tests for compactification components.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

from src.structure_net.core import (
    EvolutionContext, AnalysisReport, EvolutionPlan
)
from src.structure_net.components.evolvers import (
    CompactificationEvolver, InputHighwayEvolver
)
from src.structure_net.components.strategies import CompactificationStrategy
from src.structure_net.components.metrics import (
    CompressionRatioMetric, PatchEffectivenessMetric
)
from src.structure_net.components.analyzers import CompactificationAnalyzer


class SimpleModel(nn.Module):
    """Simple test model."""
    def __init__(self, input_size: int = 784, hidden_size: int = 256, 
                 output_size: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class TestCompactificationEvolver:
    """Test CompactificationEvolver component."""
    
    @pytest.fixture
    def evolver(self):
        return CompactificationEvolver(
            target_sparsity=0.05,
            patch_density=0.2,
            patch_size=8
        )
    
    @pytest.fixture
    def model(self):
        return SimpleModel()
    
    @pytest.fixture
    def plan(self):
        return EvolutionPlan({
            'type': 'compactification',
            'target_sparsity': 0.05,
            'patch_density': 0.2,
            'patch_size': 8
        })
    
    def test_initialization(self, evolver):
        """Test evolver initialization."""
        assert evolver.target_sparsity == 0.05
        assert evolver.patch_density == 0.2
        assert evolver.patch_size == 8
        assert 'compactification' in evolver._supported_plan_types
    
    def test_can_execute_plan(self, evolver, plan):
        """Test plan execution check."""
        assert evolver.can_execute_plan(plan)
        
        invalid_plan = EvolutionPlan({'type': 'invalid'})
        assert not evolver.can_execute_plan(invalid_plan)
    
    def test_compactify_layer(self, evolver, model):
        """Test layer compactification."""
        # Get a layer to compactify
        layer = model.fc1
        original_weight = layer.weight.data.clone()
        
        # Compactify the layer
        compact_data = evolver._compactify_layer(
            layer, 'fc1', 0.05, 0.2, 8, None
        )
        
        # Check results
        assert compact_data.layer_name == 'fc1'
        assert torch.allclose(compact_data.original_weight, original_weight)
        assert compact_data.compression_ratio < 1.0
        assert compact_data.sparsity_level > 0.9  # ~95% sparse
        assert len(compact_data.patches) > 0
    
    def test_apply_compactification(self, evolver, model, plan):
        """Test full compactification application."""
        # Store original parameter count
        original_params = sum(p.numel() for p in model.parameters())
        
        # Apply compactification
        results = evolver.apply_plan(plan, model, None, None)
        
        # Check results
        assert 'compactified_layers' in results
        assert len(results['compactified_layers']) > 0
        assert results['total_compression_ratio'] < 1.0
        assert results['patch_count'] > 0
        
        # Check that weights are actually sparse
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and name in results['compactified_layers']:
                sparsity = (module.weight.abs() < 1e-6).float().mean()
                assert sparsity > 0.9  # Should be ~95% sparse
    
    def test_patch_optimization(self, evolver, model):
        """Test patch optimization functionality."""
        # First compactify
        plan = EvolutionPlan({
            'type': 'compactification',
            'target_sparsity': 0.05
        })
        evolver.apply_plan(plan, model, None, None)
        
        # Then optimize patches
        opt_plan = EvolutionPlan({
            'type': 'patch_optimization',
            'importance_threshold': 0.5
        })
        
        results = evolver.apply_plan(opt_plan, model, None, None)
        assert 'optimized_layers' in results
        assert 'patches_removed' in results
    
    def test_extrema_detection(self, evolver):
        """Test extrema location detection."""
        # Create test weight matrix
        weight = torch.randn(64, 64)
        
        # Add some clear extrema
        weight[10, 10] = 5.0
        weight[50, 50] = -5.0
        weight[30, 30] = 3.0
        
        locations = evolver._find_extrema_locations(weight, 0.05, 8)
        
        assert len(locations) > 0
        assert all(isinstance(loc, tuple) for loc in locations)
        assert all(len(loc) == 2 for loc in locations)


class TestInputHighwayEvolver:
    """Test InputHighwayEvolver component."""
    
    @pytest.fixture
    def evolver(self):
        return InputHighwayEvolver(preserve_topology=True)
    
    @pytest.fixture
    def model(self):
        return SimpleModel(input_size=784)  # MNIST-like
    
    def test_initialization(self, evolver):
        """Test evolver initialization."""
        assert evolver.preserve_topology
        assert evolver.merge_strategy == 'adaptive'
        assert 'add_input_highway' in evolver._supported_plan_types
    
    def test_highway_module(self):
        """Test InputHighwayModule functionality."""
        from src.structure_net.components.evolvers.input_highway_evolver import InputHighwayModule
        
        highway = InputHighwayModule(784, preserve_topology=True)
        
        # Test forward pass
        x = torch.randn(32, 784)
        output = highway(x)
        
        assert 'highway' in output
        assert output['highway'].shape == (32, 784)
        assert 'groups' in output
        
        # Check topological groups for MNIST
        assert 'corners' in highway.input_groups
        assert 'edges' in highway.input_groups
        assert 'center' in highway.input_groups
    
    def test_add_highway(self, evolver, model):
        """Test adding highway to model."""
        plan = EvolutionPlan({
            'type': 'add_input_highway',
            'preserve_topology': True
        })
        
        results = evolver.apply_plan(plan, model, None, None)
        
        assert results['highway_added']
        assert results['input_dim'] == 784
        assert results['preservation_score'] == 1.0
    
    def test_highway_optimization(self, evolver, model):
        """Test highway optimization."""
        # First add highway
        add_plan = EvolutionPlan({'type': 'add_input_highway'})
        evolver.apply_plan(add_plan, model, None, None)
        
        # Then optimize
        opt_plan = EvolutionPlan({
            'type': 'optimize_highway',
            'gradient_data': {'available': True}
        })
        
        results = evolver.apply_plan(opt_plan, model, None, None)
        assert results['optimized']


class TestCompactificationStrategy:
    """Test CompactificationStrategy component."""
    
    @pytest.fixture
    def strategy(self):
        return CompactificationStrategy(
            size_threshold=100_000,
            performance_threshold=0.8
        )
    
    @pytest.fixture
    def report(self):
        report = AnalysisReport()
        report['model_stats'] = {
            'total_parameters': 500_000,
            'layer_count': 10
        }
        report['performance_metrics'] = {
            'accuracy': 0.85,
            'loss': 0.3
        }
        report['memory_usage'] = {
            'total_mb': 500
        }
        return report
    
    @pytest.fixture
    def context(self):
        return EvolutionContext({
            'epoch': 10,
            'step': 1000
        })
    
    def test_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.size_threshold == 100_000
        assert strategy.performance_threshold == 0.8
        assert strategy.get_strategy_type() == 'compactification'
    
    def test_should_compactify(self, strategy, report, context):
        """Test compactification decision logic."""
        # Should compactify - large model
        assert strategy._should_compactify(report, context)
        
        # Small model - should not compactify
        report['model_stats']['total_parameters'] = 50_000
        assert not strategy._should_compactify(report, context)
        
        # Force compactification
        context['force_compactification'] = True
        assert strategy._should_compactify(report, context)
    
    def test_propose_plan(self, strategy, report, context):
        """Test plan proposal."""
        plan = strategy.propose_plan(report, context)
        
        assert plan['type'] == 'compactification'
        assert 'target_sparsity' in plan
        assert 'patch_density' in plan
        assert 'patch_size' in plan
        assert plan.priority > 0
        assert plan.estimated_impact > 0
    
    def test_parameter_determination(self, strategy, report, context):
        """Test parameter determination logic."""
        params = strategy._determine_parameters(report, context)
        
        assert 'sparsity' in params
        assert 'patch_density' in params
        assert 'patch_size' in params
        assert params['sparsity'] == 0.05  # For 500k params
        assert params['patch_density'] == 0.2
        assert params['patch_size'] == 8
    
    def test_layer_selection(self, strategy, report, context):
        """Test layer selection logic."""
        # Add layer statistics
        report['layer_statistics'] = {
            'fc1': {'parameter_count': 200_000, 'redundancy': 0.3},
            'fc2': {'parameter_count': 150_000, 'redundancy': 0.5},
            'fc3': {'parameter_count': 50_000, 'is_critical': True}
        }
        
        selection = strategy._select_layers(report, context)
        
        assert 'layers' in selection
        assert len(selection['layers']) > 0
        # Critical layer should be ranked lower
        assert 'fc3' not in selection['layers'][:2]


class TestCompactificationMetrics:
    """Test compactification metrics."""
    
    @pytest.fixture
    def compression_metric(self):
        return CompressionRatioMetric()
    
    @pytest.fixture
    def patch_metric(self):
        return PatchEffectivenessMetric()
    
    @pytest.fixture
    def context(self):
        return EvolutionContext({
            'original_size': 1000,
            'compressed_size': 100,
            'patch_info': {
                'metadata_size': 10
            }
        })
    
    def test_compression_ratio_metric(self, compression_metric, context):
        """Test compression ratio calculation."""
        results = compression_metric.analyze(None, context)
        
        assert results['compression_ratio'] == 0.1
        assert results['space_saved'] == 900
        assert results['efficiency_score'] > 0.8
        assert 'patch_overhead' in results
    
    def test_patch_effectiveness_metric(self, patch_metric):
        """Test patch effectiveness analysis."""
        context = EvolutionContext({
            'compact_data': {
                'patches': [
                    {'density': 0.8, 'importance_score': 0.9},
                    {'density': 0.7, 'importance_score': 0.8}
                ]
            }
        })
        
        results = patch_metric.analyze(None, context)
        
        assert 'patch_count' in results
        assert 'avg_patch_density' in results
        assert results['patch_count'] == 2


class TestCompactificationAnalyzer:
    """Test CompactificationAnalyzer component."""
    
    @pytest.fixture
    def analyzer(self):
        return CompactificationAnalyzer()
    
    @pytest.fixture
    def report(self):
        return AnalysisReport()
    
    @pytest.fixture
    def context(self):
        return EvolutionContext({
            'compact_data': {
                'compression_ratio': 0.1,
                'patches': [
                    {'data': torch.randn(8, 8), 'density': 0.2}
                ],
                'sparse_skeleton': torch.randn(64, 64).to_sparse()
            }
        })
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.quality_thresholds['excellent'] == 0.9
        assert analyzer.quality_thresholds['good'] == 0.8
        assert len(analyzer._required_metrics) == 4
    
    def test_analysis(self, analyzer, report, context):
        """Test comprehensive analysis."""
        # This would require mocking the metrics
        # For now, just check the structure
        assert analyzer.contract.component_name == "CompactificationAnalyzer"
        assert analyzer.contract.maturity.value == "experimental"