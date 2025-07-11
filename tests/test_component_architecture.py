#!/usr/bin/env python3
"""
Test suite for the new component architecture.

Verifies that the core interfaces, base components, and compatibility
system work correctly.
"""

import pytest
from typing import Dict, Any, List, Set
import torch
import torch.nn as nn

from src.structure_net.core import (
    # Interfaces
    Maturity, ComponentVersion, ResourceLevel, ResourceRequirements,
    ComponentContract, EvolutionContext, AnalysisReport, EvolutionPlan,
    IComponent, ILayer, IModel, IMetric,
    
    # Base implementations
    BaseComponent, BaseLayer, BaseModel, BaseMetric,
    
    # Compatibility system
    CompatibilityLevel, CompatibilityIssue,
    ComponentRegistry, CompatibilityManager
)


class TestComponentInterfaces:
    """Test the core interfaces and data structures."""
    
    def test_component_version(self):
        """Test version compatibility checking."""
        v1 = ComponentVersion(1, 0, 0)
        v2 = ComponentVersion(1, 5, 3)
        v3 = ComponentVersion(2, 0, 0)
        
        assert str(v1) == "1.0.0"
        assert str(v2) == "1.5.3"
        assert v1.is_compatible_with(v2)  # Same major version
        assert not v1.is_compatible_with(v3)  # Different major version
    
    def test_evolution_context(self):
        """Test evolution context functionality."""
        context = EvolutionContext()
        
        # Test metadata
        assert context.epoch == 0
        assert context.step == 0
        
        # Test setters
        context.epoch = 5
        context.step = 100
        assert context.epoch == 5
        assert context.step == 100
        
        # Test dict functionality
        context['loss'] = 0.5
        assert context['loss'] == 0.5
    
    def test_analysis_report(self):
        """Test analysis report functionality."""
        report = AnalysisReport()
        
        # Add metric data
        report.add_metric_data("sparsity", {"ratio": 0.98})
        assert "metrics.sparsity" in report
        assert report["metrics.sparsity"]["ratio"] == 0.98
        assert "sparsity" in report.sources
        
        # Add analyzer data
        report.add_analyzer_data("extrema", {"dead_neurons": 10})
        assert "analyzers.extrema" in report
        assert report["analyzers.extrema"]["dead_neurons"] == 10
        assert "extrema" in report.sources


class TestBaseComponents:
    """Test the base component implementations."""
    
    def test_base_component(self):
        """Test base component functionality."""
        
        class TestComponent(BaseComponent):
            @property
            def contract(self) -> ComponentContract:
                return ComponentContract(
                    component_name="TestComponent",
                    version=ComponentVersion(1, 0, 0),
                    maturity=Maturity.STABLE
                )
        
        comp = TestComponent("test")
        assert comp.name == "test"
        assert comp.maturity == Maturity.STABLE
        
        # Test performance metrics
        metrics = comp.get_performance_metrics()
        assert metrics['call_count'] == 0
        assert metrics['error_count'] == 0
    
    def test_base_layer(self):
        """Test base layer implementation."""
        
        class SimpleLayer(BaseLayer):
            def __init__(self, in_features: int, out_features: int):
                super().__init__(f"SimpleLayer_{in_features}x{out_features}")
                self.linear = nn.Linear(in_features, out_features)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)
            
            def get_analysis_properties(self) -> Dict[str, torch.Tensor]:
                return {"weight": self.linear.weight.data}
            
            def supports_modification(self) -> bool:
                return True
            
            def add_connections(self, num_connections: int, **kwargs) -> bool:
                return True  # Dummy implementation
        
        layer = SimpleLayer(10, 5)
        assert layer.name == "SimpleLayer_10x5"
        assert layer.supports_modification()
        
        # Test forward pass
        x = torch.randn(2, 10)
        y = layer(x)
        assert y.shape == (2, 5)
    
    def test_base_metric(self):
        """Test base metric implementation."""
        
        class SimpleMetric(BaseMetric):
            def __init__(self):
                super().__init__("SimpleMetric")
                self._measurement_schema = {"count": int, "ratio": float}
            
            def _compute_metric(self, target, context):
                # Dummy metric computation
                return {"count": 10, "ratio": 0.5}
        
        class DummyLayer(BaseLayer):
            def forward(self, x):
                return x
            
            def get_analysis_properties(self):
                return {}
            
            def supports_modification(self):
                return False
            
            def add_connections(self, num_connections, **kwargs):
                return False
        
        metric = SimpleMetric()
        context = EvolutionContext()
        
        # Create a dummy target
        target = DummyLayer("dummy")
        
        result = metric.analyze(target, context)
        assert result["count"] == 10
        assert result["ratio"] == 0.5


class TestCompatibilitySystem:
    """Test the compatibility management system."""
    
    def test_component_registry(self):
        """Test component registration and retrieval."""
        registry = ComponentRegistry()
        
        # Create a test component class
        class TestModel(BaseModel):
            def forward(self, x):
                return x
            
            def get_layers(self):
                return []
            
            def supports_dynamic_growth(self):
                return False
        
        # Register component
        registry.register(TestModel)
        
        # Check registration
        assert registry.get_component("TestModel") is not None
        contract = registry.get_contract("TestModel")
        assert contract is not None
        assert contract.component_name == "TestModel"
        
        # Test search functionality
        providers = registry.search_by_output("model.output")
        assert "TestModel" in providers
        
        stable_components = registry.search_by_maturity(Maturity.EXPERIMENTAL)
        assert "TestModel" in stable_components
    
    def test_compatibility_checking(self):
        """Test compatibility checking between components."""
        registry = ComponentRegistry()
        compatibility_manager = CompatibilityManager(registry)
        
        # Create test components with different compatibility
        class ComponentA(BaseComponent):
            @property
            def contract(self):
                return ComponentContract(
                    component_name="ComponentA",
                    version=ComponentVersion(1, 0, 0),
                    maturity=Maturity.STABLE,
                    provided_outputs={"data.processed"},
                    # Only accept stable components
                    compatible_maturity_levels={Maturity.STABLE}
                )
        
        class ComponentB(BaseComponent):
            @property
            def contract(self):
                return ComponentContract(
                    component_name="ComponentB",
                    version=ComponentVersion(1, 0, 0),
                    maturity=Maturity.EXPERIMENTAL,
                    required_inputs={"data.processed"}
                )
        
        comp_a = ComponentA()
        comp_b = ComponentB()
        
        # Check compatibility
        issues = compatibility_manager.check_compatibility(comp_a, comp_b)
        
        # Should have warning about maturity mismatch
        assert any(issue.level == CompatibilityLevel.WARNING for issue in issues)
        
        # Check data flow validation
        issues = compatibility_manager._check_data_flow([comp_a, comp_b])
        assert len(issues) == 0  # Should be complete
        
        # Check incomplete data flow
        issues = compatibility_manager._check_data_flow([comp_b])  # Missing provider
        assert len(issues) > 0
        assert any("data.processed" in issue.description for issue in issues)
    
    def test_compatibility_validation(self):
        """Test full composition validation."""
        registry = ComponentRegistry()
        compatibility_manager = CompatibilityManager(registry)
        
        class StableComponent(BaseComponent):
            @property
            def contract(self):
                return ComponentContract(
                    component_name=self.__class__.__name__,
                    version=ComponentVersion(1, 0, 0),
                    maturity=Maturity.STABLE
                )
        
        class ExperimentalComponent(BaseComponent):
            @property  
            def contract(self):
                return ComponentContract(
                    component_name=self.__class__.__name__,
                    version=ComponentVersion(1, 0, 0),
                    maturity=Maturity.EXPERIMENTAL
                )
        
        class DeprecatedComponent(BaseComponent):
            @property
            def contract(self):
                return ComponentContract(
                    component_name=self.__class__.__name__,
                    version=ComponentVersion(1, 0, 0),
                    maturity=Maturity.DEPRECATED
                )
        
        # Test composition with mixed maturity
        components = [
            StableComponent(),
            ExperimentalComponent(),
            DeprecatedComponent()
        ]
        
        issues = compatibility_manager.validate_composition(components)
        
        # Should have warnings about deprecated component
        deprecated_issues = [i for i in issues if "deprecated" in i.description.lower()]
        assert len(deprecated_issues) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])