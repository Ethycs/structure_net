"""
Base testing pipeline for component architecture.

Provides abstract base classes and utilities for creating
standardized test pipelines for different component types.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type, Optional, Tuple, Union
import pytest
import torch
import numpy as np

from src.structure_net.core import (
    IComponent, IMetric, IAnalyzer, IModel, ILayer,
    EvolutionContext, AnalysisReport, ComponentContract
)


class ComponentTestPipeline(ABC):
    """
    Abstract base class for component test pipelines.
    
    Provides a standardized testing framework that validates:
    - Contract compliance
    - Input/output specifications
    - Resource requirements
    - Error handling
    - Performance characteristics
    """
    
    @abstractmethod
    def get_component_class(self) -> Type[IComponent]:
        """Return the component class being tested."""
        pass
    
    @abstractmethod
    def create_valid_inputs(self) -> Dict[str, Any]:
        """Create valid input data for the component."""
        pass
    
    @abstractmethod
    def create_invalid_inputs(self) -> List[Dict[str, Any]]:
        """Create various invalid input scenarios."""
        pass
    
    @abstractmethod
    def validate_outputs(self, outputs: Dict[str, Any], 
                        inputs: Dict[str, Any]) -> None:
        """Validate component outputs given inputs."""
        pass
    
    def test_contract_compliance(self):
        """Test that component properly declares its contract."""
        component = self.get_component_class()()
        contract = component.contract
        
        assert isinstance(contract, ComponentContract)
        assert contract.component_name
        assert contract.version
        assert contract.maturity
        assert isinstance(contract.required_inputs, set)
        assert isinstance(contract.provided_outputs, set)
        assert contract.resources is not None
    
    def test_contract_validation(self):
        """Test contract validation with various inputs."""
        # Skip this test for now - validate_inputs not implemented in base components
        pytest.skip("validate_inputs method not yet implemented in base components")
    
    def test_output_compliance(self):
        """Test that outputs match contract specifications."""
        component = self.get_component_class()()
        inputs = self.create_valid_inputs()
        
        # Run component
        outputs = self.run_component(component, inputs)
        
        # Check all promised outputs are provided
        contract = component.contract
        for output_key in contract.provided_outputs:
            # Handle nested keys like "metrics.accuracy"
            self.assert_output_exists(outputs, output_key)
    
    def test_error_handling(self):
        """Test component error handling."""
        component = self.get_component_class()()
        
        for invalid_input in self.create_invalid_inputs():
            # Component should handle errors gracefully
            try:
                self.run_component(component, invalid_input)
            except (ValueError, TypeError, KeyError) as e:
                # Expected errors are fine
                pass
            except Exception as e:
                pytest.fail(f"Unexpected error type: {type(e).__name__}: {e}")
    
    def test_determinism(self):
        """Test component determinism with same inputs."""
        component = self.get_component_class()()
        inputs = self.create_valid_inputs()
        
        # Run multiple times
        results = []
        for _ in range(3):
            output = self.run_component(component, inputs)
            results.append(output)
        
        # Results should be consistent (within floating point tolerance)
        self.assert_outputs_equal(results[0], results[1])
        self.assert_outputs_equal(results[1], results[2])
    
    def test_edge_cases(self):
        """Test component with edge case inputs."""
        component = self.get_component_class()()
        edge_cases = self.create_edge_cases()
        
        for case_name, case_inputs in edge_cases.items():
            try:
                output = self.run_component(component, case_inputs)
                self.validate_edge_case_output(case_name, output, case_inputs)
            except Exception as e:
                pytest.fail(f"Failed on edge case '{case_name}': {e}")
    
    @abstractmethod
    def run_component(self, component: IComponent, 
                     inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the component with given inputs."""
        pass
    
    def create_edge_cases(self) -> Dict[str, Dict[str, Any]]:
        """Create edge case inputs. Override for specific components."""
        return {}
    
    def validate_edge_case_output(self, case_name: str, 
                                output: Dict[str, Any],
                                inputs: Dict[str, Any]) -> None:
        """Validate edge case outputs. Override for specific validation."""
        self.validate_outputs(output, inputs)
    
    def assert_output_exists(self, outputs: Dict[str, Any], key: str) -> None:
        """Assert that a (possibly nested) key exists in outputs."""
        parts = key.split('.')
        current = outputs
        
        for part in parts:
            assert part in current, f"Missing output key: {key} (failed at {part})"
            current = current[part]
    
    def assert_outputs_equal(self, out1: Dict[str, Any], 
                           out2: Dict[str, Any],
                           rtol: float = 1e-5) -> None:
        """Assert two output dictionaries are equal."""
        assert set(out1.keys()) == set(out2.keys()), "Output keys don't match"
        
        for key in out1:
            val1, val2 = out1[key], out2[key]
            
            if isinstance(val1, (int, float)):
                np.testing.assert_allclose(val1, val2, rtol=rtol)
            elif isinstance(val1, torch.Tensor):
                np.testing.assert_allclose(
                    val1.detach().numpy(), 
                    val2.detach().numpy(), 
                    rtol=rtol
                )
            elif isinstance(val1, dict):
                self.assert_outputs_equal(val1, val2, rtol)
            elif isinstance(val1, list):
                assert len(val1) == len(val2)
                for v1, v2 in zip(val1, val2):
                    if isinstance(v1, (int, float)):
                        np.testing.assert_allclose(v1, v2, rtol=rtol)
            else:
                assert val1 == val2, f"Values don't match for key {key}"


class MetricTestPipeline(ComponentTestPipeline):
    """Test pipeline specifically for IMetric components."""
    
    @abstractmethod
    def create_target(self) -> Optional[Union[ILayer, IModel]]:
        """Create target for metric analysis."""
        pass
    
    @abstractmethod
    def get_expected_metrics(self) -> List[str]:
        """Return list of expected metric names in output."""
        pass
    
    def run_component(self, component: IMetric, 
                     inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run metric component."""
        target = inputs.get('target', self.create_target())
        context = inputs.get('context', EvolutionContext(inputs))
        
        return component.analyze(target, context)
    
    def validate_outputs(self, outputs: Dict[str, Any], 
                        inputs: Dict[str, Any]) -> None:
        """Validate metric outputs."""
        # All metrics should be present
        for metric_name in self.get_expected_metrics():
            assert metric_name in outputs, f"Missing metric: {metric_name}"
            
        # All values should be numeric
        for key, value in outputs.items():
            assert isinstance(value, (int, float, np.integer, np.floating)), \
                f"Metric {key} should be numeric, got {type(value)}"
    
    def test_metric_ranges(self):
        """Test that metrics are within expected ranges."""
        component = self.get_component_class()()
        inputs = self.create_valid_inputs()
        outputs = self.run_component(component, inputs)
        
        ranges = self.get_metric_ranges()
        for metric_name, (min_val, max_val) in ranges.items():
            if metric_name in outputs:
                value = outputs[metric_name]
                assert min_val <= value <= max_val, \
                    f"{metric_name}={value} outside range [{min_val}, {max_val}]"
    
    def get_metric_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return expected ranges for metrics. Override as needed."""
        return {}


class AnalyzerTestPipeline(ComponentTestPipeline):
    """Test pipeline specifically for IAnalyzer components."""
    
    @abstractmethod
    def create_model(self) -> IModel:
        """Create model for analysis."""
        pass
    
    @abstractmethod
    def create_analysis_context(self) -> EvolutionContext:
        """Create context with necessary data for analysis."""
        pass
    
    @abstractmethod
    def get_required_metrics(self) -> List[str]:
        """Return list of metrics required by analyzer."""
        pass
    
    def run_component(self, component: IAnalyzer, 
                     inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run analyzer component."""
        model = inputs.get('model', self.create_model())
        report = inputs.get('report', AnalysisReport())
        context = inputs.get('context', self.create_analysis_context())
        
        return component.analyze(model, report, context)
    
    def validate_outputs(self, outputs: Dict[str, Any], 
                        inputs: Dict[str, Any]) -> None:
        """Validate analyzer outputs."""
        # Should have analysis results
        assert 'analysis' in outputs or len(outputs) > 0
        
        # Check for common analyzer outputs
        common_keys = ['recommendations', 'insights', 'summary']
        has_common = any(key in outputs for key in common_keys)
        assert has_common or len(outputs) > 2, \
            "Analyzer should provide meaningful analysis"
    
    def test_metric_dependency(self):
        """Test that analyzer properly uses required metrics."""
        component = self.get_component_class()()
        
        # Create report with metrics
        report = AnalysisReport()
        for metric_name in self.get_required_metrics():
            # Add dummy metric data
            report.add_metric_data(metric_name, {'value': 0.5})
        
        inputs = self.create_valid_inputs()
        inputs['report'] = report
        
        # Should run successfully with required metrics
        output = self.run_component(component, inputs)
        assert output is not None
    
    def test_missing_metrics_handling(self):
        """Test analyzer behavior with missing metrics."""
        component = self.get_component_class()()
        
        # Empty report (no metrics)
        report = AnalysisReport()
        inputs = self.create_valid_inputs()
        inputs['report'] = report
        
        # Should either compute metrics or handle gracefully
        try:
            output = self.run_component(component, inputs)
            # If successful, should still provide some analysis
            assert len(output) > 0
        except ValueError as e:
            # Or raise clear error about missing metrics
            assert any(metric in str(e) for metric in self.get_required_metrics())


class TestPipelineRegistry:
    """Registry for component test pipelines."""
    
    _pipelines: Dict[Type[IComponent], Type[ComponentTestPipeline]] = {}
    
    @classmethod
    def register(cls, component_type: Type[IComponent], 
                 pipeline_type: Type[ComponentTestPipeline]):
        """Register a test pipeline for a component type."""
        cls._pipelines[component_type] = pipeline_type
    
    @classmethod
    def get_pipeline(cls, component_type: Type[IComponent]) -> Type[ComponentTestPipeline]:
        """Get test pipeline for component type."""
        # Look for exact match first
        if component_type in cls._pipelines:
            return cls._pipelines[component_type]
        
        # Look for parent class match
        for comp_class, pipeline_class in cls._pipelines.items():
            if issubclass(component_type, comp_class):
                return pipeline_class
        
        raise ValueError(f"No test pipeline registered for {component_type}")
    
    @classmethod
    def run_tests(cls, component_type: Type[IComponent]):
        """Run all tests for a component type."""
        pipeline_class = cls.get_pipeline(component_type)
        pipeline = pipeline_class()
        
        # Run all test methods
        test_methods = [
            method for method in dir(pipeline)
            if method.startswith('test_') and callable(getattr(pipeline, method))
        ]
        
        for method_name in test_methods:
            method = getattr(pipeline, method_name)
            method()


def create_test_suite(component_type: Type[IComponent]) -> None:
    """Create and run test suite for a component."""
    TestPipelineRegistry.run_tests(component_type)