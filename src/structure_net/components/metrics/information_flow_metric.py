"""
Information flow metric component.

This component measures information flow characteristics between layers,
including efficiency, capacity utilization, and information gaps.
"""

from typing import Dict, Any, Union, Optional
import torch
import logging

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)
from .layer_mi_metric import LayerMIMetric
from .entropy_metric import EntropyMetric


class InformationFlowMetric(BaseMetric):
    """
    Measures information flow characteristics in neural networks.
    
    This metric analyzes how information propagates through layers,
    identifying bottlenecks, gaps, and efficiency of information transfer.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize information flow metric.
        
        Args:
            name: Optional custom name
        """
        super().__init__(name or "InformationFlowMetric")
        self._mi_metric = LayerMIMetric()
        self._entropy_metric = EntropyMetric()
        self._measurement_schema = {
            "mi_efficiency": float,
            "information_gap": float,
            "capacity_utilization": float,
            "max_possible_mi": float,
            "layer_flow": dict  # Per-layer-pair flow metrics
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"target", "layer_activations"},
            provided_outputs={
                "metrics.mi_efficiency",
                "metrics.information_gap",
                "metrics.capacity_utilization",
                "metrics.max_possible_mi",
                "metrics.layer_flow"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def _compute_metric(self, target: Union[ILayer, IModel], 
                       context: EvolutionContext) -> Dict[str, Any]:
        """
        Compute information flow metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'layer_activations' data
            
        Returns:
            Dictionary containing information flow measurements
        """
        # Get activations from context
        layer_activations = context.get('layer_activations')
        if layer_activations is None:
            raise ValueError("InformationFlowMetric requires 'layer_activations' in context")
        
        if isinstance(target, IModel):
            return self._compute_model_flow(target, layer_activations)
        elif isinstance(target, ILayer):
            return self._compute_layer_flow(target, layer_activations)
        else:
            raise ValueError(f"Target must be ILayer or IModel, got {type(target)}")
    
    def _compute_layer_flow(self, layer: ILayer, 
                           layer_activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute information flow for a single layer."""
        # Find this layer and its input
        layer_acts = layer_activations.get(layer.name)
        if layer_acts is None:
            raise ValueError(f"No activations found for layer {layer.name}")
        
        # Find input layer
        layer_names = list(layer_activations.keys())
        input_acts = None
        if layer.name in layer_names:
            idx = layer_names.index(layer.name)
            if idx > 0:
                input_acts = layer_activations[layer_names[idx - 1]]
        
        if input_acts is None:
            # No input layer found
            return {
                "mi_efficiency": 0.0,
                "information_gap": 0.0,
                "capacity_utilization": 0.0,
                "max_possible_mi": 0.0,
                "layer_flow": {}
            }
        
        # Compute MI and entropies
        mi_result = self._mi_metric._compute_mi(input_acts, layer_acts)
        h_input = self._entropy_metric._estimate_entropy(input_acts)
        h_output = self._entropy_metric._estimate_entropy(layer_acts)
        
        # Information flow metrics
        mi_efficiency = mi_result / h_input if h_input > 0 else 0.0
        max_possible_mi = min(h_input, h_output)
        information_gap = max_possible_mi - mi_result
        capacity_utilization = mi_result / max_possible_mi if max_possible_mi > 0 else 0.0
        
        return {
            "mi_efficiency": mi_efficiency,
            "information_gap": information_gap,
            "capacity_utilization": capacity_utilization,
            "max_possible_mi": max_possible_mi,
            "layer_flow": {
                f"input->{layer.name}": {
                    "efficiency": mi_efficiency,
                    "gap": information_gap,
                    "utilization": capacity_utilization
                }
            }
        }
    
    def _compute_model_flow(self, model: IModel, 
                           layer_activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute information flow for entire model."""
        layer_names = list(layer_activations.keys())
        total_efficiency = 0.0
        total_gap = 0.0
        total_utilization = 0.0
        layer_flow = {}
        num_pairs = 0
        
        # Analyze flow between consecutive layers
        for i in range(len(layer_names) - 1):
            layer1_name = layer_names[i]
            layer2_name = layer_names[i + 1]
            
            acts1 = layer_activations[layer1_name]
            acts2 = layer_activations[layer2_name]
            
            if acts1 is None or acts2 is None:
                continue
            
            # Compute MI and entropies
            mi_value = self._mi_metric._compute_mi(acts1, acts2)
            h1 = self._entropy_metric._estimate_entropy(acts1)
            h2 = self._entropy_metric._estimate_entropy(acts2)
            
            # Flow metrics
            efficiency = mi_value / h1 if h1 > 0 else 0.0
            max_mi = min(h1, h2)
            gap = max_mi - mi_value
            utilization = mi_value / max_mi if max_mi > 0 else 0.0
            
            pair_key = f"{layer1_name}->{layer2_name}"
            layer_flow[pair_key] = {
                "efficiency": efficiency,
                "gap": gap,
                "utilization": utilization,
                "mi": mi_value,
                "max_mi": max_mi
            }
            
            total_efficiency += efficiency
            total_gap += gap
            total_utilization += utilization
            num_pairs += 1
        
        # Average metrics
        avg_efficiency = total_efficiency / num_pairs if num_pairs > 0 else 0.0
        avg_gap = total_gap / num_pairs if num_pairs > 0 else 0.0
        avg_utilization = total_utilization / num_pairs if num_pairs > 0 else 0.0
        
        # Find worst bottlenecks
        if layer_flow:
            worst_efficiency = min(layer_flow.values(), key=lambda x: x['efficiency'])
            best_efficiency = max(layer_flow.values(), key=lambda x: x['efficiency'])
        else:
            worst_efficiency = best_efficiency = {}
        
        self.log(logging.DEBUG, 
                f"Model information flow: avg efficiency={avg_efficiency:.3f}, "
                f"avg utilization={avg_utilization:.3f}")
        
        return {
            "mi_efficiency": avg_efficiency,
            "information_gap": avg_gap,
            "capacity_utilization": avg_utilization,
            "max_possible_mi": sum(v['max_mi'] for v in layer_flow.values()) / num_pairs if num_pairs > 0 else 0.0,
            "layer_flow": layer_flow
        }