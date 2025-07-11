"""
Redundancy metric component.

This component measures redundancy between neural network layers,
which indicates how much information is repeated or shared.
"""

from typing import Dict, Any, Union, Optional
import torch
import logging

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)
from .entropy_metric import EntropyMetric


class RedundancyMetric(BaseMetric):
    """
    Measures redundancy in neural network activations.
    
    Redundancy indicates how much information is shared or repeated
    between layers, which can indicate inefficient information encoding.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize redundancy metric.
        
        Args:
            name: Optional custom name
        """
        super().__init__(name or "RedundancyMetric")
        self._entropy_metric = EntropyMetric()
        self._measurement_schema = {
            "redundancy": float,
            "independence_ratio": float,
            "layer_redundancy": dict  # Per-layer-pair redundancy
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
                "metrics.redundancy",
                "metrics.independence_ratio",
                "metrics.layer_redundancy"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def _compute_metric(self, target: Union[ILayer, IModel], 
                       context: EvolutionContext) -> Dict[str, Any]:
        """
        Compute redundancy metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'layer_activations' data
            
        Returns:
            Dictionary containing redundancy measurements
        """
        # Get activations from context
        layer_activations = context.get('layer_activations')
        if layer_activations is None:
            raise ValueError("RedundancyMetric requires 'layer_activations' in context")
        
        if isinstance(target, IModel):
            return self._compute_model_redundancy(target, layer_activations)
        elif isinstance(target, ILayer):
            return self._compute_layer_redundancy(target, layer_activations)
        else:
            raise ValueError(f"Target must be ILayer or IModel, got {type(target)}")
    
    def _compute_layer_redundancy(self, layer: ILayer, 
                                 layer_activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute redundancy for a single layer with its input."""
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
                "redundancy": 0.0,
                "independence_ratio": 1.0,
                "layer_redundancy": {}
            }
        
        # Compute entropies
        h_input = self._entropy_metric._estimate_entropy(input_acts)
        h_output = self._entropy_metric._estimate_entropy(layer_acts)
        h_joint = self._entropy_metric.compute_joint_entropy(input_acts, layer_acts)
        
        # Redundancy = H(X) + H(Y) - H(X,Y)
        redundancy = h_input + h_output - h_joint
        
        # Independence ratio = MI / redundancy
        # MI = H(X) + H(Y) - H(X,Y) = redundancy
        # So independence_ratio indicates how much of the total information is shared
        total_info = h_input + h_output
        independence_ratio = redundancy / total_info if total_info > 0 else 0.0
        
        return {
            "redundancy": redundancy,
            "independence_ratio": independence_ratio,
            "layer_redundancy": {
                f"input->{layer.name}": redundancy
            }
        }
    
    def _compute_model_redundancy(self, model: IModel, 
                                 layer_activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute redundancy for entire model."""
        layer_names = list(layer_activations.keys())
        total_redundancy = 0.0
        layer_redundancy = {}
        num_pairs = 0
        
        # Compute redundancy between consecutive layers
        for i in range(len(layer_names) - 1):
            layer1_name = layer_names[i]
            layer2_name = layer_names[i + 1]
            
            acts1 = layer_activations[layer1_name]
            acts2 = layer_activations[layer2_name]
            
            if acts1 is None or acts2 is None:
                continue
            
            # Compute entropies
            h1 = self._entropy_metric._estimate_entropy(acts1)
            h2 = self._entropy_metric._estimate_entropy(acts2)
            h_joint = self._entropy_metric.compute_joint_entropy(acts1, acts2)
            
            # Redundancy
            redundancy = h1 + h2 - h_joint
            
            pair_key = f"{layer1_name}->{layer2_name}"
            layer_redundancy[pair_key] = redundancy
            
            total_redundancy += redundancy
            num_pairs += 1
        
        # Average redundancy
        avg_redundancy = total_redundancy / num_pairs if num_pairs > 0 else 0.0
        
        # Average independence ratio
        avg_independence_ratio = 0.0
        if num_pairs > 0:
            for i in range(len(layer_names) - 1):
                layer1_name = layer_names[i]
                layer2_name = layer_names[i + 1]
                
                acts1 = layer_activations.get(layer1_name)
                acts2 = layer_activations.get(layer2_name)
                
                if acts1 is None or acts2 is None:
                    continue
                
                h1 = self._entropy_metric._estimate_entropy(acts1)
                h2 = self._entropy_metric._estimate_entropy(acts2)
                total_info = h1 + h2
                
                pair_key = f"{layer1_name}->{layer2_name}"
                if pair_key in layer_redundancy and total_info > 0:
                    avg_independence_ratio += layer_redundancy[pair_key] / total_info
            
            avg_independence_ratio /= num_pairs
        
        self.log(logging.DEBUG, 
                f"Model redundancy: avg={avg_redundancy:.3f}, "
                f"independence ratio={avg_independence_ratio:.3f}")
        
        return {
            "redundancy": avg_redundancy,
            "independence_ratio": avg_independence_ratio,
            "layer_redundancy": layer_redundancy
        }