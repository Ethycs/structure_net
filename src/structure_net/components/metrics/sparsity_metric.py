"""
Sparsity metric component for measuring network sparsity.

This component provides focused measurements of sparsity ratios
for layers and models in the Structure Net framework.
"""

from typing import Dict, Any, Union
import torch
import torch.nn as nn
import logging

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


class SparsityMetric(BaseMetric):
    """
    Measures sparsity ratio of network weights.
    
    Provides focused sparsity measurements for individual layers
    or entire models. This is a low-level metric that other
    analyzers can use.
    """
    
    def __init__(self, threshold: float = 1e-6, name: str = None):
        """
        Initialize sparsity metric.
        
        Args:
            threshold: Values below this are considered zero
            name: Optional custom name for the metric
        """
        super().__init__(name or "SparsityMetric")
        self.threshold = threshold
        self._measurement_schema = {
            "sparsity_ratio": float,
            "num_zeros": int,
            "num_total": int,
            "density_ratio": float
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"target"},  # Either ILayer or IModel
            provided_outputs={
                "metrics.sparsity_ratio",
                "metrics.num_zeros",
                "metrics.num_total",
                "metrics.density_ratio"
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
        Compute sparsity metrics for the target.
        
        Args:
            target: Layer or model to analyze
            context: Evolution context (not used for this metric)
            
        Returns:
            Dictionary containing sparsity measurements
        """
        if isinstance(target, IModel):
            # Compute sparsity for entire model
            return self._compute_model_sparsity(target)
        elif isinstance(target, ILayer):
            # Compute sparsity for single layer
            return self._compute_layer_sparsity(target)
        else:
            raise ValueError(f"Target must be ILayer or IModel, got {type(target)}")
    
    def _compute_layer_sparsity(self, layer: ILayer) -> Dict[str, Any]:
        """Compute sparsity for a single layer."""
        # Get weight parameters
        weights = []
        for name, param in layer.named_parameters():
            if 'weight' in name:
                weights.append(param.data.flatten())
        
        if not weights:
            # No weight parameters found
            return {
                "sparsity_ratio": 0.0,
                "num_zeros": 0,
                "num_total": 0,
                "density_ratio": 1.0
            }
        
        # Concatenate all weights
        all_weights = torch.cat(weights)
        
        # Count zeros
        num_zeros = (all_weights.abs() < self.threshold).sum().item()
        num_total = all_weights.numel()
        
        sparsity_ratio = num_zeros / num_total if num_total > 0 else 0.0
        density_ratio = 1.0 - sparsity_ratio
        
        return {
            "sparsity_ratio": sparsity_ratio,
            "num_zeros": num_zeros,
            "num_total": num_total,
            "density_ratio": density_ratio
        }
    
    def _compute_model_sparsity(self, model: IModel) -> Dict[str, Any]:
        """Compute sparsity for entire model."""
        total_zeros = 0
        total_params = 0
        
        # Iterate through all parameters
        for name, param in model.named_parameters():
            if 'weight' in name:  # Only consider weight matrices
                num_zeros = (param.data.abs() < self.threshold).sum().item()
                num_total = param.numel()
                
                total_zeros += num_zeros
                total_params += num_total
        
        sparsity_ratio = total_zeros / total_params if total_params > 0 else 0.0
        density_ratio = 1.0 - sparsity_ratio
        
        self.log(logging.DEBUG, 
                f"Model sparsity: {sparsity_ratio:.2%} ({total_zeros}/{total_params} zeros)")
        
        return {
            "sparsity_ratio": sparsity_ratio,
            "num_zeros": total_zeros,
            "num_total": total_params,
            "density_ratio": density_ratio
        }
    
    def get_layer_sparsity_map(self, model: IModel) -> Dict[str, float]:
        """
        Get sparsity ratio for each layer in the model.
        
        This is a convenience method for detailed analysis.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary mapping layer names to sparsity ratios
        """
        sparsity_map = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or hasattr(module, 'weight'):
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight.data
                    num_zeros = (weight.abs() < self.threshold).sum().item()
                    num_total = weight.numel()
                    sparsity = num_zeros / num_total if num_total > 0 else 0.0
                    sparsity_map[name] = sparsity
        
        return sparsity_map