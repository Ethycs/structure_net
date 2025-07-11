"""
Activation distribution metric component.

This component analyzes the statistical distribution of activations including
percentiles, saturation analysis, and distribution characteristics.
"""

from typing import Dict, Any, Union, Optional, List
import torch
import torch.nn as nn
import logging

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


class ActivationDistributionMetric(BaseMetric):
    """
    Analyzes activation value distributions and saturation patterns.
    
    Provides detailed statistics about how activation values are distributed,
    helping identify saturation, gradient explosion risks, and other issues.
    """
    
    def __init__(self, saturation_threshold: float = 10.0, name: str = None):
        """
        Initialize activation distribution metric.
        
        Args:
            saturation_threshold: Threshold for considering neurons saturated
            name: Optional custom name
        """
        super().__init__(name or "ActivationDistributionMetric")
        self.saturation_threshold = saturation_threshold
        self._measurement_schema = {
            "max_activation": float,
            "mean_activation": float,
            "std_activation": float,
            "activation_percentiles": dict,
            "saturated_neurons": int,
            "saturation_ratio": float,
            "gradient_explosion_risk": bool
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"activations"},
            provided_outputs={
                "metrics.max_activation",
                "metrics.mean_activation",
                "metrics.std_activation",
                "metrics.activation_percentiles",
                "metrics.saturated_neurons",
                "metrics.saturation_ratio",
                "metrics.gradient_explosion_risk"
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
        Compute activation distribution metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'activations' tensor
            
        Returns:
            Dictionary containing distribution measurements
        """
        # Get activations
        activations = context.get('activations')
        if activations is None:
            raise ValueError("ActivationDistributionMetric requires 'activations' in context")
        
        # Ensure 2D tensor
        if activations.dim() != 2:
            raise ValueError(f"Expected 2D activations, got {activations.dim()}D")
        
        # Activation statistics
        abs_activations = activations.abs()
        max_activation = abs_activations.max().item()
        mean_activation = abs_activations.mean().item()
        std_activation = activations.std().item()
        
        # Percentile analysis
        quantile_values = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], 
                                      device=activations.device)
        activation_percentiles = torch.quantile(abs_activations.flatten(), quantile_values)
        
        percentiles_dict = {
            "p10": activation_percentiles[0].item(),
            "p25": activation_percentiles[1].item(),
            "p50": activation_percentiles[2].item(),
            "p75": activation_percentiles[3].item(),
            "p90": activation_percentiles[4].item(),
            "p95": activation_percentiles[5].item(),
            "p99": activation_percentiles[6].item()
        }
        
        # Saturation analysis
        saturated_neurons = (abs_activations > self.saturation_threshold).any(dim=0).sum().item()
        total_neurons = activations.shape[1]
        saturation_ratio = saturated_neurons / total_neurons if total_neurons > 0 else 0
        
        # Gradient explosion detection
        gradient_explosion_risk = max_activation > self.saturation_threshold
        
        self.log(logging.DEBUG, 
                f"Distribution: max={max_activation:.3f}, mean={mean_activation:.3f}, "
                f"saturated={saturated_neurons}")
        
        return {
            "max_activation": max_activation,
            "mean_activation": mean_activation,
            "std_activation": std_activation,
            "activation_percentiles": percentiles_dict,
            "saturated_neurons": saturated_neurons,
            "saturation_ratio": saturation_ratio,
            "gradient_explosion_risk": gradient_explosion_risk
        }