"""
Activity pattern metric component.

This component analyzes patterns in neuron activity including entropy,
Gini coefficient, and overall diversity of activations.
"""

from typing import Dict, Any, Union, Optional
import torch
import torch.nn as nn
import logging
import numpy as np

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


class ActivityPatternMetric(BaseMetric):
    """
    Analyzes patterns and diversity in neuron activity.
    
    Measures entropy, inequality (Gini coefficient), and other pattern
    characteristics that indicate the health and diversity of neural
    network activations.
    """
    
    def __init__(self, activation_threshold: float = 0.01, name: str = None):
        """
        Initialize activity pattern metric.
        
        Args:
            activation_threshold: Threshold for activity detection
            name: Optional custom name
        """
        super().__init__(name or "ActivityPatternMetric")
        self.activation_threshold = activation_threshold
        self._measurement_schema = {
            "activity_entropy": float,
            "activity_gini": float,
            "diversity_score": float,
            "sparsity_imbalance": float
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
                "metrics.activity_entropy",
                "metrics.activity_gini",
                "metrics.diversity_score",
                "metrics.sparsity_imbalance"
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
        Compute activity pattern metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'activations' tensor
            
        Returns:
            Dictionary containing pattern measurements
        """
        # Get activations
        activations = context.get('activations')
        if activations is None:
            raise ValueError("ActivityPatternMetric requires 'activations' in context")
        
        # Ensure 2D tensor
        if activations.dim() != 2:
            raise ValueError(f"Expected 2D activations, got {activations.dim()}D")
        
        # Compute activity rates per neuron
        active_mask = activations.abs() > self.activation_threshold
        neuron_activity_rates = active_mask.float().mean(dim=0)
        
        # Activity entropy
        activity_entropy = self._compute_activity_entropy(neuron_activity_rates)
        
        # Gini coefficient (inequality measure)
        activity_gini = self._compute_gini_coefficient(neuron_activity_rates)
        
        # Diversity score (combination of entropy and 1-gini)
        diversity_score = (activity_entropy / 5.0) * (1 - activity_gini)  # Normalized
        
        # Sparsity imbalance (how uneven the activity is)
        mean_activity = neuron_activity_rates.mean().item()
        std_activity = neuron_activity_rates.std().item()
        sparsity_imbalance = std_activity / (mean_activity + 1e-10)
        
        self.log(logging.DEBUG, 
                f"Patterns: entropy={activity_entropy:.3f}, gini={activity_gini:.3f}, "
                f"diversity={diversity_score:.3f}")
        
        return {
            "activity_entropy": activity_entropy,
            "activity_gini": activity_gini,
            "diversity_score": diversity_score,
            "sparsity_imbalance": sparsity_imbalance
        }
    
    def _compute_activity_entropy(self, activity_rates: torch.Tensor) -> float:
        """Compute entropy of activity distribution."""
        # Normalize to probabilities
        probs = activity_rates / (activity_rates.sum() + 1e-10)
        probs = probs[probs > 0]
        
        if len(probs) == 0:
            return 0.0
        
        # Shannon entropy
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
        
        return entropy
    
    def _compute_gini_coefficient(self, activity_rates: torch.Tensor) -> float:
        """
        Compute Gini coefficient of activity distribution.
        
        0 = perfect equality, 1 = perfect inequality
        """
        sorted_rates = torch.sort(activity_rates)[0]
        n = len(sorted_rates)
        
        if n == 0:
            return 0.0
        
        cumsum = torch.cumsum(sorted_rates, dim=0)
        
        if cumsum[-1] == 0:
            return 0.0
        
        # Gini coefficient formula
        gini = (n + 1 - 2 * torch.sum(cumsum) / cumsum[-1]) / n
        
        return gini.item()