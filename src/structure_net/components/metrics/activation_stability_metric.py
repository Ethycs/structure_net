"""
Activation stability metric component.

This component measures how stable activation patterns are over time,
detecting rapid changes that may indicate catastrophic events.
"""

from typing import Dict, Any, Union, Optional, List
import torch
import torch.nn as nn
import logging
import numpy as np

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


class ActivationStabilityMetric(BaseMetric):
    """
    Measures activation pattern stability and change rates.
    
    Tracks how quickly activation patterns change between consecutive
    inputs, which can indicate potential catastrophic forgetting or
    sudden performance drops.
    """
    
    def __init__(self, activation_threshold: float = 0.0, name: str = None):
        """
        Initialize activation stability metric.
        
        Args:
            activation_threshold: Threshold for binary activation patterns
            name: Optional custom name
        """
        super().__init__(name or "ActivationStabilityMetric")
        self.activation_threshold = activation_threshold
        self._measurement_schema = {
            "mean_change_rate": float,
            "max_change_rate": float,
            "change_variance": float,
            "stability_score": float,
            "rapid_change_count": int
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"activation_trajectory"},
            provided_outputs={
                "metrics.mean_change_rate",
                "metrics.max_change_rate",
                "metrics.change_variance",
                "metrics.stability_score",
                "metrics.rapid_change_count"
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
        Compute activation stability metrics.
        
        Args:
            target: Model to analyze
            context: Must contain 'activation_trajectory'
            
        Returns:
            Dictionary containing stability measurements
        """
        # Get activation trajectory
        trajectory = context.get('activation_trajectory')
        if trajectory is None:
            raise ValueError("ActivationStabilityMetric requires 'activation_trajectory' in context")
        
        if len(trajectory) < 2:
            return self._empty_metrics()
        
        # Compute change rates between consecutive activations
        pattern_changes = []
        
        for t in range(len(trajectory) - 1):
            act_t = trajectory[t]
            act_t1 = trajectory[t + 1]
            
            # Create binary patterns
            pattern_t = (act_t > self.activation_threshold).float()
            pattern_t1 = (act_t1 > self.activation_threshold).float()
            
            # Ensure same shape
            if pattern_t.shape != pattern_t1.shape:
                self.log(logging.WARNING, 
                        f"Shape mismatch at step {t}: {pattern_t.shape} vs {pattern_t1.shape}")
                continue
            
            # Compute change rate
            change_rate = (pattern_t != pattern_t1).float().mean().item()
            pattern_changes.append(change_rate)
        
        if not pattern_changes:
            return self._empty_metrics()
        
        # Compute statistics
        mean_change_rate = np.mean(pattern_changes)
        max_change_rate = np.max(pattern_changes)
        change_variance = np.var(pattern_changes)
        
        # Count rapid changes (>50% neurons changing)
        rapid_change_count = sum(1 for c in pattern_changes if c > 0.5)
        
        # Stability score (inverse of mean change rate)
        stability_score = 1.0 / (1.0 + mean_change_rate * 10)
        
        self.log(logging.DEBUG, 
                f"Stability: mean_change={mean_change_rate:.3f}, "
                f"max_change={max_change_rate:.3f}, score={stability_score:.3f}")
        
        return {
            "mean_change_rate": mean_change_rate,
            "max_change_rate": max_change_rate,
            "change_variance": change_variance,
            "stability_score": stability_score,
            "rapid_change_count": rapid_change_count
        }
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics when computation cannot proceed."""
        return {
            "mean_change_rate": 0.0,
            "max_change_rate": 0.0,
            "change_variance": 0.0,
            "stability_score": 1.0,
            "rapid_change_count": 0
        }