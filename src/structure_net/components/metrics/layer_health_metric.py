"""
Layer health metric component.

This component computes an overall health score for neural network layers
based on activity, saturation, and diversity metrics.
"""

from typing import Dict, Any, Union, Optional
import torch
import torch.nn as nn
import logging

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


class LayerHealthMetric(BaseMetric):
    """
    Computes overall layer health scores.
    
    Combines multiple factors including activity levels, saturation,
    and diversity to provide a single health score that indicates
    how well a layer is functioning.
    """
    
    def __init__(self, target_active_ratio: float = 0.1,
                 max_healthy_activation: float = 5.0,
                 target_entropy: float = 3.0,
                 name: str = None):
        """
        Initialize layer health metric.
        
        Args:
            target_active_ratio: Target ratio of active neurons
            max_healthy_activation: Maximum activation before penalizing
            target_entropy: Target entropy for diversity
            name: Optional custom name
        """
        super().__init__(name or "LayerHealthMetric")
        self.target_active_ratio = target_active_ratio
        self.max_healthy_activation = max_healthy_activation
        self.target_entropy = target_entropy
        self._measurement_schema = {
            "layer_health_score": float,
            "activity_health": float,
            "saturation_health": float,
            "diversity_health": float,
            "health_diagnosis": str
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={
                "metrics.active_ratio",
                "metrics.max_activation",
                "metrics.activity_entropy"
            },
            optional_inputs={
                "metrics.saturation_ratio",
                "metrics.activity_gini"
            },
            provided_outputs={
                "metrics.layer_health_score",
                "metrics.activity_health",
                "metrics.saturation_health",
                "metrics.diversity_health",
                "metrics.health_diagnosis"
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
        Compute layer health metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain required metric data
            
        Returns:
            Dictionary containing health measurements
        """
        # Get required metrics
        active_ratio = context.get('metrics.active_ratio')
        max_activation = context.get('metrics.max_activation')
        activity_entropy = context.get('metrics.activity_entropy')
        
        if any(x is None for x in [active_ratio, max_activation, activity_entropy]):
            raise ValueError("LayerHealthMetric requires activity metrics in context")
        
        # Optional metrics
        saturation_ratio = context.get('metrics.saturation_ratio', 0.0)
        activity_gini = context.get('metrics.activity_gini', 0.5)
        
        # Compute activity health (penalize too few active neurons)
        activity_health = min(1.0, active_ratio / self.target_active_ratio)
        
        # Compute saturation health (penalize high activations)
        if max_activation < self.max_healthy_activation:
            saturation_health = 1.0
        else:
            # Linear decay after threshold
            saturation_health = max(0.0, 1.0 - (max_activation - self.max_healthy_activation) / 10.0)
        
        # Also consider saturation ratio if available
        if saturation_ratio > 0.1:  # More than 10% saturated
            saturation_health *= (1.0 - saturation_ratio)
        
        # Compute diversity health (reward entropy)
        diversity_health = min(1.0, activity_entropy / self.target_entropy)
        
        # Consider Gini coefficient if available (lower is better)
        if activity_gini is not None:
            diversity_health *= (1.0 - activity_gini * 0.5)  # Partial penalty
        
        # Overall health score (weighted average)
        layer_health_score = (
            activity_health * 0.4 +
            saturation_health * 0.3 +
            diversity_health * 0.3
        )
        
        # Diagnose main issues
        health_diagnosis = self._diagnose_health(
            activity_health, saturation_health, diversity_health
        )
        
        self.log(logging.DEBUG, 
                f"Health: score={layer_health_score:.3f}, diagnosis={health_diagnosis}")
        
        return {
            "layer_health_score": layer_health_score,
            "activity_health": activity_health,
            "saturation_health": saturation_health,
            "diversity_health": diversity_health,
            "health_diagnosis": health_diagnosis
        }
    
    def _diagnose_health(self, activity_health: float, saturation_health: float,
                        diversity_health: float) -> str:
        """Diagnose the main health issues."""
        min_score = min(activity_health, saturation_health, diversity_health)
        
        if min_score > 0.8:
            return "healthy"
        elif activity_health == min_score:
            return "low_activity"
        elif saturation_health == min_score:
            return "saturation_risk"
        elif diversity_health == min_score:
            return "poor_diversity"
        else:
            return "multiple_issues"