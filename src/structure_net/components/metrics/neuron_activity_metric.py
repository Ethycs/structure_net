"""
Neuron activity metric component.

This component measures basic neuron activity patterns including active/dead
neuron ratios and activation statistics.
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


class NeuronActivityMetric(BaseMetric):
    """
    Measures neuron activity levels and identifies active/dead neurons.
    
    This metric provides basic activity statistics that are fundamental
    for understanding neural network behavior.
    """
    
    def __init__(self, activation_threshold: float = 0.01, name: str = None):
        """
        Initialize neuron activity metric.
        
        Args:
            activation_threshold: Threshold for considering a neuron active
            name: Optional custom name
        """
        super().__init__(name or "NeuronActivityMetric")
        self.activation_threshold = activation_threshold
        self._measurement_schema = {
            "active_neurons": int,
            "total_neurons": int,
            "active_ratio": float,
            "dead_ratio": float,
            "persistent_active": int,
            "neuron_activity_rates": list
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
                "metrics.active_neurons",
                "metrics.total_neurons",
                "metrics.active_ratio",
                "metrics.dead_ratio",
                "metrics.persistent_active",
                "metrics.neuron_activity_rates"
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
        Compute neuron activity metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'activations' tensor
            
        Returns:
            Dictionary containing activity measurements
        """
        # Get activations
        activations = context.get('activations')
        if activations is None:
            raise ValueError("NeuronActivityMetric requires 'activations' in context")
        
        # Ensure 2D tensor [batch_size, num_neurons]
        if activations.dim() != 2:
            raise ValueError(f"Expected 2D activations, got {activations.dim()}D")
        
        # Basic activity detection
        active_mask = activations.abs() > self.activation_threshold
        
        # Per-neuron statistics
        neuron_activity_rates = active_mask.float().mean(dim=0)
        
        # Neuron counts
        active_neurons = active_mask.any(dim=0).sum().item()
        total_neurons = activations.shape[1]
        active_ratio = active_neurons / total_neurons if total_neurons > 0 else 0
        dead_ratio = 1 - active_ratio
        
        # Persistent active neurons (active in most samples)
        persistence_ratio = context.get('persistence_ratio', 0.8)
        persistent_active = (neuron_activity_rates > persistence_ratio).sum().item()
        
        self.log(logging.DEBUG, 
                f"Activity: {active_neurons}/{total_neurons} active ({active_ratio:.2%})")
        
        return {
            "active_neurons": active_neurons,
            "total_neurons": total_neurons,
            "active_ratio": active_ratio,
            "dead_ratio": dead_ratio,
            "persistent_active": persistent_active,
            "neuron_activity_rates": neuron_activity_rates.tolist()
        }