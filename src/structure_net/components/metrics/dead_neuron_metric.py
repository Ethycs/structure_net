"""
Dead neuron detection metric component.

This component identifies neurons that have stopped contributing
to the network's computation (dead neurons).
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


class DeadNeuronMetric(BaseMetric):
    """
    Detects dead neurons in neural network layers.
    
    A neuron is considered "dead" if its activation is consistently
    below a threshold across multiple inputs. This metric helps
    identify network capacity that isn't being utilized.
    """
    
    def __init__(self, activation_threshold: float = 0.01, 
                 sample_size: int = 100, name: str = None):
        """
        Initialize dead neuron metric.
        
        Args:
            activation_threshold: Activations below this are considered dead
            sample_size: Number of samples to use for detection
            name: Optional custom name
        """
        super().__init__(name or "DeadNeuronMetric")
        self.activation_threshold = activation_threshold
        self.sample_size = sample_size
        self._measurement_schema = {
            "dead_neuron_ratio": float,
            "num_dead": int,
            "num_total": int,
            "layer_dead_ratios": dict  # Per-layer breakdown
        }
        self._activation_cache = {}
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"target", "activations"},
            provided_outputs={
                "metrics.dead_neuron_ratio",
                "metrics.num_dead",
                "metrics.num_total",
                "metrics.layer_dead_ratios"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,  # Needs to store activations
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def _compute_metric(self, target: Union[ILayer, IModel], 
                       context: EvolutionContext) -> Dict[str, Any]:
        """
        Compute dead neuron metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'activations' data
            
        Returns:
            Dictionary containing dead neuron measurements
        """
        # Get activations from context
        activations = context.get('activations')
        if activations is None:
            raise ValueError("DeadNeuronMetric requires 'activations' in context")
        
        if isinstance(target, IModel):
            return self._compute_model_dead_neurons(target, activations)
        elif isinstance(target, ILayer):
            return self._compute_layer_dead_neurons(target, activations)
        else:
            raise ValueError(f"Target must be ILayer or IModel, got {type(target)}")
    
    def _compute_layer_dead_neurons(self, layer: ILayer, 
                                   activations: torch.Tensor) -> Dict[str, Any]:
        """Compute dead neurons for a single layer."""
        # Ensure we have 2D activations (batch_size, num_neurons)
        if activations.dim() > 2:
            # Flatten spatial dimensions if present
            activations = activations.view(activations.size(0), -1)
        
        # Compute max activation per neuron across batch
        max_activations = activations.abs().max(dim=0)[0]
        
        # Count dead neurons
        num_dead = (max_activations < self.activation_threshold).sum().item()
        num_total = max_activations.numel()
        dead_ratio = num_dead / num_total if num_total > 0 else 0.0
        
        return {
            "dead_neuron_ratio": dead_ratio,
            "num_dead": num_dead,
            "num_total": num_total,
            "layer_dead_ratios": {layer.name: dead_ratio}
        }
    
    def _compute_model_dead_neurons(self, model: IModel, 
                                   activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute dead neurons for entire model."""
        total_dead = 0
        total_neurons = 0
        layer_dead_ratios = {}
        
        # Process each layer's activations
        for layer_name, layer_acts in activations.items():
            if layer_acts is None:
                continue
                
            # Flatten if needed
            if layer_acts.dim() > 2:
                layer_acts = layer_acts.view(layer_acts.size(0), -1)
            
            # Compute max activation per neuron
            max_acts = layer_acts.abs().max(dim=0)[0]
            
            # Count dead neurons
            num_dead = (max_acts < self.activation_threshold).sum().item()
            num_total = max_acts.numel()
            
            total_dead += num_dead
            total_neurons += num_total
            
            if num_total > 0:
                layer_dead_ratios[layer_name] = num_dead / num_total
        
        overall_ratio = total_dead / total_neurons if total_neurons > 0 else 0.0
        
        self.log(logging.DEBUG, 
                f"Model dead neurons: {overall_ratio:.2%} ({total_dead}/{total_neurons})")
        
        return {
            "dead_neuron_ratio": overall_ratio,
            "num_dead": total_dead,
            "num_total": total_neurons,
            "layer_dead_ratios": layer_dead_ratios
        }
    
    def track_activations(self, layer_name: str, activations: torch.Tensor):
        """
        Track activations over time for more accurate dead neuron detection.
        
        Args:
            layer_name: Name of the layer
            activations: Activation tensor for this batch
        """
        if layer_name not in self._activation_cache:
            self._activation_cache[layer_name] = []
        
        # Keep only recent samples
        self._activation_cache[layer_name].append(activations.detach().cpu())
        if len(self._activation_cache[layer_name]) > self.sample_size:
            self._activation_cache[layer_name].pop(0)
    
    def get_persistent_dead_neurons(self, layer_name: str, 
                                   persistence_threshold: float = 0.9) -> torch.Tensor:
        """
        Get neurons that are persistently dead across multiple batches.
        
        Args:
            layer_name: Name of the layer to check
            persistence_threshold: Fraction of batches where neuron must be dead
            
        Returns:
            Boolean tensor indicating which neurons are persistently dead
        """
        if layer_name not in self._activation_cache:
            return None
        
        cache = self._activation_cache[layer_name]
        if not cache:
            return None
        
        # Stack all cached activations
        all_acts = torch.stack(cache, dim=0)  # (num_samples, batch_size, num_neurons)
        
        # Flatten batch dimension
        all_acts = all_acts.view(-1, all_acts.size(-1))  # (total_samples, num_neurons)
        
        # Count how often each neuron is below threshold
        below_threshold = (all_acts.abs() < self.activation_threshold)
        dead_fraction = below_threshold.float().mean(dim=0)
        
        # Identify persistently dead neurons
        persistently_dead = dead_fraction > persistence_threshold
        
        return persistently_dead