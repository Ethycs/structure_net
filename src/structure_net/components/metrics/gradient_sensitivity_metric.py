"""
Gradient sensitivity metric component.

This component measures sensitivity of neural network layers to gradient changes,
which helps identify critical connections and potential improvements.
"""

from typing import Dict, Any, Union, Optional, Tuple
import torch
import torch.nn as nn
import logging

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


class GradientSensitivityMetric(BaseMetric):
    """
    Measures gradient-based sensitivity between neural network layers.
    
    This metric analyzes how sensitive layer connections are to changes,
    which helps identify where architectural modifications would have
    the most impact.
    """
    
    def __init__(self, threshold: float = 0.01, name: str = None):
        """
        Initialize gradient sensitivity metric.
        
        Args:
            threshold: Activation threshold for active neurons
            name: Optional custom name
        """
        super().__init__(name or "GradientSensitivityMetric")
        self.threshold = threshold
        self._measurement_schema = {
            "gradient_sensitivity": float,
            "virtual_parameter_sensitivity": float,
            "sensitivity_variance": float,
            "gradient_flow_health": float,
            "active_ratio": float
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"activations_i", "activations_j", "gradients_i", "gradients_j"},
            provided_outputs={
                "metrics.gradient_sensitivity",
                "metrics.virtual_parameter_sensitivity",
                "metrics.sensitivity_variance",
                "metrics.gradient_flow_health",
                "metrics.active_ratio"
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
        Compute gradient sensitivity metrics.
        
        Args:
            target: Layer or model to analyze (not used directly)
            context: Must contain activation and gradient data
            
        Returns:
            Dictionary containing sensitivity measurements
        """
        # Get required data from context
        acts_i = context.get('activations_i')
        acts_j = context.get('activations_j')
        grads_i = context.get('gradients_i')
        grads_j = context.get('gradients_j')
        
        if any(x is None for x in [acts_i, acts_j, grads_i, grads_j]):
            raise ValueError(
                "GradientSensitivityMetric requires 'activations_i', 'activations_j', "
                "'gradients_i', and 'gradients_j' in context"
            )
        
        # Validate tensor dimensions
        for tensor, name in [(acts_i, 'acts_i'), (acts_j, 'acts_j'), 
                            (grads_i, 'grads_i'), (grads_j, 'grads_j')]:
            if tensor.dim() != 2:
                raise ValueError(f"{name} must be 2D tensor [batch_size, features]")
        
        # Apply activation threshold
        active_mask_i = acts_i.abs() > self.threshold
        active_mask_j = acts_j.abs() > self.threshold
        
        # Check if we have any active neurons
        if not active_mask_i.any() or not active_mask_j.any():
            return {
                "gradient_sensitivity": 0.0,
                "virtual_parameter_sensitivity": 0.0,
                "sensitivity_variance": 0.0,
                "gradient_flow_health": 0.0,
                "active_ratio": 0.0
            }
        
        # Compute virtual parameter sensitivity
        virtual_sensitivity = self._compute_virtual_parameter_sensitivity(
            acts_i, acts_j, grads_i, grads_j
        )
        
        # Compute gradient sensitivity statistics
        gradient_norms_i = grads_i.norm(dim=1)
        gradient_norms_j = grads_j.norm(dim=1)
        
        # Combined gradient sensitivity
        combined_sensitivity = (gradient_norms_i.mean() * gradient_norms_j.mean()).item()
        
        # Sensitivity variance (stability measure)
        sensitivity_variance = (gradient_norms_i.var() + gradient_norms_j.var()).item() / 2
        
        # Active neuron ratio
        active_ratio_i = active_mask_i.float().mean().item()
        active_ratio_j = active_mask_j.float().mean().item()
        avg_active_ratio = (active_ratio_i + active_ratio_j) / 2
        
        # Gradient flow health
        gradient_flow_health = self._assess_gradient_flow_health(
            gradient_norms_i, gradient_norms_j, active_ratio_i, active_ratio_j
        )
        
        self.log(logging.DEBUG, 
                f"Gradient sensitivity: {combined_sensitivity:.6f}, "
                f"virtual: {virtual_sensitivity:.6f}, health: {gradient_flow_health:.3f}")
        
        return {
            "gradient_sensitivity": combined_sensitivity,
            "virtual_parameter_sensitivity": virtual_sensitivity,
            "sensitivity_variance": sensitivity_variance,
            "gradient_flow_health": gradient_flow_health,
            "active_ratio": avg_active_ratio
        }
    
    def _compute_virtual_parameter_sensitivity(self, acts_i: torch.Tensor, 
                                             acts_j: torch.Tensor,
                                             grads_i: torch.Tensor, 
                                             grads_j: torch.Tensor) -> float:
        """
        Compute sensitivity to virtual parameters between layers.
        
        This measures how sensitive the network would be to adding
        new connections between the layers.
        """
        # Average activations and gradients across batch
        acts_i_mean = acts_i.mean(dim=0)  # Shape: [features_i]
        grads_j_mean = grads_j.mean(dim=0)  # Shape: [features_j]
        
        # Compute sensitivity as the norm of the outer product
        # This represents sensitivity to adding connections between all pairs
        if acts_i_mean.numel() > 0 and grads_j_mean.numel() > 0:
            # Use broadcasting to compute outer product efficiently
            outer_product = acts_i_mean.unsqueeze(1) * grads_j_mean.unsqueeze(0)
            sensitivity = torch.norm(outer_product).item()
        else:
            sensitivity = 0.0
        
        return sensitivity
    
    def _assess_gradient_flow_health(self, gradient_norms_i: torch.Tensor,
                                   gradient_norms_j: torch.Tensor,
                                   active_ratio_i: float,
                                   active_ratio_j: float) -> float:
        """
        Assess the health of gradient flow between layers.
        
        Returns a score between 0 and 1, where 1 is healthy flow.
        """
        # Mean gradient magnitudes
        mean_grad_i = gradient_norms_i.mean().item()
        mean_grad_j = gradient_norms_j.mean().item()
        
        # Gradient variance (lower is more stable)
        var_grad_i = gradient_norms_i.var().item()
        var_grad_j = gradient_norms_j.var().item()
        
        # Healthy gradients should be:
        # 1. Not too small (vanishing) or too large (exploding)
        # 2. Relatively stable (low variance)
        # 3. Have reasonable active neuron ratios
        
        # Magnitude score (penalize very high or very low)
        ideal_magnitude = 1.0
        magnitude_score_i = 1.0 / (1.0 + abs(torch.log10(torch.tensor(mean_grad_i + 1e-8))))
        magnitude_score_j = 1.0 / (1.0 + abs(torch.log10(torch.tensor(mean_grad_j + 1e-8))))
        magnitude_score = (magnitude_score_i + magnitude_score_j) / 2
        
        # Stability score (reward low variance)
        stability_score = 1.0 / (1.0 + (var_grad_i + var_grad_j) / 2)
        
        # Activity score (penalize very low active ratios)
        activity_score = (active_ratio_i + active_ratio_j) / 2
        
        # Combined health score
        health = (magnitude_score * 0.4 + 
                 stability_score * 0.3 + 
                 activity_score * 0.3)
        
        return float(health)
    
    def compute_layer_pair_sensitivity(self, layer_i_data: Dict[str, torch.Tensor],
                                     layer_j_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Compute sensitivity between a specific pair of layers.
        
        Args:
            layer_i_data: Dict with 'activations' and 'gradients' for layer i
            layer_j_data: Dict with 'activations' and 'gradients' for layer j
            
        Returns:
            Dictionary with sensitivity metrics
        """
        context = EvolutionContext({
            'activations_i': layer_i_data['activations'],
            'activations_j': layer_j_data['activations'],
            'gradients_i': layer_i_data['gradients'],
            'gradients_j': layer_j_data['gradients']
        })
        
        return self._compute_metric(None, context)