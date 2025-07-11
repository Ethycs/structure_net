"""
Gradient metric component for analyzing gradient flow.

This component measures gradient statistics to understand training
dynamics and identify potential optimization issues.
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


class GradientMetric(BaseMetric):
    """
    Measures gradient statistics during backpropagation.
    
    Provides insights into gradient flow, vanishing/exploding gradients,
    and optimization dynamics for neural network training.
    """
    
    def __init__(self, track_history: bool = True, 
                 history_size: int = 100, name: str = None):
        """
        Initialize gradient metric.
        
        Args:
            track_history: Whether to track gradient history
            history_size: Number of steps to keep in history
            name: Optional custom name
        """
        super().__init__(name or "GradientMetric")
        self.track_history = track_history
        self.history_size = history_size
        self._gradient_history = {} if track_history else None
        self._measurement_schema = {
            "gradient_norm": float,
            "gradient_variance": float,
            "gradient_mean": float,
            "gradient_max": float,
            "gradient_min": float,
            "layer_gradients": dict,  # Per-layer gradient stats
            "vanishing_ratio": float,
            "exploding_ratio": float
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"target", "gradients"},
            provided_outputs={
                "metrics.gradient_norm",
                "metrics.gradient_variance",
                "metrics.gradient_mean",
                "metrics.gradient_max",
                "metrics.gradient_min",
                "metrics.layer_gradients",
                "metrics.vanishing_ratio",
                "metrics.exploding_ratio"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM if self.track_history else ResourceLevel.LOW,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def _compute_metric(self, target: Union[ILayer, IModel], 
                       context: EvolutionContext) -> Dict[str, Any]:
        """
        Compute gradient metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'gradients' data
            
        Returns:
            Dictionary containing gradient measurements
        """
        # Get gradients from context
        gradients = context.get('gradients')
        if gradients is None:
            raise ValueError("GradientMetric requires 'gradients' in context")
        
        if isinstance(target, IModel):
            return self._compute_model_gradients(target, gradients)
        elif isinstance(target, ILayer):
            return self._compute_layer_gradients(target, gradients)
        else:
            raise ValueError(f"Target must be ILayer or IModel, got {type(target)}")
    
    def _compute_layer_gradients(self, layer: ILayer, 
                                gradients: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Compute gradient statistics for a single layer."""
        # Handle different gradient formats
        if isinstance(gradients, dict):
            layer_grad = gradients.get(layer.name)
            if layer_grad is None:
                raise ValueError(f"No gradients found for layer {layer.name}")
        else:
            layer_grad = gradients
        
        # Flatten gradient tensor
        flat_grad = layer_grad.flatten()
        
        # Compute statistics
        grad_norm = torch.norm(flat_grad).item()
        grad_mean = flat_grad.mean().item()
        grad_var = flat_grad.var().item()
        grad_max = flat_grad.max().item()
        grad_min = flat_grad.min().item()
        
        # Check for vanishing/exploding gradients
        vanishing_threshold = 1e-7
        exploding_threshold = 1e3
        
        vanishing_ratio = (flat_grad.abs() < vanishing_threshold).float().mean().item()
        exploding_ratio = (flat_grad.abs() > exploding_threshold).float().mean().item()
        
        # Update history if tracking
        if self.track_history and self._gradient_history is not None:
            if layer.name not in self._gradient_history:
                self._gradient_history[layer.name] = []
            
            self._gradient_history[layer.name].append({
                'norm': grad_norm,
                'mean': grad_mean,
                'variance': grad_var
            })
            
            # Keep only recent history
            if len(self._gradient_history[layer.name]) > self.history_size:
                self._gradient_history[layer.name].pop(0)
        
        return {
            "gradient_norm": grad_norm,
            "gradient_variance": grad_var,
            "gradient_mean": grad_mean,
            "gradient_max": grad_max,
            "gradient_min": grad_min,
            "layer_gradients": {layer.name: {
                "norm": grad_norm,
                "variance": grad_var,
                "mean": grad_mean
            }},
            "vanishing_ratio": vanishing_ratio,
            "exploding_ratio": exploding_ratio
        }
    
    def _compute_model_gradients(self, model: IModel, 
                               gradients: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute gradient statistics for entire model."""
        total_norm = 0.0
        all_gradients = []
        layer_gradients = {}
        total_vanishing = 0
        total_exploding = 0
        total_params = 0
        
        # Process each layer's gradients
        for layer_name, layer_grad in gradients.items():
            if layer_grad is None:
                continue
            
            flat_grad = layer_grad.flatten()
            
            # Layer statistics
            layer_norm = torch.norm(flat_grad).item()
            layer_mean = flat_grad.mean().item()
            layer_var = flat_grad.var().item()
            
            layer_gradients[layer_name] = {
                "norm": layer_norm,
                "variance": layer_var,
                "mean": layer_mean,
                "max": flat_grad.max().item(),
                "min": flat_grad.min().item()
            }
            
            # Accumulate for global statistics
            total_norm += layer_norm ** 2
            all_gradients.append(flat_grad)
            
            # Count vanishing/exploding
            vanishing_threshold = 1e-7
            exploding_threshold = 1e3
            
            total_vanishing += (flat_grad.abs() < vanishing_threshold).sum().item()
            total_exploding += (flat_grad.abs() > exploding_threshold).sum().item()
            total_params += flat_grad.numel()
            
            # Update history
            if self.track_history and self._gradient_history is not None:
                if layer_name not in self._gradient_history:
                    self._gradient_history[layer_name] = []
                
                self._gradient_history[layer_name].append({
                    'norm': layer_norm,
                    'mean': layer_mean,
                    'variance': layer_var
                })
                
                if len(self._gradient_history[layer_name]) > self.history_size:
                    self._gradient_history[layer_name].pop(0)
        
        # Global statistics
        total_norm = total_norm ** 0.5  # L2 norm
        
        if all_gradients:
            all_grads_tensor = torch.cat(all_gradients)
            global_mean = all_grads_tensor.mean().item()
            global_var = all_grads_tensor.var().item()
            global_max = all_grads_tensor.max().item()
            global_min = all_grads_tensor.min().item()
        else:
            global_mean = global_var = global_max = global_min = 0.0
        
        vanishing_ratio = total_vanishing / total_params if total_params > 0 else 0.0
        exploding_ratio = total_exploding / total_params if total_params > 0 else 0.0
        
        self.log(logging.DEBUG, 
                f"Model gradient norm: {total_norm:.6f}, "
                f"vanishing: {vanishing_ratio:.2%}, exploding: {exploding_ratio:.2%}")
        
        return {
            "gradient_norm": total_norm,
            "gradient_variance": global_var,
            "gradient_mean": global_mean,
            "gradient_max": global_max,
            "gradient_min": global_min,
            "layer_gradients": layer_gradients,
            "vanishing_ratio": vanishing_ratio,
            "exploding_ratio": exploding_ratio
        }
    
    def get_gradient_flow_health(self) -> Dict[str, Any]:
        """
        Analyze gradient flow health based on history.
        
        Returns:
            Dictionary with health indicators
        """
        if not self.track_history or self._gradient_history is None:
            return {"healthy": True, "issues": []}
        
        issues = []
        layer_health = {}
        
        for layer_name, history in self._gradient_history.items():
            if len(history) < 5:  # Need some history
                continue
            
            # Get recent statistics
            recent_norms = [h['norm'] for h in history[-10:]]
            recent_vars = [h['variance'] for h in history[-10:]]
            
            # Check for issues
            avg_norm = sum(recent_norms) / len(recent_norms)
            norm_variance = torch.tensor(recent_norms).var().item()
            
            layer_issues = []
            
            # Vanishing gradients
            if avg_norm < 1e-6:
                layer_issues.append("vanishing_gradients")
            
            # Exploding gradients
            elif avg_norm > 1e2:
                layer_issues.append("exploding_gradients")
            
            # Unstable gradients
            if norm_variance > avg_norm * 0.5:
                layer_issues.append("unstable_gradients")
            
            # Dead gradients (no change)
            if norm_variance < 1e-10:
                layer_issues.append("dead_gradients")
            
            layer_health[layer_name] = {
                "avg_norm": avg_norm,
                "norm_variance": norm_variance,
                "issues": layer_issues
            }
            
            issues.extend([(layer_name, issue) for issue in layer_issues])
        
        healthy = len(issues) == 0
        
        return {
            "healthy": healthy,
            "issues": issues,
            "layer_health": layer_health
        }
    
    def get_gradient_trajectory(self, layer_name: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Get gradient trajectory over time.
        
        Args:
            layer_name: Specific layer name, or None for all layers
            
        Returns:
            Dictionary with trajectory data
        """
        if not self.track_history or self._gradient_history is None:
            return {}
        
        if layer_name:
            history = self._gradient_history.get(layer_name, [])
            return {
                "norms": [h['norm'] for h in history],
                "means": [h['mean'] for h in history],
                "variances": [h['variance'] for h in history]
            }
        else:
            # Return trajectories for all layers
            trajectories = {}
            for name, history in self._gradient_history.items():
                trajectories[name] = {
                    "norms": [h['norm'] for h in history],
                    "means": [h['mean'] for h in history],
                    "variances": [h['variance'] for h in history]
                }
            return trajectories