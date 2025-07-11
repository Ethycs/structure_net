"""
Lyapunov exponent metric component.

This component estimates local Lyapunov exponents to measure the sensitivity
of the neural network to small perturbations, indicating dynamical stability.
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


class LyapunovMetric(BaseMetric):
    """
    Estimates local Lyapunov exponents for neural networks.
    
    Lyapunov exponents measure the rate of separation of infinitesimally
    close trajectories, indicating chaos and instability in dynamical systems.
    """
    
    def __init__(self, n_directions: int = 10, epsilon: float = 1e-6, 
                 name: str = None):
        """
        Initialize Lyapunov metric.
        
        Args:
            n_directions: Number of random directions to sample
            epsilon: Perturbation magnitude
            name: Optional custom name
        """
        super().__init__(name or "LyapunovMetric")
        self.n_directions = n_directions
        self.epsilon = epsilon
        self._measurement_schema = {
            "max_lyapunov": float,
            "mean_lyapunov": float,
            "lyapunov_variance": float,
            "positive_lyapunov_ratio": float,
            "stability_indicator": float
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"model", "input_samples"},
            provided_outputs={
                "metrics.max_lyapunov",
                "metrics.mean_lyapunov",
                "metrics.lyapunov_variance",
                "metrics.positive_lyapunov_ratio",
                "metrics.stability_indicator"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=True,
                parallel_safe=False
            )
        )
    
    def _compute_metric(self, target: Union[ILayer, IModel], 
                       context: EvolutionContext) -> Dict[str, Any]:
        """
        Compute Lyapunov exponent metrics.
        
        Args:
            target: Model to analyze
            context: Must contain 'model' and 'input_samples'
            
        Returns:
            Dictionary containing Lyapunov measurements
        """
        # Get model and input samples
        model = context.get('model', target)
        input_samples = context.get('input_samples')
        
        if model is None or input_samples is None:
            raise ValueError("LyapunovMetric requires 'model' and 'input_samples' in context")
        
        if len(input_samples) == 0:
            return self._empty_metrics()
        
        # Compute Lyapunov exponents for each sample
        all_growth_rates = []
        
        model.eval()
        with torch.no_grad():
            for x in input_samples[:100]:  # Limit to 100 samples for efficiency
                growth_rates = self._compute_local_lyapunov(model, x)
                all_growth_rates.extend(growth_rates)
        
        if not all_growth_rates:
            return self._empty_metrics()
        
        # Compute statistics
        all_growth_rates = np.array(all_growth_rates)
        max_lyapunov = np.max(all_growth_rates)
        mean_lyapunov = np.mean(all_growth_rates)
        lyapunov_variance = np.var(all_growth_rates)
        
        # Ratio of positive Lyapunov exponents (indicates chaos)
        positive_lyapunov_ratio = np.mean(all_growth_rates > 0)
        
        # Stability indicator (lower is more stable)
        stability_indicator = 1.0 / (1.0 + np.exp(-max_lyapunov))
        
        self.log(logging.DEBUG, 
                f"Lyapunov: max={max_lyapunov:.3f}, mean={mean_lyapunov:.3f}, "
                f"positive_ratio={positive_lyapunov_ratio:.3f}")
        
        return {
            "max_lyapunov": max_lyapunov,
            "mean_lyapunov": mean_lyapunov,
            "lyapunov_variance": lyapunov_variance,
            "positive_lyapunov_ratio": positive_lyapunov_ratio,
            "stability_indicator": stability_indicator
        }
    
    def _compute_local_lyapunov(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        """Compute local Lyapunov exponents for a single input."""
        device = next(model.parameters()).device
        x = x.to(device)
        
        # Ensure x is the right shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Generate random perturbation directions
        input_dim = x.shape[1] if x.dim() > 1 else x.shape[0]
        perturbations = torch.randn(self.n_directions, input_dim, device=device)
        perturbations = perturbations / torch.norm(perturbations, dim=1, keepdim=True)
        
        growth_rates = []
        
        # Original output
        y = model(x)
        
        for delta in perturbations:
            # Perturbed input
            x_pert = x + self.epsilon * delta.unsqueeze(0)
            
            # Perturbed output
            y_pert = model(x_pert)
            
            # Compute growth rate
            output_diff = torch.norm(y_pert - y)
            growth = torch.log(output_diff / self.epsilon + 1e-10)
            growth_rates.append(growth.item())
        
        return growth_rates
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics when computation cannot proceed."""
        return {
            "max_lyapunov": 0.0,
            "mean_lyapunov": 0.0,
            "lyapunov_variance": 0.0,
            "positive_lyapunov_ratio": 0.0,
            "stability_indicator": 0.5
        }