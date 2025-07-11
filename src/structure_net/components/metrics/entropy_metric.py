"""
Entropy metric component for measuring activation entropy.

This component provides focused entropy measurements for layers
and models, which can be used by higher-level analyzers.
"""

from typing import Dict, Any, Union, Optional
import torch
import torch.nn as nn
import numpy as np
import logging

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


class EntropyMetric(BaseMetric):
    """
    Measures entropy of neural network activations.
    
    Entropy quantifies the uncertainty or information content in
    activation distributions. Low entropy indicates predictable
    patterns, while high entropy suggests more varied activations.
    """
    
    def __init__(self, base: float = 2.0, bins: int = 50, 
                 epsilon: float = 1e-10, name: str = None):
        """
        Initialize entropy metric.
        
        Args:
            base: Logarithm base for entropy calculation (2 for bits)
            bins: Number of bins for histogram estimation
            epsilon: Small value for numerical stability
            name: Optional custom name
        """
        super().__init__(name or "EntropyMetric")
        self.base = base
        self.bins = bins
        self.epsilon = epsilon
        self._measurement_schema = {
            "entropy": float,
            "normalized_entropy": float,
            "effective_bits": float,
            "layer_entropies": dict  # Per-layer breakdown
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"target", "activations"},
            provided_outputs={
                "metrics.entropy",
                "metrics.normalized_entropy",
                "metrics.effective_bits",
                "metrics.layer_entropies"
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
        Compute entropy metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'activations' data
            
        Returns:
            Dictionary containing entropy measurements
        """
        # Get activations from context
        activations = context.get('activations')
        if activations is None:
            raise ValueError("EntropyMetric requires 'activations' in context")
        
        if isinstance(target, IModel):
            return self._compute_model_entropy(target, activations)
        elif isinstance(target, ILayer):
            return self._compute_layer_entropy(target, activations)
        else:
            raise ValueError(f"Target must be ILayer or IModel, got {type(target)}")
    
    def _compute_layer_entropy(self, layer: ILayer, 
                              activations: torch.Tensor) -> Dict[str, Any]:
        """Compute entropy for a single layer."""
        # Flatten activations if needed
        if activations.dim() > 2:
            activations = activations.view(activations.size(0), -1)
        
        # Compute entropy across the batch dimension
        entropy = self._estimate_entropy(activations)
        
        # Compute maximum possible entropy
        max_entropy = np.log(self.bins) / np.log(self.base)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Effective bits is entropy in base 2
        if self.base != 2:
            effective_bits = entropy * np.log(self.base) / np.log(2)
        else:
            effective_bits = entropy
        
        return {
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "effective_bits": effective_bits,
            "layer_entropies": {layer.name: entropy}
        }
    
    def _compute_model_entropy(self, model: IModel, 
                              activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute entropy for entire model."""
        total_entropy = 0.0
        layer_entropies = {}
        num_layers = 0
        
        # Process each layer's activations
        for layer_name, layer_acts in activations.items():
            if layer_acts is None:
                continue
            
            # Flatten if needed
            if layer_acts.dim() > 2:
                layer_acts = layer_acts.view(layer_acts.size(0), -1)
            
            # Compute layer entropy
            layer_entropy = self._estimate_entropy(layer_acts)
            layer_entropies[layer_name] = layer_entropy
            
            total_entropy += layer_entropy
            num_layers += 1
        
        # Average entropy across layers
        avg_entropy = total_entropy / num_layers if num_layers > 0 else 0.0
        
        # Normalized entropy
        max_entropy = np.log(self.bins) / np.log(self.base)
        normalized_entropy = avg_entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Effective bits
        if self.base != 2:
            effective_bits = avg_entropy * np.log(self.base) / np.log(2)
        else:
            effective_bits = avg_entropy
        
        self.log(logging.DEBUG, 
                f"Model average entropy: {avg_entropy:.3f} ({effective_bits:.2f} bits)")
        
        return {
            "entropy": avg_entropy,
            "normalized_entropy": normalized_entropy,
            "effective_bits": effective_bits,
            "layer_entropies": layer_entropies
        }
    
    def _estimate_entropy(self, activations: torch.Tensor) -> float:
        """
        Estimate entropy using histogram method.
        
        Args:
            activations: Activation tensor (batch_size, num_features)
            
        Returns:
            Estimated entropy value
        """
        # Flatten to 1D for histogram
        flat_acts = activations.flatten().cpu().numpy()
        
        # Create histogram
        counts, _ = np.histogram(flat_acts, bins=self.bins)
        
        # Convert to probabilities
        probabilities = counts / (counts.sum() + self.epsilon)
        
        # Remove zero probabilities
        probabilities = probabilities[probabilities > self.epsilon]
        
        if len(probabilities) == 0:
            return 0.0
        
        # Compute entropy
        log_probs = np.log(probabilities + self.epsilon) / np.log(self.base)
        entropy = -np.sum(probabilities * log_probs)
        
        return float(entropy)
    
    def compute_joint_entropy(self, activations1: torch.Tensor, 
                             activations2: torch.Tensor) -> float:
        """
        Compute joint entropy between two activation sets.
        
        This is useful for mutual information calculations.
        
        Args:
            activations1: First activation tensor
            activations2: Second activation tensor
            
        Returns:
            Joint entropy value
        """
        # Ensure same batch size
        if activations1.size(0) != activations2.size(0):
            raise ValueError("Activation tensors must have same batch size")
        
        # Flatten
        if activations1.dim() > 2:
            activations1 = activations1.view(activations1.size(0), -1)
        if activations2.dim() > 2:
            activations2 = activations2.view(activations2.size(0), -1)
        
        # Create 2D histogram
        hist, _, _ = np.histogram2d(
            activations1.flatten().cpu().numpy(),
            activations2.flatten().cpu().numpy(),
            bins=self.bins
        )
        
        # Convert to probabilities
        probabilities = hist / (hist.sum() + self.epsilon)
        
        # Flatten and remove zeros
        probabilities = probabilities.flatten()
        probabilities = probabilities[probabilities > self.epsilon]
        
        if len(probabilities) == 0:
            return 0.0
        
        # Compute joint entropy
        log_probs = np.log(probabilities + self.epsilon) / np.log(self.base)
        joint_entropy = -np.sum(probabilities * log_probs)
        
        return float(joint_entropy)
    
    def compute_conditional_entropy(self, y_given_x: torch.Tensor, 
                                   x: torch.Tensor) -> float:
        """
        Compute conditional entropy H(Y|X).
        
        Args:
            y_given_x: Y activations (conditioned variable)
            x: X activations (conditioning variable)
            
        Returns:
            Conditional entropy value
        """
        # H(Y|X) = H(X,Y) - H(X)
        joint_entropy = self.compute_joint_entropy(x, y_given_x)
        x_entropy = self._estimate_entropy(x)
        
        conditional_entropy = joint_entropy - x_entropy
        
        # Ensure non-negative (numerical errors can cause small negative values)
        return max(0.0, conditional_entropy)