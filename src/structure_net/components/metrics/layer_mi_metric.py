"""
Layer mutual information metric component.

This component measures mutual information between neural network layers,
which quantifies how much information one layer shares with another.
"""

from typing import Dict, Any, Union, Optional, Tuple, List
import torch
import torch.nn as nn
import numpy as np
import logging

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)
from .entropy_metric import EntropyMetric


class LayerMIMetric(BaseMetric):
    """
    Measures mutual information (MI) between neural network layers.
    
    MI quantifies the statistical dependence between layer activations,
    helping identify information bottlenecks and redundant layers.
    """
    
    def __init__(self, method: str = 'histogram', bins: int = 50,
                 k_neighbors: int = 3, mi_method: str = None, n_bins: int = None,
                 name: str = None):
        """
        Initialize layer MI metric.
        
        Args:
            method: MI estimation method ('histogram', 'knn')
            bins: Number of bins for histogram method
            k_neighbors: Number of neighbors for KNN method
            mi_method: Alias for method (for compatibility)
            n_bins: Alias for bins (for compatibility) 
            name: Optional custom name
        """
        super().__init__(name or "LayerMIMetric")
        # Handle aliases
        if mi_method is not None:
            method = mi_method
        if n_bins is not None:
            bins = n_bins
        
        self.method = method
        self.bins = bins
        self.k_neighbors = k_neighbors
        self._entropy_metric = EntropyMetric(bins=bins)
        self._measurement_schema = {
            "mutual_information": float,
            "normalized_mi": float,
            "information_ratio": float,
            "layer_pairs": dict  # Per-layer-pair MI values
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"target", "layer_activations"},
            provided_outputs={
                "metrics.mutual_information",
                "metrics.normalized_mi",
                "metrics.information_ratio",
                "metrics.layer_pairs"
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
        Compute mutual information metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'layer_activations' data
            
        Returns:
            Dictionary containing MI measurements
        """
        # Get activations from context
        layer_activations = context.get('layer_activations')
        if layer_activations is None:
            raise ValueError("LayerMIMetric requires 'layer_activations' in context")
        
        if isinstance(target, IModel):
            return self._compute_model_mi(target, layer_activations)
        elif isinstance(target, ILayer):
            # For single layer, compute MI with input/output
            return self._compute_layer_mi(target, layer_activations)
        elif target is None:
            # Direct computation with input/output activations
            if 'input' in layer_activations and 'output' in layer_activations:
                return self._compute_direct_mi(layer_activations['input'], 
                                             layer_activations['output'])
            else:
                raise ValueError("When target is None, layer_activations must contain 'input' and 'output' keys")
        else:
            raise ValueError(f"Target must be ILayer, IModel, or None, got {type(target)}")
    
    def _compute_direct_mi(self, input_acts: torch.Tensor, 
                          output_acts: torch.Tensor) -> Dict[str, Any]:
        """Compute MI directly between input and output activations."""
        # Compute MI
        mi_value = self._compute_mi(input_acts, output_acts)
        
        # Compute normalized MI
        h_input = self._entropy_metric._estimate_entropy(input_acts)
        h_output = self._entropy_metric._estimate_entropy(output_acts)
        min_entropy = min(h_input, h_output)
        
        normalized_mi = mi_value / min_entropy if min_entropy > 0 else 0.0
        information_ratio = mi_value / h_input if h_input > 0 else 0.0
        
        return {
            "mutual_information": mi_value,
            "normalized_mi": normalized_mi,
            "information_ratio": information_ratio,
            "computation_method": self.method
        }
    
    def _compute_layer_mi(self, layer: ILayer, 
                         layer_activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute MI for a single layer with its input/output."""
        # Find this layer's activations and its input
        layer_acts = layer_activations.get(layer.name)
        if layer_acts is None:
            raise ValueError(f"No activations found for layer {layer.name}")
        
        # Find input layer (simplified - assumes sequential structure)
        input_acts = None
        layer_names = list(layer_activations.keys())
        if layer.name in layer_names:
            idx = layer_names.index(layer.name)
            if idx > 0:
                input_acts = layer_activations[layer_names[idx - 1]]
        
        if input_acts is None:
            # No input layer found, return zero MI
            return {
                "mutual_information": 0.0,
                "normalized_mi": 0.0,
                "information_ratio": 0.0,
                "layer_pairs": {}
            }
        
        # Compute MI between input and output
        mi_value = self._compute_mi(input_acts, layer_acts)
        
        # Compute normalized MI
        h_input = self._entropy_metric._estimate_entropy(input_acts)
        h_output = self._entropy_metric._estimate_entropy(layer_acts)
        min_entropy = min(h_input, h_output)
        
        normalized_mi = mi_value / min_entropy if min_entropy > 0 else 0.0
        information_ratio = mi_value / h_input if h_input > 0 else 0.0
        
        return {
            "mutual_information": mi_value,
            "normalized_mi": normalized_mi,
            "information_ratio": information_ratio,
            "layer_pairs": {f"input->{layer.name}": mi_value}
        }
    
    def _compute_model_mi(self, model: IModel, 
                         layer_activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute MI between consecutive layers in the model."""
        layer_names = list(layer_activations.keys())
        total_mi = 0.0
        layer_pairs = {}
        num_pairs = 0
        
        # Compute MI between consecutive layers
        for i in range(len(layer_names) - 1):
            layer1_name = layer_names[i]
            layer2_name = layer_names[i + 1]
            
            acts1 = layer_activations[layer1_name]
            acts2 = layer_activations[layer2_name]
            
            if acts1 is None or acts2 is None:
                continue
            
            # Compute MI
            mi_value = self._compute_mi(acts1, acts2)
            pair_key = f"{layer1_name}->{layer2_name}"
            layer_pairs[pair_key] = mi_value
            
            total_mi += mi_value
            num_pairs += 1
        
        # Average MI across all pairs
        avg_mi = total_mi / num_pairs if num_pairs > 0 else 0.0
        
        # Compute average normalized MI
        avg_normalized_mi = 0.0
        for pair_key, mi_value in layer_pairs.items():
            layer1_name, layer2_name = pair_key.split('->')
            h1 = self._entropy_metric._estimate_entropy(layer_activations[layer1_name])
            h2 = self._entropy_metric._estimate_entropy(layer_activations[layer2_name])
            min_entropy = min(h1, h2)
            if min_entropy > 0:
                avg_normalized_mi += mi_value / min_entropy
        
        avg_normalized_mi = avg_normalized_mi / num_pairs if num_pairs > 0 else 0.0
        
        # Information ratio (how much info is preserved on average)
        avg_info_ratio = avg_mi / np.log2(self.bins) if self.bins > 1 else 0.0
        
        self.log(logging.DEBUG, 
                f"Model average MI: {avg_mi:.3f} (normalized: {avg_normalized_mi:.3f})")
        
        return {
            "mutual_information": avg_mi,
            "normalized_mi": avg_normalized_mi,
            "information_ratio": avg_info_ratio,
            "layer_pairs": layer_pairs
        }
    
    def _compute_mi(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute mutual information between two activation tensors.
        
        Args:
            x: First activation tensor
            y: Second activation tensor
            
        Returns:
            Mutual information value
        """
        if self.method == 'histogram' or self.method == 'binning':
            return self._compute_mi_histogram(x, y)
        elif self.method == 'knn':
            return self._compute_mi_knn(x, y)
        else:
            raise ValueError(f"Unknown MI method: {self.method}")
    
    def _compute_mi_histogram(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute MI using histogram method.
        
        MI(X,Y) = H(X) + H(Y) - H(X,Y)
        """
        # Flatten tensors
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if y.dim() > 2:
            y = y.view(y.size(0), -1)
        
        # Compute individual entropies
        h_x = self._entropy_metric._estimate_entropy(x)
        h_y = self._entropy_metric._estimate_entropy(y)
        
        # Compute joint entropy
        h_xy = self._entropy_metric.compute_joint_entropy(x, y)
        
        # MI = H(X) + H(Y) - H(X,Y)
        mi = h_x + h_y - h_xy
        
        # Ensure non-negative (numerical errors can cause small negative values)
        return max(0.0, mi)
    
    def _compute_mi_knn(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute MI using k-nearest neighbors method.
        
        This is more accurate for continuous distributions but slower.
        """
        # Flatten tensors
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if y.dim() > 2:
            y = y.view(y.size(0), -1)
        
        # Convert to numpy
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Simple KNN-based MI estimation
        # This is a simplified version - full implementation would use
        # sklearn.feature_selection.mutual_info_regression
        
        # For now, fall back to histogram method
        self.log(logging.WARNING, 
                "KNN MI estimation not fully implemented, using histogram method")
        return self._compute_mi_histogram(x, y)
    
    def find_information_bottlenecks(self, 
                                   layer_activations: Dict[str, torch.Tensor],
                                   threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        """
        Find layer pairs with low MI (information bottlenecks).
        
        Args:
            layer_activations: Dictionary of layer activations
            threshold: MI threshold below which is considered a bottleneck
            
        Returns:
            List of (layer1, layer2, mi_value) tuples for bottlenecks
        """
        bottlenecks = []
        layer_names = list(layer_activations.keys())
        
        for i in range(len(layer_names) - 1):
            layer1_name = layer_names[i]
            layer2_name = layer_names[i + 1]
            
            acts1 = layer_activations[layer1_name]
            acts2 = layer_activations[layer2_name]
            
            if acts1 is None or acts2 is None:
                continue
            
            # Compute normalized MI
            mi_value = self._compute_mi(acts1, acts2)
            h1 = self._entropy_metric._estimate_entropy(acts1)
            h2 = self._entropy_metric._estimate_entropy(acts2)
            min_entropy = min(h1, h2)
            
            if min_entropy > 0:
                normalized_mi = mi_value / min_entropy
                if normalized_mi < threshold:
                    bottlenecks.append((layer1_name, layer2_name, normalized_mi))
        
        return bottlenecks