"""
Redundancy metric component.

This component measures redundancy between neural network layers,
which indicates how much information is repeated or shared.
"""

from typing import Dict, Any, Union, Optional, List, Tuple
import torch
import logging

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)
from .entropy_metric import EntropyMetric


class RedundancyMetric(BaseMetric):
    """
    Measures redundancy in neural network activations.
    
    Redundancy indicates how much information is shared or repeated
    between layers, which can indicate inefficient information encoding.
    """
    
    def __init__(self, threshold: float = 0.8, name: str = None):
        """
        Initialize redundancy metric.
        
        Args:
            threshold: Correlation threshold for considering features redundant
            name: Optional custom name
        """
        super().__init__(name or "RedundancyMetric")
        self.threshold = threshold
        self._entropy_metric = EntropyMetric()
        self._measurement_schema = {
            "redundancy": float,
            "independence_ratio": float,
            "layer_redundancy": dict  # Per-layer-pair redundancy
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
                "metrics.redundancy",
                "metrics.independence_ratio",
                "metrics.layer_redundancy"
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
        Compute redundancy metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'layer_activations' data
            
        Returns:
            Dictionary containing redundancy measurements
        """
        # Get activations from context
        layer_activations = context.get('layer_activations')
        if layer_activations is None:
            # Try simple activations for single tensor analysis
            activations = context.get('activations')
            if activations is not None:
                # Convert single activations to expected format
                layer_activations = {'default': activations}
            else:
                raise ValueError("RedundancyMetric requires 'layer_activations' or 'activations' in context")
        
        if isinstance(target, IModel):
            return self._compute_model_redundancy(target, layer_activations)
        elif isinstance(target, ILayer):
            return self._compute_layer_redundancy(target, layer_activations)
        elif target is None and len(layer_activations) == 1:
            # Simple activation analysis
            activations = list(layer_activations.values())[0]
            return self._compute_activation_redundancy(activations)
        else:
            raise ValueError(f"Target must be ILayer or IModel, got {type(target)}")
    
    def _compute_activation_redundancy(self, activations: torch.Tensor) -> Dict[str, Any]:
        """Compute redundancy in a single activation tensor."""
        if activations.dim() > 2:
            # Flatten to (batch, features)
            activations = activations.view(activations.size(0), -1)
        
        # Compute pairwise correlations between features
        correlations = self._compute_correlations(activations)
        
        # Find redundant pairs
        redundant_pairs = self._find_redundant_pairs(correlations)
        
        # Compute metrics
        total_features = activations.shape[1]
        total_pairs = total_features * (total_features - 1) // 2
        redundancy_ratio = len(redundant_pairs) / total_pairs if total_pairs > 0 else 0
        
        # Mean and max redundancy
        if correlations.shape[0] > 1:
            # Remove diagonal
            mask = ~torch.eye(correlations.shape[0], dtype=bool)
            off_diagonal = correlations[mask]
            mean_redundancy = off_diagonal.abs().mean().item()
            max_redundancy = off_diagonal.abs().max().item()
        else:
            mean_redundancy = 0.0
            max_redundancy = 0.0
        
        return {
            "total_redundancy": len(redundant_pairs),
            "mean_redundancy": mean_redundancy,
            "max_redundancy": max_redundancy,
            "redundancy_ratio": redundancy_ratio
        }
    
    def _compute_layer_redundancy(self, layer: ILayer, 
                                 layer_activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute redundancy for a single layer with its input."""
        # Find this layer and its input
        layer_acts = layer_activations.get(layer.name)
        if layer_acts is None:
            raise ValueError(f"No activations found for layer {layer.name}")
        
        # Find input layer
        layer_names = list(layer_activations.keys())
        input_acts = None
        if layer.name in layer_names:
            idx = layer_names.index(layer.name)
            if idx > 0:
                input_acts = layer_activations[layer_names[idx - 1]]
        
        if input_acts is None:
            # No input layer found
            return {
                "redundancy": 0.0,
                "independence_ratio": 1.0,
                "layer_redundancy": {}
            }
        
        # Compute entropies
        h_input = self._entropy_metric._estimate_entropy(input_acts)
        h_output = self._entropy_metric._estimate_entropy(layer_acts)
        h_joint = self._entropy_metric.compute_joint_entropy(input_acts, layer_acts)
        
        # Redundancy = H(X) + H(Y) - H(X,Y)
        redundancy = h_input + h_output - h_joint
        
        # Independence ratio = MI / redundancy
        # MI = H(X) + H(Y) - H(X,Y) = redundancy
        # So independence_ratio indicates how much of the total information is shared
        total_info = h_input + h_output
        independence_ratio = redundancy / total_info if total_info > 0 else 0.0
        
        return {
            "redundancy": redundancy,
            "independence_ratio": independence_ratio,
            "layer_redundancy": {
                f"input->{layer.name}": redundancy
            }
        }
    
    def _compute_model_redundancy(self, model: IModel, 
                                 layer_activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute redundancy for entire model."""
        layer_names = list(layer_activations.keys())
        total_redundancy = 0.0
        layer_redundancy = {}
        num_pairs = 0
        
        # Compute redundancy between consecutive layers
        for i in range(len(layer_names) - 1):
            layer1_name = layer_names[i]
            layer2_name = layer_names[i + 1]
            
            acts1 = layer_activations[layer1_name]
            acts2 = layer_activations[layer2_name]
            
            if acts1 is None or acts2 is None:
                continue
            
            # Compute entropies
            h1 = self._entropy_metric._estimate_entropy(acts1)
            h2 = self._entropy_metric._estimate_entropy(acts2)
            h_joint = self._entropy_metric.compute_joint_entropy(acts1, acts2)
            
            # Redundancy
            redundancy = h1 + h2 - h_joint
            
            pair_key = f"{layer1_name}->{layer2_name}"
            layer_redundancy[pair_key] = redundancy
            
            total_redundancy += redundancy
            num_pairs += 1
        
        # Average redundancy
        avg_redundancy = total_redundancy / num_pairs if num_pairs > 0 else 0.0
        
        # Average independence ratio
        avg_independence_ratio = 0.0
        if num_pairs > 0:
            for i in range(len(layer_names) - 1):
                layer1_name = layer_names[i]
                layer2_name = layer_names[i + 1]
                
                acts1 = layer_activations.get(layer1_name)
                acts2 = layer_activations.get(layer2_name)
                
                if acts1 is None or acts2 is None:
                    continue
                
                h1 = self._entropy_metric._estimate_entropy(acts1)
                h2 = self._entropy_metric._estimate_entropy(acts2)
                total_info = h1 + h2
                
                pair_key = f"{layer1_name}->{layer2_name}"
                if pair_key in layer_redundancy and total_info > 0:
                    avg_independence_ratio += layer_redundancy[pair_key] / total_info
            
            avg_independence_ratio /= num_pairs
        
        self.log(logging.DEBUG, 
                f"Model redundancy: avg={avg_redundancy:.3f}, "
                f"independence ratio={avg_independence_ratio:.3f}")
        
        return {
            "redundancy": avg_redundancy,
            "independence_ratio": avg_independence_ratio,
            "layer_redundancy": layer_redundancy
        }
    
    def _compute_correlations(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute pairwise correlations between features."""
        # Standardize features
        mean = activations.mean(dim=0, keepdim=True)
        std = activations.std(dim=0, keepdim=True) + 1e-8
        normalized = (activations - mean) / std
        
        # Compute correlation matrix
        batch_size = activations.shape[0]
        correlations = torch.mm(normalized.T, normalized) / (batch_size - 1)
        
        return correlations
    
    def _find_redundant_pairs(self, correlations: torch.Tensor) -> List[Tuple[int, int]]:
        """Find pairs of features with high correlation."""
        redundant_pairs = []
        n_features = correlations.shape[0]
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if abs(correlations[i, j]) > self.threshold:
                    redundant_pairs.append((i, j))
        
        return redundant_pairs