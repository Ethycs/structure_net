"""
Activation metric component for analyzing neural activations.

This component provides detailed statistics about activation patterns
to understand network behavior and identify potential issues.
"""

from typing import Dict, Any, Union, Optional, List, Tuple
import torch
import torch.nn as nn
import numpy as np
import logging

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


class ActivationMetric(BaseMetric):
    """
    Measures activation statistics and patterns in neural networks.
    
    Provides comprehensive analysis of activation distributions,
    patterns, and anomalies across layers and time.
    """
    
    def __init__(self, track_distribution: bool = True,
                 percentiles: List[float] = None,
                 pattern_detection: bool = True,
                 name: str = None):
        """
        Initialize activation metric.
        
        Args:
            track_distribution: Whether to track full distribution stats
            percentiles: List of percentiles to compute (e.g., [25, 50, 75])
            pattern_detection: Whether to detect activation patterns
            name: Optional custom name
        """
        super().__init__(name or "ActivationMetric")
        self.track_distribution = track_distribution
        self.percentiles = percentiles or [10, 25, 50, 75, 90]
        self.pattern_detection = pattern_detection
        self._measurement_schema = {
            "mean": float,
            "std": float,
            "min": float,
            "max": float,
            "sparsity": float,
            "percentiles": dict,
            "skewness": float,
            "kurtosis": float,
            "layer_stats": dict,
            "activation_patterns": dict
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
                "metrics.activation_mean",
                "metrics.activation_std",
                "metrics.activation_sparsity",
                "metrics.activation_percentiles",
                "metrics.activation_distribution",
                "metrics.activation_patterns"
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
        Compute activation metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'activations' data
            
        Returns:
            Dictionary containing activation measurements
        """
        # Get activations from context
        activations = context.get('activations')
        if activations is None:
            raise ValueError("ActivationMetric requires 'activations' in context")
        
        if isinstance(target, IModel):
            return self._compute_model_activations(target, activations)
        elif isinstance(target, ILayer):
            return self._compute_layer_activations(target, activations)
        else:
            raise ValueError(f"Target must be ILayer or IModel, got {type(target)}")
    
    def _compute_layer_activations(self, layer: ILayer, 
                                  activations: torch.Tensor) -> Dict[str, Any]:
        """Compute activation statistics for a single layer."""
        # Flatten activations if needed
        if activations.dim() > 2:
            activations = activations.view(activations.size(0), -1)
        
        # Basic statistics
        mean = activations.mean().item()
        std = activations.std().item()
        min_val = activations.min().item()
        max_val = activations.max().item()
        
        # Sparsity (fraction of near-zero activations)
        sparsity = (activations.abs() < 1e-6).float().mean().item()
        
        # Percentiles
        percentile_values = {}
        if self.percentiles:
            for p in self.percentiles:
                percentile_values[f"p{int(p)}"] = torch.quantile(
                    activations.flatten(), p / 100.0
                ).item()
        
        # Distribution statistics
        if self.track_distribution:
            flat_acts = activations.flatten()
            
            # Skewness (third moment)
            if std > 0:
                centered = (flat_acts - mean) / std
                skewness = (centered ** 3).mean().item()
                kurtosis = (centered ** 4).mean().item() - 3.0  # Excess kurtosis
            else:
                skewness = 0.0
                kurtosis = 0.0
        else:
            skewness = 0.0
            kurtosis = 0.0
        
        # Pattern detection
        patterns = {}
        if self.pattern_detection:
            patterns = self._detect_patterns(activations, layer.name)
        
        return {
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "sparsity": sparsity,
            "percentiles": percentile_values,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "layer_stats": {layer.name: {
                "mean": mean,
                "std": std,
                "sparsity": sparsity
            }},
            "activation_patterns": patterns
        }
    
    def _compute_model_activations(self, model: IModel, 
                                  activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute activation statistics for entire model."""
        all_stats = []
        layer_stats = {}
        all_patterns = {}
        
        # Process each layer
        for layer_name, layer_acts in activations.items():
            if layer_acts is None:
                continue
            
            # Flatten if needed
            if layer_acts.dim() > 2:
                layer_acts = layer_acts.view(layer_acts.size(0), -1)
            
            # Compute layer statistics
            mean = layer_acts.mean().item()
            std = layer_acts.std().item()
            sparsity = (layer_acts.abs() < 1e-6).float().mean().item()
            
            layer_stats[layer_name] = {
                "mean": mean,
                "std": std,
                "sparsity": sparsity,
                "min": layer_acts.min().item(),
                "max": layer_acts.max().item()
            }
            
            all_stats.append(layer_acts.flatten())
            
            # Pattern detection for this layer
            if self.pattern_detection:
                patterns = self._detect_patterns(layer_acts, layer_name)
                all_patterns.update(patterns)
        
        # Global statistics
        if all_stats:
            all_acts = torch.cat(all_stats)
            global_mean = all_acts.mean().item()
            global_std = all_acts.std().item()
            global_min = all_acts.min().item()
            global_max = all_acts.max().item()
            global_sparsity = (all_acts.abs() < 1e-6).float().mean().item()
            
            # Percentiles
            percentile_values = {}
            for p in self.percentiles:
                percentile_values[f"p{int(p)}"] = torch.quantile(all_acts, p / 100.0).item()
            
            # Distribution statistics
            if self.track_distribution and global_std > 0:
                centered = (all_acts - global_mean) / global_std
                skewness = (centered ** 3).mean().item()
                kurtosis = (centered ** 4).mean().item() - 3.0
            else:
                skewness = 0.0
                kurtosis = 0.0
        else:
            global_mean = global_std = global_min = global_max = global_sparsity = 0.0
            percentile_values = {}
            skewness = kurtosis = 0.0
        
        self.log(logging.DEBUG, 
                f"Model activation stats: mean={global_mean:.3f}, std={global_std:.3f}, "
                f"sparsity={global_sparsity:.2%}")
        
        return {
            "mean": global_mean,
            "std": global_std,
            "min": global_min,
            "max": global_max,
            "sparsity": global_sparsity,
            "percentiles": percentile_values,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "layer_stats": layer_stats,
            "activation_patterns": all_patterns
        }
    
    def _detect_patterns(self, activations: torch.Tensor, 
                        layer_name: str) -> Dict[str, Any]:
        """
        Detect activation patterns in the layer.
        
        Args:
            activations: Activation tensor
            layer_name: Name of the layer
            
        Returns:
            Dictionary of detected patterns
        """
        patterns = {}
        
        # Ensure 2D tensor (batch_size, features)
        if activations.dim() > 2:
            activations = activations.view(activations.size(0), -1)
        
        # 1. Saturation patterns
        saturation_low = (activations < 0.1).float().mean(dim=0)
        saturation_high = (activations > 0.9).float().mean(dim=0)
        
        saturated_low_neurons = (saturation_low > 0.9).sum().item()
        saturated_high_neurons = (saturation_high > 0.9).sum().item()
        
        patterns[f"{layer_name}_saturation"] = {
            "low_saturated": saturated_low_neurons,
            "high_saturated": saturated_high_neurons,
            "total_neurons": activations.size(1)
        }
        
        # 2. Correlation patterns
        if activations.size(0) > 1:  # Need multiple samples
            # Compute correlation matrix
            act_centered = activations - activations.mean(dim=0, keepdim=True)
            act_norm = act_centered / (act_centered.std(dim=0, keepdim=True) + 1e-8)
            
            # Correlation matrix (features x features)
            corr_matrix = torch.mm(act_norm.t(), act_norm) / activations.size(0)
            
            # Find highly correlated neuron pairs
            high_corr_threshold = 0.9
            corr_upper = torch.triu(corr_matrix, diagonal=1)
            high_corr_pairs = (corr_upper.abs() > high_corr_threshold).sum().item()
            
            patterns[f"{layer_name}_correlation"] = {
                "high_correlation_pairs": high_corr_pairs,
                "avg_correlation": corr_upper.abs().mean().item()
            }
        
        # 3. Clustering patterns
        if activations.size(1) > 10:  # Need enough neurons
            # Simple clustering detection using activation similarity
            # Compute pairwise distances between neurons
            neuron_acts = activations.t()  # (features, batch_size)
            
            # Normalize
            neuron_acts_norm = neuron_acts / (neuron_acts.norm(dim=1, keepdim=True) + 1e-8)
            
            # Cosine similarity
            similarity = torch.mm(neuron_acts_norm, neuron_acts_norm.t())
            
            # Count neurons with similar activation patterns
            similar_threshold = 0.8
            similar_groups = []
            processed = set()
            
            for i in range(similarity.size(0)):
                if i in processed:
                    continue
                    
                similar_neurons = (similarity[i] > similar_threshold).nonzero(as_tuple=True)[0]
                if len(similar_neurons) > 1:
                    group = similar_neurons.tolist()
                    similar_groups.append(group)
                    processed.update(group)
            
            patterns[f"{layer_name}_clustering"] = {
                "num_clusters": len(similar_groups),
                "largest_cluster": max(len(g) for g in similar_groups) if similar_groups else 0
            }
        
        # 4. Oscillation patterns
        if hasattr(self, '_activation_history'):
            # This would track temporal patterns across multiple forward passes
            # For now, we skip this as it requires temporal data
            pass
        
        return patterns
    
    def compute_activation_distance(self, acts1: torch.Tensor, 
                                   acts2: torch.Tensor,
                                   method: str = 'l2') -> float:
        """
        Compute distance between two activation tensors.
        
        Args:
            acts1: First activation tensor
            acts2: Second activation tensor
            method: Distance method ('l2', 'cosine', 'kl')
            
        Returns:
            Distance value
        """
        # Flatten tensors
        acts1_flat = acts1.flatten()
        acts2_flat = acts2.flatten()
        
        if method == 'l2':
            return torch.norm(acts1_flat - acts2_flat).item()
        
        elif method == 'cosine':
            # Cosine distance = 1 - cosine similarity
            cos_sim = torch.cosine_similarity(acts1_flat, acts2_flat, dim=0)
            return (1 - cos_sim).item()
        
        elif method == 'kl':
            # KL divergence (treat as probability distributions)
            # Add small epsilon for numerical stability
            eps = 1e-8
            p = torch.softmax(acts1_flat, dim=0) + eps
            q = torch.softmax(acts2_flat, dim=0) + eps
            
            kl_div = torch.sum(p * torch.log(p / q))
            return kl_div.item()
        
        else:
            raise ValueError(f"Unknown distance method: {method}")
    
    def identify_critical_neurons(self, activations: torch.Tensor,
                                 threshold: float = 0.9) -> List[int]:
        """
        Identify neurons that are critical for information flow.
        
        Critical neurons have high variance and are rarely saturated.
        
        Args:
            activations: Activation tensor (batch_size, num_neurons)
            threshold: Percentile threshold for criticality
            
        Returns:
            List of critical neuron indices
        """
        if activations.dim() > 2:
            activations = activations.view(activations.size(0), -1)
        
        # Compute neuron statistics
        neuron_var = activations.var(dim=0)
        neuron_mean = activations.mean(dim=0)
        
        # Non-saturated neurons (not always near 0 or 1)
        non_saturated = (neuron_mean > 0.1) & (neuron_mean < 0.9)
        
        # High variance neurons
        var_threshold = torch.quantile(neuron_var, threshold)
        high_variance = neuron_var > var_threshold
        
        # Critical neurons have both properties
        critical = non_saturated & high_variance
        critical_indices = critical.nonzero(as_tuple=True)[0].tolist()
        
        return critical_indices