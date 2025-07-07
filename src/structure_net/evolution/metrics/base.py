"""
Base classes and interfaces for the metrics system.

This module provides the foundational classes that all metric analyzers inherit from,
ensuring consistent interfaces and shared functionality.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """Configuration for threshold-based analysis."""
    activation_threshold: float = 0.01
    weight_threshold: float = 0.001
    persistence_ratio: float = 0.1
    adaptive: bool = True
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.activation_threshold <= 0:
            raise ValueError("activation_threshold must be positive")
        if self.weight_threshold <= 0:
            raise ValueError("weight_threshold must be positive")
        if not 0 <= self.persistence_ratio <= 1:
            raise ValueError("persistence_ratio must be between 0 and 1")


@dataclass
class MetricsConfig:
    """Configuration for which metrics to compute."""
    compute_mi: bool = True
    compute_activity: bool = True
    compute_sensli: bool = True
    compute_graph: bool = True
    
    # Performance settings
    max_batches: int = 10
    sample_size: int = 1000
    enable_caching: bool = True
    
    # Advanced settings
    mi_method: str = 'auto'  # 'auto', 'exact', 'knn', 'advanced'
    graph_sampling: bool = True
    sensli_optimization: bool = True


@dataclass
class MetricResult:
    """Standardized container for metric results."""
    name: str
    value: float
    metadata: Dict[str, Any]
    confidence: float = 1.0
    computation_time: float = 0.0
    
    def __post_init__(self):
        """Validate result values."""
        if not isinstance(self.value, (int, float)):
            raise ValueError("value must be numeric")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


class BaseMetricAnalyzer(ABC):
    """
    Abstract base class for all metric analyzers.
    
    Provides common functionality and enforces consistent interfaces
    across all metric computation components.
    """
    
    def __init__(self, threshold_config: ThresholdConfig):
        self.config = threshold_config
        self.cache = {} if hasattr(threshold_config, 'enable_caching') else None
        self._computation_stats = {
            'total_calls': 0,
            'total_time': 0.0,
            'cache_hits': 0
        }
    
    @abstractmethod
    def compute_metrics(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Compute metrics for the given inputs.
        
        Returns:
            Dict containing computed metrics
        """
        pass
    
    def _cache_key(self, *args, **kwargs) -> str:
        """Generate cache key for inputs."""
        # Simple hash-based cache key
        key_parts = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                key_parts.append(f"tensor_{arg.shape}_{arg.sum().item():.6f}")
            else:
                key_parts.append(str(arg))
        for k, v in kwargs.items():
            key_parts.append(f"{k}={v}")
        return "_".join(key_parts)
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached result if available."""
        if self.cache is None:
            return None
        
        if cache_key in self.cache:
            self._computation_stats['cache_hits'] += 1
            return self.cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache computation result."""
        if self.cache is not None:
            self.cache[cache_key] = result
    
    def get_computation_stats(self) -> Dict[str, Any]:
        """Get statistics about computation performance."""
        stats = self._computation_stats.copy()
        if stats['total_calls'] > 0:
            stats['avg_time_per_call'] = stats['total_time'] / stats['total_calls']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_calls']
        else:
            stats['avg_time_per_call'] = 0.0
            stats['cache_hit_rate'] = 0.0
        return stats
    
    def clear_cache(self) -> None:
        """Clear computation cache."""
        if self.cache is not None:
            self.cache.clear()
    
    def _validate_tensor_input(self, tensor: torch.Tensor, name: str, 
                              expected_dims: int = None) -> None:
        """Validate tensor input."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        
        if expected_dims is not None and tensor.dim() != expected_dims:
            raise ValueError(f"{name} must be {expected_dims}D, got {tensor.dim()}D")
        
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} contains NaN values")
        
        if torch.isinf(tensor).any():
            raise ValueError(f"{name} contains infinite values")


class NetworkAnalyzerMixin:
    """
    Mixin providing common network analysis utilities.
    """
    
    def _get_sparse_layers(self, network: nn.Module) -> List[nn.Module]:
        """Extract sparse layers from network."""
        from ...core.layers import StandardSparseLayer
        return [layer for layer in network if isinstance(layer, StandardSparseLayer)]
    
    def _get_layer_weights(self, layer) -> torch.Tensor:
        """Extract weights from a layer, handling sparse layers."""
        if hasattr(layer, 'mask'):
            return layer.linear.weight * layer.mask
        elif hasattr(layer, 'weight'):
            return layer.weight
        else:
            raise ValueError(f"Cannot extract weights from layer type {type(layer)}")
    
    def _apply_activation_threshold(self, activations: torch.Tensor, 
                                   threshold: float) -> torch.Tensor:
        """Apply threshold to get active neurons."""
        return activations.abs() > threshold
    
    def _compute_sparsity(self, tensor: torch.Tensor, threshold: float = 1e-8) -> float:
        """Compute sparsity ratio of a tensor."""
        total_elements = tensor.numel()
        if total_elements == 0:
            return 0.0
        
        active_elements = (tensor.abs() > threshold).sum().item()
        return 1.0 - (active_elements / total_elements)


class StatisticalUtilsMixin:
    """
    Mixin providing common statistical utilities.
    """
    
    def _safe_entropy(self, probs: torch.Tensor, base: float = 2.0) -> float:
        """Compute entropy with numerical stability."""
        probs = probs[probs > 0]  # Remove zeros
        if len(probs) == 0:
            return 0.0
        
        log_probs = torch.log(probs) / torch.log(torch.tensor(base))
        return -torch.sum(probs * log_probs).item()
    
    def _safe_correlation(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute correlation with numerical stability."""
        if x.numel() != y.numel():
            raise ValueError("Tensors must have same number of elements")
        
        x_flat = x.flatten()
        y_flat = y.flatten()
        
        if x_flat.std() == 0 or y_flat.std() == 0:
            return 0.0
        
        return torch.corrcoef(torch.stack([x_flat, y_flat]))[0, 1].item()
    
    def _compute_percentiles(self, tensor: torch.Tensor, 
                           percentiles: List[float]) -> Dict[str, float]:
        """Compute multiple percentiles efficiently."""
        quantiles = torch.tensor(percentiles, device=tensor.device) / 100.0
        values = torch.quantile(tensor.flatten(), quantiles)
        
        return {f'p{int(p)}': v.item() for p, v in zip(percentiles, values)}
    
    def _robust_mean_std(self, tensor: torch.Tensor) -> Tuple[float, float]:
        """Compute robust mean and std using median absolute deviation."""
        median = torch.median(tensor)
        mad = torch.median(torch.abs(tensor - median))
        
        # Convert MAD to std estimate (assumes normal distribution)
        robust_std = 1.4826 * mad
        
        return median.item(), robust_std.item()


# Export all base classes
__all__ = [
    'ThresholdConfig',
    'MetricsConfig', 
    'MetricResult',
    'BaseMetricAnalyzer',
    'NetworkAnalyzerMixin',
    'StatisticalUtilsMixin'
]
