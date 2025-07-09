"""
Activity Analysis Module

This module provides comprehensive analysis of neuron activity patterns,
including dead neuron detection, saturation analysis, and activity distribution metrics.
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Dict, Any
import time
import logging

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from .base import BaseMetricAnalyzer, StatisticalUtilsMixin

logger = logging.getLogger(__name__)


class ActivityAnalyzer(BaseMetricAnalyzer, StatisticalUtilsMixin):
    """
    Complete analysis of neuron activity patterns.
    
    Analyzes activation distributions, detects dead and saturated neurons,
    and computes activity health metrics.
    """
    
    def __init__(self, threshold_config):
        super().__init__(threshold_config)
        self.activation_history = defaultdict(list)
        
    def compute_metrics(self, activations: torch.Tensor, layer_idx: int) -> Dict[str, Any]:
        """
        Compute ALL activity metrics for a layer.
        
        Args:
            activations: Layer activations [batch_size, num_neurons]
            layer_idx: Index of the layer being analyzed
            
        Returns:
            Dict containing comprehensive activity metrics
        """
        start_time = time.time()
        self._computation_stats['total_calls'] += 1
        
        # Validate inputs
        self._validate_tensor_input(activations, "activations", expected_dims=2)
        
        # Check cache
        cache_key = self._cache_key(activations, layer_idx)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Store activation history
        self.activation_history[layer_idx].append(activations.detach())
        
        # Basic activity detection
        active_mask = activations.abs() > self.config.activation_threshold
        
        # Neuron Activity Metrics
        active_neurons = active_mask.any(dim=0).sum().item()
        total_neurons = activations.shape[1]
        active_ratio = active_neurons / total_neurons
        dead_ratio = 1 - active_ratio
        
        # Activation Statistics
        max_activation = activations.abs().max().item()
        mean_activation = activations.abs().mean().item()
        std_activation = activations.std().item()
        
        # Per-neuron statistics
        neuron_activity_rates = active_mask.float().mean(dim=0)
        persistent_active = (neuron_activity_rates > self.config.persistence_ratio).sum().item()
        
        # Activation distribution analysis
        quantile_values = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=activations.device)
        activation_percentiles = torch.quantile(activations.abs().flatten(), quantile_values)
        
        # Saturation analysis
        saturation_threshold = 10.0
        saturated_neurons = (activations.abs() > saturation_threshold).any(dim=0).sum().item()
        saturation_ratio = saturated_neurons / total_neurons
        
        # Gradient explosion detection
        gradient_explosion_risk = max_activation > 10.0
        
        # Activity pattern analysis
        activity_entropy = self._compute_activity_entropy(neuron_activity_rates)
        activity_gini = self._compute_gini_coefficient(neuron_activity_rates)
        
        # Layer health score
        layer_health_score = self._compute_layer_health(active_ratio, max_activation, activity_entropy)
        
        result = {
            # Neuron Activity
            'active_neurons': active_neurons,
            'total_neurons': total_neurons,
            'active_ratio': active_ratio,
            'dead_ratio': dead_ratio,
            'persistent_active': persistent_active,
            
            # Activation Statistics
            'max_activation': max_activation,
            'mean_activation': mean_activation,
            'std_activation': std_activation,
            
            # Distribution Analysis
            'activation_p10': activation_percentiles[0].item(),
            'activation_p25': activation_percentiles[1].item(),
            'activation_median': activation_percentiles[2].item(),
            'activation_p75': activation_percentiles[3].item(),
            'activation_p90': activation_percentiles[4].item(),
            
            # Saturation Analysis
            'saturated_neurons': saturated_neurons,
            'saturation_ratio': saturation_ratio,
            'gradient_explosion_risk': gradient_explosion_risk,
            
            # Pattern Analysis
            'activity_entropy': activity_entropy,
            'activity_gini': activity_gini,
            
            # Health Indicators
            'layer_health_score': layer_health_score,
            
            # Meta information
            'layer_idx': layer_idx,
            'batch_size': activations.shape[0]
        }
        
        # Update timing stats
        computation_time = time.time() - start_time
        self._computation_stats['total_time'] += computation_time
        result['computation_time'] = computation_time
        
        # Cache result
        self._cache_result(cache_key, result)
        
        return result
    
    def _compute_activity_entropy(self, activity_rates: torch.Tensor) -> float:
        """Compute entropy of activity distribution."""
        # Normalize to probabilities
        probs = activity_rates / (activity_rates.sum() + 1e-10)
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0
        return -torch.sum(probs * torch.log2(probs)).item()
    
    def _compute_gini_coefficient(self, activity_rates: torch.Tensor) -> float:
        """Compute Gini coefficient of activity distribution."""
        if activity_rates.is_cuda and CUPY_AVAILABLE:
            activity_rates_cp = cp.asarray(activity_rates)
            sorted_rates = cp.sort(activity_rates_cp)
            n = len(sorted_rates)
            if n == 0:
                return 0.0
            cumsum = cp.cumsum(sorted_rates)
            if cumsum[-1] == 0:
                return 0.0
            return ((n + 1 - 2 * cp.sum(cumsum) / cumsum[-1]) / n).get()
        else:
            sorted_rates = torch.sort(activity_rates)[0]
            n = len(sorted_rates)
            if n == 0:
                return 0.0
            cumsum = torch.cumsum(sorted_rates, dim=0)
            if cumsum[-1] == 0:
                return 0.0
            return (n + 1 - 2 * torch.sum(cumsum) / cumsum[-1]) / n
    
    def _compute_layer_health(self, active_ratio: float, max_activation: float, 
                            activity_entropy: float) -> float:
        """Compute overall layer health score (0-1)."""
        # Penalize dead layers
        activity_score = min(1.0, active_ratio / 0.1)  # Target 10% active
        
        # Penalize saturation
        saturation_score = 1.0 if max_activation < 5.0 else max(0.0, 1.0 - (max_activation - 5.0) / 10.0)
        
        # Reward entropy (diversity)
        entropy_score = min(1.0, activity_entropy / 3.0)  # Target entropy of 3
        
        return (activity_score + saturation_score + entropy_score) / 3.0
    
    def get_layer_history(self, layer_idx: int) -> list:
        """Get activation history for a specific layer."""
        return self.activation_history.get(layer_idx, [])
    
    def clear_history(self, layer_idx: int = None) -> None:
        """Clear activation history for a layer or all layers."""
        if layer_idx is not None:
            self.activation_history[layer_idx].clear()
        else:
            self.activation_history.clear()
    
    def compute_temporal_metrics(self, layer_idx: int) -> Dict[str, Any]:
        """
        Compute temporal metrics based on activation history.
        
        Args:
            layer_idx: Layer to analyze
            
        Returns:
            Dict with temporal analysis metrics
        """
        history = self.activation_history.get(layer_idx, [])
        if len(history) < 2:
            return {'error': 'Insufficient history for temporal analysis'}
        
        # Compute activity trends
        activity_ratios = []
        mean_activations = []
        
        for acts in history:
            active_mask = acts.abs() > self.config.activation_threshold
            activity_ratios.append(active_mask.any(dim=0).float().mean().item())
            mean_activations.append(acts.abs().mean().item())
        
        activity_ratios = torch.tensor(activity_ratios)
        mean_activations = torch.tensor(mean_activations)
        
        # Compute trends
        time_steps = torch.arange(len(activity_ratios), dtype=torch.float)
        
        # Linear trend for activity ratio
        activity_trend = self._compute_linear_trend(time_steps, activity_ratios)
        activation_trend = self._compute_linear_trend(time_steps, mean_activations)
        
        # Stability metrics
        activity_stability = 1.0 / (activity_ratios.std() + 1e-10)
        activation_stability = 1.0 / (mean_activations.std() + 1e-10)
        
        return {
            'activity_trend': activity_trend,
            'activation_trend': activation_trend,
            'activity_stability': activity_stability,
            'activation_stability': activation_stability,
            'history_length': len(history),
            'current_activity_ratio': activity_ratios[-1].item(),
            'current_mean_activation': mean_activations[-1].item()
        }
    
    def _compute_linear_trend(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute linear trend (slope) between x and y."""
        if len(x) < 2:
            return 0.0
        
        x_mean = x.mean()
        y_mean = y.mean()
        
        numerator = torch.sum((x - x_mean) * (y - y_mean))
        denominator = torch.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        return (numerator / denominator).item()


# Export the analyzer
__all__ = ['ActivityAnalyzer']
