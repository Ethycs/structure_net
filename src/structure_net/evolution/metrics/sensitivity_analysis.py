"""
Sensitivity Analysis Module (SensLI)

This module provides comprehensive sensitivity analysis for neural networks,
including gradient-based sensitivity metrics and bottleneck detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional
import time
import logging

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from .base import BaseMetricAnalyzer, NetworkAnalyzerMixin

logger = logging.getLogger(__name__)


class SensitivityAnalyzer(BaseMetricAnalyzer, NetworkAnalyzerMixin):
    """
    Complete SensLI (Sensitivity) analysis with gradient-based metrics.
    
    Analyzes gradient flow, sensitivity to virtual parameters, and bottleneck detection.
    """
    
    def __init__(self, network: nn.Module, threshold_config):
        super().__init__(threshold_config)
        self.network = network
        
    def compute_metrics(self, layer_i: int, layer_j: int, data_loader, 
                       num_batches: int = 10) -> Dict[str, Any]:
        """
        Compute ALL SensLI metrics between two layers.
        
        Args:
            layer_i: Source layer index
            layer_j: Target layer index  
            data_loader: Data loader for gradient computation
            num_batches: Number of batches to analyze
            
        Returns:
            Dict containing comprehensive sensitivity metrics
        """
        start_time = time.time()
        self._computation_stats['total_calls'] += 1
        
        # Check cache
        cache_key = self._cache_key(layer_i, layer_j, num_batches)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        gradient_sensitivities = []
        bottleneck_scores = []
        
        self.network.train()  # Need gradients
        
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
                
            # Get layer activations and gradients
            acts_i, grads_i = self._get_layer_activations_and_gradients(data, target, layer_i)
            acts_j, grads_j = self._get_layer_activations_and_gradients(data, target, layer_j)
            
            if acts_i is None or acts_j is None:
                continue
                
            # Apply thresholds
            active_mask_i = acts_i.abs() > self.config.activation_threshold
            active_mask_j = acts_j.abs() > self.config.activation_threshold
            
            # Compute gradient sensitivity
            if active_mask_i.any() and active_mask_j.any():
                # Virtual parameter sensitivity
                virtual_sensitivity = self._compute_virtual_parameter_sensitivity(
                    acts_i, acts_j, grads_i, grads_j
                )
                gradient_sensitivities.append(virtual_sensitivity)
                
                # Bottleneck score computation
                bottleneck_score = self._compute_bottleneck_score(
                    acts_i, acts_j, grads_i, grads_j, active_mask_i, active_mask_j
                )
                bottleneck_scores.append(bottleneck_score)
        
        if not gradient_sensitivities:
            result = self._zero_sensli_metrics()
        else:
            # Aggregate metrics
            use_gpu = torch.cuda.is_available() and CUPY_AVAILABLE
            if use_gpu:
                gradient_sensitivities_gpu = cp.array(gradient_sensitivities)
                bottleneck_scores_gpu = cp.array(bottleneck_scores)
                avg_gradient_sensitivity = gradient_sensitivities_gpu.mean().get()
                max_gradient_sensitivity = gradient_sensitivities_gpu.max().get()
                std_gradient_sensitivity = gradient_sensitivities_gpu.std().get()
                avg_bottleneck_score = bottleneck_scores_gpu.mean().get()
                max_bottleneck_score = bottleneck_scores_gpu.max().get()
            else:
                avg_gradient_sensitivity = np.mean(gradient_sensitivities)
                max_gradient_sensitivity = np.max(gradient_sensitivities)
                std_gradient_sensitivity = np.std(gradient_sensitivities)
                avg_bottleneck_score = np.mean(bottleneck_scores)
                max_bottleneck_score = np.max(bottleneck_scores)
            
            # Determine suggested action
            suggested_action = self._determine_suggested_action(
                avg_gradient_sensitivity, avg_bottleneck_score
            )
            
            # Compute intervention priority
            intervention_priority = self._compute_intervention_priority(
                avg_gradient_sensitivity, avg_bottleneck_score
            )
            
            result = {
                # Gradient-Based Sensitivity
                'gradient_sensitivity': avg_gradient_sensitivity,
                'max_gradient_sensitivity': max_gradient_sensitivity,
                'std_gradient_sensitivity': std_gradient_sensitivity,
                
                # Bottleneck Scores
                'bottleneck_score': avg_bottleneck_score,
                'max_bottleneck_score': max_bottleneck_score,
                'critical_bottleneck': max_bottleneck_score == float('inf'),
                
                # Action Recommendations
                'suggested_action': suggested_action,
                'intervention_priority': intervention_priority,
                
                # Sensitivity Analysis
                'sensitivity_stability': 1.0 / (std_gradient_sensitivity + 1e-10),
                'gradient_flow_health': self._assess_gradient_flow_health(gradient_sensitivities),
                
                # Meta information
                'batches_analyzed': len(gradient_sensitivities),
                'layer_pair': (layer_i, layer_j)
            }
        
        # Update timing stats
        computation_time = time.time() - start_time
        self._computation_stats['total_time'] += computation_time
        result['computation_time'] = computation_time
        
        # Cache result
        self._cache_result(cache_key, result)
        
        return result
    
    def compute_metrics_from_precomputed_data(self, acts_i: torch.Tensor, acts_j: torch.Tensor,
                                            grads_i: torch.Tensor, grads_j: torch.Tensor,
                                            layer_i: int, layer_j: int) -> Dict[str, Any]:
        """
        OPTIMIZED: Compute SensLI metrics from precomputed activation and gradient data.
        This eliminates redundant forward/backward passes.
        
        Args:
            acts_i: Precomputed activations for layer i
            acts_j: Precomputed activations for layer j
            grads_i: Precomputed gradients for layer i
            grads_j: Precomputed gradients for layer j
            layer_i: Source layer index
            layer_j: Target layer index
            
        Returns:
            Dict containing sensitivity metrics
        """
        start_time = time.time()
        self._computation_stats['total_calls'] += 1
        
        # Validate inputs
        self._validate_tensor_input(acts_i, "acts_i", expected_dims=2)
        self._validate_tensor_input(acts_j, "acts_j", expected_dims=2)
        self._validate_tensor_input(grads_i, "grads_i", expected_dims=2)
        self._validate_tensor_input(grads_j, "grads_j", expected_dims=2)
        
        # Apply thresholds
        active_mask_i = acts_i.abs() > self.config.activation_threshold
        active_mask_j = acts_j.abs() > self.config.activation_threshold
        
        if not active_mask_i.any() or not active_mask_j.any():
            result = self._zero_sensli_metrics()
        else:
            # Compute gradient sensitivity
            virtual_sensitivity = self._compute_virtual_parameter_sensitivity(
                acts_i, acts_j, grads_i, grads_j
            )
            
            # Bottleneck score computation
            bottleneck_score = self._compute_bottleneck_score(
                acts_i, acts_j, grads_i, grads_j, active_mask_i, active_mask_j
            )
            
            # Determine suggested action
            suggested_action = self._determine_suggested_action(
                virtual_sensitivity, bottleneck_score
            )
            
            # Compute intervention priority
            intervention_priority = self._compute_intervention_priority(
                virtual_sensitivity, bottleneck_score
            )
            
            result = {
                # Gradient-Based Sensitivity
                'gradient_sensitivity': virtual_sensitivity,
                'max_gradient_sensitivity': virtual_sensitivity,
                'std_gradient_sensitivity': 0.0,  # Single batch, no variance
                
                # Bottleneck Scores
                'bottleneck_score': bottleneck_score,
                'max_bottleneck_score': bottleneck_score,
                'critical_bottleneck': bottleneck_score == float('inf'),
                
                # Action Recommendations
                'suggested_action': suggested_action,
                'intervention_priority': intervention_priority,
                
                # Sensitivity Analysis
                'sensitivity_stability': 1.0,  # Single measurement, assume stable
                'gradient_flow_health': self._assess_gradient_flow_health([virtual_sensitivity]),
                
                # Meta information
                'batches_analyzed': 1,  # Using aggregated data from multiple batches
                'layer_pair': (layer_i, layer_j),
                'optimization': 'precomputed_data'  # Flag to indicate optimization used
            }
        
        # Update timing stats
        computation_time = time.time() - start_time
        self._computation_stats['total_time'] += computation_time
        result['computation_time'] = computation_time
        
        return result
    
    def _get_layer_activations_and_gradients(self, data, target, layer_idx):
        """Get activations and gradients for a specific layer."""
        # Ensure data and target are on the same device as the network
        device = next(self.network.parameters()).device
        data = data.to(device)
        target = target.to(device)
        data.requires_grad_(True)
        
        x = data.view(data.size(0), -1)
        activations = []
        
        sparse_layers = self._get_sparse_layers(self.network)
        
        for i, layer in enumerate(sparse_layers):
            x = layer(x)
            if i == layer_idx:
                activations = x.clone()
                
                # Compute loss and gradients
                if i < len(sparse_layers) - 1:
                    # Continue forward
                    temp_x = x.clone()
                    for j in range(i + 1, len(sparse_layers)):
                        temp_x = sparse_layers[j](temp_x)
                        if j < len(sparse_layers) - 1:
                            temp_x = F.relu(temp_x)
                    
                    loss = F.cross_entropy(temp_x, target)
                    grads = torch.autograd.grad(loss, x, retain_graph=True)[0]
                else:
                    # Output layer
                    loss = F.cross_entropy(x, target)
                    grads = torch.autograd.grad(loss, data, retain_graph=True)[0]
                    grads = grads.view(grads.size(0), -1)
                
                return activations, grads
                
            if i < len(sparse_layers) - 1:
                x = F.relu(x)
        
        return None, None
    
    def _compute_virtual_parameter_sensitivity(self, acts_i, acts_j, grads_i, grads_j):
        """Compute sensitivity to virtual parameters."""
        # Sensitivity of adding a connection between layers
        # Use element-wise multiplication instead of matrix multiplication for compatibility
        acts_i_mean = acts_i.mean(dim=0)  # Shape: [features_i]
        grads_j_mean = grads_j.mean(dim=0)  # Shape: [features_j]
        
        # Compute sensitivity as the norm of the outer product (flattened)
        # This represents sensitivity to adding connections between all pairs
        if acts_i_mean.numel() > 0 and grads_j_mean.numel() > 0:
            # Use broadcasting to compute outer product efficiently
            outer_product = acts_i_mean.unsqueeze(1) * grads_j_mean.unsqueeze(0)
            sensitivity = torch.norm(outer_product)
        else:
            sensitivity = torch.tensor(0.0)
        
        return sensitivity.item()
    
    def _compute_bottleneck_score(self, acts_i, acts_j, grads_i, grads_j, 
                                active_mask_i, active_mask_j):
        """Compute bottleneck severity score."""
        # Information flow restriction
        active_ratio_i = active_mask_i.float().mean()
        active_ratio_j = active_mask_j.float().mean()
        
        # Gradient magnitude
        grad_norm_i = grads_i.norm(dim=1).mean()
        grad_norm_j = grads_j.norm(dim=1).mean()
        
        # Combined bottleneck score
        if active_ratio_i < 0.001 or active_ratio_j < 0.001:
            return float('inf')  # Critical bottleneck
        
        bottleneck = (1 - active_ratio_i) * (1 - active_ratio_j) * (grad_norm_i + grad_norm_j)
        return bottleneck.item()
    
    def _determine_suggested_action(self, gradient_sensitivity, bottleneck_score):
        """Determine suggested action based on metrics."""
        if bottleneck_score == float('inf'):
            return 'emergency_intervention'
        elif bottleneck_score > 2.0:
            return 'insert_multiple_layers'
        elif bottleneck_score > 1.0:
            return 'insert_single_layer'
        elif gradient_sensitivity > 1.0:
            return 'add_parallel_paths'
        else:
            return 'monitor'
    
    def _compute_intervention_priority(self, gradient_sensitivity, bottleneck_score):
        """Compute intervention priority (0-1)."""
        if bottleneck_score == float('inf'):
            return 1.0
        
        priority = min(1.0, (bottleneck_score + gradient_sensitivity) / 5.0)
        return priority
    
    def _assess_gradient_flow_health(self, gradient_sensitivities):
        """Assess overall gradient flow health."""
        if not gradient_sensitivities:
            return 0.0
        
        use_gpu = torch.cuda.is_available() and CUPY_AVAILABLE
        if use_gpu:
            gradient_sensitivities_gpu = cp.array(gradient_sensitivities)
            mean_sens = gradient_sensitivities_gpu.mean()
            std_sens = gradient_sensitivities_gpu.std()
        else:
            mean_sens = np.mean(gradient_sensitivities)
            std_sens = np.std(gradient_sensitivities)
        
        # Healthy gradient flow should be stable and moderate
        # Penalize very high or very low sensitivity
        magnitude_score = 1.0 - min(1.0, abs(mean_sens - 1.0))
        
        # Reward stability
        stability_score = 1.0 / (1.0 + std_sens)
        
        return (magnitude_score + stability_score) / 2.0
    
    def _zero_sensli_metrics(self):
        """Return zero metrics when computation fails."""
        return {
            'gradient_sensitivity': 0.0,
            'max_gradient_sensitivity': 0.0,
            'std_gradient_sensitivity': 0.0,
            'bottleneck_score': float('inf'),
            'max_bottleneck_score': float('inf'),
            'critical_bottleneck': True,
            'suggested_action': 'emergency_intervention',
            'intervention_priority': 1.0,
            'sensitivity_stability': 0.0,
            'gradient_flow_health': 0.0,
            'batches_analyzed': 0,
            'layer_pair': (-1, -1)
        }


# Export the analyzer
__all__ = ['SensitivityAnalyzer']
