"""
Mutual Information Analysis Module

This module provides comprehensive mutual information analysis for neural networks,
including exact MI computation, entropy estimation, and information flow analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Any
import time
import logging

try:
    import cupy as cp
    from cuml.feature_selection import mutual_info_regression
    from cuml.neighbors import NearestNeighbors
    CUPY_AVAILABLE = True
except ImportError:
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.neighbors import NearestNeighbors
    CUPY_AVAILABLE = False

from .base import BaseMetricAnalyzer, StatisticalUtilsMixin

logger = logging.getLogger(__name__)


class MutualInformationAnalyzer(BaseMetricAnalyzer, StatisticalUtilsMixin):
    """
    Complete mutual information analysis with multiple estimation methods.
    
    Provides exact MI computation for small dimensions, k-NN estimation for
    medium dimensions, and advanced estimators for high dimensions.
    """
    
    def __init__(self, threshold_config):
        super().__init__(threshold_config)
        self.threshold = threshold_config.activation_threshold
        
    def compute_metrics(self, X: torch.Tensor, Y: torch.Tensor) -> Dict[str, Any]:
        """
        Compute ALL mutual information metrics.
        
        Args:
            X: Input activations [batch_size, features_X]
            Y: Output activations [batch_size, features_Y]
            
        Returns:
            Dict containing comprehensive MI metrics
        """
        start_time = time.time()
        self._computation_stats['total_calls'] += 1
        
        # Validate inputs
        self._validate_tensor_input(X, "X", expected_dims=2)
        self._validate_tensor_input(Y, "Y", expected_dims=2)
        
        # Check cache
        cache_key = self._cache_key(X, Y)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Apply threshold to get active neurons only
        X_active_mask = (X.abs() > self.threshold).any(dim=0)
        Y_active_mask = (Y.abs() > self.threshold).any(dim=0)
        
        X_active = X[:, X_active_mask]
        Y_active = Y[:, Y_active_mask]
        
        n_samples = X.shape[0]
        n_active_X = X_active.shape[1]
        n_active_Y = Y_active.shape[1]
        
        if n_active_X == 0 or n_active_Y == 0:
            result = self._zero_metrics()
        else:
            # Choose method based on dimensionality
            if n_active_X <= 10 and n_active_Y <= 10 and n_samples > 50:
                result = self._exact_discrete_mi(X_active, Y_active)
                method = 'exact_discrete'
            elif n_active_X + n_active_Y <= 50:
                result = self._knn_mi(X_active, Y_active)
                method = 'knn_exact'
            else:
                result = self._advanced_mi_estimator(X_active, Y_active)
                method = 'advanced_estimator'
            
            # Add all MI metrics
            mi = result['mi']
            entropy_X = result['entropy_X']
            entropy_Y = result['entropy_Y']
            entropy_XY = result['entropy_XY']
            
            # Core MI Metrics
            normalized_mi = 2 * mi / (entropy_X + entropy_Y + 1e-10)
            mi_efficiency = mi / (entropy_X + 1e-10)  # What fraction of input info reaches next layer
            
            # Information gaps and capacity
            max_possible_mi = min(entropy_X, entropy_Y)
            information_gap = max_possible_mi - mi
            capacity_utilization = mi / (max_possible_mi + 1e-10)
            
            # Redundancy and independence
            redundancy = entropy_X + entropy_Y - entropy_XY
            independence_ratio = mi / (redundancy + 1e-10)
            
            result = {
                # Core MI Metrics
                'mi': mi,
                'normalized_mi': normalized_mi,
                'mi_efficiency': mi_efficiency,
                
                # Entropy Metrics
                'entropy_X': entropy_X,
                'entropy_Y': entropy_Y,
                'entropy_XY': entropy_XY,
                
                # Capacity and Gap Analysis
                'max_possible_mi': max_possible_mi,
                'information_gap': information_gap,
                'capacity_utilization': capacity_utilization,
                
                # Redundancy Analysis
                'redundancy': redundancy,
                'independence_ratio': independence_ratio,
                
                # Neuron counts
                'active_neurons_X': n_active_X,
                'active_neurons_Y': n_active_Y,
                'total_neurons_X': X.shape[1],
                'total_neurons_Y': Y.shape[1],
                'sparsity_X': n_active_X / X.shape[1],
                'sparsity_Y': n_active_Y / Y.shape[1],
                
                # Method info
                'method': method,
                'samples_used': n_samples
            }
        
        # Update timing stats
        computation_time = time.time() - start_time
        self._computation_stats['total_time'] += computation_time
        result['computation_time'] = computation_time
        
        # Cache result
        self._cache_result(cache_key, result)
        
        return result
    
    def _zero_metrics(self):
        """Return zero metrics when no active neurons."""
        return {
            'mi': 0.0, 'normalized_mi': 0.0, 'mi_efficiency': 0.0,
            'entropy_X': 0.0, 'entropy_Y': 0.0, 'entropy_XY': 0.0,
            'max_possible_mi': 0.0, 'information_gap': 0.0, 'capacity_utilization': 0.0,
            'redundancy': 0.0, 'independence_ratio': 0.0,
            'active_neurons_X': 0, 'active_neurons_Y': 0,
            'total_neurons_X': 0, 'total_neurons_Y': 0,
            'sparsity_X': 0.0, 'sparsity_Y': 0.0,
            'method': 'zero_active', 'samples_used': 0
        }
    
    def _exact_discrete_mi(self, X: torch.Tensor, Y: torch.Tensor) -> Dict[str, float]:
        """Exact MI computation for discretized values."""
        n_bins = min(int(np.sqrt(X.shape[0]) / 2), 10)
        
        # Discretize
        X_discrete = self._adaptive_discretize(X, n_bins)
        Y_discrete = self._adaptive_discretize(Y, n_bins)
        
        # Compute joint probability distribution
        joint_hist = torch.zeros(n_bins, n_bins)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(Y.shape[1]):
                    joint_hist[X_discrete[i, j], Y_discrete[i, k]] += 1
        
        joint_prob = joint_hist / joint_hist.sum()
        
        # Marginal probabilities
        p_x = joint_prob.sum(dim=1)
        p_y = joint_prob.sum(dim=0)
        
        # Compute MI, H(X), H(Y), H(X,Y)
        mi = 0.0
        h_x = 0.0
        h_y = 0.0
        h_xy = 0.0
        
        for i in range(n_bins):
            if p_x[i] > 0:
                h_x -= p_x[i] * torch.log2(p_x[i])
            if p_y[i] > 0:
                h_y -= p_y[i] * torch.log2(p_y[i])
                
            for j in range(n_bins):
                if joint_prob[i, j] > 0:
                    h_xy -= joint_prob[i, j] * torch.log2(joint_prob[i, j])
                    mi += joint_prob[i, j] * torch.log2(
                        joint_prob[i, j] / (p_x[i] * p_y[j] + 1e-10)
                    )
        
        return {
            'mi': mi.item(),
            'entropy_X': h_x.item(),
            'entropy_Y': h_y.item(),
            'entropy_XY': h_xy.item()
        }
    
    def _knn_mi(self, X: torch.Tensor, Y: torch.Tensor) -> Dict[str, float]:
        """k-NN based MI estimator."""
        if X.is_cuda and CUPY_AVAILABLE:
            X_gpu = cp.asarray(X)
            Y_gpu = cp.asarray(Y)
            
            mi_values = []
            for i in range(Y_gpu.shape[1]):
                mi = mutual_info_regression(X_gpu, Y_gpu[:, i], n_neighbors=3)
                mi_values.extend(mi)
            
            mi = cp.mean(cp.array(mi_values)).get()
            
            h_x = self._knn_entropy(X_gpu)
            h_y = self._knn_entropy(Y_gpu)
            h_xy = self._knn_entropy(cp.hstack([X_gpu, Y_gpu]))
        else:
            X_np = X.cpu().numpy()
            Y_np = Y.cpu().numpy()
            
            # Use sklearn's implementation
            mi_values = []
            for i in range(Y_np.shape[1]):
                mi = mutual_info_regression(X_np, Y_np[:, i], n_neighbors=3)
                mi_values.extend(mi)
            
            mi = np.mean(mi_values)
            
            # Estimate entropies using k-NN
            h_x = self._knn_entropy(X_np)
            h_y = self._knn_entropy(Y_np)
            h_xy = self._knn_entropy(np.hstack([X_np, Y_np]))
        
        return {
            'mi': mi,
            'entropy_X': h_x,
            'entropy_Y': h_y,
            'entropy_XY': h_xy
        }
    
    def _advanced_mi_estimator(self, X: torch.Tensor, Y: torch.Tensor) -> Dict[str, float]:
        """Advanced MI estimator for higher dimensions."""
        X_norm = F.normalize(X, dim=0)
        Y_norm = F.normalize(Y, dim=0)
        
        # Canonical correlation
        corr_matrix = torch.abs(X_norm.T @ Y_norm)
        max_corr = corr_matrix.max(dim=1)[0].mean()
        
        # Upper bound on MI
        mi_upper = -0.5 * torch.log(1 - max_corr**2 + 1e-8)
        
        # Estimate entropies
        h_x = self._differential_entropy(X)
        h_y = self._differential_entropy(Y)
        
        return {
            'mi': mi_upper.item(),
            'entropy_X': h_x,
            'entropy_Y': h_y,
            'entropy_XY': h_x + h_y - mi_upper.item()
        }
    
    def _knn_entropy(self, X, k: int = 3) -> float:
        """Estimate entropy using k-NN distances."""
        if X.shape[0] < k + 1:
            return 0.0
        
        is_cupy = 'cupy' in str(type(X))
        
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
        distances, _ = nbrs.kneighbors(X)
        rho = distances[:, k]
        
        d = X.shape[1]
        
        if is_cupy:
            volume = (cp.pi ** (d/2)) / cp.exp(cp.log(math.gamma(d/2 + 1)))
            h = cp.log(rho + 1e-10).mean() * d + cp.log(volume) + cp.log(X.shape[0]) - cp.log(k)
            return h.get() / np.log(2)
        else:
            volume = (np.pi ** (d/2)) / np.exp(np.log(math.gamma(d/2 + 1)))
            h = np.log(rho + 1e-10).mean() * d + np.log(volume) + np.log(X.shape[0]) - np.log(k)
            return h / np.log(2)
    
    def _differential_entropy(self, X: torch.Tensor) -> float:
        """Estimate differential entropy."""
        if X.shape[1] == 1:
            return self._histogram_entropy(X.squeeze())
        else:
            cov = torch.cov(X.T)
            # Ensure identity matrix is on the same device as the covariance matrix
            identity = torch.eye(X.shape[1], device=cov.device, dtype=cov.dtype)
            sign, logdet = torch.linalg.slogdet(cov + 1e-6 * identity)
            d = X.shape[1]
            h = 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)
            return h.item() / np.log(2)
    
    def _histogram_entropy(self, x: torch.Tensor) -> float:
        """Compute entropy using histogram."""
        hist = torch.histc(x, bins=int(np.sqrt(len(x))))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -torch.sum(hist * torch.log2(hist)).item()
    
    def _adaptive_discretize(self, X: torch.Tensor, n_bins: int) -> torch.Tensor:
        """Adaptive discretization based on data distribution."""
        X_discrete = torch.zeros_like(X, dtype=torch.long)
        
        for j in range(X.shape[1]):
            col = X[:, j]
            if col.std() > 0:
                bins = torch.quantile(col, torch.linspace(0, 1, n_bins + 1))
                bins[-1] += 1e-5
                X_discrete[:, j] = torch.bucketize(col, bins) - 1
            else:
                X_discrete[:, j] = 0
        
        return X_discrete.clamp(0, n_bins - 1)


# Export the analyzer
__all__ = ['MutualInformationAnalyzer']
