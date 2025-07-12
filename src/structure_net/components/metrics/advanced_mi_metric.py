"""
Advanced mutual information metric component.

This component provides advanced MI estimation methods including
KNN-based estimation, exact discrete MI, and high-dimensional estimators.
"""

from typing import Dict, Any, Union, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import math
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

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


class AdvancedMIMetric(BaseMetric):
    """
    Advanced mutual information estimation for neural networks.
    
    Provides multiple estimation methods optimized for different
    dimensionalities and data characteristics.
    """
    
    def __init__(self, method: str = 'auto', threshold: float = 0.01,
                 k_neighbors: int = 3, n_bins: int = None, 
                 compute_gradients: bool = False, n_neighbors: int = None,
                 name: str = None):
        """
        Initialize advanced MI metric.
        
        Args:
            method: MI estimation method ('auto', 'exact_discrete', 'knn', 'advanced')
            threshold: Activation threshold for determining active neurons
            k_neighbors: Number of neighbors for KNN method
            n_bins: Number of bins for discretization (None for auto)
            compute_gradients: Whether to compute MI gradient
            n_neighbors: Alias for k_neighbors (for compatibility)
            name: Optional custom name
        """
        super().__init__(name or "AdvancedMIMetric")
        self.method = method
        self.threshold = threshold
        # Handle alias
        if n_neighbors is not None:
            k_neighbors = n_neighbors
        self.k_neighbors = k_neighbors
        self.n_bins = n_bins
        self.compute_gradients = compute_gradients
        self._measurement_schema = {
            "mi": float,
            "entropy_X": float,
            "entropy_Y": float,
            "entropy_XY": float,
            "method_used": str,
            "active_neurons_X": int,
            "active_neurons_Y": int,
            "sparsity_X": float,
            "sparsity_Y": float
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"X", "Y"},  # Direct activation tensors
            provided_outputs={
                "metrics.mi",
                "metrics.entropy_X",
                "metrics.entropy_Y",
                "metrics.entropy_XY",
                "metrics.method_used",
                "metrics.active_neurons",
                "metrics.sparsity"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.HIGH,
                requires_gpu=CUPY_AVAILABLE,
                parallel_safe=False  # KNN methods aren't thread-safe
            )
        )
    
    def _compute_metric(self, target: Union[ILayer, IModel], 
                       context: EvolutionContext) -> Dict[str, Any]:
        """
        Compute advanced MI metrics.
        
        Args:
            target: Not used directly (metric operates on raw tensors)
            context: Must contain 'X' and 'Y' activation tensors
            
        Returns:
            Dictionary containing MI measurements
        """
        # Get activation tensors from context
        X = context.get('X')
        Y = context.get('Y')
        
        if X is None or Y is None:
            raise ValueError("AdvancedMIMetric requires 'X' and 'Y' tensors in context")
        
        # Validate inputs
        if X.dim() != 2 or Y.dim() != 2:
            raise ValueError("X and Y must be 2D tensors [batch_size, features]")
        
        if X.size(0) != Y.size(0):
            raise ValueError("X and Y must have same batch size")
        
        # Apply threshold to get active neurons only
        X_active_mask = (X.abs() > self.threshold).any(dim=0)
        Y_active_mask = (Y.abs() > self.threshold).any(dim=0)
        
        X_active = X[:, X_active_mask]
        Y_active = Y[:, Y_active_mask]
        
        n_samples = X.shape[0]
        n_active_X = X_active.shape[1]
        n_active_Y = Y_active.shape[1]
        
        if n_active_X == 0 or n_active_Y == 0:
            # No active neurons
            return {
                "mi": 0.0,
                "entropy_X": 0.0,
                "entropy_Y": 0.0,
                "entropy_XY": 0.0,
                "method_used": "zero_active",
                "active_neurons_X": 0,
                "active_neurons_Y": 0,
                "sparsity_X": 0.0,
                "sparsity_Y": 0.0
            }
        
        # Choose method
        if self.method == 'auto':
            if n_active_X <= 10 and n_active_Y <= 10 and n_samples > 50:
                method = 'exact_discrete'
            elif n_active_X + n_active_Y <= 50:
                method = 'knn'
            else:
                method = 'advanced'
        else:
            method = self.method
        
        # Compute MI using selected method
        if method == 'exact_discrete':
            result = self._exact_discrete_mi(X_active, Y_active)
        elif method == 'knn':
            result = self._knn_mi(X_active, Y_active)
        elif method == 'advanced':
            result = self._advanced_mi_estimator(X_active, Y_active)
        else:
            raise ValueError(f"Unknown MI method: {method}")
        
        # Check if we need gradient computation
        compute_gradients = context.get('compute_gradients', False)
        if compute_gradients and 'mi_gradient' not in result:
            # Compute MI gradient by looking at MI across feature subsets
            n_features_y = Y_active.shape[1]
            if n_features_y > 1:
                mi_values = []
                # Compute MI for progressively more features
                for i in range(1, min(n_features_y + 1, 6)):
                    Y_subset = Y_active[:, :i]
                    if method == 'exact_discrete':
                        subset_result = self._exact_discrete_mi(X_active, Y_subset)
                    elif method == 'knn':
                        subset_result = self._knn_mi(X_active, Y_subset)
                    else:
                        subset_result = self._advanced_mi_estimator(X_active, Y_subset)
                    mi_values.append(subset_result['mi'])
                
                # Compute gradient as average rate of change
                if len(mi_values) > 1:
                    diffs = [mi_values[i] - mi_values[i-1] for i in range(1, len(mi_values))]
                    mi_gradient = sum(diffs) / len(diffs)
                else:
                    mi_gradient = 0.0
            else:
                mi_gradient = 0.0
            result['mi_gradient'] = mi_gradient
        
        # Add metadata
        result.update({
            "method_used": method,
            "active_neurons_X": n_active_X,
            "active_neurons_Y": n_active_Y,
            "sparsity_X": n_active_X / X.shape[1],
            "sparsity_Y": n_active_Y / Y.shape[1]
        })
        
        self.log(logging.DEBUG, 
                f"MI computed using {method}: {result['mi']:.3f} "
                f"(active: {n_active_X}x{n_active_Y})")
        
        return result
    
    def _exact_discrete_mi(self, X: torch.Tensor, Y: torch.Tensor) -> Dict[str, float]:
        """Exact MI computation for discretized values."""
        n_bins = self.n_bins or min(int(np.sqrt(X.shape[0]) / 2), 10)
        
        # Discretize
        X_discrete = self._adaptive_discretize(X, n_bins)
        Y_discrete = self._adaptive_discretize(Y, n_bins)
        
        # Compute joint probability distribution
        joint_hist = torch.zeros(n_bins, n_bins, device=X.device)
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
            X_gpu = cp.asarray(X.cpu().numpy())
            Y_gpu = cp.asarray(Y.cpu().numpy())
            
            mi_values = []
            for i in range(Y_gpu.shape[1]):
                mi = mutual_info_regression(X_gpu, Y_gpu[:, i], n_neighbors=self.k_neighbors)
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
                mi = mutual_info_regression(X_np, Y_np[:, i], n_neighbors=self.k_neighbors)
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
    
    def _knn_entropy(self, X, k: int = None) -> float:
        """Estimate entropy using k-NN distances."""
        if k is None:
            k = self.k_neighbors
            
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
                bins = torch.quantile(col, torch.linspace(0, 1, n_bins + 1, device=col.device))
                bins[-1] += 1e-5
                X_discrete[:, j] = torch.bucketize(col, bins) - 1
            else:
                X_discrete[:, j] = 0
        
        return X_discrete.clamp(0, n_bins - 1)