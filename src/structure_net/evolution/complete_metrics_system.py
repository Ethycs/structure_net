#!/usr/bin/env python3
"""
Complete Metrics System - All Advanced Network Analysis Metrics

This module implements the complete suite of metrics for network analysis:
1. Exact Mutual Information Metrics
2. Threshold-Based Activity Metrics  
3. SensLI (Sensitivity) Metrics
4. Graph-Based Sparse Metrics
5. Component Analysis Metrics
6. Betweenness Centrality Metrics
7. Spectral Analysis Metrics
8. Path Analysis Metrics
9. Motif Analysis Metrics
10. Percolation Metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import networkx as nx
from scipy import sparse as sp
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import time
from typing import Dict, List, Tuple, Any, Optional
import copy
import logging

# Import canonical structure_net components
from ..core.layers import StandardSparseLayer
from ..core.network_factory import create_standard_network
from ..core.network_analysis import get_network_stats

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PART 1: COMPLETE MUTUAL INFORMATION METRICS
# ============================================================================

class CompleteMIAnalyzer:
    """Complete mutual information analysis with all metrics."""
    
    def __init__(self, threshold=0.01):
        self.threshold = threshold
        
    def compute_complete_mi_metrics(self, X: torch.Tensor, Y: torch.Tensor) -> Dict[str, float]:
        """
        Compute ALL mutual information metrics.
        """
        # Apply threshold to get active neurons only
        X_active_mask = (X.abs() > self.threshold).any(dim=0)
        Y_active_mask = (Y.abs() > self.threshold).any(dim=0)
        
        X_active = X[:, X_active_mask]
        Y_active = Y[:, Y_active_mask]
        
        n_samples = X.shape[0]
        n_active_X = X_active.shape[1]
        n_active_Y = Y_active.shape[1]
        
        if n_active_X == 0 or n_active_Y == 0:
            return self._zero_metrics()
        
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
        
        return {
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
    
    def _knn_entropy(self, X: np.ndarray, k: int = 3) -> float:
        """Estimate entropy using k-NN distances."""
        if X.shape[0] < k + 1:
            return 0.0
            
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X)
        distances, _ = nbrs.kneighbors(X)
        rho = distances[:, k]
        
        d = X.shape[1]
        volume = (np.pi ** (d/2)) / np.exp(np.log(math.gamma(d/2 + 1)))
        h = np.log(rho + 1e-10).mean() * d + np.log(volume) + np.log(X.shape[0]) - np.log(k)
        
        return h / np.log(2)
    
    def _differential_entropy(self, X: torch.Tensor) -> float:
        """Estimate differential entropy."""
        if X.shape[1] == 1:
            return self._histogram_entropy(X.squeeze())
        else:
            cov = torch.cov(X.T)
            sign, logdet = torch.linalg.slogdet(cov + 1e-6 * torch.eye(X.shape[1]))
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

# ============================================================================
# PART 2: COMPLETE ACTIVITY METRICS
# ============================================================================

class CompleteActivityAnalyzer:
    """Complete analysis of neuron activity patterns."""
    
    def __init__(self, threshold_config):
        self.config = threshold_config
        self.activation_history = defaultdict(list)
        
    def compute_complete_activity_metrics(self, activations: torch.Tensor, 
                                        layer_idx: int) -> Dict[str, float]:
        """
        Compute ALL activity metrics for a layer.
        """
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
        activation_percentiles = torch.quantile(activations.abs().flatten(), 
                                               torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
        
        # Saturation analysis
        saturation_threshold = 10.0
        saturated_neurons = (activations.abs() > saturation_threshold).any(dim=0).sum().item()
        saturation_ratio = saturated_neurons / total_neurons
        
        # Gradient explosion detection
        gradient_explosion_risk = max_activation > 10.0
        
        # Activity pattern analysis
        activity_entropy = self._compute_activity_entropy(neuron_activity_rates)
        activity_gini = self._compute_gini_coefficient(neuron_activity_rates)
        
        return {
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
            'layer_health_score': self._compute_layer_health(active_ratio, max_activation, activity_entropy)
        }
    
    def _compute_activity_entropy(self, activity_rates: torch.Tensor) -> float:
        """Compute entropy of activity distribution."""
        # Normalize to probabilities
        probs = activity_rates / (activity_rates.sum() + 1e-10)
        probs = probs[probs > 0]
        return -torch.sum(probs * torch.log2(probs)).item()
    
    def _compute_gini_coefficient(self, activity_rates: torch.Tensor) -> float:
        """Compute Gini coefficient of activity distribution."""
        sorted_rates = torch.sort(activity_rates)[0]
        n = len(sorted_rates)
        cumsum = torch.cumsum(sorted_rates, dim=0)
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

# ============================================================================
# PART 3: COMPLETE SENSLI METRICS
# ============================================================================

class CompleteSensLIAnalyzer:
    """Complete SensLI (Sensitivity) analysis with gradient-based metrics."""
    
    def __init__(self, network, threshold_config):
        self.network = network
        self.config = threshold_config
        
    def compute_complete_sensli_metrics(self, layer_i: int, layer_j: int,
                                      data_loader, num_batches: int = 10) -> Dict[str, float]:
        """
        Compute ALL SensLI metrics between two layers.
        """
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
            return self._zero_sensli_metrics()
        
        # Aggregate metrics
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
        
        return {
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
    
    def _get_layer_activations_and_gradients(self, data, target, layer_idx):
        """Get activations and gradients for a specific layer."""
        data.requires_grad_(True)
        
        x = data.view(data.size(0), -1)
        activations = []
        
        for i, layer in enumerate(self.network):
            if isinstance(layer, StandardSparseLayer):
                x = layer(x)
                if i == layer_idx:
                    activations = x.clone()
                    
                    # Compute loss and gradients
                    if i < len(self.network) - 1:
                        # Continue forward
                        temp_x = x.clone()
                        for j in range(i + 1, len(self.network)):
                            if isinstance(self.network[j], StandardSparseLayer):
                                temp_x = self.network[j](temp_x)
                            elif isinstance(self.network[j], nn.ReLU):
                                temp_x = self.network[j](temp_x)
                        
                        loss = F.cross_entropy(temp_x, target)
                        grads = torch.autograd.grad(loss, x, retain_graph=True)[0]
                    else:
                        # Output layer
                        loss = F.cross_entropy(x, target)
                        grads = torch.autograd.grad(loss, data, retain_graph=True)[0]
                        grads = grads.view(grads.size(0), -1)
                    
                    return activations, grads
                    
                if i < len(self.network) - 1:
                    x = F.relu(x)
            elif isinstance(layer, nn.ReLU):
                x = layer(x)
        
        return None, None
    
    def _compute_virtual_parameter_sensitivity(self, acts_i, acts_j, grads_i, grads_j):
        """Compute sensitivity to virtual parameters."""
        # Sensitivity of adding a connection between layers
        sensitivity = torch.norm(acts_i.mean(dim=0).unsqueeze(0) @ grads_j.mean(dim=0).unsqueeze(1))
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
        
        # Healthy gradient flow should be stable and moderate
        mean_sens = np.mean(gradient_sensitivities)
        std_sens = np.std(gradient_sensitivities)
        
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

# ============================================================================
# PART 4: COMPLETE GRAPH METRICS
# ============================================================================

class CompleteGraphAnalyzer:
    """Complete graph-based analysis of sparse networks."""
    
    def __init__(self, network, threshold_config):
        self.network = network
        self.config = threshold_config
        
    def compute_complete_graph_metrics(self, activation_data: Dict) -> Dict[str, Any]:
        """
        Compute ALL graph-based metrics.
        """
        # Build active network graph
        G, active_neurons = self._build_active_graph(activation_data)
        
        if G.number_of_nodes() == 0:
            return self._zero_graph_metrics()
        
        metrics = {
            'graph_built': True,
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'active_neurons_per_layer': {k: len(v) for k, v in active_neurons.items()}
        }
        
        # Basic Graph Statistics
        metrics.update(self._compute_basic_graph_metrics(G))
        
        # Degree Statistics
        metrics.update(self._compute_degree_metrics(G))
        
        # Component Analysis
        metrics.update(self._compute_component_metrics(G))
        
        # Betweenness Centrality
        if G.number_of_nodes() > 10:
            metrics.update(self._compute_betweenness_metrics(G))
        
        # Spectral Analysis
        if G.number_of_nodes() > 5:
            metrics.update(self._compute_spectral_metrics(G))
        
        # Path Analysis
        metrics.update(self._compute_path_metrics(G, active_neurons))
        
        # Motif Analysis
        if G.number_of_nodes() < 1000:
            metrics.update(self._compute_motif_metrics(G))
        
        # Percolation Analysis
        metrics.update(self._compute_percolation_metrics(G, active_neurons))
        
        return metrics
    
    def _build_active_graph(self, activation_data):
        """Build directed graph of active neurons using proper 2D activation processing."""
        G = nx.DiGraph()
        active_neurons = {}
        
        # Add nodes for active neurons - using proper 2D tensor processing
        for layer_idx, acts in activation_data.items():
            # acts should be 2D: [batch_size, num_neurons]
            if acts.dim() != 2:
                logger.warning(f"Skipping layer {layer_idx}: activation tensor has unexpected dim {acts.dim()}")
                continue
            
            # Compute mean absolute activation across batch dimension
            mean_abs_acts = acts.abs().mean(dim=0)  # Shape: [num_neurons]
            
            # Find active neurons based on threshold
            active_mask = mean_abs_acts > self.config.activation_threshold
            active_indices = torch.where(active_mask)[0].tolist()
            active_neurons[layer_idx] = active_indices
            
            for neuron_idx in active_indices:
                node_id = f"L{layer_idx}_N{neuron_idx}"
                activation_val = mean_abs_acts[neuron_idx].item()
                
                G.add_node(node_id, 
                          layer=layer_idx, 
                          neuron=neuron_idx,
                          activation=activation_val)
        
        # Add edges based on weights
        sparse_layers = [layer for layer in self.network if isinstance(layer, StandardSparseLayer)]
        
        for i in range(len(sparse_layers) - 1):
            # The weights connecting FROM layer i TO layer i+1 are in layer i+1
            dst_layer = sparse_layers[i + 1]
            
            # Get sparse weights from destination layer
            if hasattr(dst_layer, 'mask'):
                weights = dst_layer.linear.weight * dst_layer.mask
            else:
                weights = dst_layer.linear.weight
            
            # Only consider weights above threshold
            active_connections = weights.abs() > self.config.weight_threshold
            
            # Use efficient sparse indexing instead of nested loops
            dst_indices, src_indices = torch.where(active_connections)
            
            # Get active neuron sets for fast lookups
            active_src_set = set(active_neurons.get(i, []))
            active_dst_set = set(active_neurons.get(i + 1, []))
            
            # Iterate only through existing sparse connections
            for dst_idx, src_idx in zip(dst_indices.tolist(), src_indices.tolist()):
                # Check if both source and destination neurons are active
                if src_idx in active_src_set and dst_idx in active_dst_set:
                    src_node = f"L{i}_N{src_idx}"
                    dst_node = f"L{i+1}_N{dst_idx}"
                    weight_val = weights[dst_idx, src_idx].item()
                    
                    # Check if nodes exist before adding edge
                    if G.has_node(src_node) and G.has_node(dst_node):
                        G.add_edge(src_node, dst_node, weight=weight_val)
        
        return G, active_neurons
    
    def _compute_basic_graph_metrics(self, G):
        """Basic graph statistics."""
        metrics = {}
        
        # Density
        metrics['density'] = nx.density(G)
        
        # Degree statistics
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        
        metrics['avg_in_degree'] = np.mean(in_degrees) if in_degrees else 0
        metrics['avg_out_degree'] = np.mean(out_degrees) if out_degrees else 0
        metrics['max_in_degree'] = max(in_degrees) if in_degrees else 0
        metrics['max_out_degree'] = max(out_degrees) if out_degrees else 0
        
        return metrics
    
    def _compute_degree_metrics(self, G):
        """Degree distribution analysis."""
        metrics = {}
        
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        
        # Degree statistics
        metrics['in_degree_std'] = np.std(in_degrees) if in_degrees else 0
        metrics['out_degree_std'] = np.std(out_degrees) if out_degrees else 0
        
        # Hub detection
        if in_degrees:
            in_degree_threshold = np.mean(in_degrees) + 2 * np.std(in_degrees)
            metrics['num_in_hubs'] = sum(1 for d in in_degrees if d > in_degree_threshold)
        else:
            metrics['num_in_hubs'] = 0
            
        if out_degrees:
            out_degree_threshold = np.mean(out_degrees) + 2 * np.std(out_degrees)
            metrics['num_out_hubs'] = sum(1 for d in out_degrees if d > out_degree_threshold)
        else:
            metrics['num_out_hubs'] = 0
        
        return metrics
    
    def _compute_component_metrics(self, G):
        """Connected component analysis."""
        metrics = {}
        
        # Weakly connected components
        wcc = list(nx.weakly_connected_components(G))
        metrics['num_weakly_connected_components'] = len(wcc)
        metrics['largest_wcc_size'] = len(max(wcc, key=len)) if wcc else 0
        metrics['isolated_neurons'] = sum(1 for c in wcc if len(c) == 1)
        
        # Strongly connected components
        scc = list(nx.strongly_connected_components(G))
        metrics['num_strongly_connected_components'] = len(scc)
        metrics['largest_scc_size'] = len(max(scc, key=len)) if scc else 0
        
        return metrics
    
    def _compute_betweenness_metrics(self, G):
        """Betweenness centrality analysis."""
        metrics = {}
        
        # Sample nodes for efficiency
        nodes = list(G.nodes())
        k = min(100, len(nodes))
        sample_nodes = np.random.choice(nodes, k, replace=False).tolist()
        
        # Compute betweenness
        betweenness = nx.betweenness_centrality_subset(
            G, sources=sample_nodes, targets=sample_nodes, normalized=True
        )
        
        # Statistics
        bc_values = list(betweenness.values())
        metrics['avg_betweenness'] = np.mean(bc_values)
        metrics['max_betweenness'] = np.max(bc_values)
        metrics['betweenness_std'] = np.std(bc_values)
        
        # Find bottleneck neurons
        threshold = metrics['avg_betweenness'] + 2 * metrics['betweenness_std']
        bottlenecks = [(node, bc) for node, bc in betweenness.items() if bc > threshold]
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        
        metrics['bottleneck_neurons'] = bottlenecks[:10]
        metrics['num_bottlenecks'] = len(bottlenecks)
        
        return metrics
    
    def _compute_spectral_metrics(self, G):
        """Spectral analysis of the graph."""
        metrics = {}
        
        # Get largest weakly connected component
        largest_wcc = max(nx.weakly_connected_components(G), key=len)
        G_wcc = G.subgraph(largest_wcc)
        
        if len(G_wcc) < 2:
            return metrics
        
        try:
            # Compute Laplacian
            L = nx.laplacian_matrix(G_wcc.to_undirected()).astype(float)
            
            # Compute smallest k eigenvalues
            k = min(10, len(G_wcc) - 1)
            eigenvals = sp.linalg.eigsh(L, k=k, which='SM', return_eigenvectors=False)
            eigenvals = np.sort(eigenvals)
            
            metrics['algebraic_connectivity'] = eigenvals[1] if len(eigenvals) > 1 else 0
            metrics['spectral_gap'] = eigenvals[1] - eigenvals[0] if len(eigenvals) > 1 else 0
            metrics['spectral_radius'] = eigenvals[-1]
            
            # Cheeger constant approximation
            metrics['cheeger_constant'] = metrics['algebraic_connectivity'] / 2
            
            # Information mixing time
            if metrics['algebraic_connectivity'] > 0:
                metrics['mixing_time'] = 1 / metrics['algebraic_connectivity']
            else:
                metrics['mixing_time'] = float('inf')
                
        except Exception as e:
            logger.warning(f"Spectral computation failed: {e}")
            
        return metrics
    
    def _compute_path_metrics(self, G, active_neurons):
        """Path-based analysis."""
        metrics = {}
        
        # Find input and output layers
        layer_indices = sorted(active_neurons.keys())
        if not layer_indices:
            return metrics
            
        input_layer = layer_indices[0]
        output_layer = layer_indices[-1]
        
        input_nodes = [f"L{input_layer}_N{n}" for n in active_neurons[input_layer]]
        output_nodes = [f"L{output_layer}_N{n}" for n in active_neurons[output_layer]]
        
        # Sample paths
        num_paths = 0
        path_lengths = []
        critical_paths = []
        
        # Sample random input-output pairs
        n_samples = min(10, len(input_nodes) * len(output_nodes))
        
        for _ in range(n_samples):
            if not input_nodes or not output_nodes:
                break
                
            src = np.random.choice(input_nodes)
            dst = np.random.choice(output_nodes)
            
            try:
                path = nx.shortest_path(G, src, dst)
                num_paths += 1
                path_lengths.append(len(path) - 1)
                
                # Compute path weight
                path_weight = 1.0
                for i in range(len(path) - 1):
                    if G.has_edge(path[i], path[i+1]):
                        path_weight *= abs(G[path[i]][path[i+1]]['weight'])
                
                critical_paths.append({
                    'path': path,
                    'length': len(path) - 1,
                    'weight': path_weight
                })
                
            except nx.NetworkXNoPath:
                pass
        
        metrics['num_paths_sampled'] = num_paths
        metrics['avg_path_length'] = np.mean(path_lengths) if path_lengths else 0
        metrics['min_path_length'] = min(path_lengths) if path_lengths else 0
        metrics['max_path_length'] = max(path_lengths) if path_lengths else 0
        
        # Top critical paths
        critical_paths.sort(key=lambda x: x['weight'], reverse=True)
        metrics['top_critical_paths'] = critical_paths[:5]
        
        # Reachability
        metrics['input_output_reachability'] = num_paths / n_samples if n_samples > 0 else 0
        
        return metrics
    
    def _compute_motif_metrics(self, G):
        """Motif analysis (small subgraph patterns)."""
        metrics = {}
        
        # Count 3-node motifs
        motif_counts = {
            'feedforward': 0,
            'feedback': 0,
            'mutual': 0,
            'chain': 0
        }
        
        nodes = list(G.nodes())
        
        # Sample if too many nodes
        if len(nodes) > 100:
            nodes = np.random.choice(nodes, 100, replace=False).tolist()
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                for k in range(j + 1, len(nodes)):
                    subgraph = G.subgraph([nodes[i], nodes[j], nodes[k]])
                    edges = list(subgraph.edges())
                    
                    if len(edges) == 2:
                        motif_counts['chain'] += 1
                    elif len(edges) == 3:
                        if nx.is_directed_acyclic_graph(subgraph):
                            motif_counts['feedforward'] += 1
                        else:
                            if len(list(nx.simple_cycles(subgraph))) > 0:
                                motif_counts['feedback'] += 1
                            else:
                                motif_counts['mutual'] += 1
        
        metrics['motif_counts'] = motif_counts
        total_motifs = sum(motif_counts.values())
        
        if total_motifs > 0:
            metrics['motif_fractions'] = {
                k: v / total_motifs for k, v in motif_counts.items()
            }
        
        return metrics
    
    def _compute_percolation_metrics(self, G, active_neurons):
        """Percolation analysis."""
        metrics = {}
        
        # Compute actual density
        total_possible_edges = sum(
            len(active_neurons.get(i, [])) * len(active_neurons.get(i+1, []))
            for i in range(len(self.network) - 1)
        )
        
        actual_edges = G.number_of_edges()
        current_density = actual_edges / total_possible_edges if total_possible_edges > 0 else 0
        
        metrics['current_edge_density'] = current_density
        
        # Theoretical percolation threshold
        avg_layer_size = np.mean([len(neurons) for neurons in active_neurons.values()])
        theoretical_threshold = 1 / avg_layer_size if avg_layer_size > 0 else 0
        
        metrics['percolation_threshold'] = theoretical_threshold
        metrics['above_percolation'] = current_density > theoretical_threshold
        metrics['distance_to_percolation'] = current_density - theoretical_threshold
        
        # Giant component fraction
        if G.number_of_nodes() > 0:
            largest_wcc_size = max(len(c) for c in nx.weakly_connected_components(G))
            metrics['giant_component_fraction'] = largest_wcc_size / G.number_of_nodes()
        else:
            metrics['giant_component_fraction'] = 0
        
        return metrics
    
    def _zero_graph_metrics(self):
        """Return zero metrics when no graph can be built."""
        return {
            'graph_built': False,
            'num_nodes': 0,
            'num_edges': 0,
            'density': 0.0,
            'avg_in_degree': 0.0,
            'avg_out_degree': 0.0,
            'num_weakly_connected_components': 0,
            'input_output_reachability': 0.0
        }
# ============================================================================
# PART 5: COMPLETE INTEGRATED SYSTEM
# ============================================================================

class CompleteMetricsSystem:
    """Complete metrics system integrating all analyzers."""
    
    def __init__(self, network, threshold_config, metrics_config):
        self.network = network
        self.threshold_config = threshold_config
        self.metrics_config = metrics_config
        
        # Initialize all analyzers
        self.mi_analyzer = CompleteMIAnalyzer(threshold_config.activation_threshold)
        self.activity_analyzer = CompleteActivityAnalyzer(threshold_config)
        self.sensli_analyzer = CompleteSensLIAnalyzer(network, threshold_config)
        self.graph_analyzer = CompleteGraphAnalyzer(network, threshold_config)
        
    def compute_all_metrics(self, data_loader, num_batches: int = 10) -> Dict[str, Any]:
        """
        Compute ALL metrics for the network.
        """
        logger.info("🔬 Computing complete metrics suite...")
        
        results = {
            'mi_metrics': {},
            'activity_metrics': {},
            'sensli_metrics': {},
            'graph_metrics': {},
            'summary': {}
        }
        
        # Collect activation data
        activation_data = self._collect_activation_data(data_loader, num_batches)
        
        # Get sparse layers
        sparse_layers = [layer for layer in self.network if isinstance(layer, StandardSparseLayer)]
        
        # 1. MI Metrics for each layer pair
        logger.info("  Computing MI metrics...")
        for i in range(len(sparse_layers) - 1):
            if i in activation_data and i+1 in activation_data:
                acts_i = activation_data[i]
                acts_j = activation_data[i+1]
                
                # Apply ReLU if not output layer
                if i < len(sparse_layers) - 1:
                    acts_i = F.relu(acts_i)
                if i+1 < len(sparse_layers) - 1:
                    acts_j = F.relu(acts_j)
                
                mi_metrics = self.mi_analyzer.compute_complete_mi_metrics(acts_i, acts_j)
                results['mi_metrics'][f'layer_{i}_{i+1}'] = mi_metrics
        
        # 2. Activity Metrics for each layer
        logger.info("  Computing activity metrics...")
        for layer_idx, acts in activation_data.items():
            activity_metrics = self.activity_analyzer.compute_complete_activity_metrics(acts, layer_idx)
            results['activity_metrics'][f'layer_{layer_idx}'] = activity_metrics
        
        # 3. SensLI Metrics for each layer pair
        logger.info("  Computing SensLI metrics...")
        for i in range(len(sparse_layers) - 1):
            sensli_metrics = self.sensli_analyzer.compute_complete_sensli_metrics(
                i, i+1, data_loader, num_batches
            )
            results['sensli_metrics'][f'layer_{i}_{i+1}'] = sensli_metrics
        
        # 4. Graph Metrics
        logger.info("  Computing graph metrics...")
        graph_metrics = self.graph_analyzer.compute_complete_graph_metrics(activation_data)
        results['graph_metrics'] = graph_metrics
        
        # 5. Summary metrics
        results['summary'] = self._compute_summary_metrics(results)
        
        logger.info("✅ Complete metrics computation finished")
        
        return results
    
    def _collect_activation_data(self, data_loader, num_batches):
        """Collect activation data for analysis."""
        activation_data = defaultdict(list)
        
        self.network.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                if batch_idx >= num_batches:
                    break
                
                x = data.view(data.size(0), -1)
                layer_idx = 0
                
                for layer in self.network:
                    if isinstance(layer, StandardSparseLayer):
                        x = layer(x)
                        activation_data[layer_idx].append(x.clone())
                        layer_idx += 1
                        if layer_idx < len([l for l in self.network if isinstance(l, StandardSparseLayer)]):
                            x = F.relu(x)
                    elif isinstance(layer, nn.ReLU):
                        x = layer(x)
        
        # Aggregate activations - concatenate all batches to keep 2D shape [total_samples, features]
        for layer_idx in activation_data:
            # Concatenate all batches: [batch1, batch2, ...] -> [total_samples, features]
            activation_data[layer_idx] = torch.cat(activation_data[layer_idx], dim=0)
            
            # Ensure we have 2D tensors for proper analysis
            assert activation_data[layer_idx].dim() == 2, f"Layer {layer_idx} activations should be 2D, got {activation_data[layer_idx].shape}"
        
        return dict(activation_data)
    
    def _compute_summary_metrics(self, results):
        """Compute high-level summary metrics."""
        summary = {}
        
        # MI Summary
        mi_results = results['mi_metrics']
        if mi_results:
            mi_efficiencies = [r['mi_efficiency'] for r in mi_results.values()]
            summary['avg_mi_efficiency'] = np.mean(mi_efficiencies)
            summary['min_mi_efficiency'] = np.min(mi_efficiencies)
            summary['bottleneck_layers'] = sum(1 for eff in mi_efficiencies if eff < 0.3)
        
        # Activity Summary
        activity_results = results['activity_metrics']
        if activity_results:
            active_ratios = [r['active_ratio'] for r in activity_results.values()]
            health_scores = [r['layer_health_score'] for r in activity_results.values()]
            summary['avg_active_ratio'] = np.mean(active_ratios)
            summary['avg_health_score'] = np.mean(health_scores)
            summary['dead_layers'] = sum(1 for ratio in active_ratios if ratio < 0.01)
        
        # SensLI Summary
        sensli_results = results['sensli_metrics']
        if sensli_results:
            critical_bottlenecks = sum(1 for r in sensli_results.values() if r['critical_bottleneck'])
            summary['critical_bottlenecks'] = critical_bottlenecks
            
            priorities = [r['intervention_priority'] for r in sensli_results.values()]
            summary['avg_intervention_priority'] = np.mean(priorities)
        
        # Graph Summary
        graph_results = results['graph_metrics']
        if graph_results.get('graph_built', False):
            summary['network_connected'] = graph_results['num_weakly_connected_components'] <= 1
            summary['reachability'] = graph_results.get('input_output_reachability', 0)
            summary['graph_efficiency'] = graph_results.get('avg_path_length', float('inf'))
        
        return summary

# Export all classes
__all__ = [
    'CompleteMIAnalyzer',
    'CompleteActivityAnalyzer', 
    'CompleteSensLIAnalyzer',
    'CompleteGraphAnalyzer',
    'CompleteMetricsSystem'
]
