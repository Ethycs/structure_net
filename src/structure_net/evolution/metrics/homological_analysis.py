#!/usr/bin/env python3
"""
Homological Analysis for Neural Networks

Provides chain complex analysis and homological metrics for neural networks.
Extracted from the compactification module for reusability and modularity.

Key Features:
- Chain complex analysis (kernel, image, homology computation)
- Betti number computation
- Information flow analysis
- Rank efficiency metrics
- SVD-based numerical stability
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from .base import BaseMetricAnalyzer, NetworkAnalyzerMixin, StatisticalUtilsMixin, MetricResult
from ..profiling import profile_component, profile_operation, ProfilerLevel


@dataclass
class ChainData:
    """Chain complex data for a layer."""
    kernel_basis: torch.Tensor
    image_basis: torch.Tensor
    homology_basis: torch.Tensor
    rank: int
    betti_numbers: List[int]
    persistence_diagram: Optional[List[Tuple[float, float]]] = None
    information_efficiency: float = 0.0
    cascade_zeros: torch.Tensor = None


@profile_component(component_name="homological_analyzer", level=ProfilerLevel.DETAILED)
class HomologicalAnalyzer(BaseMetricAnalyzer, NetworkAnalyzerMixin, StatisticalUtilsMixin):
    """
    Specialized analyzer for homological properties of neural networks.
    
    Performs chain complex analysis to understand information flow,
    bottlenecks, and topological structure of weight matrices.
    """
    
    def __init__(self, threshold_config=None, tolerance: float = 1e-6):
        if threshold_config is None:
            from .base import ThresholdConfig
            threshold_config = ThresholdConfig()
        
        super().__init__(threshold_config)
        self.tolerance = tolerance
        self.chain_history = []
        
    def compute_metrics(self, weight_matrix: torch.Tensor, 
                       prev_chain: Optional[ChainData] = None) -> Dict[str, Any]:
        """
        Compute comprehensive homological metrics for a weight matrix.
        
        Args:
            weight_matrix: Weight matrix to analyze
            prev_chain: Previous layer's chain data for homology computation
            
        Returns:
            Dictionary containing all homological metrics
        """
        start_time = time.time()
        
        # Validate input
        self._validate_tensor_input(weight_matrix, "weight_matrix", 2)
        
        # Check cache
        cache_key = self._cache_key(weight_matrix, prev_chain)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        with profile_operation("homological_analysis", "topology"):
            # Core chain complex analysis
            chain_data = self._analyze_chain_complex(weight_matrix, prev_chain)
            
            # Compute derived metrics
            metrics = {
                'chain_data': chain_data,
                'rank': chain_data.rank,
                'betti_numbers': chain_data.betti_numbers,
                'information_efficiency': chain_data.information_efficiency,
                'homological_complexity': sum(chain_data.betti_numbers),
                'kernel_dimension': chain_data.kernel_basis.shape[1],
                'image_dimension': chain_data.image_basis.shape[1],
                'homology_dimension': chain_data.homology_basis.shape[1],
                'rank_ratio': self._compute_rank_ratio(chain_data, weight_matrix),
                'information_flow': self._analyze_information_flow(chain_data),
                'bottleneck_severity': self._compute_bottleneck_severity(chain_data),
                'cascade_zeros': chain_data.cascade_zeros,
                'topological_stability': self._compute_topological_stability(chain_data)
            }
            
            # Add to history
            self.chain_history.append(chain_data)
            
            # Update computation stats
            computation_time = time.time() - start_time
            self._computation_stats['total_calls'] += 1
            self._computation_stats['total_time'] += computation_time
            
            # Cache result
            self._cache_result(cache_key, metrics)
            
            return metrics
    
    def _analyze_chain_complex(self, weight_matrix: torch.Tensor, 
                              prev_chain: Optional[ChainData] = None) -> ChainData:
        """Perform complete chain complex analysis."""
        
        # Step 1: Singular Value Decomposition for numerical stability
        U, S, Vt = torch.linalg.svd(weight_matrix, full_matrices=False)
        
        # Step 2: Determine effective rank
        rank = torch.sum(S > self.tolerance).item()
        
        # Step 3: Compute kernel basis (nullspace)
        if rank < weight_matrix.shape[1]:
            kernel_basis = Vt[rank:].T  # Orthogonal complement
        else:
            kernel_basis = torch.zeros(weight_matrix.shape[1], 0, device=weight_matrix.device)
        
        # Step 4: Compute image basis (column space)
        image_basis = U[:, :rank]
        
        # Step 5: Homology computation H = ker(∂) / im(∂_{+1})
        if prev_chain is not None:
            homology_basis = self._compute_homology(kernel_basis, prev_chain.image_basis)
        else:
            homology_basis = kernel_basis
        
        # Step 6: Topological invariants (Betti numbers)
        betti_0 = self._count_connected_components(weight_matrix)
        betti_1 = max(0, homology_basis.shape[1] - betti_0)
        betti_numbers = [betti_0, betti_1]
        
        # Step 7: Information efficiency
        total_capacity = weight_matrix.shape[1]
        information_efficiency = rank / total_capacity if total_capacity > 0 else 0.0
        
        # Step 8: Cascade zero prediction
        cascade_zeros = self._predict_cascade_zeros(kernel_basis)
        
        return ChainData(
            kernel_basis=kernel_basis,
            image_basis=image_basis,
            homology_basis=homology_basis,
            rank=rank,
            betti_numbers=betti_numbers,
            information_efficiency=information_efficiency,
            cascade_zeros=cascade_zeros
        )
    
    def _compute_homology(self, kernel: torch.Tensor, prev_image: torch.Tensor) -> torch.Tensor:
        """
        Compute homology as quotient space: H = ker(∂) / im(∂_{+1})
        
        This identifies "true" information content that's not just
        inherited from previous layers.
        """
        if kernel.shape[1] == 0 or prev_image.shape[1] == 0:
            return kernel
        
        try:
            # Compute orthogonal complement of previous image
            Q, R = torch.linalg.qr(prev_image)
            
            # Project kernel onto complement: H = ker ∩ (im)⊥
            proj = torch.eye(kernel.shape[0], device=kernel.device) - Q @ Q.T
            homology = proj @ kernel
            
            # Remove near-zero vectors (numerical stability)
            norms = torch.norm(homology, dim=0)
            mask = norms > self.tolerance
            
            return homology[:, mask]
            
        except Exception as e:
            # Fallback to kernel if computation fails
            return kernel
    
    def _count_connected_components(self, weight_matrix: torch.Tensor) -> int:
        """Count connected components (simplified β₀)."""
        # Convert to adjacency matrix
        adj = (weight_matrix.abs() > self.tolerance).float()
        
        # Simple connected components via matrix powers
        n = adj.shape[0]
        reachability = adj + torch.eye(n, device=adj.device)
        
        # Power iteration to find transitive closure
        for _ in range(min(10, n)):  # Limit iterations for efficiency
            new_reach = torch.matmul(reachability, reachability)
            if torch.allclose(new_reach, reachability, atol=self.tolerance):
                break
            reachability = new_reach
        
        # Count unique rows (components)
        unique_rows = torch.unique(reachability, dim=0)
        return unique_rows.shape[0]
    
    def _predict_cascade_zeros(self, kernel_basis: torch.Tensor) -> torch.Tensor:
        """Predict which neurons will be forced to zero due to kernel structure."""
        if kernel_basis.shape[1] == 0:
            return torch.tensor([], device=kernel_basis.device)
        
        # Neurons that only receive input from kernel elements
        kernel_mask = torch.any(kernel_basis.abs() > self.tolerance, dim=1)
        cascade_candidates = torch.where(kernel_mask)[0]
        
        return cascade_candidates
    
    def _compute_rank_ratio(self, chain_data: ChainData, weight_matrix: torch.Tensor) -> float:
        """Compute ratio of effective rank to total capacity."""
        total_capacity = weight_matrix.shape[1]
        return chain_data.rank / total_capacity if total_capacity > 0 else 0.0
    
    def _analyze_information_flow(self, chain_data: ChainData) -> Dict[str, float]:
        """Analyze information flow characteristics."""
        total_dim = chain_data.kernel_basis.shape[0]
        
        if total_dim == 0:
            return {
                'flow_efficiency': 0.0,
                'information_loss': 1.0,
                'bottleneck_ratio': 1.0,
                'redundancy_ratio': 0.0
            }
        
        # Flow efficiency: how much information passes through
        flow_efficiency = chain_data.rank / total_dim
        
        # Information loss: how much is lost
        information_loss = 1.0 - flow_efficiency
        
        # Bottleneck ratio: kernel dimension relative to total
        bottleneck_ratio = chain_data.kernel_basis.shape[1] / total_dim
        
        # Redundancy ratio: how much information is redundant
        redundancy_ratio = (total_dim - chain_data.homology_basis.shape[1]) / total_dim
        
        return {
            'flow_efficiency': flow_efficiency,
            'information_loss': information_loss,
            'bottleneck_ratio': bottleneck_ratio,
            'redundancy_ratio': redundancy_ratio
        }
    
    def _compute_bottleneck_severity(self, chain_data: ChainData) -> float:
        """Compute severity of information bottlenecks."""
        if chain_data.kernel_basis.shape[0] == 0:
            return 0.0
        
        # Combine multiple bottleneck indicators
        rank_deficiency = 1.0 - chain_data.information_efficiency
        kernel_ratio = chain_data.kernel_basis.shape[1] / chain_data.kernel_basis.shape[0]
        betti_complexity = sum(chain_data.betti_numbers) / 10.0  # Normalize
        
        # Weighted combination
        severity = (0.5 * rank_deficiency + 
                   0.3 * kernel_ratio + 
                   0.2 * betti_complexity)
        
        return min(1.0, severity)
    
    def _compute_topological_stability(self, chain_data: ChainData) -> float:
        """Compute stability of topological structure."""
        if len(self.chain_history) < 2:
            return 1.0
        
        # Compare with previous chain data
        prev_chain = self.chain_history[-2]
        
        # Betti number stability
        betti_diff = sum(abs(a - b) for a, b in zip(
            chain_data.betti_numbers, prev_chain.betti_numbers
        ))
        betti_stability = 1.0 / (1.0 + betti_diff)
        
        # Rank stability
        rank_diff = abs(chain_data.rank - prev_chain.rank)
        rank_stability = 1.0 / (1.0 + rank_diff / max(chain_data.rank, prev_chain.rank, 1))
        
        # Combined stability
        return (betti_stability + rank_stability) / 2.0
    
    def design_next_layer_structure(self, 
                                   prev_chain: ChainData,
                                   target_dim: int,
                                   sparsity: float = 0.02) -> Dict[str, Any]:
        """
        Design optimal structure for next layer based on chain analysis.
        
        This is the key function that guides architecture construction.
        """
        # Information-carrying subspace
        effective_dim = prev_chain.rank
        
        # Avoid connecting from kernel (dead information)
        avoid_indices = prev_chain.cascade_zeros
        
        # Design patch locations at information-rich regions
        patch_locations = self._find_information_extrema(prev_chain)
        
        # Calculate optimal patch count based on information density
        information_density = effective_dim / prev_chain.kernel_basis.shape[0]
        recommended_patches = max(1, int(target_dim * sparsity / 0.2))
        
        # Adjust sparsity based on homological complexity
        betti_complexity = sum(prev_chain.betti_numbers)
        skeleton_sparsity = sparsity * (0.25 + 0.1 * betti_complexity)
        
        return {
            'effective_input_dim': effective_dim,
            'avoid_connections_from': avoid_indices,
            'patch_locations': patch_locations,
            'recommended_patches': recommended_patches,
            'skeleton_sparsity': skeleton_sparsity,
            'information_density': information_density,
            'homological_complexity': betti_complexity,
            'bottleneck_severity': self._compute_bottleneck_severity(prev_chain)
        }
    
    def _find_information_extrema(self, chain_data: ChainData) -> List[int]:
        """Find locations of high information content using image basis."""
        if chain_data.image_basis.shape[1] == 0:
            return []
        
        # Compute information content per dimension
        info_content = torch.norm(chain_data.image_basis, dim=1)
        
        # Find local maxima (information concentration points)
        extrema = []
        for i in range(1, len(info_content) - 1):
            if (info_content[i] > info_content[i-1] and 
                info_content[i] > info_content[i+1] and
                info_content[i] > info_content.mean()):
                extrema.append(i)
        
        return extrema
    
    def detect_information_bottlenecks(self, chain_data: ChainData, 
                                     threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Detect information bottlenecks using homological analysis.
        
        A bottleneck occurs when:
        1. Rank drops significantly (information loss)
        2. Kernel dimension increases (dead information)
        3. Betti numbers indicate topological holes
        """
        bottlenecks = []
        
        # Check for rank deficiency
        input_dim = chain_data.kernel_basis.shape[0]
        if input_dim == 0:
            return bottlenecks
            
        rank_ratio = chain_data.rank / input_dim
        
        if rank_ratio < threshold:
            bottlenecks.append({
                'type': 'rank_deficiency',
                'severity': 1.0 - rank_ratio,
                'location': 'global',
                'recommendation': 'add_patches',
                'metric_value': rank_ratio
            })
        
        # Check for large kernel (dead information)
        kernel_ratio = chain_data.kernel_basis.shape[1] / input_dim
        if kernel_ratio > threshold:
            bottlenecks.append({
                'type': 'large_kernel',
                'severity': kernel_ratio,
                'location': 'kernel_space',
                'recommendation': 'avoid_connections',
                'metric_value': kernel_ratio
            })
        
        # Check for topological holes
        if sum(chain_data.betti_numbers) > 2:
            bottlenecks.append({
                'type': 'topological_holes',
                'severity': sum(chain_data.betti_numbers) / 10,
                'location': 'topology',
                'recommendation': 'add_skip_connections',
                'metric_value': sum(chain_data.betti_numbers)
            })
        
        return bottlenecks
    
    def get_homological_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of homological analysis."""
        if not self.chain_history:
            return {}
        
        # Collect statistics across all layers
        layer_ranks = [data.rank for data in self.chain_history]
        betti_numbers = [data.betti_numbers for data in self.chain_history]
        information_flow = [data.information_efficiency for data in self.chain_history]
        
        # Overall homological complexity
        homological_complexity = sum(sum(betti) for betti in betti_numbers)
        
        # Information preservation ratio
        total_input_dim = sum(data.kernel_basis.shape[0] for data in self.chain_history)
        total_preserved_dim = sum(layer_ranks)
        preservation_ratio = total_preserved_dim / total_input_dim if total_input_dim > 0 else 0
        
        # Topology stability
        stability_scores = []
        for i in range(1, len(self.chain_history)):
            prev_betti = self.chain_history[i-1].betti_numbers
            curr_betti = self.chain_history[i].betti_numbers
            
            if len(prev_betti) == len(curr_betti):
                if torch.cuda.is_available() and CUPY_AVAILABLE:
                    prev_betti_gpu = cp.array(prev_betti)
                    curr_betti_gpu = cp.array(curr_betti)
                    similarity = 1.0 - cp.mean(cp.abs(prev_betti_gpu - curr_betti_gpu)).get()
                else:
                    similarity = 1.0 - np.mean([abs(a - b) for a, b in zip(prev_betti, curr_betti)])
                stability_scores.append(max(0, similarity))
        
        if torch.cuda.is_available() and CUPY_AVAILABLE:
            topology_stability = cp.mean(cp.array(stability_scores)).get() if stability_scores else 1.0
            avg_efficiency = cp.mean(cp.array(information_flow)).get() if information_flow else 0
        else:
            topology_stability = np.mean(stability_scores) if stability_scores else 1.0
            avg_efficiency = np.mean(information_flow) if information_flow else 0
        
        return {
            'layer_ranks': layer_ranks,
            'betti_numbers': betti_numbers,
            'information_flow': information_flow,
            'homological_complexity': homological_complexity,
            'preservation_ratio': preservation_ratio,
            'average_efficiency': avg_efficiency,
            'topology_stability': topology_stability,
            'total_layers_analyzed': len(self.chain_history),
            'computation_stats': self.get_computation_stats()
        }
    
    def clear_history(self):
        """Clear chain history for new analysis."""
        self.chain_history.clear()
        self.clear_cache()


# Factory function for easy creation
def create_homological_analyzer(tolerance: float = 1e-6, **kwargs) -> HomologicalAnalyzer:
    """
    Factory function to create homological analyzer.
    
    Args:
        tolerance: Numerical tolerance for computations
        **kwargs: Additional configuration options
        
    Returns:
        Configured HomologicalAnalyzer
    """
    return HomologicalAnalyzer(tolerance=tolerance, **kwargs)


# Export main classes
__all__ = [
    'ChainData',
    'HomologicalAnalyzer', 
    'create_homological_analyzer'
]
