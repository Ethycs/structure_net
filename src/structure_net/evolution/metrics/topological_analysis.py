#!/usr/bin/env python3
"""
Topological Analysis for Neural Networks

Provides topological data analysis (TDA) and extrema detection for neural networks.
Extracted from the compactification module for reusability and modularity.

Key Features:
- Extrema detection and scoring
- Persistence diagram computation
- Topological signature analysis
- Gradient-based feature detection
- Connectivity analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
class ExtremaInfo:
    """Information about detected extrema."""
    position: Tuple[int, int]
    magnitude: float
    importance_score: float
    local_density: float
    connectivity_score: float
    gradient_magnitude: float
    extrema_type: str  # 'high', 'low', 'saddle'


@dataclass
class PersistencePoint:
    """Point in persistence diagram."""
    birth: float
    death: float
    dimension: int
    feature_type: str  # 'component', 'hole', 'void'


@dataclass
class TopologicalSignature:
    """Topological signature of a network layer."""
    betti_numbers: List[int]
    persistence_entropy: float
    spectral_gap: float
    connectivity_density: float
    extrema_density: float
    topological_complexity: float


@profile_component(component_name="topological_analyzer", level=ProfilerLevel.DETAILED)
class TopologicalAnalyzer(BaseMetricAnalyzer, NetworkAnalyzerMixin, StatisticalUtilsMixin):
    """
    Specialized analyzer for topological properties of neural networks.
    
    Performs topological data analysis including extrema detection,
    persistence analysis, and connectivity analysis.
    """
    
    def __init__(self, threshold_config=None, patch_size: int = 8, min_density: float = 0.15):
        if threshold_config is None:
            from .base import ThresholdConfig
            threshold_config = ThresholdConfig()
        
        super().__init__(threshold_config)
        self.patch_size = patch_size
        self.min_density = min_density
        
    def compute_metrics(self, weight_matrix: torch.Tensor, 
                       target_patches: int = None) -> Dict[str, Any]:
        """
        Compute comprehensive topological metrics for a weight matrix.
        
        Args:
            weight_matrix: Weight matrix to analyze
            target_patches: Target number of patches for extrema detection
            
        Returns:
            Dictionary containing all topological metrics
        """
        start_time = time.time()
        
        # Validate input
        self._validate_tensor_input(weight_matrix, "weight_matrix", 2)
        
        # Check cache
        cache_key = self._cache_key(weight_matrix, target_patches)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        with profile_operation("topological_analysis", "topology"):
            # Core topological analysis
            extrema_info = self._detect_extrema(weight_matrix, target_patches)
            persistence_diagram = self._compute_persistence_diagram(weight_matrix)
            topological_signature = self._compute_topological_signature(weight_matrix)
            connectivity_analysis = self._analyze_connectivity(weight_matrix)
            
            # Compile metrics
            metrics = {
                'extrema_locations': [e.position for e in extrema_info],
                'extrema_info': extrema_info,
                'extrema_count': len(extrema_info),
                'extrema_density': len(extrema_info) / weight_matrix.numel(),
                'persistence_diagram': persistence_diagram,
                'topological_signature': topological_signature,
                'connectivity_analysis': connectivity_analysis,
                'gradient_statistics': self._compute_gradient_statistics(weight_matrix),
                'local_structure_analysis': self._analyze_local_structure(weight_matrix),
                'global_topology_metrics': self._compute_global_topology_metrics(weight_matrix)
            }
            
            # Update computation stats
            computation_time = time.time() - start_time
            self._computation_stats['total_calls'] += 1
            self._computation_stats['total_time'] += computation_time
            
            # Cache result
            self._cache_result(cache_key, metrics)
            
            return metrics
    
    def _detect_extrema(self, weight_matrix: torch.Tensor, 
                       target_patches: int = None) -> List[ExtremaInfo]:
        """Detect extrema locations for patch placement."""
        
        if target_patches is None:
            # Estimate reasonable number of patches
            total_params = weight_matrix.numel()
            if torch.cuda.is_available() and CUPY_AVAILABLE:
                target_patches = max(1, int(cp.sqrt(total_params) / 10))
            else:
                target_patches = max(1, int(np.sqrt(total_params) / 10))
        
        # Compute gradient magnitude
        grad_magnitude = self._compute_gradient_magnitude(weight_matrix)
        
        # Find local maxima
        local_maxima = self._find_local_maxima(grad_magnitude)
        
        # Score potential patch locations
        scored_locations = self._score_patch_locations(weight_matrix, local_maxima)
        
        # Select top locations and create ExtremaInfo objects
        selected = sorted(scored_locations, key=lambda x: x[2], reverse=True)
        extrema_info = []
        
        for (row, col), density, importance in selected[:target_patches]:
            # Extract local patch for detailed analysis
            patch = self._extract_patch(weight_matrix, row, col)
            
            extrema_info.append(ExtremaInfo(
                position=(row, col),
                magnitude=weight_matrix[row, col].abs().item(),
                importance_score=importance,
                local_density=density,
                connectivity_score=self._compute_connectivity_score(patch),
                gradient_magnitude=grad_magnitude[row, col].item(),
                extrema_type=self._classify_extrema_type(weight_matrix, row, col)
            ))
        
        return extrema_info
    
    def _compute_gradient_magnitude(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude across the weight matrix."""
        # Sobel operators for gradient computation
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
            dtype=weight_matrix.dtype, device=weight_matrix.device
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
            dtype=weight_matrix.dtype, device=weight_matrix.device
        )
        
        # Pad weight matrix
        padded = F.pad(weight_matrix, (1, 1, 1, 1), mode='reflect')
        
        # Compute gradients
        grad_x = F.conv2d(padded.unsqueeze(0).unsqueeze(0),
                          sobel_x.unsqueeze(0).unsqueeze(0), padding=0)
        grad_y = F.conv2d(padded.unsqueeze(0).unsqueeze(0),
                          sobel_y.unsqueeze(0).unsqueeze(0), padding=0)
        
        # Magnitude
        magnitude = torch.sqrt(grad_x.squeeze() ** 2 + grad_y.squeeze() ** 2)
        
        return magnitude
    
    def _find_local_maxima(self, gradient_magnitude: torch.Tensor) -> List[Tuple[int, int]]:
        """Find local maxima in gradient magnitude."""
        maxima = []
        h, w = gradient_magnitude.shape
        
        # Use a sliding window to find local maxima
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = gradient_magnitude[i, j]
                
                # Check if it's a local maximum
                neighborhood = gradient_magnitude[i-1:i+2, j-1:j+2]
                if center == neighborhood.max() and center > neighborhood.mean():
                    maxima.append((i, j))
        
        return maxima
    
    def _score_patch_locations(self, weight_matrix: torch.Tensor,
                              candidates: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], float, float]]:
        """Score potential patch locations."""
        scored = []
        h, w = weight_matrix.shape
        
        for row, col in candidates:
            # Check if patch fits
            if (row + self.patch_size <= h and col + self.patch_size <= w):
                # Extract potential patch
                patch = weight_matrix[row:row + self.patch_size, col:col + self.patch_size]
                
                # Compute density
                density = (patch.abs() > 1e-6).float().mean().item()
                
                # Compute importance score
                importance = self._compute_importance_score(patch)
                
                if density >= self.min_density:
                    scored.append(((row, col), density, importance))
        
        return scored
    
    def _compute_importance_score(self, patch: torch.Tensor) -> float:
        """Compute importance score for a patch."""
        # Combine multiple factors
        magnitude_score = patch.abs().mean().item()
        variance_score = patch.var().item()
        connectivity_score = self._compute_connectivity_score(patch)
        
        # Weighted combination
        importance = (0.4 * magnitude_score +
                     0.3 * variance_score +
                     0.3 * connectivity_score)
        
        return importance
    
    def _compute_connectivity_score(self, patch: torch.Tensor) -> float:
        """Compute connectivity score based on non-zero pattern."""
        # Convert to binary
        binary = (patch.abs() > 1e-6).float()
        
        # Count connected components (simplified)
        total_nonzero = binary.sum().item()
        if total_nonzero == 0:
            return 0.0
        
        # Estimate connectivity by checking neighbors
        connectivity = 0.0
        h, w = binary.shape
        
        for i in range(h):
            for j in range(w):
                if binary[i, j] > 0:
                    neighbors = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if (di != 0 or dj != 0) and 0 <= i + di < h and 0 <= j + dj < w:
                                if binary[i + di, j + dj] > 0:
                                    neighbors += 1
                    connectivity += neighbors
        
        return connectivity / (total_nonzero * 8) if total_nonzero > 0 else 0.0
    
    def _extract_patch(self, weight_matrix: torch.Tensor, row: int, col: int) -> torch.Tensor:
        """Extract patch at specified location."""
        h, w = weight_matrix.shape
        row_end = min(row + self.patch_size, h)
        col_end = min(col + self.patch_size, w)
        return weight_matrix[row:row_end, col:col_end]
    
    def _classify_extrema_type(self, weight_matrix: torch.Tensor, row: int, col: int) -> str:
        """Classify extrema type based on local structure."""
        # Extract 3x3 neighborhood
        h, w = weight_matrix.shape
        
        if row == 0 or row == h-1 or col == 0 or col == w-1:
            return 'boundary'
        
        neighborhood = weight_matrix[row-1:row+2, col-1:col+2]
        center = neighborhood[1, 1]
        
        # Count neighbors higher and lower than center
        higher = (neighborhood > center).sum().item()
        lower = (neighborhood < center).sum().item()
        
        if higher <= 2:
            return 'high'
        elif lower <= 2:
            return 'low'
        else:
            return 'saddle'
    
    def _compute_persistence_diagram(self, weight_matrix: torch.Tensor) -> List[PersistencePoint]:
        """Compute persistence diagram for network topology."""
        persistence_points = []
        
        # Compute persistence across threshold levels
        thresholds = torch.linspace(0, weight_matrix.abs().max().item(), 50)
        
        prev_components = 0
        prev_holes = 0
        
        for i, threshold in enumerate(thresholds):
            # Create binary matrix at threshold
            binary_matrix = (weight_matrix.abs() > threshold).float()
            
            # Count connected components (β₀)
            components = self._count_connected_components_binary(binary_matrix)
            
            # Estimate holes (β₁) - simplified
            holes = max(0, self._estimate_holes(binary_matrix))
            
            # Track births and deaths
            if i > 0:
                # Component deaths
                if components < prev_components:
                    for _ in range(prev_components - components):
                        persistence_points.append(PersistencePoint(
                            birth=thresholds[i-1].item(),
                            death=threshold.item(),
                            dimension=0,
                            feature_type='component'
                        ))
                
                # Hole births
                if holes > prev_holes:
                    for _ in range(holes - prev_holes):
                        persistence_points.append(PersistencePoint(
                            birth=threshold.item(),
                            death=float('inf'),  # Still alive
                            dimension=1,
                            feature_type='hole'
                        ))
                
                # Hole deaths
                if holes < prev_holes:
                    # Update existing holes with death time
                    alive_holes = [p for p in persistence_points 
                                  if p.feature_type == 'hole' and p.death == float('inf')]
                    for j in range(min(prev_holes - holes, len(alive_holes))):
                        alive_holes[j] = PersistencePoint(
                            birth=alive_holes[j].birth,
                            death=threshold.item(),
                            dimension=1,
                            feature_type='hole'
                        )
            
            prev_components = components
            prev_holes = holes
        
        return persistence_points
    
    def _count_connected_components_binary(self, binary_matrix: torch.Tensor) -> int:
        """Count connected components in binary matrix."""
        visited = torch.zeros_like(binary_matrix, dtype=torch.bool)
        components = 0
        h, w = binary_matrix.shape
        
        def dfs(i, j):
            if (i < 0 or i >= h or j < 0 or j >= w or 
                visited[i, j] or binary_matrix[i, j] == 0):
                return
            
            visited[i, j] = True
            # 8-connectivity
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di != 0 or dj != 0:
                        dfs(i + di, j + dj)
        
        for i in range(h):
            for j in range(w):
                if binary_matrix[i, j] > 0 and not visited[i, j]:
                    dfs(i, j)
                    components += 1
        
        return components
    
    def _estimate_holes(self, binary_matrix: torch.Tensor) -> int:
        """Estimate number of holes (β₁) in binary matrix."""
        # Simplified hole detection using Euler characteristic
        # χ = V - E + F, where β₁ = 2 - χ for 2D
        
        # Count vertices (non-zero pixels)
        vertices = binary_matrix.sum().item()
        
        if vertices == 0:
            return 0
        
        # Estimate edges (simplified)
        h, w = binary_matrix.shape
        edges = 0
        
        for i in range(h):
            for j in range(w):
                if binary_matrix[i, j] > 0:
                    # Count connections to neighbors
                    for di, dj in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w and binary_matrix[ni, nj] > 0:
                            edges += 1
        
        # Estimate faces (very simplified)
        faces = max(0, edges - vertices + 1)
        
        # Euler characteristic
        euler_char = vertices - edges + faces
        
        # β₁ = 2 - χ (for connected component)
        holes = max(0, 2 - euler_char)
        
        return holes
    
    def _compute_topological_signature(self, weight_matrix: torch.Tensor) -> TopologicalSignature:
        """Compute topological signature of the weight matrix."""
        
        # Compute Betti numbers
        binary_matrix = (weight_matrix.abs() > weight_matrix.abs().mean()).float()
        betti_0 = self._count_connected_components_binary(binary_matrix)
        betti_1 = self._estimate_holes(binary_matrix)
        betti_numbers = [betti_0, betti_1]
        
        # Persistence entropy
        persistence_diagram = self._compute_persistence_diagram(weight_matrix)
        persistence_entropy = self._compute_persistence_entropy(persistence_diagram)
        
        # Spectral gap
        eigenvals = torch.linalg.eigvals(weight_matrix @ weight_matrix.T).real
        eigenvals = torch.sort(eigenvals, descending=True)[0]
        spectral_gap = (eigenvals[0] - eigenvals[1]).item() if len(eigenvals) > 1 else 0
        
        # Connectivity density
        connectivity_density = (weight_matrix.abs() > 1e-6).float().mean().item()
        
        # Extrema density
        grad_magnitude = self._compute_gradient_magnitude(weight_matrix)
        extrema_density = (grad_magnitude > grad_magnitude.mean()).float().mean().item()
        
        # Topological complexity
        topological_complexity = sum(betti_numbers) + persistence_entropy
        
        return TopologicalSignature(
            betti_numbers=betti_numbers,
            persistence_entropy=persistence_entropy,
            spectral_gap=spectral_gap,
            connectivity_density=connectivity_density,
            extrema_density=extrema_density,
            topological_complexity=topological_complexity
        )
    
    def _compute_persistence_entropy(self, persistence_diagram: List[PersistencePoint]) -> float:
        """Compute persistence entropy."""
        if not persistence_diagram:
            return 0.0
        
        # Compute lifespans
        lifespans = []
        for point in persistence_diagram:
            if point.death != float('inf'):
                lifespan = point.death - point.birth
                if lifespan > 0:
                    lifespans.append(lifespan)
        
        if not lifespans:
            return 0.0
        
        # Normalize to probabilities
        total_lifespan = sum(lifespans)
        probs = [l / total_lifespan for l in lifespans]
        
        # Compute entropy
        return self._safe_entropy(torch.tensor(probs))
    
    def _analyze_connectivity(self, weight_matrix: torch.Tensor) -> Dict[str, Any]:
        """Analyze connectivity patterns in the weight matrix."""
        
        # Basic connectivity metrics
        total_elements = weight_matrix.numel()
        nonzero_elements = (weight_matrix.abs() > 1e-6).sum().item()
        sparsity = 1.0 - (nonzero_elements / total_elements)
        
        # Weight distribution
        weight_stats = {
            'mean': weight_matrix.mean().item(),
            'std': weight_matrix.std().item(),
            'min': weight_matrix.min().item(),
            'max': weight_matrix.max().item()
        }
        
        # Connectivity patterns
        row_connectivity = (weight_matrix.abs() > 1e-6).float().sum(dim=1)
        col_connectivity = (weight_matrix.abs() > 1e-6).float().sum(dim=0)
        
        return {
            'sparsity': sparsity,
            'density': 1.0 - sparsity,
            'weight_statistics': weight_stats,
            'row_connectivity_stats': {
                'mean': row_connectivity.mean().item(),
                'std': row_connectivity.std().item(),
                'min': row_connectivity.min().item(),
                'max': row_connectivity.max().item()
            },
            'col_connectivity_stats': {
                'mean': col_connectivity.mean().item(),
                'std': col_connectivity.std().item(),
                'min': col_connectivity.min().item(),
                'max': col_connectivity.max().item()
            },
            'connectivity_distribution': self._compute_percentiles(
                row_connectivity, [10, 25, 50, 75, 90]
            )
        }
    
    def _compute_gradient_statistics(self, weight_matrix: torch.Tensor) -> Dict[str, Any]:
        """Compute gradient-based statistics."""
        grad_magnitude = self._compute_gradient_magnitude(weight_matrix)
        
        return {
            'gradient_mean': grad_magnitude.mean().item(),
            'gradient_std': grad_magnitude.std().item(),
            'gradient_max': grad_magnitude.max().item(),
            'gradient_percentiles': self._compute_percentiles(
                grad_magnitude, [10, 25, 50, 75, 90, 95, 99]
            ),
            'high_gradient_ratio': (grad_magnitude > grad_magnitude.mean()).float().mean().item()
        }
    
    def _analyze_local_structure(self, weight_matrix: torch.Tensor) -> Dict[str, Any]:
        """Analyze local structure patterns."""
        h, w = weight_matrix.shape
        
        # Analyze patches across the matrix
        patch_stats = []
        patch_size = min(8, h // 4, w // 4)
        
        if patch_size >= 2:
            for i in range(0, h - patch_size + 1, patch_size):
                for j in range(0, w - patch_size + 1, patch_size):
                    patch = weight_matrix[i:i+patch_size, j:j+patch_size]
                    
                    patch_stats.append({
                        'position': (i, j),
                        'density': (patch.abs() > 1e-6).float().mean().item(),
                        'magnitude': patch.abs().mean().item(),
                        'variance': patch.var().item(),
                        'connectivity': self._compute_connectivity_score(patch)
                    })
        
        if patch_stats:
            use_gpu = torch.cuda.is_available() and CUPY_AVAILABLE
            if use_gpu:
                densities = cp.array([p['density'] for p in patch_stats])
                magnitudes = cp.array([p['magnitude'] for p in patch_stats])
                variances = cp.array([p['variance'] for p in patch_stats])
                connectivities = cp.array([p['connectivity'] for p in patch_stats])
                
                return {
                    'patch_count': len(patch_stats),
                    'avg_patch_density': densities.mean().get(),
                    'avg_patch_magnitude': magnitudes.mean().get(),
                    'avg_patch_variance': variances.mean().get(),
                    'avg_patch_connectivity': connectivities.mean().get(),
                    'patch_density_std': densities.std().get(),
                    'high_density_patches': (densities > 0.5).sum().get()
                }
            else:
                return {
                    'patch_count': len(patch_stats),
                    'avg_patch_density': np.mean([p['density'] for p in patch_stats]),
                    'avg_patch_magnitude': np.mean([p['magnitude'] for p in patch_stats]),
                    'avg_patch_variance': np.mean([p['variance'] for p in patch_stats]),
                    'avg_patch_connectivity': np.mean([p['connectivity'] for p in patch_stats]),
                    'patch_density_std': np.std([p['density'] for p in patch_stats]),
                    'high_density_patches': sum(1 for p in patch_stats if p['density'] > 0.5)
                }
        else:
            return {'patch_count': 0}
    
    def _compute_global_topology_metrics(self, weight_matrix: torch.Tensor) -> Dict[str, Any]:
        """Compute global topological metrics."""
        
        # Matrix rank and condition number
        try:
            U, S, Vt = torch.linalg.svd(weight_matrix)
            rank = torch.sum(S > 1e-6).item()
            condition_number = (S[0] / S[-1]).item() if len(S) > 0 and S[-1] > 1e-12 else float('inf')
        except:
            rank = min(weight_matrix.shape)
            condition_number = float('inf')
        
        # Frobenius norm and nuclear norm
        frobenius_norm = torch.norm(weight_matrix, p='fro').item()
        nuclear_norm = torch.sum(S).item() if 'S' in locals() else frobenius_norm
        
        return {
            'matrix_rank': rank,
            'condition_number': condition_number,
            'frobenius_norm': frobenius_norm,
            'nuclear_norm': nuclear_norm,
            'rank_ratio': rank / min(weight_matrix.shape),
            'spectral_radius': S[0].item() if 'S' in locals() and len(S) > 0 else 0
        }
    
    def find_optimal_patch_locations(self, weight_matrix: torch.Tensor, 
                                   num_patches: int) -> List[Tuple[int, int]]:
        """Find optimal locations for patch placement."""
        metrics = self.compute_metrics(weight_matrix, target_patches=num_patches)
        return metrics['extrema_locations']
    
    def compute_patch_importance(self, weight_matrix: torch.Tensor, 
                               location: Tuple[int, int]) -> Dict[str, float]:
        """Compute importance metrics for a specific patch location."""
        row, col = location
        patch = self._extract_patch(weight_matrix, row, col)
        
        return {
            'importance_score': self._compute_importance_score(patch),
            'density': (patch.abs() > 1e-6).float().mean().item(),
            'connectivity': self._compute_connectivity_score(patch),
            'magnitude': patch.abs().mean().item(),
            'variance': patch.var().item()
        }


# Factory function for easy creation
def create_topological_analyzer(patch_size: int = 8, min_density: float = 0.15, 
                               **kwargs) -> TopologicalAnalyzer:
    """
    Factory function to create topological analyzer.
    
    Args:
        patch_size: Size of patches for analysis
        min_density: Minimum density threshold
        **kwargs: Additional configuration options
        
    Returns:
        Configured TopologicalAnalyzer
    """
    return TopologicalAnalyzer(patch_size=patch_size, min_density=min_density, **kwargs)


# Export main classes
__all__ = [
    'ExtremaInfo',
    'PersistencePoint', 
    'TopologicalSignature',
    'TopologicalAnalyzer',
    'create_topological_analyzer'
]
