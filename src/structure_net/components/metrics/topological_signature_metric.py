"""
Topological signature metric component.

This component computes topological invariants and signatures that characterize
the structural properties of neural network weight matrices.
"""

from typing import Dict, Any, Union, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


class TopologicalSignatureMetric(BaseMetric):
    """
    Computes topological signatures and invariants.
    
    This includes Betti numbers, Euler characteristic, and
    other topological properties that are invariant under
    continuous deformations.
    """
    
    def __init__(self, resolution: int = 20, max_dimension: int = 2,
                 name: str = None):
        """
        Initialize topological signature metric.
        
        Args:
            resolution: Number of threshold levels for filtration
            max_dimension: Maximum homological dimension to compute
            name: Optional custom name
        """
        super().__init__(name or "TopologicalSignatureMetric")
        self.resolution = resolution
        self.max_dimension = max_dimension
        self._measurement_schema = {
            "betti_numbers": dict,
            "euler_characteristic": float,
            "topological_complexity": float,
            "homological_dimension": int,
            "structural_signature": list,
            "topological_entropy": float
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"weight_matrix"},
            provided_outputs={
                "metrics.betti_numbers",
                "metrics.euler_characteristic",
                "metrics.topological_complexity",
                "metrics.homological_dimension",
                "metrics.structural_signature",
                "metrics.topological_entropy"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.HIGH,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def _compute_metric(self, target: Union[ILayer, IModel], 
                       context: EvolutionContext) -> Dict[str, Any]:
        """
        Compute topological signature metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'weight_matrix'
            
        Returns:
            Dictionary containing topological measurements
        """
        # Get weight matrix
        weight_matrix = context.get('weight_matrix')
        if weight_matrix is None:
            if isinstance(target, ILayer):
                weight_matrix = self._extract_weight_matrix(target)
            else:
                raise ValueError("TopologicalSignatureMetric requires 'weight_matrix' in context or a layer target")
        
        if weight_matrix.dim() != 2:
            raise ValueError("Weight matrix must be 2D")
        
        # Compute topological invariants
        betti_numbers = self._compute_betti_numbers(weight_matrix)
        euler_char = self._compute_euler_characteristic(betti_numbers)
        complexity = self._compute_topological_complexity(weight_matrix, betti_numbers)
        homological_dim = self._compute_homological_dimension(betti_numbers)
        signature = self._compute_structural_signature(weight_matrix)
        entropy = self._compute_topological_entropy(weight_matrix)
        
        self.log(logging.DEBUG, 
                f"Topological signature: Betti={betti_numbers}, "
                f"Euler={euler_char}, complexity={complexity:.3f}")
        
        return {
            "betti_numbers": betti_numbers,
            "euler_characteristic": euler_char,
            "topological_complexity": complexity,
            "homological_dimension": homological_dim,
            "structural_signature": signature,
            "topological_entropy": entropy
        }
    
    def _extract_weight_matrix(self, layer: ILayer) -> torch.Tensor:
        """Extract weight matrix from layer."""
        for attr_name in ['weight', 'linear.weight', 'W']:
            if hasattr(layer, attr_name):
                weight = getattr(layer, attr_name)
                if isinstance(weight, torch.Tensor) and weight.dim() >= 2:
                    return weight.flatten(0, -2) if weight.dim() > 2 else weight
        
        raise ValueError(f"Could not extract weight matrix from layer")
    
    def _compute_betti_numbers(self, weight_matrix: torch.Tensor) -> Dict[str, int]:
        """
        Compute Betti numbers for different dimensions.
        
        β₀: Number of connected components
        β₁: Number of 1-dimensional holes
        β₂: Number of 2-dimensional voids
        """
        betti = {}
        
        # Create threshold filtration
        abs_weights = weight_matrix.abs()
        thresholds = torch.linspace(
            abs_weights.min().item(),
            abs_weights.max().item(),
            self.resolution
        )
        
        # Compute Betti numbers at middle threshold
        mid_threshold = thresholds[self.resolution // 2]
        binary_matrix = (abs_weights > mid_threshold).float()
        
        # β₀: Connected components
        betti['beta_0'] = self._count_connected_components(binary_matrix)
        
        # β₁: 1-dimensional holes (simplified computation)
        if self.max_dimension >= 1:
            betti['beta_1'] = self._estimate_holes(binary_matrix)
        
        # β₂: 2-dimensional voids (simplified computation)
        if self.max_dimension >= 2:
            betti['beta_2'] = self._estimate_voids(binary_matrix)
        
        return betti
    
    def _count_connected_components(self, binary_matrix: torch.Tensor) -> int:
        """Count connected components using flood fill."""
        if binary_matrix.sum() == 0:
            return 0
        
        visited = torch.zeros_like(binary_matrix, dtype=torch.bool)
        h, w = binary_matrix.shape
        components = 0
        
        def flood_fill(i, j):
            """Flood fill from position (i, j)."""
            stack = [(i, j)]
            while stack:
                ci, cj = stack.pop()
                if (0 <= ci < h and 0 <= cj < w and 
                    not visited[ci, cj] and binary_matrix[ci, cj] > 0):
                    visited[ci, cj] = True
                    # 4-connectivity
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        stack.append((ci + di, cj + dj))
        
        # Find all components
        for i in range(h):
            for j in range(w):
                if binary_matrix[i, j] > 0 and not visited[i, j]:
                    flood_fill(i, j)
                    components += 1
        
        return components
    
    def _estimate_holes(self, binary_matrix: torch.Tensor) -> int:
        """
        Estimate number of 1-dimensional holes.
        
        Uses a simplified approach based on Euler characteristic.
        """
        if binary_matrix.sum() == 0:
            return 0
        
        h, w = binary_matrix.shape
        
        # Count vertices (non-zero entries)
        vertices = binary_matrix.sum().item()
        
        # Count edges (4-connectivity)
        edges = 0
        for i in range(h):
            for j in range(w):
                if binary_matrix[i, j] > 0:
                    # Right neighbor
                    if j + 1 < w and binary_matrix[i, j + 1] > 0:
                        edges += 1
                    # Bottom neighbor
                    if i + 1 < h and binary_matrix[i + 1, j] > 0:
                        edges += 1
        
        # Count faces (2x2 squares)
        faces = 0
        for i in range(h - 1):
            for j in range(w - 1):
                if (binary_matrix[i, j] > 0 and binary_matrix[i+1, j] > 0 and
                    binary_matrix[i, j+1] > 0 and binary_matrix[i+1, j+1] > 0):
                    faces += 1
        
        # Euler characteristic for planar graph
        euler_char = int(vertices - edges + faces)
        
        # Get number of components
        num_components = self._count_connected_components(binary_matrix)
        
        # Estimate holes: β₁ = 1 - χ + β₀
        holes = max(0, 1 - euler_char + num_components)
        
        return holes
    
    def _estimate_voids(self, binary_matrix: torch.Tensor) -> int:
        """
        Estimate number of 2-dimensional voids.
        
        This is a simplified heuristic for 2D weight matrices.
        """
        # For 2D matrices, we can't have true 3D voids
        # Instead, we estimate based on the complexity of the hole structure
        
        holes = self._estimate_holes(binary_matrix)
        
        # Heuristic: complex hole patterns might indicate higher-dimensional structure
        if holes > 3:
            return max(0, holes // 3)
        else:
            return 0
    
    def _compute_euler_characteristic(self, betti_numbers: Dict[str, int]) -> float:
        """
        Compute Euler characteristic from Betti numbers.
        
        χ = β₀ - β₁ + β₂ - ...
        """
        euler = 0
        for i in range(self.max_dimension + 1):
            key = f'beta_{i}'
            if key in betti_numbers:
                euler += ((-1) ** i) * betti_numbers[key]
        
        return float(euler)
    
    def _compute_topological_complexity(self, weight_matrix: torch.Tensor,
                                      betti_numbers: Dict[str, int]) -> float:
        """
        Compute a measure of topological complexity.
        
        Combines multiple factors including Betti numbers and
        multi-scale topological features.
        """
        # Base complexity from Betti numbers
        betti_sum = sum(betti_numbers.values())
        
        # Multi-scale complexity
        multi_scale_features = self._compute_multiscale_features(weight_matrix)
        
        # Normalize by matrix size
        size_factor = np.sqrt(weight_matrix.numel())
        
        # Combined complexity
        complexity = (betti_sum + multi_scale_features) / size_factor
        
        return complexity
    
    def _compute_multiscale_features(self, weight_matrix: torch.Tensor) -> float:
        """Compute topological features at multiple scales."""
        abs_weights = weight_matrix.abs()
        thresholds = torch.linspace(
            abs_weights.min().item(),
            abs_weights.max().item(),
            self.resolution
        )
        
        # Track how Betti numbers change across scales
        betti_variations = []
        prev_betti = None
        
        for threshold in thresholds:
            binary_matrix = (abs_weights > threshold).float()
            
            # Compute Betti_0 at this scale
            betti_0 = self._count_connected_components(binary_matrix)
            
            if prev_betti is not None:
                variation = abs(betti_0 - prev_betti)
                betti_variations.append(variation)
            
            prev_betti = betti_0
        
        # Total variation as complexity measure
        return sum(betti_variations) if betti_variations else 0.0
    
    def _compute_homological_dimension(self, betti_numbers: Dict[str, int]) -> int:
        """
        Compute the homological dimension.
        
        The highest dimension with non-zero Betti number.
        """
        max_dim = -1
        for i in range(self.max_dimension + 1):
            key = f'beta_{i}'
            if key in betti_numbers and betti_numbers[key] > 0:
                max_dim = i
        
        return max_dim
    
    def _compute_structural_signature(self, weight_matrix: torch.Tensor) -> List[float]:
        """
        Compute a structural signature vector.
        
        This signature captures key topological and geometric features.
        """
        signature = []
        
        # Weight statistics
        abs_weights = weight_matrix.abs()
        signature.append(abs_weights.mean().item())
        signature.append(abs_weights.std().item())
        signature.append(abs_weights.max().item())
        
        # Spectral features (top eigenvalues)
        try:
            # Compute singular values (more stable than eigenvalues)
            U, S, V = torch.svd(weight_matrix)
            top_singular_values = S[:min(5, len(S))]
            signature.extend(top_singular_values.tolist())
            
            # Spectral gap
            if len(S) > 1:
                spectral_gap = (S[0] - S[1]).item()
                signature.append(spectral_gap)
            else:
                signature.append(0.0)
        except:
            # Fallback if SVD fails
            signature.extend([0.0] * 6)
        
        # Ensure fixed length
        while len(signature) < 10:
            signature.append(0.0)
        
        return signature[:10]  # Fixed length of 10
    
    def _compute_topological_entropy(self, weight_matrix: torch.Tensor) -> float:
        """
        Compute topological entropy.
        
        This measures the complexity of the topological structure.
        """
        # Create filtration
        abs_weights = weight_matrix.abs()
        thresholds = torch.linspace(
            abs_weights.min().item(),
            abs_weights.max().item(),
            self.resolution
        )
        
        # Track connected components across filtration
        component_counts = []
        for threshold in thresholds:
            binary_matrix = (abs_weights > threshold).float()
            num_components = self._count_connected_components(binary_matrix)
            if num_components > 0:
                component_counts.append(num_components)
        
        if not component_counts:
            return 0.0
        
        # Compute entropy from distribution of components
        total_components = sum(component_counts)
        probabilities = [c / total_components for c in component_counts]
        
        # Shannon entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy