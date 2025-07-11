"""
Betti number metric component.

This component computes topological invariants (Betti numbers) for
neural network architectures and weight matrices.
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


class BettiNumberMetric(BaseMetric):
    """
    Computes Betti numbers and other topological invariants.
    
    Betti numbers are topological invariants that describe the number of
    k-dimensional holes in a topological space. For neural networks:
    - β₀: Number of connected components
    - β₁: Number of 1-dimensional holes (loops)
    - β₂: Number of 2-dimensional voids
    """
    
    def __init__(self, tolerance: float = 1e-6, max_dimension: int = 2,
                 compute_persistence: bool = False, name: str = None):
        """
        Initialize Betti number metric.
        
        Args:
            tolerance: Threshold for connectivity
            max_dimension: Maximum Betti number dimension to compute
            compute_persistence: Whether to compute persistence diagrams
            name: Optional custom name
        """
        super().__init__(name or "BettiNumberMetric")
        self.tolerance = tolerance
        self.max_dimension = max_dimension
        self.compute_persistence = compute_persistence
        self._measurement_schema = {
            "betti_numbers": list,
            "euler_characteristic": int,
            "connected_components": int,
            "topological_complexity": float,
            "persistence_diagram": list
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
                "metrics.connected_components",
                "metrics.topological_complexity",
                "metrics.persistence_diagram"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.HIGH,  # Can be expensive for large networks
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def _compute_metric(self, target: Union[ILayer, IModel], 
                       context: EvolutionContext) -> Dict[str, Any]:
        """
        Compute Betti number metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'weight_matrix' data
            
        Returns:
            Dictionary containing topological measurements
        """
        # Get weight matrix from context
        weight_matrix = context.get('weight_matrix')
        if weight_matrix is None:
            # Try to extract from layer/model
            if isinstance(target, ILayer):
                weight_matrix = self._extract_weight_matrix(target)
            elif isinstance(target, IModel):
                # For models, compute adjacency matrix
                weight_matrix = self._compute_model_adjacency(target)
            else:
                raise ValueError("BettiNumberMetric requires 'weight_matrix' in context")
        
        return self._compute_topological_invariants(weight_matrix)
    
    def _extract_weight_matrix(self, layer: ILayer) -> torch.Tensor:
        """Extract weight matrix from a layer."""
        for attr_name in ['weight', 'linear.weight', 'W']:
            if hasattr(layer, attr_name):
                weight = getattr(layer, attr_name)
                if isinstance(weight, torch.Tensor):
                    return weight
        
        for name, param in layer.named_parameters():
            if 'weight' in name:
                return param
        
        raise ValueError(f"Could not extract weight matrix from layer {layer.name}")
    
    def _compute_model_adjacency(self, model: IModel) -> torch.Tensor:
        """Compute adjacency matrix for entire model."""
        # This is a simplified version - in practice, you'd want more sophisticated
        # graph construction based on actual architecture
        layers = []
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                layers.append(module.weight)
        
        if not layers:
            raise ValueError("No weight matrices found in model")
        
        # Create block diagonal adjacency matrix
        total_size = sum(w.shape[0] for w in layers)
        adjacency = torch.zeros(total_size, total_size)
        
        offset = 0
        for weight in layers:
            m, n = weight.shape[:2]
            if weight.dim() > 2:
                weight = weight.flatten(1)
            adjacency[offset:offset+m, offset:offset+m] = torch.abs(weight @ weight.T)
            offset += m
        
        return adjacency
    
    def _compute_topological_invariants(self, weight_matrix: torch.Tensor) -> Dict[str, Any]:
        """
        Compute Betti numbers and related topological invariants.
        
        Args:
            weight_matrix: Weight or adjacency matrix
            
        Returns:
            Dictionary with topological metrics
        """
        # Convert to adjacency matrix if needed
        if weight_matrix.dim() > 2:
            weight_matrix = weight_matrix.flatten(0, -2)
        
        # Ensure square matrix for graph analysis
        if weight_matrix.shape[0] != weight_matrix.shape[1]:
            # Create adjacency from weight matrix
            adjacency = self._weight_to_adjacency(weight_matrix)
        else:
            adjacency = weight_matrix.abs()
        
        # Compute Betti numbers
        betti_0 = self._compute_betti_0(adjacency)
        betti_1 = self._compute_betti_1(adjacency) if self.max_dimension >= 1 else 0
        betti_2 = self._compute_betti_2(adjacency) if self.max_dimension >= 2 else 0
        
        betti_numbers = [betti_0, betti_1, betti_2][:self.max_dimension + 1]
        
        # Euler characteristic: alternating sum of Betti numbers
        euler_characteristic = sum((-1)**i * b for i, b in enumerate(betti_numbers))
        
        # Topological complexity: sum of all Betti numbers
        topological_complexity = sum(betti_numbers) / len(betti_numbers)
        
        # Persistence diagram (if requested)
        persistence_diagram = []
        if self.compute_persistence:
            persistence_diagram = self._compute_persistence_diagram(adjacency)
        
        self.log(logging.DEBUG, 
                f"Topological invariants: Betti={betti_numbers}, Euler={euler_characteristic}")
        
        return {
            "betti_numbers": betti_numbers,
            "euler_characteristic": euler_characteristic,
            "connected_components": betti_0,
            "topological_complexity": topological_complexity,
            "persistence_diagram": persistence_diagram
        }
    
    def _weight_to_adjacency(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """Convert weight matrix to adjacency matrix."""
        # Create bipartite graph adjacency
        m, n = weight_matrix.shape
        adjacency = torch.zeros(m + n, m + n, device=weight_matrix.device)
        
        # Upper right block: weight matrix
        adjacency[:m, m:] = weight_matrix.abs()
        # Lower left block: transpose
        adjacency[m:, :m] = weight_matrix.abs().T
        
        return adjacency
    
    def _compute_betti_0(self, adjacency: torch.Tensor) -> int:
        """
        Compute β₀: number of connected components.
        
        Uses iterative matrix multiplication to find connected components.
        """
        n = adjacency.shape[0]
        if n == 0:
            return 0
        
        # Binary adjacency matrix
        adj_binary = (adjacency > self.tolerance).float()
        
        # Add self-loops
        connectivity = adj_binary + torch.eye(n, device=adjacency.device)
        
        # Compute transitive closure
        prev_connectivity = torch.zeros_like(connectivity)
        max_iterations = min(n, 20)  # Limit iterations
        
        for _ in range(max_iterations):
            if torch.allclose(connectivity, prev_connectivity):
                break
            prev_connectivity = connectivity.clone()
            connectivity = torch.clamp(connectivity @ connectivity, 0, 1)
        
        # Count unique connectivity patterns (connected components)
        # Each row represents reachability from that node
        unique_patterns = []
        for i in range(n):
            pattern = connectivity[i]
            is_new = True
            for existing in unique_patterns:
                if torch.allclose(pattern, existing):
                    is_new = False
                    break
            if is_new:
                unique_patterns.append(pattern)
        
        return len(unique_patterns)
    
    def _compute_betti_1(self, adjacency: torch.Tensor) -> int:
        """
        Compute β₁: number of 1-dimensional holes (cycles).
        
        Simplified computation using cycle rank formula:
        β₁ = |E| - |V| + β₀
        """
        n = adjacency.shape[0]
        
        # Count edges (non-zero entries above diagonal)
        adj_binary = (adjacency > self.tolerance).float()
        num_edges = torch.sum(torch.triu(adj_binary, diagonal=1)).item()
        
        # Get number of vertices and components
        num_vertices = n
        beta_0 = self._compute_betti_0(adjacency)
        
        # Cycle rank formula
        beta_1 = int(num_edges - num_vertices + beta_0)
        
        return max(0, beta_1)  # Ensure non-negative
    
    def _compute_betti_2(self, adjacency: torch.Tensor) -> int:
        """
        Compute β₂: number of 2-dimensional voids.
        
        This is a simplified heuristic for neural networks.
        """
        # For neural networks, β₂ is typically 0 unless we have
        # very specific architectural patterns
        
        # Heuristic: look for clique-like structures
        n = adjacency.shape[0]
        if n < 4:  # Need at least 4 nodes for a 2-void
            return 0
        
        adj_binary = (adjacency > self.tolerance).float()
        
        # Count 4-cliques (simplest 2-dimensional void)
        clique_count = 0
        for i in range(n-3):
            for j in range(i+1, n-2):
                for k in range(j+1, n-1):
                    for l in range(k+1, n):
                        # Check if all pairs are connected
                        if (adj_binary[i,j] * adj_binary[i,k] * adj_binary[i,l] *
                            adj_binary[j,k] * adj_binary[j,l] * adj_binary[k,l] > 0):
                            clique_count += 1
        
        # Very rough heuristic: some fraction of 4-cliques form voids
        beta_2 = clique_count // 10
        
        return beta_2
    
    def _compute_persistence_diagram(self, adjacency: torch.Tensor) -> List[Tuple[float, float]]:
        """
        Compute persistence diagram for topological features.
        
        Returns list of (birth, death) pairs for topological features.
        """
        persistence = []
        
        # Simplified persistence: use eigenvalues as filtration
        try:
            eigenvalues = torch.linalg.eigvalsh(adjacency)
            eigenvalues = torch.sort(eigenvalues, descending=True)[0]
            
            # Each eigenvalue represents birth/death of a feature
            for i in range(min(10, len(eigenvalues) - 1)):
                birth = eigenvalues[i].item()
                death = eigenvalues[i + 1].item()
                if abs(birth - death) > self.tolerance:
                    persistence.append((birth, death))
        
        except Exception as e:
            self.log(logging.WARNING, f"Persistence computation failed: {e}")
        
        return persistence