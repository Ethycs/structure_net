"""
Spectral graph metric component.

This component computes spectral properties of neural network graphs including
eigenvalues, spectral gap, and algebraic connectivity.
"""

from typing import Dict, Any, Union, Optional, List
import torch
import logging
import numpy as np

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


class SpectralGraphMetric(BaseMetric):
    """
    Computes spectral properties of neural network graphs.
    
    Spectral analysis provides insights into network connectivity,
    robustness, and information flow capacity.
    """
    
    def __init__(self, num_eigenvalues: int = 10, name: str = None):
        """
        Initialize spectral graph metric.
        
        Args:
            num_eigenvalues: Number of eigenvalues to compute
            name: Optional custom name
        """
        super().__init__(name or "SpectralGraphMetric")
        self.num_eigenvalues = num_eigenvalues
        self._measurement_schema = {
            "largest_eigenvalues": list,
            "spectral_gap": float,
            "algebraic_connectivity": float,
            "spectral_radius": float,
            "graph_energy": float
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"metrics.graph"},
            provided_outputs={
                "metrics.largest_eigenvalues",
                "metrics.spectral_gap",
                "metrics.algebraic_connectivity",
                "metrics.spectral_radius",
                "metrics.graph_energy"
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
        Compute spectral graph metrics.
        
        Args:
            target: Not used directly
            context: Must contain graph from GraphStructureMetric
            
        Returns:
            Dictionary containing spectral measurements
        """
        if not NETWORKX_AVAILABLE:
            self.log(logging.WARNING, "NetworkX not available, returning empty metrics")
            return self._empty_metrics()
        
        # Get graph
        G = context.get('metrics.graph')
        if G is None or not isinstance(G, nx.DiGraph):
            return self._empty_metrics()
        
        if G.number_of_nodes() < 2:
            return self._empty_metrics()
        
        try:
            # Convert to adjacency matrix
            adj_matrix = nx.adjacency_matrix(G).astype(float)
            
            # For directed graphs, use symmetrized adjacency for some metrics
            adj_symmetric = (adj_matrix + adj_matrix.T) / 2
            
            # Compute eigenvalues of adjacency matrix
            k = min(self.num_eigenvalues, G.number_of_nodes() - 1)
            eigenvalues = self._compute_top_eigenvalues(adj_matrix, k)
            
            # Spectral radius (largest eigenvalue magnitude)
            spectral_radius = max(abs(eigenvalues)) if eigenvalues else 0
            
            # Spectral gap (difference between first and second eigenvalue)
            spectral_gap = 0.0
            if len(eigenvalues) >= 2:
                sorted_eigs = sorted(eigenvalues, key=abs, reverse=True)
                spectral_gap = abs(sorted_eigs[0]) - abs(sorted_eigs[1])
            
            # Algebraic connectivity (second smallest eigenvalue of Laplacian)
            # For directed graph, use symmetric version
            laplacian = nx.laplacian_matrix(G.to_undirected())
            algebraic_connectivity = self._compute_algebraic_connectivity(laplacian)
            
            # Graph energy (sum of absolute eigenvalues)
            graph_energy = sum(abs(e) for e in eigenvalues)
            
            self.log(logging.DEBUG, 
                    f"Spectral: radius={spectral_radius:.3f}, gap={spectral_gap:.3f}, "
                    f"connectivity={algebraic_connectivity:.3f}")
            
            return {
                "largest_eigenvalues": eigenvalues[:10],  # Top 10
                "spectral_gap": spectral_gap,
                "algebraic_connectivity": algebraic_connectivity,
                "spectral_radius": spectral_radius,
                "graph_energy": graph_energy
            }
            
        except Exception as e:
            self.log(logging.WARNING, f"Spectral computation failed: {e}")
            return self._empty_metrics()
    
    def _compute_top_eigenvalues(self, matrix, k: int) -> List[float]:
        """Compute top k eigenvalues of sparse matrix."""
        try:
            from scipy.sparse.linalg import eigs
            
            # Handle small matrices
            if matrix.shape[0] <= k + 1:
                # Convert to dense for small matrices
                dense_matrix = matrix.toarray()
                eigenvalues = np.linalg.eigvals(dense_matrix)
                return sorted(eigenvalues.real, key=abs, reverse=True)[:k]
            
            # For larger matrices, use sparse eigenvalue computation
            eigenvalues, _ = eigs(matrix, k=k, which='LM')  # Largest magnitude
            return sorted(eigenvalues.real, key=abs, reverse=True)
            
        except Exception as e:
            self.log(logging.DEBUG, f"Eigenvalue computation failed: {e}")
            return []
    
    def _compute_algebraic_connectivity(self, laplacian) -> float:
        """Compute algebraic connectivity (second smallest eigenvalue of Laplacian)."""
        try:
            from scipy.sparse.linalg import eigsh
            
            # For small matrices
            if laplacian.shape[0] <= 3:
                dense_lap = laplacian.toarray()
                eigenvalues = np.linalg.eigvalsh(dense_lap)
                return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
            
            # Compute smallest eigenvalues
            eigenvalues, _ = eigsh(laplacian, k=2, which='SM')
            return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
            
        except Exception as e:
            self.log(logging.DEBUG, f"Algebraic connectivity computation failed: {e}")
            return 0.0
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics when computation cannot proceed."""
        return {
            "largest_eigenvalues": [],
            "spectral_gap": 0.0,
            "algebraic_connectivity": 0.0,
            "spectral_radius": 0.0,
            "graph_energy": 0.0
        }