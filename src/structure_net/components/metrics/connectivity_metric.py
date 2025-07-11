"""
Connectivity metric component.

This component analyzes connectivity patterns in neural network weight matrices,
identifying structural relationships and connection density.
"""

from typing import Dict, Any, Union, Optional, List, Tuple
import torch
import torch.nn as nn
import logging
import numpy as np

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


class ConnectivityMetric(BaseMetric):
    """
    Analyzes connectivity patterns in weight matrices.
    
    This metric examines how neurons are connected to each other,
    identifying patterns like hubs, clusters, and sparse connections.
    """
    
    def __init__(self, connectivity_threshold: float = 0.01,
                 percentile_threshold: float = 0.9,
                 name: str = None):
        """
        Initialize connectivity metric.
        
        Args:
            connectivity_threshold: Minimum weight magnitude for connection
            percentile_threshold: Percentile for significant connections
            name: Optional custom name
        """
        super().__init__(name or "ConnectivityMetric")
        self.connectivity_threshold = connectivity_threshold
        self.percentile_threshold = percentile_threshold
        self._measurement_schema = {
            "connectivity_density": float,
            "average_degree": float,
            "hub_neurons": list,
            "clustering_coefficient": float,
            "connectivity_distribution": dict,
            "sparsity": float
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"weight_matrix"},
            provided_outputs={
                "metrics.connectivity_density",
                "metrics.average_degree",
                "metrics.hub_neurons",
                "metrics.clustering_coefficient",
                "metrics.connectivity_distribution",
                "metrics.sparsity"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def _compute_metric(self, target: Union[ILayer, IModel], 
                       context: EvolutionContext) -> Dict[str, Any]:
        """
        Compute connectivity metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'weight_matrix'
            
        Returns:
            Dictionary containing connectivity measurements
        """
        # Get weight matrix
        weight_matrix = context.get('weight_matrix')
        if weight_matrix is None:
            if isinstance(target, ILayer):
                weight_matrix = self._extract_weight_matrix(target)
            else:
                raise ValueError("ConnectivityMetric requires 'weight_matrix' in context or a layer target")
        
        if weight_matrix.dim() != 2:
            raise ValueError("Weight matrix must be 2D")
        
        # Create adjacency matrix
        adj_matrix = self._create_adjacency_matrix(weight_matrix)
        
        # Compute connectivity metrics
        density = self._compute_density(adj_matrix)
        avg_degree = self._compute_average_degree(adj_matrix)
        hub_neurons = self._identify_hub_neurons(adj_matrix, weight_matrix)
        clustering_coeff = self._compute_clustering_coefficient(adj_matrix)
        conn_distribution = self._analyze_connectivity_distribution(adj_matrix)
        sparsity = self._compute_sparsity(weight_matrix)
        
        self.log(logging.DEBUG, 
                f"Connectivity: density={density:.3f}, avg_degree={avg_degree:.2f}, "
                f"hubs={len(hub_neurons)}, clustering={clustering_coeff:.3f}")
        
        return {
            "connectivity_density": density,
            "average_degree": avg_degree,
            "hub_neurons": hub_neurons,
            "clustering_coefficient": clustering_coeff,
            "connectivity_distribution": conn_distribution,
            "sparsity": sparsity
        }
    
    def _extract_weight_matrix(self, layer: ILayer) -> torch.Tensor:
        """Extract weight matrix from layer."""
        for attr_name in ['weight', 'linear.weight', 'W']:
            if hasattr(layer, attr_name):
                weight = getattr(layer, attr_name)
                if isinstance(weight, torch.Tensor) and weight.dim() >= 2:
                    return weight.flatten(0, -2) if weight.dim() > 2 else weight
        
        raise ValueError(f"Could not extract weight matrix from layer")
    
    def _create_adjacency_matrix(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """Create binary adjacency matrix from weights."""
        # Use dynamic threshold based on percentile
        abs_weights = weight_matrix.abs()
        threshold = max(
            self.connectivity_threshold,
            torch.quantile(abs_weights, self.percentile_threshold).item()
        )
        
        # Create adjacency matrix
        adj_matrix = (abs_weights > threshold).float()
        
        return adj_matrix
    
    def _compute_density(self, adj_matrix: torch.Tensor) -> float:
        """Compute connection density."""
        n_rows, n_cols = adj_matrix.shape
        max_connections = n_rows * n_cols
        actual_connections = adj_matrix.sum().item()
        
        return actual_connections / max_connections if max_connections > 0 else 0.0
    
    def _compute_average_degree(self, adj_matrix: torch.Tensor) -> float:
        """Compute average node degree."""
        # Out-degree (row sums)
        out_degrees = adj_matrix.sum(dim=1)
        # In-degree (column sums)
        in_degrees = adj_matrix.sum(dim=0)
        
        # Average total degree
        avg_out = out_degrees.mean().item()
        avg_in = in_degrees.mean().item()
        
        return (avg_out + avg_in) / 2
    
    def _identify_hub_neurons(self, adj_matrix: torch.Tensor, 
                             weight_matrix: torch.Tensor) -> List[Dict[str, Any]]:
        """Identify hub neurons (highly connected nodes)."""
        # Compute degrees
        out_degrees = adj_matrix.sum(dim=1)
        in_degrees = adj_matrix.sum(dim=0)
        total_degrees = out_degrees + in_degrees
        
        # Find hubs (top 5% by degree)
        threshold = torch.quantile(total_degrees, 0.95).item()
        hub_indices = torch.where(total_degrees > threshold)[0]
        
        hubs = []
        for idx in hub_indices:
            # Compute hub strength (sum of connected weights)
            out_strength = weight_matrix[idx, :].abs().sum().item()
            in_strength = weight_matrix[:, idx].abs().sum().item()
            
            hubs.append({
                'index': idx.item(),
                'total_degree': total_degrees[idx].item(),
                'out_degree': out_degrees[idx].item(),
                'in_degree': in_degrees[idx].item(),
                'strength': out_strength + in_strength
            })
        
        # Sort by total degree
        hubs.sort(key=lambda x: x['total_degree'], reverse=True)
        
        return hubs
    
    def _compute_clustering_coefficient(self, adj_matrix: torch.Tensor) -> float:
        """
        Compute local clustering coefficient.
        
        For directed graphs, this measures the fraction of triangles
        among connected triples.
        """
        n = adj_matrix.shape[0]
        
        if n < 3:
            return 0.0
        
        # Compute number of triangles
        # A triangle exists when i->j, j->k, and i->k all exist
        triangles = 0
        possible_triangles = 0
        
        for i in range(n):
            # Find neighbors of i
            out_neighbors = torch.where(adj_matrix[i] > 0)[0]
            
            if len(out_neighbors) < 2:
                continue
            
            # Count triangles
            for j_idx in range(len(out_neighbors)):
                for k_idx in range(j_idx + 1, len(out_neighbors)):
                    j = out_neighbors[j_idx]
                    k = out_neighbors[k_idx]
                    
                    # Check if j and k are connected
                    if adj_matrix[j, k] > 0 or adj_matrix[k, j] > 0:
                        triangles += 1
                    
                    possible_triangles += 1
        
        if possible_triangles == 0:
            return 0.0
        
        return triangles / possible_triangles
    
    def _analyze_connectivity_distribution(self, adj_matrix: torch.Tensor) -> Dict[str, Any]:
        """Analyze the distribution of connections."""
        # Degree distributions
        out_degrees = adj_matrix.sum(dim=1)
        in_degrees = adj_matrix.sum(dim=0)
        
        # Compute statistics
        out_stats = {
            'mean': out_degrees.mean().item(),
            'std': out_degrees.std().item() if out_degrees.numel() > 1 else 0.0,
            'max': out_degrees.max().item(),
            'min': out_degrees.min().item()
        }
        
        in_stats = {
            'mean': in_degrees.mean().item(),
            'std': in_degrees.std().item() if in_degrees.numel() > 1 else 0.0,
            'max': in_degrees.max().item(),
            'min': in_degrees.min().item()
        }
        
        # Check for scale-free property (power law distribution)
        unique_out_degrees, out_counts = torch.unique(out_degrees, return_counts=True)
        unique_in_degrees, in_counts = torch.unique(in_degrees, return_counts=True)
        
        # Simple power law check: variance >> mean suggests heavy tail
        power_law_indicator = (out_stats['std'] ** 2) / (out_stats['mean'] + 1e-8)
        
        return {
            'out_degree_stats': out_stats,
            'in_degree_stats': in_stats,
            'power_law_indicator': power_law_indicator,
            'degree_heterogeneity': (out_stats['std'] + in_stats['std']) / 2
        }
    
    def _compute_sparsity(self, weight_matrix: torch.Tensor) -> float:
        """Compute weight matrix sparsity."""
        total_elements = weight_matrix.numel()
        zero_elements = (weight_matrix.abs() < 1e-8).sum().item()
        
        return zero_elements / total_elements if total_elements > 0 else 0.0