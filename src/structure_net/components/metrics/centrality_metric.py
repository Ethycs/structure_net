"""
Centrality metric component.

This component computes various centrality measures for neural network graphs
including betweenness, closeness, and eigenvector centrality.
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


class CentralityMetric(BaseMetric):
    """
    Computes centrality measures for neural network graphs.
    
    Centrality measures help identify important neurons that play
    critical roles in information flow through the network.
    """
    
    def __init__(self, sample_size: int = 100, name: str = None):
        """
        Initialize centrality metric.
        
        Args:
            sample_size: Number of nodes to sample for expensive computations
            name: Optional custom name
        """
        super().__init__(name or "CentralityMetric")
        self.sample_size = sample_size
        self._measurement_schema = {
            "betweenness_centrality": dict,
            "avg_betweenness": float,
            "max_betweenness": float,
            "hub_neurons": list,
            "centrality_concentration": float
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"metrics.graph"},
            provided_outputs={
                "metrics.betweenness_centrality",
                "metrics.avg_betweenness",
                "metrics.max_betweenness",
                "metrics.hub_neurons",
                "metrics.centrality_concentration"
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
        Compute centrality metrics.
        
        Args:
            target: Not used directly
            context: Must contain graph from GraphStructureMetric
            
        Returns:
            Dictionary containing centrality measurements
        """
        if not NETWORKX_AVAILABLE:
            self.log(logging.WARNING, "NetworkX not available, returning empty metrics")
            return self._empty_metrics()
        
        # Get graph
        G = context.get('metrics.graph')
        if G is None or not isinstance(G, nx.DiGraph):
            return self._empty_metrics()
        
        if G.number_of_nodes() == 0:
            return self._empty_metrics()
        
        # Compute betweenness centrality
        if G.number_of_nodes() > self.sample_size:
            # Sample nodes for efficiency
            nodes = list(G.nodes())
            sample_nodes = np.random.choice(nodes, self.sample_size, replace=False).tolist()
            
            betweenness = nx.betweenness_centrality_subset(
                G, sources=sample_nodes, targets=sample_nodes, normalized=True
            )
        else:
            betweenness = nx.betweenness_centrality(G, normalized=True)
        
        # Analyze betweenness
        bc_values = list(betweenness.values())
        avg_betweenness = np.mean(bc_values) if bc_values else 0
        max_betweenness = max(bc_values) if bc_values else 0
        
        # Identify hub neurons (top 5% by betweenness)
        threshold = np.percentile(bc_values, 95) if bc_values else 0
        hub_neurons = []
        
        for node, bc in betweenness.items():
            if bc > threshold:
                hub_neurons.append({
                    'node': node,
                    'betweenness': bc,
                    'layer': G.nodes[node].get('layer', -1),
                    'neuron': G.nodes[node].get('neuron', -1)
                })
        
        # Sort hubs by betweenness
        hub_neurons.sort(key=lambda x: x['betweenness'], reverse=True)
        
        # Centrality concentration (Gini coefficient of betweenness)
        centrality_concentration = self._compute_gini_coefficient(bc_values)
        
        self.log(logging.DEBUG, 
                f"Centrality: avg={avg_betweenness:.3f}, max={max_betweenness:.3f}, "
                f"hubs={len(hub_neurons)}")
        
        return {
            "betweenness_centrality": betweenness,
            "avg_betweenness": avg_betweenness,
            "max_betweenness": max_betweenness,
            "hub_neurons": hub_neurons[:10],  # Top 10 hubs
            "centrality_concentration": centrality_concentration
        }
    
    def _compute_gini_coefficient(self, values: List[float]) -> float:
        """Compute Gini coefficient to measure inequality."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        if cumsum[-1] == 0:
            return 0.0
        
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics when computation cannot proceed."""
        return {
            "betweenness_centrality": {},
            "avg_betweenness": 0.0,
            "max_betweenness": 0.0,
            "hub_neurons": [],
            "centrality_concentration": 0.0
        }