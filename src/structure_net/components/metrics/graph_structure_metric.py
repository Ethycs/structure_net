"""
Graph structure metric component.

This component builds a graph representation of active neural network connections
and computes basic structural properties like density, degree statistics, and components.
"""

from typing import Dict, Any, Union, Optional, Tuple, List
import torch
import torch.nn as nn
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


class GraphStructureMetric(BaseMetric):
    """
    Builds and analyzes the graph structure of active neural network connections.
    
    Creates a directed graph where nodes are active neurons and edges are
    active connections, then computes basic structural properties.
    """
    
    def __init__(self, activation_threshold: float = 0.01,
                 weight_threshold: float = 0.01,
                 name: str = None):
        """
        Initialize graph structure metric.
        
        Args:
            activation_threshold: Threshold for active neurons
            weight_threshold: Threshold for active connections
            name: Optional custom name
        """
        super().__init__(name or "GraphStructureMetric")
        self.activation_threshold = activation_threshold
        self.weight_threshold = weight_threshold
        self._measurement_schema = {
            "num_nodes": int,
            "num_edges": int,
            "density": float,
            "avg_in_degree": float,
            "avg_out_degree": float,
            "max_in_degree": int,
            "max_out_degree": int,
            "num_components": int,
            "largest_component_size": int
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"activation_data", "model"},
            provided_outputs={
                "metrics.num_nodes",
                "metrics.num_edges", 
                "metrics.density",
                "metrics.avg_in_degree",
                "metrics.avg_out_degree",
                "metrics.max_in_degree",
                "metrics.max_out_degree",
                "metrics.num_components",
                "metrics.largest_component_size",
                "metrics.graph"  # The actual graph object
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
        Compute graph structure metrics.
        
        Args:
            target: Model to analyze (not used directly)
            context: Must contain 'activation_data' and 'model'
            
        Returns:
            Dictionary containing graph structure measurements
        """
        if not NETWORKX_AVAILABLE:
            self.log(logging.WARNING, "NetworkX not available, returning empty metrics")
            return self._empty_metrics()
        
        # Get required data
        activation_data = context.get('activation_data')
        model = context.get('model')
        
        if activation_data is None or model is None:
            raise ValueError("GraphStructureMetric requires 'activation_data' and 'model' in context")
        
        # Build graph
        G, active_neurons = self._build_active_graph(activation_data, model)
        
        if G.number_of_nodes() == 0:
            return self._empty_metrics()
        
        # Basic statistics
        metrics = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G),
            "graph": G,  # Store graph for other metrics
            "active_neurons": active_neurons
        }
        
        # Degree statistics
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        
        metrics["avg_in_degree"] = np.mean(in_degrees) if in_degrees else 0
        metrics["avg_out_degree"] = np.mean(out_degrees) if out_degrees else 0
        metrics["max_in_degree"] = max(in_degrees) if in_degrees else 0
        metrics["max_out_degree"] = max(out_degrees) if out_degrees else 0
        
        # Component analysis
        wcc = list(nx.weakly_connected_components(G))
        metrics["num_components"] = len(wcc)
        metrics["largest_component_size"] = len(max(wcc, key=len)) if wcc else 0
        
        self.log(logging.DEBUG, 
                f"Graph: {metrics['num_nodes']} nodes, {metrics['num_edges']} edges, "
                f"density={metrics['density']:.3f}")
        
        return metrics
    
    def _build_active_graph(self, activation_data: Dict[int, torch.Tensor], 
                           model: nn.Module) -> Tuple[nx.DiGraph, Dict[int, List[int]]]:
        """Build directed graph of active neurons."""
        G = nx.DiGraph()
        active_neurons = {}
        
        # Add nodes for active neurons
        for layer_idx, acts in activation_data.items():
            if acts.dim() != 2:
                self.log(logging.WARNING, 
                        f"Skipping layer {layer_idx}: unexpected dim {acts.dim()}")
                continue
            
            # Find active neurons
            mean_abs_acts = acts.abs().mean(dim=0)
            active_mask = mean_abs_acts > self.activation_threshold
            active_indices = torch.where(active_mask)[0].tolist()
            active_neurons[layer_idx] = active_indices
            
            # Add nodes
            for neuron_idx in active_indices:
                node_id = f"L{layer_idx}_N{neuron_idx}"
                activation_val = mean_abs_acts[neuron_idx].item()
                
                G.add_node(node_id,
                          layer=layer_idx,
                          neuron=neuron_idx,
                          activation=activation_val)
        
        # Add edges based on weights
        layers = self._get_sparse_layers(model)
        
        for i in range(len(layers) - 1):
            if i not in active_neurons or i + 1 not in active_neurons:
                continue
            
            layer = layers[i + 1]
            weights = self._get_layer_weights(layer)
            
            if weights is None:
                continue
            
            # Find active connections
            active_connections = weights.abs() > self.weight_threshold
            dst_indices, src_indices = torch.where(active_connections)
            
            active_src_set = set(active_neurons[i])
            active_dst_set = set(active_neurons[i + 1])
            
            # Add edges
            for dst_idx, src_idx in zip(dst_indices.tolist(), src_indices.tolist()):
                if src_idx in active_src_set and dst_idx in active_dst_set:
                    src_node = f"L{i}_N{src_idx}"
                    dst_node = f"L{i+1}_N{dst_idx}"
                    weight = weights[dst_idx, src_idx].item()
                    
                    G.add_edge(src_node, dst_node, weight=abs(weight))
        
        return G, active_neurons
    
    def _get_sparse_layers(self, model: nn.Module) -> List[nn.Module]:
        """Get sparse layers from model."""
        layers = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layers.append(module)
        return layers
    
    def _get_layer_weights(self, layer: nn.Module) -> Optional[torch.Tensor]:
        """Get weight matrix from layer."""
        if hasattr(layer, 'weight'):
            weight = layer.weight
            if weight.dim() > 2:
                # Flatten conv weights
                weight = weight.flatten(1)
            return weight
        return None
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics when graph cannot be built."""
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "density": 0.0,
            "avg_in_degree": 0.0,
            "avg_out_degree": 0.0,
            "max_in_degree": 0,
            "max_out_degree": 0,
            "num_components": 0,
            "largest_component_size": 0,
            "graph": None,
            "active_neurons": {}
        }