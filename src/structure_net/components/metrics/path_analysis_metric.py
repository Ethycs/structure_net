"""
Path analysis metric component.

This component analyzes path properties in neural network graphs including
shortest paths, path lengths, and information flow characteristics.
"""

from typing import Dict, Any, Union, Optional, List, Tuple
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


class PathAnalysisMetric(BaseMetric):
    """
    Analyzes path properties in neural network graphs.
    
    Computes shortest paths between layers, average path lengths,
    and identifies critical paths for information flow.
    """
    
    def __init__(self, sample_pairs: int = 100, name: str = None):
        """
        Initialize path analysis metric.
        
        Args:
            sample_pairs: Number of node pairs to sample for path analysis
            name: Optional custom name
        """
        super().__init__(name or "PathAnalysisMetric")
        self.sample_pairs = sample_pairs
        self._measurement_schema = {
            "avg_shortest_path": float,
            "diameter": int,
            "characteristic_path_length": float,
            "layer_connectivity": dict,
            "critical_paths": list,
            "path_efficiency": float
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"metrics.graph", "metrics.active_neurons"},
            provided_outputs={
                "metrics.avg_shortest_path",
                "metrics.diameter",
                "metrics.characteristic_path_length",
                "metrics.layer_connectivity",
                "metrics.critical_paths",
                "metrics.path_efficiency"
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
        Compute path analysis metrics.
        
        Args:
            target: Not used directly
            context: Must contain graph and active_neurons
            
        Returns:
            Dictionary containing path measurements
        """
        if not NETWORKX_AVAILABLE:
            self.log(logging.WARNING, "NetworkX not available, returning empty metrics")
            return self._empty_metrics()
        
        # Get graph and active neurons
        G = context.get('metrics.graph')
        active_neurons = context.get('metrics.active_neurons', {})
        
        if G is None or not isinstance(G, nx.DiGraph):
            return self._empty_metrics()
        
        if G.number_of_nodes() < 2:
            return self._empty_metrics()
        
        # Compute path metrics
        avg_shortest_path = self._compute_average_shortest_path(G)
        diameter = self._compute_diameter(G)
        characteristic_path_length = self._compute_characteristic_path_length(G)
        
        # Layer connectivity analysis
        layer_connectivity = self._analyze_layer_connectivity(G, active_neurons)
        
        # Find critical paths
        critical_paths = self._find_critical_paths(G, active_neurons)
        
        # Path efficiency (how direct paths are compared to minimum possible)
        path_efficiency = self._compute_path_efficiency(G, active_neurons)
        
        self.log(logging.DEBUG, 
                f"Paths: avg_length={avg_shortest_path:.2f}, diameter={diameter}, "
                f"efficiency={path_efficiency:.3f}")
        
        return {
            "avg_shortest_path": avg_shortest_path,
            "diameter": diameter,
            "characteristic_path_length": characteristic_path_length,
            "layer_connectivity": layer_connectivity,
            "critical_paths": critical_paths,
            "path_efficiency": path_efficiency
        }
    
    def _compute_average_shortest_path(self, G: nx.DiGraph) -> float:
        """Compute average shortest path length."""
        # Sample node pairs for large graphs
        nodes = list(G.nodes())
        
        if len(nodes) > self.sample_pairs * 2:
            # Random sampling
            sampled_pairs = []
            for _ in range(self.sample_pairs):
                src = np.random.choice(nodes)
                dst = np.random.choice(nodes)
                if src != dst:
                    sampled_pairs.append((src, dst))
        else:
            # All pairs for small graphs
            sampled_pairs = [(u, v) for u in nodes for v in nodes if u != v]
        
        # Compute path lengths
        path_lengths = []
        for src, dst in sampled_pairs:
            try:
                length = nx.shortest_path_length(G, src, dst)
                path_lengths.append(length)
            except nx.NetworkXNoPath:
                continue
        
        return np.mean(path_lengths) if path_lengths else 0.0
    
    def _compute_diameter(self, G: nx.DiGraph) -> int:
        """Compute graph diameter (longest shortest path)."""
        try:
            # For efficiency, compute on largest strongly connected component
            if nx.is_strongly_connected(G):
                return nx.diameter(G)
            else:
                # Find largest SCC
                sccs = list(nx.strongly_connected_components(G))
                if sccs:
                    largest_scc = max(sccs, key=len)
                    subgraph = G.subgraph(largest_scc)
                    if subgraph.number_of_nodes() > 1:
                        return nx.diameter(subgraph)
            return 0
        except:
            return 0
    
    def _compute_characteristic_path_length(self, G: nx.DiGraph) -> float:
        """Compute characteristic path length (CPL)."""
        # CPL is average shortest path in largest connected component
        if nx.is_weakly_connected(G):
            try:
                return nx.average_shortest_path_length(G)
            except:
                return 0.0
        else:
            # Use largest weakly connected component
            wccs = list(nx.weakly_connected_components(G))
            if wccs:
                largest_wcc = max(wccs, key=len)
                subgraph = G.subgraph(largest_wcc)
                try:
                    return nx.average_shortest_path_length(subgraph)
                except:
                    return 0.0
        return 0.0
    
    def _analyze_layer_connectivity(self, G: nx.DiGraph, 
                                   active_neurons: Dict[int, List[int]]) -> Dict[str, Any]:
        """Analyze connectivity between layers."""
        layer_connectivity = {}
        
        # Get unique layers
        layers = sorted(active_neurons.keys())
        
        for i in range(len(layers) - 1):
            src_layer = layers[i]
            dst_layer = layers[i + 1]
            
            # Count paths between layers
            src_nodes = [f"L{src_layer}_N{n}" for n in active_neurons.get(src_layer, [])]
            dst_nodes = [f"L{dst_layer}_N{n}" for n in active_neurons.get(dst_layer, [])]
            
            connected_pairs = 0
            total_pairs = len(src_nodes) * len(dst_nodes)
            
            if total_pairs > 0:
                # Sample for efficiency
                sample_size = min(100, total_pairs)
                for _ in range(sample_size):
                    src = np.random.choice(src_nodes) if src_nodes else None
                    dst = np.random.choice(dst_nodes) if dst_nodes else None
                    
                    if src and dst and nx.has_path(G, src, dst):
                        connected_pairs += 1
                
                connectivity_ratio = connected_pairs / sample_size
                layer_connectivity[f"L{src_layer}->L{dst_layer}"] = connectivity_ratio
        
        return layer_connectivity
    
    def _find_critical_paths(self, G: nx.DiGraph,
                           active_neurons: Dict[int, List[int]]) -> List[Dict[str, Any]]:
        """Find critical paths through the network."""
        critical_paths = []
        
        # Get input and output layers
        layers = sorted(active_neurons.keys())
        if len(layers) < 2:
            return critical_paths
        
        input_layer = layers[0]
        output_layer = layers[-1]
        
        input_nodes = [f"L{input_layer}_N{n}" for n in active_neurons.get(input_layer, [])]
        output_nodes = [f"L{output_layer}_N{n}" for n in active_neurons.get(output_layer, [])]
        
        # Sample paths
        num_samples = min(10, len(input_nodes) * len(output_nodes))
        
        for _ in range(num_samples):
            if not input_nodes or not output_nodes:
                break
                
            src = np.random.choice(input_nodes)
            dst = np.random.choice(output_nodes)
            
            try:
                path = nx.shortest_path(G, src, dst)
                path_weight = sum(G[u][v].get('weight', 1.0) for u, v in zip(path[:-1], path[1:]))
                
                critical_paths.append({
                    'path': path,
                    'length': len(path) - 1,
                    'weight': path_weight,
                    'source': src,
                    'target': dst
                })
            except nx.NetworkXNoPath:
                continue
        
        # Sort by weight (importance)
        critical_paths.sort(key=lambda x: x['weight'], reverse=True)
        
        return critical_paths[:5]  # Top 5 critical paths
    
    def _compute_path_efficiency(self, G: nx.DiGraph,
                               active_neurons: Dict[int, List[int]]) -> float:
        """Compute path efficiency (directness of connections)."""
        layers = sorted(active_neurons.keys())
        if len(layers) < 2:
            return 0.0
        
        # Theoretical minimum path length between layers
        min_path_length = len(layers) - 1
        
        # Sample actual paths
        actual_lengths = []
        
        input_layer = layers[0]
        output_layer = layers[-1]
        
        input_nodes = [f"L{input_layer}_N{n}" for n in active_neurons.get(input_layer, [])]
        output_nodes = [f"L{output_layer}_N{n}" for n in active_neurons.get(output_layer, [])]
        
        num_samples = min(50, len(input_nodes) * len(output_nodes))
        
        for _ in range(num_samples):
            if not input_nodes or not output_nodes:
                break
                
            src = np.random.choice(input_nodes)
            dst = np.random.choice(output_nodes)
            
            try:
                length = nx.shortest_path_length(G, src, dst)
                actual_lengths.append(length)
            except nx.NetworkXNoPath:
                continue
        
        if not actual_lengths:
            return 0.0
        
        avg_actual_length = np.mean(actual_lengths)
        
        # Efficiency is ratio of minimum to actual
        efficiency = min_path_length / avg_actual_length if avg_actual_length > 0 else 0.0
        
        return min(1.0, efficiency)  # Cap at 1.0
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics when computation cannot proceed."""
        return {
            "avg_shortest_path": 0.0,
            "diameter": 0,
            "characteristic_path_length": 0.0,
            "layer_connectivity": {},
            "critical_paths": [],
            "path_efficiency": 0.0
        }