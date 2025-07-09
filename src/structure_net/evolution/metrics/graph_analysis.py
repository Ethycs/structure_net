"""
Graph Analysis Module

This module provides comprehensive graph-theoretic analysis of sparse neural networks,
including connectivity, centrality, spectral, and percolation analysis.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import sparse as sp
from typing import Dict, Any, List
import time
import logging

try:
    import cugraph
    import cudf
    CUGRAPH_AVAILABLE = True
except ImportError:
    import networkx as nx
    CUGRAPH_AVAILABLE = False

from .base import BaseMetricAnalyzer, NetworkAnalyzerMixin

logger = logging.getLogger(__name__)


class GraphAnalyzer(BaseMetricAnalyzer, NetworkAnalyzerMixin):
    """
    Complete graph-based analysis of sparse networks.
    
    Analyzes network topology, connectivity patterns, centrality measures,
    spectral properties, and percolation characteristics.
    """
    
    def __init__(self, network: nn.Module, threshold_config):
        super().__init__(threshold_config)
        self.network = network
        self.use_gpu = CUGRAPH_AVAILABLE and torch.cuda.is_available()
        if self.use_gpu:
            logger.info("🚀 GraphAnalyzer: GPU acceleration (cuGraph) enabled")
        else:
            logger.info("📊 GraphAnalyzer: Using CPU (networkx)")
        
    def compute_metrics(self, activation_data: Dict) -> Dict[str, Any]:
        """
        Compute ALL graph-based metrics.
        
        Args:
            activation_data: Dict mapping layer_idx -> activations tensor
            
        Returns:
            Dict containing comprehensive graph metrics
        """
        start_time = time.time()
        self._computation_stats['total_calls'] += 1
        
        # Check cache
        cache_key = self._cache_key(str(activation_data.keys()))
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Build active network graph
        if self.use_gpu:
            G, active_neurons = self._build_active_graph_gpu(activation_data)
        else:
            G, active_neurons = self._build_active_graph_cpu(activation_data)
        
        num_nodes = G.number_of_nodes() if not self.use_gpu else G.number_of_vertices()
        
        if num_nodes == 0:
            result = self._zero_graph_metrics()
        else:
            result = {
                'graph_built': True,
                'num_nodes': num_nodes,
                'num_edges': G.number_of_edges(),
                'active_neurons_per_layer': {k: len(v) for k, v in active_neurons.items()}
            }
            
            # Basic Graph Statistics
            result.update(self._compute_basic_graph_metrics(G))
            
            # Degree Statistics
            result.update(self._compute_degree_metrics(G))
            
            # Component Analysis
            result.update(self._compute_component_metrics(G))
            
            # Betweenness Centrality
            if num_nodes > 10:
                result.update(self._compute_betweenness_metrics(G))
            
            # Spectral Analysis
            if num_nodes > 5:
                result.update(self._compute_spectral_metrics(G))
            
            # Path Analysis
            result.update(self._compute_path_metrics(G, active_neurons))
            
            # Motif Analysis
            if num_nodes < 1000 and not self.use_gpu: # Motif analysis not yet in cugraph
                result.update(self._compute_motif_metrics(G))
            
            # Percolation Analysis
            result.update(self._compute_percolation_metrics(G, active_neurons))
        
        # Update timing stats
        computation_time = time.time() - start_time
        self._computation_stats['total_time'] += computation_time
        result['computation_time'] = computation_time
        
        # Cache result
        self._cache_result(cache_key, result)
        
        return result
    
    def _build_active_graph_cpu(self, activation_data):
        """Build directed graph of active neurons using networkx."""
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
        sparse_layers = self._get_sparse_layers(self.network)
        
        for i in range(len(sparse_layers) - 1):
            # The weights connecting FROM layer i TO layer i+1 are in layer i+1
            dst_layer = sparse_layers[i + 1]
            
            # Get sparse weights from destination layer
            weights = self._get_layer_weights(dst_layer)
            
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

    def _build_active_graph_gpu(self, activation_data):
        """Build directed graph of active neurons using cugraph."""
        active_neurons = {}
        node_map = {}
        current_node_id = 0
        
        # Add nodes for active neurons
        for layer_idx, acts in activation_data.items():
            if acts.dim() != 2:
                logger.warning(f"Skipping layer {layer_idx}: activation tensor has unexpected dim {acts.dim()}")
                continue
            
            mean_abs_acts = acts.abs().mean(dim=0)
            active_mask = mean_abs_acts > self.config.activation_threshold
            active_indices = torch.where(active_mask)[0].tolist()
            active_neurons[layer_idx] = active_indices
            
            for neuron_idx in active_indices:
                node_name = f"L{layer_idx}_N{neuron_idx}"
                node_map[node_name] = current_node_id
                current_node_id += 1

        # Add edges based on weights
        edges = []
        sparse_layers = self._get_sparse_layers(self.network)
        
        for i in range(len(sparse_layers) - 1):
            dst_layer = sparse_layers[i + 1]
            weights = self._get_layer_weights(dst_layer)
            active_connections = weights.abs() > self.config.weight_threshold
            dst_indices, src_indices = torch.where(active_connections)
            
            active_src_set = set(active_neurons.get(i, []))
            active_dst_set = set(active_neurons.get(i + 1, []))
            
            for dst_idx, src_idx in zip(dst_indices.tolist(), src_indices.tolist()):
                if src_idx in active_src_set and dst_idx in active_dst_set:
                    src_node_name = f"L{i}_N{src_idx}"
                    dst_node_name = f"L{i+1}_N{dst_idx}"
                    if src_node_name in node_map and dst_node_name in node_map:
                        edges.append((node_map[src_node_name], node_map[dst_node_name]))

        if not edges:
            return cugraph.Graph(directed=True), active_neurons

        edge_df = cudf.DataFrame(edges, columns=['src', 'dst'])
        G = cugraph.Graph(directed=True)
        G.from_cudf_edgelist(edge_df, source='src', destination='dst')
        
        return G, active_neurons
    
    def _compute_basic_graph_metrics(self, G):
        """Basic graph statistics."""
        metrics = {}
        
        if self.use_gpu:
            metrics['density'] = G.density()
            degrees = G.degrees()
            metrics['avg_in_degree'] = degrees['in_degree'].mean()
            metrics['avg_out_degree'] = degrees['out_degree'].mean()
            metrics['max_in_degree'] = degrees['in_degree'].max()
            metrics['max_out_degree'] = degrees['out_degree'].max()
        else:
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
        
        if self.use_gpu:
            degrees = G.degrees()
            in_degrees = degrees['in_degree']
            out_degrees = degrees['out_degree']
            
            metrics['in_degree_std'] = in_degrees.std()
            metrics['out_degree_std'] = out_degrees.std()
            
            in_degree_threshold = in_degrees.mean() + 2 * in_degrees.std()
            metrics['num_in_hubs'] = (in_degrees > in_degree_threshold).sum()
            
            out_degree_threshold = out_degrees.mean() + 2 * out_degrees.std()
            metrics['num_out_hubs'] = (out_degrees > out_degree_threshold).sum()
        else:
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
        
        if self.use_gpu:
            wcc = cugraph.weakly_connected_components(G)
            wcc_counts = wcc['labels'].value_counts()
            metrics['num_weakly_connected_components'] = len(wcc_counts)
            metrics['largest_wcc_size'] = wcc_counts.max() if not wcc_counts.empty else 0
            metrics['isolated_neurons'] = (wcc_counts == 1).sum()
            
            scc = cugraph.strongly_connected_components(G)
            scc_counts = scc['labels'].value_counts()
            metrics['num_strongly_connected_components'] = len(scc_counts)
            metrics['largest_scc_size'] = scc_counts.max() if not scc_counts.empty else 0
        else:
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
        
        if self.use_gpu:
            k = min(100, G.number_of_vertices())
            betweenness = cugraph.betweenness_centrality(G, k=k, normalized=True)
            bc_values = betweenness['betweenness_centrality'].to_arrow().to_pylist()
        else:
            # Sample nodes for efficiency
            nodes = list(G.nodes())
            k = min(100, len(nodes))
            sample_nodes = np.random.choice(nodes, k, replace=False).tolist()
            
            # Compute betweenness
            betweenness = nx.betweenness_centrality_subset(
                G, sources=sample_nodes, targets=sample_nodes, normalized=True
            )
            bc_values = list(betweenness.values())

        metrics['avg_betweenness'] = np.mean(bc_values)
        metrics['max_betweenness'] = np.max(bc_values)
        metrics['betweenness_std'] = np.std(bc_values)
        
        # Find bottleneck neurons
        threshold = metrics['avg_betweenness'] + 2 * metrics['betweenness_std']
        if self.use_gpu:
            bottlenecks = betweenness[betweenness['betweenness_centrality'] > threshold]
            metrics['num_bottlenecks'] = len(bottlenecks)
            metrics['bottleneck_neurons'] = bottlenecks.nlargest(10, 'betweenness_centrality').to_pandas().to_dict('records')
        else:
            bottlenecks = [(node, bc) for node, bc in betweenness.items() if bc > threshold]
            bottlenecks.sort(key=lambda x: x[1], reverse=True)
            metrics['bottleneck_neurons'] = bottlenecks[:10]
            metrics['num_bottlenecks'] = len(bottlenecks)
        
        return metrics
    
    def _compute_spectral_metrics(self, G):
        """Spectral analysis of the graph."""
        metrics = {}
        
        if self.use_gpu:
            # cugraph does not have a direct equivalent of eigsh for Laplacian.
            # This part will remain on CPU for now.
            # TODO: Explore GPU-accelerated spectral methods.
            return metrics

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
        
        if self.use_gpu:
            # cugraph SSSP is efficient for path calculations
            # We'll sample a few source nodes and compute paths to all other nodes
            input_nodes = [f"L{input_layer}_N{n}" for n in active_neurons[input_layer]]
            k = min(10, len(input_nodes))
            if k == 0:
                return metrics
            sample_sources = np.random.choice(input_nodes, k, replace=False).tolist()
            
            # This part is complex with cugraph and requires node ID mapping.
            # For now, we'll skip GPU path metrics.
            # TODO: Implement GPU-accelerated path metrics.
            return metrics
        else:
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
            for i in range(len(self._get_sparse_layers(self.network)) - 1)
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
        num_nodes = G.number_of_vertices() if self.use_gpu else G.number_of_nodes()
        if num_nodes > 0:
            if self.use_gpu:
                wcc = cugraph.weakly_connected_components(G)
                largest_wcc_size = wcc['labels'].value_counts().max() if not wcc.empty else 0
            else:
                largest_wcc_size = max(len(c) for c in nx.weakly_connected_components(G))
            metrics['giant_component_fraction'] = largest_wcc_size / num_nodes
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


# Export the analyzer
__all__ = ['GraphAnalyzer']
