"""
Graph analyzer component.

This analyzer combines multiple graph-theoretic metrics to provide comprehensive
analysis of neural network topology from a graph perspective.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import logging
import numpy as np

from src.structure_net.core import (
    BaseAnalyzer, IModel, ILayer, EvolutionContext, AnalysisReport,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)
from src.structure_net.components.metrics import (
    GraphStructureMetric, CentralityMetric, 
    SpectralGraphMetric, PathAnalysisMetric
)


class GraphAnalyzer(BaseAnalyzer):
    """
    Analyzes neural networks from a graph-theoretic perspective.
    
    Combines graph structure, centrality, spectral properties, and path
    analysis to provide insights about network topology, connectivity
    patterns, and information flow characteristics.
    """
    
    def __init__(self, activation_threshold: float = 0.01,
                 weight_threshold: float = 0.01,
                 sample_size: int = 100,
                 name: str = None):
        """
        Initialize graph analyzer.
        
        Args:
            activation_threshold: Threshold for active neurons
            weight_threshold: Threshold for active connections
            sample_size: Sample size for expensive computations
            name: Optional custom name
        """
        super().__init__(name or "GraphAnalyzer")
        self.activation_threshold = activation_threshold
        self.weight_threshold = weight_threshold
        self.sample_size = sample_size
        
        # Initialize metrics
        self._structure_metric = GraphStructureMetric(
            activation_threshold, weight_threshold
        )
        self._centrality_metric = CentralityMetric(sample_size)
        self._spectral_metric = SpectralGraphMetric()
        self._path_metric = PathAnalysisMetric(sample_size)
        
        self._required_metrics = {
            "GraphStructureMetric",
            "CentralityMetric",
            "SpectralGraphMetric",
            "PathAnalysisMetric"
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"model", "activation_data"},
            provided_outputs={
                "analysis.graph_summary",
                "analysis.connectivity_analysis",
                "analysis.centrality_analysis",
                "analysis.spectral_analysis",
                "analysis.path_analysis",
                "analysis.network_diagnostics",
                "analysis.architectural_insights"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.HIGH,
                requires_gpu=False,
                parallel_safe=False
            )
        )
    
    def _perform_analysis(self, model: IModel, report: AnalysisReport, 
                         context: EvolutionContext) -> Dict[str, Any]:
        """
        Perform comprehensive graph analysis.
        
        Args:
            model: Model to analyze
            report: Analysis report containing metric data
            context: Evolution context with activation data
            
        Returns:
            Dictionary containing analysis results
        """
        # Get activation data
        activation_data = context.get('activation_data')
        if activation_data is None:
            raise ValueError("GraphAnalyzer requires 'activation_data' in context")
        
        # Run structure metric first (builds the graph)
        structure_context = EvolutionContext({
            'activation_data': activation_data,
            'model': model
        })
        
        structure_key = "GraphStructureMetric"
        if structure_key not in report.metrics:
            structure_result = self._structure_metric.analyze(model, structure_context)
            report.add_metric_data(structure_key, structure_result)
        else:
            structure_result = report.get(f"metrics.{structure_key}")
        
        # Check if graph was built successfully
        if structure_result.get('num_nodes', 0) == 0:
            return self._empty_analysis()
        
        # Pass graph to other metrics
        graph_context = EvolutionContext({
            'metrics.graph': structure_result['graph'],
            'metrics.active_neurons': structure_result['active_neurons']
        })
        
        # Run centrality metric
        centrality_key = "CentralityMetric"
        if centrality_key not in report.metrics:
            centrality_result = self._centrality_metric.analyze(None, graph_context)
            report.add_metric_data(centrality_key, centrality_result)
        else:
            centrality_result = report.get(f"metrics.{centrality_key}")
        
        # Run spectral metric
        spectral_key = "SpectralGraphMetric"
        if spectral_key not in report.metrics:
            spectral_result = self._spectral_metric.analyze(None, graph_context)
            report.add_metric_data(spectral_key, spectral_result)
        else:
            spectral_result = report.get(f"metrics.{spectral_key}")
        
        # Run path metric
        path_key = "PathAnalysisMetric"
        if path_key not in report.metrics:
            path_result = self._path_metric.analyze(None, graph_context)
            report.add_metric_data(path_key, path_result)
        else:
            path_result = report.get(f"metrics.{path_key}")
        
        # Create comprehensive analysis
        graph_summary = self._create_graph_summary(structure_result)
        connectivity_analysis = self._analyze_connectivity(
            structure_result, path_result
        )
        centrality_analysis = self._analyze_centrality(
            centrality_result, structure_result
        )
        spectral_analysis = self._analyze_spectral_properties(spectral_result)
        path_analysis = self._analyze_paths(path_result)
        network_diagnostics = self._diagnose_network(
            structure_result, centrality_result, spectral_result, path_result
        )
        architectural_insights = self._generate_architectural_insights(
            structure_result, centrality_result, spectral_result, path_result
        )
        
        return {
            "graph_summary": graph_summary,
            "connectivity_analysis": connectivity_analysis,
            "centrality_analysis": centrality_analysis,
            "spectral_analysis": spectral_analysis,
            "path_analysis": path_analysis,
            "network_diagnostics": network_diagnostics,
            "architectural_insights": architectural_insights
        }
    
    def _create_graph_summary(self, structure_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of graph structure."""
        return {
            "num_nodes": structure_result['num_nodes'],
            "num_edges": structure_result['num_edges'],
            "density": structure_result['density'],
            "avg_degree": (structure_result['avg_in_degree'] + 
                          structure_result['avg_out_degree']) / 2,
            "max_degree": max(structure_result['max_in_degree'],
                             structure_result['max_out_degree']),
            "num_components": structure_result['num_components'],
            "largest_component_ratio": (structure_result['largest_component_size'] / 
                                      structure_result['num_nodes'] 
                                      if structure_result['num_nodes'] > 0 else 0),
            "network_type": self._classify_network_type(structure_result)
        }
    
    def _analyze_connectivity(self, structure_result: Dict[str, Any],
                            path_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze connectivity patterns."""
        # Degree distribution analysis
        in_degree_cv = (structure_result.get('in_degree_std', 0) / 
                       (structure_result['avg_in_degree'] + 1e-10))
        out_degree_cv = (structure_result.get('out_degree_std', 0) / 
                        (structure_result['avg_out_degree'] + 1e-10))
        
        # Hub presence
        has_in_hubs = structure_result.get('num_in_hubs', 0) > 0
        has_out_hubs = structure_result.get('num_out_hubs', 0) > 0
        
        return {
            "connectivity_type": self._classify_connectivity(
                structure_result['density'], in_degree_cv, out_degree_cv
            ),
            "has_hub_structure": has_in_hubs or has_out_hubs,
            "in_degree_heterogeneity": in_degree_cv,
            "out_degree_heterogeneity": out_degree_cv,
            "layer_connectivity": path_result.get('layer_connectivity', {}),
            "connectivity_health": self._assess_connectivity_health(
                structure_result, path_result
            )
        }
    
    def _analyze_centrality(self, centrality_result: Dict[str, Any],
                          structure_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze centrality patterns."""
        return {
            "avg_betweenness": centrality_result['avg_betweenness'],
            "max_betweenness": centrality_result['max_betweenness'],
            "centrality_concentration": centrality_result['centrality_concentration'],
            "num_hub_neurons": len(centrality_result['hub_neurons']),
            "top_hubs": centrality_result['hub_neurons'][:5],  # Top 5
            "hub_distribution": self._analyze_hub_distribution(
                centrality_result['hub_neurons']
            ),
            "centrality_insights": self._generate_centrality_insights(
                centrality_result, structure_result
            )
        }
    
    def _analyze_spectral_properties(self, spectral_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze spectral properties."""
        return {
            "spectral_radius": spectral_result['spectral_radius'],
            "spectral_gap": spectral_result['spectral_gap'],
            "algebraic_connectivity": spectral_result['algebraic_connectivity'],
            "graph_energy": spectral_result['graph_energy'],
            "spectral_insights": self._generate_spectral_insights(spectral_result)
        }
    
    def _analyze_paths(self, path_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze path properties."""
        return {
            "avg_shortest_path": path_result['avg_shortest_path'],
            "diameter": path_result['diameter'],
            "characteristic_path_length": path_result['characteristic_path_length'],
            "path_efficiency": path_result['path_efficiency'],
            "critical_paths": path_result['critical_paths'][:3],  # Top 3
            "path_insights": self._generate_path_insights(path_result)
        }
    
    def _diagnose_network(self, structure_result: Dict[str, Any],
                         centrality_result: Dict[str, Any],
                         spectral_result: Dict[str, Any],
                         path_result: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose network health and issues."""
        issues = []
        health_score = 1.0
        
        # Check density
        if structure_result['density'] < 0.01:
            issues.append("extremely_sparse")
            health_score *= 0.8
        elif structure_result['density'] > 0.5:
            issues.append("overly_dense")
            health_score *= 0.9
        
        # Check connectivity
        if structure_result['num_components'] > structure_result['num_nodes'] * 0.1:
            issues.append("highly_fragmented")
            health_score *= 0.7
        
        # Check centrality concentration
        if centrality_result['centrality_concentration'] > 0.8:
            issues.append("over_centralized")
            health_score *= 0.8
        
        # Check spectral gap
        if spectral_result['spectral_gap'] < 0.1:
            issues.append("poor_spectral_gap")
            health_score *= 0.9
        
        # Check path efficiency
        if path_result['path_efficiency'] < 0.5:
            issues.append("inefficient_paths")
            health_score *= 0.8
        
        return {
            "health_score": max(0.0, health_score),
            "issues": issues,
            "status": self._get_health_status(health_score),
            "recommendations": self._generate_recommendations(issues)
        }
    
    def _generate_architectural_insights(self, structure_result: Dict[str, Any],
                                       centrality_result: Dict[str, Any],
                                       spectral_result: Dict[str, Any],
                                       path_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate architectural insights and suggestions."""
        insights = []
        
        # Sparsity insights
        if structure_result['density'] < 0.05:
            insights.append({
                "type": "sparsity",
                "insight": "Network is very sparse, may benefit from additional connections",
                "suggestion": "Consider adding skip connections or increasing connectivity",
                "impact": "high"
            })
        
        # Hub insights
        if len(centrality_result['hub_neurons']) > structure_result['num_nodes'] * 0.05:
            insights.append({
                "type": "centralization",
                "insight": "Network has significant hub structure",
                "suggestion": "Consider distributing connections more evenly",
                "impact": "medium"
            })
        
        # Path efficiency insights
        if path_result['path_efficiency'] < 0.7:
            insights.append({
                "type": "efficiency",
                "insight": "Information flow is indirect",
                "suggestion": "Add shortcut connections between distant layers",
                "impact": "high"
            })
        
        # Spectral insights
        if spectral_result['algebraic_connectivity'] < 0.01:
            insights.append({
                "type": "robustness",
                "insight": "Network has poor algebraic connectivity",
                "suggestion": "Strengthen connections to improve robustness",
                "impact": "medium"
            })
        
        return insights
    
    def _classify_network_type(self, structure_result: Dict[str, Any]) -> str:
        """Classify the type of network based on structure."""
        density = structure_result['density']
        
        if density < 0.01:
            return "ultra_sparse"
        elif density < 0.05:
            return "sparse"
        elif density < 0.2:
            return "moderate"
        elif density < 0.5:
            return "dense"
        else:
            return "fully_connected"
    
    def _classify_connectivity(self, density: float, in_cv: float, out_cv: float) -> str:
        """Classify connectivity pattern."""
        avg_cv = (in_cv + out_cv) / 2
        
        if avg_cv > 1.5:
            return "scale_free"  # High degree heterogeneity
        elif density > 0.3:
            return "small_world"  # Dense with structure
        elif density < 0.05:
            return "sparse_regular"  # Sparse but regular
        else:
            return "random"  # No clear pattern
    
    def _assess_connectivity_health(self, structure_result: Dict[str, Any],
                                  path_result: Dict[str, Any]) -> float:
        """Assess overall connectivity health."""
        health = 1.0
        
        # Penalize extreme density
        if structure_result['density'] < 0.01 or structure_result['density'] > 0.5:
            health *= 0.8
        
        # Penalize poor layer connectivity
        layer_conn_values = list(path_result.get('layer_connectivity', {}).values())
        if layer_conn_values:
            avg_layer_conn = np.mean(layer_conn_values)
            if avg_layer_conn < 0.5:
                health *= 0.7
        
        # Reward good path efficiency
        if path_result['path_efficiency'] > 0.8:
            health *= 1.1
        
        return min(1.0, health)
    
    def _analyze_hub_distribution(self, hub_neurons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze distribution of hub neurons across layers."""
        if not hub_neurons:
            return {"num_layers_with_hubs": 0, "hub_concentration": 0.0}
        
        layers_with_hubs = set(hub['layer'] for hub in hub_neurons)
        
        # Count hubs per layer
        hubs_per_layer = {}
        for hub in hub_neurons:
            layer = hub['layer']
            hubs_per_layer[layer] = hubs_per_layer.get(layer, 0) + 1
        
        # Compute concentration (Gini-like)
        total_hubs = len(hub_neurons)
        concentration = 0.0
        if len(hubs_per_layer) > 1:
            counts = list(hubs_per_layer.values())
            mean_count = np.mean(counts)
            concentration = np.std(counts) / (mean_count + 1e-10)
        
        return {
            "num_layers_with_hubs": len(layers_with_hubs),
            "hub_concentration": concentration,
            "hubs_per_layer": hubs_per_layer
        }
    
    def _generate_centrality_insights(self, centrality_result: Dict[str, Any],
                                    structure_result: Dict[str, Any]) -> str:
        """Generate insights from centrality analysis."""
        if centrality_result['centrality_concentration'] > 0.7:
            return "High centralization - few neurons control information flow"
        elif len(centrality_result['hub_neurons']) == 0:
            return "No clear hub structure - distributed information flow"
        elif centrality_result['avg_betweenness'] > 0.1:
            return "Moderate centralization with multiple pathways"
        else:
            return "Well-distributed information flow"
    
    def _generate_spectral_insights(self, spectral_result: Dict[str, Any]) -> str:
        """Generate insights from spectral analysis."""
        if spectral_result['spectral_gap'] > 0.5:
            return "Good spectral gap - clear community structure"
        elif spectral_result['algebraic_connectivity'] < 0.01:
            return "Poor connectivity - network is fragile"
        elif spectral_result['spectral_radius'] > 10:
            return "High spectral radius - potential for instability"
        else:
            return "Balanced spectral properties"
    
    def _generate_path_insights(self, path_result: Dict[str, Any]) -> str:
        """Generate insights from path analysis."""
        if path_result['path_efficiency'] > 0.8:
            return "Efficient information flow with direct paths"
        elif path_result['avg_shortest_path'] > 10:
            return "Long paths - consider adding shortcuts"
        elif path_result['diameter'] > 20:
            return "Very large diameter - network is stretched"
        else:
            return "Reasonable path structure"
    
    def _get_health_status(self, health_score: float) -> str:
        """Get health status from score."""
        if health_score > 0.8:
            return "healthy"
        elif health_score > 0.6:
            return "good"
        elif health_score > 0.4:
            return "needs_attention"
        else:
            return "critical"
    
    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations based on issues."""
        recommendations = []
        
        if "extremely_sparse" in issues:
            recommendations.append("Increase network connectivity or add skip connections")
        
        if "highly_fragmented" in issues:
            recommendations.append("Add connections to bridge disconnected components")
        
        if "over_centralized" in issues:
            recommendations.append("Distribute connections to reduce hub dependency")
        
        if "inefficient_paths" in issues:
            recommendations.append("Add shortcut connections between distant layers")
        
        if "poor_spectral_gap" in issues:
            recommendations.append("Strengthen community structure with clustered connections")
        
        return recommendations
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis when graph cannot be built."""
        return {
            "graph_summary": {"error": "No active graph could be built"},
            "connectivity_analysis": {},
            "centrality_analysis": {},
            "spectral_analysis": {},
            "path_analysis": {},
            "network_diagnostics": {"health_score": 0.0, "status": "no_graph"},
            "architectural_insights": []
        }