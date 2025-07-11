"""
Topological analyzer component.

This analyzer combines multiple topological metrics to provide comprehensive
analysis of neural network architecture from a topological perspective.
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
    ExtremaMetric, PersistenceMetric, ConnectivityMetric,
    TopologicalSignatureMetric
)


class TopologicalAnalyzer(BaseAnalyzer):
    """
    Analyzes topological properties of neural networks.
    
    Combines extrema detection, persistence analysis, connectivity patterns,
    and topological signatures to provide insights for topology-aware
    architecture design.
    """
    
    def __init__(self, patch_size: int = 8, num_thresholds: int = 50,
                 resolution: int = 20, name: str = None):
        """
        Initialize topological analyzer.
        
        Args:
            patch_size: Size of patches for extrema analysis
            num_thresholds: Number of thresholds for persistence
            resolution: Resolution for topological signatures
            name: Optional custom name
        """
        super().__init__(name or "TopologicalAnalyzer")
        self.patch_size = patch_size
        self.num_thresholds = num_thresholds
        self.resolution = resolution
        
        # Initialize metrics
        self._extrema_metric = ExtremaMetric(patch_size=patch_size)
        self._persistence_metric = PersistenceMetric(num_thresholds=num_thresholds)
        self._connectivity_metric = ConnectivityMetric()
        self._signature_metric = TopologicalSignatureMetric(resolution=resolution)
        
        self._required_metrics = {
            "ExtremaMetric",
            "PersistenceMetric", 
            "ConnectivityMetric",
            "TopologicalSignatureMetric"
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"model"},
            provided_outputs={
                "analysis.topological_summary",
                "analysis.extrema_analysis",
                "analysis.persistence_analysis",
                "analysis.connectivity_patterns",
                "analysis.topological_recommendations",
                "analysis.patch_placement_suggestions"
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
        Perform comprehensive topological analysis.
        
        Args:
            model: Model to analyze
            report: Analysis report containing metric data
            context: Evolution context
            
        Returns:
            Dictionary containing analysis results
        """
        # Extract weight matrices from all layers
        layer_weights = self._extract_layer_weights(model)
        
        # Analyze each layer
        layer_analyses = {}
        for layer_name, weight_matrix in layer_weights.items():
            layer_analyses[layer_name] = self._analyze_layer_topology(
                layer_name, weight_matrix, report, context
            )
        
        # Aggregate results
        extrema_analysis = self._aggregate_extrema_analysis(layer_analyses)
        persistence_analysis = self._aggregate_persistence_analysis(layer_analyses)
        connectivity_patterns = self._aggregate_connectivity_patterns(layer_analyses)
        
        # Generate recommendations
        recommendations = self._generate_topological_recommendations(
            extrema_analysis, persistence_analysis, connectivity_patterns
        )
        
        # Generate patch placement suggestions
        patch_suggestions = self._generate_patch_placement_suggestions(
            layer_analyses, extrema_analysis
        )
        
        # Create summary
        summary = self._create_topological_summary(
            layer_analyses, extrema_analysis, persistence_analysis, connectivity_patterns
        )
        
        return {
            "topological_summary": summary,
            "extrema_analysis": extrema_analysis,
            "persistence_analysis": persistence_analysis,
            "connectivity_patterns": connectivity_patterns,
            "topological_recommendations": recommendations,
            "patch_placement_suggestions": patch_suggestions,
            "layer_topologies": layer_analyses
        }
    
    def _extract_layer_weights(self, model: IModel) -> Dict[str, torch.Tensor]:
        """Extract weight matrices from all layers."""
        weights = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight'):
                    weight = module.weight
                    if weight.dim() > 2:
                        weight = weight.flatten(0, -2)
                    weights[name] = weight
        
        return weights
    
    def _analyze_layer_topology(self, layer_name: str, weight_matrix: torch.Tensor,
                               report: AnalysisReport, 
                               context: EvolutionContext) -> Dict[str, Any]:
        """Analyze topology of a single layer."""
        # Create context for metrics
        metric_context = EvolutionContext({'weight_matrix': weight_matrix})
        
        # Run extrema metric
        extrema_key = f"ExtremaMetric_{layer_name}"
        if extrema_key not in report.metrics:
            extrema_result = self._extrema_metric.analyze(None, metric_context)
            report.add_metric_data(extrema_key, extrema_result)
        else:
            extrema_result = report.get(f"metrics.{extrema_key}")
        
        # Run persistence metric
        persistence_key = f"PersistenceMetric_{layer_name}"
        if persistence_key not in report.metrics:
            persistence_result = self._persistence_metric.analyze(None, metric_context)
            report.add_metric_data(persistence_key, persistence_result)
        else:
            persistence_result = report.get(f"metrics.{persistence_key}")
        
        # Run connectivity metric
        connectivity_key = f"ConnectivityMetric_{layer_name}"
        if connectivity_key not in report.metrics:
            connectivity_result = self._connectivity_metric.analyze(None, metric_context)
            report.add_metric_data(connectivity_key, connectivity_result)
        else:
            connectivity_result = report.get(f"metrics.{connectivity_key}")
        
        # Run signature metric
        signature_key = f"TopologicalSignatureMetric_{layer_name}"
        if signature_key not in report.metrics:
            signature_result = self._signature_metric.analyze(None, metric_context)
            report.add_metric_data(signature_key, signature_result)
        else:
            signature_result = report.get(f"metrics.{signature_key}")
        
        return {
            'extrema': extrema_result,
            'persistence': persistence_result,
            'connectivity': connectivity_result,
            'signature': signature_result
        }
    
    def _aggregate_extrema_analysis(self, 
                                   layer_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate extrema analysis across layers."""
        all_extrema = []
        total_extrema = 0
        extrema_densities = []
        
        for layer_name, analysis in layer_analyses.items():
            extrema_data = analysis['extrema']
            
            # Collect extrema points with layer info
            for extrema_point in extrema_data['extrema_points']:
                extrema_with_layer = extrema_point.copy()
                extrema_with_layer['layer'] = layer_name
                all_extrema.append(extrema_with_layer)
            
            total_extrema += extrema_data['num_extrema']
            extrema_densities.append(extrema_data['extrema_density'])
        
        # Sort by importance
        all_extrema.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # Find extrema clusters
        clusters = self._find_extrema_clusters(all_extrema)
        
        return {
            'total_extrema': total_extrema,
            'average_density': np.mean(extrema_densities) if extrema_densities else 0.0,
            'top_extrema': all_extrema[:10],  # Top 10 most important
            'extrema_clusters': clusters,
            'gradient_flow_indicators': self._analyze_gradient_flow_from_extrema(layer_analyses)
        }
    
    def _aggregate_persistence_analysis(self,
                                      layer_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate persistence analysis across layers."""
        all_features = []
        total_persistence = 0.0
        persistence_entropies = []
        
        for layer_name, analysis in layer_analyses.items():
            persistence_data = analysis['persistence']
            
            # Collect persistence features
            for feature in persistence_data['persistence_features']:
                feature_with_layer = feature.copy()
                feature_with_layer['layer'] = layer_name
                all_features.append(feature_with_layer)
            
            total_persistence += persistence_data['total_persistence']
            persistence_entropies.append(persistence_data['persistence_entropy'])
        
        # Analyze persistence patterns
        persistence_patterns = self._analyze_persistence_patterns(all_features)
        
        return {
            'total_features': len(all_features),
            'total_persistence': total_persistence,
            'average_entropy': np.mean(persistence_entropies) if persistence_entropies else 0.0,
            'persistence_patterns': persistence_patterns,
            'critical_features': [f for f in all_features if f['persistence'] == 'inf'],
            'feature_distribution': self._compute_feature_distribution(all_features)
        }
    
    def _aggregate_connectivity_patterns(self,
                                       layer_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate connectivity patterns across layers."""
        densities = []
        avg_degrees = []
        all_hubs = []
        clustering_coeffs = []
        
        for layer_name, analysis in layer_analyses.items():
            connectivity_data = analysis['connectivity']
            
            densities.append(connectivity_data['connectivity_density'])
            avg_degrees.append(connectivity_data['average_degree'])
            clustering_coeffs.append(connectivity_data['clustering_coefficient'])
            
            # Collect hub neurons
            for hub in connectivity_data['hub_neurons']:
                hub_with_layer = hub.copy()
                hub_with_layer['layer'] = layer_name
                all_hubs.append(hub_with_layer)
        
        # Network-wide connectivity analysis
        network_connectivity = self._analyze_network_connectivity(layer_analyses)
        
        return {
            'average_density': np.mean(densities) if densities else 0.0,
            'average_degree': np.mean(avg_degrees) if avg_degrees else 0.0,
            'average_clustering': np.mean(clustering_coeffs) if clustering_coeffs else 0.0,
            'total_hubs': len(all_hubs),
            'top_hubs': sorted(all_hubs, key=lambda x: x['total_degree'], reverse=True)[:10],
            'network_connectivity': network_connectivity
        }
    
    def _generate_topological_recommendations(self,
                                            extrema_analysis: Dict[str, Any],
                                            persistence_analysis: Dict[str, Any],
                                            connectivity_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on topological analysis."""
        recommendations = []
        
        # Extrema-based recommendations
        if extrema_analysis['average_density'] > 0.1:
            recommendations.append({
                'type': 'architecture',
                'action': 'consider_patch_based_architecture',
                'reason': f"High extrema density ({extrema_analysis['average_density']:.3f}) indicates structured weight patterns",
                'priority': 0.8
            })
        
        # Persistence-based recommendations
        if persistence_analysis['average_entropy'] > 2.0:
            recommendations.append({
                'type': 'regularization',
                'action': 'add_topological_regularization',
                'reason': f"High persistence entropy ({persistence_analysis['average_entropy']:.2f}) suggests complex topological structure",
                'priority': 0.7
            })
        
        if len(persistence_analysis['critical_features']) > 5:
            recommendations.append({
                'type': 'stability',
                'action': 'stabilize_persistent_features',
                'reason': f"{len(persistence_analysis['critical_features'])} infinite persistence features detected",
                'priority': 0.9
            })
        
        # Connectivity-based recommendations
        if connectivity_patterns['average_clustering'] < 0.1:
            recommendations.append({
                'type': 'connectivity',
                'action': 'increase_local_connectivity',
                'reason': "Low clustering coefficient indicates sparse local connections",
                'priority': 0.6
            })
        
        if connectivity_patterns['total_hubs'] > len(extrema_analysis['extrema_clusters']) * 2:
            recommendations.append({
                'type': 'architecture',
                'action': 'implement_hub_based_routing',
                'reason': "Significant hub structure detected in network",
                'priority': 0.7
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        return recommendations
    
    def _generate_patch_placement_suggestions(self,
                                            layer_analyses: Dict[str, Dict[str, Any]],
                                            extrema_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggestions for patch placement based on extrema."""
        suggestions = []
        
        # Group extrema by layer
        layer_extrema = {}
        for extrema in extrema_analysis['top_extrema']:
            layer = extrema['layer']
            if layer not in layer_extrema:
                layer_extrema[layer] = []
            layer_extrema[layer].append(extrema)
        
        # Generate suggestions for each layer
        for layer_name, extrema_list in layer_extrema.items():
            if len(extrema_list) >= 2:
                # Suggest patches around important extrema
                for extrema in extrema_list[:3]:  # Top 3 per layer
                    suggestions.append({
                        'layer': layer_name,
                        'position': extrema['position'],
                        'type': extrema['type'],
                        'importance': extrema['importance_score'],
                        'suggested_patch_size': self.patch_size,
                        'reason': f"{extrema['type']} with high importance score"
                    })
        
        # Add cluster-based suggestions
        for cluster in extrema_analysis['extrema_clusters']:
            if len(cluster['members']) >= 3:
                suggestions.append({
                    'layer': cluster['layer'],
                    'position': cluster['center'],
                    'type': 'cluster',
                    'importance': cluster['total_importance'],
                    'suggested_patch_size': self.patch_size * 2,  # Larger patch for clusters
                    'reason': f"Extrema cluster with {len(cluster['members'])} points"
                })
        
        # Sort by importance
        suggestions.sort(key=lambda x: x['importance'], reverse=True)
        
        return suggestions[:10]  # Top 10 suggestions
    
    def _create_topological_summary(self,
                                  layer_analyses: Dict[str, Dict[str, Any]],
                                  extrema_analysis: Dict[str, Any],
                                  persistence_analysis: Dict[str, Any],
                                  connectivity_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of topological analysis."""
        # Compute topological health score
        health_score = self._compute_topological_health(
            extrema_analysis, persistence_analysis, connectivity_patterns
        )
        
        # Identify dominant topological features
        dominant_features = []
        
        if extrema_analysis['average_density'] > 0.05:
            dominant_features.append('extrema_rich')
        
        if persistence_analysis['average_entropy'] > 1.5:
            dominant_features.append('topologically_complex')
        
        if connectivity_patterns['average_clustering'] > 0.3:
            dominant_features.append('highly_clustered')
        
        if connectivity_patterns['total_hubs'] > 10:
            dominant_features.append('hub_structured')
        
        return {
            'topological_health_score': health_score,
            'num_layers_analyzed': len(layer_analyses),
            'dominant_features': dominant_features,
            'total_extrema': extrema_analysis['total_extrema'],
            'total_persistence_features': persistence_analysis['total_features'],
            'average_connectivity': connectivity_patterns['average_density'],
            'assessment': self._get_topological_assessment(health_score, dominant_features)
        }
    
    def _find_extrema_clusters(self, extrema_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find clusters of extrema points."""
        clusters = []
        
        # Group by layer first
        layer_extrema = {}
        for extrema in extrema_points:
            layer = extrema['layer']
            if layer not in layer_extrema:
                layer_extrema[layer] = []
            layer_extrema[layer].append(extrema)
        
        # Find clusters within each layer
        for layer, points in layer_extrema.items():
            if len(points) < 2:
                continue
            
            # Simple clustering based on position proximity
            used = set()
            for i, point_i in enumerate(points):
                if i in used:
                    continue
                
                cluster_members = [point_i]
                used.add(i)
                
                # Find nearby points
                for j, point_j in enumerate(points):
                    if j <= i or j in used:
                        continue
                    
                    # Check if positions are close
                    pos_i = point_i['position']
                    pos_j = point_j['position']
                    distance = abs(pos_i[0] - pos_j[0]) + abs(pos_i[1] - pos_j[1])
                    
                    if distance < self.patch_size:
                        cluster_members.append(point_j)
                        used.add(j)
                
                if len(cluster_members) >= 2:
                    # Compute cluster center
                    center_row = int(np.mean([p['position'][0] for p in cluster_members]))
                    center_col = int(np.mean([p['position'][1] for p in cluster_members]))
                    
                    clusters.append({
                        'layer': layer,
                        'center': (center_row, center_col),
                        'members': cluster_members,
                        'size': len(cluster_members),
                        'total_importance': sum(p['importance_score'] for p in cluster_members)
                    })
        
        return clusters
    
    def _analyze_gradient_flow_from_extrema(self, 
                                          layer_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze gradient flow patterns from extrema data."""
        gradient_stats = []
        
        for analysis in layer_analyses.values():
            if 'gradient_statistics' in analysis['extrema']:
                gradient_stats.append(analysis['extrema']['gradient_statistics'])
        
        if not gradient_stats:
            return {'healthy_flow': False, 'issues': ['no_gradient_data']}
        
        # Aggregate statistics
        mean_gradients = [s['mean'] for s in gradient_stats]
        high_gradient_ratios = [s['high_gradient_ratio'] for s in gradient_stats]
        
        # Assess health
        issues = []
        if np.mean(mean_gradients) < 0.01:
            issues.append('vanishing_gradients')
        if np.mean(mean_gradients) > 10.0:
            issues.append('exploding_gradients')
        if np.std(mean_gradients) / (np.mean(mean_gradients) + 1e-8) > 2.0:
            issues.append('unstable_gradients')
        
        return {
            'healthy_flow': len(issues) == 0,
            'issues': issues,
            'average_gradient_magnitude': np.mean(mean_gradients),
            'gradient_stability': 1.0 / (1.0 + np.std(mean_gradients))
        }
    
    def _analyze_persistence_patterns(self, 
                                    features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in persistence features."""
        if not features:
            return {'dominant_pattern': 'none', 'pattern_strength': 0.0}
        
        # Count feature types
        type_counts = {}
        for feature in features:
            ftype = feature['feature_type']
            type_counts[ftype] = type_counts.get(ftype, 0) + 1
        
        # Find dominant type
        dominant_type = max(type_counts.items(), key=lambda x: x[1])[0]
        pattern_strength = type_counts[dominant_type] / len(features)
        
        # Analyze lifespans
        finite_features = [f for f in features if f['persistence'] != 'inf']
        if finite_features:
            avg_lifespan = np.mean([f['persistence'] for f in finite_features])
        else:
            avg_lifespan = 0.0
        
        return {
            'dominant_pattern': dominant_type,
            'pattern_strength': pattern_strength,
            'type_distribution': type_counts,
            'average_lifespan': avg_lifespan,
            'infinite_ratio': sum(1 for f in features if f['persistence'] == 'inf') / len(features)
        }
    
    def _compute_feature_distribution(self, 
                                    features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute distribution of persistence features."""
        if not features:
            return {'by_dimension': {}, 'by_layer': {}}
        
        # Distribution by dimension
        by_dimension = {}
        for feature in features:
            dim = feature['dimension']
            if dim not in by_dimension:
                by_dimension[dim] = 0
            by_dimension[dim] += 1
        
        # Distribution by layer
        by_layer = {}
        for feature in features:
            layer = feature['layer']
            if layer not in by_layer:
                by_layer[layer] = 0
            by_layer[layer] += 1
        
        return {
            'by_dimension': by_dimension,
            'by_layer': by_layer
        }
    
    def _analyze_network_connectivity(self, 
                                    layer_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze network-wide connectivity patterns."""
        # Collect all connectivity data
        all_densities = []
        all_sparsities = []
        power_law_indicators = []
        
        for analysis in layer_analyses.values():
            conn_data = analysis['connectivity']
            all_densities.append(conn_data['connectivity_density'])
            all_sparsities.append(conn_data['sparsity'])
            
            if 'connectivity_distribution' in conn_data:
                dist = conn_data['connectivity_distribution']
                if 'power_law_indicator' in dist:
                    power_law_indicators.append(dist['power_law_indicator'])
        
        # Determine network type
        avg_density = np.mean(all_densities)
        avg_sparsity = np.mean(all_sparsities)
        
        if avg_sparsity > 0.9:
            network_type = 'highly_sparse'
        elif avg_density > 0.5:
            network_type = 'densely_connected'
        elif power_law_indicators and np.mean(power_law_indicators) > 5.0:
            network_type = 'scale_free'
        else:
            network_type = 'regular'
        
        return {
            'network_type': network_type,
            'average_density': avg_density,
            'average_sparsity': avg_sparsity,
            'density_variance': np.std(all_densities),
            'connectivity_uniformity': 1.0 / (1.0 + np.std(all_densities))
        }
    
    def _compute_topological_health(self,
                                  extrema_analysis: Dict[str, Any],
                                  persistence_analysis: Dict[str, Any],
                                  connectivity_patterns: Dict[str, Any]) -> float:
        """Compute overall topological health score."""
        # Component scores
        extrema_score = min(1.0, extrema_analysis['average_density'] * 10)  # Normalize
        
        persistence_score = 1.0 / (1.0 + persistence_analysis['average_entropy'])  # Lower entropy is simpler
        
        connectivity_score = connectivity_patterns['average_clustering']  # Already 0-1
        
        gradient_flow = extrema_analysis['gradient_flow_indicators']
        flow_score = 1.0 if gradient_flow['healthy_flow'] else 0.5
        
        # Weighted combination
        health_score = (
            extrema_score * 0.2 +
            persistence_score * 0.3 +
            connectivity_score * 0.2 +
            flow_score * 0.3
        )
        
        return health_score
    
    def _get_topological_assessment(self, health_score: float, 
                                  dominant_features: List[str]) -> str:
        """Get assessment based on health score and features."""
        if health_score > 0.8:
            assessment = "excellent_topology"
        elif health_score > 0.6:
            assessment = "good_topology"
        elif health_score > 0.4:
            assessment = "needs_optimization"
        else:
            assessment = "poor_topology"
        
        # Modify based on dominant features
        if 'topologically_complex' in dominant_features:
            assessment += "_complex_structure"
        elif 'hub_structured' in dominant_features:
            assessment += "_hub_based"
        elif 'highly_clustered' in dominant_features:
            assessment += "_clustered"
        
        return assessment