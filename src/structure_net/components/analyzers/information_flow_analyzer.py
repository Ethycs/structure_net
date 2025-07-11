"""
Information flow analyzer component.

This analyzer combines multiple metrics to provide comprehensive
analysis of information flow through neural networks.
"""

from typing import Dict, Any, List, Tuple, Optional
import torch
import logging

from src.structure_net.core import (
    BaseAnalyzer, IModel, EvolutionContext, AnalysisReport,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)
from src.structure_net.components.metrics import (
    LayerMIMetric, EntropyMetric, InformationFlowMetric,
    RedundancyMetric, AdvancedMIMetric
)


class InformationFlowAnalyzer(BaseAnalyzer):
    """
    Analyzes information flow through neural networks.
    
    Combines mutual information, entropy, flow efficiency, and redundancy
    metrics to identify bottlenecks, inefficiencies, and optimization
    opportunities in network architecture.
    """
    
    def __init__(self, mi_threshold: float = 0.5, 
                 efficiency_threshold: float = 0.7,
                 redundancy_threshold: float = 0.3,
                 name: str = None):
        """
        Initialize information flow analyzer.
        
        Args:
            mi_threshold: Threshold for identifying MI bottlenecks
            efficiency_threshold: Threshold for poor information efficiency
            redundancy_threshold: Threshold for excessive redundancy
            name: Optional custom name
        """
        super().__init__(name or "InformationFlowAnalyzer")
        self.mi_threshold = mi_threshold
        self.efficiency_threshold = efficiency_threshold
        self.redundancy_threshold = redundancy_threshold
        
        # Initialize metrics
        self._mi_metric = LayerMIMetric()
        self._entropy_metric = EntropyMetric()
        self._flow_metric = InformationFlowMetric()
        self._redundancy_metric = RedundancyMetric()
        self._advanced_mi_metric = AdvancedMIMetric()
        
        self._required_metrics = {
            "LayerMIMetric",
            "EntropyMetric", 
            "InformationFlowMetric",
            "RedundancyMetric"
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"model", "layer_activations"},
            provided_outputs={
                "analysis.information_bottlenecks",
                "analysis.information_flow_map",
                "analysis.layer_importance",
                "analysis.redundancy_analysis",
                "analysis.recommendations"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def _perform_analysis(self, model: IModel, report: AnalysisReport, 
                         context: EvolutionContext) -> Dict[str, Any]:
        """
        Perform comprehensive information flow analysis.
        
        Args:
            model: Model to analyze
            report: Analysis report containing metric data
            context: Evolution context with layer activations
            
        Returns:
            Dictionary containing analysis results
        """
        # Get layer activations
        layer_activations = context.get('layer_activations')
        if layer_activations is None:
            raise ValueError("InformationFlowAnalyzer requires 'layer_activations' in context")
        
        # Run metrics if not in report
        if "LayerMIMetric" not in report.metrics:
            mi_result = self._mi_metric.analyze(model, context)
            report.add_metric_data("LayerMIMetric", mi_result)
        
        if "InformationFlowMetric" not in report.metrics:
            flow_result = self._flow_metric.analyze(model, context)
            report.add_metric_data("InformationFlowMetric", flow_result)
        
        if "RedundancyMetric" not in report.metrics:
            redundancy_result = self._redundancy_metric.analyze(model, context)
            report.add_metric_data("RedundancyMetric", redundancy_result)
        
        # Get metric data from report
        mi_data = report.get("metrics.LayerMIMetric", {})
        flow_data = report.get("metrics.InformationFlowMetric", {})
        redundancy_data = report.get("metrics.RedundancyMetric", {})
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(mi_data, flow_data)
        
        # Create flow map
        flow_map = self._create_flow_map(mi_data, flow_data, redundancy_data)
        
        # Compute layer importance
        layer_importance = self._compute_layer_importance(flow_map, bottlenecks)
        
        # Analyze redundancy patterns
        redundancy_analysis = self._analyze_redundancy(redundancy_data, flow_data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            bottlenecks, flow_map, redundancy_analysis
        )
        
        return {
            "information_bottlenecks": bottlenecks,
            "information_flow_map": flow_map,
            "layer_importance": layer_importance,
            "redundancy_analysis": redundancy_analysis,
            "recommendations": recommendations,
            "summary": self._create_summary(bottlenecks, flow_map, redundancy_analysis)
        }
    
    def _identify_bottlenecks(self, mi_data: Dict[str, Any], 
                             flow_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify information bottlenecks in the network."""
        bottlenecks = []
        
        # Check MI values
        layer_pairs = mi_data.get("layer_pairs", {})
        for pair, mi_value in layer_pairs.items():
            if mi_value < self.mi_threshold:
                bottlenecks.append({
                    "type": "low_mi",
                    "location": pair,
                    "value": mi_value,
                    "severity": "high" if mi_value < self.mi_threshold * 0.5 else "medium"
                })
        
        # Check flow efficiency
        layer_flow = flow_data.get("layer_flow", {})
        for pair, flow_info in layer_flow.items():
            efficiency = flow_info.get("efficiency", 1.0)
            if efficiency < self.efficiency_threshold:
                bottlenecks.append({
                    "type": "low_efficiency",
                    "location": pair,
                    "value": efficiency,
                    "severity": "high" if efficiency < self.efficiency_threshold * 0.5 else "medium"
                })
        
        return bottlenecks
    
    def _create_flow_map(self, mi_data: Dict[str, Any], 
                        flow_data: Dict[str, Any],
                        redundancy_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Create comprehensive information flow map."""
        flow_map = {}
        
        layer_pairs = mi_data.get("layer_pairs", {})
        layer_flow = flow_data.get("layer_flow", {})
        layer_redundancy = redundancy_data.get("layer_redundancy", {})
        
        # Combine all data sources
        all_pairs = set(layer_pairs.keys()) | set(layer_flow.keys()) | set(layer_redundancy.keys())
        
        for pair in all_pairs:
            flow_map[pair] = {
                "mi": layer_pairs.get(pair, 0.0),
                "efficiency": layer_flow.get(pair, {}).get("efficiency", 0.0),
                "gap": layer_flow.get(pair, {}).get("gap", 0.0),
                "utilization": layer_flow.get(pair, {}).get("utilization", 0.0),
                "redundancy": layer_redundancy.get(pair, 0.0)
            }
            
            # Compute health score
            health_score = (
                flow_map[pair]["mi"] / self.mi_threshold * 0.3 +
                flow_map[pair]["efficiency"] / self.efficiency_threshold * 0.3 +
                flow_map[pair]["utilization"] * 0.2 +
                (1 - flow_map[pair]["redundancy"] / self.redundancy_threshold) * 0.2
            )
            flow_map[pair]["health_score"] = min(1.0, health_score)
        
        return flow_map
    
    def _compute_layer_importance(self, flow_map: Dict[str, Dict[str, Any]], 
                                 bottlenecks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute importance score for each layer based on information flow."""
        layer_importance = {}
        
        # Count connections and aggregate scores
        for pair, flow_info in flow_map.items():
            if "->" in pair:
                source, target = pair.split("->")
                
                # Source layer importance (how well it transmits info)
                if source not in layer_importance:
                    layer_importance[source] = {"out_score": 0, "out_count": 0, "in_score": 0, "in_count": 0}
                
                layer_importance[source]["out_score"] += flow_info["health_score"]
                layer_importance[source]["out_count"] += 1
                
                # Target layer importance (how well it receives info)
                if target not in layer_importance:
                    layer_importance[target] = {"out_score": 0, "out_count": 0, "in_score": 0, "in_count": 0}
                
                layer_importance[target]["in_score"] += flow_info["health_score"]
                layer_importance[target]["in_count"] += 1
        
        # Compute final importance scores
        final_scores = {}
        for layer, scores in layer_importance.items():
            out_avg = scores["out_score"] / scores["out_count"] if scores["out_count"] > 0 else 0
            in_avg = scores["in_score"] / scores["in_count"] if scores["in_count"] > 0 else 0
            
            # Combined importance
            final_scores[layer] = (out_avg + in_avg) / 2
            
            # Penalize layers involved in bottlenecks
            bottleneck_count = sum(1 for b in bottlenecks if layer in b["location"])
            if bottleneck_count > 0:
                final_scores[layer] *= (1 - 0.1 * bottleneck_count)
        
        return final_scores
    
    def _analyze_redundancy(self, redundancy_data: Dict[str, Any], 
                           flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze redundancy patterns in the network."""
        layer_redundancy = redundancy_data.get("layer_redundancy", {})
        avg_redundancy = redundancy_data.get("redundancy", 0.0)
        
        # Find high redundancy pairs
        high_redundancy_pairs = [
            pair for pair, value in layer_redundancy.items() 
            if value > self.redundancy_threshold
        ]
        
        # Compute redundancy efficiency (is redundancy useful?)
        redundancy_efficiency = {}
        layer_flow = flow_data.get("layer_flow", {})
        
        for pair in high_redundancy_pairs:
            flow_info = layer_flow.get(pair, {})
            efficiency = flow_info.get("efficiency", 0.0)
            
            # High redundancy with low efficiency is bad
            redundancy_efficiency[pair] = {
                "redundancy": layer_redundancy[pair],
                "efficiency": efficiency,
                "is_problematic": efficiency < self.efficiency_threshold
            }
        
        return {
            "average_redundancy": avg_redundancy,
            "high_redundancy_pairs": high_redundancy_pairs,
            "redundancy_efficiency": redundancy_efficiency,
            "excessive_redundancy": avg_redundancy > self.redundancy_threshold
        }
    
    def _generate_recommendations(self, bottlenecks: List[Dict[str, Any]], 
                                 flow_map: Dict[str, Dict[str, Any]],
                                 redundancy_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Bottleneck recommendations
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "low_mi":
                recommendations.append({
                    "action": "increase_capacity",
                    "location": bottleneck["location"],
                    "reason": f"Low mutual information ({bottleneck['value']:.3f}) indicates information bottleneck",
                    "confidence": 0.8 if bottleneck["severity"] == "high" else 0.6
                })
            elif bottleneck["type"] == "low_efficiency":
                recommendations.append({
                    "action": "improve_connectivity",
                    "location": bottleneck["location"],
                    "reason": f"Poor information efficiency ({bottleneck['value']:.2%}) suggests weak connections",
                    "confidence": 0.7
                })
        
        # Redundancy recommendations
        problematic_redundancy = [
            pair for pair, info in redundancy_analysis["redundancy_efficiency"].items()
            if info["is_problematic"]
        ]
        
        for pair in problematic_redundancy:
            recommendations.append({
                "action": "reduce_redundancy",
                "location": pair,
                "reason": "High redundancy with low efficiency wastes capacity",
                "confidence": 0.7
            })
        
        # Global recommendations
        if redundancy_analysis["excessive_redundancy"]:
            recommendations.append({
                "action": "global_architecture_review",
                "location": "network",
                "reason": f"Overall redundancy ({redundancy_analysis['average_redundancy']:.3f}) is excessive",
                "confidence": 0.6
            })
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        
        return recommendations
    
    def _create_summary(self, bottlenecks: List[Dict[str, Any]], 
                       flow_map: Dict[str, Dict[str, Any]],
                       redundancy_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create analysis summary."""
        # Calculate average health score
        health_scores = [info["health_score"] for info in flow_map.values()]
        avg_health = sum(health_scores) / len(health_scores) if health_scores else 0.0
        
        # Count severe issues
        severe_bottlenecks = sum(1 for b in bottlenecks if b.get("severity") == "high")
        
        return {
            "overall_health": avg_health,
            "num_bottlenecks": len(bottlenecks),
            "severe_bottlenecks": severe_bottlenecks,
            "excessive_redundancy": redundancy_analysis["excessive_redundancy"],
            "health_assessment": self._assess_health(avg_health, len(bottlenecks))
        }
    
    def _assess_health(self, avg_health: float, num_bottlenecks: int) -> str:
        """Assess overall information flow health."""
        if avg_health > 0.8 and num_bottlenecks == 0:
            return "excellent"
        elif avg_health > 0.6 and num_bottlenecks < 3:
            return "good"
        elif avg_health > 0.4:
            return "fair"
        else:
            return "poor"