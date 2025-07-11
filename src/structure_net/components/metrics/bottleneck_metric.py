"""
Bottleneck metric component.

This component identifies and quantifies information bottlenecks in neural networks,
helping to locate architectural constraints that limit performance.
"""

from typing import Dict, Any, Union, Optional, List, Tuple
import torch
import torch.nn as nn
import logging

from src.structure_net.core import (
    BaseMetric, ILayer, IModel, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)


class BottleneckMetric(BaseMetric):
    """
    Identifies and measures bottlenecks in neural network architectures.
    
    Bottlenecks occur when information flow is severely restricted,
    often due to low neuron activity or poor gradient flow.
    """
    
    def __init__(self, activation_threshold: float = 0.01, 
                 critical_threshold: float = 0.001,
                 name: str = None):
        """
        Initialize bottleneck metric.
        
        Args:
            activation_threshold: Threshold for active neurons
            critical_threshold: Threshold for critical bottlenecks
            name: Optional custom name
        """
        super().__init__(name or "BottleneckMetric")
        self.activation_threshold = activation_threshold
        self.critical_threshold = critical_threshold
        self._measurement_schema = {
            "bottleneck_score": float,
            "bottleneck_severity": str,
            "information_flow_ratio": float,
            "gradient_blockage": float,
            "suggested_action": str,
            "intervention_priority": float
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"activations", "gradients"},
            provided_outputs={
                "metrics.bottleneck_score",
                "metrics.bottleneck_severity",
                "metrics.information_flow_ratio",
                "metrics.gradient_blockage",
                "metrics.suggested_action",
                "metrics.intervention_priority"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def _compute_metric(self, target: Union[ILayer, IModel], 
                       context: EvolutionContext) -> Dict[str, Any]:
        """
        Compute bottleneck metrics.
        
        Args:
            target: Layer or model to analyze
            context: Must contain 'activations' and 'gradients'
            
        Returns:
            Dictionary containing bottleneck measurements
        """
        # Get required data
        activations = context.get('activations')
        gradients = context.get('gradients')
        
        if activations is None or gradients is None:
            raise ValueError("BottleneckMetric requires 'activations' and 'gradients' in context")
        
        # Handle different input formats
        if isinstance(activations, dict):
            # Multiple layers - analyze each
            return self._compute_multi_layer_bottlenecks(activations, gradients)
        else:
            # Single layer
            return self._compute_single_layer_bottleneck(activations, gradients)
    
    def _compute_single_layer_bottleneck(self, activations: torch.Tensor,
                                       gradients: torch.Tensor) -> Dict[str, Any]:
        """Compute bottleneck metrics for a single layer."""
        # Ensure 2D tensors
        if activations.dim() > 2:
            activations = activations.flatten(1)
        if gradients.dim() > 2:
            gradients = gradients.flatten(1)
        
        # Compute active neuron ratios
        active_mask = activations.abs() > self.activation_threshold
        active_ratio = active_mask.float().mean().item()
        
        # Critical bottleneck detection
        critical_mask = activations.abs() > self.critical_threshold
        critical_ratio = critical_mask.float().mean().item()
        
        # Gradient flow analysis
        grad_norms = gradients.norm(dim=1)
        mean_grad_norm = grad_norms.mean().item()
        
        # Gradient blockage (neurons with very small gradients)
        blocked_gradients = (grad_norms < 1e-7).float().mean().item()
        
        # Compute bottleneck score
        if critical_ratio < 0.001:
            bottleneck_score = float('inf')  # Critical bottleneck
            severity = "critical"
        else:
            # Combine multiple factors
            bottleneck_score = (
                (1 - active_ratio) * 0.4 +  # Low activity
                blocked_gradients * 0.3 +    # Blocked gradients
                (1 / (mean_grad_norm + 1e-8)) * 0.3  # Weak gradients
            )
            
            if bottleneck_score > 2.0:
                severity = "severe"
            elif bottleneck_score > 1.0:
                severity = "moderate"
            elif bottleneck_score > 0.5:
                severity = "mild"
            else:
                severity = "none"
        
        # Information flow ratio
        information_flow_ratio = active_ratio * (1 - blocked_gradients)
        
        # Determine suggested action
        suggested_action = self._determine_suggested_action(
            bottleneck_score, severity, active_ratio, blocked_gradients
        )
        
        # Intervention priority
        intervention_priority = self._compute_intervention_priority(
            bottleneck_score, severity
        )
        
        self.log(logging.DEBUG, 
                f"Bottleneck analysis: score={bottleneck_score:.3f}, "
                f"severity={severity}, flow={information_flow_ratio:.3f}")
        
        return {
            "bottleneck_score": bottleneck_score,
            "bottleneck_severity": severity,
            "information_flow_ratio": information_flow_ratio,
            "gradient_blockage": blocked_gradients,
            "suggested_action": suggested_action,
            "intervention_priority": intervention_priority
        }
    
    def _compute_multi_layer_bottlenecks(self, activations: Dict[str, torch.Tensor],
                                       gradients: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute bottleneck metrics for multiple layers."""
        layer_results = {}
        worst_score = 0.0
        worst_layer = None
        total_flow = 0.0
        
        for layer_name in activations:
            if layer_name in gradients:
                result = self._compute_single_layer_bottleneck(
                    activations[layer_name],
                    gradients[layer_name]
                )
                layer_results[layer_name] = result
                
                # Track worst bottleneck
                score = result["bottleneck_score"]
                if score == float('inf') or score > worst_score:
                    worst_score = score
                    worst_layer = layer_name
                
                total_flow += result["information_flow_ratio"]
        
        # Overall metrics
        num_layers = len(layer_results)
        avg_flow = total_flow / num_layers if num_layers > 0 else 0.0
        
        # Count bottlenecks by severity
        severity_counts = {
            "critical": 0,
            "severe": 0,
            "moderate": 0,
            "mild": 0,
            "none": 0
        }
        
        for result in layer_results.values():
            severity_counts[result["bottleneck_severity"]] += 1
        
        return {
            "layer_bottlenecks": layer_results,
            "worst_bottleneck_score": worst_score,
            "worst_bottleneck_layer": worst_layer,
            "average_information_flow": avg_flow,
            "bottleneck_severity_counts": severity_counts,
            "global_intervention_priority": max(
                r["intervention_priority"] for r in layer_results.values()
            ) if layer_results else 0.0
        }
    
    def _determine_suggested_action(self, score: float, severity: str,
                                  active_ratio: float, blocked_gradients: float) -> str:
        """Determine suggested action based on bottleneck analysis."""
        if severity == "critical":
            return "emergency_intervention"
        elif severity == "severe":
            if active_ratio < 0.1:
                return "add_bypass_connections"
            else:
                return "increase_layer_width"
        elif severity == "moderate":
            if blocked_gradients > 0.5:
                return "improve_initialization"
            else:
                return "add_normalization"
        elif severity == "mild":
            return "monitor_and_optimize"
        else:
            return "no_action_needed"
    
    def _compute_intervention_priority(self, score: float, severity: str) -> float:
        """Compute intervention priority (0-1)."""
        if score == float('inf'):
            return 1.0
        
        severity_weights = {
            "critical": 1.0,
            "severe": 0.8,
            "moderate": 0.5,
            "mild": 0.2,
            "none": 0.0
        }
        
        base_priority = severity_weights.get(severity, 0.0)
        
        # Adjust based on score magnitude
        if score > 3.0:
            return min(1.0, base_priority + 0.2)
        else:
            return base_priority
    
    def identify_bottleneck_patterns(self, 
                                   layer_sequence: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """
        Identify bottleneck patterns across a sequence of layers.
        
        Args:
            layer_sequence: List of dicts with 'activations' and 'gradients'
            
        Returns:
            Dictionary with bottleneck pattern analysis
        """
        bottleneck_scores = []
        flow_ratios = []
        
        for layer_data in layer_sequence:
            result = self._compute_single_layer_bottleneck(
                layer_data['activations'],
                layer_data['gradients']
            )
            bottleneck_scores.append(result['bottleneck_score'])
            flow_ratios.append(result['information_flow_ratio'])
        
        # Identify patterns
        increasing_bottleneck = all(
            bottleneck_scores[i] <= bottleneck_scores[i+1] 
            for i in range(len(bottleneck_scores)-1)
        )
        
        decreasing_flow = all(
            flow_ratios[i] >= flow_ratios[i+1] 
            for i in range(len(flow_ratios)-1)
        )
        
        # Find bottleneck clusters
        clusters = []
        current_cluster = []
        
        for i, score in enumerate(bottleneck_scores):
            if score > 1.0:  # Bottleneck threshold
                current_cluster.append(i)
            elif current_cluster:
                clusters.append(current_cluster)
                current_cluster = []
        
        if current_cluster:
            clusters.append(current_cluster)
        
        return {
            "bottleneck_scores": bottleneck_scores,
            "flow_ratios": flow_ratios,
            "increasing_bottleneck_pattern": increasing_bottleneck,
            "decreasing_flow_pattern": decreasing_flow,
            "bottleneck_clusters": clusters,
            "num_bottlenecks": sum(1 for s in bottleneck_scores if s > 1.0),
            "avg_bottleneck_score": sum(bottleneck_scores) / len(bottleneck_scores) if bottleneck_scores else 0.0
        }