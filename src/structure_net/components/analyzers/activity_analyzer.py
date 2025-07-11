"""
Activity analyzer component.

This analyzer combines multiple activity-related metrics to provide comprehensive
analysis of neural network activation patterns and layer health.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import logging
from collections import defaultdict

from src.structure_net.core import (
    BaseAnalyzer, IModel, ILayer, EvolutionContext, AnalysisReport,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)
from src.structure_net.components.metrics import (
    NeuronActivityMetric, ActivationDistributionMetric,
    ActivityPatternMetric, LayerHealthMetric
)


class ActivityAnalyzer(BaseAnalyzer):
    """
    Analyzes neural network activity patterns comprehensively.
    
    Combines neuron activity, activation distributions, pattern analysis,
    and health metrics to provide insights about network behavior and
    identify issues like dead neurons, saturation, and poor diversity.
    """
    
    def __init__(self, activation_threshold: float = 0.01,
                 saturation_threshold: float = 10.0,
                 track_history: bool = True,
                 name: str = None):
        """
        Initialize activity analyzer.
        
        Args:
            activation_threshold: Threshold for active neurons
            saturation_threshold: Threshold for saturation detection
            track_history: Whether to track activation history
            name: Optional custom name
        """
        super().__init__(name or "ActivityAnalyzer")
        self.activation_threshold = activation_threshold
        self.saturation_threshold = saturation_threshold
        self.track_history = track_history
        
        # Initialize metrics
        self._activity_metric = NeuronActivityMetric(activation_threshold)
        self._distribution_metric = ActivationDistributionMetric(saturation_threshold)
        self._pattern_metric = ActivityPatternMetric(activation_threshold)
        self._health_metric = LayerHealthMetric()
        
        self._required_metrics = {
            "NeuronActivityMetric",
            "ActivationDistributionMetric",
            "ActivityPatternMetric",
            "LayerHealthMetric"
        }
        
        # History tracking
        if track_history:
            self.activation_history = defaultdict(list)
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"model", "data_loader"},
            optional_inputs={"num_batches"},
            provided_outputs={
                "analysis.layer_activities",
                "analysis.dead_neurons",
                "analysis.saturated_neurons",
                "analysis.activity_summary",
                "analysis.health_report",
                "analysis.temporal_trends"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=False,
                parallel_safe=False
            )
        )
    
    def _perform_analysis(self, model: IModel, report: AnalysisReport, 
                         context: EvolutionContext) -> Dict[str, Any]:
        """
        Perform comprehensive activity analysis.
        
        Args:
            model: Model to analyze
            report: Analysis report containing metric data
            context: Evolution context with data loader
            
        Returns:
            Dictionary containing analysis results
        """
        # Get data loader
        data_loader = context.get('data_loader')
        if data_loader is None:
            raise ValueError("ActivityAnalyzer requires 'data_loader' in context")
        
        num_batches = context.get('num_batches', 10)
        
        # Collect activations from all layers
        layer_activations = self._collect_layer_activations(model, data_loader, num_batches)
        
        # Analyze each layer
        layer_analyses = {}
        for layer_name, activations in layer_activations.items():
            layer_analyses[layer_name] = self._analyze_layer_activity(
                layer_name, activations, report, context
            )
            
            # Track history if enabled
            if self.track_history:
                self.activation_history[layer_name].append(activations.detach())
        
        # Aggregate results
        dead_neurons = self._identify_dead_neurons(layer_analyses)
        saturated_neurons = self._identify_saturated_neurons(layer_analyses)
        activity_summary = self._create_activity_summary(layer_analyses)
        health_report = self._create_health_report(layer_analyses)
        
        # Temporal analysis if we have history
        temporal_trends = {}
        if self.track_history and any(len(h) > 1 for h in self.activation_history.values()):
            temporal_trends = self._analyze_temporal_trends()
        
        return {
            "layer_activities": layer_analyses,
            "dead_neurons": dead_neurons,
            "saturated_neurons": saturated_neurons,
            "activity_summary": activity_summary,
            "health_report": health_report,
            "temporal_trends": temporal_trends
        }
    
    def _collect_layer_activations(self, model: IModel, data_loader, 
                                  num_batches: int) -> Dict[str, torch.Tensor]:
        """Collect activations from all layers."""
        activations = {}
        hooks = []
        
        # Register forward hooks
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                def get_activation(name):
                    def hook(module, input, output):
                        # Store activations for this layer
                        if name not in activations:
                            activations[name] = []
                        
                        # Flatten to 2D if needed
                        act = output.detach()
                        if act.dim() > 2:
                            act = act.flatten(1)
                        activations[name].append(act)
                    
                    return hook
                
                hooks.append(module.register_forward_hook(get_activation(name)))
        
        # Run forward passes
        model.eval()
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                if batch_idx >= num_batches:
                    break
                
                data = data.to(device)
                _ = model(data)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Concatenate activations across batches
        for layer_name in activations:
            activations[layer_name] = torch.cat(activations[layer_name], dim=0)
        
        return activations
    
    def _analyze_layer_activity(self, layer_name: str, activations: torch.Tensor,
                               report: AnalysisReport, 
                               context: EvolutionContext) -> Dict[str, Any]:
        """Analyze activity for a single layer."""
        # Create context for metrics
        metric_context = EvolutionContext({'activations': activations})
        
        # Run activity metric
        activity_key = f"NeuronActivityMetric_{layer_name}"
        if activity_key not in report.metrics:
            activity_result = self._activity_metric.analyze(None, metric_context)
            report.add_metric_data(activity_key, activity_result)
        else:
            activity_result = report.get(f"metrics.{activity_key}")
        
        # Run distribution metric
        distribution_key = f"ActivationDistributionMetric_{layer_name}"
        if distribution_key not in report.metrics:
            distribution_result = self._distribution_metric.analyze(None, metric_context)
            report.add_metric_data(distribution_key, distribution_result)
        else:
            distribution_result = report.get(f"metrics.{distribution_key}")
        
        # Run pattern metric
        pattern_key = f"ActivityPatternMetric_{layer_name}"
        if pattern_key not in report.metrics:
            pattern_result = self._pattern_metric.analyze(None, metric_context)
            report.add_metric_data(pattern_key, pattern_result)
        else:
            pattern_result = report.get(f"metrics.{pattern_key}")
        
        # Prepare context for health metric
        health_context = EvolutionContext({
            'metrics.active_ratio': activity_result['active_ratio'],
            'metrics.max_activation': distribution_result['max_activation'],
            'metrics.activity_entropy': pattern_result['activity_entropy'],
            'metrics.saturation_ratio': distribution_result['saturation_ratio'],
            'metrics.activity_gini': pattern_result['activity_gini']
        })
        
        # Run health metric
        health_key = f"LayerHealthMetric_{layer_name}"
        if health_key not in report.metrics:
            health_result = self._health_metric.analyze(None, health_context)
            report.add_metric_data(health_key, health_result)
        else:
            health_result = report.get(f"metrics.{health_key}")
        
        # Combine all results
        return {
            'activity': activity_result,
            'distribution': distribution_result,
            'patterns': pattern_result,
            'health': health_result
        }
    
    def _identify_dead_neurons(self, layer_analyses: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify dead neurons across all layers."""
        dead_neurons = []
        
        for layer_name, analysis in layer_analyses.items():
            activity_data = analysis['activity']
            
            # Get neuron activity rates
            neuron_rates = activity_data.get('neuron_activity_rates', [])
            
            for idx, rate in enumerate(neuron_rates):
                if rate < self.activation_threshold:
                    dead_neurons.append({
                        'layer': layer_name,
                        'neuron_index': idx,
                        'activity_rate': rate
                    })
        
        return dead_neurons
    
    def _identify_saturated_neurons(self, layer_analyses: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify saturated neurons across all layers."""
        saturated_neurons = []
        
        for layer_name, analysis in layer_analyses.items():
            distribution_data = analysis['distribution']
            
            if distribution_data['saturated_neurons'] > 0:
                # We don't have per-neuron saturation data, so report layer-level
                saturated_neurons.append({
                    'layer': layer_name,
                    'count': distribution_data['saturated_neurons'],
                    'ratio': distribution_data['saturation_ratio'],
                    'max_activation': distribution_data['max_activation']
                })
        
        return saturated_neurons
    
    def _create_activity_summary(self, layer_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary of activity across all layers."""
        total_neurons = 0
        total_active = 0
        total_dead = 0
        total_saturated = 0
        
        layer_summaries = []
        
        for layer_name, analysis in layer_analyses.items():
            activity = analysis['activity']
            distribution = analysis['distribution']
            
            total_neurons += activity['total_neurons']
            total_active += activity['active_neurons']
            total_dead += int(activity['total_neurons'] * activity['dead_ratio'])
            total_saturated += distribution['saturated_neurons']
            
            layer_summaries.append({
                'layer': layer_name,
                'total_neurons': activity['total_neurons'],
                'active_ratio': activity['active_ratio'],
                'health_score': analysis['health']['layer_health_score'],
                'diagnosis': analysis['health']['health_diagnosis']
            })
        
        return {
            'total_neurons': total_neurons,
            'total_active': total_active,
            'total_dead': total_dead,
            'total_saturated': total_saturated,
            'overall_active_ratio': total_active / total_neurons if total_neurons > 0 else 0,
            'overall_dead_ratio': total_dead / total_neurons if total_neurons > 0 else 0,
            'overall_saturation_ratio': total_saturated / total_neurons if total_neurons > 0 else 0,
            'layer_summaries': layer_summaries
        }
    
    def _create_health_report(self, layer_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create overall health report."""
        health_scores = []
        issues = defaultdict(list)
        
        for layer_name, analysis in layer_analyses.items():
            health = analysis['health']
            health_scores.append(health['layer_health_score'])
            
            # Track issues
            if health['health_diagnosis'] != 'healthy':
                issues[health['health_diagnosis']].append(layer_name)
        
        # Overall health
        overall_health = sum(health_scores) / len(health_scores) if health_scores else 0
        
        # Determine status
        if overall_health > 0.8:
            status = 'excellent'
        elif overall_health > 0.6:
            status = 'good'
        elif overall_health > 0.4:
            status = 'needs_attention'
        else:
            status = 'critical'
        
        return {
            'overall_health_score': overall_health,
            'status': status,
            'num_healthy_layers': sum(1 for s in health_scores if s > 0.8),
            'num_unhealthy_layers': sum(1 for s in health_scores if s < 0.5),
            'issues': dict(issues),
            'recommendations': self._generate_recommendations(issues, overall_health)
        }
    
    def _generate_recommendations(self, issues: Dict[str, List[str]], 
                                overall_health: float) -> List[str]:
        """Generate recommendations based on issues."""
        recommendations = []
        
        if 'low_activity' in issues:
            recommendations.append(
                f"Increase learning rate or reduce regularization for layers: {issues['low_activity']}"
            )
        
        if 'saturation_risk' in issues:
            recommendations.append(
                f"Add gradient clipping or batch normalization for layers: {issues['saturation_risk']}"
            )
        
        if 'poor_diversity' in issues:
            recommendations.append(
                f"Consider dropout or noise injection for layers: {issues['poor_diversity']}"
            )
        
        if overall_health < 0.5:
            recommendations.append(
                "Overall network health is poor. Consider architectural changes or training adjustments."
            )
        
        return recommendations
    
    def _analyze_temporal_trends(self) -> Dict[str, Any]:
        """Analyze temporal trends in activation history."""
        trends = {}
        
        for layer_name, history in self.activation_history.items():
            if len(history) < 2:
                continue
            
            # Compute trends over time
            activity_ratios = []
            mean_activations = []
            
            for acts in history:
                active_mask = acts.abs() > self.activation_threshold
                activity_ratios.append(active_mask.any(dim=0).float().mean().item())
                mean_activations.append(acts.abs().mean().item())
            
            # Convert to tensors for trend analysis
            activity_ratios = torch.tensor(activity_ratios)
            mean_activations = torch.tensor(mean_activations)
            
            # Compute linear trends
            time_steps = torch.arange(len(activity_ratios), dtype=torch.float)
            activity_trend = self._compute_linear_trend(time_steps, activity_ratios)
            activation_trend = self._compute_linear_trend(time_steps, mean_activations)
            
            trends[layer_name] = {
                'activity_trend': activity_trend,
                'activation_trend': activation_trend,
                'activity_stability': 1.0 / (activity_ratios.std().item() + 1e-10),
                'activation_stability': 1.0 / (mean_activations.std().item() + 1e-10),
                'history_length': len(history),
                'improving': activity_trend > 0 and activation_trend > 0
            }
        
        return trends
    
    def _compute_linear_trend(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute linear trend (slope) between x and y."""
        if len(x) < 2:
            return 0.0
        
        x_mean = x.mean()
        y_mean = y.mean()
        
        numerator = torch.sum((x - x_mean) * (y - y_mean))
        denominator = torch.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        return (numerator / denominator).item()
    
    def clear_history(self, layer_name: str = None):
        """Clear activation history for a layer or all layers."""
        if layer_name is not None:
            self.activation_history[layer_name].clear()
        else:
            self.activation_history.clear()