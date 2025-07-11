"""
Sensitivity analyzer component.

This analyzer combines gradient sensitivity and bottleneck metrics to provide
comprehensive analysis of neural network sensitivity and architectural constraints.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from src.structure_net.core import (
    BaseAnalyzer, IModel, ILayer, EvolutionContext, AnalysisReport,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)
from src.structure_net.components.metrics import (
    GradientSensitivityMetric, BottleneckMetric, GradientMetric
)


class SensitivityAnalyzer(BaseAnalyzer):
    """
    Analyzes sensitivity and bottlenecks in neural networks.
    
    Combines gradient-based sensitivity analysis with bottleneck detection
    to identify architectural improvements and intervention points.
    """
    
    def __init__(self, activation_threshold: float = 0.01,
                 critical_threshold: float = 0.001,
                 sensitivity_threshold: float = 1.0,
                 name: str = None):
        """
        Initialize sensitivity analyzer.
        
        Args:
            activation_threshold: Threshold for active neurons
            critical_threshold: Threshold for critical bottlenecks
            sensitivity_threshold: Threshold for high sensitivity
            name: Optional custom name
        """
        super().__init__(name or "SensitivityAnalyzer")
        self.activation_threshold = activation_threshold
        self.critical_threshold = critical_threshold
        self.sensitivity_threshold = sensitivity_threshold
        
        # Initialize metrics
        self._gradient_sensitivity_metric = GradientSensitivityMetric(
            threshold=activation_threshold
        )
        self._bottleneck_metric = BottleneckMetric(
            activation_threshold=activation_threshold,
            critical_threshold=critical_threshold
        )
        self._gradient_metric = GradientMetric()
        
        self._required_metrics = {
            "GradientSensitivityMetric",
            "BottleneckMetric",
            "GradientMetric"
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"model", "data_loader"},
            provided_outputs={
                "analysis.layer_sensitivities",
                "analysis.bottleneck_locations",
                "analysis.intervention_recommendations",
                "analysis.gradient_flow_health",
                "analysis.architectural_suggestions"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=True,
                parallel_safe=False
            )
        )
    
    def _perform_analysis(self, model: IModel, report: AnalysisReport, 
                         context: EvolutionContext) -> Dict[str, Any]:
        """
        Perform comprehensive sensitivity analysis.
        
        Args:
            model: Model to analyze
            report: Analysis report containing metric data
            context: Evolution context with data loader
            
        Returns:
            Dictionary containing analysis results
        """
        # Get data loader from context
        data_loader = context.get('data_loader')
        if data_loader is None:
            raise ValueError("SensitivityAnalyzer requires 'data_loader' in context")
        
        num_batches = context.get('num_batches', 10)
        
        # Collect activations and gradients
        layer_data = self._collect_layer_data(model, data_loader, num_batches)
        
        # Analyze layer sensitivities
        layer_sensitivities = self._analyze_layer_sensitivities(
            layer_data, report, context
        )
        
        # Detect bottlenecks
        bottleneck_locations = self._detect_bottlenecks(
            layer_data, report, context
        )
        
        # Generate intervention recommendations
        recommendations = self._generate_recommendations(
            layer_sensitivities, bottleneck_locations
        )
        
        # Assess overall gradient flow health
        gradient_flow_health = self._assess_gradient_flow_health(
            layer_data, report
        )
        
        # Generate architectural suggestions
        architectural_suggestions = self._generate_architectural_suggestions(
            layer_sensitivities, bottleneck_locations, gradient_flow_health
        )
        
        return {
            "layer_sensitivities": layer_sensitivities,
            "bottleneck_locations": bottleneck_locations,
            "intervention_recommendations": recommendations,
            "gradient_flow_health": gradient_flow_health,
            "architectural_suggestions": architectural_suggestions,
            "summary": self._create_summary(
                layer_sensitivities, bottleneck_locations, gradient_flow_health
            )
        }
    
    def _collect_layer_data(self, model: IModel, data_loader, 
                           num_batches: int) -> Dict[str, Dict[str, torch.Tensor]]:
        """Collect activations and gradients for all layers."""
        device = next(model.parameters()).device
        model.train()  # Need gradients
        
        # Storage for aggregated data
        layer_data = {}
        layer_names = []
        
        # Get layer names
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and 'weight' in name:
                layer_names.append(name)
                layer_data[name] = {
                    'activations': [],
                    'gradients': []
                }
        
        # Collect data over batches
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            
            data, target = data.to(device), target.to(device)
            data.requires_grad_(True)
            
            # Forward hooks to capture activations
            activations = {}
            def get_activation(name):
                def hook(module, input, output):
                    activations[name] = output.detach()
                return hook
            
            handles = []
            for name, module in model.named_modules():
                if name in layer_names:
                    handles.append(module.register_forward_hook(get_activation(name)))
            
            # Forward pass
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            # Backward pass to get gradients
            model.zero_grad()
            loss.backward()
            
            # Collect gradients
            gradients = {}
            for name, param in model.named_parameters():
                if 'weight' in name and name.replace('.weight', '') in layer_names:
                    layer_name = name.replace('.weight', '')
                    if param.grad is not None:
                        gradients[layer_name] = param.grad.detach()
            
            # Store batch data
            for name in layer_names:
                if name in activations:
                    act = activations[name]
                    if act.dim() > 2:
                        act = act.flatten(1)
                    layer_data[name]['activations'].append(act)
                
                if name in gradients:
                    grad = gradients[name]
                    if grad.dim() > 2:
                        grad = grad.flatten(0, -2)
                    layer_data[name]['gradients'].append(grad.flatten())
            
            # Remove hooks
            for handle in handles:
                handle.remove()
        
        # Aggregate data
        for name in layer_names:
            if layer_data[name]['activations']:
                layer_data[name]['activations'] = torch.cat(
                    layer_data[name]['activations'], dim=0
                )
            if layer_data[name]['gradients']:
                layer_data[name]['gradients'] = torch.stack(
                    layer_data[name]['gradients']
                ).mean(dim=0)
        
        return layer_data
    
    def _analyze_layer_sensitivities(self, layer_data: Dict[str, Dict[str, torch.Tensor]],
                                   report: AnalysisReport,
                                   context: EvolutionContext) -> Dict[str, Dict[str, Any]]:
        """Analyze sensitivity between consecutive layers."""
        sensitivities = {}
        layer_names = list(layer_data.keys())
        
        for i in range(len(layer_names) - 1):
            layer_i = layer_names[i]
            layer_j = layer_names[i + 1]
            
            if (layer_i in layer_data and layer_j in layer_data and
                'activations' in layer_data[layer_i] and 
                'activations' in layer_data[layer_j]):
                
                # Create context for sensitivity metric
                metric_context = EvolutionContext({
                    'activations_i': layer_data[layer_i]['activations'],
                    'activations_j': layer_data[layer_j]['activations'],
                    'gradients_i': layer_data[layer_i].get('gradients', 
                                                          torch.zeros_like(layer_data[layer_i]['activations'])),
                    'gradients_j': layer_data[layer_j].get('gradients',
                                                          torch.zeros_like(layer_data[layer_j]['activations']))
                })
                
                # Run metric
                pair_key = f"{layer_i}->{layer_j}"
                if f"GradientSensitivityMetric_{pair_key}" not in report.metrics:
                    sensitivity_result = self._gradient_sensitivity_metric.analyze(
                        None, metric_context
                    )
                    report.add_metric_data(f"GradientSensitivityMetric_{pair_key}", 
                                         sensitivity_result)
                else:
                    sensitivity_result = report.get(
                        f"metrics.GradientSensitivityMetric_{pair_key}"
                    )
                
                sensitivities[pair_key] = sensitivity_result
        
        return sensitivities
    
    def _detect_bottlenecks(self, layer_data: Dict[str, Dict[str, torch.Tensor]],
                           report: AnalysisReport,
                           context: EvolutionContext) -> List[Dict[str, Any]]:
        """Detect bottlenecks in the network."""
        bottlenecks = []
        
        for layer_name, data in layer_data.items():
            if 'activations' in data:
                # Create context for bottleneck metric
                metric_context = EvolutionContext({
                    'activations': data['activations'],
                    'gradients': data.get('gradients', torch.zeros_like(data['activations']))
                })
                
                # Run metric
                if f"BottleneckMetric_{layer_name}" not in report.metrics:
                    bottleneck_result = self._bottleneck_metric.analyze(
                        None, metric_context
                    )
                    report.add_metric_data(f"BottleneckMetric_{layer_name}", 
                                         bottleneck_result)
                else:
                    bottleneck_result = report.get(f"metrics.BottleneckMetric_{layer_name}")
                
                # Add to bottlenecks if significant
                if bottleneck_result['bottleneck_score'] > 1.0:
                    bottlenecks.append({
                        'layer': layer_name,
                        'score': bottleneck_result['bottleneck_score'],
                        'severity': bottleneck_result['bottleneck_severity'],
                        'suggested_action': bottleneck_result['suggested_action'],
                        'priority': bottleneck_result['intervention_priority']
                    })
        
        # Sort by priority
        bottlenecks.sort(key=lambda x: x['priority'], reverse=True)
        
        return bottlenecks
    
    def _generate_recommendations(self, sensitivities: Dict[str, Dict[str, Any]],
                                bottlenecks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # High sensitivity recommendations
        for pair, sensitivity in sensitivities.items():
            if sensitivity['gradient_sensitivity'] > self.sensitivity_threshold:
                recommendations.append({
                    'type': 'high_sensitivity',
                    'location': pair,
                    'action': 'add_regularization',
                    'reason': f"High gradient sensitivity ({sensitivity['gradient_sensitivity']:.3f})",
                    'priority': 0.7
                })
            
            if sensitivity['virtual_parameter_sensitivity'] > 2.0:
                recommendations.append({
                    'type': 'parameter_sensitivity',
                    'location': pair,
                    'action': 'add_skip_connection',
                    'reason': f"High virtual parameter sensitivity ({sensitivity['virtual_parameter_sensitivity']:.3f})",
                    'priority': 0.6
                })
        
        # Bottleneck recommendations
        for bottleneck in bottlenecks:
            recommendations.append({
                'type': 'bottleneck',
                'location': bottleneck['layer'],
                'action': bottleneck['suggested_action'],
                'reason': f"{bottleneck['severity']} bottleneck detected",
                'priority': bottleneck['priority']
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        return recommendations
    
    def _assess_gradient_flow_health(self, layer_data: Dict[str, Dict[str, torch.Tensor]],
                                   report: AnalysisReport) -> Dict[str, Any]:
        """Assess overall gradient flow health."""
        # Collect gradient statistics from all layers
        gradient_norms = []
        vanishing_layers = []
        exploding_layers = []
        
        for layer_name, data in layer_data.items():
            if 'gradients' in data and data['gradients'] is not None:
                norm = data['gradients'].norm().item()
                gradient_norms.append(norm)
                
                if norm < 1e-7:
                    vanishing_layers.append(layer_name)
                elif norm > 1e3:
                    exploding_layers.append(layer_name)
        
        if not gradient_norms:
            return {
                'overall_health': 0.0,
                'gradient_flow_score': 0.0,
                'vanishing_gradient_layers': [],
                'exploding_gradient_layers': []
            }
        
        # Compute health metrics
        mean_norm = sum(gradient_norms) / len(gradient_norms)
        norm_variance = torch.var(torch.tensor(gradient_norms)).item()
        
        # Health score based on gradient characteristics
        health_score = 1.0
        
        # Penalize vanishing/exploding gradients
        vanishing_penalty = len(vanishing_layers) / len(gradient_norms)
        exploding_penalty = len(exploding_layers) / len(gradient_norms)
        health_score -= (vanishing_penalty + exploding_penalty) * 0.5
        
        # Penalize high variance
        if mean_norm > 0:
            relative_variance = norm_variance / (mean_norm ** 2)
            health_score -= min(0.3, relative_variance * 0.1)
        
        health_score = max(0.0, health_score)
        
        return {
            'overall_health': health_score,
            'gradient_flow_score': 1.0 / (1.0 + norm_variance),
            'mean_gradient_norm': mean_norm,
            'gradient_norm_variance': norm_variance,
            'vanishing_gradient_layers': vanishing_layers,
            'exploding_gradient_layers': exploding_layers
        }
    
    def _generate_architectural_suggestions(self, 
                                         sensitivities: Dict[str, Dict[str, Any]],
                                         bottlenecks: List[Dict[str, Any]],
                                         gradient_flow_health: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate architectural improvement suggestions."""
        suggestions = []
        
        # Global gradient flow issues
        if gradient_flow_health['overall_health'] < 0.5:
            suggestions.append({
                'type': 'global',
                'suggestion': 'add_batch_normalization',
                'reason': 'Poor gradient flow health detected',
                'expected_impact': 'high'
            })
        
        if gradient_flow_health['vanishing_gradient_layers']:
            suggestions.append({
                'type': 'initialization',
                'suggestion': 'use_kaiming_initialization',
                'reason': f"Vanishing gradients in {len(gradient_flow_health['vanishing_gradient_layers'])} layers",
                'affected_layers': gradient_flow_health['vanishing_gradient_layers'],
                'expected_impact': 'high'
            })
        
        # Bottleneck clustering
        if len(bottlenecks) > 2:
            critical_bottlenecks = [b for b in bottlenecks if b['severity'] == 'critical']
            if critical_bottlenecks:
                suggestions.append({
                    'type': 'architecture',
                    'suggestion': 'add_residual_connections',
                    'reason': f"{len(critical_bottlenecks)} critical bottlenecks found",
                    'affected_layers': [b['layer'] for b in critical_bottlenecks],
                    'expected_impact': 'very_high'
                })
        
        # High sensitivity patterns
        high_sensitivity_pairs = [
            pair for pair, sens in sensitivities.items()
            if sens['gradient_sensitivity'] > self.sensitivity_threshold * 2
        ]
        
        if len(high_sensitivity_pairs) > len(sensitivities) * 0.3:
            suggestions.append({
                'type': 'regularization',
                'suggestion': 'add_dropout_layers',
                'reason': 'Widespread high sensitivity detected',
                'expected_impact': 'medium'
            })
        
        return suggestions
    
    def _create_summary(self, sensitivities: Dict[str, Dict[str, Any]],
                       bottlenecks: List[Dict[str, Any]],
                       gradient_flow_health: Dict[str, Any]) -> Dict[str, Any]:
        """Create analysis summary."""
        # Calculate average sensitivity
        if sensitivities:
            avg_sensitivity = sum(
                s['gradient_sensitivity'] for s in sensitivities.values()
            ) / len(sensitivities)
        else:
            avg_sensitivity = 0.0
        
        # Count bottleneck severities
        severity_counts = {
            'critical': sum(1 for b in bottlenecks if b['severity'] == 'critical'),
            'severe': sum(1 for b in bottlenecks if b['severity'] == 'severe'),
            'moderate': sum(1 for b in bottlenecks if b['severity'] == 'moderate'),
            'mild': sum(1 for b in bottlenecks if b['severity'] == 'mild')
        }
        
        # Overall assessment
        health_score = gradient_flow_health['overall_health']
        if health_score > 0.8 and severity_counts['critical'] == 0:
            assessment = 'healthy'
        elif health_score > 0.6 and severity_counts['critical'] == 0:
            assessment = 'good'
        elif health_score > 0.4:
            assessment = 'needs_attention'
        else:
            assessment = 'critical'
        
        return {
            'overall_assessment': assessment,
            'average_sensitivity': avg_sensitivity,
            'num_bottlenecks': len(bottlenecks),
            'bottleneck_severity_counts': severity_counts,
            'gradient_flow_health': health_score,
            'num_interventions_needed': sum(
                1 for b in bottlenecks if b['priority'] > 0.7
            )
        }