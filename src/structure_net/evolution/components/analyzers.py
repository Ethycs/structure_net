#!/usr/bin/env python3
"""
Concrete Analyzer Implementations

This module provides concrete implementations of network analyzers
that wrap existing analysis functionality into the composable interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional
import time

from ..interfaces import (
    NetworkAnalyzer, ExtremaAnalyzer, InformationFlowAnalyzer,
    NetworkContext, AnalysisResult, FullyConfigurableComponent
)
from ..extrema_analyzer import detect_network_extrema
from ...core.network_analysis import get_network_stats
from ...core.layers import StandardSparseLayer


class StandardExtremaAnalyzer(ExtremaAnalyzer, FullyConfigurableComponent):
    """
    Standard implementation of extrema analysis using existing functionality.
    """
    
    def __init__(self, 
                 dead_threshold: float = 0.01,
                 saturated_multiplier: float = 2.5,
                 max_batches: int = 5):
        self.dead_threshold = dead_threshold
        self.saturated_multiplier = saturated_multiplier
        self.max_batches = max_batches
        self._metrics = {}
    
    def can_apply(self, context: NetworkContext) -> bool:
        """Check if extrema analysis can be applied."""
        return (self.validate_context(context) and 
                len(list(context.network.parameters())) > 0)
    
    def apply(self, context: NetworkContext) -> bool:
        """Apply extrema analysis and store results in context."""
        if not self.can_apply(context):
            return False
        
        result = self.analyze(context)
        context.metadata['extrema_analysis'] = result
        return True
    
    def analyze(self, context: NetworkContext) -> AnalysisResult:
        """Perform extrema analysis on the network."""
        start_time = time.time()
        
        # Use existing extrema detection
        extrema_patterns = detect_network_extrema(
            context.network,
            context.data_loader,
            str(context.device),
            max_batches=self.max_batches
        )
        
        # Convert list of patterns to summary statistics
        from ..extrema_analyzer import get_extrema_statistics
        extrema_stats = get_extrema_statistics(extrema_patterns)
        
        # Calculate extrema ratio
        total_neurons = sum(len(layer.get('low', [])) + len(layer.get('high', [])) 
                           for layer in extrema_patterns)
        network_size = sum(p.numel() for p in context.network.parameters() if p.dim() > 1)
        extrema_ratio = total_neurons / max(network_size, 1)
        
        # Extract key metrics
        metrics = {
            'extrema_patterns': extrema_patterns,
            'extrema_stats': extrema_stats,
            'total_extrema': extrema_stats['total_dead_neurons'] + extrema_stats['total_saturated_neurons'],
            'extrema_ratio': extrema_ratio,
            'dead_neurons': {i: pattern['low'] for i, pattern in enumerate(extrema_patterns)},
            'saturated_neurons': {i: pattern['high'] for i, pattern in enumerate(extrema_patterns)},
            'layer_health': {i: 1.0 - (len(pattern['low']) + len(pattern['high'])) / max(100, 1) 
                           for i, pattern in enumerate(extrema_patterns)},
            'analysis_time': time.time() - start_time
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)
        
        # Update monitoring metrics
        self._metrics.update({
            'last_analysis_time': metrics['analysis_time'],
            'last_extrema_ratio': metrics['extrema_ratio'],
            'total_analyses': self._metrics.get('total_analyses', 0) + 1
        })
        
        return AnalysisResult(
            analyzer_name=self.get_name(),
            metrics=metrics,
            recommendations=recommendations,
            confidence=self._calculate_confidence(metrics),
            timestamp=start_time
        )
    
    def detect_extrema(self, context: NetworkContext) -> Dict[str, Any]:
        """Specialized extrema detection method."""
        result = self.analyze(context)
        return result.metrics['extrema_patterns']
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on extrema analysis."""
        recommendations = []
        
        extrema_ratio = metrics.get('extrema_ratio', 0.0)
        dead_neurons = metrics.get('dead_neurons', {})
        saturated_neurons = metrics.get('saturated_neurons', {})
        
        # High extrema ratio
        if extrema_ratio > 0.3:
            recommendations.append("add_extrema_aware_patches")
            recommendations.append("consider_layer_addition")
        
        # Many dead neurons
        total_dead = sum(len(neurons) for neurons in dead_neurons.values())
        if total_dead > 10:
            recommendations.append("revive_dead_neurons")
            recommendations.append("add_skip_connections")
        
        # Many saturated neurons
        total_saturated = sum(len(neurons) for neurons in saturated_neurons.values())
        if total_saturated > 10:
            recommendations.append("add_relief_connections")
            recommendations.append("normalize_activations")
        
        # Layer-specific recommendations
        for layer_idx, health in metrics.get('layer_health', {}).items():
            if health < 0.5:
                recommendations.append(f"repair_layer_{layer_idx}")
        
        return recommendations
    
    def _calculate_confidence(self, metrics: Dict[str, Any]) -> float:
        """Calculate confidence in the analysis results."""
        # Base confidence on data quality and consistency
        base_confidence = 0.8
        
        # Reduce confidence if analysis was too fast (might be insufficient data)
        if metrics.get('analysis_time', 0) < 0.1:
            base_confidence *= 0.7
        
        # Increase confidence if we have clear patterns
        extrema_ratio = metrics.get('extrema_ratio', 0.0)
        if extrema_ratio > 0.2 or extrema_ratio < 0.05:
            base_confidence *= 1.1
        
        return min(1.0, base_confidence)
    
    def get_required_batches(self) -> int:
        """Get number of batches required for analysis."""
        return self.max_batches
    
    # Configurable interface
    def configure(self, config: Dict[str, Any]):
        """Configure the analyzer."""
        self.dead_threshold = config.get('dead_threshold', self.dead_threshold)
        self.saturated_multiplier = config.get('saturated_multiplier', self.saturated_multiplier)
        self.max_batches = config.get('max_batches', self.max_batches)
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            'dead_threshold': self.dead_threshold,
            'saturated_multiplier': self.saturated_multiplier,
            'max_batches': self.max_batches
        }
    
    # Serializable interface
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': 'StandardExtremaAnalyzer',
            'config': self.get_configuration(),
            'metrics': self._metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StandardExtremaAnalyzer':
        """Deserialize from dictionary."""
        analyzer = cls(**data.get('config', {}))
        analyzer._metrics = data.get('metrics', {})
        return analyzer
    
    # Monitorable interface
    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics."""
        return self._metrics.copy()
    
    def reset_metrics(self):
        """Reset monitoring metrics."""
        self._metrics.clear()


class NetworkStatsAnalyzer(NetworkAnalyzer, FullyConfigurableComponent):
    """
    Analyzer for basic network statistics and architecture information.
    """
    
    def __init__(self):
        self._metrics = {}
    
    def can_apply(self, context: NetworkContext) -> bool:
        """Check if network stats analysis can be applied."""
        return self.validate_context(context)
    
    def apply(self, context: NetworkContext) -> bool:
        """Apply network stats analysis."""
        if not self.can_apply(context):
            return False
        
        result = self.analyze(context)
        context.metadata['network_stats'] = result
        return True
    
    def analyze(self, context: NetworkContext) -> AnalysisResult:
        """Analyze network statistics."""
        start_time = time.time()
        
        # Get network statistics using existing function
        network_stats = get_network_stats(context.network)
        
        # Add additional metrics
        metrics = {
            **network_stats,
            'device': str(context.device),
            'epoch': context.epoch,
            'iteration': context.iteration,
            'analysis_time': time.time() - start_time
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)
        
        # Update monitoring
        self._metrics.update({
            'last_analysis_time': metrics['analysis_time'],
            'last_parameter_count': metrics.get('total_parameters', 0),
            'total_analyses': self._metrics.get('total_analyses', 0) + 1
        })
        
        return AnalysisResult(
            analyzer_name=self.get_name(),
            metrics=metrics,
            recommendations=recommendations,
            confidence=1.0,  # Network stats are always reliable
            timestamp=start_time
        )
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on network statistics."""
        recommendations = []
        
        # Check sparsity
        sparsity = metrics.get('overall_sparsity', 0.0)
        if sparsity > 0.95:
            recommendations.append("network_too_sparse")
        elif sparsity < 0.5:
            recommendations.append("consider_pruning")
        
        # Check architecture depth
        architecture = metrics.get('architecture', [])
        if len(architecture) < 3:
            recommendations.append("consider_deeper_network")
        elif len(architecture) > 10:
            recommendations.append("consider_residual_connections")
        
        # Check parameter count
        total_params = metrics.get('total_parameters', 0)
        if total_params < 1000:
            recommendations.append("network_might_be_too_small")
        elif total_params > 1000000:
            recommendations.append("consider_parameter_sharing")
        
        return recommendations
    
    def get_analysis_type(self) -> str:
        return "network_stats"
    
    # Configurable interface
    def configure(self, config: Dict[str, Any]):
        """Configure the analyzer."""
        pass  # No configuration needed for basic stats
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {}
    
    # Serializable interface
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': 'NetworkStatsAnalyzer',
            'metrics': self._metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkStatsAnalyzer':
        """Deserialize from dictionary."""
        analyzer = cls()
        analyzer._metrics = data.get('metrics', {})
        return analyzer
    
    # Monitorable interface
    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics."""
        return self._metrics.copy()
    
    def reset_metrics(self):
        """Reset monitoring metrics."""
        self._metrics.clear()


class SimpleInformationFlowAnalyzer(InformationFlowAnalyzer, FullyConfigurableComponent):
    """
    Simple information flow analyzer using correlation-based MI approximation.
    """
    
    def __init__(self, min_bottleneck_severity: float = 0.05):
        self.min_bottleneck_severity = min_bottleneck_severity
        self._metrics = {}
    
    def can_apply(self, context: NetworkContext) -> bool:
        """Check if information flow analysis can be applied."""
        return (self.validate_context(context) and 
                len(list(context.network.modules())) > 1)
    
    def apply(self, context: NetworkContext) -> bool:
        """Apply information flow analysis."""
        if not self.can_apply(context):
            return False
        
        result = self.analyze(context)
        context.metadata['information_flow'] = result
        return True
    
    def analyze(self, context: NetworkContext) -> AnalysisResult:
        """Analyze information flow through the network."""
        start_time = time.time()
        
        # Get layer activations
        activations = self._get_layer_activations(context)
        
        # Calculate MI flow
        mi_flow = self._calculate_mi_flow(activations)
        
        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(mi_flow)
        
        metrics = {
            'mi_flow': mi_flow,
            'bottlenecks': bottlenecks,
            'total_information_loss': sum(max(0, mi_flow[i] - mi_flow[i+1]) 
                                         for i in range(len(mi_flow)-1)),
            'information_efficiency': self._calculate_efficiency(mi_flow),
            'analysis_time': time.time() - start_time
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)
        
        # Update monitoring
        self._metrics.update({
            'last_analysis_time': metrics['analysis_time'],
            'last_bottleneck_count': len(bottlenecks),
            'total_analyses': self._metrics.get('total_analyses', 0) + 1
        })
        
        return AnalysisResult(
            analyzer_name=self.get_name(),
            metrics=metrics,
            recommendations=recommendations,
            confidence=0.7,  # MI approximation has moderate confidence
            timestamp=start_time
        )
    
    def analyze_information_flow(self, context: NetworkContext) -> Dict[str, Any]:
        """Specialized information flow analysis method."""
        result = self.analyze(context)
        return {
            'mi_flow': result.metrics['mi_flow'],
            'bottlenecks': result.metrics['bottlenecks']
        }
    
    def _get_layer_activations(self, context: NetworkContext) -> List[torch.Tensor]:
        """Get activations from each layer."""
        activations = []
        
        with torch.no_grad():
            # Get one batch for analysis
            data, _ = next(iter(context.data_loader))
            data = data.to(context.device).view(data.size(0), -1)
            
            h = data
            activations.append(h.clone())
            
            for layer in context.network:
                if isinstance(layer, StandardSparseLayer):
                    h = layer(h)
                    activations.append(h.clone())
                    h = F.relu(h)
                elif isinstance(layer, nn.ReLU):
                    h = layer(h)
                    if activations:
                        activations[-1] = h.clone()
        
        return activations
    
    def _calculate_mi_flow(self, activations: List[torch.Tensor]) -> List[float]:
        """Calculate MI flow between layers."""
        mi_flow = []
        
        for i in range(len(activations)):
            if i == 0:
                # Input layer - use entropy approximation
                mi = self._estimate_entropy(activations[i])
            else:
                # MI between consecutive layers
                mi = self._estimate_mi_proxy(activations[i-1], activations[i])
            mi_flow.append(mi)
        
        return mi_flow
    
    def _estimate_mi_proxy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Fast proxy for Mutual Information based on correlation."""
        if x.numel() == 0 or y.numel() == 0:
            return 0.0
        
        x_norm = F.normalize(x, dim=1)
        y_norm = F.normalize(y, dim=1)
        min_dim = min(x_norm.shape[1], y_norm.shape[1])
        
        if min_dim == 0:
            return 0.0
        
        correlation = (x_norm[:, :min_dim] * y_norm[:, :min_dim]).sum(dim=1).mean()
        mi_approx = -0.5 * torch.log(1 - correlation**2 + 1e-8)
        return mi_approx.item()
    
    def _estimate_entropy(self, x: torch.Tensor) -> float:
        """Estimate entropy of activations."""
        if x.numel() == 0:
            return 0.0
        
        # Simple entropy approximation based on variance
        variance = x.var(dim=0).mean()
        entropy_approx = 0.5 * torch.log(2 * np.pi * np.e * variance + 1e-8)
        return entropy_approx.item()
    
    def _detect_bottlenecks(self, mi_flow: List[float]) -> List[Dict[str, Any]]:
        """Detect information bottlenecks in the flow."""
        bottlenecks = []
        
        for i in range(len(mi_flow) - 1):
            info_loss = mi_flow[i] - mi_flow[i+1]
            if info_loss > self.min_bottleneck_severity:
                severity = info_loss / (mi_flow[0] + 1e-6)
                bottlenecks.append({
                    'position': i + 1,
                    'info_loss': info_loss,
                    'severity': severity
                })
        
        return sorted(bottlenecks, key=lambda x: x['severity'], reverse=True)
    
    def _calculate_efficiency(self, mi_flow: List[float]) -> float:
        """Calculate information efficiency."""
        if len(mi_flow) < 2:
            return 1.0
        
        total_loss = sum(max(0, mi_flow[i] - mi_flow[i+1]) 
                        for i in range(len(mi_flow)-1))
        initial_info = mi_flow[0] if mi_flow else 0
        
        if initial_info <= 0:
            return 1.0
        
        return max(0.0, 1.0 - total_loss / initial_info)
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on information flow analysis."""
        recommendations = []
        
        bottlenecks = metrics.get('bottlenecks', [])
        efficiency = metrics.get('information_efficiency', 1.0)
        
        if bottlenecks:
            recommendations.append("add_layer_for_information_flow")
            recommendations.append("insert_layer_at_bottleneck")
        
        if efficiency < 0.7:
            recommendations.append("improve_information_flow")
            recommendations.append("add_skip_connections")
        
        return recommendations
    
    # Configurable interface
    def configure(self, config: Dict[str, Any]):
        """Configure the analyzer."""
        self.min_bottleneck_severity = config.get('min_bottleneck_severity', 
                                                 self.min_bottleneck_severity)
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            'min_bottleneck_severity': self.min_bottleneck_severity
        }
    
    # Serializable interface
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': 'SimpleInformationFlowAnalyzer',
            'config': self.get_configuration(),
            'metrics': self._metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimpleInformationFlowAnalyzer':
        """Deserialize from dictionary."""
        analyzer = cls(**data.get('config', {}))
        analyzer._metrics = data.get('metrics', {})
        return analyzer
    
    # Monitorable interface
    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics."""
        return self._metrics.copy()
    
    def reset_metrics(self):
        """Reset monitoring metrics."""
        self._metrics.clear()


# Export all analyzers
__all__ = [
    'StandardExtremaAnalyzer',
    'NetworkStatsAnalyzer', 
    'SimpleInformationFlowAnalyzer'
]
