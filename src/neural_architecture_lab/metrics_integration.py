"""
Integration between Neural Architecture Lab and Structure Net's metrics system.

This module bridges NAL's experiment framework with the comprehensive metrics
and analysis tools already available in structure_net.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from src.structure_net.evolution.metrics.integrated_system import CompleteMetricsSystem
from src.structure_net.evolution.metrics.base import ThresholdConfig, MetricsConfig
from src.structure_net.evolution.metrics.activity_analysis import ActivityAnalyzer
from src.structure_net.evolution.metrics.sensitivity_analysis import SensitivityAnalyzer
from src.structure_net.evolution.metrics.graph_analysis import GraphAnalyzer
from src.structure_net.evolution.metrics.mutual_information import MutualInformationAnalyzer
from src.structure_net.evolution.metrics.topological_analysis import TopologicalAnalyzer
from src.structure_net.evolution.metrics.homological_analysis import HomologicalAnalyzer
from src.structure_net.evolution.metrics.compactification_metrics import CompactificationMetrics
from src.structure_net.evolution.autocorrelation.performance_analyzer import PerformanceAnalyzer
from src.structure_net.evolution.extrema_analyzer import detect_network_extrema

from .core import ExperimentResult, Hypothesis


class StructureNetMetricsAnalyzer:
    """
    Analyzer that leverages structure_net's full metrics system for NAL experiments.
    """
    
    def __init__(self):
        # Default configurations
        self.threshold_config = ThresholdConfig(
            saturation_threshold=0.9,
            zero_threshold=0.01,
            activity_threshold=0.1,
            gradient_threshold=0.001,
            connection_threshold=0.05,
            growth_threshold=0.01
        )
        
        self.metrics_config = MetricsConfig(
            track_layer_metrics=True,
            track_connection_metrics=True,
            enable_topological_analysis=True,
            enable_homological_analysis=True,
            analyze_information_flow=True,
            enable_autocorrelation=True,
            enable_spectral_analysis=True,
            enable_extrema_detection=True
        )
        
        # Initialize performance analyzer
        self.performance_analyzer = PerformanceAnalyzer()
    
    def analyze_model(self, model: nn.Module, test_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a model using all available metrics.
        
        Args:
            model: The neural network to analyze
            test_data: Optional test data for performance analysis
            
        Returns:
            Dictionary containing all analysis results
        """
        results = {}
        
        # Create complete metrics system
        metrics_system = CompleteMetricsSystem(
            model, self.threshold_config, self.metrics_config
        )
        
        # Basic network statistics
        with torch.no_grad():
            # Get sample input for analysis
            sample_input = self._get_sample_input(model)
            
            # Run comprehensive analysis
            metrics_output = metrics_system.analyze(sample_input)
            results.update(metrics_output)
            
            # Add extrema analysis
            extrema_info = detect_network_extrema(model)
            results['extrema_analysis'] = extrema_info
            
            # Add topological analysis if enabled
            if self.metrics_config.enable_topological_analysis:
                topo_analyzer = TopologicalAnalyzer(self.threshold_config)
                results['topological_features'] = topo_analyzer.analyze_topology(model)
            
            # Add homological analysis if enabled
            if self.metrics_config.enable_homological_analysis:
                homo_analyzer = HomologicalAnalyzer(self.threshold_config)
                results['homological_features'] = homo_analyzer.analyze_homology(model)
            
            # Add compactification metrics
            compact_metrics = CompactificationMetrics(self.threshold_config)
            results['compactification'] = compact_metrics.analyze_compactification_potential(model)
        
        return results
    
    def compare_architectures(self, results_list: List[ExperimentResult]) -> Dict[str, Any]:
        """
        Compare different architectures using structure_net's metrics.
        
        Args:
            results_list: List of experiment results to compare
            
        Returns:
            Comparative analysis across architectures
        """
        comparison = {
            'architecture_rankings': {},
            'metric_correlations': {},
            'best_practices': []
        }
        
        # Extract architecture types and their metrics
        arch_metrics = {}
        for result in results_list:
            arch_type = self._classify_architecture(result.model_architecture)
            if arch_type not in arch_metrics:
                arch_metrics[arch_type] = []
            arch_metrics[arch_type].append(result.metrics)
        
        # Rank architectures by different criteria
        criteria = ['accuracy', 'efficiency', 'gradient_flow', 'information_flow']
        for criterion in criteria:
            rankings = []
            for arch_type, metrics_list in arch_metrics.items():
                values = [m.get(criterion, 0) for m in metrics_list if criterion in m]
                if values:
                    avg_value = np.mean(values)
                    rankings.append((arch_type, avg_value))
            
            rankings.sort(key=lambda x: x[1], reverse=True)
            comparison['architecture_rankings'][criterion] = rankings
        
        # Find correlations between metrics
        all_metrics = []
        for result in results_list:
            if result.metrics:
                all_metrics.append(result.metrics)
        
        if len(all_metrics) > 3:
            # Calculate correlations between key metrics
            metric_names = ['accuracy', 'sparsity', 'gradient_flow', 'information_flow']
            for m1 in metric_names:
                for m2 in metric_names:
                    if m1 < m2:  # Avoid duplicates
                        values1 = [m.get(m1, 0) for m in all_metrics if m1 in m and m2 in m]
                        values2 = [m.get(m2, 0) for m in all_metrics if m1 in m and m2 in m]
                        if len(values1) > 3:
                            corr = np.corrcoef(values1, values2)[0, 1]
                            comparison['metric_correlations'][f"{m1}_vs_{m2}"] = corr
        
        # Extract best practices
        best_accuracy_idx = max(range(len(results_list)), 
                               key=lambda i: results_list[i].metrics.get('accuracy', 0))
        best_result = results_list[best_accuracy_idx]
        
        if best_result.metrics.get('gradient_flow', 0) > 0.5:
            comparison['best_practices'].append("High gradient flow correlates with better accuracy")
        if best_result.metrics.get('sparsity', 0) < 0.1:
            comparison['best_practices'].append("Lower sparsity (<10%) tends to perform better")
        
        return comparison
    
    def analyze_growth_dynamics(self, growth_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze network growth dynamics using metrics evolution.
        
        Args:
            growth_history: List of metrics snapshots during growth
            
        Returns:
            Analysis of growth patterns and effectiveness
        """
        if not growth_history:
            return {}
        
        analysis = {
            'growth_efficiency': [],
            'capacity_utilization': [],
            'information_gain': [],
            'growth_recommendations': []
        }
        
        # Analyze efficiency of each growth event
        for i in range(1, len(growth_history)):
            prev = growth_history[i-1]
            curr = growth_history[i]
            
            # Calculate efficiency metrics
            param_increase = (curr.get('parameters', 0) - prev.get('parameters', 0)) / prev.get('parameters', 1)
            accuracy_gain = curr.get('accuracy', 0) - prev.get('accuracy', 0)
            
            if param_increase > 0:
                efficiency = accuracy_gain / param_increase
                analysis['growth_efficiency'].append(efficiency)
            
            # Track capacity utilization
            if 'layer_saturation' in curr:
                avg_saturation = np.mean(list(curr['layer_saturation'].values()))
                analysis['capacity_utilization'].append(avg_saturation)
            
            # Information flow changes
            if 'mutual_information' in curr and 'mutual_information' in prev:
                info_gain = curr['mutual_information'] - prev['mutual_information']
                analysis['information_gain'].append(info_gain)
        
        # Generate recommendations
        if analysis['growth_efficiency']:
            avg_efficiency = np.mean(analysis['growth_efficiency'])
            if avg_efficiency < 0.1:
                analysis['growth_recommendations'].append(
                    "Growth efficiency is low - consider more selective growth strategies"
                )
            
            if len(analysis['growth_efficiency']) > 3:
                # Check for diminishing returns
                early_efficiency = np.mean(analysis['growth_efficiency'][:len(analysis['growth_efficiency'])//2])
                late_efficiency = np.mean(analysis['growth_efficiency'][len(analysis['growth_efficiency'])//2:])
                
                if late_efficiency < early_efficiency * 0.5:
                    analysis['growth_recommendations'].append(
                        "Diminishing returns detected - consider stopping growth earlier"
                    )
        
        if analysis['capacity_utilization']:
            final_utilization = analysis['capacity_utilization'][-1]
            if final_utilization < 0.5:
                analysis['growth_recommendations'].append(
                    "Low capacity utilization - network may be over-parameterized"
                )
        
        return analysis
    
    def analyze_training_dynamics(self, training_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze training dynamics using autocorrelation and performance metrics.
        
        Args:
            training_history: Training history with metrics per epoch
            
        Returns:
            Analysis of training patterns and stability
        """
        if not training_history:
            return {}
        
        # Extract time series
        losses = [h.get('loss', 0) for h in training_history]
        accuracies = [h.get('accuracy', 0) for h in training_history]
        learning_rates = [h.get('lr', 0) for h in training_history]
        
        # Use performance analyzer
        loss_analysis = self.performance_analyzer.analyze_convergence(
            losses, method='exponential_smoothing'
        )
        
        acc_analysis = self.performance_analyzer.analyze_convergence(
            accuracies, method='polynomial_fit'
        )
        
        # Analyze oscillations
        loss_oscillation = self.performance_analyzer.detect_oscillation(losses)
        lr_stability = self.performance_analyzer.analyze_lr_stability(learning_rates)
        
        # Compile results
        analysis = {
            'convergence': {
                'loss_converged': loss_analysis['converged'],
                'accuracy_converged': acc_analysis['converged'],
                'convergence_epoch': loss_analysis.get('convergence_point', -1)
            },
            'stability': {
                'loss_oscillation': loss_oscillation['oscillation_strength'],
                'lr_stability': lr_stability['stability_score']
            },
            'efficiency': {
                'epochs_to_90_percent': self._find_threshold_epoch(accuracies, 0.9),
                'final_accuracy': accuracies[-1] if accuracies else 0
            }
        }
        
        # Add recommendations
        if loss_oscillation['oscillation_strength'] > 0.3:
            analysis['recommendations'] = analysis.get('recommendations', [])
            analysis['recommendations'].append(
                "High loss oscillation detected - consider reducing learning rate"
            )
        
        return analysis
    
    def generate_insights(self, 
                         hypothesis: Hypothesis,
                         results: List[ExperimentResult],
                         metrics_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generate insights by combining NAL results with structure_net metrics.
        
        Args:
            hypothesis: The hypothesis being tested
            results: Experiment results
            metrics_analysis: Comprehensive metrics analysis
            
        Returns:
            Dictionary of insights
        """
        insights = {
            'key_findings': [],
            'unexpected_results': [],
            'recommendations': [],
            'future_hypotheses': []
        }
        
        # Analyze based on hypothesis category
        if hypothesis.category.value == 'architecture':
            # Architecture-specific insights
            if 'architecture_rankings' in metrics_analysis:
                rankings = metrics_analysis['architecture_rankings']
                if 'gradient_flow' in rankings and rankings['gradient_flow']:
                    best_arch = rankings['gradient_flow'][0][0]
                    insights['key_findings'].append(
                        f"{best_arch} architecture shows best gradient flow characteristics"
                    )
            
            # Check for extrema
            extrema_counts = [r.metrics.get('extrema_count', 0) for r in results if 'extrema_count' in r.metrics]
            if extrema_counts and max(extrema_counts) > 5:
                insights['unexpected_results'].append(
                    "High number of extrema neurons detected - may indicate optimization issues"
                )
                insights['recommendations'].append(
                    "Consider gradient clipping or different initialization"
                )
        
        elif hypothesis.category.value == 'growth':
            # Growth-specific insights
            if 'growth_efficiency' in metrics_analysis:
                efficiencies = metrics_analysis['growth_efficiency']
                if efficiencies:
                    avg_efficiency = np.mean(efficiencies)
                    insights['key_findings'].append(
                        f"Average growth efficiency: {avg_efficiency:.3f} accuracy gain per parameter ratio"
                    )
                    
                    if avg_efficiency < 0.05:
                        insights['recommendations'].append(
                            "Consider more targeted growth strategies or earlier stopping"
                        )
        
        elif hypothesis.category.value == 'training':
            # Training-specific insights
            if 'stability' in metrics_analysis:
                stability = metrics_analysis['stability']
                if stability.get('loss_oscillation', 0) > 0.5:
                    insights['unexpected_results'].append(
                        "High training instability detected"
                    )
                    insights['future_hypotheses'].append(
                        "Investigate learning rate warmup and decay strategies"
                    )
        
        # General insights from metrics
        if 'metric_correlations' in metrics_analysis:
            correlations = metrics_analysis['metric_correlations']
            for metric_pair, corr in correlations.items():
                if abs(corr) > 0.7:
                    direction = "positive" if corr > 0 else "negative"
                    insights['key_findings'].append(
                        f"Strong {direction} correlation ({corr:.2f}) between {metric_pair}"
                    )
        
        # Add compactification insights
        compact_potential = [r.metrics.get('compactification_potential', 0) 
                           for r in results if 'compactification_potential' in r.metrics]
        if compact_potential and max(compact_potential) > 0.3:
            insights['recommendations'].append(
                "High compactification potential detected - consider patch-based approaches"
            )
        
        return insights
    
    def _classify_architecture(self, architecture: List[int]) -> str:
        """Classify architecture type."""
        if not architecture or len(architecture) < 3:
            return "unknown"
        
        depth = len(architecture) - 1
        
        # Check patterns
        if depth <= 3:
            return "shallow"
        elif depth >= 6:
            return "deep"
        
        # Check shape
        sizes = architecture[1:-1]  # Hidden layers only
        if all(sizes[i] >= sizes[i+1] for i in range(len(sizes)-1)):
            return "pyramid"
        elif all(sizes[i] <= sizes[i+1] for i in range(len(sizes)-1)):
            return "inverse_pyramid"
        else:
            # Check for bottleneck
            min_idx = sizes.index(min(sizes))
            if min_idx > 0 and min_idx < len(sizes) - 1:
                return "bottleneck"
            return "irregular"
    
    def _get_sample_input(self, model: nn.Module) -> torch.Tensor:
        """Generate sample input for model analysis."""
        # Try to infer input size from first layer
        first_layer = next(model.modules())
        if isinstance(first_layer, nn.Linear):
            input_size = first_layer.in_features
            return torch.randn(1, input_size)
        elif isinstance(first_layer, nn.Conv2d):
            # Assume standard image input
            return torch.randn(1, 3, 32, 32)
        else:
            # Default
            return torch.randn(1, 784)
    
    def _find_threshold_epoch(self, values: List[float], threshold_fraction: float) -> int:
        """Find epoch where value reaches threshold fraction of final value."""
        if not values:
            return -1
        
        final_value = values[-1]
        threshold = final_value * threshold_fraction
        
        for i, val in enumerate(values):
            if val >= threshold:
                return i
        
        return len(values) - 1