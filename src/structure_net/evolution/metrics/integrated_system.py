"""
Integrated Metrics System

This module provides the main CompleteMetricsSystem that orchestrates all
metric analyzers and integrates with the autocorrelation framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, Any, List
import time
import logging
import numpy as np

from .base import ThresholdConfig, MetricsConfig
from .mutual_information import MutualInformationAnalyzer
from .activity_analysis import ActivityAnalyzer
from .sensitivity_analysis import SensitivityAnalyzer
from .graph_analysis import GraphAnalyzer

logger = logging.getLogger(__name__)


class CompleteMetricsSystem:
    """
    DEPRECATED: The metrics system has been migrated to a component-based architecture.
    
    The old monolithic analyzers have been split into:
    1. Low-level metrics (focused measurements) in src.structure_net.components.metrics
    2. High-level analyzers (combining metrics) in src.structure_net.components.analyzers
    
    For a complete replacement, use the new MetricsOrchestrator pattern:
    
    Example migration:
        # Old way:
        metrics_system = CompleteMetricsSystem(network, threshold_config, metrics_config)
        results = metrics_system.compute_all_metrics(data_loader)
        
        # New way:
        from src.structure_net.components.analyzers import (
            InformationFlowAnalyzer, HomologicalAnalyzer
        )
        from src.structure_net.core import EvolutionContext, AnalysisReport
        
        # Create analyzers
        info_analyzer = InformationFlowAnalyzer()
        homo_analyzer = HomologicalAnalyzer()
        
        # Run analysis
        context = EvolutionContext({'model': network, 'data_loader': data_loader})
        report = AnalysisReport()
        
        info_results = info_analyzer.analyze(network, report, context)
        homo_results = homo_analyzer.analyze(network, report, context)
    
    See MIGRATION_STATUS.md for full migration guide.
    """
    
    def __init__(self, network: nn.Module, threshold_config: ThresholdConfig, metrics_config: MetricsConfig):
        # Provide helpful migration message before failing
        migration_msg = (
            "\n" + "="*80 + "\n"
            "MIGRATION NOTICE: CompleteMetricsSystem has been replaced by component architecture.\n"
            "\n"
            "The metrics system has been redesigned for better modularity and performance.\n"
            "Old monolithic analyzers are now split into focused metrics and analyzers.\n"
            "\n"
            "Quick migration path:\n"
            "1. For MI/entropy analysis: use InformationFlowAnalyzer\n"
            "2. For homological analysis: use HomologicalAnalyzer\n" 
            "3. For specific metrics: import from src.structure_net.components.metrics\n"
            "\n"
            "See src/structure_net/evolution/metrics/MIGRATION_STATUS.md for details.\n"
            + "="*80
        )
        logger.warning(migration_msg)
        
        self.network = network
        self.threshold_config = threshold_config
        self.metrics_config = metrics_config
        
        # Initialize all analyzers - these will raise deprecation warnings
        try:
            self.mi_analyzer = MutualInformationAnalyzer(threshold_config)
        except DeprecationWarning as e:
            logger.error(f"Failed to initialize MI analyzer: {e}")
            raise
        
        self.activity_analyzer = ActivityAnalyzer(threshold_config)
        self.sensli_analyzer = SensitivityAnalyzer(network, threshold_config)
        self.graph_analyzer = GraphAnalyzer(network, threshold_config)
        
        # Initialize autocorrelation framework
        self.performance_analyzer = None
        if self._autocorrelation_available():
            try:
                from ..autocorrelation.performance_analyzer import PerformanceAnalyzer
                self.performance_analyzer = PerformanceAnalyzer()
                logger.info("🔬 Autocorrelation framework enabled")
            except ImportError:
                logger.info("📊 Using basic metrics system (autocorrelation framework not available)")
        
    def _autocorrelation_available(self) -> bool:
        """Check if autocorrelation framework is available."""
        try:
            from ..autocorrelation.performance_analyzer import PerformanceAnalyzer
            return True
        except ImportError:
            return False
    
    def compute_all_metrics(self, data_loader, num_batches: int = 10) -> Dict[str, Any]:
        """
        Compute ALL metrics for the network.
        
        Args:
            data_loader: Data loader for metric computation
            num_batches: Number of batches to analyze
            
        Returns:
            Dict containing all computed metrics
        """
        logger.info("🔬 Computing complete metrics suite...")
        
        results = {
            'mi_metrics': {},
            'activity_metrics': {},
            'sensli_metrics': {},
            'graph_metrics': {},
            'summary': {},
            'computation_stats': {}
        }
        
        # Collect activation and gradient data in a single pass (OPTIMIZED)
        activation_data, gradient_data = self._collect_activation_and_gradient_data(data_loader, num_batches)
        
        # Get sparse layers
        sparse_layers = self._get_sparse_layers()
        
        # 1. MI Metrics for each layer pair
        if self.metrics_config.compute_mi:
            logger.info("  Computing MI metrics...")
            for i in range(len(sparse_layers) - 1):
                if i in activation_data and i+1 in activation_data:
                    acts_i = activation_data[i]
                    acts_j = activation_data[i+1]
                    
                    # Apply ReLU if not output layer
                    if i < len(sparse_layers) - 1:
                        acts_i = F.relu(acts_i)
                    if i+1 < len(sparse_layers) - 1:
                        acts_j = F.relu(acts_j)
                    
                    mi_metrics = self.mi_analyzer.compute_metrics(acts_i, acts_j)
                    results['mi_metrics'][f'layer_{i}_{i+1}'] = mi_metrics
        
        # 2. Activity Metrics for each layer
        if self.metrics_config.compute_activity:
            logger.info("  Computing activity metrics...")
            for layer_idx, acts in activation_data.items():
                activity_metrics = self.activity_analyzer.compute_metrics(acts, layer_idx)
                results['activity_metrics'][f'layer_{layer_idx}'] = activity_metrics
        
        # 3. SensLI Metrics for each layer pair (OPTIMIZED - reuse pre-computed data)
        sensli_optimization = getattr(self.metrics_config, 'sensli_optimization', True)
        if self.metrics_config.compute_sensli and sensli_optimization:
            logger.info("  Computing SensLI metrics...")
            for i in range(len(sparse_layers) - 1):
                if i in activation_data and i+1 in activation_data and i in gradient_data and i+1 in gradient_data:
                    sensli_metrics = self.sensli_analyzer.compute_metrics_from_precomputed_data(
                        activation_data[i], activation_data[i+1],
                        gradient_data[i], gradient_data[i+1],
                        i, i+1
                    )
                    results['sensli_metrics'][f'layer_{i}_{i+1}'] = sensli_metrics
        
        # 4. Graph Metrics
        if self.metrics_config.compute_graph:
            logger.info("  Computing graph metrics...")
            graph_metrics = self.graph_analyzer.compute_metrics(activation_data)
            results['graph_metrics'] = graph_metrics
        
        # 5. Summary metrics
        results['summary'] = self._compute_summary_metrics(results)
        
        # 6. Computation statistics
        results['computation_stats'] = self._get_computation_stats()
        
        logger.info("✅ Complete metrics computation finished")
        
        return results
    
    def _collect_activation_and_gradient_data(self, data_loader, num_batches):
        """
        OPTIMIZED: Collect both activation and gradient data in a single pass.
        This eliminates redundant forward passes and significantly improves efficiency.
        """
        device = next(self.network.parameters()).device
        activation_data = defaultdict(list)
        gradient_data = defaultdict(list)
        
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            
            data = data.to(device)
            target = target.to(device)
            data.requires_grad_(True)
            
            # Forward pass collecting activations
            x = data.view(data.size(0), -1)
            layer_activations = []
            layer_idx = 0
            
            sparse_layers = self._get_sparse_layers()
            
            for layer in self.network:
                if self._is_sparse_layer(layer):
                    x = layer(x)
                    layer_activations.append(x.clone())
                    activation_data[layer_idx].append(x.detach().clone())
                    layer_idx += 1
                    if layer_idx < len(sparse_layers):
                        x = F.relu(x)
                elif isinstance(layer, nn.ReLU):
                    x = layer(x)
            
            # Backward pass to compute gradients for each layer
            loss = F.cross_entropy(x, target)
            
            # Compute gradients for each layer activation
            for i, acts in enumerate(layer_activations):
                try:
                    if not acts.requires_grad:
                        acts.requires_grad_(True)
                    
                    grads = torch.autograd.grad(
                        loss, acts, 
                        retain_graph=(i < len(layer_activations) - 1),
                        create_graph=False,
                        allow_unused=True
                    )
                    
                    if grads[0] is not None:
                        gradient_data[i].append(grads[0].detach().clone())
                    else:
                        # Create zero gradients if unused
                        zero_grads = torch.zeros_like(acts)
                        gradient_data[i].append(zero_grads.detach().clone())
                        
                except RuntimeError as e:
                    # Fallback: create zero gradients
                    zero_grads = torch.zeros_like(acts)
                    gradient_data[i].append(zero_grads.detach().clone())
        
        # Aggregate data - concatenate all batches
        for layer_idx in activation_data:
            activation_data[layer_idx] = torch.cat(activation_data[layer_idx], dim=0)
            assert activation_data[layer_idx].dim() == 2, f"Layer {layer_idx} activations should be 2D"
        
        for layer_idx in gradient_data:
            gradient_data[layer_idx] = torch.cat(gradient_data[layer_idx], dim=0)
            assert gradient_data[layer_idx].dim() == 2, f"Layer {layer_idx} gradients should be 2D"
        
        return dict(activation_data), dict(gradient_data)
    
    def _get_sparse_layers(self) -> List[nn.Module]:
        """Extract sparse layers from network."""
        from ...core.layers import StandardSparseLayer
        return [layer for layer in self.network if isinstance(layer, StandardSparseLayer)]
    
    def _is_sparse_layer(self, layer) -> bool:
        """Check if layer is a sparse layer."""
        from ...core.layers import StandardSparseLayer
        return isinstance(layer, StandardSparseLayer)
    
    def _compute_summary_metrics(self, results):
        """Compute high-level summary metrics."""
        summary = {}
        
        # MI Summary
        mi_results = results['mi_metrics']
        if mi_results:
            mi_efficiencies = [r['mi_efficiency'] for r in mi_results.values()]
            summary['avg_mi_efficiency'] = float(np.mean(mi_efficiencies)) if mi_efficiencies else 0.0
            summary['min_mi_efficiency'] = float(np.min(mi_efficiencies)) if mi_efficiencies else 0.0
            summary['bottleneck_layers'] = sum(1 for eff in mi_efficiencies if eff < 0.3)
        
        # Activity Summary
        activity_results = results['activity_metrics']
        if activity_results:
            active_ratios = [r['active_ratio'] for r in activity_results.values()]
            health_scores = [r['layer_health_score'] for r in activity_results.values()]
            summary['avg_active_ratio'] = float(np.mean(active_ratios)) if active_ratios else 0.0
            summary['avg_health_score'] = float(np.mean(health_scores)) if health_scores else 0.0
            summary['dead_layers'] = sum(1 for ratio in active_ratios if ratio < 0.01)
        
        # SensLI Summary
        sensli_results = results['sensli_metrics']
        if sensli_results:
            critical_bottlenecks = sum(1 for r in sensli_results.values() if r['critical_bottleneck'])
            summary['critical_bottlenecks'] = critical_bottlenecks
            
            priorities = [r['intervention_priority'] for r in sensli_results.values()]
            summary['avg_intervention_priority'] = float(np.mean(priorities)) if priorities else 0.0
        
        # Graph Summary
        graph_results = results['graph_metrics']
        if graph_results.get('graph_built', False):
            summary['network_connected'] = graph_results['num_weakly_connected_components'] <= 1
            summary['reachability'] = graph_results.get('input_output_reachability', 0)
            summary['graph_efficiency'] = graph_results.get('avg_path_length', float('inf'))
        
        return summary
    
    def _get_computation_stats(self):
        """Get computation statistics from all analyzers."""
        stats = {}
        
        if hasattr(self.mi_analyzer, 'get_computation_stats'):
            stats['mi_analyzer'] = self.mi_analyzer.get_computation_stats()
        
        if hasattr(self.activity_analyzer, 'get_computation_stats'):
            stats['activity_analyzer'] = self.activity_analyzer.get_computation_stats()
        
        if hasattr(self.sensli_analyzer, 'get_computation_stats'):
            stats['sensli_analyzer'] = self.sensli_analyzer.get_computation_stats()
        
        if hasattr(self.graph_analyzer, 'get_computation_stats'):
            stats['graph_analyzer'] = self.graph_analyzer.get_computation_stats()
        
        return stats
    
    def clear_caches(self):
        """Clear all analyzer caches."""
        for analyzer in [self.mi_analyzer, self.activity_analyzer, self.sensli_analyzer, self.graph_analyzer]:
            if hasattr(analyzer, 'clear_cache'):
                analyzer.clear_cache()
    
    # Autocorrelation Framework Integration
    def collect_checkpoint_data(self, dataloader, epoch, performance_metrics):
        """Collect data for autocorrelation analysis."""
        if self.performance_analyzer is not None:
            self.performance_analyzer.collect_checkpoint_data(
                self.network, dataloader, epoch, performance_metrics
            )
    
    def update_autocorrelation_metrics(self, epoch, complete_metrics):
        """Update autocorrelation framework with computed metrics."""
        if self.performance_analyzer is not None:
            self.performance_analyzer.update_metrics_from_complete_system(epoch, complete_metrics)
    
    def analyze_correlations(self, min_history_length=20):
        """Analyze metric-performance correlations."""
        if self.performance_analyzer is not None:
            return self.performance_analyzer.analyze_metric_correlations(min_history_length)
        return {}
    
    def get_growth_recommendations(self, current_metrics):
        """Get growth recommendations based on learned patterns."""
        if self.performance_analyzer is not None:
            return self.performance_analyzer.get_growth_recommendations(current_metrics)
        return []
    
    def record_strategy_outcome(self, strategy_name, metrics_before, metrics_after, performance_improvement):
        """Record strategy outcome for learning."""
        if self.performance_analyzer is not None:
            self.performance_analyzer.record_strategy_outcome(
                strategy_name, metrics_before, metrics_after, performance_improvement
            )


# Backward compatibility - alias the old class name
MetricPerformanceAnalyzer = CompleteMetricsSystem

# Export classes
__all__ = ['CompleteMetricsSystem', 'MetricPerformanceAnalyzer']
