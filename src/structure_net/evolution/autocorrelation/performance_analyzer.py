"""
Performance Analyzer - Core of the Autocorrelation Framework

This module implements the main PerformanceAnalyzer class that orchestrates
the entire autocorrelation framework for discovering metric-performance relationships.
"""

import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional
import time
import logging
import pickle

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Core of the autocorrelation framework that discovers which metrics predict learning success.
    
    This class orchestrates the collection of metrics and performance data, then
    uses various analysis components to discover predictive patterns and relationships.
    """
    
    def __init__(self):
        self.metric_history = []
        self.performance_history = []
        self.correlation_results = {}
        self.learned_patterns = {}
        self.strategy_effectiveness = defaultdict(list)
        
        # Initialize analysis components (will be created when needed)
        self._correlation_analyzer = None
        self._pattern_discovery = None
        self._strategy_learner = None
        self._temporal_analyzer = None
        self._threshold_detector = None
        
    def collect_checkpoint_data(self, network, dataloader, epoch, performance_metrics):
        """
        Collect comprehensive metrics at each checkpoint during training.
        
        Args:
            network: The neural network
            dataloader: Data loader for evaluation
            epoch: Current epoch number
            performance_metrics: Dict with 'train_acc', 'val_acc', 'train_loss', 'val_loss'
        """
        timestamp = time.time()
        
        # Store performance data
        perf_data = {
            'epoch': epoch,
            'timestamp': timestamp,
            'train_accuracy': performance_metrics.get('train_acc', 0.0),
            'val_accuracy': performance_metrics.get('val_acc', 0.0),
            'train_loss': performance_metrics.get('train_loss', float('inf')),
            'val_loss': performance_metrics.get('val_loss', float('inf')),
            'generalization_gap': performance_metrics.get('train_acc', 0.0) - performance_metrics.get('val_acc', 0.0)
        }
        self.performance_history.append(perf_data)
        
        # Collect comprehensive metrics (placeholder - will be filled by CompleteMetricsSystem)
        metrics_data = {
            'epoch': epoch,
            'timestamp': timestamp,
            
            # Placeholders for all metric categories
            'mi_efficiency_mean': 0.0,
            'mi_efficiency_min': 0.0,
            'mi_bottleneck_severity': 0.0,
            'total_information_loss': 0.0,
            
            'algebraic_connectivity': 0.0,
            'spectral_gap': 0.0,
            'betweenness_max': 0.0,
            'betweenness_variance': 0.0,
            'num_components': 0.0,
            'giant_component_size': 0.0,
            'avg_path_length': 0.0,
            'network_efficiency': 0.0,
            'percolation_distance': 0.0,
            
            'active_neuron_ratio': 0.0,
            'dead_neuron_count': 0.0,
            'activation_variance': 0.0,
            'activation_kurtosis': 0.0,
            'extrema_ratio': 0.0,
            
            'gradient_norm_mean': 0.0,
            'gradient_norm_variance': 0.0,
            'gradient_correlation': 0.0,
            'update_ratio': 0.0,
            
            'effective_rank': 0.0,
            'singular_value_decay': 0.0,
            'condition_number': 0.0,
            
            'feedforward_motifs': 0.0,
            'feedback_motifs': 0.0,
            'skip_connections_natural': 0.0
        }
        
        self.metric_history.append(metrics_data)
        
        logger.info(f"📊 Collected checkpoint data for epoch {epoch}")
        
    def update_metrics_from_complete_system(self, epoch, complete_metrics):
        """
        Update the latest metric entry with data from CompleteMetricsSystem.
        
        Args:
            epoch: Current epoch
            complete_metrics: Results from CompleteMetricsSystem.compute_all_metrics()
        """
        if not self.metric_history or self.metric_history[-1]['epoch'] != epoch:
            logger.warning(f"No metric entry found for epoch {epoch}")
            return
        
        # Extract key metrics from complete_metrics and update the latest entry
        latest_metrics = self.metric_history[-1]
        
        # MI Metrics
        mi_metrics = complete_metrics.get('mi_metrics', {})
        if mi_metrics:
            mi_efficiencies = [m['mi_efficiency'] for m in mi_metrics.values()]
            latest_metrics['mi_efficiency_mean'] = np.mean(mi_efficiencies) if mi_efficiencies else 0.0
            latest_metrics['mi_efficiency_min'] = np.min(mi_efficiencies) if mi_efficiencies else 0.0
            latest_metrics['mi_bottleneck_severity'] = sum(1 for eff in mi_efficiencies if eff < 0.3)
            latest_metrics['total_information_loss'] = sum(m.get('information_gap', 0) for m in mi_metrics.values())
        
        # Graph Metrics
        graph_metrics = complete_metrics.get('graph_metrics', {})
        latest_metrics['algebraic_connectivity'] = graph_metrics.get('algebraic_connectivity', 0.0)
        latest_metrics['spectral_gap'] = graph_metrics.get('spectral_gap', 0.0)
        latest_metrics['betweenness_max'] = graph_metrics.get('max_betweenness', 0.0)
        latest_metrics['betweenness_variance'] = graph_metrics.get('betweenness_std', 0.0) ** 2
        latest_metrics['num_components'] = graph_metrics.get('num_weakly_connected_components', 0.0)
        latest_metrics['giant_component_size'] = graph_metrics.get('largest_wcc_size', 0.0)
        latest_metrics['avg_path_length'] = graph_metrics.get('avg_path_length', 0.0)
        latest_metrics['network_efficiency'] = 1.0 / (latest_metrics['avg_path_length'] + 1e-10)
        latest_metrics['percolation_distance'] = graph_metrics.get('distance_to_percolation', 0.0)
        
        # Activity Metrics
        activity_metrics = complete_metrics.get('activity_metrics', {})
        if activity_metrics:
            active_ratios = [m['active_ratio'] for m in activity_metrics.values()]
            dead_counts = [m['total_neurons'] - m['active_neurons'] for m in activity_metrics.values()]
            activations = [m['mean_activation'] for m in activity_metrics.values()]
            
            latest_metrics['active_neuron_ratio'] = np.mean(active_ratios) if active_ratios else 0.0
            latest_metrics['dead_neuron_count'] = sum(dead_counts)
            latest_metrics['activation_variance'] = np.var(activations) if activations else 0.0
            latest_metrics['extrema_ratio'] = sum(m.get('saturated_neurons', 0) + (m['total_neurons'] - m['active_neurons']) 
                                                 for m in activity_metrics.values()) / sum(m['total_neurons'] for m in activity_metrics.values())
        
        # SensLI Metrics
        sensli_metrics = complete_metrics.get('sensli_metrics', {})
        if sensli_metrics:
            grad_sensitivities = [m['gradient_sensitivity'] for m in sensli_metrics.values()]
            latest_metrics['gradient_norm_mean'] = np.mean(grad_sensitivities) if grad_sensitivities else 0.0
            latest_metrics['gradient_norm_variance'] = np.var(grad_sensitivities) if grad_sensitivities else 0.0
            latest_metrics['gradient_correlation'] = np.mean([m.get('sensitivity_stability', 0) for m in sensli_metrics.values()])
        
        logger.info(f"📈 Updated metrics for epoch {epoch} with complete system data")
    
    def analyze_metric_correlations(self, min_history_length=20):
        """
        Find which metrics correlate with future performance improvements.
        
        Args:
            min_history_length: Minimum number of data points needed for analysis
            
        Returns:
            Dict with correlation analysis results
        """
        if len(self.metric_history) < min_history_length:
            logger.info(f"Not enough data for correlation analysis ({len(self.metric_history)} < {min_history_length})")
            return {}
        
        logger.info("🔍 Analyzing metric-performance correlations...")
        
        # Use CorrelationAnalyzer if available
        if self._correlation_analyzer is None:
            try:
                from .correlation_analyzer import CorrelationAnalyzer
                self._correlation_analyzer = CorrelationAnalyzer()
            except ImportError:
                # Fallback to basic analysis
                return self._basic_correlation_analysis()
        
        # Delegate to specialized analyzer
        results = self._correlation_analyzer.analyze_correlations(
            self.metric_history, self.performance_history
        )
        
        self.correlation_results = results.get('correlations', {})
        
        logger.info(f"✅ Correlation analysis complete. Found {len(results.get('top_predictive_metrics', []))} significant predictors.")
        
        return results
    
    def _basic_correlation_analysis(self):
        """Fallback basic correlation analysis when specialized analyzer is not available."""
        # Convert to DataFrame for easier analysis
        metrics_df = pd.DataFrame(self.metric_history)
        performance_df = pd.DataFrame(self.performance_history)
        
        # Merge on epoch
        combined_df = pd.merge(metrics_df, performance_df, on='epoch', suffixes=('_metric', '_perf'))
        
        # Calculate future performance improvement (5 epochs ahead)
        combined_df['future_val_improvement'] = combined_df['val_accuracy'].shift(-5) - combined_df['val_accuracy']
        
        # Remove last 5 rows (no future data)
        analysis_df = combined_df[:-5].copy()
        
        if len(analysis_df) < 10:
            logger.warning("Insufficient data after future performance calculation")
            return {}
        
        # Feature columns (all metrics except performance and metadata)
        feature_cols = [col for col in analysis_df.columns 
                       if col not in ['epoch', 'timestamp_metric', 'timestamp_perf', 
                                     'train_accuracy', 'val_accuracy', 'train_loss', 'val_loss',
                                     'generalization_gap', 'future_val_improvement']]
        
        correlations = {}
        
        # Basic Pearson correlation
        for col in feature_cols:
            try:
                corr_val = analysis_df[col].fillna(0).corr(analysis_df['future_val_improvement'].fillna(0))
                correlations[col] = {
                    'pearson_val': corr_val,
                    'significant_val': abs(corr_val) > 0.3  # Simple threshold
                }
            except Exception as e:
                logger.warning(f"Failed to compute correlation for {col}: {e}")
                correlations[col] = {'pearson_val': 0.0, 'significant_val': False}
        
        # Find top predictive metrics
        top_metrics = sorted(
            [(metric, abs(data['pearson_val'])) for metric, data in correlations.items()],
            key=lambda x: x[1], reverse=True
        )[:10]
        
        return {
            'correlations': correlations,
            'top_predictive_metrics': [{'metric': m, 'correlation': c} for m, c in top_metrics],
            'analysis_summary': {
                'total_metrics_analyzed': len(correlations),
                'significant_predictors': sum(1 for c in correlations.values() if c['significant_val'])
            }
        }
    
    def get_growth_recommendations(self, current_metrics):
        """
        Use learned patterns to suggest optimal growth actions.
        
        Args:
            current_metrics: Dict of current network metrics
            
        Returns:
            List of recommended actions with confidence scores
        """
        if not self.correlation_results:
            return [{'action': 'collect_more_data', 'confidence': 0.0, 'reason': 'Insufficient correlation data'}]
        
        # Use StrategyLearner if available
        if self._strategy_learner is None:
            try:
                from .strategy_learner import StrategyLearner
                self._strategy_learner = StrategyLearner(self.correlation_results, self.strategy_effectiveness)
            except ImportError:
                # Fallback to basic recommendations
                return self._basic_growth_recommendations(current_metrics)
        
        return self._strategy_learner.get_recommendations(current_metrics)
    
    def _basic_growth_recommendations(self, current_metrics):
        """Fallback basic recommendations when specialized learner is not available."""
        recommendations = []
        
        # Simple rule-based recommendations
        if current_metrics.get('mi_efficiency_mean', 0) < 0.3:
            recommendations.append({
                'action': 'add_layer_for_information_flow',
                'confidence': 0.7,
                'reason': 'Low MI efficiency detected'
            })
        
        if current_metrics.get('active_neuron_ratio', 1.0) < 0.1:
            recommendations.append({
                'action': 'add_extrema_aware_patches',
                'confidence': 0.6,
                'reason': 'High dead neuron ratio detected'
            })
        
        if current_metrics.get('algebraic_connectivity', 0) < 0.1:
            recommendations.append({
                'action': 'add_skip_connections',
                'confidence': 0.5,
                'reason': 'Poor network connectivity detected'
            })
        
        return recommendations[:3]  # Top 3 recommendations
    
    def record_strategy_outcome(self, strategy_name, metrics_before, metrics_after, performance_improvement):
        """
        Record the outcome of a growth strategy for learning.
        
        Args:
            strategy_name: Name of the strategy that was applied
            metrics_before: Metrics before applying the strategy
            metrics_after: Metrics after applying the strategy
            performance_improvement: Actual performance improvement achieved
        """
        outcome = {
            'strategy': strategy_name,
            'metrics_before': metrics_before.copy(),
            'metrics_after': metrics_after.copy(),
            'performance_improvement': performance_improvement,
            'timestamp': time.time()
        }
        
        self.strategy_effectiveness[strategy_name].append(outcome)
        
        logger.info(f"📝 Recorded outcome for strategy '{strategy_name}': {performance_improvement:+.3f} improvement")
    
    def get_strategy_effectiveness_summary(self):
        """Get summary of strategy effectiveness based on recorded outcomes."""
        summary = {}
        
        for strategy, outcomes in self.strategy_effectiveness.items():
            if outcomes:
                improvements = [o['performance_improvement'] for o in outcomes]
                summary[strategy] = {
                    'num_applications': len(outcomes),
                    'avg_improvement': np.mean(improvements),
                    'std_improvement': np.std(improvements),
                    'success_rate': sum(1 for imp in improvements if imp > 0.01) / len(improvements),
                    'best_improvement': max(improvements),
                    'worst_improvement': min(improvements)
                }
        
        return summary
    
    def save_learned_patterns(self, filepath):
        """Save learned patterns to file for cross-experiment learning."""
        patterns = {
            'correlation_results': self.correlation_results,
            'learned_patterns': self.learned_patterns,
            'strategy_effectiveness': dict(self.strategy_effectiveness),
            'metric_history_summary': {
                'total_checkpoints': len(self.metric_history),
                'epochs_covered': [m['epoch'] for m in self.metric_history] if self.metric_history else []
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(patterns, f)
        
        logger.info(f"💾 Saved learned patterns to {filepath}")
    
    def load_learned_patterns(self, filepath):
        """Load learned patterns from file for cross-experiment learning."""
        try:
            with open(filepath, 'rb') as f:
                patterns = pickle.load(f)
            
            self.correlation_results.update(patterns.get('correlation_results', {}))
            self.learned_patterns.update(patterns.get('learned_patterns', {}))
            
            # Merge strategy effectiveness
            for strategy, outcomes in patterns.get('strategy_effectiveness', {}).items():
                self.strategy_effectiveness[strategy].extend(outcomes)
            
            logger.info(f"📂 Loaded learned patterns from {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to load patterns from {filepath}: {e}")
    
    def get_insights_summary(self):
        """Get a comprehensive summary of insights discovered."""
        summary = {
            'data_collection': {
                'total_checkpoints': len(self.metric_history),
                'total_performance_points': len(self.performance_history),
                'epochs_covered': list(range(len(self.metric_history))) if self.metric_history else []
            },
            'correlation_analysis': {
                'total_metrics_analyzed': len(self.correlation_results),
                'significant_correlations': sum(1 for c in self.correlation_results.values() 
                                              if c.get('significant_val', False) or c.get('significant_train', False))
            },
            'strategy_learning': {
                'strategies_tested': len(self.strategy_effectiveness),
                'total_applications': sum(len(outcomes) for outcomes in self.strategy_effectiveness.values())
            }
        }
        
        return summary


# Export the analyzer
__all__ = ['PerformanceAnalyzer']
