#!/usr/bin/env python3
"""
Performance Correlation Analyzer Component

Migrated from evolution.autocorrelation.performance_analyzer to use the IAnalyzer interface.
Implements autocorrelation framework for discovering metric-performance relationships.
"""

import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import logging
import time

from ...core.interfaces import (
    IAnalyzer, IModel, AnalysisReport, EvolutionContext,
    ComponentContract, ComponentVersion, Maturity, ResourceRequirements, ResourceLevel
)


class PerformanceCorrelationAnalyzer(IAnalyzer):
    """
    Analyzer component for discovering metric-performance relationships.
    
    This analyzer implements an autocorrelation framework that discovers which
    metrics predict learning success and training dynamics.
    
    Features:
    - Comprehensive metric collection and performance correlation
    - Pattern discovery in training dynamics
    - Strategy effectiveness learning
    - Temporal analysis of metric evolution
    - Predictive threshold detection
    """
    
    def __init__(self,
                 correlation_window: int = 50,
                 min_data_points: int = 10,
                 significance_threshold: float = 0.7,
                 pattern_discovery_enabled: bool = True,
                 name: str = None):
        super().__init__()
        self.correlation_window = correlation_window
        self.min_data_points = min_data_points
        self.significance_threshold = significance_threshold
        self.pattern_discovery_enabled = pattern_discovery_enabled
        self._name = name or "PerformanceCorrelationAnalyzer"
        
        # Data storage
        self.metric_history = []
        self.performance_history = []
        self.correlation_results = {}
        self.learned_patterns = {}
        self.strategy_effectiveness = defaultdict(list)
        
        # Component contract
        self._contract = ComponentContract(
            component_name=self._name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={'model', 'performance_metrics', 'training_history'},
            provided_outputs={'correlation_analysis', 'predictive_patterns', 'strategy_recommendations'},
            optional_inputs={'metric_history', 'correlation_window'},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=False,
                parallel_safe=True,
                estimated_runtime_seconds=3.0
            )
        )
    
    @property
    def contract(self) -> ComponentContract:
        """Component contract declaration."""
        return self._contract
    
    @property
    def name(self) -> str:
        """Component name."""
        return self._name
    
    def analyze(self, model: IModel, report: AnalysisReport, context: EvolutionContext) -> Dict[str, Any]:
        """
        Perform performance correlation analysis.
        
        Args:
            model: Model to analyze
            report: Current analysis report with metrics
            context: Evolution context with performance data
            
        Returns:
            Dictionary containing correlation analysis results
        """
        self._track_execution(self._perform_analysis)
        return self._perform_analysis(model, report, context)
    
    def _perform_analysis(self, model: IModel, report: AnalysisReport, context: EvolutionContext) -> Dict[str, Any]:
        """Internal analysis implementation."""
        try:
            # Collect current checkpoint data
            self._collect_checkpoint_data(model, report, context)
            
            # Perform correlation analysis if we have enough data
            analysis_results = {}
            
            if len(self.performance_history) >= self.min_data_points:
                # Core correlation analysis
                analysis_results['correlation_analysis'] = self._analyze_correlations()
                
                # Pattern discovery
                if self.pattern_discovery_enabled:
                    analysis_results['pattern_discovery'] = self._discover_patterns()
                
                # Strategy effectiveness
                analysis_results['strategy_effectiveness'] = self._analyze_strategy_effectiveness()
                
                # Predictive insights
                analysis_results['predictive_insights'] = self._generate_predictive_insights()
                
                # Temporal analysis
                analysis_results['temporal_analysis'] = self._analyze_temporal_patterns()
            
            # Always include current data summary
            analysis_results['current_summary'] = self._get_current_summary()
            analysis_results['data_quality'] = self._assess_data_quality()
            
            self.log(logging.INFO, f"Analyzed {len(self.performance_history)} data points, found {len(self.correlation_results)} significant correlations")
            
            return analysis_results
            
        except Exception as e:
            self.log(logging.ERROR, f"Performance correlation analysis failed: {str(e)}")
            raise
    
    def _collect_checkpoint_data(self, model: IModel, report: AnalysisReport, context: EvolutionContext):
        """
        Collect comprehensive metrics and performance data at current checkpoint.
        
        Args:
            model: Current model
            report: Analysis report with metrics
            context: Evolution context with performance data
        """
        timestamp = time.time()
        epoch = context.get('epoch', 0)
        
        # Extract performance metrics
        performance_metrics = context.get('performance_metrics', {})
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
        
        # Extract comprehensive metrics from report
        metrics_data = {
            'epoch': epoch,
            'timestamp': timestamp,
        }
        
        # Collect metrics from different analyzers
        for key, value in report.items():
            if key.startswith('metrics.') or key.startswith('analyzers.'):
                # Flatten nested metric data
                if isinstance(value, dict):
                    for metric_name, metric_value in value.items():
                        if isinstance(metric_value, (int, float)):
                            metrics_data[f"{key}.{metric_name}"] = metric_value
                elif isinstance(value, (int, float)):
                    metrics_data[key] = value
        
        self.metric_history.append(metrics_data)
        
        # Limit history size to manage memory
        max_history = self.correlation_window * 3
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
            self.metric_history = self.metric_history[-max_history:]
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """
        Analyze correlations between metrics and performance.
        
        Returns:
            Correlation analysis results
        """
        if len(self.performance_history) < self.min_data_points:
            return {'status': 'insufficient_data'}
        
        # Convert to DataFrame for analysis
        perf_df = pd.DataFrame(self.performance_history)
        metrics_df = pd.DataFrame(self.metric_history)
        
        # Combine dataframes
        combined_df = pd.merge(perf_df, metrics_df, on=['epoch', 'timestamp'], how='inner')
        
        if combined_df.empty:
            return {'status': 'no_matching_data'}
        
        # Calculate correlations
        correlations = {}
        performance_cols = ['train_accuracy', 'val_accuracy', 'generalization_gap']
        
        for perf_col in performance_cols:
            if perf_col in combined_df.columns:
                correlations[perf_col] = {}
                
                for col in combined_df.columns:
                    if col.startswith('metrics.') or col.startswith('analyzers.'):
                        try:
                            corr = combined_df[perf_col].corr(combined_df[col])
                            if not pd.isna(corr) and abs(corr) > self.significance_threshold:
                                correlations[perf_col][col] = {
                                    'correlation': corr,
                                    'strength': self._classify_correlation_strength(abs(corr)),
                                    'direction': 'positive' if corr > 0 else 'negative'
                                }
                        except Exception:
                            continue
        
        # Update stored results
        self.correlation_results.update(correlations)
        
        return {
            'status': 'success',
            'correlations': correlations,
            'data_points': len(combined_df),
            'significant_correlations': sum(len(v) for v in correlations.values())
        }
    
    def _classify_correlation_strength(self, abs_corr: float) -> str:
        """Classify correlation strength."""
        if abs_corr >= 0.9:
            return 'very_strong'
        elif abs_corr >= 0.7:
            return 'strong'
        elif abs_corr >= 0.5:
            return 'moderate'
        else:
            return 'weak'
    
    def _discover_patterns(self) -> Dict[str, Any]:
        """
        Discover patterns in training dynamics.
        
        Returns:
            Pattern discovery results
        """
        patterns = {}
        
        if len(self.performance_history) < self.min_data_points:
            return {'status': 'insufficient_data'}
        
        # Analyze performance trends
        recent_performance = self.performance_history[-self.min_data_points:]
        val_accuracies = [p['val_accuracy'] for p in recent_performance]
        
        # Detect plateau patterns
        if len(val_accuracies) >= 5:
            recent_std = np.std(val_accuracies[-5:])
            patterns['plateau_detected'] = recent_std < 0.01
            patterns['plateau_strength'] = 1.0 - min(recent_std / 0.01, 1.0)
        
        # Detect learning phases
        if len(val_accuracies) >= 10:
            # Simple phase detection based on derivative
            diffs = np.diff(val_accuracies)
            avg_improvement = np.mean(diffs)
            
            if avg_improvement > 0.01:
                patterns['learning_phase'] = 'rapid_improvement'
            elif avg_improvement > 0.001:
                patterns['learning_phase'] = 'steady_improvement'
            elif avg_improvement > -0.001:
                patterns['learning_phase'] = 'plateau'
            else:
                patterns['learning_phase'] = 'declining'
        
        # Store learned patterns
        self.learned_patterns.update(patterns)
        
        return {
            'status': 'success',
            'patterns': patterns,
            'pattern_count': len(patterns)
        }
    
    def _analyze_strategy_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze effectiveness of different strategies.
        
        Returns:
            Strategy effectiveness analysis
        """
        effectiveness = {}
        
        # This would analyze which strategies led to performance improvements
        # For now, provide a basic framework
        
        if self.strategy_effectiveness:
            for strategy, results in self.strategy_effectiveness.items():
                if results:
                    effectiveness[strategy] = {
                        'avg_improvement': np.mean(results),
                        'success_rate': sum(1 for r in results if r > 0) / len(results),
                        'applications': len(results)
                    }
        
        return {
            'strategies_analyzed': len(effectiveness),
            'effectiveness': effectiveness
        }
    
    def _generate_predictive_insights(self) -> Dict[str, Any]:
        """
        Generate predictive insights based on correlation analysis.
        
        Returns:
            Predictive insights
        """
        insights = {
            'predictions': [],
            'recommendations': [],
            'warning_indicators': []
        }
        
        # Analyze current metric values against learned correlations
        if self.correlation_results and self.metric_history:
            current_metrics = self.metric_history[-1]
            
            for perf_metric, correlations in self.correlation_results.items():
                for metric_name, corr_data in correlations.items():
                    if metric_name in current_metrics:
                        current_value = current_metrics[metric_name]
                        correlation = corr_data['correlation']
                        
                        # Generate prediction based on correlation
                        if abs(correlation) > 0.8:
                            if correlation > 0:
                                prediction = f"High {metric_name} ({current_value:.3f}) suggests good {perf_metric}"
                            else:
                                prediction = f"High {metric_name} ({current_value:.3f}) suggests poor {perf_metric}"
                            
                            insights['predictions'].append({
                                'metric': metric_name,
                                'performance': perf_metric,
                                'prediction': prediction,
                                'confidence': abs(correlation)
                            })
        
        return insights
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """
        Analyze temporal patterns in metrics and performance.
        
        Returns:
            Temporal analysis results
        """
        temporal = {}
        
        if len(self.performance_history) >= 20:
            # Analyze trends over time
            recent_epochs = [p['epoch'] for p in self.performance_history[-20:]]
            recent_val_acc = [p['val_accuracy'] for p in self.performance_history[-20:]]
            
            # Linear trend analysis
            if len(recent_epochs) == len(recent_val_acc):
                correlation = np.corrcoef(recent_epochs, recent_val_acc)[0, 1]
                temporal['trend_correlation'] = correlation
                temporal['trend_direction'] = 'improving' if correlation > 0.1 else 'stable' if correlation > -0.1 else 'declining'
        
        return temporal
    
    def _get_current_summary(self) -> Dict[str, Any]:
        """Get summary of current state."""
        summary = {
            'total_checkpoints': len(self.performance_history),
            'correlation_results_count': sum(len(v) for v in self.correlation_results.values()),
            'patterns_discovered': len(self.learned_patterns)
        }
        
        if self.performance_history:
            latest_perf = self.performance_history[-1]
            summary['latest_performance'] = {
                'epoch': latest_perf['epoch'],
                'val_accuracy': latest_perf['val_accuracy'],
                'train_accuracy': latest_perf['train_accuracy'],
                'generalization_gap': latest_perf['generalization_gap']
            }
        
        return summary
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess quality of collected data."""
        quality = {
            'sufficient_data': len(self.performance_history) >= self.min_data_points,
            'data_consistency': True,  # Would implement actual consistency checks
            'missing_values': 0  # Would count actual missing values
        }
        
        return quality
    
    def get_required_metrics(self) -> Set[str]:
        """Get metrics required by this analyzer."""
        return {
            'performance_metrics',
            'training_history',
            'metric_correlation_data'
        }
    
    def can_apply(self, context: EvolutionContext) -> bool:
        """Check if this analyzer can be applied to the given context."""
        return (
            self.validate_context(context) and
            'model' in context
        )
    
    def apply(self, context: EvolutionContext) -> bool:
        """Apply this analyzer (analysis happens via analyze method)."""
        return self.can_apply(context)
    
    def get_analysis_type(self) -> str:
        """Get the type of analysis this analyzer performs."""
        return "performance_correlation"
    
    def get_required_batches(self) -> int:
        """Get number of data batches required for analysis."""
        return 1
    
    # Utility methods for external use
    
    def record_strategy_effectiveness(self, strategy_name: str, improvement: float):
        """
        Record effectiveness of a strategy application.
        
        Args:
            strategy_name: Name of the strategy
            improvement: Performance improvement achieved
        """
        self.strategy_effectiveness[strategy_name].append(improvement)
    
    def get_top_predictive_metrics(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top predictive metrics across all performance measures.
        
        Args:
            n: Number of top metrics to return
            
        Returns:
            List of top predictive metrics
        """
        all_correlations = []
        
        for perf_metric, correlations in self.correlation_results.items():
            for metric_name, corr_data in correlations.items():
                all_correlations.append({
                    'metric': metric_name,
                    'performance': perf_metric,
                    'correlation': corr_data['correlation'],
                    'strength': corr_data['strength'],
                    'direction': corr_data['direction']
                })
        
        # Sort by absolute correlation value
        all_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return all_correlations[:n]
    
    def reset_history(self):
        """Reset all collected history data."""
        self.metric_history.clear()
        self.performance_history.clear()
        self.correlation_results.clear()
        self.learned_patterns.clear()
        self.strategy_effectiveness.clear()
        self.log(logging.INFO, "Reset performance correlation history")