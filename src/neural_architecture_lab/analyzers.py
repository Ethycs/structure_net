"""
Analysis tools for extracting insights from experiment results.
"""

import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

from .core import (
    Hypothesis, ExperimentResult, HypothesisResult,
    HypothesisCategory
)


class StatisticalAnalyzer:
    """Statistical analysis of experiment results."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def analyze_results(
        self, 
        results: List[ExperimentResult],
        success_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Perform statistical analysis on experiment results.
        
        Returns:
            Dictionary with statistical summary
        """
        if not results:
            return {
                'meets_success_criteria': False,
                'statistically_significant': False,
                'confidence': 0.0,
                'effect_size': 0.0
            }
        
        # Extract primary metrics
        primary_metrics = [r.primary_metric for r in results]
        
        # Basic statistics
        mean = np.mean(primary_metrics)
        std = np.std(primary_metrics)
        median = np.median(primary_metrics)
        q1, q3 = np.percentile(primary_metrics, [25, 75])
        
        # Check success criteria
        meets_criteria = all(
            mean >= threshold 
            for metric, threshold in success_metrics.items()
            if metric in ['accuracy', 'primary_metric']
        )
        
        # Statistical significance (one-sample t-test against baseline)
        baseline = success_metrics.get('baseline', 0.0)
        if len(primary_metrics) > 1 and std > 0:
            t_stat, p_value = stats.ttest_1samp(primary_metrics, baseline)
            statistically_significant = p_value < self.significance_level
            
            # Effect size (Cohen's d)
            effect_size = (mean - baseline) / std
        else:
            statistically_significant = False
            p_value = 1.0
            effect_size = 0.0
        
        # Confidence interval
        if len(primary_metrics) > 1:
            confidence_interval = stats.t.interval(
                1 - self.significance_level,
                len(primary_metrics) - 1,
                loc=mean,
                scale=stats.sem(primary_metrics)
            )
        else:
            confidence_interval = (mean, mean)
        
        # Additional metrics analysis
        metrics_summary = {}
        if results[0].metrics:
            metric_names = list(results[0].metrics.keys())
            for metric in metric_names:
                values = [r.metrics.get(metric, 0) for r in results]
                metrics_summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return {
            'meets_success_criteria': meets_criteria,
            'statistically_significant': statistically_significant,
            'confidence': 1 - p_value if 'p_value' in locals() else 0.0,
            'effect_size': effect_size,
            'mean': mean,
            'std': std,
            'median': median,
            'iqr': (q1, q3),
            'confidence_interval': confidence_interval,
            'p_value': p_value if 'p_value' in locals() else 1.0,
            'n_samples': len(primary_metrics),
            'metrics_summary': metrics_summary
        }
    
    def compare_groups(
        self, 
        group1: List[float], 
        group2: List[float]
    ) -> Dict[str, Any]:
        """Compare two groups of results."""
        if not group1 or not group2:
            return {
                'significant': False,
                'p_value': 1.0,
                'effect_size': 0.0
            }
        
        # T-test
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.std(group1)**2 + np.std(group2)**2) / 2
        )
        if pooled_std > 0:
            effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std
        else:
            effect_size = 0.0
        
        return {
            'significant': p_value < self.significance_level,
            'p_value': p_value,
            'effect_size': effect_size,
            'group1_mean': np.mean(group1),
            'group2_mean': np.mean(group2),
            'difference': np.mean(group1) - np.mean(group2)
        }


class InsightExtractor:
    """Extract actionable insights from experiment results."""
    
    def extract_insights(
        self,
        hypothesis: Hypothesis,
        results: List[ExperimentResult],
        statistical_summary: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Extract insights from experiment results.
        
        Returns:
            Dictionary with key_insights, unexpected_findings, and suggested_hypotheses
        """
        insights = {
            'key_insights': [],
            'unexpected_findings': [],
            'suggested_hypotheses': []
        }
        
        if not results:
            return insights
        
        # Analyze based on hypothesis category
        if hypothesis.category == HypothesisCategory.ARCHITECTURE:
            insights.update(self._analyze_architecture_results(hypothesis, results, statistical_summary))
        elif hypothesis.category == HypothesisCategory.GROWTH:
            insights.update(self._analyze_growth_results(hypothesis, results, statistical_summary))
        elif hypothesis.category == HypothesisCategory.SPARSITY:
            insights.update(self._analyze_sparsity_results(hypothesis, results, statistical_summary))
        elif hypothesis.category == HypothesisCategory.TRAINING:
            insights.update(self._analyze_training_results(hypothesis, results, statistical_summary))
        
        # General insights
        if statistical_summary['statistically_significant']:
            effect_size = statistical_summary['effect_size']
            if abs(effect_size) > 0.8:
                insights['key_insights'].append(
                    f"Large effect size ({effect_size:.2f}) indicates strong practical significance"
                )
            elif abs(effect_size) < 0.2:
                insights['unexpected_findings'].append(
                    "Despite statistical significance, effect size is small - practical impact may be limited"
                )
        
        # Variance insights
        if statistical_summary['std'] / statistical_summary['mean'] > 0.3:
            insights['unexpected_findings'].append(
                "High variance in results suggests sensitivity to initialization or hyperparameters"
            )
            insights['suggested_hypotheses'].append(
                "Investigate factors contributing to high variance in results"
            )
        
        return insights
    
    def _analyze_architecture_results(
        self, 
        hypothesis: Hypothesis,
        results: List[ExperimentResult],
        summary: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Analyze architecture-specific results."""
        insights = {
            'key_insights': [],
            'unexpected_findings': [],
            'suggested_hypotheses': []
        }
        
        # Group by architecture type
        arch_groups = defaultdict(list)
        for result in results:
            arch_type = self._classify_architecture(result.model_architecture)
            arch_groups[arch_type].append(result)
        
        # Compare architectures
        if len(arch_groups) > 1:
            best_arch = max(arch_groups.items(), key=lambda x: np.mean([r.primary_metric for r in x[1]]))
            worst_arch = min(arch_groups.items(), key=lambda x: np.mean([r.primary_metric for r in x[1]]))
            
            if best_arch[0] != worst_arch[0]:
                improvement = (
                    np.mean([r.primary_metric for r in best_arch[1]]) / 
                    np.mean([r.primary_metric for r in worst_arch[1]]) - 1
                )
                insights['key_insights'].append(
                    f"{best_arch[0]} architecture outperforms {worst_arch[0]} by {improvement:.1%}"
                )
        
        # Parameter efficiency
        if 'metrics_summary' in summary and 'final_parameters' in summary['metrics_summary']:
            param_stats = summary['metrics_summary']['final_parameters']
            if param_stats['std'] / param_stats['mean'] > 0.2:
                insights['unexpected_findings'].append(
                    "Large variation in model sizes despite similar architectures"
                )
        
        # Growth analysis
        if 'metrics_summary' in summary and 'growth_events' in summary['metrics_summary']:
            growth_stats = summary['metrics_summary']['growth_events']
            if growth_stats['mean'] > 0:
                insights['key_insights'].append(
                    f"Models underwent average of {growth_stats['mean']:.1f} growth events"
                )
                if growth_stats['std'] > growth_stats['mean'] * 0.5:
                    insights['suggested_hypotheses'].append(
                        "Investigate optimal growth triggers and thresholds"
                    )
        
        return insights
    
    def _analyze_growth_results(
        self,
        hypothesis: Hypothesis,
        results: List[ExperimentResult],
        summary: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Analyze growth-specific results."""
        insights = {
            'key_insights': [],
            'unexpected_findings': [],
            'suggested_hypotheses': []
        }
        
        # Analyze growth patterns
        growth_counts = [r.metrics.get('growth_events', 0) for r in results]
        final_params = [r.metrics.get('final_parameters', 0) for r in results]
        accuracies = [r.metrics.get('accuracy', 0) for r in results]
        
        if any(growth_counts):
            # Correlation between growth and accuracy
            if len(set(growth_counts)) > 1:
                corr, p_value = stats.pearsonr(growth_counts, accuracies)
                if abs(corr) > 0.5 and p_value < 0.05:
                    direction = "positive" if corr > 0 else "negative"
                    insights['key_insights'].append(
                        f"Strong {direction} correlation ({corr:.2f}) between growth events and accuracy"
                    )
            
            # Efficiency analysis
            params_per_growth = [
                (p - min(final_params)) / (g + 1) 
                for p, g in zip(final_params, growth_counts)
            ]
            avg_params_per_growth = np.mean(params_per_growth)
            
            insights['key_insights'].append(
                f"Average parameter increase per growth event: {avg_params_per_growth:.0f}"
            )
            
            # Check for diminishing returns
            if len(results) > 5:
                sorted_by_growth = sorted(zip(growth_counts, accuracies))
                if len(sorted_by_growth) > 3:
                    early_growth = sorted_by_growth[:len(sorted_by_growth)//2]
                    late_growth = sorted_by_growth[len(sorted_by_growth)//2:]
                    
                    early_improvement = np.mean([a for _, a in early_growth])
                    late_improvement = np.mean([a for _, a in late_growth])
                    
                    if early_improvement > late_improvement * 1.1:
                        insights['unexpected_findings'].append(
                            "Diminishing returns observed with increased growth events"
                        )
                        insights['suggested_hypotheses'].append(
                            "Investigate adaptive growth thresholds based on current performance"
                        )
        
        return insights
    
    def _analyze_sparsity_results(
        self,
        hypothesis: Hypothesis,
        results: List[ExperimentResult],
        summary: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Analyze sparsity-specific results."""
        insights = {
            'key_insights': [],
            'unexpected_findings': [],
            'suggested_hypotheses': []
        }
        
        # Extract sparsity levels
        sparsity_levels = []
        for result in results:
            # Try to get from metrics or parameters
            sparsity = result.metrics.get('sparsity', 
                      self._extract_sparsity_from_params(result))
            if sparsity is not None:
                sparsity_levels.append(sparsity)
        
        if sparsity_levels:
            accuracies = [r.metrics.get('accuracy', 0) for r in results]
            
            # Find optimal sparsity
            if len(set(sparsity_levels)) > 3:
                # Fit polynomial to find optimal point
                z = np.polyfit(sparsity_levels, accuracies, 2)
                p = np.poly1d(z)
                
                # Find maximum in reasonable range
                test_sparsities = np.linspace(
                    min(sparsity_levels), 
                    max(sparsity_levels), 
                    100
                )
                optimal_sparsity = test_sparsities[np.argmax(p(test_sparsities))]
                
                insights['key_insights'].append(
                    f"Optimal sparsity appears to be around {optimal_sparsity:.1%}"
                )
                
                # Check if extreme sparsity works
                high_sparsity_results = [
                    (s, a) for s, a in zip(sparsity_levels, accuracies) 
                    if s > 0.3
                ]
                if high_sparsity_results:
                    avg_high_sparsity_acc = np.mean([a for _, a in high_sparsity_results])
                    overall_avg = np.mean(accuracies)
                    if avg_high_sparsity_acc > overall_avg * 0.95:
                        insights['unexpected_findings'].append(
                            "High sparsity (>30%) maintains competitive accuracy"
                        )
                        insights['suggested_hypotheses'].append(
                            "Explore extreme sparsity with specialized training techniques"
                        )
        
        return insights
    
    def _analyze_training_results(
        self,
        hypothesis: Hypothesis,
        results: List[ExperimentResult],
        summary: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Analyze training-specific results."""
        insights = {
            'key_insights': [],
            'unexpected_findings': [],
            'suggested_hypotheses': []
        }
        
        # Learning rate analysis
        lr_strategies = defaultdict(list)
        for result in results:
            strategy = self._extract_lr_strategy(result)
            if strategy:
                lr_strategies[strategy].append(result)
        
        if len(lr_strategies) > 1:
            # Compare strategies
            strategy_performance = {
                strategy: np.mean([r.primary_metric for r in results])
                for strategy, results in lr_strategies.items()
            }
            
            best_strategy = max(strategy_performance.items(), key=lambda x: x[1])
            worst_strategy = min(strategy_performance.items(), key=lambda x: x[1])
            
            if best_strategy[0] != worst_strategy[0]:
                improvement = (best_strategy[1] / worst_strategy[1] - 1) * 100
                insights['key_insights'].append(
                    f"{best_strategy[0]} learning rate strategy outperforms {worst_strategy[0]} by {improvement:.1f}%"
                )
        
        # Convergence analysis
        convergence_epochs = []
        for result in results:
            if result.training_history:
                # Find epoch where 90% of final accuracy was reached
                final_acc = result.training_history[-1].get('test_accuracy', 0)
                target_acc = final_acc * 0.9
                
                for i, epoch_data in enumerate(result.training_history):
                    if epoch_data.get('test_accuracy', 0) >= target_acc:
                        convergence_epochs.append(i)
                        break
        
        if convergence_epochs:
            avg_convergence = np.mean(convergence_epochs)
            insights['key_insights'].append(
                f"Average convergence to 90% of final accuracy: {avg_convergence:.1f} epochs"
            )
            
            if np.std(convergence_epochs) > avg_convergence * 0.5:
                insights['unexpected_findings'].append(
                    "High variance in convergence speed across experiments"
                )
                insights['suggested_hypotheses'].append(
                    "Investigate initialization strategies for more consistent convergence"
                )
        
        return insights
    
    def _classify_architecture(self, architecture: List[int]) -> str:
        """Classify architecture type."""
        if not architecture or len(architecture) < 3:
            return "unknown"
        
        # Check depth
        depth = len(architecture) - 1
        
        # Check shape
        decreasing = all(architecture[i] >= architecture[i+1] for i in range(len(architecture)-2))
        increasing = all(architecture[i] <= architecture[i+1] for i in range(1, len(architecture)-1))
        
        if depth <= 3:
            return "shallow"
        elif depth >= 6:
            return "deep"
        elif decreasing:
            return "pyramid"
        elif increasing:
            return "inverse_pyramid"
        else:
            return "irregular"
    
    def _extract_sparsity_from_params(self, result: ExperimentResult) -> Optional[float]:
        """Extract sparsity from experiment parameters."""
        # This would need to check the experiment's parameters
        # For now, return None
        return None
    
    def _extract_lr_strategy(self, result: ExperimentResult) -> Optional[str]:
        """Extract learning rate strategy from result."""
        # This would need to check the experiment's parameters
        # For now, return None
        return None


class PerformanceAnalyzer:
    """Analyze performance characteristics of models."""
    
    def analyze_efficiency(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Analyze efficiency metrics across results."""
        if not results:
            return {}
        
        # Extract key metrics
        accuracies = [r.metrics.get('accuracy', 0) for r in results]
        parameters = [r.model_parameters for r in results]
        training_times = [r.training_time for r in results]
        
        # Calculate efficiency metrics
        efficiency_metrics = {
            'accuracy_per_parameter': [
                a / (p / 1e6) if p > 0 else 0 
                for a, p in zip(accuracies, parameters)
            ],
            'accuracy_per_second': [
                a / t if t > 0 else 0 
                for a, t in zip(accuracies, training_times)
            ],
            'parameters_per_second': [
                p / t if t > 0 else 0 
                for p, t in zip(parameters, training_times)
            ]
        }
        
        # Find Pareto optimal models
        pareto_optimal = self._find_pareto_optimal(accuracies, parameters)
        
        return {
            'efficiency_metrics': efficiency_metrics,
            'pareto_optimal_indices': pareto_optimal,
            'best_accuracy_per_param': max(efficiency_metrics['accuracy_per_parameter']),
            'best_accuracy_per_second': max(efficiency_metrics['accuracy_per_second'])
        }
    
    def _find_pareto_optimal(self, objective1: List[float], objective2: List[float]) -> List[int]:
        """Find Pareto optimal points (maximize objective1, minimize objective2)."""
        n = len(objective1)
        pareto_optimal = []
        
        for i in range(n):
            is_optimal = True
            for j in range(n):
                if i != j:
                    # Check if j dominates i
                    if objective1[j] >= objective1[i] and objective2[j] <= objective2[i]:
                        if objective1[j] > objective1[i] or objective2[j] < objective2[i]:
                            is_optimal = False
                            break
            
            if is_optimal:
                pareto_optimal.append(i)
        
        return pareto_optimal