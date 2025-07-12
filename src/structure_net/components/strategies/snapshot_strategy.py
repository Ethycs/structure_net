"""
Snapshot Strategy Component.

Determines when and how to take snapshots based on network
evolution state and performance metrics.
"""

import torch
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from ...core import (
    BaseStrategy, ComponentContract, ComponentVersion,
    Maturity, ResourceRequirements, ResourceLevel,
    AnalysisReport, EvolutionContext, EvolutionPlan
)


class SnapshotStrategy(BaseStrategy):
    """
    Strategy for intelligent snapshot decisions.
    
    Analyzes network state and evolution progress to determine
    optimal snapshot timing and policies.
    """
    
    def __init__(self,
                 min_improvement: float = 0.02,
                 growth_priority: float = 0.9,
                 performance_priority: float = 0.7,
                 milestone_priority: float = 0.5,
                 min_interval_epochs: int = 5,
                 name: str = None):
        """
        Initialize snapshot strategy.
        
        Args:
            min_improvement: Minimum performance improvement for snapshot
            growth_priority: Priority for growth event snapshots (0-1)
            performance_priority: Priority for performance snapshots (0-1)
            milestone_priority: Priority for milestone snapshots (0-1)
            min_interval_epochs: Minimum epochs between snapshots
            name: Optional custom name
        """
        super().__init__(name or "SnapshotStrategy")
        
        self.min_improvement = min_improvement
        self.growth_priority = growth_priority
        self.performance_priority = performance_priority
        self.milestone_priority = milestone_priority
        self.min_interval_epochs = min_interval_epochs
        
        # Track snapshot history
        self.last_snapshot_epoch: Optional[int] = None
        self.last_performance: Optional[float] = None
        self.last_architecture_hash: Optional[int] = None
        
        # Required analysis
        self._required_analysis = {
            'model_stats',
            'performance_metrics',
            'growth_analysis'
        }
        
        # Define contract
        self._contract = ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs=set(self._required_analysis),
            optional_inputs={'snapshot_history', 'resource_usage'},
            provided_outputs={'snapshot_plan'},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    @property
    def contract(self) -> ComponentContract:
        """Return component contract."""
        return self._contract
    
    def get_strategy_type(self) -> str:
        """Return strategy type."""
        return 'snapshot'
    
    def propose_plan(self, report: AnalysisReport, 
                    context: EvolutionContext) -> EvolutionPlan:
        """
        Propose snapshot plan based on analysis.
        
        Args:
            report: Analysis report with metrics
            context: Evolution context
            
        Returns:
            Evolution plan for snapshots
        """
        # Check if enough time has passed
        if not self._check_interval(context):
            return self._create_no_op_plan("Too soon since last snapshot")
        
        # Evaluate snapshot criteria
        criteria = self._evaluate_criteria(report, context)
        
        # Determine if snapshot is needed
        if not any(criteria.values()):
            return self._create_no_op_plan("No snapshot criteria met")
        
        # Create snapshot plan
        plan = self._create_snapshot_plan(criteria, report, context)
        
        return plan
    
    def _check_interval(self, context: EvolutionContext) -> bool:
        """Check if enough epochs have passed since last snapshot."""
        if self.last_snapshot_epoch is None:
            return True
        
        epochs_since = context.epoch - self.last_snapshot_epoch
        return epochs_since >= self.min_interval_epochs
    
    def _evaluate_criteria(self, report: AnalysisReport,
                          context: EvolutionContext) -> Dict[str, bool]:
        """Evaluate snapshot criteria."""
        criteria = {
            'growth_event': False,
            'performance_improvement': False,
            'structural_change': False,
            'milestone_epoch': False,
            'critical_state': False
        }
        
        # Check for growth events
        growth_analysis = report.get('growth_analysis', {})
        if growth_analysis.get('growth_occurred', False):
            criteria['growth_event'] = True
        
        # Check performance improvement
        performance = report.get('performance_metrics', {})
        current_perf = performance.get('accuracy', performance.get('loss'))
        
        if current_perf is not None and self.last_performance is not None:
            improvement = abs(current_perf - self.last_performance)
            if improvement >= self.min_improvement:
                criteria['performance_improvement'] = True
        
        # Check structural changes
        model_stats = report.get('model_stats', {})
        arch_hash = self._compute_architecture_hash(model_stats)
        
        if self.last_architecture_hash is not None:
            if arch_hash != self.last_architecture_hash:
                criteria['structural_change'] = True
        
        # Check milestone epochs
        epoch = context.epoch
        milestones = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]
        if epoch in milestones:
            criteria['milestone_epoch'] = True
        
        # Check critical states (high loss variance, instability)
        if self._detect_critical_state(report):
            criteria['critical_state'] = True
        
        return criteria
    
    def _create_snapshot_plan(self, criteria: Dict[str, bool],
                            report: AnalysisReport,
                            context: EvolutionContext) -> EvolutionPlan:
        """Create snapshot plan based on criteria."""
        # Calculate priority based on which criteria are met
        priority = 0.0
        reasons = []
        
        if criteria['growth_event']:
            priority = max(priority, self.growth_priority)
            reasons.append('growth_event')
        
        if criteria['performance_improvement']:
            priority = max(priority, self.performance_priority)
            reasons.append('performance_improvement')
        
        if criteria['structural_change']:
            priority = max(priority, 0.8)  # High priority
            reasons.append('structural_change')
        
        if criteria['milestone_epoch']:
            priority = max(priority, self.milestone_priority)
            reasons.append(f'milestone_epoch_{context.epoch}')
        
        if criteria['critical_state']:
            priority = 1.0  # Maximum priority
            reasons.append('critical_state')
        
        # Create plan
        plan = EvolutionPlan({
            'type': 'snapshot',
            'action': 'save',
            'criteria_met': criteria,
            'reasons': reasons,
            'metadata': {
                'epoch': context.epoch,
                'performance': report.get('performance_metrics', {}),
                'model_stats': report.get('model_stats', {}),
                'timestamp': datetime.now().isoformat()
            }
        })
        
        plan.priority = priority
        plan.estimated_impact = 0.0  # Snapshots don't impact performance
        plan.created_by = self.name
        
        # Update tracking
        self.last_snapshot_epoch = context.epoch
        
        performance = report.get('performance_metrics', {})
        current_perf = performance.get('accuracy', performance.get('loss'))
        if current_perf is not None:
            self.last_performance = current_perf
        
        model_stats = report.get('model_stats', {})
        self.last_architecture_hash = self._compute_architecture_hash(model_stats)
        
        return plan
    
    def _compute_architecture_hash(self, model_stats: Dict[str, Any]) -> int:
        """Compute hash of architecture for change detection."""
        # Use architecture list if available
        arch = model_stats.get('architecture', [])
        if arch:
            return hash(tuple(arch))
        
        # Fallback to parameter count
        total_params = model_stats.get('total_parameters', 0)
        active_params = model_stats.get('active_connections', 0)
        
        return hash((total_params, active_params))
    
    def _detect_critical_state(self, report: AnalysisReport) -> bool:
        """Detect if network is in critical state requiring snapshot."""
        # Check for training instability
        performance = report.get('performance_metrics', {})
        
        # High loss variance
        if 'loss_variance' in performance:
            if performance['loss_variance'] > 0.5:
                return True
        
        # Gradient explosion/vanishing
        if 'gradient_stats' in report:
            grad_stats = report['gradient_stats']
            grad_norm = grad_stats.get('gradient_norm', 0)
            
            if grad_norm > 100 or (grad_norm < 1e-6 and grad_norm > 0):
                return True
        
        # Dead neurons
        if 'layer_health' in report:
            health = report['layer_health']
            dead_ratio = health.get('dead_neuron_ratio', 0)
            
            if dead_ratio > 0.5:
                return True
        
        return False
    
    def _create_no_op_plan(self, reason: str) -> EvolutionPlan:
        """Create no-operation plan."""
        plan = EvolutionPlan({
            'type': 'snapshot',
            'action': 'skip',
            'reason': reason
        })
        
        plan.priority = 0.0
        plan.estimated_impact = 0.0
        plan.created_by = self.name
        
        return plan
    
    def analyze_snapshot_history(self, snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze snapshot history for insights.
        
        Args:
            snapshots: List of snapshot metadata
            
        Returns:
            Analysis of snapshot patterns
        """
        if not snapshots:
            return {'status': 'no_history'}
        
        # Analyze snapshot frequency
        epochs = [s['epoch'] for s in snapshots]
        intervals = [epochs[i] - epochs[i-1] for i in range(1, len(epochs))]
        
        # Analyze reasons
        reason_counts = {}
        for snap in snapshots:
            reasons = snap.get('reasons', [])
            for reason in reasons:
                reason_type = reason.split('_')[0]
                reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1
        
        # Analyze sizes
        sizes = [s.get('file_size_mb', 0) for s in snapshots]
        
        return {
            'total_snapshots': len(snapshots),
            'average_interval': sum(intervals) / len(intervals) if intervals else 0,
            'min_interval': min(intervals) if intervals else 0,
            'max_interval': max(intervals) if intervals else 0,
            'reason_distribution': reason_counts,
            'average_size_mb': sum(sizes) / len(sizes) if sizes else 0,
            'total_size_mb': sum(sizes),
            'recommendations': self._generate_recommendations(intervals, reason_counts)
        }
    
    def _generate_recommendations(self, intervals: List[int], 
                                reason_counts: Dict[str, int]) -> List[str]:
        """Generate recommendations based on history."""
        recommendations = []
        
        # Check interval consistency
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            if avg_interval < self.min_interval_epochs:
                recommendations.append(
                    f"Consider increasing min_interval_epochs to {int(avg_interval * 1.5)} "
                    "to reduce snapshot frequency"
                )
        
        # Check reason balance
        total_reasons = sum(reason_counts.values())
        if total_reasons > 0:
            growth_ratio = reason_counts.get('growth', 0) / total_reasons
            perf_ratio = reason_counts.get('performance', 0) / total_reasons
            
            if growth_ratio > 0.7:
                recommendations.append(
                    "High proportion of growth snapshots. Consider reducing growth_priority "
                    "if disk space is a concern"
                )
            
            if perf_ratio < 0.1:
                recommendations.append(
                    "Few performance-based snapshots. Consider lowering min_improvement "
                    "threshold to capture more performance milestones"
                )
        
        return recommendations