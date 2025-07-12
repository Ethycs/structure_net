"""
Snapshot Metric Component.

Tracks metrics related to snapshot efficiency and effectiveness.
"""

import torch
from typing import Dict, Any, Optional, Union
from datetime import datetime

from ...core import (
    BaseMetric, ComponentContract, ComponentVersion,
    Maturity, ResourceRequirements, ResourceLevel,
    ILayer, IModel, EvolutionContext
)


class SnapshotMetric(BaseMetric):
    """
    Metric for tracking snapshot system performance.
    
    Measures:
    - Snapshot compression ratios
    - Recovery time estimates
    - Storage efficiency
    - Snapshot coverage
    """
    
    def __init__(self, name: str = None):
        """Initialize snapshot metric."""
        super().__init__(name or "SnapshotMetric")
        
        # Track cumulative statistics
        self.total_snapshots = 0
        self.total_size_mb = 0.0
        self.delta_count = 0
        self.full_count = 0
        
        self._contract = ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={'snapshot_info'},
            optional_inputs={'model', 'snapshot_history'},
            provided_outputs={
                'compression_ratio',
                'storage_efficiency',
                'snapshot_coverage',
                'recovery_metrics'
            },
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
    
    def analyze(self, target: Optional[Union[ILayer, IModel]], 
                context: EvolutionContext) -> Dict[str, Any]:
        """
        Analyze snapshot metrics.
        
        Args:
            target: Model (optional)
            context: Context with snapshot information
            
        Returns:
            Snapshot metrics
        """
        snapshot_info = context.get('snapshot_info', {})
        
        # Basic metrics
        metrics = {
            'snapshot_count': snapshot_info.get('total_saved', 0),
            'total_size_mb': snapshot_info.get('total_size_mb', 0),
            'delta_ratio': 0.0,
            'compression_ratio': 0.0,
            'storage_efficiency': 0.0
        }
        
        # Calculate delta ratio
        delta_count = snapshot_info.get('deltas_saved', 0)
        total_count = snapshot_info.get('total_saved', 0)
        if total_count > 0:
            metrics['delta_ratio'] = delta_count / total_count
        
        # Calculate compression ratio if model provided
        if target is not None and isinstance(target, IModel):
            model_size = self._calculate_model_size(target)
            if model_size > 0 and metrics['total_size_mb'] > 0:
                # Compression ratio: saved_size / (model_size * num_snapshots)
                expected_size = model_size * total_count
                metrics['compression_ratio'] = metrics['total_size_mb'] / expected_size
                metrics['storage_efficiency'] = 1.0 - metrics['compression_ratio']
        
        # Analyze snapshot history if provided
        history = context.get('snapshot_history', [])
        if history:
            coverage_metrics = self._analyze_coverage(history, context)
            metrics.update(coverage_metrics)
        
        # Recovery metrics
        recovery_metrics = self._calculate_recovery_metrics(snapshot_info, history)
        metrics['recovery_metrics'] = recovery_metrics
        
        # Update tracking
        self.total_snapshots = total_count
        self.total_size_mb = metrics['total_size_mb']
        self.delta_count = delta_count
        self.full_count = total_count - delta_count
        
        return metrics
    
    def get_measurement_schema(self) -> Dict[str, type]:
        """Get schema of measurements."""
        return {
            'snapshot_count': int,
            'total_size_mb': float,
            'delta_ratio': float,
            'compression_ratio': float,
            'storage_efficiency': float,
            'epoch_coverage': float,
            'performance_coverage': float,
            'growth_coverage': float,
            'recovery_metrics': dict
        }
    
    def _calculate_model_size(self, model: IModel) -> float:
        """Calculate model size in MB."""
        total_size = 0
        
        for param in model.parameters():
            # Each parameter is 4 bytes (float32)
            total_size += param.numel() * 4
        
        # Convert to MB
        return total_size / (1024 * 1024)
    
    def _analyze_coverage(self, history: list, context: EvolutionContext) -> Dict[str, Any]:
        """Analyze how well snapshots cover the training process."""
        if not history:
            return {
                'epoch_coverage': 0.0,
                'performance_coverage': 0.0,
                'growth_coverage': 0.0
            }
        
        current_epoch = context.epoch
        
        # Epoch coverage: how well distributed across training
        epochs = [s['epoch'] for s in history]
        if current_epoch > 0:
            # Calculate coverage as ratio of unique epochs with snapshots
            epoch_coverage = len(set(epochs)) / current_epoch
        else:
            epoch_coverage = 0.0
        
        # Performance coverage: snapshots at key performance points
        perf_snapshots = sum(1 for s in history 
                           if 'performance' in s.get('reason', ''))
        perf_coverage = perf_snapshots / max(len(history), 1)
        
        # Growth coverage: snapshots at growth events
        growth_snapshots = sum(1 for s in history 
                             if 'growth' in s.get('reason', ''))
        growth_coverage = growth_snapshots / max(len(history), 1)
        
        return {
            'epoch_coverage': epoch_coverage,
            'performance_coverage': perf_coverage,
            'growth_coverage': growth_coverage,
            'snapshot_distribution': self._analyze_distribution(epochs, current_epoch)
        }
    
    def _analyze_distribution(self, epochs: list, current_epoch: int) -> Dict[str, float]:
        """Analyze distribution of snapshots across training."""
        if not epochs or current_epoch == 0:
            return {'uniformity': 0.0, 'recency_bias': 0.0}
        
        # Sort epochs
        sorted_epochs = sorted(epochs)
        
        # Calculate uniformity (0 = clustered, 1 = uniform)
        if len(sorted_epochs) > 1:
            intervals = [sorted_epochs[i] - sorted_epochs[i-1] 
                        for i in range(1, len(sorted_epochs))]
            mean_interval = sum(intervals) / len(intervals)
            
            if mean_interval > 0:
                variance = sum((i - mean_interval)**2 for i in intervals) / len(intervals)
                uniformity = 1.0 / (1.0 + variance / mean_interval**2)
            else:
                uniformity = 0.0
        else:
            uniformity = 0.0
        
        # Calculate recency bias (higher = more recent snapshots)
        recent_threshold = current_epoch * 0.8  # Last 20% of training
        recent_count = sum(1 for e in epochs if e >= recent_threshold)
        recency_bias = recent_count / len(epochs) if epochs else 0.0
        
        return {
            'uniformity': uniformity,
            'recency_bias': recency_bias
        }
    
    def _calculate_recovery_metrics(self, snapshot_info: Dict[str, Any],
                                  history: list) -> Dict[str, Any]:
        """Calculate metrics related to recovery capabilities."""
        metrics = {
            'average_recovery_size_mb': 0.0,
            'max_recovery_gap_epochs': 0,
            'recovery_time_estimate_seconds': 0.0,
            'recovery_confidence': 0.0
        }
        
        if not history:
            return metrics
        
        # Average size for recovery
        sizes = [s.get('file_size_mb', 0) for s in history]
        if sizes:
            metrics['average_recovery_size_mb'] = sum(sizes) / len(sizes)
        
        # Maximum gap between snapshots
        epochs = sorted([s['epoch'] for s in history])
        if len(epochs) > 1:
            gaps = [epochs[i] - epochs[i-1] for i in range(1, len(epochs))]
            metrics['max_recovery_gap_epochs'] = max(gaps)
        
        # Estimate recovery time (based on size and type)
        delta_ratio = snapshot_info.get('deltas_saved', 0) / max(snapshot_info.get('total_saved', 1), 1)
        avg_size = metrics['average_recovery_size_mb']
        
        # Assume 100 MB/s read speed, add overhead for deltas
        base_time = avg_size / 100.0
        delta_overhead = 1.5 if delta_ratio > 0.5 else 1.0
        metrics['recovery_time_estimate_seconds'] = base_time * delta_overhead
        
        # Recovery confidence (based on coverage and recency)
        if 'epoch_coverage' in metrics:
            coverage_score = (
                metrics.get('epoch_coverage', 0) * 0.3 +
                metrics.get('performance_coverage', 0) * 0.4 +
                metrics.get('growth_coverage', 0) * 0.3
            )
            metrics['recovery_confidence'] = min(coverage_score * 1.2, 1.0)
        
        return metrics
    
    def get_efficiency_report(self) -> Dict[str, Any]:
        """Get a summary report of snapshot efficiency."""
        if self.total_snapshots == 0:
            return {'status': 'no_snapshots'}
        
        return {
            'total_snapshots': self.total_snapshots,
            'total_size_mb': self.total_size_mb,
            'average_size_mb': self.total_size_mb / self.total_snapshots,
            'delta_usage': {
                'delta_count': self.delta_count,
                'full_count': self.full_count,
                'delta_percentage': (self.delta_count / self.total_snapshots) * 100
            },
            'recommendations': self._generate_efficiency_recommendations()
        }
    
    def _generate_efficiency_recommendations(self) -> list:
        """Generate recommendations for improving efficiency."""
        recommendations = []
        
        if self.total_snapshots > 0:
            avg_size = self.total_size_mb / self.total_snapshots
            delta_ratio = self.delta_count / self.total_snapshots
            
            if avg_size > 100:
                recommendations.append(
                    "Large average snapshot size. Consider enabling compression "
                    "or increasing delta snapshot usage."
                )
            
            if delta_ratio < 0.5:
                recommendations.append(
                    "Low delta snapshot usage. Enable delta-based storage "
                    "to reduce storage requirements."
                )
            
            if self.total_size_mb > 10000:  # 10 GB
                recommendations.append(
                    "High total storage usage. Consider pruning old snapshots "
                    "or reducing snapshot frequency."
                )
        
        return recommendations