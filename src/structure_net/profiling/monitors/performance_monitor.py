#!/usr/bin/env python3
"""
Performance Monitor

Real-time performance monitoring for the StructureNet profiling system.
Provides continuous monitoring of system performance and component health.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import statistics

from ..kernel_profiler import KernelProfiler


@dataclass
class PerformanceAlert:
    """Represents a performance alert."""
    alert_type: str
    component: str
    message: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: float
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    operation_rate: float  # operations per second
    error_rate: float
    component_metrics: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Real-time performance monitor for the profiling system.
    
    Continuously monitors system performance and component health,
    providing alerts when performance degrades.
    """
    
    def __init__(self, 
                 kernel_profiler: KernelProfiler,
                 monitoring_interval: float = 1.0,
                 history_size: int = 300):  # 5 minutes at 1s intervals
        """
        Initialize the performance monitor.
        
        Args:
            kernel_profiler: The kernel profiler to monitor
            monitoring_interval: How often to collect metrics (seconds)
            history_size: Number of snapshots to keep in history
        """
        self.kernel_profiler = kernel_profiler
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        
        # Performance history
        self.performance_history: deque = deque(maxlen=history_size)
        self.component_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        
        # Alert system
        self.alerts: List[PerformanceAlert] = []
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Performance thresholds
        self.thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'error_rate_warning': 0.05,  # 5%
            'error_rate_critical': 0.20,  # 20%
            'operation_rate_drop': 0.5,  # 50% drop from baseline
            'response_time_warning': 1.0,  # seconds
            'response_time_critical': 5.0  # seconds
        }
        
        # Baseline performance (calculated from history)
        self.baseline_metrics: Dict[str, float] = {}
        
        if hasattr(kernel_profiler, 'logger') and kernel_profiler.logger:
            kernel_profiler.logger.info("PerformanceMonitor initialized")
    
    def start_monitoring(self):
        """Start real-time performance monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="PerformanceMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        if hasattr(self.kernel_profiler, 'logger') and self.kernel_profiler.logger:
            self.kernel_profiler.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time performance monitoring."""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        if hasattr(self.kernel_profiler, 'logger') and self.kernel_profiler.logger:
            self.kernel_profiler.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                snapshot = self._take_performance_snapshot()
                
                with self.lock:
                    self.performance_history.append(snapshot)
                
                # Check for performance issues
                self._analyze_performance(snapshot)
                
                # Update baselines periodically
                if len(self.performance_history) % 60 == 0:  # Every minute
                    self._update_baseline_metrics()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                if hasattr(self.kernel_profiler, 'logger') and self.kernel_profiler.logger:
                    self.kernel_profiler.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _take_performance_snapshot(self) -> PerformanceSnapshot:
        """Take a snapshot of current system performance."""
        timestamp = time.time()
        
        # Get kernel health metrics
        health_metrics = self.kernel_profiler.get_kernel_health_metrics()
        
        # Calculate operation rate (operations per second)
        operation_rate = 0.0
        if len(self.performance_history) > 0:
            prev_snapshot = self.performance_history[-1]
            time_diff = timestamp - prev_snapshot.timestamp
            if time_diff > 0:
                prev_ops = self._get_total_operations_from_snapshot(prev_snapshot)
                current_ops = health_metrics.get('total_profiles', 0)
                operation_rate = (current_ops - prev_ops) / time_diff
        
        # Calculate error rate
        error_rate = 0.0
        total_ops = health_metrics.get('total_profiles', 0)
        if total_ops > 0:
            # This would need to be tracked in the kernel profiler
            # For now, we'll estimate based on component metrics
            error_rate = 0.0  # Placeholder
        
        # Get component-specific metrics
        specialized_metrics = self.kernel_profiler.get_specialized_metrics()
        component_metrics = specialized_metrics.get('component_breakdown', {})
        
        snapshot = PerformanceSnapshot(
            timestamp=timestamp,
            cpu_usage=health_metrics.get('cpu_usage_percent', 0.0),
            memory_usage=health_metrics.get('memory_usage_mb', 0.0),
            gpu_usage=health_metrics.get('gpu_memory_mb', 0.0),
            operation_rate=operation_rate,
            error_rate=error_rate,
            component_metrics=component_metrics
        )
        
        return snapshot
    
    def _analyze_performance(self, snapshot: PerformanceSnapshot):
        """Analyze performance snapshot and generate alerts if needed."""
        alerts = []
        
        # CPU usage alerts
        if snapshot.cpu_usage > self.thresholds['cpu_critical']:
            alerts.append(PerformanceAlert(
                alert_type='cpu_usage',
                component='system',
                message=f"Critical CPU usage: {snapshot.cpu_usage:.1f}%",
                severity='critical',
                timestamp=snapshot.timestamp,
                metrics={'cpu_usage': snapshot.cpu_usage}
            ))
        elif snapshot.cpu_usage > self.thresholds['cpu_warning']:
            alerts.append(PerformanceAlert(
                alert_type='cpu_usage',
                component='system',
                message=f"High CPU usage: {snapshot.cpu_usage:.1f}%",
                severity='medium',
                timestamp=snapshot.timestamp,
                metrics={'cpu_usage': snapshot.cpu_usage}
            ))
        
        # Memory usage alerts (if we have baseline)
        if 'memory_usage' in self.baseline_metrics:
            baseline_memory = self.baseline_metrics['memory_usage']
            memory_increase = snapshot.memory_usage - baseline_memory
            
            if memory_increase > baseline_memory * 2:  # Memory doubled
                alerts.append(PerformanceAlert(
                    alert_type='memory_growth',
                    component='system',
                    message=f"Significant memory growth: {memory_increase:.1f}MB increase",
                    severity='high',
                    timestamp=snapshot.timestamp,
                    metrics={
                        'current_memory': snapshot.memory_usage,
                        'baseline_memory': baseline_memory,
                        'increase': memory_increase
                    }
                ))
        
        # Operation rate alerts
        if 'operation_rate' in self.baseline_metrics and self.baseline_metrics['operation_rate'] > 0:
            baseline_rate = self.baseline_metrics['operation_rate']
            rate_ratio = snapshot.operation_rate / baseline_rate
            
            if rate_ratio < self.thresholds['operation_rate_drop']:
                alerts.append(PerformanceAlert(
                    alert_type='performance_drop',
                    component='system',
                    message=f"Operation rate dropped to {rate_ratio:.1%} of baseline",
                    severity='medium',
                    timestamp=snapshot.timestamp,
                    metrics={
                        'current_rate': snapshot.operation_rate,
                        'baseline_rate': baseline_rate,
                        'ratio': rate_ratio
                    }
                ))
        
        # Component-specific alerts
        for component_name, component_data in snapshot.component_metrics.items():
            if isinstance(component_data, dict):
                avg_time = component_data.get('average_time', 0)
                
                if avg_time > self.thresholds['response_time_critical']:
                    alerts.append(PerformanceAlert(
                        alert_type='slow_component',
                        component=component_name,
                        message=f"Critical response time: {avg_time:.3f}s",
                        severity='critical',
                        timestamp=snapshot.timestamp,
                        metrics={'average_time': avg_time}
                    ))
                elif avg_time > self.thresholds['response_time_warning']:
                    alerts.append(PerformanceAlert(
                        alert_type='slow_component',
                        component=component_name,
                        message=f"Slow response time: {avg_time:.3f}s",
                        severity='medium',
                        timestamp=snapshot.timestamp,
                        metrics={'average_time': avg_time}
                    ))
        
        # Process alerts
        for alert in alerts:
            self._process_alert(alert)
    
    def _process_alert(self, alert: PerformanceAlert):
        """Process a performance alert."""
        with self.lock:
            self.alerts.append(alert)
            
            # Keep only recent alerts (last hour)
            cutoff_time = time.time() - 3600
            self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        # Log the alert
        if hasattr(self.kernel_profiler, 'logger') and self.kernel_profiler.logger:
            self.kernel_profiler.logger.warning(
                f"Performance alert: {alert.message}",
                alert_type=alert.alert_type,
                component=alert.component,
                severity=alert.severity,
                metrics=alert.metrics
            )
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                if hasattr(self.kernel_profiler, 'logger') and self.kernel_profiler.logger:
                    self.kernel_profiler.logger.error(f"Error in alert callback: {e}")
    
    def _update_baseline_metrics(self):
        """Update baseline performance metrics from recent history."""
        if len(self.performance_history) < 10:
            return
        
        recent_snapshots = list(self.performance_history)[-60:]  # Last minute
        
        self.baseline_metrics = {
            'cpu_usage': statistics.mean(s.cpu_usage for s in recent_snapshots),
            'memory_usage': statistics.mean(s.memory_usage for s in recent_snapshots),
            'operation_rate': statistics.mean(s.operation_rate for s in recent_snapshots),
            'error_rate': statistics.mean(s.error_rate for s in recent_snapshots)
        }
    
    def _get_total_operations_from_snapshot(self, snapshot: PerformanceSnapshot) -> int:
        """Extract total operations count from a snapshot."""
        # This would need to be stored in the snapshot
        # For now, return 0 as placeholder
        return 0
    
    def get_current_performance(self) -> Optional[PerformanceSnapshot]:
        """Get the most recent performance snapshot."""
        with self.lock:
            return self.performance_history[-1] if self.performance_history else None
    
    def get_performance_trend(self, 
                             metric: str, 
                             duration_seconds: int = 300) -> List[float]:
        """
        Get performance trend for a specific metric.
        
        Args:
            metric: Metric name ('cpu_usage', 'memory_usage', etc.)
            duration_seconds: How far back to look
            
        Returns:
            List of metric values over time
        """
        cutoff_time = time.time() - duration_seconds
        
        with self.lock:
            recent_snapshots = [
                s for s in self.performance_history 
                if s.timestamp > cutoff_time
            ]
        
        return [getattr(s, metric, 0) for s in recent_snapshots]
    
    def get_recent_alerts(self, 
                         severity: Optional[str] = None,
                         component: Optional[str] = None,
                         duration_seconds: int = 3600) -> List[PerformanceAlert]:
        """
        Get recent performance alerts.
        
        Args:
            severity: Filter by severity level
            component: Filter by component name
            duration_seconds: How far back to look
            
        Returns:
            List of matching alerts
        """
        cutoff_time = time.time() - duration_seconds
        
        with self.lock:
            alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if component:
            alerts = [a for a in alerts if a.component == component]
        
        return alerts
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add a callback function to be called when alerts are generated."""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Remove an alert callback function."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    def get_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        current = self.get_current_performance()
        if not current:
            return "No performance data available."
        
        recent_alerts = self.get_recent_alerts(duration_seconds=3600)
        
        lines = [
            "📊 PERFORMANCE MONITOR REPORT",
            "=" * 50,
            f"Timestamp: {time.ctime(current.timestamp)}",
            f"Monitoring Duration: {len(self.performance_history) * self.monitoring_interval:.0f}s",
            "",
            "📈 Current Metrics:",
            f"  CPU Usage: {current.cpu_usage:.1f}%",
            f"  Memory Usage: {current.memory_usage:.1f}MB",
            f"  GPU Memory: {current.gpu_usage:.1f}MB",
            f"  Operation Rate: {current.operation_rate:.1f} ops/sec",
            f"  Error Rate: {current.error_rate:.2%}",
        ]
        
        if self.baseline_metrics:
            lines.extend([
                "",
                "📊 Baseline Comparison:",
                f"  CPU vs Baseline: {current.cpu_usage - self.baseline_metrics.get('cpu_usage', 0):+.1f}%",
                f"  Memory vs Baseline: {current.memory_usage - self.baseline_metrics.get('memory_usage', 0):+.1f}MB",
                f"  Rate vs Baseline: {(current.operation_rate / max(self.baseline_metrics.get('operation_rate', 1), 0.1)):.1%}",
            ])
        
        if recent_alerts:
            lines.extend([
                "",
                f"⚠️  Recent Alerts ({len(recent_alerts)}):",
            ])
            
            for alert in recent_alerts[-10:]:  # Last 10 alerts
                lines.append(f"  [{alert.severity.upper()}] {alert.component}: {alert.message}")
        
        return "\n".join(lines)
    
    def __del__(self):
        """Cleanup when monitor is destroyed."""
        self.stop_monitoring()