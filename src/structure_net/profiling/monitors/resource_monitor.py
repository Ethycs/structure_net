#!/usr/bin/env python3
"""
Resource Monitor

Monitors system resources (CPU, memory, GPU) and provides resource
utilization tracking and optimization recommendations.
"""

import time
import threading
import psutil
import torch
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import statistics


@dataclass
class ResourceSnapshot:
    """Snapshot of system resource usage."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    process_count: int = 0
    thread_count: int = 0


@dataclass
class ResourceAlert:
    """Resource usage alert."""
    resource_type: str
    severity: str  # 'warning', 'critical'
    message: str
    timestamp: float
    current_value: float
    threshold: float
    recommendations: List[str] = field(default_factory=list)


class ResourceMonitor:
    """
    System resource monitoring for the profiling system.
    
    Tracks CPU, memory, disk, and GPU usage, providing alerts
    when resources are constrained.
    """
    
    def __init__(self, 
                 monitoring_interval: float = 5.0,
                 history_size: int = 720):  # 1 hour at 5s intervals
        """
        Initialize the resource monitor.
        
        Args:
            monitoring_interval: How often to collect metrics (seconds)
            history_size: Number of snapshots to keep in history
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        
        # Resource history
        self.resource_history: deque = deque(maxlen=history_size)
        
        # Alert system
        self.alerts: List[ResourceAlert] = []
        self.alert_callbacks: List[callable] = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Resource thresholds
        self.thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
            'gpu_memory_warning': 80.0,
            'gpu_memory_critical': 95.0,
            'gpu_utilization_warning': 90.0,
            'gpu_utilization_critical': 98.0
        }
        
        # GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_monitoring = True
            except:
                self.gpu_monitoring = False
        else:
            self.gpu_monitoring = False
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="ResourceMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        print("🔍 Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        print("🔍 Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Main resource monitoring loop."""
        while self.is_monitoring:
            try:
                snapshot = self._take_resource_snapshot()
                
                with self.lock:
                    self.resource_history.append(snapshot)
                
                # Check for resource issues
                self._check_resource_thresholds(snapshot)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Error in resource monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _take_resource_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current resource usage."""
        timestamp = time.time()
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Process information
        process_count = len(psutil.pids())
        current_process = psutil.Process()
        thread_count = current_process.num_threads()
        
        snapshot = ResourceSnapshot(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            memory_available_mb=memory.available / 1024 / 1024,
            disk_usage_percent=disk.percent,
            process_count=process_count,
            thread_count=thread_count
        )
        
        # GPU metrics if available
        if self.gpu_monitoring:
            try:
                import pynvml
                
                # GPU memory
                gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                snapshot.gpu_memory_used_mb = gpu_memory.used / 1024 / 1024
                snapshot.gpu_memory_total_mb = gpu_memory.total / 1024 / 1024
                
                # GPU utilization
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                snapshot.gpu_utilization_percent = gpu_util.gpu
                
            except Exception as e:
                # GPU monitoring failed, disable it
                self.gpu_monitoring = False
        
        return snapshot
    
    def _check_resource_thresholds(self, snapshot: ResourceSnapshot):
        """Check resource usage against thresholds and generate alerts."""
        alerts = []
        
        # CPU alerts
        if snapshot.cpu_percent > self.thresholds['cpu_critical']:
            alerts.append(ResourceAlert(
                resource_type='cpu',
                severity='critical',
                message=f"Critical CPU usage: {snapshot.cpu_percent:.1f}%",
                timestamp=snapshot.timestamp,
                current_value=snapshot.cpu_percent,
                threshold=self.thresholds['cpu_critical'],
                recommendations=[
                    'Reduce profiling level',
                    'Check for runaway processes',
                    'Consider reducing workload'
                ]
            ))
        elif snapshot.cpu_percent > self.thresholds['cpu_warning']:
            alerts.append(ResourceAlert(
                resource_type='cpu',
                severity='warning',
                message=f"High CPU usage: {snapshot.cpu_percent:.1f}%",
                timestamp=snapshot.timestamp,
                current_value=snapshot.cpu_percent,
                threshold=self.thresholds['cpu_warning'],
                recommendations=[
                    'Monitor CPU-intensive operations',
                    'Consider optimizing algorithms'
                ]
            ))
        
        # Memory alerts
        if snapshot.memory_percent > self.thresholds['memory_critical']:
            alerts.append(ResourceAlert(
                resource_type='memory',
                severity='critical',
                message=f"Critical memory usage: {snapshot.memory_percent:.1f}%",
                timestamp=snapshot.timestamp,
                current_value=snapshot.memory_percent,
                threshold=self.thresholds['memory_critical'],
                recommendations=[
                    'Free up memory immediately',
                    'Reduce profiling detail',
                    'Check for memory leaks',
                    'Consider restarting the application'
                ]
            ))
        elif snapshot.memory_percent > self.thresholds['memory_warning']:
            alerts.append(ResourceAlert(
                resource_type='memory',
                severity='warning',
                message=f"High memory usage: {snapshot.memory_percent:.1f}%",
                timestamp=snapshot.timestamp,
                current_value=snapshot.memory_percent,
                threshold=self.thresholds['memory_warning'],
                recommendations=[
                    'Monitor memory usage trends',
                    'Consider garbage collection'
                ]
            ))
        
        # Disk alerts
        if snapshot.disk_usage_percent > self.thresholds['disk_critical']:
            alerts.append(ResourceAlert(
                resource_type='disk',
                severity='critical',
                message=f"Critical disk usage: {snapshot.disk_usage_percent:.1f}%",
                timestamp=snapshot.timestamp,
                current_value=snapshot.disk_usage_percent,
                threshold=self.thresholds['disk_critical'],
                recommendations=[
                    'Free up disk space immediately',
                    'Clean up profiling output files',
                    'Archive old data'
                ]
            ))
        elif snapshot.disk_usage_percent > self.thresholds['disk_warning']:
            alerts.append(ResourceAlert(
                resource_type='disk',
                severity='warning',
                message=f"High disk usage: {snapshot.disk_usage_percent:.1f}%",
                timestamp=snapshot.timestamp,
                current_value=snapshot.disk_usage_percent,
                threshold=self.thresholds['disk_warning'],
                recommendations=[
                    'Plan for disk cleanup',
                    'Monitor disk usage growth'
                ]
            ))
        
        # GPU alerts
        if self.gpu_monitoring and snapshot.gpu_memory_total_mb > 0:
            gpu_memory_percent = (snapshot.gpu_memory_used_mb / snapshot.gpu_memory_total_mb) * 100
            
            if gpu_memory_percent > self.thresholds['gpu_memory_critical']:
                alerts.append(ResourceAlert(
                    resource_type='gpu_memory',
                    severity='critical',
                    message=f"Critical GPU memory usage: {gpu_memory_percent:.1f}%",
                    timestamp=snapshot.timestamp,
                    current_value=gpu_memory_percent,
                    threshold=self.thresholds['gpu_memory_critical'],
                    recommendations=[
                        'Reduce batch sizes',
                        'Clear GPU memory',
                        'Reduce model complexity'
                    ]
                ))
            elif gpu_memory_percent > self.thresholds['gpu_memory_warning']:
                alerts.append(ResourceAlert(
                    resource_type='gpu_memory',
                    severity='warning',
                    message=f"High GPU memory usage: {gpu_memory_percent:.1f}%",
                    timestamp=snapshot.timestamp,
                    current_value=gpu_memory_percent,
                    threshold=self.thresholds['gpu_memory_warning'],
                    recommendations=[
                        'Monitor GPU memory usage',
                        'Consider optimizing GPU usage'
                    ]
                ))
        
        # Process alerts if needed
        for alert in alerts:
            self._process_alert(alert)
    
    def _process_alert(self, alert: ResourceAlert):
        """Process a resource alert."""
        with self.lock:
            self.alerts.append(alert)
            
            # Keep only recent alerts (last 4 hours)
            cutoff_time = time.time() - 14400
            self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Error in resource alert callback: {e}")
        
        # Print critical alerts
        if alert.severity == 'critical':
            print(f"🚨 CRITICAL RESOURCE ALERT: {alert.message}")
            for rec in alert.recommendations[:2]:
                print(f"   → {rec}")
    
    def get_current_usage(self) -> Optional[ResourceSnapshot]:
        """Get current resource usage."""
        with self.lock:
            return self.resource_history[-1] if self.resource_history else None
    
    def get_usage_trends(self, duration_minutes: int = 60) -> Dict[str, List[float]]:
        """
        Get resource usage trends over time.
        
        Args:
            duration_minutes: How far back to look
            
        Returns:
            Dictionary with trend data for each resource
        """
        cutoff_time = time.time() - (duration_minutes * 60)
        
        with self.lock:
            recent_snapshots = [
                s for s in self.resource_history
                if s.timestamp > cutoff_time
            ]
        
        if not recent_snapshots:
            return {}
        
        return {
            'cpu_percent': [s.cpu_percent for s in recent_snapshots],
            'memory_percent': [s.memory_percent for s in recent_snapshots],
            'disk_percent': [s.disk_usage_percent for s in recent_snapshots],
            'gpu_memory_percent': [
                (s.gpu_memory_used_mb / max(s.gpu_memory_total_mb, 1)) * 100
                for s in recent_snapshots
            ] if self.gpu_monitoring else [],
            'gpu_utilization': [s.gpu_utilization_percent for s in recent_snapshots] if self.gpu_monitoring else []
        }
    
    def get_usage_statistics(self, duration_minutes: int = 60) -> Dict[str, Dict[str, float]]:
        """
        Get statistical summary of resource usage.
        
        Args:
            duration_minutes: How far back to look
            
        Returns:
            Statistics for each resource type
        """
        trends = self.get_usage_trends(duration_minutes)
        stats = {}
        
        for resource, values in trends.items():
            if values:
                stats[resource] = {
                    'current': values[-1],
                    'average': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0
                }
        
        return stats
    
    def get_resource_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current resource usage."""
        current = self.get_current_usage()
        if not current:
            return []
        
        recommendations = []
        
        # CPU recommendations
        if current.cpu_percent > 70:
            recommendations.append("Consider reducing profiling level to decrease CPU usage")
            recommendations.append("Profile only critical components during high CPU periods")
        
        # Memory recommendations
        if current.memory_percent > 70:
            recommendations.append("Enable garbage collection more frequently")
            recommendations.append("Reduce profiling history size to save memory")
            recommendations.append("Consider batch processing to reduce memory pressure")
        
        # GPU recommendations
        if self.gpu_monitoring and current.gpu_memory_total_mb > 0:
            gpu_usage_percent = (current.gpu_memory_used_mb / current.gpu_memory_total_mb) * 100
            if gpu_usage_percent > 70:
                recommendations.append("Clear GPU cache regularly")
                recommendations.append("Reduce batch sizes for GPU operations")
        
        # General recommendations
        if current.process_count > 500:
            recommendations.append("High process count detected - check for process leaks")
        
        if current.thread_count > 100:
            recommendations.append("High thread count detected - review threading usage")
        
        return recommendations
    
    def get_recent_alerts(self, 
                         severity: Optional[str] = None,
                         resource_type: Optional[str] = None,
                         duration_hours: int = 1) -> List[ResourceAlert]:
        """
        Get recent resource alerts.
        
        Args:
            severity: Filter by severity ('warning', 'critical')
            resource_type: Filter by resource type
            duration_hours: How far back to look
            
        Returns:
            List of matching alerts
        """
        cutoff_time = time.time() - (duration_hours * 3600)
        
        with self.lock:
            alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if resource_type:
            alerts = [a for a in alerts if a.resource_type == resource_type]
        
        return alerts
    
    def add_alert_callback(self, callback: callable):
        """Add a callback function for resource alerts."""
        self.alert_callbacks.append(callback)
    
    def get_resource_report(self) -> str:
        """Generate a comprehensive resource usage report."""
        current = self.get_current_usage()
        if not current:
            return "No resource data available."
        
        stats = self.get_usage_statistics(60)  # Last hour
        recent_alerts = self.get_recent_alerts(duration_hours=4)
        recommendations = self.get_resource_recommendations()
        
        lines = [
            "📊 RESOURCE USAGE REPORT",
            "=" * 50,
            f"Timestamp: {time.ctime(current.timestamp)}",
            "",
            "📈 Current Usage:",
            f"  CPU: {current.cpu_percent:.1f}%",
            f"  Memory: {current.memory_percent:.1f}% ({current.memory_used_mb:.0f}MB used)",
            f"  Disk: {current.disk_usage_percent:.1f}%",
        ]
        
        if self.gpu_monitoring and current.gpu_memory_total_mb > 0:
            gpu_percent = (current.gpu_memory_used_mb / current.gpu_memory_total_mb) * 100
            lines.extend([
                f"  GPU Memory: {gpu_percent:.1f}% ({current.gpu_memory_used_mb:.0f}MB used)",
                f"  GPU Utilization: {current.gpu_utilization_percent:.1f}%"
            ])
        
        lines.extend([
            f"  Processes: {current.process_count}",
            f"  Threads: {current.thread_count}",
        ])
        
        # Statistics
        if stats:
            lines.extend([
                "",
                "📊 Hour Average:",
            ])
            for resource, stat in stats.items():
                lines.append(f"  {resource}: {stat['average']:.1f}% (max: {stat['max']:.1f}%)")
        
        # Recent alerts
        if recent_alerts:
            lines.extend([
                "",
                f"⚠️  Recent Alerts ({len(recent_alerts)}):",
            ])
            for alert in recent_alerts[-10:]:
                lines.append(f"  [{alert.severity.upper()}] {alert.resource_type}: {alert.message}")
        
        # Recommendations
        if recommendations:
            lines.extend([
                "",
                "💡 Recommendations:",
            ])
            for rec in recommendations[:5]:
                lines.append(f"  • {rec}")
        
        return "\n".join(lines)
    
    def __del__(self):
        """Cleanup when monitor is destroyed."""
        self.stop_monitoring()