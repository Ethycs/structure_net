#!/usr/bin/env python3
"""
Health Monitor

Monitors the health of the profiling system and components,
providing health checks and diagnostics.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..kernel_profiler import KernelProfiler


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Represents a health check result."""
    check_name: str
    status: HealthStatus
    message: str
    timestamp: float
    component: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ComponentHealth:
    """Health information for a component."""
    component_name: str
    status: HealthStatus
    last_check: float
    checks: List[HealthCheck] = field(default_factory=list)
    uptime: float = 0.0
    error_count: int = 0
    success_rate: float = 1.0


class HealthMonitor:
    """
    Health monitoring system for the profiling infrastructure.
    
    Performs periodic health checks on the kernel profiler and components,
    tracking system health and providing diagnostics.
    """
    
    def __init__(self, 
                 kernel_profiler: KernelProfiler,
                 check_interval: float = 30.0):  # 30 second health checks
        """
        Initialize the health monitor.
        
        Args:
            kernel_profiler: The kernel profiler to monitor
            check_interval: How often to perform health checks (seconds)
        """
        self.kernel_profiler = kernel_profiler
        self.check_interval = check_interval
        
        # Health tracking
        self.system_health = HealthStatus.UNKNOWN
        self.component_health: Dict[str, ComponentHealth] = {}
        self.recent_checks: List[HealthCheck] = []
        self.health_history: List[Tuple[float, HealthStatus]] = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Health thresholds
        self.thresholds = {
            'memory_growth_rate': 10.0,  # MB per minute
            'error_rate_degraded': 0.01,  # 1%
            'error_rate_unhealthy': 0.05,  # 5%
            'error_rate_critical': 0.20,  # 20%
            'response_time_degraded': 0.5,  # seconds
            'response_time_unhealthy': 2.0,  # seconds
            'response_time_critical': 10.0,  # seconds
            'profiler_overhead_warning': 5.0,  # 5%
            'profiler_overhead_critical': 15.0,  # 15%
        }
        
        # Baseline measurements
        self.baseline_memory = 0.0
        self.baseline_timestamp = time.time()
        
        if hasattr(kernel_profiler, 'logger') and kernel_profiler.logger:
            kernel_profiler.logger.info("HealthMonitor initialized")
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self.is_monitoring:
            return
        
        # Set initial baseline
        health_metrics = self.kernel_profiler.get_kernel_health_metrics()
        self.baseline_memory = health_metrics.get('memory_usage_mb', 0.0)
        self.baseline_timestamp = time.time()
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="HealthMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        if hasattr(self.kernel_profiler, 'logger') and self.kernel_profiler.logger:
            self.kernel_profiler.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        if hasattr(self.kernel_profiler, 'logger') and self.kernel_profiler.logger:
            self.kernel_profiler.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main health monitoring loop."""
        while self.is_monitoring:
            try:
                self._perform_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                if hasattr(self.kernel_profiler, 'logger') and self.kernel_profiler.logger:
                    self.kernel_profiler.logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _perform_health_checks(self):
        """Perform comprehensive health checks."""
        timestamp = time.time()
        checks = []
        
        # System-level health checks
        checks.extend(self._check_system_health(timestamp))
        
        # Profiler-specific health checks
        checks.extend(self._check_profiler_health(timestamp))
        
        # Component health checks
        checks.extend(self._check_component_health(timestamp))
        
        # Update overall system health
        self._update_system_health(checks, timestamp)
        
        # Store recent checks
        with self.lock:
            self.recent_checks.extend(checks)
            # Keep only recent checks (last hour)
            cutoff_time = timestamp - 3600
            self.recent_checks = [c for c in self.recent_checks if c.timestamp > cutoff_time]
    
    def _check_system_health(self, timestamp: float) -> List[HealthCheck]:
        """Perform system-level health checks."""
        checks = []
        health_metrics = self.kernel_profiler.get_kernel_health_metrics()
        
        # Memory growth check
        current_memory = health_metrics.get('memory_usage_mb', 0.0)
        time_diff = timestamp - self.baseline_timestamp
        if time_diff > 60:  # At least 1 minute of data
            memory_growth_rate = (current_memory - self.baseline_memory) / (time_diff / 60)
            
            if memory_growth_rate > self.thresholds['memory_growth_rate']:
                checks.append(HealthCheck(
                    check_name='memory_growth',
                    status=HealthStatus.DEGRADED,
                    message=f"High memory growth rate: {memory_growth_rate:.1f} MB/min",
                    timestamp=timestamp,
                    metrics={'growth_rate': memory_growth_rate, 'current_memory': current_memory},
                    recommendations=['Check for memory leaks', 'Review profiling overhead']
                ))
            else:
                checks.append(HealthCheck(
                    check_name='memory_growth',
                    status=HealthStatus.HEALTHY,
                    message=f"Normal memory growth rate: {memory_growth_rate:.1f} MB/min",
                    timestamp=timestamp,
                    metrics={'growth_rate': memory_growth_rate}
                ))
        
        # CPU usage check
        cpu_usage = health_metrics.get('cpu_usage_percent', 0.0)
        if cpu_usage > 90:
            checks.append(HealthCheck(
                check_name='cpu_usage',
                status=HealthStatus.UNHEALTHY,
                message=f"High CPU usage: {cpu_usage:.1f}%",
                timestamp=timestamp,
                metrics={'cpu_usage': cpu_usage},
                recommendations=['Reduce profiling level', 'Check for runaway processes']
            ))
        elif cpu_usage > 70:
            checks.append(HealthCheck(
                check_name='cpu_usage',
                status=HealthStatus.DEGRADED,
                message=f"Elevated CPU usage: {cpu_usage:.1f}%",
                timestamp=timestamp,
                metrics={'cpu_usage': cpu_usage}
            ))
        else:
            checks.append(HealthCheck(
                check_name='cpu_usage',
                status=HealthStatus.HEALTHY,
                message=f"Normal CPU usage: {cpu_usage:.1f}%",
                timestamp=timestamp,
                metrics={'cpu_usage': cpu_usage}
            ))
        
        return checks
    
    def _check_profiler_health(self, timestamp: float) -> List[HealthCheck]:
        """Perform profiler-specific health checks."""
        checks = []
        health_metrics = self.kernel_profiler.get_kernel_health_metrics()
        
        # Profiler overhead check
        overhead_percent = health_metrics.get('profiler_overhead_percent', 0.0)
        if overhead_percent > self.thresholds['profiler_overhead_critical']:
            checks.append(HealthCheck(
                check_name='profiler_overhead',
                status=HealthStatus.CRITICAL,
                message=f"Critical profiler overhead: {overhead_percent:.2f}%",
                timestamp=timestamp,
                metrics={'overhead_percent': overhead_percent},
                recommendations=['Reduce profiling level', 'Enable adaptive sampling']
            ))
        elif overhead_percent > self.thresholds['profiler_overhead_warning']:
            checks.append(HealthCheck(
                check_name='profiler_overhead',
                status=HealthStatus.DEGRADED,
                message=f"High profiler overhead: {overhead_percent:.2f}%",
                timestamp=timestamp,
                metrics={'overhead_percent': overhead_percent},
                recommendations=['Consider reducing profiling detail']
            ))
        else:
            checks.append(HealthCheck(
                check_name='profiler_overhead',
                status=HealthStatus.HEALTHY,
                message=f"Normal profiler overhead: {overhead_percent:.2f}%",
                timestamp=timestamp,
                metrics={'overhead_percent': overhead_percent}
            ))
        
        # Active profiles check
        active_profiles = health_metrics.get('active_profiles', 0)
        if active_profiles > 100:
            checks.append(HealthCheck(
                check_name='active_profiles',
                status=HealthStatus.DEGRADED,
                message=f"Many active profiles: {active_profiles}",
                timestamp=timestamp,
                metrics={'active_profiles': active_profiles},
                recommendations=['Check for profile leaks', 'Verify proper profile cleanup']
            ))
        
        # Profiler enabled check
        if not health_metrics.get('profiler_enabled', True):
            checks.append(HealthCheck(
                check_name='profiler_enabled',
                status=HealthStatus.UNHEALTHY,
                message="Profiler is disabled",
                timestamp=timestamp,
                recommendations=['Re-enable profiler if monitoring is needed']
            ))
        
        return checks
    
    def _check_component_health(self, timestamp: float) -> List[HealthCheck]:
        """Perform component-specific health checks."""
        checks = []
        specialized_metrics = self.kernel_profiler.get_specialized_metrics()
        component_breakdown = specialized_metrics.get('component_breakdown', {})
        
        for component_name, component_data in component_breakdown.items():
            if not isinstance(component_data, dict):
                continue
            
            # Response time check
            avg_time = component_data.get('average_time', 0.0)
            if avg_time > self.thresholds['response_time_critical']:
                status = HealthStatus.CRITICAL
                message = f"Critical response time: {avg_time:.3f}s"
                recommendations = ['Optimize component performance', 'Check for bottlenecks']
            elif avg_time > self.thresholds['response_time_unhealthy']:
                status = HealthStatus.UNHEALTHY
                message = f"Poor response time: {avg_time:.3f}s"
                recommendations = ['Review component performance']
            elif avg_time > self.thresholds['response_time_degraded']:
                status = HealthStatus.DEGRADED
                message = f"Slow response time: {avg_time:.3f}s"
                recommendations = []
            else:
                status = HealthStatus.HEALTHY
                message = f"Good response time: {avg_time:.3f}s"
                recommendations = []
            
            checks.append(HealthCheck(
                check_name='response_time',
                status=status,
                message=message,
                timestamp=timestamp,
                component=component_name,
                metrics={'average_time': avg_time},
                recommendations=recommendations
            ))
            
            # Update component health tracking
            self._update_component_health(component_name, status, timestamp, checks[-1])
        
        return checks
    
    def _update_component_health(self, 
                                component_name: str, 
                                status: HealthStatus, 
                                timestamp: float, 
                                check: HealthCheck):
        """Update health tracking for a specific component."""
        with self.lock:
            if component_name not in self.component_health:
                self.component_health[component_name] = ComponentHealth(
                    component_name=component_name,
                    status=status,
                    last_check=timestamp
                )
            
            component = self.component_health[component_name]
            component.status = status
            component.last_check = timestamp
            component.checks.append(check)
            
            # Keep only recent checks
            cutoff_time = timestamp - 3600
            component.checks = [c for c in component.checks if c.timestamp > cutoff_time]
    
    def _update_system_health(self, checks: List[HealthCheck], timestamp: float):
        """Update overall system health based on checks."""
        if not checks:
            return
        
        # Determine overall health from individual checks
        statuses = [check.status for check in checks]
        
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        with self.lock:
            self.system_health = overall_status
            self.health_history.append((timestamp, overall_status))
            
            # Keep only recent history (last 24 hours)
            cutoff_time = timestamp - 86400
            self.health_history = [(t, s) for t, s in self.health_history if t > cutoff_time]
    
    def get_system_health(self) -> HealthStatus:
        """Get current overall system health."""
        with self.lock:
            return self.system_health
    
    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get health information for a specific component."""
        with self.lock:
            return self.component_health.get(component_name)
    
    def get_all_component_health(self) -> Dict[str, ComponentHealth]:
        """Get health information for all components."""
        with self.lock:
            return self.component_health.copy()
    
    def get_recent_checks(self, 
                         status: Optional[HealthStatus] = None,
                         component: Optional[str] = None,
                         duration_seconds: int = 3600) -> List[HealthCheck]:
        """
        Get recent health checks.
        
        Args:
            status: Filter by health status
            component: Filter by component name
            duration_seconds: How far back to look
            
        Returns:
            List of matching health checks
        """
        cutoff_time = time.time() - duration_seconds
        
        with self.lock:
            checks = [c for c in self.recent_checks if c.timestamp > cutoff_time]
        
        if status:
            checks = [c for c in checks if c.status == status]
        
        if component:
            checks = [c for c in checks if c.component == component]
        
        return checks
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a comprehensive health summary."""
        with self.lock:
            system_health = self.system_health
            component_health = self.component_health.copy()
            recent_checks = self.recent_checks[-10:]  # Last 10 checks
        
        # Count checks by status
        status_counts = {}
        for status in HealthStatus:
            status_counts[status.value] = len([
                c for c in recent_checks if c.status == status
            ])
        
        return {
            'system_health': system_health.value,
            'component_count': len(component_health),
            'components_healthy': len([c for c in component_health.values() if c.status == HealthStatus.HEALTHY]),
            'components_degraded': len([c for c in component_health.values() if c.status == HealthStatus.DEGRADED]),
            'components_unhealthy': len([c for c in component_health.values() if c.status == HealthStatus.UNHEALTHY]),
            'components_critical': len([c for c in component_health.values() if c.status == HealthStatus.CRITICAL]),
            'recent_check_counts': status_counts,
            'last_check_time': recent_checks[-1].timestamp if recent_checks else None
        }
    
    def get_health_report(self) -> str:
        """Generate a comprehensive health report."""
        summary = self.get_health_summary()
        recent_issues = self.get_recent_checks(duration_seconds=3600)
        recent_issues = [c for c in recent_issues if c.status != HealthStatus.HEALTHY]
        
        lines = [
            "🏥 SYSTEM HEALTH REPORT",
            "=" * 50,
            f"Overall Health: {summary['system_health'].upper()}",
            f"Components Monitored: {summary['component_count']}",
            "",
            "📊 Component Health:",
            f"  ✅ Healthy: {summary['components_healthy']}",
            f"  ⚠️  Degraded: {summary['components_degraded']}",
            f"  🔴 Unhealthy: {summary['components_unhealthy']}",
            f"  🚨 Critical: {summary['components_critical']}",
        ]
        
        if recent_issues:
            lines.extend([
                "",
                f"⚠️  Recent Issues ({len(recent_issues)}):",
            ])
            
            for issue in recent_issues[-10:]:  # Last 10 issues
                component_str = f" [{issue.component}]" if issue.component else ""
                lines.append(f"  [{issue.status.value.upper()}]{component_str} {issue.message}")
                
                if issue.recommendations:
                    for rec in issue.recommendations[:2]:  # First 2 recommendations
                        lines.append(f"    → {rec}")
        
        if summary['last_check_time']:
            lines.extend([
                "",
                f"Last Check: {time.ctime(summary['last_check_time'])}"
            ])
        
        return "\n".join(lines)
    
    def __del__(self):
        """Cleanup when monitor is destroyed."""
        self.stop_monitoring()