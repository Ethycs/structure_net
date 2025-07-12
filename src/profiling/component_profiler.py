#!/usr/bin/env python3
"""
Component Profiler Implementation

Profiler wrapper for individual components that provides component-specific
profiling capabilities while integrating with the kernel profiler.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager
from dataclasses import dataclass

from .kernel_profiler import KernelProfiler
from .core.base_profiler import ProfilerLevel


@dataclass
class ComponentMetrics:
    """Metrics specific to a component."""
    component_name: str
    total_operations: int
    total_time: float
    average_time: float
    success_rate: float
    memory_usage: float
    error_count: int
    last_operation_time: Optional[float] = None


class ComponentProfiler:
    """
    Profiler wrapper for individual components.
    
    Provides a component-specific interface to the kernel profiler
    with automatic categorization and component-specific metrics.
    """
    
    def __init__(self, kernel_profiler: KernelProfiler, component_name: str):
        """
        Initialize component profiler.
        
        Args:
            kernel_profiler: The kernel profiler instance
            component_name: Name of this component
        """
        self.kernel_profiler = kernel_profiler
        self.component_name = component_name
        
        # Component-specific tracking
        self.operation_count = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.total_execution_time = 0.0
        self.last_operation_time = None
        
        # Method-specific tracking
        self.method_stats: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
        # Performance thresholds
        self.slow_operation_threshold = 1.0  # seconds
        self.memory_warning_threshold = 100.0  # MB
        
        if hasattr(kernel_profiler, 'logger') and kernel_profiler.logger:
            kernel_profiler.logger.debug(f"ComponentProfiler created for {component_name}")
    
    @contextmanager
    def profile_method(self, method_name: str, **kwargs):
        """
        Profile a component method.
        
        Args:
            method_name: Name of the method being profiled
            **kwargs: Additional metrics to track
            
        Usage:
            with self._profiler.profile_method("my_method") as profile_id:
                # method implementation
                pass
        """
        operation_name = f"{self.component_name}.{method_name}"
        
        with self.kernel_profiler.profile_component_operation(
            self.component_name, 
            method_name
        ) as ctx:
            
            start_time = time.perf_counter()
            
            try:
                # Track method-specific stats
                self._track_method_start(method_name)
                
                # Add initial metrics
                if kwargs:
                    for key, value in kwargs.items():
                        ctx.add_metric(key, value)
                
                yield ctx
                
                # Track successful completion
                execution_time = time.perf_counter() - start_time
                self._track_method_completion(method_name, execution_time, True)
                
                # Add execution metrics
                ctx.add_metric('method_execution_time', execution_time)
                ctx.add_metric('component_operation_count', self.operation_count)
                
            except Exception as e:
                # Track failed completion
                execution_time = time.perf_counter() - start_time
                self._track_method_completion(method_name, execution_time, False)
                
                # Add error metrics
                ctx.add_metric('method_execution_time', execution_time)
                ctx.add_metric('error_type', type(e).__name__)
                ctx.add_metric('error_message', str(e))
                
                raise
    
    @contextmanager
    def profile_operation(self, operation_name: str, tags: Optional[List[str]] = None):
        """
        Profile a general component operation.
        
        Args:
            operation_name: Name of the operation
            tags: Optional tags for categorization
            
        Usage:
            with profiler.profile_operation("data_processing"):
                # operation code
                pass
        """
        with self.kernel_profiler.profile_component_operation(
            self.component_name,
            operation_name,
            tags
        ) as ctx:
            
            start_time = time.perf_counter()
            
            try:
                self._track_operation_start()
                yield ctx
                
                execution_time = time.perf_counter() - start_time
                self._track_operation_completion(execution_time, True)
                
                ctx.add_metric('operation_execution_time', execution_time)
                
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                self._track_operation_completion(execution_time, False)
                
                ctx.add_metric('operation_execution_time', execution_time)
                ctx.add_metric('error_type', type(e).__name__)
                
                raise
    
    def profile_function_call(self, 
                             func: Callable, 
                             function_name: Optional[str] = None,
                             *args, **kwargs):
        """
        Profile a single function call.
        
        Args:
            func: Function to profile
            function_name: Optional custom name for the function
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Function result
        """
        fname = function_name or func.__name__
        
        with self.profile_method(fname) as ctx:
            ctx.add_metric('function_name', func.__name__)
            ctx.add_metric('args_count', len(args))
            ctx.add_metric('kwargs_count', len(kwargs))
            
            result = func(*args, **kwargs)
            
            # Add result metadata if possible
            if hasattr(result, '__len__'):
                try:
                    ctx.add_metric('result_size', len(result))
                except:
                    pass
            
            return result
    
    def get_component_metrics(self) -> ComponentMetrics:
        """
        Get comprehensive metrics for this component.
        
        Returns:
            ComponentMetrics object with current stats
        """
        with self.lock:
            return ComponentMetrics(
                component_name=self.component_name,
                total_operations=self.operation_count,
                total_time=self.total_execution_time,
                average_time=self.total_execution_time / max(self.operation_count, 1),
                success_rate=self.successful_operations / max(self.operation_count, 1),
                memory_usage=self._get_current_memory_usage(),
                error_count=self.failed_operations,
                last_operation_time=self.last_operation_time
            )
    
    def get_method_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all profiled methods.
        
        Returns:
            Dictionary mapping method names to their statistics
        """
        with self.lock:
            return {
                method: {
                    'call_count': stats['count'],
                    'total_time': stats['total_time'],
                    'average_time': stats['total_time'] / max(stats['count'], 1),
                    'success_rate': stats['successes'] / max(stats['count'], 1),
                    'min_time': stats['min_time'],
                    'max_time': stats['max_time'],
                    'last_call': stats['last_call']
                }
                for method, stats in self.method_stats.items()
            }
    
    def get_performance_summary(self) -> str:
        """
        Get a human-readable performance summary for this component.
        
        Returns:
            Formatted performance summary string
        """
        metrics = self.get_component_metrics()
        method_stats = self.get_method_stats()
        
        lines = [
            f"🔍 Component Performance Summary: {self.component_name}",
            "=" * 50,
            f"Total Operations: {metrics.total_operations:,}",
            f"Total Time: {metrics.total_time:.3f}s",
            f"Average Time: {metrics.average_time:.4f}s",
            f"Success Rate: {metrics.success_rate:.1%}",
            f"Memory Usage: {metrics.memory_usage:.1f}MB",
            f"Error Count: {metrics.error_count}",
        ]
        
        if self.last_operation_time:
            lines.append(f"Last Operation: {time.time() - self.last_operation_time:.1f}s ago")
        
        if method_stats:
            lines.append("\n📊 Method Breakdown:")
            sorted_methods = sorted(
                method_stats.items(),
                key=lambda x: x[1]['total_time'],
                reverse=True
            )
            
            for method, stats in sorted_methods[:10]:  # Top 10 methods
                lines.append(
                    f"  {method}: {stats['call_count']} calls, "
                    f"{stats['total_time']:.3f}s total, "
                    f"{stats['average_time']:.4f}s avg"
                )
        
        # Performance warnings
        warnings = self._get_performance_warnings(metrics)
        if warnings:
            lines.append("\n⚠️  Performance Warnings:")
            for warning in warnings:
                lines.append(f"  {warning}")
        
        return "\n".join(lines)
    
    def log_performance_summary(self):
        """Log the performance summary using the kernel logger."""
        if hasattr(self.kernel_profiler, 'logger') and self.kernel_profiler.logger:
            summary = self.get_performance_summary()
            self.kernel_profiler.logger.info(
                f"Component performance summary for {self.component_name}",
                summary=summary,
                component=self.component_name
            )
    
    def reset_stats(self):
        """Reset all component statistics."""
        with self.lock:
            self.operation_count = 0
            self.successful_operations = 0
            self.failed_operations = 0
            self.total_execution_time = 0.0
            self.last_operation_time = None
            self.method_stats.clear()
        
        if hasattr(self.kernel_profiler, 'logger') and self.kernel_profiler.logger:
            self.kernel_profiler.logger.info(f"Reset stats for component {self.component_name}")
    
    def _track_method_start(self, method_name: str):
        """Track the start of a method call."""
        with self.lock:
            if method_name not in self.method_stats:
                self.method_stats[method_name] = {
                    'count': 0,
                    'total_time': 0.0,
                    'successes': 0,
                    'failures': 0,
                    'min_time': float('inf'),
                    'max_time': 0.0,
                    'last_call': None
                }
    
    def _track_method_completion(self, method_name: str, execution_time: float, success: bool):
        """Track the completion of a method call."""
        with self.lock:
            if method_name in self.method_stats:
                stats = self.method_stats[method_name]
                stats['count'] += 1
                stats['total_time'] += execution_time
                stats['last_call'] = time.time()
                
                if success:
                    stats['successes'] += 1
                else:
                    stats['failures'] += 1
                
                stats['min_time'] = min(stats['min_time'], execution_time)
                stats['max_time'] = max(stats['max_time'], execution_time)
    
    def _track_operation_start(self):
        """Track the start of an operation."""
        with self.lock:
            self.operation_count += 1
    
    def _track_operation_completion(self, execution_time: float, success: bool):
        """Track the completion of an operation."""
        with self.lock:
            self.total_execution_time += execution_time
            self.last_operation_time = time.time()
            
            if success:
                self.successful_operations += 1
            else:
                self.failed_operations += 1
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _get_performance_warnings(self, metrics: ComponentMetrics) -> List[str]:
        """Get performance warnings based on current metrics."""
        warnings = []
        
        if metrics.average_time > self.slow_operation_threshold:
            warnings.append(
                f"Slow average operation time: {metrics.average_time:.3f}s "
                f"(threshold: {self.slow_operation_threshold}s)"
            )
        
        if metrics.memory_usage > self.memory_warning_threshold:
            warnings.append(
                f"High memory usage: {metrics.memory_usage:.1f}MB "
                f"(threshold: {self.memory_warning_threshold}MB)"
            )
        
        if metrics.success_rate < 0.95:  # Less than 95% success rate
            warnings.append(
                f"Low success rate: {metrics.success_rate:.1%} "
                f"({metrics.error_count} errors)"
            )
        
        # Check for methods with concerning stats
        method_stats = self.get_method_stats()
        slow_methods = [
            method for method, stats in method_stats.items()
            if stats['average_time'] > self.slow_operation_threshold
        ]
        
        if slow_methods:
            warnings.append(
                f"Slow methods detected: {', '.join(slow_methods[:3])}"
                + ("..." if len(slow_methods) > 3 else "")
            )
        
        return warnings


# Utility functions for component profiler creation
def create_component_profiler(kernel_profiler: KernelProfiler, 
                             component_name: str) -> ComponentProfiler:
    """
    Create a component profiler for a specific component.
    
    Args:
        kernel_profiler: The kernel profiler instance
        component_name: Name of the component
        
    Returns:
        ComponentProfiler instance
    """
    return ComponentProfiler(kernel_profiler, component_name)


def inject_profiler_into_component(component, kernel_profiler: KernelProfiler):
    """
    Inject a component profiler into a component instance.
    
    Args:
        component: Component instance to inject profiler into
        kernel_profiler: The kernel profiler instance
    """
    component_name = getattr(component, 'name', component.__class__.__name__)
    component._profiler = ComponentProfiler(kernel_profiler, component_name)
    
    if hasattr(kernel_profiler, 'logger') and kernel_profiler.logger:
        kernel_profiler.logger.debug(f"Injected profiler into component {component_name}")