#!/usr/bin/env python3
"""
Profiler Context Manager

Provides context management utilities for profiling operations with
optimized overhead and flexible configuration.
"""

import time
import threading
from typing import Any, Optional, List, Dict, Callable
from contextlib import contextmanager

from .base_profiler import BaseProfiler, ProfilerLevel
from .profiler_manager import get_global_profiler_manager


class ProfilerContext:
    """
    High-performance context manager for profiling operations.
    
    Optimized for minimal overhead when profiling is disabled
    and efficient batch processing when enabled.
    """
    
    def __init__(self, 
                 operation_name: str,
                 component: str = "general",
                 profiler_names: Optional[List[str]] = None,
                 tags: Optional[List[str]] = None,
                 level: ProfilerLevel = ProfilerLevel.BASIC,
                 lazy_init: bool = True):
        self.operation_name = operation_name
        self.component = component
        self.profiler_names = profiler_names
        self.tags = tags or []
        self.level = level
        self.lazy_init = lazy_init
        
        # State tracking
        self.operation_ids: Dict[str, str] = {}
        self.custom_metrics: Dict[str, Any] = {}
        self.start_time = 0.0
        self.manager = None
        
        # Optimization flags
        self._is_enabled = None
        self._profilers_to_use = None
    
    def __enter__(self):
        # Fast path: check if profiling is globally disabled
        if self._is_enabled is False:
            return self
        
        # Lazy initialization for better performance
        if self.lazy_init and self.manager is None:
            self.manager = get_global_profiler_manager()
            
            # Cache enabled state for fast future checks
            self._is_enabled = (self.manager.global_config.enabled and 
                              self.manager.global_config.level.value >= self.level.value)
            
            if not self._is_enabled:
                return self
            
            # Cache profilers to use
            self._profilers_to_use = (
                {name: self.manager.profilers[name] for name in self.profiler_names 
                 if name in self.manager.profilers}
                if self.profiler_names
                else self.manager.profilers
            )
        
        # Start profiling in applicable profilers
        self.start_time = time.perf_counter()
        
        for profiler_name, profiler in (self._profilers_to_use or {}).items():
            if profiler.is_enabled:
                operation_id = profiler.start_operation(
                    self.operation_name, 
                    self.component, 
                    self.tags
                )
                self.operation_ids[profiler_name] = operation_id
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Fast path: if profiling was never enabled
        if not self.operation_ids:
            return
        
        # Add execution time and exception info
        if self.start_time > 0:
            self.custom_metrics['execution_time'] = time.perf_counter() - self.start_time
        
        if exc_type is not None:
            self.custom_metrics['exception'] = str(exc_type.__name__)
            self.custom_metrics['error'] = str(exc_val)
            self.custom_metrics['success'] = False
        else:
            self.custom_metrics['success'] = True
        
        # End operations in all profilers
        for profiler_name, operation_id in self.operation_ids.items():
            if profiler_name in (self._profilers_to_use or {}):
                self._profilers_to_use[profiler_name].end_operation(
                    operation_id, **self.custom_metrics
                )
    
    def add_metric(self, name: str, value: Any):
        """Add a custom metric to the current operation."""
        self.custom_metrics[name] = value
        
        # Also add to active operations if available
        for profiler_name, operation_id in self.operation_ids.items():
            if (profiler_name in (self._profilers_to_use or {}) and 
                operation_id in self._profilers_to_use[profiler_name].active_operations):
                self._profilers_to_use[profiler_name].active_operations[operation_id].custom_metrics[name] = value
    
    def add_metrics(self, metrics: Dict[str, Any]):
        """Add multiple custom metrics at once."""
        for name, value in metrics.items():
            self.add_metric(name, value)


class BatchProfilerContext:
    """
    Batch profiler context for high-frequency operations.
    
    Collects multiple operations and processes them in batches
    to reduce overhead for high-frequency profiling.
    """
    
    def __init__(self, 
                 batch_size: int = 100,
                 flush_interval: float = 5.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.operations_batch = []
        self.last_flush = time.time()
        self.lock = threading.Lock()
    
    def add_operation(self, 
                     operation_name: str,
                     component: str,
                     duration: float,
                     custom_metrics: Optional[Dict[str, Any]] = None):
        """Add an operation to the batch."""
        operation = {
            'name': operation_name,
            'component': component,
            'duration': duration,
            'timestamp': time.time(),
            'custom_metrics': custom_metrics or {}
        }
        
        with self.lock:
            self.operations_batch.append(operation)
            
            # Check if we need to flush
            should_flush = (
                len(self.operations_batch) >= self.batch_size or
                time.time() - self.last_flush >= self.flush_interval
            )
            
            if should_flush:
                self._flush_batch()
    
    def _flush_batch(self):
        """Flush the current batch to profilers."""
        if not self.operations_batch:
            return
        
        manager = get_global_profiler_manager()
        
        # Process batch efficiently
        for operation in self.operations_batch:
            # Use context manager for each operation
            with ProfilerContext(
                operation['name'], 
                operation['component'],
                lazy_init=False  # Already have manager
            ) as ctx:
                ctx.manager = manager
                ctx.add_metrics(operation['custom_metrics'])
        
        # Clear batch
        self.operations_batch.clear()
        self.last_flush = time.time()
    
    def flush(self):
        """Manually flush the current batch."""
        with self.lock:
            self._flush_batch()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()


# Global batch profiler for high-frequency operations
_global_batch_profiler = None


def get_global_batch_profiler() -> BatchProfilerContext:
    """Get the global batch profiler instance."""
    global _global_batch_profiler
    if _global_batch_profiler is None:
        _global_batch_profiler = BatchProfilerContext()
    return _global_batch_profiler


@contextmanager
def profile_operation(operation_name: str,
                     component: str = "general",
                     profiler_names: Optional[List[str]] = None,
                     tags: Optional[List[str]] = None,
                     level: ProfilerLevel = ProfilerLevel.BASIC):
    """
    Optimized context manager for profiling operations.
    
    Usage:
        with profile_operation("my_operation", "evolution"):
            # code to profile
            pass
    """
    with ProfilerContext(operation_name, component, profiler_names, tags, level) as ctx:
        yield ctx


@contextmanager
def profile_batch_operation(operation_name: str,
                           component: str = "general",
                           custom_metrics: Optional[Dict[str, Any]] = None):
    """
    High-performance batch profiling for frequent operations.
    
    Usage:
        with profile_batch_operation("frequent_op", "training"):
            # high-frequency code
            pass
    """
    start_time = time.perf_counter()
    
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        batch_profiler = get_global_batch_profiler()
        batch_profiler.add_operation(operation_name, component, duration, custom_metrics)


def profile_function_call(func: Callable, 
                         operation_name: Optional[str] = None,
                         component: str = "general",
                         *args, **kwargs):
    """
    Profile a single function call without decorators.
    
    Usage:
        result = profile_function_call(expensive_function, "computation", "analysis", arg1, arg2)
    """
    op_name = operation_name or func.__name__
    
    with profile_operation(op_name, component) as ctx:
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


# Convenience functions for common profiling patterns
def profile_if_slow(threshold_seconds: float = 0.1):
    """
    Decorator that only profiles operations that take longer than threshold.
    
    Usage:
        @profile_if_slow(0.5)  # Only profile if takes > 0.5 seconds
        def potentially_slow_function():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            
            if duration > threshold_seconds:
                # Retroactively profile the slow operation
                batch_profiler = get_global_batch_profiler()
                batch_profiler.add_operation(
                    func.__name__,
                    "slow_operations",
                    duration,
                    {'threshold_exceeded': True, 'threshold': threshold_seconds}
                )
            
            return result
        return wrapper
    return decorator


def profile_memory_intensive(func):
    """
    Decorator for profiling memory-intensive operations.
    
    Usage:
        @profile_memory_intensive
        def memory_heavy_function():
            pass
    """
    def wrapper(*args, **kwargs):
        with profile_operation(func.__name__, "memory_intensive", 
                             level=ProfilerLevel.DETAILED) as ctx:
            ctx.add_metric('memory_intensive', True)
            return func(*args, **kwargs)
    return wrapper
