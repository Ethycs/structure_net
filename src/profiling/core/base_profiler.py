#!/usr/bin/env python3
"""
Base Profiler Infrastructure

This module provides the foundational classes and interfaces for the
modular profiling system.
"""

import time
import threading
import psutil
import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class ProfilerLevel(Enum):
    """Profiling detail levels."""
    DISABLED = 0
    BASIC = 1
    DETAILED = 2
    COMPREHENSIVE = 3


class MetricType(Enum):
    """Types of metrics that can be profiled."""
    TIME = "time"
    MEMORY = "memory"
    COMPUTE = "compute"
    IO = "io"
    CUSTOM = "custom"


@dataclass
class ProfilerConfig:
    """Configuration for profiler behavior."""
    enabled: bool = True
    level: ProfilerLevel = ProfilerLevel.BASIC
    
    # What to profile
    profile_time: bool = True
    profile_memory: bool = True
    profile_compute: bool = False
    profile_io: bool = False
    
    # Sampling and collection
    sampling_interval: float = 0.1  # seconds
    max_samples: int = 1000
    auto_save: bool = True
    save_interval: int = 100  # operations
    
    # Output configuration
    output_dir: str = "profiling_results"
    output_format: str = "json"  # json, csv, wandb
    
    # Performance settings
    max_overhead_percent: float = 5.0  # Max acceptable overhead
    adaptive_sampling: bool = True
    
    # Integration settings
    integrate_with_wandb: bool = False
    integrate_with_logging: bool = True
    
    # Custom settings
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfiledOperation:
    """Represents a single profiled operation."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    
    # Resource usage
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    memory_peak: Optional[float] = None
    
    # GPU metrics (if available)
    gpu_memory_before: Optional[float] = None
    gpu_memory_after: Optional[float] = None
    gpu_utilization: Optional[float] = None
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Context information
    thread_id: Optional[int] = None
    process_id: Optional[int] = None
    component: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def finalize(self):
        """Finalize the operation by calculating derived metrics."""
        if self.end_time and self.start_time:
            self.duration = self.end_time - self.start_time
        
        if self.memory_before is not None and self.memory_after is not None:
            self.custom_metrics['memory_delta'] = self.memory_after - self.memory_before
        
        if self.gpu_memory_before is not None and self.gpu_memory_after is not None:
            self.custom_metrics['gpu_memory_delta'] = self.gpu_memory_after - self.gpu_memory_before
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'memory_before': self.memory_before,
            'memory_after': self.memory_after,
            'memory_peak': self.memory_peak,
            'gpu_memory_before': self.gpu_memory_before,
            'gpu_memory_after': self.gpu_memory_after,
            'gpu_utilization': self.gpu_utilization,
            'custom_metrics': self.custom_metrics,
            'thread_id': self.thread_id,
            'process_id': self.process_id,
            'component': self.component,
            'tags': self.tags
        }


class BaseProfiler(ABC):
    """
    Abstract base class for all profilers.
    
    Provides common functionality for profiling operations while allowing
    specialized profilers to implement domain-specific logic.
    """
    
    def __init__(self, name: str, config: ProfilerConfig):
        self.name = name
        self.config = config
        self.operations: List[ProfiledOperation] = []
        self.active_operations: Dict[str, ProfiledOperation] = {}
        self.lock = threading.Lock()
        
        # Performance tracking
        self.total_overhead = 0.0
        self.operation_count = 0
        
        # Setup output directory
        if self.config.auto_save:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    @property
    def is_enabled(self) -> bool:
        """Check if profiler is enabled."""
        return self.config.enabled and self.config.level != ProfilerLevel.DISABLED
    
    @property
    def average_overhead(self) -> float:
        """Get average overhead per operation."""
        if self.operation_count == 0:
            return 0.0
        return self.total_overhead / self.operation_count
    
    def start_operation(self, 
                       name: str, 
                       component: Optional[str] = None,
                       tags: Optional[List[str]] = None) -> str:
        """
        Start profiling an operation.
        
        Args:
            name: Operation name
            component: Component name (e.g., 'evolution', 'metrics')
            tags: Optional tags for categorization
            
        Returns:
            Operation ID for later reference
        """
        if not self.is_enabled:
            return ""
        
        profile_start = time.perf_counter()
        
        operation_id = f"{name}_{int(time.time() * 1000000)}"
        
        operation = ProfiledOperation(
            name=name,
            start_time=time.perf_counter(),
            thread_id=threading.get_ident(),
            process_id=psutil.Process().pid,
            component=component,
            tags=tags or []
        )
        
        # Collect initial metrics
        if self.config.profile_memory:
            operation.memory_before = self._get_memory_usage()
        
        if self.config.profile_compute and torch.cuda.is_available():
            operation.gpu_memory_before = self._get_gpu_memory_usage()
            operation.gpu_utilization = self._get_gpu_utilization()
        
        # Store operation
        with self.lock:
            self.active_operations[operation_id] = operation
        
        # Track overhead
        profile_end = time.perf_counter()
        self.total_overhead += profile_end - profile_start
        
        return operation_id
    
    def end_operation(self, operation_id: str, **custom_metrics):
        """
        End profiling an operation.
        
        Args:
            operation_id: ID returned from start_operation
            **custom_metrics: Additional custom metrics to record
        """
        if not self.is_enabled or not operation_id:
            return
        
        profile_start = time.perf_counter()
        
        with self.lock:
            if operation_id not in self.active_operations:
                return
            
            operation = self.active_operations.pop(operation_id)
        
        # Finalize operation
        operation.end_time = time.perf_counter()
        
        # Collect final metrics
        if self.config.profile_memory:
            operation.memory_after = self._get_memory_usage()
        
        if self.config.profile_compute and torch.cuda.is_available():
            operation.gpu_memory_after = self._get_gpu_memory_usage()
        
        # Add custom metrics
        operation.custom_metrics.update(custom_metrics)
        
        # Finalize derived metrics
        operation.finalize()
        
        # Store completed operation
        with self.lock:
            self.operations.append(operation)
            self.operation_count += 1
        
        # Auto-save if configured
        if (self.config.auto_save and 
            len(self.operations) % self.config.save_interval == 0):
            self.save_results()
        
        # Track overhead
        profile_end = time.perf_counter()
        self.total_overhead += profile_end - profile_start
        
        # Check overhead and adapt if needed
        if self.config.adaptive_sampling:
            self._adapt_sampling_if_needed()
    
    def profile_operation(self, 
                         name: str, 
                         component: Optional[str] = None,
                         tags: Optional[List[str]] = None):
        """
        Context manager for profiling operations.
        
        Usage:
            with profiler.profile_operation("my_operation"):
                # code to profile
                pass
        """
        return ProfiledOperationContext(self, name, component, tags)
    
    def get_metrics(self, 
                   component: Optional[str] = None,
                   operation_name: Optional[str] = None,
                   tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get profiling metrics with optional filtering.
        
        Args:
            component: Filter by component name
            operation_name: Filter by operation name
            tags: Filter by tags (operation must have all specified tags)
            
        Returns:
            List of operation dictionaries
        """
        with self.lock:
            operations = self.operations.copy()
        
        # Apply filters
        if component:
            operations = [op for op in operations if op.component == component]
        
        if operation_name:
            operations = [op for op in operations if op.name == operation_name]
        
        if tags:
            operations = [op for op in operations 
                         if all(tag in op.tags for tag in tags)]
        
        return [op.to_dict() for op in operations]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all profiled operations."""
        with self.lock:
            operations = self.operations.copy()
        
        if not operations:
            return {}
        
        # Calculate summary statistics
        durations = [op.duration for op in operations if op.duration is not None]
        memory_deltas = [op.custom_metrics.get('memory_delta', 0) for op in operations]
        
        stats = {
            'total_operations': len(operations),
            'total_time': sum(durations),
            'average_duration': sum(durations) / len(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'total_memory_delta': sum(memory_deltas),
            'average_overhead': self.average_overhead,
            'overhead_percentage': (self.average_overhead / (sum(durations) / len(durations))) * 100 if durations else 0
        }
        
        # Component breakdown
        component_stats = {}
        for op in operations:
            if op.component:
                if op.component not in component_stats:
                    component_stats[op.component] = {'count': 0, 'total_time': 0}
                component_stats[op.component]['count'] += 1
                if op.duration:
                    component_stats[op.component]['total_time'] += op.duration
        
        stats['component_breakdown'] = component_stats
        
        return stats
    
    def save_results(self, filename: Optional[str] = None):
        """Save profiling results to file."""
        if not self.config.auto_save and filename is None:
            return
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"{self.name}_profile_{timestamp}.{self.config.output_format}"
        
        filepath = Path(self.config.output_dir) / filename
        
        data = {
            'profiler_name': self.name,
            'config': {
                'level': self.config.level.name,
                'profile_time': self.config.profile_time,
                'profile_memory': self.config.profile_memory,
                'profile_compute': self.config.profile_compute,
                'profile_io': self.config.profile_io
            },
            'summary_stats': self.get_summary_stats(),
            'operations': self.get_metrics()
        }
        
        if self.config.output_format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        # TODO: Add CSV and other format support
        
        print(f"📊 Profiling results saved to {filepath}")
    
    def clear_results(self):
        """Clear all profiling results."""
        with self.lock:
            self.operations.clear()
            self.active_operations.clear()
            self.operation_count = 0
            self.total_overhead = 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
        except:
            pass
        return 0.0
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu
        except:
            return 0.0
    
    def _adapt_sampling_if_needed(self):
        """Adapt sampling rate if overhead is too high."""
        if self.average_overhead == 0:
            return
        
        # Calculate current overhead percentage
        if self.operation_count > 10:  # Only adapt after some operations
            avg_op_time = sum(op.duration for op in self.operations[-10:] 
                             if op.duration) / 10
            overhead_percent = (self.average_overhead / avg_op_time) * 100
            
            if overhead_percent > self.config.max_overhead_percent:
                # Reduce profiling detail
                if self.config.level == ProfilerLevel.COMPREHENSIVE:
                    self.config.level = ProfilerLevel.DETAILED
                elif self.config.level == ProfilerLevel.DETAILED:
                    self.config.level = ProfilerLevel.BASIC
                
                print(f"⚠️  Profiler {self.name}: Reduced detail level due to high overhead ({overhead_percent:.1f}%)")
    
    @abstractmethod
    def get_specialized_metrics(self) -> Dict[str, Any]:
        """Get profiler-specific metrics. Implemented by subclasses."""
        pass


class ProfiledOperationContext:
    """Context manager for profiling operations."""
    
    def __init__(self, 
                 profiler: BaseProfiler, 
                 name: str, 
                 component: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        self.profiler = profiler
        self.name = name
        self.component = component
        self.tags = tags
        self.operation_id = ""
    
    def __enter__(self):
        self.operation_id = self.profiler.start_operation(
            self.name, self.component, self.tags
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Add exception info if there was an error
        custom_metrics = {}
        if exc_type is not None:
            custom_metrics['exception'] = str(exc_type.__name__)
            custom_metrics['error'] = str(exc_val)
        
        self.profiler.end_operation(self.operation_id, **custom_metrics)
    
    def add_metric(self, name: str, value: Any):
        """Add a custom metric to the current operation."""
        if self.operation_id and self.operation_id in self.profiler.active_operations:
            self.profiler.active_operations[self.operation_id].custom_metrics[name] = value
