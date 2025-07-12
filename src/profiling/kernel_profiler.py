#!/usr/bin/env python3
"""
Kernel Profiler Implementation

Core profiling service for the StructureNet microkernel.
Provides centralized profiling capabilities for all components.
"""

import time
import threading
import psutil
import torch
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
import json

from .core.base_profiler import BaseProfiler, ProfilerConfig, ProfilerLevel, ProfiledOperation


@dataclass
class ProfileResult:
    """Result of a profiling operation."""
    name: str
    duration: float
    memory_delta: float
    cpu_percent: float
    success: bool = True
    custom_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}


class KernelProfiler(BaseProfiler):
    """
    Core profiling service for the StructureNet kernel.
    
    Provides profiling capabilities for all kernel and component operations
    with minimal overhead and comprehensive metric collection.
    """
    
    def __init__(self, 
                 logger=None,
                 enable_cpu: bool = True,
                 enable_memory: bool = True, 
                 enable_gpu: bool = None,
                 config: Optional[ProfilerConfig] = None):
        """
        Initialize the kernel profiler.
        
        Args:
            logger: Logger instance for profiler output
            enable_cpu: Whether to collect CPU metrics
            enable_memory: Whether to collect memory metrics  
            enable_gpu: Whether to collect GPU metrics (auto-detect if None)
            config: Profiler configuration
        """
        # Set default config if not provided
        if config is None:
            config = ProfilerConfig(
                profile_time=True,
                profile_memory=enable_memory,
                profile_compute=enable_gpu if enable_gpu is not None else torch.cuda.is_available(),
                profile_io=False,
                level=ProfilerLevel.BASIC
            )
        
        super().__init__("KernelProfiler", config)
        
        self.logger = logger
        self.enable_cpu = enable_cpu
        self.enable_memory = enable_memory
        self.enable_gpu = enable_gpu if enable_gpu is not None else torch.cuda.is_available()
        
        # Kernel-specific tracking
        self.component_profiles: Dict[str, List[ProfiledOperation]] = {}
        self.service_profiles: Dict[str, List[ProfiledOperation]] = {}
        self.kernel_startup_time = time.time()
        
        # Performance tracking
        self.total_kernel_overhead = 0.0
        self.profile_counts_by_component: Dict[str, int] = {}
        
        if self.logger:
            self.logger.info("KernelProfiler initialized", 
                           cpu_enabled=self.enable_cpu,
                           memory_enabled=self.enable_memory,
                           gpu_enabled=self.enable_gpu,
                           level=self.config.level.name)
    
    @property
    def kernel_uptime(self) -> float:
        """Get kernel uptime in seconds."""
        return time.time() - self.kernel_startup_time
    
    def start_profile(self, name: str) -> str:
        """
        Start profiling an operation.
        
        Args:
            name: Operation name
            
        Returns:
            Profile ID for later reference
        """
        profile_id = f"{name}_{int(time.time() * 1000000)}"
        
        start_data = {
            'name': name,
            'start_time': time.time(),
            'start_memory': self._get_memory_usage() if self.enable_memory else 0,
            'start_cpu': self._get_cpu_usage() if self.enable_cpu else 0
        }
        
        if self.enable_gpu and torch.cuda.is_available():
            start_data['start_gpu_memory'] = self._get_gpu_memory_usage()
            start_data['start_gpu_util'] = self._get_gpu_utilization()
        
        with self.lock:
            self.active_operations[profile_id] = start_data
        
        return profile_id
    
    def end_profile(self, profile_id: str, success: bool = True) -> ProfileResult:
        """
        End profiling an operation.
        
        Args:
            profile_id: ID returned from start_profile
            success: Whether the operation succeeded
            
        Returns:
            ProfileResult with collected metrics
        """
        if profile_id not in self.active_operations:
            raise ValueError(f"No active profile: {profile_id}")
        
        with self.lock:
            profile_data = self.active_operations.pop(profile_id)
        
        end_time = time.time()
        duration = end_time - profile_data['start_time']
        
        # Calculate resource deltas
        memory_delta = 0.0
        if self.enable_memory:
            current_memory = self._get_memory_usage()
            memory_delta = current_memory - profile_data['start_memory']
        
        cpu_percent = 0.0
        if self.enable_cpu:
            cpu_percent = self._get_cpu_usage()
        
        custom_metrics = {}
        if self.enable_gpu and torch.cuda.is_available():
            current_gpu_memory = self._get_gpu_memory_usage()
            current_gpu_util = self._get_gpu_utilization()
            custom_metrics['gpu_memory_delta'] = current_gpu_memory - profile_data.get('start_gpu_memory', 0)
            custom_metrics['gpu_utilization'] = current_gpu_util
        
        result = ProfileResult(
            name=profile_data['name'],
            duration=duration,
            memory_delta=memory_delta,
            cpu_percent=cpu_percent,
            success=success,
            custom_metrics=custom_metrics
        )
        
        # Store in completed profiles
        operation = ProfiledOperation(
            name=profile_data['name'],
            start_time=profile_data['start_time'],
            end_time=end_time,
            duration=duration,
            memory_before=profile_data.get('start_memory'),
            memory_after=profile_data.get('start_memory', 0) + memory_delta,
            custom_metrics=custom_metrics
        )
        operation.finalize()
        
        with self.lock:
            self.operations.append(operation)
        
        return result
    
    @contextmanager
    def profile(self, name: str):
        """
        Context manager for profiling operations.
        
        Args:
            name: Operation name
            
        Usage:
            with profiler.profile("my_operation"):
                # code to profile
                pass
        """
        profile_id = self.start_profile(name)
        try:
            yield profile_id
            self.end_profile(profile_id, success=True)
        except Exception:
            self.end_profile(profile_id, success=False)
            raise
    
    def profile_component_operation(self, 
                                   component_name: str,
                                   operation_name: str,
                                   tags: Optional[List[str]] = None):
        """
        Profile a component operation with automatic categorization.
        
        Args:
            component_name: Name of the component
            operation_name: Name of the operation
            tags: Optional tags for categorization
        """
        full_name = f"{component_name}.{operation_name}"
        operation_id = self.start_operation(full_name, component_name, tags)
        
        # Track component-specific metrics
        if component_name not in self.profile_counts_by_component:
            self.profile_counts_by_component[component_name] = 0
        self.profile_counts_by_component[component_name] += 1
        
        return ComponentOperationContext(self, operation_id, component_name)
    
    def get_component_metrics(self, component_name: str) -> Dict[str, Any]:
        """
        Get metrics specific to a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Component-specific metrics
        """
        component_operations = [
            op for op in self.operations 
            if op.component == component_name
        ]
        
        if not component_operations:
            return {}
        
        total_time = sum(op.duration for op in component_operations if op.duration)
        total_memory = sum(op.custom_metrics.get('memory_delta', 0) for op in component_operations)
        
        return {
            'component_name': component_name,
            'total_operations': len(component_operations),
            'total_time': total_time,
            'average_time': total_time / len(component_operations),
            'total_memory_delta': total_memory,
            'profile_count': self.profile_counts_by_component.get(component_name, 0),
            'operations': [op.to_dict() for op in component_operations[-10:]]  # Last 10
        }
    
    def get_kernel_health_metrics(self) -> Dict[str, Any]:
        """
        Get overall kernel health and performance metrics.
        
        Returns:
            Kernel health metrics
        """
        return {
            'kernel_uptime': self.kernel_uptime,
            'total_profiles': len(self.operations),
            'active_profiles': len(self.active_operations),
            'total_overhead': self.total_overhead,
            'average_overhead': self.average_overhead,
            'components_profiled': len(self.profile_counts_by_component),
            'memory_usage_mb': self._get_memory_usage(),
            'cpu_usage_percent': self._get_cpu_usage() if self.enable_cpu else 0,
            'gpu_memory_mb': self._get_gpu_memory_usage() if self.enable_gpu else 0,
            'profiler_enabled': self.is_enabled,
            'profiler_level': self.config.level.name
        }
    
    def get_specialized_metrics(self) -> Dict[str, Any]:
        """Get kernel-specific specialized metrics."""
        component_breakdown = {}
        for component, count in self.profile_counts_by_component.items():
            component_breakdown[component] = self.get_component_metrics(component)
        
        return {
            'kernel_health': self.get_kernel_health_metrics(),
            'component_breakdown': component_breakdown,
            'service_profiles': len(self.service_profiles),
            'profiler_overhead_percent': (self.average_overhead / self.kernel_uptime * 100) if self.kernel_uptime > 0 else 0
        }
    
    def save_kernel_report(self, filepath: Optional[str] = None):
        """
        Save a comprehensive kernel profiling report.
        
        Args:
            filepath: Optional custom filepath
        """
        if filepath is None:
            timestamp = int(time.time())
            filepath = f"kernel_profile_report_{timestamp}.json"
        
        report = {
            'kernel_info': {
                'uptime': self.kernel_uptime,
                'profiler_version': '1.0.0',
                'generated_at': time.time()
            },
            'health_metrics': self.get_kernel_health_metrics(),
            'specialized_metrics': self.get_specialized_metrics(),
            'summary_stats': self.get_summary_stats(),
            'component_details': {
                component: self.get_component_metrics(component)
                for component in self.profile_counts_by_component.keys()
            }
        }
        
        output_path = Path(self.config.output_dir) / filepath
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        if self.logger:
            self.logger.info(f"Kernel profiling report saved to {output_path}")
        
        print(f"🔬 Kernel profiling report saved to {output_path}")
    
    def log_performance_warning(self, threshold_seconds: float = 1.0):
        """
        Log warnings for operations that exceed performance thresholds.
        
        Args:
            threshold_seconds: Threshold for slow operations
        """
        slow_operations = [
            op for op in self.operations 
            if op.duration and op.duration > threshold_seconds
        ]
        
        if slow_operations and self.logger:
            for op in slow_operations[-5:]:  # Last 5 slow operations
                self.logger.warning(
                    f"Slow operation detected: {op.name}",
                    duration=op.duration,
                    component=op.component,
                    memory_delta=op.custom_metrics.get('memory_delta', 0)
                )


class ComponentOperationContext:
    """Context manager for component operations."""
    
    def __init__(self, profiler: KernelProfiler, operation_id: str, component_name: str):
        self.profiler = profiler
        self.operation_id = operation_id
        self.component_name = component_name
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        custom_metrics = {}
        
        if not success:
            custom_metrics['exception'] = str(exc_type.__name__)
            custom_metrics['error'] = str(exc_val)
        
        self.profiler.end_operation(self.operation_id, **custom_metrics)
    
    def add_metric(self, name: str, value: Any):
        """Add a custom metric to this operation."""
        if (self.operation_id in self.profiler.active_operations and 
            hasattr(self.profiler, 'active_operations')):
            # Add to the profiled operation
            if self.operation_id in self.profiler.active_operations:
                if 'custom_metrics' not in self.profiler.active_operations[self.operation_id]:
                    self.profiler.active_operations[self.operation_id]['custom_metrics'] = {}
                self.profiler.active_operations[self.operation_id]['custom_metrics'][name] = value