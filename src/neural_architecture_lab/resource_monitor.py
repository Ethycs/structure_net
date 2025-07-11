"""
Resource monitoring and auto-balancing for NAL.

Monitors CPU, GPU, and memory usage to dynamically adjust experiment parallelism
and batch sizes for optimal resource utilization.
"""

import psutil
import torch
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """Current system resource metrics."""
    cpu_percent: float
    memory_percent: float
    gpu_utilization: Dict[int, float]  # GPU ID -> utilization %
    gpu_memory: Dict[int, float]  # GPU ID -> memory %
    timestamp: float


@dataclass
class ResourceLimits:
    """Resource utilization targets and limits."""
    target_cpu_percent: float = 75.0
    max_cpu_percent: float = 90.0
    target_gpu_percent: float = 85.0
    max_gpu_percent: float = 95.0
    target_memory_percent: float = 80.0
    max_memory_percent: float = 90.0
    min_batch_size: int = 32
    max_batch_size: int = 512
    min_parallel_experiments: int = 1
    max_parallel_experiments: int = 16


class ResourceMonitor:
    """Monitors system resources and provides recommendations."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.cpu_history = deque(maxlen=window_size)
        self.memory_history = deque(maxlen=window_size)
        self.gpu_history = deque(maxlen=window_size)
        
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # GPU metrics
        gpu_utilization = {}
        gpu_memory = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    # Get GPU utilization using nvidia-ml-py if available
                    # For now, use a simple memory-based estimate
                    props = torch.cuda.get_device_properties(i)
                    used = torch.cuda.memory_allocated(i)
                    total = props.total_memory
                    gpu_memory[i] = (used / total) * 100
                    
                    # Estimate utilization from memory (simplified)
                    # In practice, you'd use nvidia-ml-py for accurate GPU utilization
                    gpu_utilization[i] = min(gpu_memory[i] * 1.2, 100.0)
                except Exception as e:
                    logger.warning(f"Could not get GPU {i} metrics: {e}")
                    gpu_utilization[i] = 0.0
                    gpu_memory[i] = 0.0
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_utilization=gpu_utilization,
            gpu_memory=gpu_memory,
            timestamp=time.time()
        )
    
    def update_history(self, metrics: ResourceMetrics):
        """Update resource history with new metrics."""
        self.cpu_history.append(metrics.cpu_percent)
        self.memory_history.append(metrics.memory_percent)
        if metrics.gpu_utilization:
            avg_gpu = np.mean(list(metrics.gpu_utilization.values()))
            self.gpu_history.append(avg_gpu)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average resource metrics over the window."""
        return {
            'cpu': np.mean(self.cpu_history) if self.cpu_history else 0.0,
            'memory': np.mean(self.memory_history) if self.memory_history else 0.0,
            'gpu': np.mean(self.gpu_history) if self.gpu_history else 0.0
        }


class AutoBalancer:
    """Automatically balances resource usage for optimal performance."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.monitor = ResourceMonitor()
        self.last_adjustment_time = 0
        self.adjustment_cooldown = 5.0  # seconds
        
    def should_adjust(self) -> bool:
        """Check if enough time has passed for another adjustment."""
        return time.time() - self.last_adjustment_time > self.adjustment_cooldown
    
    def get_recommendations(
        self, 
        current_parallel: int,
        current_batch_size: int,
        current_workers: int
    ) -> Dict[str, int]:
        """Get recommended settings based on current resource usage."""
        metrics = self.monitor.get_current_metrics()
        self.monitor.update_history(metrics)
        avg_metrics = self.monitor.get_average_metrics()
        
        recommendations = {
            'parallel_experiments': current_parallel,
            'batch_size': current_batch_size,
            'num_workers': current_workers
        }
        
        if not self.should_adjust():
            return recommendations
        
        # CPU-bound: reduce parallel experiments or workers
        if avg_metrics['cpu'] > self.limits.max_cpu_percent:
            if current_parallel > self.limits.min_parallel_experiments:
                recommendations['parallel_experiments'] = max(
                    self.limits.min_parallel_experiments,
                    current_parallel - 1
                )
                logger.info(f"High CPU usage ({avg_metrics['cpu']:.1f}%), reducing parallel experiments to {recommendations['parallel_experiments']}")
            elif current_workers > 0:
                recommendations['num_workers'] = max(0, current_workers - 1)
                logger.info(f"High CPU usage, reducing workers to {recommendations['num_workers']}")
        
        # Memory-bound: reduce batch size or parallel experiments
        elif avg_metrics['memory'] > self.limits.max_memory_percent:
            if current_batch_size > self.limits.min_batch_size:
                recommendations['batch_size'] = max(
                    self.limits.min_batch_size,
                    int(current_batch_size * 0.75)
                )
                logger.info(f"High memory usage ({avg_metrics['memory']:.1f}%), reducing batch size to {recommendations['batch_size']}")
            elif current_parallel > self.limits.min_parallel_experiments:
                recommendations['parallel_experiments'] = max(
                    self.limits.min_parallel_experiments,
                    current_parallel - 1
                )
                logger.info(f"High memory usage, reducing parallel experiments to {recommendations['parallel_experiments']}")
        
        # Underutilized: increase resources
        elif (avg_metrics['cpu'] < self.limits.target_cpu_percent and 
              avg_metrics['memory'] < self.limits.target_memory_percent):
            
            # Prioritize GPU utilization
            if avg_metrics['gpu'] < self.limits.target_gpu_percent:
                # Increase batch size first (better GPU utilization)
                if current_batch_size < self.limits.max_batch_size:
                    recommendations['batch_size'] = min(
                        self.limits.max_batch_size,
                        int(current_batch_size * 1.5)
                    )
                    logger.info(f"Low GPU usage ({avg_metrics['gpu']:.1f}%), increasing batch size to {recommendations['batch_size']}")
                # Then increase parallel experiments
                elif current_parallel < self.limits.max_parallel_experiments:
                    recommendations['parallel_experiments'] = min(
                        self.limits.max_parallel_experiments,
                        current_parallel + 1
                    )
                    logger.info(f"Low resource usage, increasing parallel experiments to {recommendations['parallel_experiments']}")
            
            # Increase workers if CPU is underutilized
            if avg_metrics['cpu'] < 50 and current_workers < 4:
                recommendations['num_workers'] = min(4, current_workers + 1)
                logger.info(f"Low CPU usage, increasing workers to {recommendations['num_workers']}")
        
        self.last_adjustment_time = time.time()
        return recommendations
    
    def get_optimal_initial_settings(self) -> Dict[str, int]:
        """Get optimal initial settings based on system resources."""
        n_cpus = psutil.cpu_count()
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Base calculations
        if n_gpus > 0:
            # GPU-optimized settings
            parallel_experiments = min(n_gpus * 2, n_cpus // 3)
            batch_size = 256 if total_memory_gb > 16 else 128
            num_workers = min(2, n_cpus // (parallel_experiments * 2))
        else:
            # CPU-only settings
            parallel_experiments = min(4, n_cpus // 2)
            batch_size = 64
            num_workers = 1
        
        return {
            'parallel_experiments': max(1, parallel_experiments),
            'batch_size': batch_size,
            'num_workers': max(0, num_workers)
        }


# Singleton instance
_auto_balancer = None


def get_auto_balancer(limits: Optional[ResourceLimits] = None) -> AutoBalancer:
    """Get or create the global auto-balancer instance."""
    global _auto_balancer
    if _auto_balancer is None:
        _auto_balancer = AutoBalancer(limits)
    return _auto_balancer