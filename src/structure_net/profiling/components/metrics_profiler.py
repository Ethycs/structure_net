#!/usr/bin/env python3
"""
Metrics Component Profiler

Specialized profiler for metric computation operations, tracking computation
time, memory usage, and call frequency per metric type.
"""

import time
from typing import Dict, List, Any, Optional
import numpy as np

from ..core.base_profiler import BaseProfiler, ProfilerConfig


class MetricsProfiler(BaseProfiler):
    """
    Specialized profiler for metric computation components.

    Tracks metric-specific data like computation time per metric type,
    memory cost, call frequency, and identifies expensive metrics.
    """

    def __init__(self, config: Optional[ProfilerConfig] = None):
        config = config or ProfilerConfig()
        super().__init__("metrics_profiler", config)

        # Metrics-specific tracking
        self.metric_stats: Dict[str, Dict[str, Any]] = {}
        self.metric_calls: List[Dict[str, Any]] = []

        print("📏 MetricsProfiler initialized")

    def track_metric_computation(
        self,
        metric_name: str,
        execution_time: float,
        result_size: int = 0,
        network_size: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Track a single metric computation.

        Args:
            metric_name: Name of the metric computed
            execution_time: Time taken in seconds
            result_size: Size of the result (number of values produced)
            network_size: Size of the network being measured
            metadata: Additional metadata
        """
        if not self.is_enabled:
            return

        call_record = {
            "timestamp": time.time(),
            "metric_name": metric_name,
            "execution_time": execution_time,
            "result_size": result_size,
            "network_size": network_size,
            "metadata": metadata or {},
        }
        self.metric_calls.append(call_record)

        # Update per-metric statistics
        if metric_name not in self.metric_stats:
            self.metric_stats[metric_name] = {
                "total_calls": 0,
                "total_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
                "times": [],
                "total_result_size": 0,
            }

        stats = self.metric_stats[metric_name]
        stats["total_calls"] += 1
        stats["total_time"] += execution_time
        stats["min_time"] = min(stats["min_time"], execution_time)
        stats["max_time"] = max(stats["max_time"], execution_time)
        stats["times"].append(execution_time)
        stats["total_result_size"] += result_size

    def get_specialized_metrics(self) -> Dict[str, Any]:
        """Get metrics-specific profiling data."""
        metrics: Dict[str, Any] = {
            "metric_computations": {
                "total_calls": len(self.metric_calls),
                "unique_metrics": len(self.metric_stats),
                "total_computation_time": sum(
                    s["total_time"] for s in self.metric_stats.values()
                ),
            },
            "per_metric": {},
        }

        for metric_name, stats in self.metric_stats.items():
            times = stats["times"]
            per_metric = {
                "total_calls": stats["total_calls"],
                "total_time": stats["total_time"],
                "average_time": stats["total_time"] / stats["total_calls"],
                "min_time": stats["min_time"],
                "max_time": stats["max_time"],
                "total_result_size": stats["total_result_size"],
            }
            if len(times) > 1:
                per_metric["std_time"] = float(np.std(times))
            metrics["per_metric"][metric_name] = per_metric

        return metrics

    def get_expensive_metrics(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get the most expensive metrics by total computation time."""
        ranked = []
        for metric_name, stats in self.metric_stats.items():
            ranked.append(
                {
                    "metric_name": metric_name,
                    "total_time": stats["total_time"],
                    "total_calls": stats["total_calls"],
                    "average_time": stats["total_time"] / stats["total_calls"],
                }
            )
        ranked.sort(key=lambda x: x["total_time"], reverse=True)
        return ranked[:top_n]

    def clear_results(self):
        """Clear all metrics-specific results."""
        super().clear_results()
        self.metric_stats.clear()
        self.metric_calls.clear()
