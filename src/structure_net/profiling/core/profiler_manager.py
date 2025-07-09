#!/usr/bin/env python3
"""
Profiler Manager

Central coordinator for all profiling activities in the structure_net system.
Manages multiple profilers, provides unified interface, and handles integration
with logging and monitoring systems.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Type, Union
from pathlib import Path
import json

from .base_profiler import BaseProfiler, ProfilerConfig, ProfilerLevel
from ....logging import create_profiling_logger, StandardizedLogger
# Import evolution profiler - others will be added as they're implemented
try:
    from ..components.evolution_profiler import EvolutionProfiler
except ImportError:
    EvolutionProfiler = None


class ProfilerManager:
    """
    Central manager for all profiling activities.
    
    Provides a unified interface for managing multiple profilers,
    coordinating their activities, and aggregating results.
    """
    
    def __init__(self, global_config: Optional[ProfilerConfig] = None):
        self.global_config = global_config or ProfilerConfig()
        self.profilers: Dict[str, BaseProfiler] = {}
        self.lock = threading.Lock()
        
        # Global state
        self.is_profiling = False
        self.session_start_time = None
        self.session_id = None
        
        # Integration hooks
        self.wandb_logger = None
        self.standard_logger: Optional[StandardizedLogger] = None
        
        print(f"🔬 ProfilerManager initialized (level: {self.global_config.level.name})")
    
    def register_profiler(self, profiler: BaseProfiler):
        """Register a profiler with the manager."""
        with self.lock:
            self.profilers[profiler.name] = profiler
        print(f"📊 Registered profiler: {profiler.name}")
    
    def unregister_profiler(self, name: str):
        """Unregister a profiler."""
        with self.lock:
            if name in self.profilers:
                del self.profilers[name]
                print(f"📊 Unregistered profiler: {name}")
    
    def get_profiler(self, name: str) -> Optional[BaseProfiler]:
        """Get a specific profiler by name."""
        return self.profilers.get(name)
    
    def start_session(self, session_name: Optional[str] = None):
        """Start a new profiling session."""
        self.session_start_time = time.time()
        self.session_id = session_name or f"session_{int(self.session_start_time)}"
        self.is_profiling = True
        
        # Clear all profilers
        for profiler in self.profilers.values():
            profiler.clear_results()
            
        # Initialize logger if integration is enabled
        if self.global_config.integrate_with_logging:
            self.standard_logger = create_profiling_logger(
                session_id=self.session_id,
                config=self.global_config.custom_metrics
            )
        
        print(f"🚀 Started profiling session: {self.session_id}")
    
    def end_session(self, save_results: bool = True) -> Dict[str, Any]:
        """End the current profiling session and optionally save results."""
        if not self.is_profiling:
            return {}
        
        self.is_profiling = False
        session_duration = time.time() - self.session_start_time
        
        # Collect results from all profilers
        session_results = {
            'session_id': self.session_id,
            'session_duration': session_duration,
            'start_time': self.session_start_time,
            'end_time': time.time(),
            'profilers': {}
        }
        
        for name, profiler in self.profilers.items():
            session_results['profilers'][name] = {
                'summary_stats': profiler.get_summary_stats(),
                'specialized_metrics': profiler.get_specialized_metrics(),
                'operations_count': len(profiler.operations)
            }
        
        if save_results:
            self._save_session_results(session_results)
            
        # Log to standardized logging system if integrated
        if self.global_config.integrate_with_logging and self.standard_logger:
            logged_data = self._transform_results_for_logging(session_results)
            self.standard_logger.experiment_data.update(logged_data)
            self.standard_logger.finish_experiment()
            print(f"📦 Logged profiling session to artifacts: {self.session_id}")
        
        print(f"🏁 Ended profiling session: {self.session_id} (duration: {session_duration:.2f}s)")
        return session_results
    
    def _transform_results_for_logging(self, session_results: Dict[str, Any]) -> Dict[str, Any]:
        """Transform session results into the format expected by ProfilingExperiment schema."""
        return {
            "session_duration": session_results.get("session_duration", 0),
            "profilers": {
                name: {
                    "summary_stats": data.get("summary_stats", {}),
                    "specialized_metrics": data.get("specialized_metrics", {}),
                    "operations_count": data.get("operations_count", 0)
                }
                for name, data in session_results.get("profilers", {}).items()
            }
        }
    
    def profile_operation(self, 
                         operation_name: str,
                         component: str,
                         profiler_names: Optional[List[str]] = None,
                         tags: Optional[List[str]] = None):
        """
        Context manager for profiling an operation across multiple profilers.
        
        Args:
            operation_name: Name of the operation
            component: Component name (e.g., 'evolution', 'metrics')
            profiler_names: Specific profilers to use (None = all applicable)
            tags: Optional tags for categorization
        """
        return MultiProfilerContext(self, operation_name, component, profiler_names, tags)
    
    def get_aggregated_metrics(self, 
                              component: Optional[str] = None,
                              operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregated metrics from all profilers."""
        aggregated = {
            'total_operations': 0,
            'total_time': 0.0,
            'total_memory_delta': 0.0,
            'profiler_breakdown': {},
            'component_breakdown': {},
            'operation_breakdown': {}
        }
        
        for profiler_name, profiler in self.profilers.items():
            metrics = profiler.get_metrics(component, operation_name)
            stats = profiler.get_summary_stats()
            
            # Aggregate totals
            aggregated['total_operations'] += stats.get('total_operations', 0)
            aggregated['total_time'] += stats.get('total_time', 0.0)
            aggregated['total_memory_delta'] += stats.get('total_memory_delta', 0.0)
            
            # Profiler breakdown
            aggregated['profiler_breakdown'][profiler_name] = stats
            
            # Component breakdown
            component_stats = stats.get('component_breakdown', {})
            for comp, comp_stats in component_stats.items():
                if comp not in aggregated['component_breakdown']:
                    aggregated['component_breakdown'][comp] = {'count': 0, 'total_time': 0}
                aggregated['component_breakdown'][comp]['count'] += comp_stats['count']
                aggregated['component_breakdown'][comp]['total_time'] += comp_stats['total_time']
            
            # Operation breakdown
            for metric in metrics:
                op_name = metric['name']
                if op_name not in aggregated['operation_breakdown']:
                    aggregated['operation_breakdown'][op_name] = {
                        'count': 0, 
                        'total_time': 0.0,
                        'avg_time': 0.0,
                        'profilers': []
                    }
                
                aggregated['operation_breakdown'][op_name]['count'] += 1
                if metric['duration']:
                    aggregated['operation_breakdown'][op_name]['total_time'] += metric['duration']
                
                if profiler_name not in aggregated['operation_breakdown'][op_name]['profilers']:
                    aggregated['operation_breakdown'][op_name]['profilers'].append(profiler_name)
        
        # Calculate averages
        for op_name, op_stats in aggregated['operation_breakdown'].items():
            if op_stats['count'] > 0:
                op_stats['avg_time'] = op_stats['total_time'] / op_stats['count']
        
        return aggregated
    
    def get_performance_report(self) -> str:
        """Generate a human-readable performance report."""
        if not self.is_profiling and not self.profilers:
            return "No profiling data available."
        
        aggregated = self.get_aggregated_metrics()
        
        report = []
        report.append("🔬 PROFILING PERFORMANCE REPORT")
        report.append("=" * 50)
        
        if self.session_id:
            report.append(f"Session: {self.session_id}")
            if self.session_start_time:
                duration = time.time() - self.session_start_time
                report.append(f"Duration: {duration:.2f}s")
        
        report.append(f"Total Operations: {aggregated['total_operations']:,}")
        report.append(f"Total Time: {aggregated['total_time']:.3f}s")
        report.append(f"Memory Delta: {aggregated['total_memory_delta']:.1f}MB")
        
        # Component breakdown
        if aggregated['component_breakdown']:
            report.append("\n📊 Component Breakdown:")
            for comp, stats in aggregated['component_breakdown'].items():
                avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
                report.append(f"  {comp}: {stats['count']} ops, {stats['total_time']:.3f}s total, {avg_time:.4f}s avg")
        
        # Top operations
        if aggregated['operation_breakdown']:
            report.append("\n⏱️  Top Operations by Total Time:")
            sorted_ops = sorted(aggregated['operation_breakdown'].items(), 
                              key=lambda x: x[1]['total_time'], reverse=True)
            for op_name, stats in sorted_ops[:10]:
                report.append(f"  {op_name}: {stats['total_time']:.3f}s ({stats['count']} calls, {stats['avg_time']:.4f}s avg)")
        
        # Profiler overhead
        report.append("\n🔧 Profiler Overhead:")
        for profiler_name, profiler in self.profilers.items():
            overhead_percent = profiler.get_summary_stats().get('overhead_percentage', 0)
            report.append(f"  {profiler_name}: {overhead_percent:.2f}%")
        
        return "\n".join(report)
    
    def configure_profiler(self, profiler_name: str, config: ProfilerConfig):
        """Configure a specific profiler."""
        if profiler_name in self.profilers:
            self.profilers[profiler_name].config = config
            print(f"🔧 Configured profiler: {profiler_name}")
    
    def set_global_level(self, level: ProfilerLevel):
        """Set profiling level for all profilers."""
        self.global_config.level = level
        for profiler in self.profilers.values():
            profiler.config.level = level
        print(f"🔧 Set global profiling level: {level.name}")
    
    def enable_all(self):
        """Enable all profilers."""
        for profiler in self.profilers.values():
            profiler.config.enabled = True
        print("✅ Enabled all profilers")
    
    def disable_all(self):
        """Disable all profilers."""
        for profiler in self.profilers.values():
            profiler.config.enabled = False
        print("❌ Disabled all profilers")
    
    def save_all_results(self):
        """Save results from all profilers."""
        for profiler in self.profilers.values():
            profiler.save_results()
    
    def clear_all_results(self):
        """Clear results from all profilers."""
        for profiler in self.profilers.values():
            profiler.clear_results()
        print("🧹 Cleared all profiling results")
    
    def integrate_with_wandb(self, wandb_run):
        """Integrate profiling with Weights & Biases."""
        self.wandb_logger = wandb_run
        self.global_config.integrate_with_wandb = True
        print("🔗 Integrated with Weights & Biases")
    
    def integrate_with_logging(self, logger):
        """Integrate profiling with standard logging."""
        self.standard_logger = logger
        self.global_config.integrate_with_logging = True
        print("🔗 Integrated with standard logging")
    
    def _save_session_results(self, session_results: Dict[str, Any]):
        """Save session results to file."""
        if not self.global_config.auto_save:
            return
        
        output_dir = Path(self.global_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"session_{self.session_id}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(session_results, f, indent=2)
        
        print(f"💾 Session results saved to {filepath}")
        
        # Also save individual profiler results
        for profiler in self.profilers.values():
            profiler.save_results()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current profiling status."""
        return {
            'is_profiling': self.is_profiling,
            'session_id': self.session_id,
            'session_duration': time.time() - self.session_start_time if self.session_start_time else 0,
            'profilers_count': len(self.profilers),
            'profilers': {name: {
                'enabled': profiler.config.enabled,
                'level': profiler.config.level.name,
                'operations_count': len(profiler.operations),
                'overhead': profiler.average_overhead
            } for name, profiler in self.profilers.items()},
            'global_config': {
                'level': self.global_config.level.name,
                'profile_memory': self.global_config.profile_memory,
                'profile_compute': self.global_config.profile_compute,
                'auto_save': self.global_config.auto_save
            }
        }


class MultiProfilerContext:
    """Context manager for profiling across multiple profilers."""
    
    def __init__(self, 
                 manager: ProfilerManager,
                 operation_name: str,
                 component: str,
                 profiler_names: Optional[List[str]] = None,
                 tags: Optional[List[str]] = None):
        self.manager = manager
        self.operation_name = operation_name
        self.component = component
        self.profiler_names = profiler_names
        self.tags = tags
        self.operation_ids: Dict[str, str] = {}
    
    def __enter__(self):
        # Start operation in all applicable profilers
        profilers_to_use = (
            {name: self.manager.profilers[name] for name in self.profiler_names 
             if name in self.manager.profilers}
            if self.profiler_names
            else self.manager.profilers
        )
        
        for profiler_name, profiler in profilers_to_use.items():
            if profiler.is_enabled:
                operation_id = profiler.start_operation(
                    self.operation_name, 
                    self.component, 
                    self.tags
                )
                self.operation_ids[profiler_name] = operation_id
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # End operation in all profilers
        custom_metrics = {}
        if exc_type is not None:
            custom_metrics['exception'] = str(exc_type.__name__)
            custom_metrics['error'] = str(exc_val)
        
        for profiler_name, operation_id in self.operation_ids.items():
            if profiler_name in self.manager.profilers:
                self.manager.profilers[profiler_name].end_operation(
                    operation_id, **custom_metrics
                )
    
    def add_metric(self, name: str, value: Any):
        """Add a custom metric to all active operations."""
        for profiler_name, operation_id in self.operation_ids.items():
            if (profiler_name in self.manager.profilers and 
                operation_id in self.manager.profilers[profiler_name].active_operations):
                self.manager.profilers[profiler_name].active_operations[operation_id].custom_metrics[name] = value


# Global profiler manager instance
_global_profiler_manager: Optional[ProfilerManager] = None


def get_global_profiler_manager() -> ProfilerManager:
    """Get the global profiler manager instance."""
    global _global_profiler_manager
    if _global_profiler_manager is None:
        _global_profiler_manager = ProfilerManager()
    return _global_profiler_manager


def set_global_profiler_manager(manager: ProfilerManager):
    """Set the global profiler manager instance."""
    global _global_profiler_manager
    _global_profiler_manager = manager


def profile_operation(operation_name: str, 
                     component: str,
                     profiler_names: Optional[List[str]] = None,
                     tags: Optional[List[str]] = None):
    """
    Convenience function for profiling operations using the global manager.
    
    Usage:
        with profile_operation("my_operation", "evolution"):
            # code to profile
            pass
    """
    manager = get_global_profiler_manager()
    return manager.profile_operation(operation_name, component, profiler_names, tags)
