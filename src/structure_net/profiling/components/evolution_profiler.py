#!/usr/bin/env python3
"""
Evolution Component Profiler

Specialized profiler for network evolution operations including growth strategies,
analyzers, and evolution systems.
"""

import torch
import time
from typing import Dict, List, Any, Optional
import numpy as np

from ..core.base_profiler import BaseProfiler, ProfilerConfig


class EvolutionProfiler(BaseProfiler):
    """
    Specialized profiler for evolution components.
    
    Tracks evolution-specific metrics like growth events, strategy performance,
    analyzer execution times, and network architecture changes.
    """
    
    def __init__(self, config: Optional[ProfilerConfig] = None):
        config = config or ProfilerConfig()
        super().__init__("evolution_profiler", config)
        
        # Evolution-specific tracking
        self.growth_events = []
        self.strategy_performance = {}
        self.analyzer_performance = {}
        self.architecture_changes = []
        
        print("🧬 EvolutionProfiler initialized")
    
    def track_growth_event(self, 
                          strategy_name: str,
                          growth_type: str,
                          network_before: torch.nn.Module,
                          network_after: torch.nn.Module,
                          performance_improvement: float,
                          metadata: Optional[Dict[str, Any]] = None):
        """
        Track a network growth event.
        
        Args:
            strategy_name: Name of the growth strategy
            growth_type: Type of growth (add_layer, add_patches, etc.)
            network_before: Network state before growth
            network_after: Network state after growth
            performance_improvement: Performance improvement from growth
            metadata: Additional metadata about the growth event
        """
        if not self.is_enabled:
            return
        
        # Calculate network changes
        params_before = sum(p.numel() for p in network_before.parameters())
        params_after = sum(p.numel() for p in network_after.parameters())
        
        growth_event = {
            'timestamp': time.time(),
            'strategy_name': strategy_name,
            'growth_type': growth_type,
            'parameters_before': params_before,
            'parameters_after': params_after,
            'parameter_increase': params_after - params_before,
            'performance_improvement': performance_improvement,
            'metadata': metadata or {}
        }
        
        self.growth_events.append(growth_event)
        
        # Update strategy performance tracking
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                'total_applications': 0,
                'total_improvement': 0.0,
                'best_improvement': 0.0,
                'worst_improvement': float('inf'),
                'growth_types': set()
            }
        
        stats = self.strategy_performance[strategy_name]
        stats['total_applications'] += 1
        stats['total_improvement'] += performance_improvement
        stats['best_improvement'] = max(stats['best_improvement'], performance_improvement)
        stats['worst_improvement'] = min(stats['worst_improvement'], performance_improvement)
        stats['growth_types'].add(growth_type)
    
    def track_analyzer_execution(self,
                               analyzer_name: str,
                               execution_time: float,
                               analysis_results: Dict[str, Any],
                               network_size: int):
        """
        Track analyzer execution performance.
        
        Args:
            analyzer_name: Name of the analyzer
            execution_time: Time taken to execute
            analysis_results: Results from the analyzer
            network_size: Size of the network analyzed
        """
        if not self.is_enabled:
            return
        
        if analyzer_name not in self.analyzer_performance:
            self.analyzer_performance[analyzer_name] = {
                'total_executions': 0,
                'total_time': 0.0,
                'average_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0,
                'time_per_parameter': []
            }
        
        stats = self.analyzer_performance[analyzer_name]
        stats['total_executions'] += 1
        stats['total_time'] += execution_time
        stats['average_time'] = stats['total_time'] / stats['total_executions']
        stats['min_time'] = min(stats['min_time'], execution_time)
        stats['max_time'] = max(stats['max_time'], execution_time)
        
        if network_size > 0:
            time_per_param = execution_time / network_size
            stats['time_per_parameter'].append(time_per_param)
    
    def track_architecture_change(self,
                                 old_architecture: List[int],
                                 new_architecture: List[int],
                                 change_type: str,
                                 change_reason: str):
        """
        Track changes to network architecture.
        
        Args:
            old_architecture: Architecture before change
            new_architecture: Architecture after change
            change_type: Type of change (layer_addition, width_increase, etc.)
            change_reason: Reason for the change
        """
        if not self.is_enabled:
            return
        
        change_event = {
            'timestamp': time.time(),
            'old_architecture': old_architecture,
            'new_architecture': new_architecture,
            'change_type': change_type,
            'change_reason': change_reason,
            'depth_change': len(new_architecture) - len(old_architecture),
            'total_neurons_before': sum(old_architecture),
            'total_neurons_after': sum(new_architecture),
            'neuron_increase': sum(new_architecture) - sum(old_architecture)
        }
        
        self.architecture_changes.append(change_event)
    
    def get_specialized_metrics(self) -> Dict[str, Any]:
        """Get evolution-specific metrics."""
        metrics = {
            'growth_events': {
                'total_events': len(self.growth_events),
                'events_by_type': {},
                'events_by_strategy': {},
                'total_performance_improvement': 0.0,
                'average_performance_improvement': 0.0
            },
            'strategy_performance': {},
            'analyzer_performance': {},
            'architecture_changes': {
                'total_changes': len(self.architecture_changes),
                'changes_by_type': {},
                'total_depth_increase': 0,
                'total_neuron_increase': 0
            }
        }
        
        # Analyze growth events
        if self.growth_events:
            events_by_type = {}
            events_by_strategy = {}
            total_improvement = 0.0
            
            for event in self.growth_events:
                # By type
                growth_type = event['growth_type']
                if growth_type not in events_by_type:
                    events_by_type[growth_type] = 0
                events_by_type[growth_type] += 1
                
                # By strategy
                strategy = event['strategy_name']
                if strategy not in events_by_strategy:
                    events_by_strategy[strategy] = 0
                events_by_strategy[strategy] += 1
                
                total_improvement += event['performance_improvement']
            
            metrics['growth_events']['events_by_type'] = events_by_type
            metrics['growth_events']['events_by_strategy'] = events_by_strategy
            metrics['growth_events']['total_performance_improvement'] = total_improvement
            metrics['growth_events']['average_performance_improvement'] = total_improvement / len(self.growth_events)
        
        # Strategy performance summary
        for strategy_name, stats in self.strategy_performance.items():
            metrics['strategy_performance'][strategy_name] = {
                'total_applications': stats['total_applications'],
                'average_improvement': stats['total_improvement'] / stats['total_applications'],
                'best_improvement': stats['best_improvement'],
                'worst_improvement': stats['worst_improvement'],
                'growth_types_used': list(stats['growth_types'])
            }
        
        # Analyzer performance summary
        for analyzer_name, stats in self.analyzer_performance.items():
            analyzer_metrics = {
                'total_executions': stats['total_executions'],
                'average_time': stats['average_time'],
                'min_time': stats['min_time'],
                'max_time': stats['max_time']
            }
            
            if stats['time_per_parameter']:
                analyzer_metrics['average_time_per_parameter'] = np.mean(stats['time_per_parameter'])
                analyzer_metrics['time_per_parameter_std'] = np.std(stats['time_per_parameter'])
            
            metrics['analyzer_performance'][analyzer_name] = analyzer_metrics
        
        # Architecture changes summary
        if self.architecture_changes:
            changes_by_type = {}
            total_depth_increase = 0
            total_neuron_increase = 0
            
            for change in self.architecture_changes:
                change_type = change['change_type']
                if change_type not in changes_by_type:
                    changes_by_type[change_type] = 0
                changes_by_type[change_type] += 1
                
                total_depth_increase += change['depth_change']
                total_neuron_increase += change['neuron_increase']
            
            metrics['architecture_changes']['changes_by_type'] = changes_by_type
            metrics['architecture_changes']['total_depth_increase'] = total_depth_increase
            metrics['architecture_changes']['total_neuron_increase'] = total_neuron_increase
        
        return metrics
    
    def get_strategy_ranking(self) -> List[Dict[str, Any]]:
        """Get strategies ranked by performance."""
        rankings = []
        
        for strategy_name, stats in self.strategy_performance.items():
            avg_improvement = stats['total_improvement'] / stats['total_applications']
            rankings.append({
                'strategy_name': strategy_name,
                'average_improvement': avg_improvement,
                'total_applications': stats['total_applications'],
                'best_improvement': stats['best_improvement'],
                'consistency': 1.0 - (stats['best_improvement'] - stats['worst_improvement']) / max(stats['best_improvement'], 0.001)
            })
        
        # Sort by average improvement
        rankings.sort(key=lambda x: x['average_improvement'], reverse=True)
        return rankings
    
    def get_analyzer_efficiency_report(self) -> Dict[str, Any]:
        """Get analyzer efficiency analysis."""
        report = {
            'most_efficient': None,
            'least_efficient': None,
            'scalability_analysis': {}
        }
        
        if not self.analyzer_performance:
            return report
        
        # Find most and least efficient analyzers
        efficiency_scores = []
        for analyzer_name, stats in self.analyzer_performance.items():
            if stats['time_per_parameter']:
                avg_time_per_param = np.mean(stats['time_per_parameter'])
                efficiency_scores.append((analyzer_name, avg_time_per_param))
        
        if efficiency_scores:
            efficiency_scores.sort(key=lambda x: x[1])
            report['most_efficient'] = efficiency_scores[0][0]
            report['least_efficient'] = efficiency_scores[-1][0]
        
        # Scalability analysis
        for analyzer_name, stats in self.analyzer_performance.items():
            if len(stats['time_per_parameter']) > 1:
                times = np.array(stats['time_per_parameter'])
                # Simple trend analysis
                trend = np.polyfit(range(len(times)), times, 1)[0]
                report['scalability_analysis'][analyzer_name] = {
                    'trend': 'improving' if trend < 0 else 'degrading' if trend > 0 else 'stable',
                    'trend_slope': trend
                }
        
        return report
    
    def clear_results(self):
        """Clear all evolution-specific results."""
        super().clear_results()
        self.growth_events.clear()
        self.strategy_performance.clear()
        self.analyzer_performance.clear()
        self.architecture_changes.clear()
