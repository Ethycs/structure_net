#!/usr/bin/env python3
"""
WandB Integration for Structure Net

This module provides comprehensive WandB logging for structure_net experiments,
including growth tracking, metric visualization, and automatic graphing.

Key features:
- Automatic experiment tracking
- Real-time metric logging
- Network architecture visualization
- Growth pattern analysis
- Comparative experiment dashboards
"""

import wandb
import torch
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..core.network_analysis import get_network_stats


class StructureNetWandBLogger:
    """
    Comprehensive WandB logger for structure_net experiments.
    
    Automatically logs metrics, visualizes growth patterns, and creates
    interactive dashboards for experiment analysis.
    """
    
    def __init__(self, 
                 project_name: str = "structure_net",
                 experiment_name: str = None,
                 config: Dict[str, Any] = None,
                 tags: List[str] = None):
        """
        Initialize WandB logger.
        
        Args:
            project_name: WandB project name
            experiment_name: Experiment name (auto-generated if None)
            config: Experiment configuration
            tags: Experiment tags
        """
        self.project_name = project_name
        self.experiment_name = experiment_name or f"structure_net_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize WandB with reinit to handle multiple runs
        self.run = wandb.init(
            project=project_name,
            name=self.experiment_name,
            config=config or {},
            tags=tags or [],
            reinit=True  # Allow multiple runs in same process
        )
        
        # Track experiment state
        self.iteration = 0
        self.epoch = 0
        self.growth_events = []
        self.architecture_history = []
        
        print(f"🔗 WandB logging initialized: {self.run.url}")
    
    def log_experiment_start(self, 
                           network: torch.nn.Module,
                           experiment_type: str,
                           target_accuracy: float = None,
                           seed_architecture: List[int] = None):
        """Log experiment initialization."""
        
        # Get initial network stats
        stats = get_network_stats(network)
        
        # Update config
        wandb.config.update({
            "experiment_type": experiment_type,
            "target_accuracy": target_accuracy,
            "seed_architecture": seed_architecture or stats['architecture'],
            "initial_connections": stats['total_connections'],
            "initial_sparsity": stats['overall_sparsity'],
            "initial_parameters": sum(p.numel() for p in network.parameters()),
            "device": str(next(network.parameters()).device)
        })
        
        # Log initial architecture
        self.log_architecture(stats['architecture'], iteration=0)
        
        # Create architecture visualization
        self._create_architecture_visualization(stats['architecture'])
        
        print(f"📊 Logged experiment start: {experiment_type}")
    
    def log_growth_iteration(self,
                           iteration: int,
                           network: torch.nn.Module,
                           accuracy: float,
                           loss: float = None,
                           extrema_ratio: float = None,
                           growth_occurred: bool = False,
                           growth_actions: List[Dict] = None,
                           learning_rates: Dict = None):
        """Log a complete growth iteration."""
        
        self.iteration = iteration
        
        # Get network stats
        stats = get_network_stats(network)
        
        # Core metrics
        metrics = {
            "iteration": iteration,
            "accuracy": accuracy,
            "architecture_depth": len(stats['architecture']),
            "total_connections": stats['total_connections'],
            "sparsity": stats['overall_sparsity'],
            "total_parameters": sum(p.numel() for p in network.parameters()),
            "growth_occurred": growth_occurred
        }
        
        # Optional metrics
        if loss is not None:
            metrics["loss"] = loss
        if extrema_ratio is not None:
            metrics["extrema_ratio"] = extrema_ratio
        
        # Log to WandB
        wandb.log(metrics, step=iteration)
        
        # Log architecture changes
        if growth_occurred:
            self.log_architecture(stats['architecture'], iteration)
            
            # Log growth actions
            if growth_actions:
                self.log_growth_actions(growth_actions, iteration)
        
        # Log learning rates if provided
        if learning_rates:
            self.log_learning_rates(learning_rates, iteration)
        
        # Track growth events
        if growth_occurred:
            self.growth_events.append({
                "iteration": iteration,
                "accuracy": accuracy,
                "architecture": stats['architecture'],
                "actions": growth_actions or []
            })
        
        print(f"📈 Logged iteration {iteration}: acc={accuracy:.2%}, growth={growth_occurred}")
    
    def log_training_epoch(self,
                          epoch: int,
                          train_loss: float,
                          train_acc: float,
                          val_loss: float = None,
                          val_acc: float = None,
                          learning_rate: float = None):
        """Log training epoch metrics."""
        
        self.epoch = epoch
        
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc
        }
        
        if val_loss is not None:
            metrics["val_loss"] = val_loss
        if val_acc is not None:
            metrics["val_accuracy"] = val_acc
        if learning_rate is not None:
            metrics["learning_rate"] = learning_rate
        
        wandb.log(metrics, step=epoch)
    
    def log_architecture(self, architecture: List[int], iteration: int):
        """Log network architecture with visualization."""
        
        # Store architecture history
        self.architecture_history.append({
            "iteration": iteration,
            "architecture": architecture,
            "depth": len(architecture),
            "total_neurons": sum(architecture)
        })
        
        # Log architecture metrics
        wandb.log({
            "architecture_depth": len(architecture),
            "total_neurons": sum(architecture),
            "largest_layer": max(architecture),
            "smallest_layer": min(architecture),
            "architecture_string": str(architecture)
        }, step=iteration)
        
        # Create architecture visualization
        self._create_architecture_visualization(architecture, iteration)
    
    def log_growth_actions(self, actions: List[Dict], iteration: int):
        """Log growth actions taken."""
        
        for i, action in enumerate(actions):
            action_metrics = {
                f"action_{i}_type": action.get('action', 'unknown'),
                f"action_{i}_reason": action.get('reason', 'N/A')
            }
            
            # Add action-specific metrics
            if 'position' in action:
                action_metrics[f"action_{i}_position"] = action['position']
            if 'size' in action:
                action_metrics[f"action_{i}_size"] = action['size']
            if 'count' in action:
                action_metrics[f"action_{i}_count"] = action['count']
            
            wandb.log(action_metrics, step=iteration)
    
    def log_learning_rates(self, learning_rates: Dict, iteration: int):
        """Log adaptive learning rate information."""
        
        lr_metrics = {}
        
        # Log strategy information
        if 'strategies' in learning_rates:
            for strategy_name, strategy_data in learning_rates['strategies'].items():
                prefix = f"lr_{strategy_name}"
                
                for key, value in strategy_data.items():
                    if isinstance(value, (int, float)):
                        lr_metrics[f"{prefix}_{key}"] = value
                    elif isinstance(value, str):
                        lr_metrics[f"{prefix}_{key}"] = value
        
        # Log base learning rate
        if 'base_lr' in learning_rates:
            lr_metrics["base_learning_rate"] = learning_rates['base_lr']
        
        wandb.log(lr_metrics, step=iteration)
    
    def log_tournament_results(self, 
                             tournament_results: Dict,
                             iteration: int):
        """Log tournament-based growth results."""
        
        winner = tournament_results['winner']
        all_results = tournament_results['all_results']
        
        # Log winner information
        wandb.log({
            "tournament_winner": winner['strategy'],
            "tournament_improvement": winner['improvement'],
            "tournament_final_accuracy": winner['final_accuracy']
        }, step=iteration)
        
        # Log all strategy performances
        for result in all_results:
            strategy_name = result['strategy'].replace(' ', '_').lower()
            wandb.log({
                f"tournament_{strategy_name}_improvement": result['improvement'],
                f"tournament_{strategy_name}_accuracy": result['final_accuracy']
            }, step=iteration)
        
        # Create tournament comparison chart
        self._create_tournament_visualization(all_results, iteration)
    
    def log_residual_block_performance(self,
                                     block_type: str,
                                     performance_metrics: Dict,
                                     iteration: int):
        """Log residual block specific metrics."""
        
        prefix = f"residual_{block_type}"
        
        for metric_name, value in performance_metrics.items():
            wandb.log({f"{prefix}_{metric_name}": value}, step=iteration)
    
    def log_extrema_analysis(self,
                           extrema_analysis: Dict,
                           iteration: int):
        """Log extrema detection analysis."""
        
        # Log overall extrema metrics
        wandb.log({
            "extrema_total": extrema_analysis.get('total_extrema', 0),
            "extrema_ratio": extrema_analysis.get('extrema_ratio', 0.0)
        }, step=iteration)
        
        # Log layer-wise extrema
        if 'layer_health' in extrema_analysis:
            for layer_idx, health in extrema_analysis['layer_health'].items():
                wandb.log({f"layer_{layer_idx}_health": health}, step=iteration)
        
        # Create extrema visualization
        self._create_extrema_visualization(extrema_analysis, iteration)
    
    def log_json_experiment(self, json_path: str):
        """
        Import and log an existing JSON experiment file.
        
        This converts JSON experiment data to WandB format with full visualization.
        Handles multiple data formats: time-series arrays, summary objects, and growth experiments.
        """
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Detect file type and handle accordingly
        file_type = self._detect_json_file_type(data, json_path)
        
        if file_type == "time_series":
            self._log_smart_time_series(data, json_path)
        elif file_type == "experiment_summary":
            self._log_smart_experiment_summary(data, json_path)
        else:
            # Fallback for unknown formats
            self._log_generic_data(data, json_path)
        
        print(f"📊 Imported {file_type}: {json_path}")
    
    def _detect_json_file_type(self, data, json_path: str):
        """Smart detection: just time-series arrays vs experiment summaries."""
        if isinstance(data, list):
            return "time_series"
        elif isinstance(data, dict):
            return "experiment_summary"
        else:
            return "unknown"
    
    def _log_smart_time_series(self, data: List[Dict], json_path: str):
        """Smart handler for ANY time-series JSON array - automatically detects and logs all metrics."""
        
        filename = Path(json_path).name.lower()
        
        # Update config
        wandb.config.update({
            "experiment_type": f"time_series_{filename.replace('.json', '').replace('_', '')}",
            "total_records": len(data),
            "imported_from_json": True,
            "source_file": Path(json_path).name
        }, allow_val_change=True)
        
        # Automatically detect step field (epoch, iteration, etc.)
        step_field = self._detect_step_field(data)
        
        # Log each record
        for i, record in enumerate(data):
            step = record.get(step_field, i)
            
            # Automatically extract ALL numeric fields
            metrics = self._extract_all_metrics(record, step_field)
            
            if metrics:
                wandb.log(metrics, step=step)
        
        # Create automatic visualizations
        self._create_smart_visualizations(data, step_field)
    
    def _log_smart_experiment_summary(self, data: Dict, json_path: str):
        """Smart handler for ANY experiment summary dict - automatically extracts all data."""
        
        filename = Path(json_path).name.lower()
        
        # Update config with all top-level data
        config_data = {
            "experiment_type": f"summary_{filename.replace('.json', '').replace('_', '')}",
            "imported_from_json": True,
            "source_file": Path(json_path).name
        }
        
        # Recursively extract all config-worthy data
        config_data.update(self._extract_config_data(data))
        wandb.config.update(config_data, allow_val_change=True)
        
        # Log all numeric metrics
        metrics = self._extract_all_metrics(data)
        if metrics:
            wandb.log(metrics)
        
        # Handle nested time-series data (like performance_history in hybrid files)
        self._handle_nested_time_series(data)
    
    def _detect_step_field(self, data: List[Dict]) -> str:
        """Automatically detect the step field (epoch, iteration, step, etc.)."""
        if not data:
            return "step"
        
        first_record = data[0]
        step_candidates = ["epoch", "iteration", "step", "time", "frame"]
        
        for candidate in step_candidates:
            if candidate in first_record:
                return candidate
        
        return "step"  # Default
    
    def _extract_all_metrics(self, data: Dict, exclude_field: str = None) -> Dict:
        """Recursively extract all numeric metrics from a dict."""
        metrics = {}
        
        def extract_recursive(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == exclude_field:
                        continue
                    
                    new_prefix = f"{prefix}_{key}" if prefix else key
                    
                    if isinstance(value, (int, float)):
                        metrics[new_prefix] = value
                    elif isinstance(value, dict):
                        extract_recursive(value, new_prefix)
                    elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                        # Handle lists of numbers (like architecture)
                        for i, item in enumerate(value):
                            if isinstance(item, (int, float)):
                                metrics[f"{new_prefix}_{i}"] = item
        
        extract_recursive(data)
        return metrics
    
    def _extract_config_data(self, data: Dict) -> Dict:
        """Extract configuration data (non-time-series) from dict."""
        config = {}
        
        def extract_config_recursive(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}_{key}" if prefix else key
                    
                    if isinstance(value, (str, int, float, bool)):
                        config[new_prefix] = value
                    elif isinstance(value, list) and len(value) < 20:  # Small lists only
                        if all(isinstance(x, (int, float)) for x in value):
                            config[new_prefix] = value
                    elif isinstance(value, dict) and len(value) < 10:  # Small dicts only
                        extract_config_recursive(value, new_prefix)
        
        extract_config_recursive(data)
        return config
    
    def _handle_nested_time_series(self, data: Dict):
        """Handle nested time-series data in summary files."""
        time_series_keys = ["performance_history", "growth_events", "growth_history", "training_log"]
        
        for key in time_series_keys:
            if key in data and isinstance(data[key], list):
                # Log nested time-series
                step_field = self._detect_step_field(data[key])
                
                for i, record in enumerate(data[key]):
                    step = record.get(step_field, i)
                    metrics = self._extract_all_metrics(record, step_field)
                    
                    # Prefix metrics to avoid conflicts
                    prefixed_metrics = {f"{key}_{k}": v for k, v in metrics.items()}
                    
                    if prefixed_metrics:
                        wandb.log(prefixed_metrics, step=step)
    
    def _create_smart_visualizations(self, data: List[Dict], step_field: str):
        """Create automatic visualizations for any time-series data."""
        if not data:
            return
        
        # Find all numeric fields
        all_fields = set()
        for record in data:
            metrics = self._extract_all_metrics(record, step_field)
            all_fields.update(metrics.keys())
        
        # Create visualization data
        viz_data = []
        for record in data:
            step = record.get(step_field, 0)
            metrics = self._extract_all_metrics(record, step_field)
            
            row = [step]
            for field in sorted(all_fields):
                row.append(metrics.get(field, 0))
            
            viz_data.append(row)
        
        # Create table and plots for key metrics
        columns = [step_field.title()] + [field.replace('_', ' ').title() for field in sorted(all_fields)]
        table = wandb.Table(data=viz_data, columns=columns)
        
        # Create plots for the most interesting metrics
        interesting_fields = [f for f in sorted(all_fields) if any(keyword in f.lower() 
                             for keyword in ['accuracy', 'loss', 'performance', 'count', 'ratio'])]
        
        for field in interesting_fields[:5]:  # Limit to 5 plots
            field_title = field.replace('_', ' ').title()
            wandb.log({
                f"{field}_over_time": wandb.plot.line(
                    table, step_field.title(), field_title,
                    title=f"{field_title} Over Time"
                )
            })
    
    def _log_performance_history(self, data: List[Dict], json_path: str):
        """Log performance history time-series data."""
        # Now just calls the smart handler
        self._log_smart_time_series(data, json_path)
    
    def _log_growth_events(self, data: List[Dict], json_path: str):
        """Log growth events time-series data."""
        # Update config
        wandb.config.update({
            "experiment_type": "growth_tracking",
            "total_growth_events": len(data),
            "imported_from_json": True,
            "source_file": Path(json_path).name
        }, allow_val_change=True)
        
        # Log each growth event
        for i, record in enumerate(data):
            epoch = record.get('epoch', i)
            
            metrics = {
                "connections_added": record.get('connections_added'),
                "total_connections": record.get('total_connections'),
                "growth_phase": record.get('phase'),
                "performance": record.get('performance'),
                "loss": record.get('loss'),
                "epoch": epoch
            }
            
            # Add growth stats if available
            if 'growth_stats' in record:
                growth_stats = record['growth_stats']
                if 'network_stats' in growth_stats:
                    net_stats = growth_stats['network_stats']
                    metrics.update({
                        "connectivity_ratio": net_stats.get('connectivity_ratio'),
                        "sparsity": net_stats.get('sparsity')
                    })
            
            # Remove None values
            metrics = {k: v for k, v in metrics.items() if v is not None}
            
            wandb.log(metrics, step=epoch)
    
    def _log_training_log(self, data: List[Dict], json_path: str):
        """Log training log time-series data."""
        # Similar to performance history but for training logs
        self._log_performance_history(data, json_path)
    
    def _log_hybrid_experiment_results(self, data: Dict, json_path: str):
        """Log hybrid experiment results with config, results, performance_history, and growth_events."""
        
        # Extract config information
        config_data = {
            "experiment_type": "hybrid_experiment_results",
            "imported_from_json": True,
            "source_file": Path(json_path).name
        }
        
        # Add config information
        if 'config' in data:
            config = data['config']
            config_data.update({
                "experiment_id": config.get('experiment_id'),
                "gpu_id": config.get('gpu_id'),
                "epochs": config.get('epochs'),
                "batch_size": config.get('batch_size'),
                "learning_rate": config.get('learning_rate'),
                "sparsity": config.get('sparsity'),
                "hidden_sizes": config.get('hidden_sizes'),
                "activation": config.get('activation'),
                "dataset_size": config.get('dataset_size'),
                "random_seed": config.get('random_seed')
            })
        
        # Add results summary
        if 'results' in data:
            results = data['results']
            config_data.update({
                "total_time": results.get('total_time'),
                "best_performance": results.get('best_performance'),
                "total_growth_events": results.get('total_growth_events'),
                "final_connections": results.get('final_connections')
            })
            
            # Log final results
            wandb.log({
                "final_best_performance": results.get('best_performance'),
                "final_total_connections": results.get('final_connections'),
                "experiment_total_time": results.get('total_time'),
                "total_growth_events": results.get('total_growth_events')
            })
        
        # Update config
        wandb.config.update(config_data, allow_val_change=True)
        
        # Log performance history if available
        if 'performance_history' in data:
            perf_history = data['performance_history']
            for record in perf_history:
                epoch = record.get('epoch', 0)
                
                metrics = {
                    "train_loss": record.get('train_loss'),
                    "train_accuracy": record.get('train_acc'),
                    "test_loss": record.get('test_loss'),
                    "test_accuracy": record.get('test_acc'),
                    "connections": record.get('connections'),
                    "epoch": epoch
                }
                
                # Remove None values
                metrics = {k: v for k, v in metrics.items() if v is not None}
                
                wandb.log(metrics, step=epoch)
            
            # Create performance visualizations
            self._create_performance_visualizations(perf_history)
        
        # Log growth events if available
        if 'growth_events' in data:
            growth_events = data['growth_events']
            for i, event in enumerate(growth_events):
                epoch = event.get('epoch', i)
                
                growth_metrics = {
                    "growth_connections_added": event.get('connections_added'),
                    "growth_total_connections": event.get('total_connections'),
                    "growth_performance": event.get('performance'),
                    "growth_event": 1  # Mark that a growth event occurred
                }
                
                # Remove None values
                growth_metrics = {k: v for k, v in growth_metrics.items() if v is not None}
                
                wandb.log(growth_metrics, step=epoch)
    
    def _log_extrema_evolution_data(self, data: List[Dict], json_path: str):
        """Log extrema evolution time-series data with detailed activation statistics."""
        
        # Update config
        wandb.config.update({
            "experiment_type": "extrema_evolution_tracking",
            "total_epochs": len(data),
            "imported_from_json": True,
            "source_file": Path(json_path).name
        }, allow_val_change=True)
        
        # Log each epoch's extrema data
        for record in data:
            epoch = record.get('epoch', 0)
            
            # Core extrema metrics
            metrics = {
                "adaptive_extrema_count": record.get('adaptive_extrema_count'),
                "lenient_extrema_count": record.get('lenient_extrema_count'),
                "strict_extrema_count": record.get('strict_extrema_count'),
                "epoch": epoch
            }
            
            # Log activation statistics per layer
            if 'activation_stats' in record:
                for layer_stats in record['activation_stats']:
                    layer_idx = layer_stats['layer']
                    layer_prefix = f"layer_{layer_idx}"
                    
                    metrics.update({
                        f"{layer_prefix}_mean": layer_stats.get('mean'),
                        f"{layer_prefix}_std": layer_stats.get('std'),
                        f"{layer_prefix}_min": layer_stats.get('min'),
                        f"{layer_prefix}_max": layer_stats.get('max'),
                        f"{layer_prefix}_range": layer_stats.get('range'),
                        f"{layer_prefix}_saturation_ratio": layer_stats.get('saturation_ratio')
                    })
            
            # Remove None values
            metrics = {k: v for k, v in metrics.items() if v is not None}
            
            wandb.log(metrics, step=epoch)
        
        # Create extrema evolution visualizations
        self._create_extrema_evolution_visualizations(data)
    
    def _create_extrema_evolution_visualizations(self, data: List[Dict]):
        """Create visualizations for extrema evolution data."""
        if not data:
            return
        
        # Prepare data for visualization
        extrema_data = []
        layer_stats_data = []
        
        for record in data:
            epoch = record.get('epoch', 0)
            
            # Extrema counts over time
            extrema_data.append([
                epoch,
                record.get('adaptive_extrema_count', 0),
                record.get('lenient_extrema_count', 0),
                record.get('strict_extrema_count', 0)
            ])
            
            # Layer statistics over time
            if 'activation_stats' in record:
                for layer_stats in record['activation_stats']:
                    layer_stats_data.append([
                        epoch,
                        layer_stats['layer'],
                        layer_stats.get('mean', 0),
                        layer_stats.get('std', 0),
                        layer_stats.get('saturation_ratio', 0)
                    ])
        
        # Extrema counts visualization
        extrema_table = wandb.Table(
            data=extrema_data,
            columns=["Epoch", "Adaptive_Extrema", "Lenient_Extrema", "Strict_Extrema"]
        )
        
        wandb.log({
            "extrema_counts_over_time": wandb.plot.line(
                extrema_table, "Epoch", "Adaptive_Extrema",
                title="Adaptive Extrema Count Over Time"
            )
        })
        
        # Layer statistics visualization
        if layer_stats_data:
            layer_table = wandb.Table(
                data=layer_stats_data,
                columns=["Epoch", "Layer", "Mean_Activation", "Std_Activation", "Saturation_Ratio"]
            )
            
            wandb.log({
                "layer_saturation_over_time": wandb.plot.line(
                    layer_table, "Epoch", "Saturation_Ratio",
                    title="Layer Saturation Ratio Over Time"
                )
            })
    
    def _log_experiment_summary(self, data: Dict, json_path: str):
        """Log experiment summary data."""
        # Update config
        config_data = {
            "experiment_type": "experiment_summary",
            "imported_from_json": True,
            "source_file": Path(json_path).name
        }
        
        # Extract summary information
        if 'experiment_summary' in data:
            summary = data['experiment_summary']
            config_data.update({
                "total_epochs": summary.get('total_epochs'),
                "growth_events": summary.get('growth_events'),
                "snapshots_created": summary.get('snapshots_created')
            })
            
            # Log final performance if available
            if 'final_performance' in summary:
                final_perf = summary['final_performance']
                wandb.log({
                    "final_train_accuracy": final_perf.get('train_acc'),
                    "final_test_accuracy": final_perf.get('test_acc'),
                    "final_train_loss": final_perf.get('train_loss'),
                    "final_test_loss": final_perf.get('test_loss'),
                    "final_connections": final_perf.get('connections')
                })
        
        # Log network stats if available
        if 'network_stats' in data:
            net_stats = data['network_stats']
            wandb.log({
                "total_active_connections": net_stats.get('total_active_connections'),
                "connectivity_ratio": net_stats.get('connectivity_ratio'),
                "sparsity": net_stats.get('sparsity')
            })
        
        wandb.config.update(config_data, allow_val_change=True)
    
    def _log_growth_experiment(self, data: Dict, json_path: str):
        """Log growth experiment data (original format)."""
        # Update config with experiment info
        wandb.config.update({
            "experiment_type": data.get('experiment_type', 'growth_experiment'),
            "seed_architecture": data.get('seed_architecture', []),
            "scaffold_sparsity": data.get('scaffold_sparsity', 0.02),
            "final_accuracy": data.get('final_accuracy', 0.0),
            "growth_iterations": data.get('growth_iterations', 0),
            "imported_from_json": True,
            "json_timestamp": data.get('timestamp', 'unknown'),
            "source_file": Path(json_path).name
        }, allow_val_change=True)
        
        # Log growth history
        if 'growth_history' in data:
            for record in data['growth_history']:
                iteration = record['iteration']
                
                # Core metrics
                metrics = {
                    "iteration": iteration,
                    "accuracy": record['accuracy'],
                    "total_connections": record.get('total_connections'),
                    "sparsity": record.get('sparsity'),
                    "extrema_ratio": record.get('extrema_ratio'),
                    "growth_occurred": record.get('growth_occurred'),
                    "architecture_depth": len(record.get('architecture', [])),
                    "total_neurons": sum(record.get('architecture', []))
                }
                
                # Remove None values
                metrics = {k: v for k, v in metrics.items() if v is not None}
                
                wandb.log(metrics, step=iteration)
                
                # Log architecture
                if 'architecture' in record:
                    self.log_architecture(record['architecture'], iteration)
        
        # Create summary visualizations
        self._create_json_summary_visualizations(data)
    
    def _log_generic_data(self, data, json_path: str):
        """Fallback for unknown data formats."""
        wandb.config.update({
            "experiment_type": "generic_import",
            "imported_from_json": True,
            "source_file": Path(json_path).name,
            "data_type": type(data).__name__
        }, allow_val_change=True)
        
        # Try to extract any numeric data for logging
        if isinstance(data, dict):
            numeric_data = {}
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    numeric_data[key] = value
            
            if numeric_data:
                wandb.log(numeric_data)
    
    def _create_performance_visualizations(self, data: List[Dict]):
        """Create visualizations for performance history data."""
        if not data:
            return
        
        # Prepare data for visualization
        perf_data = []
        for record in data:
            perf_data.append([
                record.get('epoch', 0),
                record.get('train_acc', 0),
                record.get('test_acc', 0),
                record.get('train_loss', 0),
                record.get('test_loss', 0),
                record.get('connections', 0)
            ])
        
        table = wandb.Table(
            data=perf_data,
            columns=["Epoch", "Train_Acc", "Test_Acc", "Train_Loss", "Test_Loss", "Connections"]
        )
        
        # Accuracy plot
        wandb.log({
            "accuracy_over_time": wandb.plot.line(
                table, "Epoch", "Test_Acc",
                title="Test Accuracy Over Time"
            )
        })
        
        # Loss plot
        wandb.log({
            "loss_over_time": wandb.plot.line(
                table, "Epoch", "Test_Loss",
                title="Test Loss Over Time"
            )
        })
        
        # Connections plot
        wandb.log({
            "connections_over_time": wandb.plot.line(
                table, "Epoch", "Connections",
                title="Network Connections Over Time"
            )
        })
    
    def _create_architecture_visualization(self, architecture: List[int], iteration: int = 0):
        """Create architecture visualization."""
        
        # Create bar chart of layer sizes
        layer_data = [[i, size] for i, size in enumerate(architecture)]
        
        table = wandb.Table(data=layer_data, columns=["Layer", "Size"])
        
        wandb.log({
            f"architecture_chart_iter_{iteration}": wandb.plot.bar(
                table, "Layer", "Size", 
                title=f"Network Architecture (Iteration {iteration})"
            )
        }, step=iteration)
    
    def _create_tournament_visualization(self, results: List[Dict], iteration: int):
        """Create tournament results visualization."""
        
        # Prepare data for visualization
        strategy_data = []
        for result in results:
            strategy_data.append([
                result['strategy'],
                result['improvement'],
                result['final_accuracy']
            ])
        
        table = wandb.Table(
            data=strategy_data, 
            columns=["Strategy", "Improvement", "Final_Accuracy"]
        )
        
        # Create improvement comparison
        wandb.log({
            f"tournament_improvement_iter_{iteration}": wandb.plot.bar(
                table, "Strategy", "Improvement",
                title=f"Tournament Strategy Improvements (Iteration {iteration})"
            )
        }, step=iteration)
        
        # Create accuracy comparison
        wandb.log({
            f"tournament_accuracy_iter_{iteration}": wandb.plot.bar(
                table, "Strategy", "Final_Accuracy",
                title=f"Tournament Strategy Accuracies (Iteration {iteration})"
            )
        }, step=iteration)
    
    def _create_extrema_visualization(self, extrema_analysis: Dict, iteration: int):
        """Create extrema analysis visualization."""
        
        if 'layer_health' in extrema_analysis:
            health_data = []
            for layer_idx, health in extrema_analysis['layer_health'].items():
                health_data.append([layer_idx, health])
            
            table = wandb.Table(data=health_data, columns=["Layer", "Health"])
            
            wandb.log({
                f"layer_health_iter_{iteration}": wandb.plot.bar(
                    table, "Layer", "Health",
                    title=f"Layer Health (Iteration {iteration})"
                )
            }, step=iteration)
    
    def _create_json_summary_visualizations(self, data: Dict):
        """Create summary visualizations from JSON data."""
        
        if 'growth_history' not in data:
            return
        
        # Growth trajectory
        growth_data = []
        for record in data['growth_history']:
            growth_data.append([
                record['iteration'],
                record['accuracy'],
                record['total_connections'],
                record['sparsity']
            ])
        
        table = wandb.Table(
            data=growth_data,
            columns=["Iteration", "Accuracy", "Connections", "Sparsity"]
        )
        
        # Accuracy trajectory
        wandb.log({
            "accuracy_trajectory": wandb.plot.line(
                table, "Iteration", "Accuracy",
                title="Accuracy Growth Trajectory"
            )
        })
        
        # Connection growth
        wandb.log({
            "connection_growth": wandb.plot.line(
                table, "Iteration", "Connections",
                title="Connection Growth Over Time"
            )
        })
        
        # Sparsity evolution
        wandb.log({
            "sparsity_evolution": wandb.plot.line(
                table, "Iteration", "Sparsity",
                title="Sparsity Evolution"
            )
        })
    
    def create_comparative_dashboard(self, experiment_names: List[str]):
        """Create comparative dashboard for multiple experiments."""
        
        # This would typically be done in WandB UI, but we can log comparison metrics
        wandb.log({
            "comparative_experiments": len(experiment_names),
            "experiment_list": experiment_names
        })
        
        print(f"📊 Comparative dashboard setup for {len(experiment_names)} experiments")
    
    def finish_experiment(self, summary_metrics: Dict = None):
        """Finish the experiment and log final summary."""
        
        if summary_metrics:
            wandb.summary.update(summary_metrics)
        
        # Log experiment summary
        wandb.summary.update({
            "total_iterations": self.iteration,
            "total_epochs": self.epoch,
            "growth_events": len(self.growth_events),
            "architecture_changes": len(self.architecture_history)
        })
        
        print(f"✅ Experiment finished: {self.run.url}")
        wandb.finish()


def convert_json_to_wandb(json_path: str, 
                         project_name: str = "structure_net",
                         experiment_name: str = None):
    """
    Convert a JSON experiment file to WandB with full visualization.
    
    Args:
        json_path: Path to JSON experiment file
        project_name: WandB project name
        experiment_name: Custom experiment name
    """
    
    # Extract experiment name from JSON if not provided
    if experiment_name is None:
        json_file = Path(json_path)
        # Create unique experiment name with parent directory for context
        parent_dir = json_file.parent.name
        experiment_name = f"{parent_dir}_{json_file.stem}"
    
    # Initialize logger with reinit to avoid conflicts
    logger = StructureNetWandBLogger(
        project_name=project_name,
        experiment_name=experiment_name,
        tags=["imported", "json_conversion", Path(json_path).parent.name]
    )
    
    try:
        # Import and log the JSON data
        logger.log_json_experiment(json_path)
        
        # Finish the experiment
        logger.finish_experiment({
            "imported_from": json_path,
            "conversion_timestamp": datetime.now().isoformat()
        })
        
        return logger.run.url
        
    except Exception as e:
        # Ensure we finish the run even if there's an error
        logger.finish_experiment({
            "error": str(e),
            "imported_from": json_path,
            "conversion_timestamp": datetime.now().isoformat()
        })
        raise e


def setup_wandb_for_modern_indefinite_growth(config: Dict = None):
    """
    Setup WandB logger specifically for modern indefinite growth experiments.
    
    Returns configured logger ready for growth tracking.
    """
    
    default_config = {
        "experiment_type": "modern_indefinite_growth",
        "framework": "structure_net",
        "growth_strategy": "extrema_driven",
        "adaptive_learning_rates": True,
        "residual_blocks": True
    }
    
    if config:
        default_config.update(config)
    
    logger = StructureNetWandBLogger(
        project_name="structure_net_growth",
        experiment_name=f"modern_growth_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=default_config,
        tags=["modern_growth", "extrema_driven", "adaptive_lr"]
    )
    
    return logger


# Export all components
__all__ = [
    'StructureNetWandBLogger',
    'convert_json_to_wandb',
    'setup_wandb_for_modern_indefinite_growth'
]
