#!/usr/bin/env python3
"""
Standardized Logger for Structure Net

This module provides an enhanced logger that integrates Pydantic validation
with the WandB artifact system, ensuring all experiment data is validated
before logging and uploaded as artifacts.

Key features:
- Artifact-first logging approach
- Automatic schema validation
- Backward compatibility with existing WandBLogger
- Local-first with queue system
- Real-time metrics + artifact persistence
- Schema migration support
"""

import torch
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

import wandb
from pydantic import ValidationError

from .schemas import (
    GrowthExperiment,
    TrainingExperiment,
    TournamentExperiment,
    NetworkArchitecture,
    PerformanceMetrics,
    ExtremaAnalysis,
    GrowthIteration,
    TrainingEpoch,
    GrowthAction,
    TournamentResult,
    ExperimentConfig,
    validate_experiment_data
)
from .artifact_manager import ArtifactManager, ArtifactConfig
from .wandb_integration import StructureNetWandBLogger


class StandardizedLogger:
    """
    Enhanced logger that combines real-time WandB logging with artifact persistence.
    
    This logger provides:
    1. Immediate WandB logging for real-time monitoring
    2. Validated artifact persistence for data integrity
    3. Automatic schema validation and migration
    4. Offline-safe operation with queue system
    """
    
    def __init__(self,
                 project_name: str = "structure_net",
                 experiment_name: str = None,
                 experiment_type: str = "growth_experiment",
                 config: Dict[str, Any] = None,
                 tags: List[str] = None,
                 artifact_config: ArtifactConfig = None):
        """
        Initialize standardized logger.
        
        Args:
            project_name: WandB project name
            experiment_name: Experiment name (auto-generated if None)
            experiment_type: Type of experiment for schema validation
            config: Experiment configuration
            tags: Experiment tags
            artifact_config: Artifact manager configuration
        """
        self.project_name = project_name
        self.experiment_type = experiment_type
        self.experiment_name = experiment_name or f"{experiment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize WandB logger for real-time logging
        self.wandb_logger = StructureNetWandBLogger(
            project_name=project_name,
            experiment_name=self.experiment_name,
            config=config or {},
            tags=tags or []
        )
        
        # Initialize artifact manager for validated persistence
        self.artifact_manager = ArtifactManager(artifact_config)
        
        # Experiment data accumulation
        self.experiment_data = {
            'experiment_id': self.experiment_name,
            'experiment_type': experiment_type,
            'timestamp': datetime.now(),
            'config': self._create_experiment_config(config or {}),
            'growth_history': [],
            'training_history': [],
            'tournament_results': [],
            'final_performance': None
        }
        
        # Track current state
        self.current_iteration = 0
        self.current_epoch = 0
        
        print(f"🔗 StandardizedLogger initialized: {self.wandb_logger.run.url}")
        print(f"📦 Artifact queue: {self.artifact_manager.config.queue_dir}")
    
    def _create_experiment_config(self, config: Dict[str, Any]) -> ExperimentConfig:
        """Create validated experiment configuration."""
        # Set defaults for required fields
        config_defaults = {
            'dataset': config.get('dataset', 'unknown'),
            'batch_size': config.get('batch_size', 64),
            'learning_rate': config.get('learning_rate', 0.001),
            'max_epochs': config.get('max_epochs', 100),
            'device': config.get('device', 'cpu'),
            'random_seed': config.get('random_seed'),
            'target_accuracy': config.get('target_accuracy')
        }
        
        try:
            return ExperimentConfig(**config_defaults)
        except ValidationError as e:
            print(f"⚠️  Config validation failed, using defaults: {e}")
            return ExperimentConfig(
                dataset='unknown',
                batch_size=64,
                learning_rate=0.001,
                max_epochs=100,
                device='cpu'
            )
    
    def _create_network_architecture(self, network: torch.nn.Module) -> NetworkArchitecture:
        """Create validated network architecture from PyTorch model."""
        from ..core.network_analysis import get_network_stats
        
        stats = get_network_stats(network)
        
        return NetworkArchitecture(
            layers=stats['architecture'],
            total_parameters=sum(p.numel() for p in network.parameters()),
            total_connections=stats['total_connections'],
            sparsity=stats['overall_sparsity'],
            depth=len(stats['architecture'])
        )
    
    def _create_performance_metrics(self, 
                                  accuracy: float,
                                  loss: float = None,
                                  precision: float = None,
                                  recall: float = None,
                                  f1_score: float = None) -> PerformanceMetrics:
        """Create validated performance metrics."""
        return PerformanceMetrics(
            accuracy=accuracy,
            loss=loss,
            precision=precision,
            recall=recall,
            f1_score=f1_score
        )
    
    def _create_extrema_analysis(self, extrema_data: Dict[str, Any]) -> ExtremaAnalysis:
        """Create validated extrema analysis."""
        return ExtremaAnalysis(
            total_extrema=extrema_data.get('total_extrema', 0),
            extrema_ratio=extrema_data.get('extrema_ratio', 0.0),
            dead_neurons=extrema_data.get('dead_neurons', {}),
            saturated_neurons=extrema_data.get('saturated_neurons', {}),
            layer_health=extrema_data.get('layer_health', {})
        )
    
    def log_experiment_start(self,
                           network: torch.nn.Module,
                           target_accuracy: float = None,
                           seed_architecture: List[int] = None):
        """Log experiment initialization with validation."""
        
        # Real-time WandB logging
        self.wandb_logger.log_experiment_start(
            network=network,
            experiment_type=self.experiment_type,
            target_accuracy=target_accuracy,
            seed_architecture=seed_architecture
        )
        
        # Update experiment data
        self.experiment_data.update({
            'seed_architecture': seed_architecture or self._create_network_architecture(network).layers,
            'scaffold_sparsity': getattr(network, 'sparsity', 0.02)
        })
        
        if target_accuracy:
            self.experiment_data['config'].target_accuracy = target_accuracy
        
        print(f"📊 Experiment started: {self.experiment_type}")
    
    def log_growth_iteration(self,
                           iteration: int,
                           network: torch.nn.Module,
                           accuracy: float,
                           loss: float = None,
                           extrema_analysis: Dict[str, Any] = None,
                           growth_actions: List[Dict] = None,
                           growth_occurred: bool = False,
                           credits: float = None):
        """Log growth iteration with validation."""
        
        self.current_iteration = iteration
        
        # Real-time WandB logging
        self.wandb_logger.log_growth_iteration(
            iteration=iteration,
            network=network,
            accuracy=accuracy,
            loss=loss,
            extrema_ratio=extrema_analysis.get('extrema_ratio') if extrema_analysis else None,
            growth_occurred=growth_occurred,
            growth_actions=growth_actions
        )
        
        # Create validated data structures
        architecture = self._create_network_architecture(network)
        performance = self._create_performance_metrics(accuracy, loss)
        
        validated_extrema = None
        if extrema_analysis:
            validated_extrema = self._create_extrema_analysis(extrema_analysis)
        
        validated_actions = []
        if growth_actions:
            for action in growth_actions:
                try:
                    validated_action = GrowthAction(
                        action_type=action.get('action', 'unknown'),
                        layer_position=action.get('position'),
                        size=action.get('size'),
                        reason=action.get('reason', 'No reason provided'),
                        success=action.get('success', True)
                    )
                    validated_actions.append(validated_action)
                except ValidationError as e:
                    print(f"⚠️  Invalid growth action: {e}")
        
        # Create growth iteration
        growth_iteration = GrowthIteration(
            iteration=iteration,
            architecture=architecture,
            performance=performance,
            extrema_analysis=validated_extrema,
            growth_actions=validated_actions,
            growth_occurred=growth_occurred,
            credits=credits
        )
        
        # Add to experiment data
        self.experiment_data['growth_history'].append(growth_iteration)
        
        print(f"📈 Growth iteration {iteration} logged: acc={accuracy:.2%}, growth={growth_occurred}")
    
    def log_training_epoch(self,
                          epoch: int,
                          train_loss: float,
                          train_acc: float,
                          val_loss: float = None,
                          val_acc: float = None,
                          learning_rate: float = None,
                          duration: float = None):
        """Log training epoch with validation."""
        
        self.current_epoch = epoch
        
        # Real-time WandB logging
        self.wandb_logger.log_training_epoch(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            learning_rate=learning_rate
        )
        
        # Create validated training epoch
        training_epoch = TrainingEpoch(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            learning_rate=learning_rate,
            duration=duration
        )
        
        # Add to experiment data
        self.experiment_data['training_history'].append(training_epoch)
    
    def log_tournament_results(self,
                             tournament_results: Dict[str, Any],
                             iteration: int):
        """Log tournament results with validation."""
        
        # Real-time WandB logging
        self.wandb_logger.log_tournament_results(tournament_results, iteration)
        
        # Validate tournament results
        validated_results = []
        for result in tournament_results.get('all_results', []):
            try:
                validated_result = TournamentResult(
                    strategy_name=result.get('strategy', 'unknown'),
                    improvement=result.get('improvement', 0.0),
                    final_accuracy=result.get('final_accuracy', 0.0),
                    execution_time=result.get('execution_time'),
                    success=result.get('success', True)
                )
                validated_results.append(validated_result)
            except ValidationError as e:
                print(f"⚠️  Invalid tournament result: {e}")
        
        # Add to experiment data
        self.experiment_data['tournament_results'].extend(validated_results)
        
        # Set winner if provided
        winner_data = tournament_results.get('winner')
        if winner_data:
            try:
                winner = TournamentResult(
                    strategy_name=winner_data.get('strategy', 'unknown'),
                    improvement=winner_data.get('improvement', 0.0),
                    final_accuracy=winner_data.get('final_accuracy', 0.0),
                    execution_time=winner_data.get('execution_time'),
                    success=winner_data.get('success', True)
                )
                self.experiment_data['winner'] = winner
            except ValidationError as e:
                print(f"⚠️  Invalid tournament winner: {e}")
    
    def log_final_performance(self,
                            accuracy: float,
                            loss: float = None,
                            precision: float = None,
                            recall: float = None,
                            f1_score: float = None):
        """Log final experiment performance."""
        
        final_performance = self._create_performance_metrics(
            accuracy=accuracy,
            loss=loss,
            precision=precision,
            recall=recall,
            f1_score=f1_score
        )
        
        self.experiment_data['final_performance'] = final_performance
        
        print(f"🎯 Final performance logged: {accuracy:.2%}")
    
    def save_experiment_artifact(self, force_upload: bool = False) -> str:
        """
        Save experiment as validated artifact.
        
        Args:
            force_upload: Force immediate upload instead of queuing
            
        Returns:
            Artifact hash ID
        """
        # Update totals
        self.experiment_data.update({
            'total_iterations': len(self.experiment_data['growth_history']),
            'total_epochs': len(self.experiment_data['training_history'])
        })
        
        # Convert to appropriate schema based on experiment type
        try:
            if self.experiment_type == "growth_experiment":
                validated_experiment = GrowthExperiment(**self.experiment_data)
            elif self.experiment_type == "training_experiment":
                validated_experiment = TrainingExperiment(**self.experiment_data)
            elif self.experiment_type == "tournament_experiment":
                validated_experiment = TournamentExperiment(**self.experiment_data)
            else:
                raise ValueError(f"Unknown experiment type: {self.experiment_type}")
            
            # Convert to dict for artifact manager
            artifact_data = validated_experiment.dict()
            
            # Queue for upload
            artifact_hash = self.artifact_manager.queue_experiment(
                data=artifact_data,
                experiment_id=self.experiment_name
            )
            
            # Force immediate upload if requested
            if force_upload:
                stats = self.artifact_manager.process_queue(max_files=1)
                print(f"📦 Immediate upload stats: {stats}")
            
            print(f"💾 Experiment artifact saved: {artifact_hash}")
            return artifact_hash
            
        except ValidationError as e:
            print(f"❌ Experiment validation failed: {e}")
            # Save raw data for debugging
            debug_file = Path("experiments/debug") / f"{self.experiment_name}_debug.json"
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(debug_file, 'w') as f:
                json.dump(self.experiment_data, f, indent=2, default=str)
            
            print(f"🐛 Debug data saved to: {debug_file}")
            raise
    
    def finish_experiment(self, 
                         final_accuracy: float = None,
                         summary_metrics: Dict[str, Any] = None,
                         save_artifact: bool = True) -> Optional[str]:
        """
        Finish experiment with final logging and artifact creation.
        
        Args:
            final_accuracy: Final accuracy achieved
            summary_metrics: Additional summary metrics
            save_artifact: Whether to save as artifact
            
        Returns:
            Artifact hash ID if saved, None otherwise
        """
        
        # Log final performance if provided
        if final_accuracy is not None:
            self.log_final_performance(accuracy=final_accuracy)
        
        # Finish WandB logging
        self.wandb_logger.finish_experiment(summary_metrics)
        
        # Save artifact if requested
        artifact_hash = None
        if save_artifact:
            try:
                artifact_hash = self.save_experiment_artifact()
            except Exception as e:
                print(f"⚠️  Failed to save artifact: {e}")
        
        print(f"✅ Experiment finished: {self.experiment_name}")
        if artifact_hash:
            print(f"📦 Artifact ID: {artifact_hash}")
        
        return artifact_hash
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get current experiment summary."""
        return {
            'experiment_id': self.experiment_name,
            'experiment_type': self.experiment_type,
            'total_iterations': len(self.experiment_data['growth_history']),
            'total_epochs': len(self.experiment_data['training_history']),
            'current_iteration': self.current_iteration,
            'current_epoch': self.current_epoch,
            'wandb_url': self.wandb_logger.run.url,
            'queue_status': self.artifact_manager.get_queue_status()
        }
    
    def validate_current_data(self) -> bool:
        """
        Validate current experiment data against schema.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Create temporary copy with totals
            temp_data = dict(self.experiment_data)
            temp_data.update({
                'total_iterations': len(temp_data['growth_history']),
                'total_epochs': len(temp_data['training_history'])
            })
            
            # Validate based on experiment type
            if self.experiment_type == "growth_experiment":
                GrowthExperiment(**temp_data)
            elif self.experiment_type == "training_experiment":
                TrainingExperiment(**temp_data)
            elif self.experiment_type == "tournament_experiment":
                TournamentExperiment(**temp_data)
            
            print("✅ Current experiment data is valid")
            return True
            
        except ValidationError as e:
            print(f"❌ Current experiment data is invalid: {e}")
            return False
    
    def process_artifact_queue(self) -> Dict[str, int]:
        """Process the artifact upload queue."""
        return self.artifact_manager.process_queue()
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get artifact queue status."""
        return self.artifact_manager.get_queue_status()


# Convenience functions for backward compatibility
def create_growth_logger(project_name: str = "structure_net",
                        experiment_name: str = None,
                        config: Dict[str, Any] = None,
                        tags: List[str] = None) -> StandardizedLogger:
    """Create a standardized logger for growth experiments."""
    return StandardizedLogger(
        project_name=project_name,
        experiment_name=experiment_name,
        experiment_type="growth_experiment",
        config=config,
        tags=tags
    )


def create_training_logger(project_name: str = "structure_net",
                          experiment_name: str = None,
                          config: Dict[str, Any] = None,
                          tags: List[str] = None) -> StandardizedLogger:
    """Create a standardized logger for training experiments."""
    return StandardizedLogger(
        project_name=project_name,
        experiment_name=experiment_name,
        experiment_type="training_experiment",
        config=config,
        tags=tags
    )


def create_tournament_logger(project_name: str = "structure_net",
                           experiment_name: str = None,
                           config: Dict[str, Any] = None,
                           tags: List[str] = None) -> StandardizedLogger:
    """Create a standardized logger for tournament experiments."""
    return StandardizedLogger(
        project_name=project_name,
        experiment_name=experiment_name,
        experiment_type="tournament_experiment",
        config=config,
        tags=tags
    )


def create_profiling_logger(project_name: str = "structure_net",
                          session_id: str = None,
                          config: Dict[str, Any] = None,
                          tags: List[str] = None) -> StandardizedLogger:
    """Create a standardized logger for profiling experiments."""
    tags = tags or []
    if 'profiling' not in tags:
        tags.append('profiling')
        
    return StandardizedLogger(
        project_name=project_name,
        experiment_name=session_id,
        experiment_type="profiling_experiment",
        config=config,
        tags=tags
    )


# Export main components
__all__ = [
    'StandardizedLogger',
    'create_growth_logger',
    'create_training_logger', 
    'create_tournament_logger',
    'create_profiling_logger'
]
