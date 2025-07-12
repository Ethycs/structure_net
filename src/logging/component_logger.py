#!/usr/bin/env python3
"""
Component-Based Logger for Structure Net

Bridges the component schemas with the existing standardized logging system,
providing strict validation while maintaining backwards compatibility.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import json

from .standardized_logging import StandardizedLogger, LoggingConfig
from .component_schemas import (
    ExperimentComposition,
    ExperimentExecution,
    ExperimentTemplate,
    IterationData,
    MetricSchema,
    EvolverSchema,
    ModelSchema,
    TrainerSchema,
    NALSchema,
    validate_component_compatibility,
    STANDARD_TEMPLATES
)


class ComponentLogger:
    """
    Logger that enforces component-based schema validation.
    
    Wraps StandardizedLogger to provide:
    1. Component-based experiment composition
    2. Strict schema validation for each layer
    3. Template-based experiment setup
    4. Automatic logging of component interactions
    """
    
    def __init__(self, config: LoggingConfig = None):
        """Initialize component logger with standard logger backend."""
        self.standard_logger = StandardizedLogger(config or LoggingConfig())
        self.active_executions: Dict[str, ExperimentExecution] = {}
        
    def create_experiment_from_template(self, 
                                      template_name: str,
                                      execution_id: str,
                                      **customizations) -> ExperimentExecution:
        """
        Create an experiment execution from a template.
        
        Args:
            template_name: Name of template in STANDARD_TEMPLATES
            execution_id: Unique ID for this execution
            **customizations: Parameter overrides
            
        Returns:
            ExperimentExecution ready to run
        """
        if template_name not in STANDARD_TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")
            
        template = STANDARD_TEMPLATES[template_name]
        composition = template.instantiate(**customizations)
        
        execution = ExperimentExecution(
            execution_id=execution_id,
            composition=composition
        )
        
        self.active_executions[execution_id] = execution
        
        # Log composition to ChromaDB for searchability
        self._log_composition(execution)
        
        return execution
    
    def create_experiment_from_components(self,
                                        execution_id: str,
                                        metric: MetricSchema,
                                        evolver: EvolverSchema,
                                        model: ModelSchema,
                                        trainer: TrainerSchema,
                                        nal: NALSchema,
                                        name: str = None) -> ExperimentExecution:
        """
        Create an experiment execution from individual components.
        
        Args:
            execution_id: Unique ID for this execution
            metric: Metric component
            evolver: Evolver component
            model: Model component
            trainer: Trainer component
            nal: NAL component
            name: Optional name for the composition
            
        Returns:
            ExperimentExecution ready to run
        """
        composition = ExperimentComposition(
            composition_id=f"comp_{execution_id}",
            name=name or f"Custom composition for {execution_id}",
            metric=metric,
            evolver=evolver,
            model=model,
            trainer=trainer,
            nal=nal
        )
        
        # Validate compatibility
        warnings = validate_component_compatibility(composition)
        if warnings:
            for warning in warnings:
                print(f"⚠️  Component compatibility warning: {warning}")
        
        execution = ExperimentExecution(
            execution_id=execution_id,
            composition=composition
        )
        
        self.active_executions[execution_id] = execution
        self._log_composition(execution)
        
        return execution
    
    def log_iteration(self, 
                     execution_id: str,
                     iteration: int,
                     metric_outputs: Dict[str, Any],
                     trainer_metrics: Dict[str, Any],
                     accuracy: float,
                     loss: float,
                     evolver_actions: List[str] = None,
                     model_changes: Dict[str, Any] = None,
                     nal_decisions: Dict[str, Any] = None):
        """
        Log one iteration of experiment execution.
        
        This is the main logging method called during training.
        """
        if execution_id not in self.active_executions:
            raise ValueError(f"No active execution with ID: {execution_id}")
            
        execution = self.active_executions[execution_id]
        
        iteration_data = IterationData(
            iteration=iteration,
            metric_outputs=metric_outputs,
            evolver_actions=evolver_actions or [],
            model_changes=model_changes or {},
            trainer_metrics=trainer_metrics,
            nal_decisions=nal_decisions or {},
            accuracy=accuracy,
            loss=loss
        )
        
        execution.add_iteration(iteration_data)
        
        # Also log to standard logger for real-time monitoring
        self.standard_logger.log_metrics(
            experiment_id=execution_id,
            metrics={
                'accuracy': accuracy,
                'loss': loss,
                'epoch': iteration,
                **trainer_metrics
            }
        )
    
    def finalize_experiment(self,
                          execution_id: str,
                          final_metrics: Dict[str, Any],
                          status: str = "completed",
                          error: str = None):
        """
        Finalize an experiment execution and create WandB artifact.
        
        Args:
            execution_id: Execution to finalize
            final_metrics: Final performance metrics
            status: Final status (completed/failed/cancelled)
            error: Error message if failed
        """
        if execution_id not in self.active_executions:
            raise ValueError(f"No active execution with ID: {execution_id}")
            
        execution = self.active_executions[execution_id]
        execution.final_metrics = final_metrics
        execution.finalize(status=status, error=error)
        
        # Convert to artifact-ready format
        artifact_data = self._execution_to_artifact_format(execution)
        
        # Log as experiment result for WandB artifact creation
        result_hash = self.standard_logger.log_experiment_result(artifact_data)
        
        # Update ChromaDB with final status
        self.standard_logger.update_experiment_status(
            execution_id,
            status,
            **final_metrics
        )
        
        # Clean up
        del self.active_executions[execution_id]
        
        return result_hash
    
    def _log_composition(self, execution: ExperimentExecution):
        """Log composition details to ChromaDB."""
        comp = execution.composition
        
        # Register in ChromaDB with component details
        self.standard_logger.register_experiment_start(
            experiment_id=execution.execution_id,
            hypothesis_id=comp.nal.hypothesis,
            architecture=str(comp.model.architecture),
            metric_type=comp.metric.metric_name,
            evolver_type=comp.evolver.evolver_name,
            model_type=comp.model.model_name,
            trainer_type=comp.trainer.trainer_name,
            composition_hash=comp.generate_hash()
        )
    
    def _execution_to_artifact_format(self, execution: ExperimentExecution) -> Dict[str, Any]:
        """Convert execution to format expected by StandardizedLogger."""
        comp = execution.composition
        
        # Build artifact data following the component structure
        artifact_data = {
            'experiment_id': execution.execution_id,
            'timestamp': execution.started_at.isoformat(),
            'schema_version': '2.0',  # Component-based schema version
            
            # Composition details
            'composition': {
                'id': comp.composition_id,
                'name': comp.name,
                'template': comp.template_name,
                'hash': comp.generate_hash()
            },
            
            # Component configurations
            'components': {
                'metric': comp.metric.model_dump(),
                'evolver': comp.evolver.model_dump(),
                'model': comp.model.model_dump(),
                'trainer': comp.trainer.model_dump(),
                'nal': comp.nal.model_dump()
            },
            
            # Execution data
            'execution': {
                'status': execution.status,
                'started_at': execution.started_at.isoformat(),
                'completed_at': execution.completed_at.isoformat() if execution.completed_at else None,
                'execution_time': execution.execution_time,
                'error': execution.error
            },
            
            # Results
            'results': {
                'final_metrics': execution.final_metrics,
                'iteration_count': len(execution.iteration_log),
                'peak_accuracy': max(i.accuracy for i in execution.iteration_log) if execution.iteration_log else 0
            },
            
            # Full iteration history (can be large)
            'iteration_log': [iter_data.model_dump() for iter_data in execution.iteration_log]
        }
        
        # Add NAL hypothesis result
        if execution.status == "completed":
            success_criteria = comp.nal.success_criteria
            hypothesis_result = all(
                execution.final_metrics.get(metric, 0) >= threshold
                for metric, threshold in success_criteria.items()
            )
            artifact_data['results']['hypothesis_confirmed'] = hypothesis_result
        
        return artifact_data
    
    def search_experiments_by_component(self, 
                                      component_type: str,
                                      component_name: str,
                                      limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for experiments using specific components.
        
        Args:
            component_type: Type of component (metric/evolver/model/trainer/nal)
            component_name: Name of the specific component
            limit: Maximum results to return
            
        Returns:
            List of matching experiments
        """
        # This would integrate with ChromaDB search
        # For now, delegating to standard logger's search
        query = f"{component_type}:{component_name}"
        return self.standard_logger.searcher.search_by_description(query, limit=limit)


# Convenience functions for common use cases

def create_evolution_experiment(hypothesis: str,
                              architecture: List[int],
                              execution_id: str = None,
                              **kwargs) -> ComponentLogger:
    """
    Quick setup for evolution experiments using the standard template.
    
    Args:
        hypothesis: What you're testing
        architecture: Initial network architecture
        execution_id: Unique ID (auto-generated if None)
        **kwargs: Additional customizations
        
    Returns:
        Configured ComponentLogger with active experiment
    """
    logger = ComponentLogger()
    
    if execution_id is None:
        execution_id = f"evo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Customize the template
    customizations = {
        'model.architecture': architecture,
        'nal.hypothesis': hypothesis,
        **kwargs
    }
    
    execution = logger.create_experiment_from_template(
        'architecture_evolution',
        execution_id,
        **customizations
    )
    
    return logger, execution


def create_custom_experiment(metric_name: str,
                           evolver_name: str,
                           model_name: str,
                           trainer_name: str,
                           hypothesis: str,
                           architecture: List[int],
                           execution_id: str = None) -> ComponentLogger:
    """
    Create a fully custom experiment from component names.
    
    This is a simplified interface that creates components with default configs.
    """
    logger = ComponentLogger()
    
    if execution_id is None:
        execution_id = f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create components with sensible defaults
    metric = MetricSchema(
        component_id=f"metric_{execution_id}",
        metric_name=metric_name,
        outputs=["score"],  # Generic output
        config={}
    )
    
    evolver = EvolverSchema(
        component_id=f"evolver_{execution_id}",
        evolver_name=evolver_name,
        inputs=["score"],
        outputs=["modify"],
        config={}
    )
    
    model = ModelSchema(
        component_id=f"model_{execution_id}",
        model_name=model_name,
        architecture=architecture,
        total_parameters=sum(architecture[i] * architecture[i+1] for i in range(len(architecture)-1)),
        sparsity=0.0,
        config={}
    )
    
    trainer = TrainerSchema(
        component_id=f"trainer_{execution_id}",
        trainer_name=trainer_name,
        optimizer="adam",
        learning_rate=0.001,
        batch_size=128,
        config={}
    )
    
    nal = NALSchema(
        component_id=f"nal_{execution_id}",
        hypothesis=hypothesis,
        success_criteria={"accuracy": 0.9},
        config={}
    )
    
    execution = logger.create_experiment_from_components(
        execution_id=execution_id,
        metric=metric,
        evolver=evolver,
        model=model,
        trainer=trainer,
        nal=nal
    )
    
    return logger, execution