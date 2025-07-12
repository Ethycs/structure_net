#!/usr/bin/env python3
"""
Structure Net Logging System

A comprehensive logging system that adopts the WandB artifact standard with
Pydantic validation to ensure data consistency and reliability.

Key features:
- Standardized logging with Pydantic validation
- WandB artifact-first approach
- Local-first with queue system
- Automatic schema migration
- Offline-safe operation
- Real-time monitoring + persistent artifacts
"""

# Import main components
from .standardized_logging import (
    StandardizedLogger,
    LoggingConfig,
    initialize_logging,
    get_logger,
    log_experiment,
    log_metrics,
    log_growth_event
)

# Import config adapter to enable unified config support
from . import config_adapter

# Import component-based logging
from .component_logger import (
    ComponentLogger,
    create_evolution_experiment,
    create_custom_experiment
)

from .component_schemas import (
    # Component schemas
    MetricSchema,
    EvolverSchema,
    ModelSchema,
    TrainerSchema,
    NALSchema,
    # Composition schemas
    ExperimentComposition,
    ExperimentExecution,
    ExperimentTemplate,
    IterationData,
    # Templates
    STANDARD_TEMPLATES,
    # Utilities
    validate_component_compatibility
)

from .artifact_manager import (
    ArtifactManager,
    ArtifactConfig,
    ArtifactUploader,
    queue_experiment,
    process_queue,
    get_queue_status
)

from .schemas import (
    # Base schemas
    BaseExperimentSchema,
    NetworkArchitecture,
    PerformanceMetrics,
    ExtremaAnalysis,
    GrowthAction,
    TrainingEpoch,
    GrowthIteration,
    TournamentResult,
    ExperimentConfig,
    
    # Experiment schemas
    GrowthExperiment,
    TrainingExperiment,
    TournamentExperiment,
    ComparativeExperiment,
    ExperimentSummary,
    ProfilingExperiment,
    ProfilingResultSchema,
    
    # Utilities
    validate_experiment_data,
    migrate_legacy_data
)

# Keep backward compatibility with existing WandB integration
from .wandb_integration import (
    StructureNetWandBLogger,
    convert_json_to_wandb,
    setup_wandb_for_modern_indefinite_growth
)

# Import standardized logger functions
from .standardized_logger import (
    create_growth_logger,
    create_training_logger,
    create_tournament_logger,
    create_profiling_logger
)

# Import argument parser
from .argument_parser import add_logging_arguments

# Export all public components
__all__ = [
    # Main logger
    'StandardizedLogger',
    'LoggingConfig',
    'initialize_logging',
    'get_logger',
    'log_experiment',
    'log_metrics', 
    'log_growth_event',
    
    # Component-based logging
    'ComponentLogger',
    'create_evolution_experiment',
    'create_custom_experiment',
    
    # Component schemas
    'MetricSchema',
    'EvolverSchema', 
    'ModelSchema',
    'TrainerSchema',
    'NALSchema',
    'ExperimentComposition',
    'ExperimentExecution',
    'ExperimentTemplate',
    'IterationData',
    'STANDARD_TEMPLATES',
    'validate_component_compatibility',
    
    # Artifact management
    'ArtifactManager',
    'ArtifactConfig',
    'ArtifactUploader',
    'queue_experiment',
    'process_queue',
    'get_queue_status',
    
    # Schemas
    'BaseExperimentSchema',
    'NetworkArchitecture',
    'PerformanceMetrics',
    'ExtremaAnalysis',
    'GrowthAction',
    'TrainingEpoch',
    'GrowthIteration',
    'TournamentResult',
    'ExperimentConfig',
    'GrowthExperiment',
    'TrainingExperiment',
    'TournamentExperiment',
    'ComparativeExperiment',
    'ExperimentSummary',
    'ProfilingExperiment',
    'ProfilingResultSchema',
    'validate_experiment_data',
    'migrate_legacy_data',
    
    # Backward compatibility
    'StructureNetWandBLogger',
    'convert_json_to_wandb',
    'setup_wandb_for_modern_indefinite_growth',
    
    # Standardized logger functions
    'create_growth_logger',
    'create_training_logger',
    'create_tournament_logger',
    'create_profiling_logger',

    # Argument parsing
    'add_logging_arguments'
]

# Version info
__version__ = "1.0.0"
