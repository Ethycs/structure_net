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
from .standardized_logger import (
    StandardizedLogger,
    create_growth_logger,
    create_training_logger,
    create_tournament_logger,
    create_profiling_logger
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

# Import argument parser
from .argument_parser import add_logging_arguments

# Export all public components
__all__ = [
    # Main logger
    'StandardizedLogger',
    'create_growth_logger',
    'create_training_logger',
    'create_tournament_logger',
    'create_profiling_logger',
    
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

    # Argument parsing
    'add_logging_arguments'
]

# Version info
__version__ = "1.0.0"
