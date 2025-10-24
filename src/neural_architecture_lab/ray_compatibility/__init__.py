"""
Ray compatibility layer for Neural Architecture Lab.

This module provides drop-in replacements for existing NAL components
using Ray for distributed execution while maintaining backward compatibility.
"""

from .compat_layer import (
    RayExperimentRunner,
    RayTuneExperimentRunner,
    RayGPUMemoryManager,
    create_experiment_runner,
    run_structure_net_experiment,
    run_seed_search_task,
    run_tournament_task,
    ray_experiment_worker
)

from .advanced_compat import (
    RayAdvancedRunner,
    RayWorkerPool,
    RayAsyncGPURunner,
    migrate_existing_runner
)

__all__ = [
    # Main runners
    'RayExperimentRunner',
    'RayTuneExperimentRunner', 
    'RayAdvancedRunner',
    
    # Resource management
    'RayGPUMemoryManager',
    'RayWorkerPool',
    'RayAsyncGPURunner',
    
    # Factory functions
    'create_experiment_runner',
    'migrate_existing_runner',
    
    # Worker compatibility functions
    'run_structure_net_experiment',
    'run_seed_search_task', 
    'run_tournament_task',
    'ray_experiment_worker'
]