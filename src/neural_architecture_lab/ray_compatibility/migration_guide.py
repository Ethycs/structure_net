"""
Migration guide and helper functions for transitioning from custom runners to Ray.

This module provides utilities to help migrate existing code with minimal changes.
"""

import os
import sys
from typing import Dict, Any, Optional, Type
from functools import wraps

# Import Ray compatibility layer
from . import (
    RayExperimentRunner,
    RayAdvancedRunner,
    create_experiment_runner,
    migrate_existing_runner
)


class CompatibilityPatcher:
    """
    Monkey-patches existing imports to use Ray compatibility layer.
    
    Usage:
        # At the start of your script:
        from neural_architecture_lab.ray_compatibility import CompatibilityPatcher
        CompatibilityPatcher.patch_imports()
        
        # Now existing imports will use Ray versions automatically
        from neural_architecture_lab.runners import ExperimentRunner  # Gets Ray version
    """
    
    @staticmethod
    def patch_imports():
        """Patch module imports to use Ray compatibility versions."""
        # Save original modules
        original_modules = {}
        
        # Patch runners module
        if 'neural_architecture_lab.runners' in sys.modules:
            original_modules['runners'] = sys.modules['neural_architecture_lab.runners']
        
        if 'neural_architecture_lab.advanced_runners' in sys.modules:
            original_modules['advanced_runners'] = sys.modules['neural_architecture_lab.advanced_runners']
        
        # Create patched module
        class PatchedRunners:
            """Patched runners module that returns Ray versions."""
            
            # Main runner classes
            ExperimentRunner = RayExperimentRunner
            AdvancedExperimentRunner = RayAdvancedRunner
            
            # Worker functions
            run_structure_net_experiment = staticmethod(
                __import__('neural_architecture_lab.ray_compatibility', 
                         fromlist=['run_structure_net_experiment']).run_structure_net_experiment
            )
            
            # Resource management
            GPUMemoryManager = __import__('neural_architecture_lab.ray_compatibility',
                                        fromlist=['RayGPUMemoryManager']).RayGPUMemoryManager
            
            @staticmethod
            def __getattr__(name):
                """Fallback to Ray compatibility layer."""
                try:
                    return getattr(
                        __import__('neural_architecture_lab.ray_compatibility', fromlist=[name]),
                        name
                    )
                except AttributeError:
                    # Fall back to original module if available
                    if 'runners' in original_modules:
                        return getattr(original_modules['runners'], name)
                    raise
        
        # Replace modules
        sys.modules['neural_architecture_lab.runners'] = PatchedRunners
        sys.modules['neural_architecture_lab.advanced_runners'] = PatchedRunners


def migrate_worker_function(original_worker_fn):
    """
    Decorator to make existing worker functions Ray-compatible.
    
    Usage:
        @migrate_worker_function
        def my_worker(experiment: Experiment, device_id: int) -> ExperimentResult:
            # Original worker code
            ...
    """
    @wraps(original_worker_fn)
    def ray_compatible_worker(experiment, device_id=0):
        # Ray handles device assignment, but we maintain compatibility
        if hasattr(experiment, 'to_dict'):
            # If experiment has serialization method, use it
            experiment_dict = experiment.to_dict()
        else:
            experiment_dict = experiment
        
        # Call original function
        result = original_worker_fn(experiment, device_id)
        
        # Ensure result is serializable for Ray
        if hasattr(result, 'to_dict'):
            return result.to_dict()
        return result
    
    # Mark as Ray-compatible
    ray_compatible_worker._ray_compatible = True
    return ray_compatible_worker


def create_migration_config(existing_lab_config: Optional[Any] = None) -> Dict[str, Any]:
    """
    Create Ray configuration from existing LabConfig.
    
    Args:
        existing_lab_config: Existing configuration object
        
    Returns:
        Ray-compatible configuration dictionary
    """
    if existing_lab_config is None:
        return {
            'num_cpus': os.cpu_count(),
            'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    
    ray_config = {}
    
    # Map common configuration fields
    field_mappings = {
        'max_cpus': 'num_cpus',
        'num_gpus': 'num_gpus',
        'max_workers': 'max_concurrent_trials',
        'memory_limit': 'object_store_memory',
    }
    
    for old_field, new_field in field_mappings.items():
        if hasattr(existing_lab_config, old_field):
            ray_config[new_field] = getattr(existing_lab_config, old_field)
    
    return ray_config


# Example migration script
MIGRATION_TEMPLATE = '''
#!/usr/bin/env python3
"""
Migrated script using Ray compatibility layer.

Original script: {original_script}
Migration date: {date}
"""

# Step 1: Import compatibility patcher
from src.neural_architecture_lab.ray_compatibility import CompatibilityPatcher

# Step 2: Patch imports (do this before other NAL imports)
CompatibilityPatcher.patch_imports()

# Step 3: Rest of your imports remain the same
{original_imports}

# Step 4: Your code remains largely the same
# The compatibility layer handles the Ray integration transparently

if __name__ == "__main__":
    # Original code continues here...
    {main_code}
'''


def generate_migration_script(original_script_path: str) -> str:
    """
    Generate a migrated version of an existing script.
    
    Args:
        original_script_path: Path to the original script
        
    Returns:
        Migrated script content
    """
    import datetime
    
    with open(original_script_path, 'r') as f:
        original_content = f.read()
    
    # Extract imports and main code (simplified)
    lines = original_content.split('\n')
    import_lines = []
    code_lines = []
    
    in_imports = True
    for line in lines:
        if in_imports and (line.startswith('import ') or line.startswith('from ')):
            import_lines.append(line)
        else:
            in_imports = False
            code_lines.append(line)
    
    return MIGRATION_TEMPLATE.format(
        original_script=original_script_path,
        date=datetime.datetime.now().strftime('%Y-%m-%d'),
        original_imports='\n'.join(import_lines),
        main_code='\n'.join(code_lines)
    )


# Environment variable for automatic patching
if os.environ.get('NAL_USE_RAY', '').lower() == 'true':
    CompatibilityPatcher.patch_imports()


# Usage examples
def example_minimal_changes():
    """Example: Minimal changes to use Ray"""
    # Original code:
    # from neural_architecture_lab.runners import ExperimentRunner
    # runner = ExperimentRunner(lab_config)
    # results = runner.run_experiments(experiments)
    
    # With compatibility layer (Option 1 - explicit):
    from src.neural_architecture_lab.ray_compatibility import RayExperimentRunner
    runner = RayExperimentRunner(lab_config)
    results = runner.run_experiments(experiments)
    
    # With compatibility layer (Option 2 - patched):
    CompatibilityPatcher.patch_imports()
    from src.neural_architecture_lab.runners import ExperimentRunner
    runner = ExperimentRunner(lab_config)  # Actually creates RayExperimentRunner
    results = runner.run_experiments(experiments)


def example_advanced_migration():
    """Example: Migrating advanced features"""
    # Original code with custom worker pool:
    # from neural_architecture_lab.advanced_runners import GPUWorkerPool
    # pool = GPUWorkerPool(num_workers=4, devices=[0,1,2,3])
    # results = pool.run_experiments(experiments)
    
    # With Ray compatibility:
    from src.neural_architecture_lab.ray_compatibility import RayWorkerPool
    pool = RayWorkerPool(num_workers=4, worker_fn=run_structure_net_experiment, devices=[0,1,2,3])
    results = pool.submit_experiments(experiments)


if __name__ == "__main__":
    # Quick test of compatibility layer
    print("Ray compatibility layer loaded successfully")
    print("Set NAL_USE_RAY=true to automatically patch imports")