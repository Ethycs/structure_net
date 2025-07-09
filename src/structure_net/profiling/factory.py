#!/usr/bin/env python3
"""
Profiler Factory Functions

Provides convenient factory functions for creating pre-configured profiler setups
for different use cases and requirements.
"""

from typing import Optional, List, Dict, Any
from .core.profiler_manager import ProfilerManager
from .core.base_profiler import ProfilerConfig, ProfilerLevel

# Import available profilers
try:
    from .components.evolution_profiler import EvolutionProfiler
except ImportError:
    EvolutionProfiler = None


def create_standard_profiler(level: ProfilerLevel = ProfilerLevel.BASIC,
                           output_dir: str = "profiling_results",
                           enable_memory: bool = True,
                           enable_compute: bool = False) -> ProfilerManager:
    """
    Create a standard profiler setup suitable for most use cases.
    
    Args:
        level: Profiling detail level
        output_dir: Directory for saving results
        enable_memory: Whether to profile memory usage
        enable_compute: Whether to profile GPU compute metrics
        
    Returns:
        Configured ProfilerManager with standard profilers
    """
    config = ProfilerConfig(
        level=level,
        profile_memory=enable_memory,
        profile_compute=enable_compute,
        output_dir=output_dir,
        auto_save=True,
        save_interval=50
    )
    
    manager = ProfilerManager(config)
    
    # Add evolution profiler if available
    if EvolutionProfiler is not None:
        evolution_profiler = EvolutionProfiler(config)
        manager.register_profiler(evolution_profiler)
    
    print(f"📊 Created standard profiler (level: {level.name})")
    return manager


def create_lightweight_profiler(output_dir: str = "profiling_results") -> ProfilerManager:
    """
    Create a lightweight profiler with minimal overhead.
    
    Args:
        output_dir: Directory for saving results
        
    Returns:
        Configured ProfilerManager with minimal profiling
    """
    config = ProfilerConfig(
        level=ProfilerLevel.BASIC,
        profile_memory=False,
        profile_compute=False,
        profile_io=False,
        output_dir=output_dir,
        auto_save=True,
        save_interval=100,
        max_overhead_percent=1.0,  # Very low overhead tolerance
        adaptive_sampling=True
    )
    
    manager = ProfilerManager(config)
    
    # Add only essential profilers if available
    if EvolutionProfiler is not None:
        evolution_profiler = EvolutionProfiler(config)
        manager.register_profiler(evolution_profiler)
    
    print("🪶 Created lightweight profiler (minimal overhead)")
    return manager


def create_comprehensive_profiler(output_dir: str = "profiling_results",
                                enable_wandb: bool = False) -> ProfilerManager:
    """
    Create a comprehensive profiler with all features enabled.
    
    Args:
        output_dir: Directory for saving results
        enable_wandb: Whether to integrate with Weights & Biases
        
    Returns:
        Configured ProfilerManager with comprehensive profiling
    """
    config = ProfilerConfig(
        level=ProfilerLevel.COMPREHENSIVE,
        profile_memory=True,
        profile_compute=True,
        profile_io=True,
        output_dir=output_dir,
        auto_save=True,
        save_interval=25,
        integrate_with_wandb=enable_wandb,
        integrate_with_logging=True,
        max_overhead_percent=10.0,  # Higher overhead tolerance
        adaptive_sampling=True
    )
    
    manager = ProfilerManager(config)
    
    # Add all available profilers
    if EvolutionProfiler is not None:
        evolution_profiler = EvolutionProfiler(config)
        manager.register_profiler(evolution_profiler)
    
    # TODO: Add other profilers when implemented
    # metrics_profiler = MetricsProfiler(config)
    # manager.register_profiler(metrics_profiler)
    # 
    # network_profiler = NetworkProfiler(config)
    # manager.register_profiler(network_profiler)
    # 
    # training_profiler = TrainingProfiler(config)
    # manager.register_profiler(training_profiler)
    
    print("🔬 Created comprehensive profiler (all features enabled)")
    return manager


def create_custom_profiler(profiler_configs: Dict[str, ProfilerConfig],
                          global_config: Optional[ProfilerConfig] = None) -> ProfilerManager:
    """
    Create a custom profiler with specific configurations for each profiler type.
    
    Args:
        profiler_configs: Dictionary mapping profiler names to their configs
        global_config: Global configuration (optional)
        
    Returns:
        Configured ProfilerManager with custom profilers
    """
    manager = ProfilerManager(global_config)
    
    for profiler_name, config in profiler_configs.items():
        if profiler_name == "evolution":
            profiler = EvolutionProfiler(config)
            manager.register_profiler(profiler)
        # TODO: Add other profiler types
        # elif profiler_name == "metrics":
        #     profiler = MetricsProfiler(config)
        #     manager.register_profiler(profiler)
        # elif profiler_name == "network":
        #     profiler = NetworkProfiler(config)
        #     manager.register_profiler(profiler)
        # elif profiler_name == "training":
        #     profiler = TrainingProfiler(config)
        #     manager.register_profiler(profiler)
        else:
            print(f"⚠️  Unknown profiler type: {profiler_name}")
    
    print(f"🔧 Created custom profiler with {len(profiler_configs)} profilers")
    return manager


def create_evolution_focused_profiler(level: ProfilerLevel = ProfilerLevel.DETAILED,
                                    output_dir: str = "evolution_profiling") -> ProfilerManager:
    """
    Create a profiler specifically optimized for evolution experiments.
    
    Args:
        level: Profiling detail level
        output_dir: Directory for saving results
        
    Returns:
        ProfilerManager optimized for evolution profiling
    """
    config = ProfilerConfig(
        level=level,
        profile_memory=True,
        profile_compute=True,
        profile_io=False,
        output_dir=output_dir,
        auto_save=True,
        save_interval=10,  # Save frequently for evolution events
        max_overhead_percent=5.0,
        adaptive_sampling=True
    )
    
    manager = ProfilerManager(config)
    
    # Add evolution profiler with enhanced configuration if available
    if EvolutionProfiler is not None:
        evolution_profiler = EvolutionProfiler(config)
        manager.register_profiler(evolution_profiler)
    
    print(f"🧬 Created evolution-focused profiler (level: {level.name})")
    return manager


def create_research_profiler(experiment_name: str,
                           level: ProfilerLevel = ProfilerLevel.COMPREHENSIVE,
                           enable_all_integrations: bool = True) -> ProfilerManager:
    """
    Create a profiler setup optimized for research experiments.
    
    Args:
        experiment_name: Name of the research experiment
        level: Profiling detail level
        enable_all_integrations: Whether to enable all integrations
        
    Returns:
        ProfilerManager configured for research use
    """
    output_dir = f"research_profiling/{experiment_name}"
    
    config = ProfilerConfig(
        level=level,
        profile_memory=True,
        profile_compute=True,
        profile_io=True,
        output_dir=output_dir,
        auto_save=True,
        save_interval=20,
        integrate_with_wandb=enable_all_integrations,
        integrate_with_logging=enable_all_integrations,
        max_overhead_percent=15.0,  # Higher tolerance for research
        adaptive_sampling=False,  # Consistent profiling for research
        output_format="json"
    )
    
    manager = ProfilerManager(config)
    
    # Add all profilers for comprehensive research data if available
    if EvolutionProfiler is not None:
        evolution_profiler = EvolutionProfiler(config)
        manager.register_profiler(evolution_profiler)
    
    print(f"🔬 Created research profiler for experiment: {experiment_name}")
    return manager


def create_production_profiler(max_overhead_percent: float = 2.0,
                             output_dir: str = "production_profiling") -> ProfilerManager:
    """
    Create a profiler suitable for production environments.
    
    Args:
        max_overhead_percent: Maximum acceptable overhead percentage
        output_dir: Directory for saving results
        
    Returns:
        ProfilerManager configured for production use
    """
    config = ProfilerConfig(
        level=ProfilerLevel.BASIC,
        profile_memory=True,
        profile_compute=False,  # Avoid GPU profiling in production
        profile_io=False,
        output_dir=output_dir,
        auto_save=True,
        save_interval=200,  # Less frequent saves
        max_overhead_percent=max_overhead_percent,
        adaptive_sampling=True,  # Critical for production
        integrate_with_logging=True
    )
    
    manager = ProfilerManager(config)
    
    # Add minimal profilers for production monitoring if available
    if EvolutionProfiler is not None:
        evolution_profiler = EvolutionProfiler(config)
        manager.register_profiler(evolution_profiler)
    
    print(f"🏭 Created production profiler (max overhead: {max_overhead_percent}%)")
    return manager


# Convenience functions for quick setup
def quick_evolution_profiler() -> ProfilerManager:
    """Quick setup for evolution profiling."""
    return create_evolution_focused_profiler()


def quick_research_profiler(experiment_name: str) -> ProfilerManager:
    """Quick setup for research profiling."""
    return create_research_profiler(experiment_name)


def quick_lightweight_profiler() -> ProfilerManager:
    """Quick setup for lightweight profiling."""
    return create_lightweight_profiler()
