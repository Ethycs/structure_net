#!/usr/bin/env python3
"""
Modular Profiling System for Structure Net

This package provides comprehensive, granular profiling capabilities
for monitoring performance, memory usage, and execution patterns
across all components of the structure_net system.

Key Features:
- Toggleable profiling with minimal overhead when disabled
- Granular profiling of individual operations
- Memory and compute profiling
- Component-specific profilers
- Customizable profiling configurations
- Integration with existing logging system
"""

from .core.profiler_manager import ProfilerManager
from .core.base_profiler import BaseProfiler, ProfilerConfig, ProfilerLevel
from .core.decorators import (
    profile_function, profile_method, profile_component,
    profile_if_enabled, profile_async, profile_evolution,
    profile_metrics, profile_training, profile_network
)
from .core.context_manager import (
    ProfilerContext, BatchProfilerContext, 
    profile_operation, profile_batch_operation,
    profile_function_call, profile_if_slow, profile_memory_intensive
)

# Component-specific profilers (import safely)
try:
    from .components.evolution_profiler import EvolutionProfiler
except ImportError:
    EvolutionProfiler = None

# Factory functions
from .factory import (
    create_standard_profiler,
    create_lightweight_profiler,
    create_comprehensive_profiler,
    create_custom_profiler,
    create_evolution_focused_profiler,
    create_research_profiler,
    create_production_profiler,
    quick_evolution_profiler,
    quick_research_profiler,
    quick_lightweight_profiler
)

# Export all components
__all__ = [
    # Core profiling system
    'ProfilerManager',
    'BaseProfiler', 
    'ProfilerConfig',
    'ProfilerLevel',
    'ProfilerContext',
    'BatchProfilerContext',
    
    # Decorators
    'profile_function',
    'profile_method', 
    'profile_component',
    'profile_if_enabled',
    'profile_async',
    'profile_evolution',
    'profile_metrics',
    'profile_training',
    'profile_network',
    
    # Context managers and utilities
    'profile_operation',
    'profile_batch_operation',
    'profile_function_call',
    'profile_if_slow',
    'profile_memory_intensive',
    
    # Component profilers
    'EvolutionProfiler',
    
    # Factory functions
    'create_standard_profiler',
    'create_lightweight_profiler',
    'create_comprehensive_profiler',
    'create_custom_profiler',
    'create_evolution_focused_profiler',
    'create_research_profiler',
    'create_production_profiler',
    'quick_evolution_profiler',
    'quick_research_profiler',
    'quick_lightweight_profiler'
]

# Version info
__version__ = "1.0.0"
