#!/usr/bin/env python3
"""
Profiling Decorators

Provides convenient decorators for adding profiling to functions, methods,
and classes with minimal code changes.
"""

import functools
import inspect
from typing import Any, Callable, Optional, List, Union, Type
import time

from .profiler_manager import get_global_profiler_manager
from .base_profiler import ProfilerLevel


def profile_function(operation_name: Optional[str] = None,
                    component: Optional[str] = None,
                    profiler_names: Optional[List[str]] = None,
                    tags: Optional[List[str]] = None,
                    level: ProfilerLevel = ProfilerLevel.BASIC):
    """
    Decorator for profiling individual functions.
    
    Args:
        operation_name: Custom operation name (defaults to function name)
        component: Component name for categorization
        profiler_names: Specific profilers to use
        tags: Tags for categorization
        level: Minimum profiling level required
        
    Usage:
        @profile_function(component="evolution")
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_global_profiler_manager()
            
            # Skip if profiling level is too low
            if manager.global_config.level.value < level.value:
                return func(*args, **kwargs)
            
            # Determine operation name
            op_name = operation_name or func.__name__
            
            # Determine component from module if not specified
            comp = component
            if comp is None and hasattr(func, '__module__'):
                module_parts = func.__module__.split('.')
                if 'evolution' in module_parts:
                    comp = 'evolution'
                elif 'metrics' in module_parts:
                    comp = 'metrics'
                elif 'training' in module_parts:
                    comp = 'training'
                elif 'network' in module_parts:
                    comp = 'network'
                else:
                    comp = 'general'
            
            # Profile the function execution
            with manager.profile_operation(op_name, comp or 'general', profiler_names, tags) as ctx:
                # Add function metadata
                ctx.add_metric('function_name', func.__name__)
                ctx.add_metric('module', func.__module__)
                ctx.add_metric('args_count', len(args))
                ctx.add_metric('kwargs_count', len(kwargs))
                
                # Execute function
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    ctx.add_metric('success', True)
                    return result
                except Exception as e:
                    ctx.add_metric('success', False)
                    ctx.add_metric('exception_type', type(e).__name__)
                    raise
                finally:
                    execution_time = time.perf_counter() - start_time
                    ctx.add_metric('execution_time', execution_time)
        
        # Add profiling metadata to function
        wrapper._profiled = True
        wrapper._profile_config = {
            'operation_name': operation_name,
            'component': component,
            'profiler_names': profiler_names,
            'tags': tags,
            'level': level
        }
        
        return wrapper
    return decorator


def profile_method(operation_name: Optional[str] = None,
                  component: Optional[str] = None,
                  profiler_names: Optional[List[str]] = None,
                  tags: Optional[List[str]] = None,
                  level: ProfilerLevel = ProfilerLevel.BASIC,
                  include_self_info: bool = True):
    """
    Decorator for profiling class methods.
    
    Args:
        operation_name: Custom operation name (defaults to class.method)
        component: Component name for categorization
        profiler_names: Specific profilers to use
        tags: Tags for categorization
        level: Minimum profiling level required
        include_self_info: Whether to include information about self object
        
    Usage:
        class MyClass:
            @profile_method(component="evolution")
            def my_method(self):
                pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            manager = get_global_profiler_manager()
            
            # Skip if profiling level is too low
            if manager.global_config.level.value < level.value:
                return func(self, *args, **kwargs)
            
            # Determine operation name
            class_name = self.__class__.__name__
            method_name = func.__name__
            op_name = operation_name or f"{class_name}.{method_name}"
            
            # Determine component
            comp = component
            if comp is None:
                # Try to infer from class name or module
                if hasattr(self, '__module__'):
                    module_parts = self.__module__.split('.')
                    if 'evolution' in module_parts or 'Evolution' in class_name:
                        comp = 'evolution'
                    elif 'metrics' in module_parts or 'Metrics' in class_name:
                        comp = 'metrics'
                    elif 'training' in module_parts or 'Train' in class_name:
                        comp = 'training'
                    elif 'network' in module_parts or 'Network' in class_name:
                        comp = 'network'
                    else:
                        comp = 'general'
            
            # Profile the method execution
            with manager.profile_operation(op_name, comp or 'general', profiler_names, tags) as ctx:
                # Add method metadata
                ctx.add_metric('class_name', class_name)
                ctx.add_metric('method_name', method_name)
                ctx.add_metric('args_count', len(args))
                ctx.add_metric('kwargs_count', len(kwargs))
                
                # Add self object information if requested
                if include_self_info:
                    if hasattr(self, '__dict__'):
                        ctx.add_metric('self_attributes_count', len(self.__dict__))
                    
                    # Add specific information for known types
                    if hasattr(self, 'name'):
                        ctx.add_metric('object_name', self.name)
                    if hasattr(self, 'config'):
                        ctx.add_metric('has_config', True)
                
                # Execute method
                start_time = time.perf_counter()
                try:
                    result = func(self, *args, **kwargs)
                    ctx.add_metric('success', True)
                    return result
                except Exception as e:
                    ctx.add_metric('success', False)
                    ctx.add_metric('exception_type', type(e).__name__)
                    raise
                finally:
                    execution_time = time.perf_counter() - start_time
                    ctx.add_metric('execution_time', execution_time)
        
        # Add profiling metadata to method
        wrapper._profiled = True
        wrapper._profile_config = {
            'operation_name': operation_name,
            'component': component,
            'profiler_names': profiler_names,
            'tags': tags,
            'level': level,
            'include_self_info': include_self_info
        }
        
        return wrapper
    return decorator


def profile_component(component_name: Optional[str] = None,
                     profiler_names: Optional[List[str]] = None,
                     tags: Optional[List[str]] = None,
                     level: ProfilerLevel = ProfilerLevel.BASIC,
                     methods_to_profile: Optional[List[str]] = None,
                     exclude_methods: Optional[List[str]] = None):
    """
    Class decorator for profiling all methods in a component.
    
    Args:
        component_name: Component name (defaults to class name)
        profiler_names: Specific profilers to use
        tags: Tags for categorization
        level: Minimum profiling level required
        methods_to_profile: Specific methods to profile (None = all public methods)
        exclude_methods: Methods to exclude from profiling
        
    Usage:
        @profile_component(component_name="evolution")
        class MyEvolutionClass:
            def method1(self):
                pass
            def method2(self):
                pass
    """
    def decorator(cls: Type) -> Type:
        class_name = cls.__name__
        comp_name = component_name or class_name.lower()
        
        # Get methods to profile
        if methods_to_profile is not None:
            methods = methods_to_profile
        else:
            # Profile all public methods by default
            methods = [name for name, method in inspect.getmembers(cls, predicate=inspect.isfunction)
                      if not name.startswith('_')]
        
        # Apply exclusions
        if exclude_methods:
            methods = [m for m in methods if m not in exclude_methods]
        
        # Apply profiling to each method
        for method_name in methods:
            if hasattr(cls, method_name):
                original_method = getattr(cls, method_name)
                
                # Skip if already profiled
                if hasattr(original_method, '_profiled'):
                    continue
                
                # Create profiled version
                profiled_method = profile_method(
                    operation_name=f"{class_name}.{method_name}",
                    component=comp_name,
                    profiler_names=profiler_names,
                    tags=tags,
                    level=level
                )(original_method)
                
                # Replace the method
                setattr(cls, method_name, profiled_method)
        
        # Add profiling metadata to class
        cls._profiled_component = True
        cls._profile_config = {
            'component_name': comp_name,
            'profiler_names': profiler_names,
            'tags': tags,
            'level': level,
            'profiled_methods': methods
        }
        
        return cls
    return decorator


def profile_if_enabled(operation_name: Optional[str] = None,
                      component: Optional[str] = None,
                      condition: Optional[Callable[[], bool]] = None):
    """
    Conditional profiling decorator that only profiles if a condition is met.
    
    Args:
        operation_name: Custom operation name
        component: Component name
        condition: Function that returns True if profiling should be enabled
        
    Usage:
        @profile_if_enabled(condition=lambda: os.getenv('PROFILE_ENABLED') == '1')
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check condition
            should_profile = True
            if condition is not None:
                try:
                    should_profile = condition()
                except:
                    should_profile = False
            
            if not should_profile:
                return func(*args, **kwargs)
            
            # Use regular profiling
            return profile_function(operation_name, component)(func)(*args, **kwargs)
        
        return wrapper
    return decorator


def profile_async(operation_name: Optional[str] = None,
                 component: Optional[str] = None,
                 profiler_names: Optional[List[str]] = None,
                 tags: Optional[List[str]] = None,
                 level: ProfilerLevel = ProfilerLevel.BASIC):
    """
    Decorator for profiling async functions.
    
    Args:
        operation_name: Custom operation name
        component: Component name
        profiler_names: Specific profilers to use
        tags: Tags for categorization
        level: Minimum profiling level required
        
    Usage:
        @profile_async(component="evolution")
        async def my_async_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            manager = get_global_profiler_manager()
            
            # Skip if profiling level is too low
            if manager.global_config.level.value < level.value:
                return await func(*args, **kwargs)
            
            # Determine operation name
            op_name = operation_name or func.__name__
            
            # Determine component
            comp = component or 'async'
            
            # Profile the async function execution
            with manager.profile_operation(op_name, comp, profiler_names, tags) as ctx:
                # Add function metadata
                ctx.add_metric('function_name', func.__name__)
                ctx.add_metric('is_async', True)
                ctx.add_metric('args_count', len(args))
                ctx.add_metric('kwargs_count', len(kwargs))
                
                # Execute async function
                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    ctx.add_metric('success', True)
                    return result
                except Exception as e:
                    ctx.add_metric('success', False)
                    ctx.add_metric('exception_type', type(e).__name__)
                    raise
                finally:
                    execution_time = time.perf_counter() - start_time
                    ctx.add_metric('execution_time', execution_time)
        
        # Add profiling metadata
        wrapper._profiled = True
        wrapper._profile_config = {
            'operation_name': operation_name,
            'component': component,
            'profiler_names': profiler_names,
            'tags': tags,
            'level': level,
            'is_async': True
        }
        
        return wrapper
    return decorator


# Convenience decorators for specific components
def profile_evolution(operation_name: Optional[str] = None, **kwargs):
    """Convenience decorator for evolution component profiling."""
    return profile_function(operation_name=operation_name, component='evolution', **kwargs)


def profile_metrics(operation_name: Optional[str] = None, **kwargs):
    """Convenience decorator for metrics component profiling."""
    return profile_function(operation_name=operation_name, component='metrics', **kwargs)


def profile_training(operation_name: Optional[str] = None, **kwargs):
    """Convenience decorator for training component profiling."""
    return profile_function(operation_name=operation_name, component='training', **kwargs)


def profile_network(operation_name: Optional[str] = None, **kwargs):
    """Convenience decorator for network component profiling."""
    return profile_function(operation_name=operation_name, component='network', **kwargs)


# Utility functions
def is_profiled(func_or_class) -> bool:
    """Check if a function or class has profiling enabled."""
    return hasattr(func_or_class, '_profiled') or hasattr(func_or_class, '_profiled_component')


def get_profile_config(func_or_class) -> Optional[dict]:
    """Get profiling configuration for a function or class."""
    return getattr(func_or_class, '_profile_config', None)


def remove_profiling(func_or_class):
    """Remove profiling from a function or class (returns original)."""
    if hasattr(func_or_class, '__wrapped__'):
        return func_or_class.__wrapped__
    return func_or_class
