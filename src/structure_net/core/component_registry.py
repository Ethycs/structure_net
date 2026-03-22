"""
Composition-Aware Component Registry

Extends the existing ComponentRegistry (compatibility.py) with category-based
lookup and model factory support, enabling the composition system to resolve
string names to actual component classes.

Categories map 1:1 to core/interfaces.py types:
    analyzer, strategy, evolver, trainer, metric, scheduler, orchestrator
"""

from typing import Dict, Type, List, Optional, Callable, Any
import logging

import torch.nn as nn

from .interfaces import (
    IComponent, IAnalyzer, IStrategy, IEvolver,
    ITrainer, IMetric, IScheduler, IOrchestrator,
)

logger = logging.getLogger(__name__)

# Map category strings to interface types
CATEGORY_INTERFACE_MAP: Dict[str, Type[IComponent]] = {
    "analyzer": IAnalyzer,
    "strategy": IStrategy,
    "evolver": IEvolver,
    "trainer": ITrainer,
    "metric": IMetric,
    "scheduler": IScheduler,
    "orchestrator": IOrchestrator,
}


class CompositionRegistry:
    """
    Central registry for resolving component names to classes.

    Supports two kinds of entries:
      1. IComponent subclasses, registered under a category + name.
      2. Model factory functions (architecture -> nn.Module), registered by name.
    """

    def __init__(self):
        # category -> name -> class
        self._components: Dict[str, Dict[str, Type[IComponent]]] = {
            cat: {} for cat in CATEGORY_INTERFACE_MAP
        }
        # name -> factory callable
        self._model_factories: Dict[str, Callable[..., nn.Module]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, category: str, name: str, cls: Type[IComponent]) -> None:
        """Register a component class under a category and name."""
        if category not in self._components:
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Valid categories: {list(CATEGORY_INTERFACE_MAP)}"
            )
        self._components[category][name] = cls
        logger.debug("Registered %s/%s -> %s", category, name, cls.__name__)

    def register_model_factory(self, name: str, factory: Callable[..., nn.Module]) -> None:
        """Register a model factory function."""
        self._model_factories[name] = factory
        logger.debug("Registered model factory '%s'", name)

    # ------------------------------------------------------------------
    # Lookup / creation
    # ------------------------------------------------------------------

    def get_class(self, category: str, name: str) -> Type[IComponent]:
        """Look up a component class. Raises KeyError if not found."""
        try:
            return self._components[category][name]
        except KeyError:
            available = list(self._components.get(category, {}).keys())
            raise KeyError(
                f"No component '{name}' in category '{category}'. "
                f"Available: {available}"
            )

    def create(self, category: str, name: str, config: Dict[str, Any] = None) -> IComponent:
        """Instantiate a component from the registry."""
        cls = self.get_class(category, name)
        config = config or {}
        return cls(**config)

    def create_model(self, name: str, **kwargs) -> nn.Module:
        """Create a model using a registered factory function."""
        if name not in self._model_factories:
            available = list(self._model_factories.keys())
            raise KeyError(
                f"No model factory '{name}'. Available: {available}"
            )
        return self._model_factories[name](**kwargs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_available(self, category: str) -> List[str]:
        """List registered names for a category."""
        return list(self._components.get(category, {}).keys())

    def list_model_factories(self) -> List[str]:
        """List registered model factory names."""
        return list(self._model_factories.keys())

    def has(self, category: str, name: str) -> bool:
        return name in self._components.get(category, {})

    def has_model_factory(self, name: str) -> bool:
        return name in self._model_factories


# ------------------------------------------------------------------
# Decorator for easy registration
# ------------------------------------------------------------------

# Deferred registrations (accumulated before the global registry is built)
_DEFERRED: List[tuple] = []


def register_component(category: str, name: str):
    """
    Class decorator that registers a component with the global registry.

    Usage::

        @register_component("analyzer", "extrema")
        class ExtremaAnalyzer(IAnalyzer):
            ...
    """
    def decorator(cls):
        _DEFERRED.append((category, name, cls))
        return cls
    return decorator


# ------------------------------------------------------------------
# Global singleton
# ------------------------------------------------------------------

_GLOBAL_REGISTRY: Optional[CompositionRegistry] = None


def get_global_registry() -> CompositionRegistry:
    """Return (and lazily build) the global CompositionRegistry."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = _build_default_registry()
    return _GLOBAL_REGISTRY


def _build_default_registry() -> CompositionRegistry:
    """Build the default registry with all known components."""
    registry = CompositionRegistry()

    # Apply decorator-based registrations
    for category, name, cls in _DEFERRED:
        registry.register(category, name, cls)

    # Register model factories from network_factory
    try:
        from .network_factory import (
            create_standard_network,
            create_extrema_aware_network,
            create_evolvable_network,
        )
        registry.register_model_factory("standard", create_standard_network)
        registry.register_model_factory("extrema_aware", create_extrema_aware_network)
        registry.register_model_factory("evolvable", create_evolvable_network)
    except ImportError:
        logger.warning("Could not import network_factory — model factories not registered")

    # Register non-deprecated analyzer components
    try:
        from ..components.analyzers.extrema_analyzer import ExtremaAnalyzer
        from ..components.analyzers.information_flow_analyzer import InformationFlowAnalyzer
        from ..components.analyzers.graph_analyzer import GraphAnalyzer
        from ..components.analyzers.homological_analyzer import HomologicalAnalyzer
        from ..components.analyzers.topological_analyzer import TopologicalAnalyzer
        from ..components.analyzers.activity_analyzer import ActivityAnalyzer
        from ..components.analyzers.compactification_analyzer import CompactificationAnalyzer
        from ..components.analyzers.catastrophe_analyzer import CatastropheAnalyzer
        from ..components.analyzers.sensitivity_analyzer import SensitivityAnalyzer
        from ..components.analyzers.performance_correlation_analyzer import PerformanceCorrelationAnalyzer

        for name, cls in [
            ("extrema", ExtremaAnalyzer),
            ("information_flow", InformationFlowAnalyzer),
            ("graph", GraphAnalyzer),
            ("homological", HomologicalAnalyzer),
            ("topological", TopologicalAnalyzer),
            ("activity", ActivityAnalyzer),
            ("compactification", CompactificationAnalyzer),
            ("catastrophe", CatastropheAnalyzer),
            ("sensitivity", SensitivityAnalyzer),
            ("performance_correlation", PerformanceCorrelationAnalyzer),
        ]:
            registry.register("analyzer", name, cls)
    except ImportError:
        logger.warning("Could not import analyzer components")

    # Register non-deprecated strategy components
    try:
        from ..components.strategies.extrema_growth_strategy import ExtremaGrowthStrategy
        from ..components.strategies.information_flow_growth_strategy import InformationFlowGrowthStrategy
        from ..components.strategies.residual_block_growth_strategy import ResidualBlockGrowthStrategy
        from ..components.strategies.hybrid_growth_strategy import HybridGrowthStrategy

        for name, cls in [
            ("extrema_growth", ExtremaGrowthStrategy),
            ("information_flow_growth", InformationFlowGrowthStrategy),
            ("residual_block_growth", ResidualBlockGrowthStrategy),
            ("hybrid_growth", HybridGrowthStrategy),
        ]:
            registry.register("strategy", name, cls)
    except ImportError:
        logger.warning("Could not import strategy components")

    return registry
