"""
The StructureNet microkernel.

The kernel is the infrastructure layer beneath the research components and
the NAL research harness: it owns logging, profiling, the event bus, the
component registry/compatibility services, tracking (scorecards, experiment
records, health), and a small service registry for dependency injection.

It deliberately does **not** replace NAL. NAL is the scientific-methodology
layer and may optionally receive a kernel (``NeuralArchitectureLab(config,
kernel=...)``) to reuse these services.

Reconciliation note: this implements the archived kernel design
(``docs/09 - Archived/kernel_implementation_guide.md``) on top of the
existing component architecture — ``ComponentRegistry`` and
``CompatibilityManager`` from :mod:`structure_net.core.compatibility`,
``KernelProfiler``/``ComponentProfiler`` from :mod:`structure_net.profiling`
— rather than duplicating them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Union

from .compatibility import CompatibilityIssue, CompatibilityManager, ComponentRegistry
from .events import EventBus
from .interfaces import IComponent
from .kernel_config import KernelConfig


ServiceKey = Union[str, Type[Any]]


def _service_name(key: ServiceKey) -> str:
    return key if isinstance(key, str) else key.__name__


class StructureNetKernel:
    """Microkernel providing core services to components and harnesses."""

    def __init__(self, config: Optional[Union[KernelConfig, Dict[str, Any]]] = None):
        if config is None:
            config = KernelConfig()
        elif isinstance(config, dict):
            config = KernelConfig(**config)
        self.config: KernelConfig = config

        # Core services — always available.
        self._logger = self._initialize_logging()
        self._profiler = self._initialize_profiling()
        self._event_bus = EventBus(self._logger)

        # Component management (reuses the existing registry/compatibility).
        self._registry = ComponentRegistry()
        self._compatibility_manager = CompatibilityManager(self._registry)

        # Tracking services.
        from ..tracking import (
            ComponentHealthMonitor,
            ExperimentManager,
            ScorecardManager,
        )

        self._scorecard_manager = ScorecardManager(self)
        self._experiment_manager = ExperimentManager(self)
        self._health_monitor = ComponentHealthMonitor(self)

        # Service registry / plugin system.
        self._services: Dict[str, Any] = {}

        self._logger.info("StructureNet Kernel initialized")

    # ------------------------------------------------------------------
    # Core service construction
    # ------------------------------------------------------------------

    def _initialize_logging(self):
        from ..logging.kernel_logger import KernelLogger

        return KernelLogger(
            name="structure_net.kernel",
            level=self.config.log_level,
            handlers=self.config.log_handlers,
            log_file=self.config.log_file,
        )

    def _initialize_profiling(self):
        from ..profiling.kernel_profiler import KernelProfiler

        return KernelProfiler(
            logger=self._logger,
            enable_cpu=self.config.profile_cpu,
            enable_memory=self.config.profile_memory,
            enable_gpu=self.config.profile_gpu,
        )

    # ------------------------------------------------------------------
    # Component management
    # ------------------------------------------------------------------

    def register_component(self, component_class: Type[IComponent]) -> None:
        """Register a component class in the kernel's registry.

        The registry instantiates the class once (with no arguments) to read
        its contract, so registered components must be default-constructible.
        """
        self._registry.register(component_class)
        contract = component_class().contract
        self.log_event(
            "component.registered",
            component=contract.component_name,
            version=str(contract.version),
        )

    def create_component(self, component_name: str, **kwargs: Any) -> IComponent:
        """Create a component instance with kernel services injected."""
        component_class = self._registry.get_component(component_name)
        if component_class is None:
            raise KeyError(f"component '{component_name}' is not registered")
        component = component_class(**kwargs)

        # Inject kernel services. ``_logger`` intentionally replaces the
        # stdlib logger installed by ``IComponent.__init__``; the adapter is
        # call-compatible with it (including ``log(level, msg)``).
        component._kernel = self
        component._logger = self.get_logger(component.name)
        component._profiler = self.get_profiler(component.name)

        self._scorecard_manager.create_scorecard(component)
        self.log_event("component.created", component=component.name)
        return component

    def get_logger(self, component_name: str):
        """Get a structured logger bound to ``component_name``."""
        from ..logging.kernel_logger import KernelComponentLogger

        return KernelComponentLogger(self._logger, component_name)

    def get_profiler(self, component_name: str):
        """Get a profiler bound to ``component_name``."""
        from ..profiling.component_profiler import ComponentProfiler

        return ComponentProfiler(self._profiler, component_name)

    def validate_composition(
        self, components: List[IComponent]
    ) -> List[CompatibilityIssue]:
        """Validate a component composition; returns discovered issues."""
        return self._compatibility_manager.validate_composition(components)

    # ------------------------------------------------------------------
    # Service registry (dependency injection)
    # ------------------------------------------------------------------

    def register_service(self, key: ServiceKey, service: Any) -> None:
        """Register a shared service under a string or type key."""
        name = _service_name(key)
        self._services[name] = service
        self.log_event("service.registered", service=name)

    def get_service(self, key: ServiceKey) -> Any:
        """Look up a shared service; raises ``KeyError`` when absent."""
        name = _service_name(key)
        if name not in self._services:
            raise KeyError(f"service '{name}' is not registered")
        return self._services[name]

    def has_service(self, key: ServiceKey) -> bool:
        return _service_name(key) in self._services

    # ------------------------------------------------------------------
    # Accessors and helpers
    # ------------------------------------------------------------------

    @property
    def logger(self):
        return self._logger

    @property
    def profiler(self):
        return self._profiler

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def registry(self) -> ComponentRegistry:
        return self._registry

    @property
    def compatibility_manager(self) -> CompatibilityManager:
        return self._compatibility_manager

    @property
    def scorecard_manager(self):
        return self._scorecard_manager

    @property
    def experiment_manager(self):
        return self._experiment_manager

    @property
    def health_monitor(self):
        return self._health_monitor

    def log_event(self, event_name: str, **payload: Any) -> None:
        """Publish a kernel event (never raises)."""
        try:
            self._event_bus.publish(event_name, payload, source="kernel")
        except Exception:  # noqa: BLE001 - eventing must never break callers
            pass

    def shutdown(self) -> None:
        """Release logging handlers and publish the shutdown event."""
        self.log_event("kernel.shutdown")
        self._logger.close()
