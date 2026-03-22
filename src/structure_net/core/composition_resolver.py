"""
Composition Resolver

Bridges composition schemas (component_schemas.py) to live Python objects
using the CompositionRegistry (component_registry.py).

Usage::

    from structure_net.core.component_registry import get_global_registry
    from structure_net.core.composition_resolver import CompositionResolver
    from structure_net.logging.component_schemas import ExperimentComposition

    resolver = CompositionResolver(get_global_registry())
    resolved = resolver.resolve(composition, device='cpu')
    # resolved.model, resolved.analyzers, resolved.strategies, ...
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import torch.nn as nn
import logging

from .interfaces import (
    IAnalyzer, IStrategy, IEvolver, ITrainer,
    IMetric, IScheduler, IOrchestrator, IComponent,
)
from .component_registry import CompositionRegistry
from ..logging.component_schemas import (
    ExperimentComposition, ComponentSpec, TrainingSpec, HypothesisSpec,
)

logger = logging.getLogger(__name__)


@dataclass
class ResolvedComposition:
    """A composition where all specs have been resolved to live instances."""

    model: nn.Module
    analyzers: List[IAnalyzer] = field(default_factory=list)
    strategies: List[IStrategy] = field(default_factory=list)
    evolvers: List[IEvolver] = field(default_factory=list)
    trainers: List[ITrainer] = field(default_factory=list)
    metrics: List[IMetric] = field(default_factory=list)
    schedulers: List[IScheduler] = field(default_factory=list)
    orchestrator: Optional[IOrchestrator] = None
    training_config: Optional[TrainingSpec] = None
    hypothesis: Optional[HypothesisSpec] = None

    # Metadata populated during resolution
    resolved_classes: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Human-readable summary of what was resolved."""
        return {
            "model": type(self.model).__name__,
            "analyzers": [type(a).__name__ for a in self.analyzers],
            "strategies": [type(s).__name__ for s in self.strategies],
            "evolvers": [type(e).__name__ for e in self.evolvers],
            "trainers": [type(t).__name__ for t in self.trainers],
            "metrics": [type(m).__name__ for m in self.metrics],
            "schedulers": [type(s).__name__ for s in self.schedulers],
            "orchestrator": type(self.orchestrator).__name__ if self.orchestrator else None,
            "has_training_config": self.training_config is not None,
            "has_hypothesis": self.hypothesis is not None,
        }


class CompositionResolver:
    """Resolves an ExperimentComposition into live component instances."""

    def __init__(self, registry: CompositionRegistry):
        self.registry = registry

    def resolve(
        self,
        composition: ExperimentComposition,
        device: str = "cpu",
    ) -> ResolvedComposition:
        """
        Instantiate all components from a composition spec.

        Args:
            composition: The composition schema to resolve.
            device: Device for model creation (e.g. 'cpu', 'cuda:0').

        Returns:
            ResolvedComposition with live objects.

        Raises:
            KeyError: If a component name is not found in the registry.
        """
        # 1. Create model
        model_spec = composition.model
        factory_kwargs = {
            "architecture": model_spec.architecture,
            "sparsity": model_spec.sparsity,
            "device": device,
            **model_spec.config,
        }
        model = self.registry.create_model(model_spec.factory_name, **factory_kwargs)

        resolved_classes: Dict[str, str] = {
            "model": type(model).__name__,
        }

        # 2. Instantiate component lists
        analyzers = self._resolve_list(composition.analyzers, resolved_classes)
        strategies = self._resolve_list(composition.strategies, resolved_classes)
        evolvers = self._resolve_list(composition.evolvers, resolved_classes)
        trainers = self._resolve_list(composition.trainers, resolved_classes)
        metrics = self._resolve_list(composition.metrics, resolved_classes)
        schedulers = self._resolve_list(composition.schedulers, resolved_classes)

        # 3. Auto-create orchestrator if analyzers + strategies present
        orchestrator = None
        if analyzers and strategies:
            orchestrator = self._try_create_orchestrator(
                analyzers, strategies, evolvers, resolved_classes
            )

        resolved = ResolvedComposition(
            model=model,
            analyzers=analyzers,
            strategies=strategies,
            evolvers=evolvers,
            trainers=trainers,
            metrics=metrics,
            schedulers=schedulers,
            orchestrator=orchestrator,
            training_config=composition.training,
            hypothesis=composition.hypothesis,
            resolved_classes=resolved_classes,
        )

        logger.info(
            "Resolved composition '%s': %s",
            composition.name,
            resolved.summary(),
        )
        return resolved

    def validate_contracts(self, resolved: ResolvedComposition) -> List[str]:
        """
        Check that all component contracts are satisfied.

        Returns a list of issue descriptions (empty means all OK).
        """
        issues: List[str] = []

        # Collect all provided outputs and required inputs
        all_provided: set = set()
        all_required: set = set()

        all_components: List[IComponent] = (
            resolved.analyzers + resolved.strategies + resolved.evolvers
            + resolved.trainers + resolved.metrics + resolved.schedulers
        )
        if resolved.orchestrator:
            all_components.append(resolved.orchestrator)

        for comp in all_components:
            try:
                contract = comp.contract
                all_provided.update(contract.provided_outputs)
                all_required.update(contract.required_inputs)
            except Exception as exc:
                issues.append(f"Cannot read contract for {comp.name}: {exc}")

        # Check for unsatisfied required inputs
        missing = all_required - all_provided
        # Filter out common system-level inputs that are provided at runtime
        runtime_inputs = {"model", "data_loader", "device", "context", "layer"}
        missing -= runtime_inputs

        for m in missing:
            issues.append(f"Required input '{m}' is not provided by any component")

        return issues

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_list(
        self,
        specs: List[ComponentSpec],
        resolved_classes: Dict[str, str],
    ) -> list:
        """Resolve a list of ComponentSpecs to instances."""
        instances = []
        for spec in specs:
            instance = self.registry.create(
                spec.component_type, spec.component_name, spec.config
            )
            instances.append(instance)
            key = f"{spec.component_type}/{spec.component_name}"
            resolved_classes[key] = type(instance).__name__
        return instances

    def _try_create_orchestrator(
        self,
        analyzers: list,
        strategies: list,
        evolvers: list,
        resolved_classes: Dict[str, str],
    ) -> Optional[IOrchestrator]:
        """Try to create an EvolutionOrchestrator from resolved components."""
        try:
            from ..components.orchestrators.evolution_orchestrator import EvolutionOrchestrator

            orchestrator = EvolutionOrchestrator(
                analyzers=analyzers,
                strategies=strategies,
                evolvers=evolvers if evolvers else None,
            )
            resolved_classes["orchestrator"] = "EvolutionOrchestrator"
            return orchestrator
        except (ImportError, TypeError):
            logger.debug(
                "Could not auto-create EvolutionOrchestrator — "
                "analyzers and strategies resolved but orchestrator unavailable"
            )
            return None
