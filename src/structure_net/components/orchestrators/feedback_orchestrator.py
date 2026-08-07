"""
Feedback-loop orchestrator.

Runs the canonical component cycle from the kernel design:

    trainer -> metrics -> analyzers -> (scheduler gate) -> strategies -> evolvers

Each cycle trains one step, assembles an :class:`AnalysisReport` from the
metric and analyzer components, and — when the scheduler triggers — asks each
strategy for an :class:`EvolutionPlan` and applies every plan through the
first evolver that can execute it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from ...core.base_components import BaseOrchestrator
from ...core.interfaces import (
    AnalysisReport,
    ComponentContract,
    ComponentVersion,
    EvolutionContext,
    Maturity,
    ResourceRequirements,
    ResourceLevel,
)


class FeedbackOrchestrator(BaseOrchestrator):
    """Orchestrates trainer, metrics, analyzers, strategies, and evolvers."""

    def __init__(
        self,
        trainer: Any = None,
        metrics: Optional[List[Any]] = None,
        analyzers: Optional[List[Any]] = None,
        strategies: Optional[List[Any]] = None,
        evolvers: Optional[List[Any]] = None,
        scheduler: Any = None,
        name: str = None,
    ):
        super().__init__(name or "FeedbackOrchestrator")
        self.trainer = trainer
        self.metrics = list(metrics or [])
        self.analyzers = list(analyzers or [])
        self.strategies = list(strategies or [])
        self.evolvers = list(evolvers or [])
        self.scheduler = scheduler
        self.evolution_history: List[Dict[str, Any]] = []

    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"model", "batch", "optimizer"},
            provided_outputs={"training.results", "evolution.history"},
            resources=ResourceRequirements(memory_level=ResourceLevel.MEDIUM),
        )

    def _log(self, level: str, message: str) -> None:
        # Works with both the kernel's injected adapter and the stdlib logger
        # installed by IComponent.__init__.
        logger = getattr(self, "_logger", None)
        if logger is not None:
            getattr(logger, level)(message)

    def run_cycle(self, context: EvolutionContext) -> Dict[str, Any]:
        """Run one complete feedback cycle."""
        profiler = getattr(self, "_profiler", None)
        if profiler is not None:
            with profiler.profile_method("run_cycle"):
                return self._run_cycle(context)
        return self._run_cycle(context)

    def _run_cycle(self, context: EvolutionContext) -> Dict[str, Any]:
        self._log("info", f"Starting feedback cycle (epoch {context.get('epoch', 0)})")

        # 1. Training step.
        train_metrics: Dict[str, float] = {}
        if self.trainer is not None:
            train_metrics = self.trainer.train_step(
                context["model"], context["batch"], context
            )

        # 2. Collect metrics.
        report = AnalysisReport()
        for metric in self.metrics:
            result = metric.analyze(context["model"], context)
            report.add_metric_data(metric.name, result)

        # 3. Run analyzers over the metric report.
        for analyzer in self.analyzers:
            result = analyzer.analyze(context["model"], report, context)
            report.add_analyzer_data(analyzer.name, result)

        # 4. Evolution gate. Evaluate the scheduler exactly once per cycle so
        # stateful schedulers cannot disagree with the returned flag.
        evolution_triggered = bool(
            self.scheduler is not None and self.scheduler.should_trigger(context)
        )
        applied_plans: List[Dict[str, Any]] = []
        if evolution_triggered:
            self._log("info", "Evolution triggered")
            for strategy in self.strategies:
                plan = strategy.propose_plan(report, context)
                for evolver in self.evolvers:
                    if evolver.can_execute_plan(plan):
                        self._log("debug", f"Applying evolution with {evolver.name}")
                        outcome = evolver.apply_plan(
                            plan,
                            context["model"],
                            self.trainer,
                            context.get("optimizer"),
                        )
                        applied_plans.append(
                            {
                                "strategy": strategy.name,
                                "evolver": evolver.name,
                                "outcome": outcome,
                            }
                        )
                        break
            self.evolution_history.append(
                {
                    "epoch": context.get("epoch", 0),
                    "step": context.get("step", 0),
                    "applied_plans": len(applied_plans),
                }
            )

        return {
            "train_metrics": train_metrics,
            "analysis_report": report,
            "evolution_triggered": evolution_triggered,
            "applied_plans": applied_plans,
        }

    def get_composition_health(self) -> Dict[str, Any]:
        components = (
            ([self.trainer] if self.trainer is not None else [])
            + self.metrics
            + self.analyzers
            + self.strategies
            + self.evolvers
            + ([self.scheduler] if self.scheduler is not None else [])
        )
        return {
            "status": "ok",
            "component_count": len(components),
            "components": [
                getattr(component, "name", type(component).__name__)
                for component in components
            ],
        }

    def _get_required_inputs(self) -> Set[str]:
        return {"model", "batch", "optimizer"}

    def _get_provided_outputs(self) -> Set[str]:
        return {"training.results", "evolution.history"}
