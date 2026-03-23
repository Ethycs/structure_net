from src.structure_net.core.base_components import BaseOrchestrator
from src.structure_net.core.interfaces import (
    IAnalyzer,
    IStrategy,
    IEvolver,
    EvolutionContext,
    AnalysisReport,
    ComponentContract,
    ComponentVersion,
    Maturity,
    ResourceRequirements,
    ResourceLevel,
)
from typing import List
import logging

class EvolutionOrchestrator(BaseOrchestrator):
    """Orchestrates the entire evolution cycle."""

    def __init__(self, analyzers: List[IAnalyzer], strategies: List[IStrategy], evolvers: List[IEvolver], name: str = None):
        super().__init__(name or "EvolutionOrchestrator")
        self.analyzers = analyzers
        self.strategies = strategies
        self.evolvers = evolvers

    @property
    def contract(self) -> ComponentContract:
        """Declares the contract for this component."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            resources=ResourceRequirements(
                memory_level=ResourceLevel.HIGH,
                requires_gpu=True
            ),
        )

    def run_cycle(self, context: EvolutionContext, metrics_report: AnalysisReport) -> EvolutionContext:
        """Runs one full evolution cycle.

        The context must contain at least ``'model'`` (the network).
        Optional keys: ``'optimizer'``, ``'trainer'``, ``'data_loader'``,
        ``'device'``.

        After a successful evolution the context is updated with
        ``'evolution_result'`` and ``'model_modified'`` flags.
        """
        self.log(logging.INFO, "Starting evolution cycle.")

        model = context.network  # EvolutionContext.network -> context['model']
        trainer = context.get('trainer')
        optimizer = context.get('optimizer')

        # 1. Run Analyzers
        for analyzer in self.analyzers:
            try:
                required = analyzer.contract.required_inputs
                # Check requirements against both the report and the context
                available = set(metrics_report.keys()) | set(context.keys())
                if all(req in available for req in required):
                    analysis_data = analyzer.analyze(model, metrics_report, context)
                    metrics_report.add_analyzer_data(analyzer.name, analysis_data)
                    self.log(logging.DEBUG, f"Successfully ran analyzer: {analyzer.name}")
                else:
                    missing = required - available
                    self.log(logging.WARNING, f"Skipping analyzer {analyzer.name}, missing: {missing}")
            except Exception as e:
                self.log(logging.ERROR, f"Analyzer {analyzer.name} failed: {e}")

        # 2. Propose Plans
        plans = []
        for strategy in self.strategies:
            try:
                required = strategy.contract.required_inputs
                available = set(metrics_report.keys()) | set(context.keys())
                if all(req in available for req in required):
                    plan = strategy.propose_plan(metrics_report, context)
                    if plan and len(plan) > 0:
                        plans.append(plan)
                        self.log(logging.DEBUG, f"Strategy {strategy.name} proposed a plan.")
                else:
                    missing = required - available
                    self.log(logging.WARNING, f"Skipping strategy {strategy.name}, missing: {missing}")
            except Exception as e:
                self.log(logging.ERROR, f"Strategy {strategy.name} failed: {e}")

        # 3. Select and Execute Plan
        if plans:
            best_plan = max(plans, key=lambda p: p.priority)
            self.log(logging.INFO, f"Selected plan from {best_plan.created_by} with priority {best_plan.priority:.2f}.")

            for evolver in self.evolvers:
                if evolver.can_execute_plan(best_plan):
                    try:
                        result = evolver.apply_plan(best_plan, model, trainer, optimizer)
                        context['evolution_result'] = result
                        context['model_modified'] = result.get('model_updated', result.get('success', False))
                        self.log(logging.INFO, f"Evolver {evolver.name} executed the plan.")
                        break
                    except Exception as e:
                        self.log(logging.ERROR, f"Evolver {evolver.name} failed to execute plan: {e}")
        else:
            self.log(logging.INFO, "No evolution plans proposed in this cycle.")

        self.log(logging.INFO, "Evolution cycle complete.")
        return context
