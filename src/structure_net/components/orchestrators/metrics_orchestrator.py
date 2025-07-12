from src.structure_net.core.base_components import BaseOrchestrator
from src.structure_net.core.interfaces import (
    IComponent,
    IMetric,
    IAnalyzer,
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

class MetricsOrchestrator(BaseOrchestrator):
    """Orchestrates the collection of metrics from various metric components."""

    def __init__(self, metrics: List[IMetric], name: str = None):
        super().__init__(name or "MetricsOrchestrator")
        self.metrics = metrics

    @property
    def contract(self) -> ComponentContract:
        """Declares the contract for this component."""
        required_inputs = set()
        for metric in self.metrics:
            required_inputs.update(metric.contract.required_inputs)
        
        provided_outputs = set()
        for metric in self.metrics:
            provided_outputs.update(metric.contract.provided_outputs)

        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs=required_inputs,
            provided_outputs=provided_outputs,
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=False
            ),
        )

    def run_cycle(self, context: EvolutionContext) -> AnalysisReport:
        """Run all metric components and return a consolidated report."""
        report = AnalysisReport()
        self.log(logging.INFO, f"Starting metrics collection cycle with {len(self.metrics)} components.")

        for metric in self.metrics:
            try:
                if all(req in context for req in metric.contract.required_inputs):
                    metric_data = metric.analyze(context.network, context)
                    report.add_metric_data(metric.name, metric_data)
                    self.log(logging.DEBUG, f"Successfully ran metric: {metric.name}")
                else:
                    self.log(logging.WARNING, f"Skipping metric {metric.name} due to missing inputs.")
            except Exception as e:
                self.log(logging.ERROR, f"Metric {metric.name} failed: {e}")

        self.log(logging.INFO, "Metrics collection cycle complete.")
        return report
