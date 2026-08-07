#!/usr/bin/env python3
"""
Complete kernel example: infrastructure + a running feedback loop.

Demonstrates the StructureNet kernel end to end on synthetic data:

1. initialize the kernel from a ``KernelConfig``;
2. register and create components (services injected automatically);
3. validate the composition;
4. track an experiment with scorecards and the experiment manager;
5. run the trainer -> metrics -> analyzers -> strategies -> evolvers loop
   through the ``FeedbackOrchestrator``;
6. read back health and event-bus activity.

Everything is self-contained and CPU-only.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

import torch
import torch.nn as nn

from structure_net.components.orchestrators import FeedbackOrchestrator
from structure_net.core import (
    AnalysisReport,
    BaseComponent,
    BaseModel,
    BaseTrainer,
    ComponentContract,
    ComponentVersion,
    EvolutionContext,
    EvolutionPlan,
    IScheduler,
    KernelConfig,
    Maturity,
    StructureNetKernel,
)


class TinyModel(BaseModel):
    """Minimal IModel implementation for the demo."""

    def __init__(self):
        super().__init__(name="TinyModel")
        self.linear = nn.Linear(8, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)

    def get_layers(self):
        return []

    def _get_required_inputs(self) -> Set[str]:
        return set()

    def _get_provided_outputs(self) -> Set[str]:
        return {"model.output"}


class SgdTrainer(BaseTrainer):
    """One-step SGD trainer over (inputs, targets) batches."""

    def _train_step(self, model, batch, context) -> Dict[str, float]:
        inputs, targets = batch
        optimizer = context["optimizer"]
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach())}

    def _get_required_inputs(self) -> Set[str]:
        return {"model", "batch", "optimizer"}

    def _get_provided_outputs(self) -> Set[str]:
        return {"training_metrics"}


class WeightNormMetric(BaseComponent):
    """Example metric: overall weight norm of the model."""

    def __init__(self):
        super().__init__(name="WeightNormMetric")

    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"model"},
            provided_outputs={"metrics.WeightNormMetric"},
        )

    def analyze(self, target: Any, context: EvolutionContext) -> Dict[str, Any]:
        self._logger.debug("Computing weight norm")
        norm = sum(float(p.detach().norm()) for p in target.parameters())
        return {"weight_norm": norm}

    def _get_required_inputs(self) -> Set[str]:
        return {"model"}

    def _get_provided_outputs(self) -> Set[str]:
        return {"metrics.WeightNormMetric"}


class GrowthAnalyzer(BaseComponent):
    """Example analyzer: flags when the weight norm exceeds a bound."""

    def __init__(self, bound: float = 10.0):
        super().__init__(name="GrowthAnalyzer")
        self.bound = bound

    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"metrics.WeightNormMetric"},
            provided_outputs={"analyzers.GrowthAnalyzer"},
        )

    def analyze(
        self, model: Any, report: AnalysisReport, context: EvolutionContext
    ) -> Dict[str, Any]:
        norm = report.get_metric("WeightNormMetric", {}).get("weight_norm", 0.0)
        return {"norm_exceeds_bound": norm > self.bound, "weight_norm": norm}

    def _get_required_inputs(self) -> Set[str]:
        return {"metrics.WeightNormMetric"}

    def _get_provided_outputs(self) -> Set[str]:
        return {"analyzers.GrowthAnalyzer"}


class EveryOtherEpochScheduler(BaseComponent, IScheduler):
    """Triggers evolution on even epochs."""

    def __init__(self):
        super().__init__(name="EveryOtherEpochScheduler")

    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs=set(),
            provided_outputs={"schedule.trigger"},
        )

    def should_trigger(self, context: EvolutionContext) -> bool:
        return context.get("epoch", 0) % 2 == 0

    def get_next_trigger_estimate(self, context: EvolutionContext) -> Optional[int]:
        return 1

    def _get_required_inputs(self) -> Set[str]:
        return set()

    def _get_provided_outputs(self) -> Set[str]:
        return {"schedule.trigger"}


class ShrinkStrategy(BaseComponent):
    """Proposes shrinking weights when the analyzer flags growth."""

    def __init__(self):
        super().__init__(name="ShrinkStrategy")

    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"analyzers.GrowthAnalyzer"},
            provided_outputs={"plan.shrink"},
        )

    def propose_plan(
        self, report: AnalysisReport, context: EvolutionContext
    ) -> EvolutionPlan:
        flagged = report.get_analyzer("GrowthAnalyzer", {}).get(
            "norm_exceeds_bound", False
        )
        plan = EvolutionPlan({"action": "shrink" if flagged else "noop", "factor": 0.9})
        plan.created_by = self.name
        return plan

    def _get_required_inputs(self) -> Set[str]:
        return {"analyzers.GrowthAnalyzer"}

    def _get_provided_outputs(self) -> Set[str]:
        return {"plan.shrink"}


class WeightScaleEvolver(BaseComponent):
    """Applies shrink plans by scaling the model weights."""

    def __init__(self):
        super().__init__(name="WeightScaleEvolver")

    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"plan.shrink"},
            provided_outputs={"evolution.applied"},
        )

    def can_execute_plan(self, plan: EvolutionPlan) -> bool:
        return plan.get("action") in {"shrink", "noop"}

    def apply_plan(
        self, plan: EvolutionPlan, model: Any, trainer: Any, optimizer: Any
    ) -> Dict[str, Any]:
        if plan.get("action") == "shrink":
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.mul_(plan.get("factor", 0.9))
            return {"applied": True}
        return {"applied": False}

    def _get_required_inputs(self) -> Set[str]:
        return {"plan.shrink"}

    def _get_provided_outputs(self) -> Set[str]:
        return {"evolution.applied"}


def main() -> None:
    # 1. Infrastructure.
    kernel = StructureNetKernel(KernelConfig(log_level="INFO", log_handlers=["console"]))
    kernel.event_bus.subscribe(
        "experiment.completed",
        lambda event: print(f"[event] experiment completed: {event.payload}"),
    )

    # 2. Register + create components (services injected by the kernel).
    for component_class in (
        WeightNormMetric,
        GrowthAnalyzer,
        EveryOtherEpochScheduler,
        ShrinkStrategy,
        WeightScaleEvolver,
    ):
        kernel.register_component(component_class)
    metric = kernel.create_component("WeightNormMetric")
    analyzer = kernel.create_component("GrowthAnalyzer")
    scheduler = kernel.create_component("EveryOtherEpochScheduler")
    strategy = kernel.create_component("ShrinkStrategy")
    evolver = kernel.create_component("WeightScaleEvolver")
    trainer = SgdTrainer()

    # 3. Validate the composition.
    issues = kernel.validate_composition([metric, analyzer, strategy, evolver])
    print(f"composition issues: {[str(issue) for issue in issues] or 'none'}")

    # 4. Track the experiment.
    experiment_id = kernel.experiment_manager.start_experiment(
        "kernel_demo", [metric, analyzer, scheduler, strategy, evolver]
    )

    # 5. Run the feedback loop.
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    orchestrator = FeedbackOrchestrator(
        trainer=trainer,
        metrics=[metric],
        analyzers=[analyzer],
        strategies=[strategy],
        evolvers=[evolver],
        scheduler=scheduler,
    )
    generator = torch.Generator().manual_seed(7)
    context = EvolutionContext({"model": model, "optimizer": optimizer, "epoch": 0})
    last_loss = float("nan")
    for epoch in range(4):
        context["epoch"] = epoch
        for _ in range(8):
            inputs = torch.randn(16, 8, generator=generator)
            targets = torch.randint(0, 2, (16,), generator=generator)
            context["batch"] = (inputs, targets)
            results = orchestrator.run_cycle(context)
            last_loss = results["train_metrics"]["loss"]
        print(
            f"epoch {epoch}: loss={last_loss:.4f} "
            f"evolution_triggered={results['evolution_triggered']}"
        )

    # 6. Complete + inspect.
    kernel.experiment_manager.complete_experiment(
        experiment_id, {"final_loss": last_loss}, success=True
    )
    print("health report:", kernel.health_monitor.report())
    print("recent events:", [event.name for event in kernel.event_bus.history][-5:])
    kernel.shutdown()


if __name__ == "__main__":
    main()
