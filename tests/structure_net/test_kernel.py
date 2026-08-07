import inspect
from typing import Any, Dict, Optional, Set

import pytest
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
from structure_net.core.events import EventBus
from structure_net.logging.kernel_logger import KernelComponentLogger, KernelLogger
from structure_net.tracking import ComponentHealth


class _Widget(BaseComponent):
    """Minimal default-constructible component for registry tests."""

    def __init__(self, threshold: float = 0.5):
        super().__init__(name="Widget")
        self.threshold = threshold

    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 2, 3),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"model"},
            provided_outputs={"metrics.widget"},
        )

    def _get_required_inputs(self) -> Set[str]:
        return {"model"}

    def _get_provided_outputs(self) -> Set[str]:
        return {"metrics.widget"}


@pytest.fixture()
def kernel():
    instance = StructureNetKernel(KernelConfig(log_handlers=[]))
    yield instance
    instance.shutdown()


def test_event_bus_delivers_wildcards_and_isolates_failures() -> None:
    bus = EventBus()
    seen = []
    bus.subscribe("a", lambda event: seen.append(("a", event.payload["value"])))
    bus.subscribe("*", lambda event: seen.append(("*", event.name)))
    bus.subscribe("a", lambda event: 1 / 0)
    bus.publish("a", {"value": 7})
    bus.publish("b")
    assert ("a", 7) in seen
    assert ("*", "a") in seen and ("*", "b") in seen
    assert len(bus.history) == 2
    handler = lambda event: None  # noqa: E731
    bus.subscribe("c", handler)
    assert bus.unsubscribe("c", handler)
    assert not bus.unsubscribe("c", handler)


def test_kernel_config_validates_inputs() -> None:
    assert KernelConfig().to_dict()["log_level"] == "INFO"
    with pytest.raises(ValueError, match="unknown log handlers"):
        KernelConfig(log_handlers=["syslog"])
    with pytest.raises(ValueError, match="unknown log level"):
        KernelConfig(log_level="LOUD")


def test_kernel_logger_buffers_structured_context() -> None:
    logger = KernelLogger(name="test.kernel.logger", handlers=[])
    adapter = KernelComponentLogger(logger, "MyComponent")
    adapter.info("hello", value=3)
    adapter.error("boom", exception=ValueError("bad"))
    records = logger.records_for("MyComponent")
    assert records[0]["message"] == "hello"
    assert records[0]["context"] == {"value": 3}
    assert records[1]["level"] == "ERROR"
    assert "bad" in records[1]["context"]["exception"]
    # Stdlib-compatible: integer level through .log()
    adapter.log(20, "int level")
    assert logger.records_for("MyComponent")[-1]["level"] == "INFO"
    logger.close()


def test_kernel_registers_creates_and_injects(kernel) -> None:
    kernel.register_component(_Widget)
    widget = kernel.create_component("Widget", threshold=0.9)
    assert widget.threshold == 0.9
    assert widget._kernel is kernel
    assert isinstance(widget._logger, KernelComponentLogger)
    assert widget._profiler is not None
    widget._logger.info("injected", ok=True)
    assert kernel.logger.records_for("Widget")
    assert kernel.scorecard_manager.get_scorecard("Widget") is not None
    with pytest.raises(KeyError, match="not registered"):
        kernel.create_component("Nonexistent")
    events = [event.name for event in kernel.event_bus.history]
    assert "component.registered" in events and "component.created" in events


def test_kernel_service_registry_accepts_types_and_strings(kernel) -> None:
    class DataService:
        pass

    service = DataService()
    kernel.register_service(DataService, service)
    kernel.register_service("cache", {"hits": 0})
    assert kernel.get_service(DataService) is service
    assert kernel.get_service("DataService") is service
    assert kernel.has_service("cache")
    with pytest.raises(KeyError, match="not registered"):
        kernel.get_service("missing")


def test_kernel_validates_composition(kernel) -> None:
    kernel.register_component(_Widget)
    widget = kernel.create_component("Widget")
    issues = kernel.validate_composition([widget])
    assert isinstance(issues, list)


def test_scorecards_track_executions_and_health(kernel) -> None:
    kernel.register_component(_Widget)
    widget = kernel.create_component("Widget")
    manager = kernel.scorecard_manager
    for _ in range(19):
        manager.update_execution("Widget", 0.010, success=True)
    card = manager.get_scorecard("Widget")
    assert card.health is ComponentHealth.HEALTHY
    assert card.average_execution_time == pytest.approx(0.010)
    manager.update_execution("Widget", 0.030, success=False, message="oops")
    assert card.total_executions == 20
    assert card.success_rate == pytest.approx(0.95)
    assert card.health is ComponentHealth.HEALTHY
    for _ in range(3):
        manager.update_execution("Widget", 0.010, success=False)
    # 19/23 ~= 0.826: below the 0.95 healthy floor, above the 0.80 degraded floor.
    assert card.health is ComponentHealth.DEGRADED
    assert card.last_error == "oops"
    assert "Widget" in kernel.health_monitor.unhealthy_components()
    report = kernel.health_monitor.report()
    assert report["total_components"] == 1
    assert not report["system_healthy"]


def test_experiment_manager_lifecycle(kernel) -> None:
    kernel.register_component(_Widget)
    widget = kernel.create_component("Widget")
    experiment_id = kernel.experiment_manager.start_experiment("demo", [widget])
    assert kernel.experiment_manager.get_experiment(experiment_id).status == "running"
    kernel.experiment_manager.update_metrics(experiment_id, {"accuracy": 0.5})
    record = kernel.experiment_manager.complete_experiment(
        experiment_id, {"accuracy": 0.9}, success=True
    )
    assert record.status == "completed"
    assert record.metrics["accuracy"] == 0.9
    assert record.best_performance == pytest.approx(0.9)
    assert record.component_versions["Widget"] == "1.2.3"
    assert kernel.experiment_manager.get_experiment(experiment_id) is record
    card = kernel.scorecard_manager.get_scorecard("Widget")
    assert card.experiment_count == 1
    assert card.last_experiment_id == experiment_id


class _Model(BaseModel):
    def __init__(self):
        super().__init__(name="Model")
        self.linear = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)

    def get_layers(self):
        return []

    def _get_required_inputs(self) -> Set[str]:
        return set()

    def _get_provided_outputs(self) -> Set[str]:
        return set()


class _Trainer(BaseTrainer):
    def _train_step(self, model, batch, context) -> Dict[str, float]:
        return {"loss": 1.0}

    def _get_required_inputs(self) -> Set[str]:
        return set()

    def _get_provided_outputs(self) -> Set[str]:
        return set()


class _Metric(BaseComponent):
    def __init__(self):
        super().__init__(name="Metric")

    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs=set(),
            provided_outputs={"metrics.Metric"},
        )

    def analyze(self, target: Any, context: EvolutionContext) -> Dict[str, Any]:
        return {"value": 41}

    def _get_required_inputs(self) -> Set[str]:
        return set()

    def _get_provided_outputs(self) -> Set[str]:
        return {"metrics.Metric"}


class _Analyzer(BaseComponent):
    def __init__(self):
        super().__init__(name="Analyzer")

    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"metrics.Metric"},
            provided_outputs={"analyzers.Analyzer"},
        )

    def analyze(self, model, report: AnalysisReport, context) -> Dict[str, Any]:
        return {"value": report.get_metric("Metric")["value"] + 1}

    def _get_required_inputs(self) -> Set[str]:
        return {"metrics.Metric"}

    def _get_provided_outputs(self) -> Set[str]:
        return {"analyzers.Analyzer"}


class _Scheduler(BaseComponent, IScheduler):
    def __init__(self, trigger: bool = True):
        super().__init__(name="Scheduler")
        self.trigger = trigger
        self.calls = 0

    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs=set(),
            provided_outputs=set(),
        )

    def should_trigger(self, context) -> bool:
        self.calls += 1
        return self.trigger

    def get_next_trigger_estimate(self, context) -> Optional[int]:
        return None

    def _get_required_inputs(self) -> Set[str]:
        return set()

    def _get_provided_outputs(self) -> Set[str]:
        return set()


class _Strategy(BaseComponent):
    def __init__(self):
        super().__init__(name="Strategy")

    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs=set(),
            provided_outputs=set(),
        )

    def propose_plan(self, report: AnalysisReport, context) -> EvolutionPlan:
        return EvolutionPlan({"action": "noop"})

    def _get_required_inputs(self) -> Set[str]:
        return set()

    def _get_provided_outputs(self) -> Set[str]:
        return set()


class _Evolver(BaseComponent):
    def __init__(self):
        super().__init__(name="Evolver")
        self.applied = 0

    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs=set(),
            provided_outputs=set(),
        )

    def can_execute_plan(self, plan: EvolutionPlan) -> bool:
        return True

    def apply_plan(self, plan, model, trainer, optimizer) -> Dict[str, Any]:
        self.applied += 1
        return {"applied": True}

    def _get_required_inputs(self) -> Set[str]:
        return set()

    def _get_provided_outputs(self) -> Set[str]:
        return set()


def _orchestrator(trigger: bool = True):
    scheduler = _Scheduler(trigger=trigger)
    evolver = _Evolver()
    orchestrator = FeedbackOrchestrator(
        trainer=_Trainer(),
        metrics=[_Metric()],
        analyzers=[_Analyzer()],
        strategies=[_Strategy()],
        evolvers=[evolver],
        scheduler=scheduler,
    )
    return orchestrator, scheduler, evolver


def test_feedback_orchestrator_runs_full_cycle() -> None:
    orchestrator, scheduler, evolver = _orchestrator(trigger=True)
    context = EvolutionContext(
        {"model": _Model(), "batch": None, "optimizer": None, "epoch": 3}
    )
    results = orchestrator.run_cycle(context)
    assert results["train_metrics"] == {"loss": 1.0}
    assert results["analysis_report"].get_metric("Metric") == {"value": 41}
    assert results["analysis_report"].get_analyzer("Analyzer") == {"value": 42}
    assert results["evolution_triggered"] is True
    assert results["applied_plans"][0]["evolver"] == "Evolver"
    assert evolver.applied == 1
    # The scheduler is consulted exactly once per cycle.
    assert scheduler.calls == 1
    assert orchestrator.evolution_history[0]["epoch"] == 3
    health = orchestrator.get_composition_health()
    assert health["component_count"] == 6


def test_feedback_orchestrator_respects_scheduler_gate() -> None:
    orchestrator, scheduler, evolver = _orchestrator(trigger=False)
    context = EvolutionContext({"model": _Model(), "batch": None, "optimizer": None})
    results = orchestrator.run_cycle(context)
    assert results["evolution_triggered"] is False
    assert results["applied_plans"] == []
    assert evolver.applied == 0
    assert orchestrator.evolution_history == []


def test_nal_accepts_an_optional_kernel() -> None:
    from neural_architecture_lab.lab import NeuralArchitectureLab

    # ``config_adapter`` monkey-patches ``__init__`` with a wrapper that must
    # forward extra keyword arguments (like ``kernel``) to the original.
    parameters = inspect.signature(NeuralArchitectureLab.__init__).parameters
    assert "kernel" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    # The unwrapped constructor carries the explicit optional parameter.
    import neural_architecture_lab.lab as lab_module

    source = inspect.getsource(lab_module.NeuralArchitectureLab)
    assert "kernel: Optional[Any] = None" in source
