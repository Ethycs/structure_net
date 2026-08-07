# StructureNet Kernel (as built)

**Status:** CURRENT  
**Date:** 2026-08-07  
**Applies to:** `structure_net.core.kernel`, `structure_net.core.events`, `structure_net.core.kernel_config`, `structure_net.logging.kernel_logger`, `structure_net.tracking`, `structure_net.components.orchestrators.feedback_orchestrator`, and the optional NAL integration  
**Derived from:** `../09 - Archived/kernel_implementation_guide.md` (aspirational design), reconciled against the component architecture in `../01 - Design/component-architecture.md`  
**Verification:** `pixi run pytest tests/structure_net/test_kernel.py -q`; `pixi run python examples/structure_net/kernel_complete_example.py`

## What the kernel is

`StructureNetKernel` is the infrastructure layer beneath the research
components and the NAL research harness. It owns cross-cutting services so
components do not have to:

```text
StructureNetKernel
├── logging        KernelLogger + per-component KernelComponentLogger
├── profiling      KernelProfiler + per-component ComponentProfiler (existing)
├── events         EventBus (sync pub/sub, wildcard, handler isolation)
├── components     ComponentRegistry + CompatibilityManager (existing)
├── tracking       ScorecardManager, ExperimentManager, ComponentHealthMonitor
└── services       register_service / get_service dependency injection
```

The kernel does **not** replace NAL. NAL remains the scientific-methodology
layer (hypotheses, preregistration, statistical gates);
`NeuralArchitectureLab(config, kernel=...)` optionally reuses the kernel's
logging/profiling, and works unchanged without one.

## Usage

```python
from structure_net.core import KernelConfig, StructureNetKernel

kernel = StructureNetKernel(KernelConfig(log_level="INFO"))
kernel.register_component(MyMetric)          # class must be default-constructible
metric = kernel.create_component("MyMetric") # injects _kernel/_logger/_profiler
issues = kernel.validate_composition([metric, analyzer, evolver])

experiment_id = kernel.experiment_manager.start_experiment("run", [metric])
kernel.experiment_manager.complete_experiment(experiment_id, {"acc": 0.9})
print(kernel.health_monitor.report())
```

The full feedback loop (`trainer -> metrics -> analyzers -> scheduler gate ->
strategies -> evolvers`) is provided by
`structure_net.components.orchestrators.FeedbackOrchestrator`; see
`examples/structure_net/kernel_complete_example.py` for a runnable
end-to-end demonstration on synthetic data.

## Reconciliation with the archived design

The archived guide predates the component migration; this implementation
keeps its service surface but reuses what already existed instead of
duplicating it:

| Archived design element | As built |
| --- | --- |
| `src/core/kernel.py` `StructureNetKernel` | `structure_net/core/kernel.py`, same service methods (`register_component`, `create_component`, `get_logger`, `get_profiler`, `validate_composition`) plus `register_service`/`get_service` and `log_event` |
| `src/core/registry.py`, `contracts.py` | not recreated — `ComponentRegistry`/`ComponentContract` already live in `core/compatibility.py` and `core/interfaces.py` |
| `src/core/events.py` `EventBus` | `structure_net/core/events.py`, with wildcard subscription, bounded history, and handler-failure isolation |
| `src/core/kernel_config.py` `KernelConfig` | `structure_net/core/kernel_config.py`, validated dataclass |
| `src/logging/kernel_logger.py` | `structure_net/logging/kernel_logger.py`; the guide's `%(extra)s` formatter bug is fixed, and context keys can never collide with the logging API (`_emit` takes context as a dict) |
| `ComponentLogger` wrapper | named `KernelComponentLogger` — the name `ComponentLogger` was already taken by the schema-validating experiment logger in `structure_net/logging/component_logger.py` |
| `src/tracking/` scorecards, experiment manager, health monitor | `structure_net/tracking/`; health is graded from observed success rate (healthy >= 95%, degraded >= 80%, failing >= 50%, else critical) |
| `FeedbackOrchestrator` | `structure_net/components/orchestrators/feedback_orchestrator.py`; evaluates the scheduler exactly once per cycle (the guide's draft consulted it twice, which breaks stateful schedulers) |
| `src/integrations/wandb_integration.py`, `chromadb_integration.py` | not recreated — WandB/ChromaDB already have production homes (`structure_net.logging.standardized_logging`, `structure_net.data_factory.search`); attach them via `kernel.register_service` instead |
| `NeuralArchitectureLab(kernel=kernel, config=...)` | `NeuralArchitectureLab(config, kernel=None)` — optional and backward compatible; the `config_adapter` monkey-patch forwards the keyword |

## Boundaries

- Component classes registered with the kernel must be default-constructible
  (the existing registry instantiates once to read the contract).
- `create_component` injection replaces the component's stdlib `_logger` with
  the kernel adapter; the adapter is call-compatible (including
  `log(int_level, message)`), so components that never see a kernel behave
  exactly as before.
- Kernel experiment records are infrastructure bookkeeping, not scientific
  evidence; NAL-STD-EXPERIMENT campaigns continue to use the standardized
  evidence ledger and are unaffected by the kernel.
- No background threads: health checks and event delivery are synchronous
  and deterministic under test.
