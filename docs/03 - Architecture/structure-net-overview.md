# Structure Net Overview

**Status:** CURRENT  
**Date:** 2026-08-03  
**Applies to:** `src/structure_net`, `src/neural_architecture_lab`  
**Supersedes as trusted overview:** `docs/NAL_ARCHITECTURE.md` and architectural claims spread across the root `README.md`

**Active change program:** `../06 - Roadmaps/component-refactor.md`; design intent lives in `../01 - Design/component-architecture.md`.

Structure Net is a research codebase for sparse neural-network construction, measurement, and structural evolution. Its newer component architecture coexists with an older evolution stack, while Neural Architecture Lab (NAL) provides a hypothesis-and-experiment harness above both. Configuration, data, logging, profiling, and snapshots form cross-cutting infrastructure rather than the evolution algorithm itself.

## 1. Architecture

```text
experiments / examples / scripts
              |
              v
 Neural Architecture Lab (hypotheses, runners, workers, orchestration)
              |
              v
 component orchestrators -> analyzers/metrics -> strategies/evolvers/trainers
              |                                      |
              +-------------- contexts/plans --------+
                                 |
                                 v
                 models and sparse layer primitives

 Cross-cutting: unified config | data factory | logging | profiling | snapshots
 Compatibility: legacy `structure_net.evolution` modules remain beside components
```

## 2. Core contracts

| Domain | Runs at | Inputs | Outputs | Structure | Hard constraint |
| --- | --- | --- | --- | --- | --- |
| Layers/models | Forward and mutation time | Tensors, masks, architecture | Activations, structure summaries | `components/layers`, `components/models`, `core/layers.py` | PyTorch module semantics must survive mutation |
| Metrics | Observation time | Layer/model plus `EvolutionContext` | Measurement dictionaries | `components/metrics` | A metric should measure, not choose a mutation |
| Analyzers | Decision preparation | Model, measurements, context | Higher-level findings | `components/analyzers` | Analysis is distinct from action selection |
| Strategies/evolvers | Evolution events | Reports/context | `EvolutionPlan` or structural changes | `components/strategies`, `components/evolvers` | Mutations must match target model capabilities |
| Orchestrators | Experiment/evolution loop | Components and context | Ordered execution and results | `components/orchestrators` | Current preferred coordination surface |

The explicit contract vocabulary lives in `core/interfaces.py`: `ComponentContract`, `EvolutionContext`, `AnalysisReport`, `EvolutionPlan`, and component ABCs. **known rough edge:** not every older implementation conforms to these newer interfaces.

## 3. Neural Architecture Lab

NAL owns research methodology: hypotheses, experiment definitions/results, runners, resource-aware workers, aggregate analysis, and follow-up hypothesis generation. It consumes Structure Net capabilities but is packaged as a sibling Python package. Its principal modules are `core.py`, `lab.py`, `runners.py`, `analyzers.py`, `workers/`, `seed_search/`, and `orchestrators/`.

## 4. Infrastructure

| Subsystem | Inputs | Outputs | Target |
| --- | --- | --- | --- |
| Configuration | defaults, environment, files | structured storage/compute/logging/experiment settings | one adaptable configuration surface |
| Data factory | dataset name and loader options | loaders, metadata, optional search/time-series records | reproducible dataset access |
| Logging | experiment and component events | validated records, queues, WandB integration | durable experiment evidence |
| Profiling | instrumented operations | timing/resource/health records | locate operational bottlenecks |
| Snapshots | model/evolution state | restorable artifacts | preserve state across structural change |

## 5. Compatibility boundary

There are parallel implementations under `components/` and `evolution/`. `evolution/components/evolution_system.py` declares itself deprecated and directs callers to component orchestrators. The legacy directory cannot yet be described as removed: examples, exports, and compatibility modules still refer to it.

**BUG:** `src/structure_net/__init__.py` advertises names in `__all__` that are not bound in the module (`analyze_layer_extrema`, `detect_network_extrema`, and `ArchitectureGenerator`). The package docstring also calls the project “Multi-Scale Snapshots Neural Network,” while project metadata uses `structure_net` and the README describes dynamic evolution.

## Design axioms

1. Measurement, interpretation, planning, and mutation are separate responsibilities.
2. Structural mutation must preserve valid PyTorch models and optimizer/training semantics.
3. Experiments need reproducible configuration, datasets, logging, and saved evidence.
4. New coordination work targets `components/orchestrators`; legacy evolution paths are compatibility surfaces until usage is removed and tested.
5. Presence of code is not proof of integration; tests or a runnable example must establish that claim.

## Verification

Run `find src/structure_net src/neural_architecture_lab -maxdepth 2 -type d | sort` and `rg -n "DEPRECATED|deprecated" src/structure_net/evolution src/structure_net/components`; then reconcile any newly added subsystem or changed compatibility boundary here.
