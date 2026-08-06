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

### TinyLLM architecture family

`components/models/tinyllm_model.py` implements the GPT-2 presets used by `weiserlab/TinyLLM`, direct import of its pinned `llm.c` v3/v5 checkpoints, and Hugging Face GPT-2 weight translation. Optional sparse feedback patches create delayed later-block-to-earlier-block edges evaluated through explicit refinement passes. `RandomFeedbackGrowthStrategy` owns random placement plans and `FeedbackGrowthEvolver` applies them while preserving existing optimizer state. Feedback models remain PyTorch research graphs; unmodified GPT-2, GGUF, and `llama.cpp` runtimes cannot represent their recurrent execution. The realized API and verification command are documented in `tinyllm-feedback-adapter.md`.

`components/analyzers/semantic_quotient_analyzer.py` supplies task-relative Fisher--Rao distances, Euclidean/cosine baselines, k-nearest-neighbor geodesics for externally computed pullback metrics, Ripser `H0/H1` diagrams, seeded bootstrap summaries, circular-map alignment and degree, nuisance-fiber collapse ratios, and persistent-cocycle circular coordinates. It does not infer a semantic quotient from arbitrary activations without an experiment-defined sampling/readout contract. The predictive-circle, matched circle-versus-interval, and frozen internal-probe runners define their quotient, designated readout, and controls; their measured verdicts are recorded in the dated TinyLLM reports under `../08 - Analysis/`.

The experimental task-geometry atlas adds paired distance correlation, scaled stress, neighborhood recall, local decoders, and sublayer tracing. It records operational carrier/retract/fiber proxies while explicitly withholding chain-level induced-map, homotopy-retract, and Reeb claims. Its as-built contract is documented in `task-geometry-atlas.md`.

The degree–defect cobordism analyzer measures winding change and indexed posterior-moment zero cells on a periodic phase/path grid. The TinyLLM runner uses a declared continuous adjacent-token-embedding lift and deterministically replayed optimizer states; it does not treat the hard quantizer as a smooth map or present grid localization as certified root isolation. The contract and mathematical boundary are documented in `degree-defect-cobordism.md`.

The depth-graded TinyLLM API evaluates exact integer prefixes and continuously gated partial residual blocks through one shared head. Matched ordinary, integer multi-exit, and real-depth training arms expose task fronts and depth-wise defect charge without claiming a neural-ODE limit or Reeb construction. The as-built contract is documented in `depth-graded-transformer.md`.

## 3. Neural Architecture Lab

NAL owns research methodology: hypotheses, experiment definitions/results, runners, resource-aware workers, aggregate analysis, and follow-up hypothesis generation. It consumes Structure Net capabilities but is packaged as a sibling Python package. Its principal modules are `core.py`, `lab.py`, `runners.py`, `analyzers.py`, `workers/`, `seed_search/`, and `orchestrators/`.

The canonical local runner schedules independent experiments onto logical CUDA
device slots using CUDA-safe spawned processes. Slot counts may be fixed or
derived from free memory and an experiment estimate. Successful experiments
can be resumed from an atomic fingerprinted ledger, and failed attempts have a
bounded retry policy. Physical GPU visibility remains a parent-launcher
decision; NAL never silently rewrites it. See `nal-local-gpu-scheduler.md`.

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
