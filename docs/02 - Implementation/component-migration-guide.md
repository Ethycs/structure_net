# Component Migration Guide

**Status:** CURRENT  
**Date:** 2026-08-03  
**Applies to:** migrations from `src/structure_net/evolution` and other monolithic modules into the component architecture  
**Depends on:** `../01 - Design/component-architecture.md`, `../06 - Roadmaps/component-refactor.md`

Use this procedure for each remaining migration. It consolidates the operational instructions previously split across `integration_plan.md`, `to_integrate.md`, and the large componentwise refactoring guide.

## Hard constraints

1. Preserve externally observable behavior before deleting or rewriting the old implementation.
2. Extract one responsibility at a time; do not copy a monolith wholesale into a class with a new base type.
3. Add or identify behavioral tests before marking a migration complete.
4. Update exports and every active caller in the same migration slice.
5. Deprecation messages must point to an import path that collection tests prove works.

## 1. Inventory the source

For a monolithic analyzer or evolution service:

1. List low-level computations, high-level decisions, mutations, coordination, and persistence separately.
2. Map each responsibility to the component role table in the design document.
3. Search active callers, tests, examples, and package exports.
4. Capture current deterministic behavior with regression or golden tests where possible.

```bash
rg -n "<OldClass|old_module>" src tests examples experiments
```

## 2. Extract components

| Source responsibility | Destination |
| --- | --- |
| Focused measurement | `src/structure_net/components/metrics/` |
| Combined interpretation | `src/structure_net/components/analyzers/` |
| Action selection | `src/structure_net/components/strategies/` |
| Structural mutation | `src/structure_net/components/evolvers/` |
| Optimization scheduling | `src/structure_net/components/schedulers/` |
| Structure Net workflow | `src/structure_net/components/orchestrators/` |
| Cross-experiment/NAL workflow | `src/neural_architecture_lab/orchestrators/` |
| Per-experiment evaluation | `src/neural_architecture_lab/workers/` |

Each extracted class must inherit from the corresponding base, implement the required interface methods, and declare its contract.

## 3. Define the data contract

Before wiring orchestration, specify exact `required_inputs` and `provided_outputs`. Use stable dotted keys for report data and test that the declared keys match what `analyze`, `propose_plan`, or `apply_plan` actually reads and writes.

**known rough edge:** the current `EvolutionOrchestrator` checks report keys directly, so mismatched names cause components to be skipped rather than failing loudly.

## 4. Integrate

1. Export the new class from its component package.
2. Add it to the appropriate orchestrator or client.
3. Update active examples and experiments to import the canonical package, not `src.<package>` unless the project deliberately retains that convention.
4. Keep the old import as a thin adapter only when backward compatibility is required.
5. Add a deprecation warning with a tested replacement example.

## 5. Validate a migration slice

A slice is complete only when all rows pass:

| Evidence | Acceptance criterion |
| --- | --- |
| Contract | Required/provided keys and resources are declared and exercised |
| Unit behavior | New component matches pinned old behavior or an approved changed contract |
| Composition | Orchestrator runs the component with a realistic context/report |
| Public import | Canonical import works; compatibility import warns and works if retained |
| Caller migration | Active callers no longer require the old implementation |
| Documentation | Architecture, roadmap status, and migration mapping are updated |

## 6. Deprecate and remove

First warn, then migrate callers, then remove. Before removal:

```bash
rg -n "<OldClass|old_module>" src tests examples experiments README.md docs
```

Classify every remaining hit as active usage, compatibility test, historical documentation, or archive. Never use an archive hit alone to keep production code alive.

## Current baseline

Metric/analyzer extraction is recorded as complete in `src/structure_net/evolution/metrics/MIGRATION_STATUS.md`. Integration is not complete: active collection currently stops on missing exports, a model-interface MRO conflict, and a stale tournament API expectation. See the refactor roadmap for the ordered repair plan.

## Verification

Run `env PIXI_CACHE_DIR=/tmp/structure-net-pixi-cache UV_CACHE_DIR=/tmp/structure-net-uv-cache pixi run pytest tests --collect-only -q`; a clean migration baseline requires zero collection errors before behavioral failures can be evaluated.
