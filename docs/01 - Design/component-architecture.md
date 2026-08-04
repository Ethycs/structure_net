# Component Architecture Design

**Status:** CURRENT  
**Date:** 2026-08-03  
**Applies to:** `src/structure_net/core`, `src/structure_net/components`, `src/neural_architecture_lab`  
**Depends on:** `../03 - Architecture/structure-net-overview.md`  
**Derived from:** `docs/New Componentwise refactoring.md`, `integration_plan.md`, `to_integrate.md`

Structure Net is moving from feature-sized monoliths toward small, contract-declaring components coordinated by explicit orchestrators. This document preserves the design intent; the architecture overview records what is actually built, and the refactor roadmap records what remains.

## Non-negotiable design constraints

1. Measurement, interpretation, planning, mutation, and coordination are separate responsibilities.
2. Every new component implements the narrowest appropriate interface and declares a `ComponentContract`.
3. Components communicate through `EvolutionContext`, `AnalysisReport`, and `EvolutionPlan`, rather than hidden shared state.
4. NAL owns experiment methodology and execution; Structure Net owns model construction, measurement, and evolution mechanics.
5. Old entry points remain compatibility surfaces only when they are tested and point clearly to the replacement.
6. A migration is not complete when files merely exist: exports, orchestration, examples, and tests must use the new path.

## 1. Component roles

| Role | Interface/base | Responsibility | Must not own |
| --- | --- | --- | --- |
| Layer | `ILayer` / `BaseLayer` | Tensor transformation and supported structural mutation | Experiment coordination |
| Model | `IModel` / `BaseModel` | Aggregate layers and expose architecture capabilities | Research methodology |
| Metric | `IMetric` / `BaseMetric` | Produce focused measurements | Decide what action to take |
| Analyzer | `IAnalyzer` / `BaseAnalyzer` | Combine measurements into findings | Mutate the model |
| Strategy | `IStrategy` / `BaseStrategy` | Propose an `EvolutionPlan` | Execute structural changes |
| Evolver | `IEvolver` / `BaseEvolver` | Validate and apply a plan | Select research hypotheses |
| Scheduler | `IScheduler` / `BaseScheduler` | Adapt optimization parameters | Coordinate unrelated components |
| Orchestrator | `IOrchestrator` / `BaseOrchestrator` | Order component execution and manage workflow state | Hide domain logic that belongs in a component |

## 2. Contract model

Each component contract declares:

| Field | Meaning |
| --- | --- |
| `component_name` | Stable component identity |
| `version` | Compatibility-relevant version |
| `maturity` | Experimental, stable, or deprecated |
| `required_inputs` | Keys consumed from context or report |
| `provided_outputs` | Keys produced for downstream components |
| `resources` | Memory, GPU, parallel-safety, and runtime expectations |
| compatibility fields | Required or incompatible component types and maturity levels |

**known rough edge:** existing contracts are inconsistent in key naming and completeness. A component marked `STABLE` is not automatically proven stable without conformance tests.

## 3. Canonical workflow

```text
model + run state
      |
      v
MetricsOrchestrator -> AnalysisReport
      |                    |
      |                    v
      |              analyzers add findings
      |                    |
      |                    v
      +------------ strategies propose plans
                           |
                           v
                    orchestrator selects
                           |
                           v
                     evolver applies plan
                           |
                           v
                  updated model/run context
```

## 4. NAL boundary

Tournament evolution demonstrates the intended split:

- `TournamentStrategy` describes the competitors to evaluate.
- `TournamentEvolver` performs crossover and mutation.
- `TournamentOrchestrator` belongs to NAL because it coordinates hypotheses and experiment execution across the lab.
- `tournament_worker.py` evaluates one competitor.
- `ultimate_stress_test_v2.py` is intended to be a thin CLI client.

This refines the original `to_integrate.md` location: the orchestrator now lives at `src/neural_architecture_lab/orchestrators/tournament_orchestrator.py`, not under Structure Net component orchestrators. That relocation is consistent with the ownership boundary but legacy imports and tests must be reconciled.

## 5. Compatibility intent

The desired end state is one recommended component path plus narrow compatibility adapters. Deprecation warnings must name a working replacement. Compatibility modules must not silently become a second implementation, and removal happens only after usage search, replacement examples, and regression tests.

## Verification

Run `rg -n "class .*\((Base|I)(Metric|Analyzer|Strategy|Evolver|Scheduler|Orchestrator|Layer|Model)" src/structure_net src/neural_architecture_lab` and compare each new role with this table; then record implementation gaps in the refactor roadmap.
