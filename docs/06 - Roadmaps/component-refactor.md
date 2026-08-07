# Component Refactor Roadmap

**Status:** MIGRATION TARGET COMPLETE — 2026-08-03  
**Applies to:** contract-driven components, NAL execution, experiment persistence, and tournament integration  
**Depends on:** `../01 - Design/component-architecture.md`, `../02 - Implementation/component-migration-guide.md`, `../03 - Architecture/structure-net-overview.md`  
**Completion report:** `../07 - Status Reports/2026-08-03_component-migration-complete.md`  
**Consolidates:** `integration_plan.md`, `to_integrate.md`, and `docs/09 - Archived/New Componentwise refactoring.md`

## Outcome

The active migration target is complete. The component framework now has runnable contracts, focused metric and analyzer implementations, working snapshot and compactification paths, a canonical NAL worker/result protocol, versioned hybrid persistence, and an asynchronous tournament cycle. Active pytest discovery is restricted to `tests/` and completes without collection errors.

This roadmap remains open only for deliberate legacy retirement. Those cleanup items do not block the component architecture or the canonical CPU experiment lifecycle.

## Completed migration tiers

| Tier | Result | Evidence |
| --- | --- | --- |
| Evidence baseline | ✅ Complete | component MRO, `ComponentStatus`, Chroma naming, tournament tests, and pytest discovery repaired |
| Component contracts | ✅ Complete | layer/model inheritance, output schemas, report namespaces, strategy/evolver protocols, and public exports reconciled |
| Orchestration | ✅ Complete for target | snapshots, compactification, runner lifecycle, and tournament orchestration have focused passing tests |
| Runner and experiment schema | ✅ Complete for local backends | canonical `ExperimentWorker(Experiment, device_id) -> ExperimentResult`; legacy `(model, metrics)` adapter retained |
| Data lifecycle | ✅ Complete | direct-ID result envelope, Chroma search aliases, JSON/HDF5 history round-trip, and memory-efficient NAL integration |
| Validation | ✅ Complete | repository-wide active suite passes; exact evidence is recorded in the completion report |

## Canonical decisions

1. Packaged imports are `structure_net.*` and `neural_architecture_lab.*`. Compatibility modules may retain old import paths while callers migrate.
2. `Experiment`, `ExperimentResult`, and `HypothesisResult` are the canonical in-memory experiment types.
3. Every execution backend consumes an `ExperimentWorker`; hypothesis functions using the old parameter-dictionary return shape are adapted at the boundary.
4. Experiment search metadata uses a versioned envelope. Large histories live in HDF5; small histories live in compressed JSON; both round-trip through one time-series API.
5. Tournament ownership is split intentionally: strategy/evolver components live in Structure Net, while experiment execution orchestration lives in NAL.
6. Generic component matrices validate universal contracts. Focused suites own behavioral and domain-specific assertions.

## Remaining retirement work

These are follow-on maintenance tasks, not unfinished migration blockers:

1. Keep active source, tests, and experiments free of `src.*` imports; historical archived documents may retain old paths as provenance.
2. Decide whether each legacy evolution module is a supported shim or removable duplication; remove only after caller inventory.
3. Replace class-based Pydantic configuration before Pydantic 3.
4. Set an explicit safe-loading policy for snapshot `torch.load` calls.
5. Either support Ray as an optional tested backend or keep `ray_compatibility/` explicitly experimental. Ray is not part of the canonical path.
6. Retire stale specialized pipeline helper definitions once focused suites cover every retained diagnostic.

## Acceptance gate

Run:

```bash
env PIXI_CACHE_DIR=/tmp/structure-net-pixi-cache \
    UV_CACHE_DIR=/tmp/structure-net-uv-cache \
    pixi run pytest -q
```

Record future changes against the exact count and warnings in the latest status report. A new migration phase must name its scope explicitly rather than reopening this completed target implicitly.
