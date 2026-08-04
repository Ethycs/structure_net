# Docs-tree Gaps — 2026-08-03

**Status:** CURRENT  
**Date:** 2026-08-03  
**Applies to:** documentation tree health; observational note, not a plan

## The tree today

```text
docs/
├── 000 - Doc Maintenance  [current drift recorded here]
├── 00 - Theory            [placeholder; legacy candidates unverified]
├── 01 - Design            [component refactor reconciled; other proposals mixed]
├── 02 - Implementation    [developer and component migration guides]
├── 03 - Architecture      [initial trusted overview]
├── 04 - Reference         [ambient-doc inventory]
├── 05 - Standards         [empty by design; no frozen contract]
├── 06 - Roadmaps          [documentation and component-refactor plans]
├── 07 - Status Reports    [documentation and refactor-recovery handoffs]
├── 08 - Analysis          [placeholder; no results re-run]
└── 09 - Archived          [index only; old paths not moved]
```

## Gap 1 — Flat legacy documents remain beside the tree

They are classified but unreconciled. Cheap fix: promote one subsystem guide per code-changing session after checking its commands and paths. Alternative: bulk-moving them would produce a tidy but untrustworthy tree.

## Gap 2 — The public README conflicts with code boundaries

Its quick start imports a deprecated evolution surface and its NAL example uses `src.neural_architecture_lab`. Cheap fix: test and rewrite the examples after resolving the public API. See the roadmap.

## Gap 3 — Test evidence is not yet trustworthy

Unscoped pytest discovery traverses `archive/` and executes import-time script behavior. Cheap fix: scope discovery in project configuration, then capture a dated baseline.

## Gap 4 — Historical material has two archive locations

`archive/` and `docs/old docs/` predate the numbered docs archive. Cheap fix: retain them until a link audit, then index rather than duplicate large artifacts.

## Gap 5 — Reconciled refactor source documents remain outside the numbered tree

`integration_plan.md`, `to_integrate.md`, and `docs/New Componentwise refactoring.md` now have current replacements but retain provenance and incoming-link value. Cheap fix: migrate links, then archive snapshots and update `ARCHIVE-INDEX.md` rather than deleting them immediately.

## Why this note exists

The initial tree is a truthful navigation layer, not a claim that all ambient prose has been reconciled. These markers prevent legacy prose from acquiring authority merely because it is nearby.

## Suggested cadence

Add a dated maintenance note whenever code changes without its architecture or implementation guide changing. Close gaps in the roadmap; never rewrite this observation after the fact.

## Decision log

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-08-03 | Synthesize before relocating legacy docs | Preserve provenance and avoid blessing stale claims |
| 2026-08-03 | Make the numbered component-refactor roadmap authoritative | Original plans describe completed file creation and no longer represent current state |

## Verification

Run `find docs -maxdepth 2 -type f -name '*.md' | sort`, compare it with `../04 - Reference/document-inventory.md`, and add a new dated note for newly observed drift.
