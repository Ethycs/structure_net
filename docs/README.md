# Structure Net Documentation

**Status:** CURRENT  
**Date:** 2026-08-03  
**Applies to:** repository documentation and the Python packages under `src/`  
**Depends on:** `Agentic Technique Master.md`

This tree separates foundations, intent, implementation, as-built architecture, contracts, plans, snapshots, and measured analysis. The code remains authoritative where a legacy document and implementation disagree.

## Start here

1. Read [`03 - Architecture/structure-net-overview.md`](03%20-%20Architecture/structure-net-overview.md) for the current system map.
2. Read [`02 - Implementation/developer-guide.md`](02%20-%20Implementation/developer-guide.md) to install, inspect, and test the repository.
3. For the completed component migration, read [`01 - Design/component-architecture.md`](01%20-%20Design/component-architecture.md), [`07 - Status Reports/2026-08-03_component-migration-complete.md`](07%20-%20Status%20Reports/2026-08-03_component-migration-complete.md), then use [`06 - Roadmaps/component-refactor.md`](06%20-%20Roadmaps/component-refactor.md) for remaining retirement work.
4. Use [`02 - Implementation/component-migration-guide.md`](02%20-%20Implementation/component-migration-guide.md) for each migration slice.
5. Consult [`04 - Reference/document-inventory.md`](04%20-%20Reference/document-inventory.md) before trusting a pre-existing guide.
6. Use [`06 - Roadmaps/documentation-reconciliation.md`](06%20-%20Roadmaps/documentation-reconciliation.md) for broader documentation gaps.
7. Read [`08 - Analysis/2026-08-03_runner-experiment-data-modernization.md`](08%20-%20Analysis/2026-08-03_runner-experiment-data-modernization.md) for the runner, experiment-schema, and data-story assessment.
8. Check [`000 - Doc Maintenance/2026-08-03_docs-tree-gaps.md`](000%20-%20Doc%20Maintenance/2026-08-03_docs-tree-gaps.md) for known drift.

## Governance

| Folder | Purpose | Policy |
| --- | --- | --- |
| `000 - Doc Maintenance` | Visible documentation drift | Append-only observations |
| `00 - Theory` | Mathematical and conceptual foundations | Correct, do not casually redesign |
| `01 - Design` | Intended shape and rationale | Archive when superseded |
| `02 - Implementation` | Setup, tools, and realized specs | Reconcile with code |
| `03 - Architecture` | Single trusted as-built map | Keep current |
| `04 - Reference` | Human-facing and imported reference | Living |
| `05 - Standards` | Normative project contracts | Freeze only after adoption |
| `06 - Roadmaps` | Ordered plans and gaps | Living |
| `07 - Status Reports` | Dated resumable snapshots | Archive when complete |
| `08 - Analysis` | Measured technical studies | Record method and date |
| `09 - Archived` | Superseded snapshots | Append-only |

`DRAFT` directory notes below are classification boundaries, not claims that each lane already has mature content.

## Verification

Run `find docs -maxdepth 2 -type f -name '*.md' | sort` to inspect the tree, then use the architecture overview's verification commands to compare it with the package layout.
