# Docs-root Relocation — 2026-08-07

**Status:** CURRENT  
**Date:** 2026-08-07  
**Applies to:** filing of the twelve loose `docs/`-root files into the numbered lanes; observational record of what moved, why, and with what verified trust

## What happened

Every loose file in the `docs/` root except `README.md` (the lane index) was
audited against the codebase and then filed. The audit preceded the moves, so
this is the "synthesize before relocating" path the 2026-08-03 decision log
required, not a bulk move: each relocated file carries a status block recording
its verified trust, and archived files state what superseded them.

## The tree today

```text
docs/
├── README.md                 [lane index; only file remaining at the root]
├── 000 - Doc Maintenance     [this note supersedes the 2026-08-03 tree sketch]
├── 00 - Theory               [+ METHODOLOGY.md (LEGACY, split still pending)]
├── 01 - Design               [+ LEARNING_RATE_STRATEGIES.md (LEGACY catalogue)]
├── 02 - Implementation       [+ CONFIG_MIGRATION_GUIDE.md (CURRENT) and the five
│                              logging/data/profiling/storage guides (LEGACY/MIXED,
│                              import paths corrected, drift itemized per file)]
├── 03 - Architecture         [unchanged]
├── 04 - Reference            [document-inventory.md rewritten for new locations]
├── 05 - Standards            [unchanged]
├── 06 - Roadmaps             [links updated]
├── 07 - Status Reports       [unchanged]
├── 08 - Analysis             [unchanged]
└── 09 - Archived             [+ NAL_ARCHITECTURE.md, CONTRIBUTING.md,
                               kernel_implementation_guide.md,
                               New Componentwise refactoring.md; ARCHIVE-INDEX
                               now has four rows]
```

## Moves and dispositions

| File | From `docs/` to | Verified trust at move time |
| --- | --- | --- |
| `METHODOLOGY.md` | `00 - Theory/` | Legacy compendium of the deprecated `evolution/` stack; fixed-threshold saturation error noted; credit-based growth never implemented |
| `LEARNING_RATE_STRATEGIES.md` | `01 - Design/` | 10/12 strategies exist, but only in the deprecated `adaptive_learning_rates` package |
| `CONFIG_MIGRATION_GUIDE.md` | `02 - Implementation/` | Current; API verified field-for-field; adoption of the migration remains partial |
| `logging_guide.md` | `02 - Implementation/` | API real; `src.logging` imports corrected; queue paths and WandB pipeline claims stale |
| `standardized_logging_system.md` | `02 - Implementation/` | Schemas accurate; omits the live ChromaDB backend; WandB half disabled in practice |
| `storage_formats_guide.md` | `02 - Implementation/` | NAL/JSON section verified; `NALChromaIntegration` path corrected; stress-test section false |
| `profiling_data_guide.md` | `02 - Implementation/` | §1 layout + `ProfiledOperation` valid; the proposed storage/search classes were never built |
| `data_system_integration_guide.md` | `02 - Implementation/` | Search/time-series verified; dataset/metadata sections drifted; paths corrected |
| `NAL_ARCHITECTURE.md` | `09 - Archived/` | Built on deleted `TournamentExecutor`; superseded by the trusted overview |
| `CONTRIBUTING.md` | `09 - Archived/` | Directory map names three nonexistent directories; routes code into deprecated shims |
| `kernel_implementation_guide.md` | `09 - Archived/` | Raw design-conversation transcript; microkernel never implemented |
| `New Componentwise refactoring.md` | `09 - Archived/` | Already self-marked superseded; Phases 3–4 never built |

## Link migration

Inbound references were updated before or with the moves:
`03 - Architecture/structure-net-overview.md` (supersession pointer),
`06 - Roadmaps/documentation-reconciliation.md` (METHODOLOGY action),
`01 - Design/component-architecture.md` and `06 - Roadmaps/component-refactor.md`
(refactoring-guide provenance), `src/structure_net/config/README.md`
(migration-guide link), and `src/structure_net/logging/README.md`
(logging-guide links). The 2026-08-03 gaps note was left untouched per its
append-only rule; its tree sketch and Gap 1/Gap 5 status are superseded by
this note. Archived snapshots were not link-edited beyond their added status
blocks, so stale links inside them are expected.

## What this does not claim

Filing is not reconciliation. The five subsystem guides in
`02 - Implementation` remain LEGACY/MIXED until their flagged sections are
rewritten; `METHODOLOGY.md` still awaits its split into validated theory and
design hypotheses; the repo-root `README.md`, `integration_plan.md`, and
`to_integrate.md` were out of scope and remain where they were.

## Decision log

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-08-07 | File all docs-root prose after a per-claim code audit | The 2026-08-03 "synthesize before relocating" condition was satisfied by the audit |
| 2026-08-07 | Correct only mechanically verified import paths in promoted guides | Deeper rewrites need subsystem owners; status blocks carry the remaining drift |
| 2026-08-07 | Archive rather than delete the four superseded documents | Preserve provenance and history via `git mv` plus ARCHIVE-INDEX rows |

## Verification

Run `find docs -maxdepth 1 -type f` (expect only `README.md`), then
`find 'docs/09 - Archived' -type f | sort` and compare against
`ARCHIVE-INDEX.md`, and grep for `docs/NAL_ARCHITECTURE|docs/METHODOLOGY|docs/logging_guide|docs/New Componentwise`
outside `docs/09 - Archived`, `docs/old docs`, and this lane (expect no hits).
