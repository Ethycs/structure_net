# Archive Index

**Status:** CURRENT  
**Date:** 2026-08-07  
**Applies to:** superseded documents stored in this numbered lane

The existing `archive/` tree and `docs/old docs/` remain historical sources but are not silently relocated because doing so may break links and erase useful path provenance. Files below were archived after a codebase audit; each carries a status block recording why it was superseded, added at archive time. Inbound links were migrated before the moves (see `../000 - Doc Maintenance/2026-08-07_docs-root-relocation.md`).

| Archived document | Superseded by | Date archived |
| --- | --- | --- |
| `NAL_ARCHITECTURE.md` | `../03 - Architecture/structure-net-overview.md`; `../03 - Architecture/nal-local-gpu-scheduler.md` | 2026-08-07 |
| `CONTRIBUTING.md` | `../02 - Implementation/component-migration-guide.md`; `../02 - Implementation/experiment-and-report-authoring-guide.md` | 2026-08-07 |
| `kernel_implementation_guide.md` | `../01 - Design/component-architecture.md` (interface layer only; microkernel never built) | 2026-08-07 |
| `New Componentwise refactoring.md` | `../01 - Design/component-architecture.md`; `../02 - Implementation/component-migration-guide.md`; `../06 - Roadmaps/component-refactor.md` | 2026-08-07 |

## Verification

Run `find 'docs/09 - Archived' -type f | sort`; every entry besides this index must have a row above and must never be edited in place.
