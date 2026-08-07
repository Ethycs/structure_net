# Ambient Documentation Inventory

**Status:** CURRENT  
**Date:** 2026-08-07  
**Applies to:** Markdown outside the numbered documentation tree, and the 2026-08-07 relocation of the former `docs/`-root files  
**Depends on:** `../03 - Architecture/structure-net-overview.md`; `../000 - Doc Maintenance/2026-08-07_docs-root-relocation.md`

This inventory classifies pre-existing prose without silently upgrading its claims. “Legacy” means unreconciled, not necessarily wrong. On 2026-08-07 every loose `docs/`-root file except `docs/README.md` was audited against the codebase and filed into its lane; each moved file carries a status block recording its verified trust.

| Document(s) | Location now | Current trust | Disposition |
| --- | --- | --- | --- |
| `README.md` (repo root) | unmoved | Legacy; quick start and NAL examples fail against the current API | Keep as entry point; reconcile examples and imports (see roadmap) |
| `00 - Theory/METHODOLOGY.md` | filed 2026-08-07 | Legacy; verified against the deprecated `evolution/` stack, one factual error corrected in its status block | Split foundations from proposed techniques (still pending) |
| `01 - Design/LEARNING_RATE_STRATEGIES.md` | filed 2026-08-07 | Legacy design catalogue; 10/12 strategies exist only in the deprecated package | Do not treat as the current scheduler API |
| `02 - Implementation/CONFIG_MIGRATION_GUIDE.md` | filed 2026-08-07 | Current; API verified | Track `UnifiedConfig` adoption |
| `02 - Implementation/{logging_guide, standardized_logging_system, storage_formats_guide, profiling_data_guide, data_system_integration_guide}.md` | filed 2026-08-07 | Mixed; import paths corrected, remaining drift itemized in each status block | Reconcile the flagged sections by subsystem |
| `09 - Archived/New Componentwise refactoring.md` | archived 2026-08-07 | Superseded design source | Authority: component design, migration guide, refactor roadmap |
| `09 - Archived/kernel_implementation_guide.md` | archived 2026-08-07 | Aspirational; microkernel never implemented | Interface layer survives in `core/interfaces.py` |
| `09 - Archived/NAL_ARCHITECTURE.md` | archived 2026-08-07 | Superseded; links migrated | Authority: `03 - Architecture/structure-net-overview.md` |
| `09 - Archived/CONTRIBUTING.md` | archived 2026-08-07 | Stale; routed contributors into deprecated shims | Authority: the two `02 - Implementation` guides |
| `integration_plan.md`, `to_integrate.md` (repo root) | unmoved | Reconciled and substantially implemented | Source provenance; current authority is `docs/06 - Roadmaps/component-refactor.md` |
| `experiments/MEMORY_OPTIMIZATION_SUMMARY.md` | unmoved | Point-in-time, target missing | Archive with provenance after confirming history |
| `docs/old docs/*` | unmoved | Historical but mutable location | Move only with link/history audit |
| `src/**/README.md`, `tests/**/README*.md` | unmoved | Closest to code, still unreconciled | Keep beside code; link from promoted guides |
| `archive/**/*.md` | unmoved | Historical | Do not treat as active requirements |

## Verification

Run `find . -type f -name '*.md' -not -path './.git/*' | sort` and add any unclassified living document to this table before relying on it.
