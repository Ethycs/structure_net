# Ambient Documentation Inventory

**Status:** CURRENT  
**Date:** 2026-08-03  
**Applies to:** Markdown outside the numbered documentation tree  
**Depends on:** `../03 - Architecture/structure-net-overview.md`

This inventory classifies pre-existing prose without silently upgrading its claims. “Legacy” means unreconciled, not necessarily wrong.

| Document(s) | Best lane | Current trust | Disposition |
| --- | --- | --- | --- |
| `README.md` | 04 Reference | Legacy | Keep as entry point; reconcile examples and imports |
| `docs/METHODOLOGY.md`, `docs/LEARNING_RATE_STRATEGIES.md` | 00 Theory / 01 Design | Legacy | Split foundations from proposed techniques |
| `docs/New Componentwise refactoring.md` | 01 Design / 02 Implementation / 06 Roadmaps | Reconciled into numbered docs; retains aspirational detail | Source provenance; current authority is the component design, migration guide, and refactor roadmap |
| `integration_plan.md`, `to_integrate.md` | 06 Roadmaps | Reconciled and substantially implemented | Source provenance; current authority is `docs/06 - Roadmaps/component-refactor.md` |
| `docs/kernel_implementation_guide.md` | 01 Design / 06 Roadmaps | Legacy, broader aspirational architecture | Reconcile separately; it is not the active component-refactor brief |
| `docs/NAL_ARCHITECTURE.md` | 03 Architecture | Superseded as trusted overview | Retain until links migrate, then archive |
| config, data, logging, profiling, storage guides | 02 Implementation | Mixed | Reconcile commands and paths, then promote by subsystem |
| `experiments/MEMORY_OPTIMIZATION_SUMMARY.md` | 07 Status / 08 Analysis | Point-in-time, target missing | Archive with provenance after confirming history |
| `docs/old docs/*` | 09 Archived | Historical but mutable location | Move only with link/history audit |
| `src/**/README.md`, `tests/**/README*.md` | 02 Implementation | Closest to code, still unreconciled | Keep beside code; link from promoted guides |
| `archive/**/*.md` | 09 Archived | Historical | Do not treat as active requirements |

## Verification

Run `find . -type f -name '*.md' -not -path './.git/*' | sort` and add any unclassified living document to this table before relying on it.
