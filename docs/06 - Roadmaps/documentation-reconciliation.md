# Documentation Reconciliation Roadmap

**Status:** Living Document — 2026-08-03  
**Applies to:** documentation and its evidence links

## Current status

| Area | Verdict | Evidence |
| --- | --- | --- |
| Numbered tree and governance | ✅ Established | `docs/README.md` |
| As-built top-level architecture | ✅ Initial reconciliation | `03 - Architecture/structure-net-overview.md` |
| Ambient-doc classification | ✅ Initial inventory | `04 - Reference/document-inventory.md` |
| Executable verification baseline | 🟡 Collection characterized | archive discovery defect plus 5 active-suite collection errors |
| Subsystem guides | 🔲 Unreconciled | flat `docs/*.md` and code-adjacent READMEs |
| Component-refactor instructions | ✅ Reconciled | design, migration guide, and current roadmap in numbered lanes |
| Normative component standard | 🔲 Not adopted | incomplete cross-stack conformance |

## Tier 0 — Restore trustworthy evidence

| Gap | Depends on | Blocks | Status |
| --- | --- | --- | --- |
| Restrict pytest discovery to active tests | project decision on archive collection | reliable baseline | 🔲 Open |
| Resolve four active-suite collection causes | API and inheritance decisions | test execution baseline | 🔲 Open |
| Repair or narrow `structure_net.__all__` | public API decision | accurate reference docs | 🔲 Open |
| Decide canonical orchestration imports | inventory of active examples | deprecation guide | 🔲 Open |
| Align package dependencies with runtime imports | supported installation policy | reproducible setup | 🔲 Open |

## Tier 1 — Reconcile active surfaces

1. Reconcile the root README quick start against importable public APIs; this unblocks a reliable user reference.
2. Reconcile configuration, data, logging, and profiling guides against their current modules; this unblocks subsystem implementation docs.
3. After the five active test-module collection errors are resolved, run the suite and separate test failures from collection and environment failures.
4. Trace imports from examples and experiments to quantify remaining legacy-evolution usage before removal planning.

## Tier 2 — Curate research knowledge

1. Split `METHODOLOGY.md` into validated theory, design hypotheses, and measured analysis.
2. Reconcile the remaining kernel proposal separately; the component refactor is now distilled into `component-architecture.md`, `component-migration-guide.md`, and `component-refactor.md`.
3. Promote stable executable component contracts into a versioned standard only after conformance tests exist.

## Verification

Run the scoped collection command in `../02 - Implementation/developer-guide.md`, then choose whether to repair discovery first or document the active suite's baseline first.
