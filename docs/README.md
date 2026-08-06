# Structure Net Documentation

**Status:** CURRENT  
**Date:** 2026-08-06
**Applies to:** repository documentation and the Python packages under `src/`  
**Depends on:** `Agentic Technique Master.md`

This tree separates foundations, intent, implementation, as-built architecture, contracts, plans, snapshots, and measured analysis. The code remains authoritative where a legacy document and implementation disagree.

## Start here

1. Read [`03 - Architecture/structure-net-overview.md`](03%20-%20Architecture/structure-net-overview.md) for the current system map.
2. Read [`02 - Implementation/developer-guide.md`](02%20-%20Implementation/developer-guide.md) to install, inspect, and test the repository.
3. Use [`02 - Implementation/experiment-and-report-authoring-guide.md`](02%20-%20Implementation/experiment-and-report-authoring-guide.md) to preregister, implement, shakedown, run, report, and register NAL experiments; its normative requirements are in [`05 - Standards/NAL-STD-EXPERIMENT-v0.md`](05%20-%20Standards/NAL-STD-EXPERIMENT-v0.md).
4. For the completed component migration, read [`01 - Design/component-architecture.md`](01%20-%20Design/component-architecture.md), [`07 - Status Reports/2026-08-03_component-migration-complete.md`](07%20-%20Status%20Reports/2026-08-03_component-migration-complete.md), then use [`06 - Roadmaps/component-refactor.md`](06%20-%20Roadmaps/component-refactor.md) for remaining retirement work.
5. Use [`02 - Implementation/component-migration-guide.md`](02%20-%20Implementation/component-migration-guide.md) for each migration slice.
6. Consult [`04 - Reference/document-inventory.md`](04%20-%20Reference/document-inventory.md) before trusting a pre-existing guide.
7. Use [`06 - Roadmaps/documentation-reconciliation.md`](06%20-%20Roadmaps/documentation-reconciliation.md) for broader documentation gaps.
8. Read [`08 - Analysis/2026-08-03_runner-experiment-data-modernization.md`](08%20-%20Analysis/2026-08-03_runner-experiment-data-modernization.md) for the runner, experiment-schema, and data-story assessment.
9. Check [`000 - Doc Maintenance/2026-08-03_docs-tree-gaps.md`](000%20-%20Doc%20Maintenance/2026-08-03_docs-tree-gaps.md) for known drift.
10. Read [`03 - Architecture/tinyllm-feedback-adapter.md`](03%20-%20Architecture/tinyllm-feedback-adapter.md) for the TinyLLM-compatible GPT-2 builder and experimental delayed-feedback extension.
11. Read [`08 - Analysis/2026-08-04_tinyllm-adapter-acceptance.md`](08%20-%20Analysis/2026-08-04_tinyllm-adapter-acceptance.md) for the upstream audit and claim-by-claim acceptance matrix.
12. Read [`08 - Analysis/2026-08-05_tinyllm-semantic-quotient-circle.md`](08%20-%20Analysis/2026-08-05_tinyllm-semantic-quotient-circle.md) for the measured d6/d8 semantic-topology experiment and its conservative verdict.
13. Read [`08 - Analysis/2026-08-05_tinyllm-task-quotient-contrast.md`](08%20-%20Analysis/2026-08-05_tinyllm-task-quotient-contrast.md) for the matched circle-versus-interval intervention and map-aware result.
14. Read [`08 - Analysis/2026-08-05_tinyllm-internal-quotient-probes.md`](08%20-%20Analysis/2026-08-05_tinyllm-internal-quotient-probes.md) for the frozen layerwise branch/cross-decoding experiment and independent cohomology result.
15. Read [`03 - Architecture/task-geometry-atlas.md`](03%20-%20Architecture/task-geometry-atlas.md) for the paired task-reference interpretability contract and its explicit proxy boundaries.
16. Read [`08 - Analysis/2026-08-05_tinyllm-layer-task-geometry-atlas.md`](08%20-%20Analysis/2026-08-05_tinyllm-layer-task-geometry-atlas.md) for the measured attention/MLP localization atlas.
17. Read [`03 - Architecture/degree-defect-cobordism.md`](03%20-%20Architecture/degree-defect-cobordism.md) for the numerical winding-change/defect-charge contract and its continuous-tokenizer boundary.
18. Read [`08 - Analysis/2026-08-05_tinyllm-degree-defect-cobordism.md`](08%20-%20Analysis/2026-08-05_tinyllm-degree-defect-cobordism.md) for the localized d6/d8 training events.
19. Read [`03 - Architecture/depth-graded-transformer.md`](03%20-%20Architecture/depth-graded-transformer.md) for the exact-prefix/partial-residual continuous-depth contract.
20. Read [`08 - Analysis/2026-08-05_tinyllm-depth-graded-quotient.md`](08%20-%20Analysis/2026-08-05_tinyllm-depth-graded-quotient.md) for the matched ordinary, multi-exit, and continuous-gate results.
21. Read [`08 - Analysis/2026-08-05_tinyllm-conditional-branch-depth-scan.md`](08%20-%20Analysis/2026-08-05_tinyllm-conditional-branch-depth-scan.md) for the five-seed direct residual-quotient test, failed nuisance-robust gate, and block-1 attention/MLP mechanism.
22. Read [`03 - Architecture/nal-local-gpu-scheduler.md`](03%20-%20Architecture/nal-local-gpu-scheduler.md) for logical GPU IDs, experiment slots, retries, and completed-result resume.
23. Read [`08 - Analysis/2026-08-06_nal-local-gpu-scheduler-acceptance.md`](08%20-%20Analysis/2026-08-06_nal-local-gpu-scheduler-acceptance.md) for the real three-process and dual-`d6` CUDA acceptance runs.
24. Read [`08 - Analysis/2026-08-06_tinyllm-nuisance-support-scaling.md`](08%20-%20Analysis/2026-08-06_tinyllm-nuisance-support-scaling.md) for the five-seed nuisance-coverage scaling result, failed invariant-quotient gate, and shifted block-1 mechanism.
25. Read [`08 - Analysis/2026-08-06_tinyllm-block1-quotient-control.md`](08%20-%20Analysis/2026-08-06_tinyllm-block1-quotient-control.md) for the causal horizontal/vertical block-1 intervention, its modest extrapolation-base improvement, and failed joint quotient gate.
25. Read [`08 - Analysis/2026-08-06_tinyllm-block1-quotient-control.md`](08%20-%20Analysis/2026-08-06_tinyllm-block1-quotient-control.md) for the five-seed causal block-1 intervention, failed constructive quotient gate, and equivariant-front-end recommendation.

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
