# Standards Lane

**Status:** DRAFT  
**Date:** 2026-08-06
**Applies to:** normative and candidate contracts

No project-authored standard is frozen here yet. `core/interfaces.py` contains executable contracts, but documenting the complete component surface as stable would overstate current conformance while legacy and component stacks coexist.

Current candidate:

- [`NAL-STD-EXPERIMENT-v0.md`](NAL-STD-EXPERIMENT-v0.md) — draft evidence, execution, artifact, preregistration, preservation, and report contract for NAL research. The operational companion is [`experiment-and-report-authoring-guide.md`](../02%20-%20Implementation/experiment-and-report-authoring-guide.md).

New standards must define scope, dependencies, MUST/MUST NOT clauses, conformance evidence, and versioning. Do not turn an aspirational design into a frozen contract.

## Verification

Run `rg -n "class I[A-Z]|ComponentContract" src/structure_net/core src/structure_net/components` for component coverage, and audit a completed NAL campaign against `NAL-STD-EXPERIMENT-v0.md` before freezing that candidate.
