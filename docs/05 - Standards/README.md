# Standards Lane

**Status:** DRAFT  
**Date:** 2026-08-03  
**Applies to:** future normative contracts

No project-authored standard is frozen here yet. `core/interfaces.py` contains executable contracts, but documenting them as a stable standard would overstate current conformance while legacy and component stacks coexist.

New standards must define scope, dependencies, MUST/MUST NOT clauses, conformance evidence, and versioning. Do not turn an aspirational design into a frozen contract.

## Verification

Run `rg -n "class I[A-Z]|ComponentContract" src/structure_net/core src/structure_net/components` and decide whether implementation coverage is sufficient before drafting a v0 standard.
