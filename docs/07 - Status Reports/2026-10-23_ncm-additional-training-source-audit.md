# Audit additional public training sources for whole-summary supervision

Date: 2026-10-23. Preparation only; no model-quality claim.
The word-final model remains at 0/128 complete-summary passes. The isolated
first-concept learner passed a check on 16 training sources after 1,024 updates,
but the remaining hierarchy and source-disjoint quality remain unqualified.

Before choosing new training-data quotas, count available public sources from
the previously audited original training stores. Restrict candidate groups to
those already admitted to the unpaired codec corpus. Its 1,024 groups excluded
the existing summary labels and held-out sources. Recheck all identities against
current 252 training cases, 64 short development cases, 64 longer diagnostic
cases, and the longer sealed cohort's metadata. Use only original training
stores whose manifest proves no source-session or eligible-input overlap with
the original test partition. Do not open any final source or output file.

Audit both original training tokenization stores, retaining complete source
texts that round-trip to the exact public LongMemEval spans, have a matching
source hash, and contain 192–2,048 cl100k tokens. Ignore every old target field:
old atom targets are not whole-summary labels. Deduplicate sources by hash,
record their original store/row, and report counts by source-length band and
source group. Export metadata only, not labels or candidate summaries.

This CPU preparation does not admit a training run or select targets based on
model output or judge feedback. Freeze the complete eligible index before a
later quota/teacher/quality protocol is specified. Use a fresh root
`20261023_additional_summary_source_availability_v1`. Bind this plan, audit code,
public dataset and prior source-split proofs. Preserve all failures. Both final
cohorts remain closed; no accuracy, compression or speed claim.
