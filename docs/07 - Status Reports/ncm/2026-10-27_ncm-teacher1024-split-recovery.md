# Correct the reserved TRAIN split-name check before teacher requests

The first preparation command failed before creating its stage root or sending
any request: tool session 28932, terminal chunk `2a2ccf`, exit 1. Its producing
source and actual failure receipt are preserved in
`20261027_teacher1024_prepare_v1_failure`. No candidate or target was generated.

Diagnosis found that all 1,024 selected source records use the original
`additional_train` split name, written by `ncm_additional_training_sources.py`.
The new validator had incorrectly required the literal `train`. A read-only
check found no selected source text-hash, truncation, or token-count mismatch.
The failure does not establish that the remaining source guards passed.

Use `ncm_teacher1024_recovery` with fresh `_v2` stage roots and its matching
`ncm_teacher1024_recovery_service` launcher. Require exactly `additional_train`
without modifying any record. All other source guards, original IDs, quotas,
ordering, teacher models, prompts, retry budgets, feature checks, eight-case
pilot, stop rules and raw audit requirements remain unchanged. Preserve the
original producing files, plan and failed attempt; bind them into new proofs.
Run the new split-name regression tests and existing scheduler tests first.

The new teacher collection still has no student accuracy result. Existing
application DEV remains 0/128, and the final320 remains sealed.
