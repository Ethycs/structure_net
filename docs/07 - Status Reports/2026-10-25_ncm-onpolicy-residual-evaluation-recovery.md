# Initialize the qualified rubric before validating sibling judgments

Execution amendment to `ncm-onpolicy-residual-v1`. The completed fixed-target
arm scored 0/128 valid Terra DEV summaries. Its four-case pilot, all outputs,
judgments and producing sources remain unchanged. Both training arms and their
independent state audits, plus all 2,048 residual-label checks, are complete.

The residual-target inference pilot exited 1 during configuration, before model
loading, inference, stage-directory creation or judge requests. The sibling
cache guard constructed packet keys before installing the qualified 128-token
rubric. An offline replay found all 128 new control keys rejected under the
default rubric and all 128 accepted after the existing profile installation.
The 3,923 inherited firsts and all 128 control firsts are unchanged. The failure
and actual process receipts are retained in
`20261025_onpolicy_residual_DEV_launch_failure_v1`.

Use a separate recovery wrapper that installs the existing qualified Sol/Terra
profile before calling the unchanged configuration function. Retain every
cache guard; do not change keys, packets, rubric text, judge identities,
attempt limits, scores or prior files. Regression tests must reproduce the
initialization-order mismatch and continue rejecting unrelated new judgments.

The fixed-target DEV roots stay at v1. Use fresh residual-target DEV roots
ending in `_v2`, with this amendment, the failure diagnosis, recovery source
and tests bound into each producer. Run an offline sibling-cache preflight,
then the unchanged repeated two-case inference pilot, complete 128 outputs,
four-case judge-validity pilot and full semantic grade. Inherit all 4,051
canonical firsts. No training, checkpoint, prediction algorithm, source split,
precision, semantic threshold or final-test rule changes.

Use a separately bound finalizer and registrar for
`20261025_onpolicy_residual_results_v2`. They reuse the existing result replay,
same-case gates and conservative registry logic while including the recovery
files and failure evidence in the aggregate. The measured report remains
`docs/08 - Analysis/2026-10-25_ncm-onpolicy-residual.md`, which does not yet exist.
No failed judgment is rerun. The sealed final320 stays closed unless a candidate
passes the original DEV qualification gate.
