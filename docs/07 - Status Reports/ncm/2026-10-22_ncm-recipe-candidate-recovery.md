# Recover valid candidate decisions from an invalid comparison packet

Date: 2026-10-22, after the test's original 64 packets. Preserve the original
`updated_test_accuracy_invalid` aggregate and report: 63 packets valid, both NCMs
0/63, full parent61/63. The remaining packet has three replies where both NCMs
are consistently judged incoherent failures, but the parent's `faithful:true`
contradicts an unsupported/contradicted claim. The unchanged parser correctly
rejects that parent evaluation. No verdict should be silently changed.

Create `ncm-recipe-update-test-candidate-recovery-v1` under
`20261022_test_recovery_v1`. Reparse each candidate independently using the same
parser, original source lines and candidate text/ID, taking its first valid raw
evaluation in chronological attempt order. Do not select by pass/fail. All NCM
decisions must come from existing raw replies. Retain the original comparison's
invalid classification; this candidate-level diagnostic cannot rescue it or the
failed training control gate.

For a parent candidate with no valid existing evaluation, request only that
candidate in a new packet, same source, rubric, judge and parser, at most three
transport/schema attempts. Never retry a valid failure. No new NCM judgment,
generation, optimization or source selection. If the parent remains invalid,
report its missing judgment separately and still expose fully validated NCM
candidate counts. This changes validation granularity after outcomes were known
and must remain explicitly exploratory. Preserve raw replies, lineage, exact
counts, independent per-model completeness, and the conservative registry result.
