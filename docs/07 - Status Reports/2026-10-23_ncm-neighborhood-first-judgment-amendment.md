# Reuse earlier identical 128-token baseline judgments for the student

Date: 2026-10-23. Prospective evaluation-only correctness amendment. The fixed
2,048-update student is still running; no student development outputs or scores
exist. Known baseline results remain 52/64 for unchanged 3B and 47/64 for 1B
under the qualified 128-token profile. Their first judgments are preserved.

The original student grading entry point starts an empty judgment cache. If a
student output exactly matches an already graded baseline packet, requesting a
fresh judgment would violate the first-judgment rule used for the comparison.
Before student evaluation, add an explicit baseline-cache import. Verify the
registered baseline aggregate hash, every artifact hash, packet/rubric/role/model
cache keys and every first raw response. Import all prior entries, including
negatives; never select which prior decisions to retain. Only exact packet keys
match, so old 32-word-profile judgments cannot become new-profile judgments.

Preserve the frozen final checkpoint, all 64 original source-disjoint development
cases, nine parallel passes, no output repair, and the qualified Terra then
Sonnet gates of 64 valid and at least 61 whole-summary passes each. Verify the
generated-output hash and exact source order before judging. The original v1
grading entry point remains unexecuted/deferred; use a new
`20261023_native_query_neighborhood_grade_v2` root. No model-producing source,
training data, weights, generation algorithm or success threshold changes.

The new grader records prior-cache size and exact reuse counts. Short-source
development success still requires appropriate final/source-range evaluation;
neither teacher labels nor cached baseline results establish student 95% or
speed. Keep both sealed cohorts closed.
