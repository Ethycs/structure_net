# Produce vetted training labels for the wider public sources

Date: 2026-10-23. Exploratory accepted-label data production only. The running
short-source student remains fixed and has no inspected final summary score.
The new pool contains 112 training sources in four 512–2048 cl100k token bands.
Diagnostic and both sealed cohorts are excluded from this label-production job.

Use the qualified 128-token profile and the existing fixed Sonnet draft → Terra
critique → one Sonnet revision for every candidate → Terra final-audit algorithm.
Use source-line evidence IDs, exact public source verification, four workers,
two passages per generation request, temperature zero and max 2048 response
tokens. This teacher data-production allowance is not an application-runtime
generation benchmark. Preserve first schema-valid teacher candidates, including
bad or over-cap ones; maximum three transport/schema attempts, no semantic
rerolls. Revisions receive exact stored/native token counts for every draft.
Prefer 40–60 words, preserve material qualifiers and omit unsupported details.

Pilot on the first fixed training source from each length band. Require four
valid final audits, four whole-summary passes and all four complete labels to
fit both 128-cl100k and 128-native-including-EOS limits, with exact native round
trip, nonempty evidence-linked claims and no embedded native special tokens.
Preserve a failed pilot and stop main production if this gate fails.

Main production covers all 112 fixed training sources. Reuse the pilot's exact
drafts, revisions and first judgments, including negatives. Never regenerate
those four texts. Each remaining source gets one draft and one revision; never
choose the better of them. Grade all final candidates, retaining failures in
the teacher denominator. Cache identity includes exact packet/rubric/role/model;
if revision text is unchanged, its earlier first judgment wins. Keep generation
and judging stages sequential; no API stage overlap.

Export as training labels only individually final-audit-passed candidates that
also meet the complete native/storage/evidence format checks. This is a declared
accepted-label study, not an assertion that the teacher population meets 95%.
Report population counts, invalids, cap failures and accepted counts per band.
There is no all-case capacity rescue or student-training admission. Any new
student run still needs its own protocol and lifecycle checks; model evaluation
retains every source in its fixed held-out denominator.

Roots `20261023_wide_labels_{pilot,main}_{draft,critique,revision,audit}_v1`.
Freeze plans, source/train manifest, original helper sources, tokenizer and both
judge qualifications. Preserve raw requests/responses, first judgments and true
terminal receipts. Terra critiques and audits, so this is not independent human
gold; the student still requires its own source-disjoint, independent-judge task
evaluation. No model success, compression, speedup or final-test claim.
