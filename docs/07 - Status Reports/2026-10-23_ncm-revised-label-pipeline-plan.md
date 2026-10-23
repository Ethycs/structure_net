# Fixed draft, critique and one-revision teacher pipeline

Date: 2026-10-23. EXPLORATORY `ncm-revised-label-pipeline-v1`.
Known evidence: native teacher v2 failed its quote-format pilot; unchanged
summaries scored8/10 under the old rubric, including one real recipe-order
contradiction and one32-word-budget salience failure. Preserve both first
negative judgments and the failed pilot. The128-token judge-profile amendment
must pass all50 calibration controls for both aliases before this pipeline runs.

This is a changed data-production algorithm, not a retry of an unchanged
benchmark model: Sonnet draft -> independent Terra claim critique -> exactly
one Sonnet revision for EVERY draft, including passing drafts -> independent
final Terra audit. Preserve each draft, critique and revision. Never select the
better-looking draft/revision, regenerate final negatives, loop until a pass,
exclude over-cap outputs or change the audit threshold. Final audit cannot
change the final candidate. Critique judgments are training-data production
inputs, not claimed unbiased final system quality.

The first10 sources/candidates are the existing v2 systems pilot, reused exactly
as the draft stage (no requests). They are development data for the pipeline.
The following128 already-fixed source-disjoint public training cases remain the
prospective audit sample; no labels have been generated for them. Never touch
the64model-development or256sealed-final sources. Keep the source-byte checks.

Use source-line IDs in new teacher evidence fields; resolve exact source quotes
in code. This removes a copying-format requirement without weakening the final
semantic rubric. Both draft and revision request40–70 words within128 stored
tokens, preserve conditions/roles/order/negation, and output one summary plus
atomic claims with evidence IDs. No literal ellipsis repair of old candidates.
Maximum three retries only for transport/schema failure, four workers, batch2,
max2048tokens, temperature0,180-second timeout. Keep empty/overlong/semantically
wrong first schema-valid candidates. All pipeline phases run sequentially; no
generation/judging API overlap.

Pilot: grade all10 unchanged drafts under the new qualified128-token profile,
revise all10 once, then grade all10 finals. Require all10finals have valid teacher
schema/evidence IDs, nonempty text and both128-cl/native-token capacity, all10
valid audit judgments and10/10whole passes before main128 production. This is a
small development gate, not a95%population estimate. Main uses the same fixed
algorithm, with no per-example admission filtering. Require128/128valid final
audits and at least122/128whole-summary passes before scaling label production
or training on this new pipeline. Failure stops this configuration.

Exact source/summary/profile/role packets retain their FIRST valid critique or
audit judgment across stages, including negatives and exhausted invalids. If a
revision equals its draft, reuse the critique judgment; do not ask again hoping
for a pass. New text gets one first independent audit. Use the existing audited
first-judgment cache implementation and fresh roots for this profile/pipeline.

Roots `20261023_revised_labels_{pilot,main}_{draft,critique,revision,audit}_v1`.
Archive all requests/responses, per-stage fingerprints, native tokenizer identity,
component/full pass numerators, cache reuse and provider errors. No student
updates, NCM success, final-test result or speed claim follows from this study.
