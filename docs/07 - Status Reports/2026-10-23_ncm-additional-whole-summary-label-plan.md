# Produce additional complete-summary training labels

Date: 2026-10-23. Training-data preparation, not model accuracy.
The first concept level passed its 16-case training-capacity check, while the
latest full summarizer remains at 0/128 held-out development passes. A broader
data pool and full-hierarchy training are still required.

Use the completed broader availability index, retaining all original training
partition, public-source, hash and held-out exclusions. Select 512 new sources:
212 in 192–511 cl100k tokens, 212 in 512–1,023, 24 in 1,024–1,535 and 64 in
1,536–2,048. Process bands in scarcity order 2,3,1,0; sort each by
SHA256(`ncm-additional-whole-label-v1:`+original case ID). Keep one source per
case, session, connected group and source hash across the complete selection.
The two rare bands have 33 and 90 available groups; these quotas leave room
for group overlap. Fail rather than silently substitute if a quota cannot fit.

Export only complete source text and provenance, ignoring every old target.
Verify the selected sources against the exact public LongMemEval release.
Freeze source identities and exact initial teacher request payloads before
external requests. Use the first selected source in each band as a four-case
pilot; retain its drafts, revisions and first judgments in the main run.

Reuse the existing whole-summary label workflow: Sonnet drafts two independent
passages per request, qualified Terra critiques each summary, Sonnet revises
every draft exactly once, and Terra audits the final text. Prefer 40–60 words,
at most 128 stored cl100k tokens, correct attribution and material conditions.
Teacher JSON includes atomic claims and exact source-line IDs. Temperature0,
four workers, generation cap2,048, judge cap4,096; at most three transport/schema
attempts, with the first schema-valid teacher response and first valid judgment
retained. No repeated semantic revisions or best-of selection.

Use the existing four-of-four final pilot gate before the 512-case main run.
Every final accepted label must pass faithfulness, salience, readability,
nonempty text and the128-token cap. Require GPT-2 round trip and at most192GPT-2
positions including one EOS. Native Llama round-trip/control checks remain;
the old128-native-token readout ceiling is removed because the new output
canvas is192GPT-2 positions. Record exact capacity information in critiques.
Preserve all failures and report the complete512-source denominator. This is
training-label curation; accepted-label quality is not model task accuracy or
a claim of95%teacher population quality. Student training needs a later protocol.

Begin every critique/audit cache with all930canonical first task judgments and
merge retained pilot/previous-stage entries without overwriting any decision,
including negatives. Use the same anonymous `summary` candidate identity.
Identical revised text reuses its earlier decision. Both final cohorts remain
closed. Send only verified public sources and their locally derived model
outputs to the already authorized central-dev inference gateway.

Use fresh source root `20261023_additional_whole_labels_sources_v1` and label
roots `20261023_additional_whole_labels_{pilot,main}_{draft,critique,revision,audit}_v1`.
Every producing source/config/input is hashed; record actual terminal receipts.
The existing workflow code is retained unchanged behind a new explicit wrapper.
No compression, model accuracy or speed claim follows from this data preparation.
