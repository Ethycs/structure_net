# Audit actual input coverage before claiming application-wide quality

Date: 2026-10-23. Read-only CPU data audit during the unchanged 2,048-update run.
No model generation, training, quality grading, source selection or final-test
access. Preserve the original 256 sealed sources and every existing artifact.

Count actual native context/target lengths for the current language and summary
training cohorts. Recover each summary source from its already fixed public
label record and verify ID, group, session, hash and native prompt IDs. Count
source and target cl100k tokens separately from native model tokens. Count the
same fields for the 64 original development sources and source lengths for the
8,192 approved public training candidates. Do not treat a full raw source as
language-model context: that arm sees only its stored 128-token native window.

Record application limits: skip summarization below 192 cl100k source tokens;
fragment ceiling 2,048, raw prompt ceiling 6,300, at most eight fragments, stored
summary ceiling 128. The 32-word extractor target is a separate task. This audit
establishes exposure/coverage only. It cannot pass the 95% student endpoint or
authorize a speed claim. A short-source development result still requires an
appropriately source-disjoint final evaluation with representative input range.

Use a fresh `20261023_application_input_range_audit_v1` root with source hashes,
strict JSON and a short report. No modifications to the running producer.
