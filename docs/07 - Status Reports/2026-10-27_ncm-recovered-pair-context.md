# Recover conditioning metadata without changing pair admission

Inventory original role/date and summarizer identities for the fixed 104,503
provenance-guarded candidate pairs. The normalized export omitted this context.
Use exact pair IDs, source hashes and target hashes to join the completed
supplement/deep-join label tables and original atomic-cache records. For a raw
label, require one pointer whose span hash is the complete candidate source
hash. For an atomic cache, require its stored text hash and summary text to
match the candidate exactly. Preserve all matching provenances and conflicting
contexts; do not pick a convenient role or date.

Separate counts for exact-substring versus non-substring targets, available
roles/dates/identity fingerprints, and ambiguous roles or dates. Metadata
fingerprints are not automatically classified as semantic/extractive protocols.
Record request hashes when present so a later original-request lookup can
establish that distinction. Role/date recovery is partial context recovery;
it does not reconstruct the full original prompt or preceding hierarchy.

This is a CPU-only read-only inventory of existing evidence. Store the result
and provenance in `20261027_recovered_pair_context_v1`. No model calls, new
judgments, candidate admission, revised historical scores, or final-case access.
All 104,503 inputs remain `TRAIN_review_candidate`, not high-quality gold.
