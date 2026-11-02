# Role-aware source prompts for recovered conversational supervision

CPU-only continuation of the recovered paraphrase catalog. No optimizer,
encoder, codec, or judge runs here. Keep the original frozen Llama body,
GPT2 source-token geometry, exact character-overlap alignment, full source text,
and fixed chat-template date23 Oct2026 for subsequent feature extraction.

The new input adds only the original role and original timestamp before the
passage, as a JSON metadata object, and tells the existing summarization system
prompt that these supplied fields are evidence for attribution. Missing dates
remain null. The original timestamp is never substituted for the chat-template
date. No target text, codes, summarizer identity, request ID, or target provenance
may enter the prompt. This is a new conditioning variant; it must not replace
the cached inputs of the running matched tree/dense comparison.

Use an exact source-field allowlist, hash-bound indexed disk reads, and full
tokenization without truncation. Keep source roles/timestamps distinct even for
identical raw source text. First run a pilot covering the longest and shortest
sources, every observed role, present/missing timestamps, and one repeated source
with different timestamps. Require unchanged original TRAIN geometry controls,
canonical template tokenization, identical raw source-token IDs, finite normalized
alignment weights, and exact repeated pilot records in the full pass.

Fresh roots `20261028_recovered_paraphrase_metadata_{pilot,full}_v1`; locked Pixi,
CPU-only,4GB host-RSS ceiling. Freeze source/config/input identities. Verify every
JSONL offset and report total positions, maximum context length, and projected
FP32 feature bytes. Preserve failures. Source features and native target codes
are subsequent GPU stages; this preparation does not admit training or establish
semantic quality. Keep the existing development and final reservations intact.
