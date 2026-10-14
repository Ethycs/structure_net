# Llama 3B throughput pilot corrections

Date: 2026-10-13. Parent hypothesis `llama32-3b-summary-throughput-v1`.
Both Q4 lifecycle pilots completed before this amendment. No development or
held-out quality outcomes have been inspected. Pilot artifacts are retained.

The first FP16 conversion failed before writing weights because the pinned
converter imports SentencePiece even for a Llama 3 BPE tokenizer. Install
`sentencepiece==0.2.0` in a new experiment-local dependency directory, record its
native-library hash, and retry conversion in a new root. Existing environments
and the failed conversion evidence remain unchanged.

The native runtime emitted debug logs inside timed inference. Suppress messages
below warning level identically for all arms, rebuild the new study binary, and
run a new lifecycle check before primary measurements. Completed pilots use the
previous frozen producer and must not be pooled with measurements.

Drafts 8 and 16 generated different formatting from scalar decoding on the tiny
Q4 control; draft 4 matched. Every emitted candidate was still selected by the
target model's own logits, and rejected KV tails were removed. Evaluate new
outputs for task quality, as authorized, without claiming bitwise identity.
