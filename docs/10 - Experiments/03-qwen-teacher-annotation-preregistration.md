# Experiment 03 — Qwen Teacher Annotation Preregistration

**Status:** AMENDED BY USER BEFORE DISTILLATION OUTCOMES  
**Date:** 2026-08-16

## Scope

The original contract requested every eligible PAWS training pair-group. On
2026-08-16, after 17,920 responses but before either local distillation outcome,
the user directed the experiment to use the dataset's existing answers. Human
PAWS labels are therefore the full-data distillation target. Qwen is retained
as a balanced 2,048-group teacher audit selected within label by
`SHA256("23:" + group_id)`. The superseded full-run attempt remains raw evidence
and cannot silently enter the auxiliary loss.

The Experiment 01 prompt, `temperature=0`, `max_tokens=16`, exact label parser,
model name, request fingerprint, latency, retry count, raw response, teacher
hard label, human label, and teacher correctness are retained. Rationales are
disabled and represented as `null`; they must never be inferred from hidden
reasoning fields. The API key is neither serialized nor hashed into artifacts.

## Completion gates

The canonical NAL run passes only if:

1. every selected immutable audit group occurs exactly once (1,024 per label);
2. no quarantined, development, or test example occurs;
3. every request has a terminal success/failure status and fingerprint;
4. malformed responses remain raw and are not repaired;
5. output dataset, prompt, model, selection, and producing-code hashes exist;
6. failures are zero after bounded retries.

Human PAWS labels remain the oracle and cover all 49,259 eligible unique train
groups. Qwen disagreement is teacher behavior, not a relabeling instruction.
Experiments 04–05 may use teacher labels only for selected audit rows in their
declared auxiliary fidelity arms.
