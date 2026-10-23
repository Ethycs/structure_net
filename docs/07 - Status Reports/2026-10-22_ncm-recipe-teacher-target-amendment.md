# Replace failed training targets with fixed local-teacher summaries

Date: 2026-10-22, after the original-domain control passed and before updated
training. The saved-target audit accepted only 5/128 candidates, with one invalid
judgment. Preserve `20261022_quality_v1`; do not train the planned quality run on
those labels or upgrade that gate to passed.

New exploratory data stage: on the first 64 already-fixed public training
passages, generate one greedy summary each from the original frozen local
Llama-3.2-3B-Instruct. Request approximately 24 words to fit the unchanged
48-GPT2-token codec; no prompt tuning, retries for quality, sampled selection or
source selection by the generated outcomes. BF16, batch four, max96 new Llama
tokens. Reject labels that cannot fit the native canvas without truncation.
Source body features retain their original approximately-32-word instruction;
both compared NCM arms share those same frozen features and new target labels.

Verify all outgoing source passages against the published LongMemEval SHA and
exact source spans. Grader payloads contain only those public passages and newly
generated teacher summaries. Reuse the qualified unchanged rubric/parser. Keep
every valid failure and invalid attempt. Select the first 32 valid passing,
native-compatible summaries in fixed source order. Invalid candidates are
reported and not eligible for training; this is label selection, not a model
accuracy estimate. If fewer than 32 pass, stop quality training again.

Artifacts: `20261022_teacher_v1` and `20261022_quality_v2` under the recipe root.
Only after this new gate passes may the unchanged matched 32-case training
design and its CUDA pilot run. That tests the recipe on regenerated targets,
not on the original poor snippets, and cannot retroactively rescue any prior
test result. The updated source producer must include this amendment and the
new data-stage code. The old target audit and raw replies remain unchanged.
