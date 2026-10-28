# Recovered conversational summaries: source and target catalog

This exploratory data preparation follows the completed 104,503-pair guarded
pool and original-context inventory. It does not change the running source-tree
comparison, its codec, prompts, training schedule, or evaluation. The goal is a
larger conversational training input for a subsequent, separately frozen run.

Known before selection: the fixed label review passed 54/128 jointly; its
non-substring subset passed 46/57. Seven rejected paraphrases lacked assistant
attribution in the review packet, and original records establish assistant roles
for all seven. Historical judgments remain unchanged. The improved codec
reconstructs 117/128 held-out strings exactly; this is not summarization accuracy.
The available non-substring pool with roles has 44,777 pairs, 39,257 distinct
source texts, and 8,169 components. These labels are noisy historical supervision,
not factual gold. No new quality claim follows from filtering.

Keep every eligible non-substring pair with one unambiguous original user or
assistant role and at most one observed original timestamp. Missing timestamps
remain null. Retain all target variants. Exclude the entire 128-component saved
label/codec diagnostic sample and exact native-frame duplicates of its targets.
Inherit all original source-component and open-text guards, including the sealed
final metadata reservation; do not open final source text. Recheck complete
source/target hashes, 192–2,048 input tokens, at most128 application output tokens,
and 1–126 native lexical tokens without special/unknown IDs. Never truncate.

A repeated source text can have different original timestamps: the inventory
contains 2,324 such source hashes before exclusions. Preserve each pair's exact
role/date context. Source feature identity depends only on source text, component,
role, and original timestamp; no target, summarizer identity, or request hash may
enter it. Keep original provenance separately. A timestamp is source metadata,
not the current chat-template date. Prompt tokenization and source alignment
remain a subsequent stage and must explicitly include this metadata.

Write indexed source-only JSONL, separate target JSONL with complete native
frames, and source-content groups listing their context and target variants.
Future sampling must give each distinct source text one presentation per epoch
and rotate through its target variants deterministically; raw variant count must
not multiply source exposure. This stage records that requirement without
claiming an optimizer schedule has been run. Report pair, source-text, conditioned
source, and component counts separately, together with role/date coverage.

CPU-only locked Pixi,4GB peak host-RSS cap, fresh root
`20261028_recovered_paraphrase_catalog_v1`. Hash and preserve producer/inputs,
account for all104,503 candidates and every exclusion, verify every emitted byte
offset and join, and stop on mismatches. Focused tests cover context conflicts,
target-independent source identity, full-frame boundaries, and whole-component
exclusion. Preparation admits no training, makes no model/judge calls, and changes
no existing evaluation split. Prompt/features, singleton target-code extraction,
and a fixed noisy-supervision training plan are still required before training.
