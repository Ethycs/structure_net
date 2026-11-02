# Recover source-safe training review candidates from the complete census

This is CPU-only data preparation, not a quality experiment. Preserve the full
259,636-row census export and its SHA-256. Account for every one of its 244,587
raw pairs. Keep the 15,049 hierarchy-merge pairs separate pending reconstruction
of their underlying source provenance and task-specific training prompts.

Trace source inputs to the original verified data package or the registered
public source graph. Reconstruct pointer-bound inputs from the original whole
turns and independently verify their hashes, character bounds and ordered
concatenation. An exact target-text match is never a source proof. Retain failed
or missing provenance explicitly. Save logical SQL selection digests for the
completed census databases and hash the original package and graph inputs.

Keep the fixed 128 DEV, 64 component and 320 final source reservations. Link
all recovered input variants by source family, original informative whole-turn
hash, original source component, group and exact input hash. The informative
whole-turn threshold remains 192 application tokens; generic short turns must
not join unrelated conversations. Propagate exclusions across entire connected
groups, including bridges introduced by newly recovered sources. Conservatively
exclude previously blocked source components as well. Check normalized exact
text and the existing >=32 shared distinct word-five-gram / >=80% shorter-text
containment rule against all 192 open DEV/component sources. Do not open or hash
sealed final source files. Metadata-based final protection is not represented
as a completed sealed-final full-text overlap audit.

For each raw pair, apply the unchanged native BERT limit of 1–126 lexical tokens
and reject native special/unknown IDs. Preserve every full target without
truncation. Keep multiple target variants for a source in the review pool; they
are not independent source coverage. Export eligible text and provenance as
`TRAIN_review_candidate` with `training_admitted: false`. Original-source
verification filters out untraceable fixture-like inputs; it does not certify
historical target quality. Semantic label review, codec reconstruction checks,
final prompt conditioning and a fixed training protocol remain separate gates.

Use locked Pixi ncm, `CUDA_VISIBLE_DEVICES=''`, a new append-only root
`20261027_recovered_pair_training_pool_v1`, and an 8 GB host-RSS ceiling. Stop on
source/export identity mismatch, incomplete accounting, memory failure, or a
changed producer; preserve the root and failure. Focused CPU tests precede the
full pass. No model generation, teacher/judge call or optimizer update occurs.
