# Trained source hierarchy versus dense source attention

Exploratory hypothesis `ncm-source-hierarchy-task-comparison-v1`: adding a
per-source hierarchy to source attention at every NCM block improves
complete-summary semantic accuracy relative to twelve-block dense source
attention under equal training data and exposure. Systems qualification is
complete for the corrected tree, including real FP32 gradient parity, FP16
updates, and exact restart on RTX2060. No hierarchy model has received full
task training or semantic DEV evaluation. Previous application DEV is 0/128;
public epoch4 evaluation and existing/expanded task supervision are in progress.

Two new arms, in fixed training order `tree`, then `dense`:

- `tree`: the corrected eight-way source hierarchy, per-source active-level
  masks, all twelve blocks, token leaves retained, activation checkpointing.
  Trainable population: 123,063,936.
- `dense`: source-token attention at all twelve blocks, activation checkpointing
  around each source branch, no ancestor path. Population: 115,939,968.

Both start afresh from the exact public epoch2 checkpoint, use all 1,376
unchanged admitted TRAIN pairs and their complete source features/native codes,
and train on physical GPU1 RTX2060. The Llama body is represented by its frozen
feature cache; codec and word-source path remain frozen. Added output projections
start at zero; query projections copy the same existing templates. Installation
must preserve original parameters, outputs and RNG. Never initialize full runs
from a systems pilot.

Reuse the exact existing task-supervision schedule: 2,048 optimizer updates,
32 examples per update, microbatch four, two prefetched batches, fixed seeds
251142/252142/253142, exactly half noisy presentations, FP32 weights/features
with FP16 autocast, FP32 CPU AdamW, warmup 128 to 3e-5 and decay to 3e-6,
clipping 1, recovery checkpoint every 256 updates, 4.9 GB own allocation cap.
Each arm gets exactly 65,536 presentations and the identical case/noise order.
Use read-only feature streaming, locked Pixi, deterministic algorithms and
TF32 disabled. No changes to concurrent training producers.

The existing completed three-source-block task arm on GPU0 is an additional
quality reference, not the primary matched hardware comparison. Its smaller
105,323,136-parameter trainable head and different GPU must remain explicit.
The tree/dense comparison also changes parameter count; it tests adding this
complete mechanism, not whether hierarchy alone explains every improvement.

Before full training, require focused CPU checkpoint/RNG tests and a real dense
lifecycle using the same four source/target-extreme TRAIN controls, clean/noisy
FP32 serial/batch gradient checks, two accumulated updates and exact restarted
second update. Reuse the completed tree lifecycle v3 rather than rerunning it.
Any failed contract is preserved and stops that attempt. No silent batch,
precision, seed, learning-rate, update-budget or threshold adjustments.

For each full run, save the exact model/optimizer/scaler/RNG state, require
unchanged protected tensors and strict final reload, and independently replay
all case positions, corruption counts, learning rates and loss denominators
before admitting semantic evaluation. Fixed final checkpoint only. Record
training times and memory, but do not present them as inference speedups.

Evaluate each completed audited arm, in tree-then-dense order, on the unchanged
source-disjoint 128-case DEV protocol. Retain every qualified Sol/Terra first
judgment and inherit all completed intervening evaluation branches before new
calls. Preserve the existing two-cohort and same-case semantic qualification
rules. Report full-summary passes out of 128, not code/token agreement. No new
test/train split, target-aware prediction, judge retry selection or best-seed
selection is permitted. The first qualified candidate proceeds to the existing
sealed-final protocol; 95% DEV alone does not complete the overall goal.

The dense baseline remains part of this comparison even if the tree is weak.
Full reconstruction work on the separately running codec is not installed in
either arm, so it cannot confound this architecture comparison.

Roots: `20261027_source_hierarchy_{arm}_{stage}_v1`, with binary checkpoints
under `/data2/structure_net/ncm_source_hierarchy_20261027_v1`. The qualified tree
pilot is `20261027_source_tree_training_lifecycle_v3`. This is one fixed
optimization seed per architecture; generalization across training seeds and
end-to-end latency remain unmeasured.
