# Train all source-conditioned concept levels before fitting generated-history words

Date: 2026-10-23. Exploratory hypothesis `ncm-source-hierarchy-bootstrap-v1`.
The latest complete summarizer failed0/128. The isolated source-root control
then reached173/176active and253/256all root labels on16TRAINcases, versus2/176
with shuffled sources. That capacity result permits moving to the complete
concept hierarchy. It does not qualify summaries or held-out accuracy.

Initialize from the registered final1,024 source-root checkpoint SHA256
91aece481ae0448126e80b5e29eb65e2c9b1d4950dbae4aa2c4a75e63cb093fb.
Use all252already vetted complete TRAINsummaries and their existing frozen
features and original target-derived concept labels. The16root-control cases
are part of these252; report their earlier training history. No new512-source
teacher label is used in this bootstrap. Their labeling and feature preparation
can proceed in parallel, and a separate later protocol may incorporate them.
Neither development source nor judge feedback enters optimization.

Keep original Llama weights/MLPs and the original codec frozen and absent from
the optimizer. The parent is not loaded. Freeze the copied word projection and
decoder and the unused finest VQ head. Train only new source bridge, shared
generator, learned root queries and the first eight concept heads. Compute one
parallel training forward over the608coarse queries, without word queries or
word logits. Current-level input contains earlier-level reconstruction only.
Targets enter code CE; word IDs enter only the active/padded-block loss mask.

For each level, average CE over codes in48-position blocks containing at least
one non-PAD target word/EOS, plus.25times mean CE over entirely padded blocks.
Weight the root level4and other levels1, normalized by their total weight11.
For each update, a separate CPU noise generator chooses clean history with
probability.5; otherwise uniformly replace conditioning entries within their
4,096-entry codebook. Rates decrease linearly from.35at level0to0at level7.
Do not change clean targets, source inputs or unused finest labels. This adapts
the paper's uniform conditioning corruption and directly addresses weak source
learning and generated-history errors; it is not a faithful paper replication.

Fix16,384updates, batch1, no accumulation, uniform sampling of252TRAINcases.
This provides about65sampled exposures per case, comparable to the successful
small source-learning control rather than the earlier short mixed objective.
Model/dropout seed16142; sampler17142; noise18142. AdamW beta=(.9,.95),wd.01,
clip1. Warm256updates to3e-4then cosine to3e-5at16,384. FP32master/optimizer,
BF16autocast and FP32loss/reconstruction. Disable both matmul and cuDNN TF32.
Retain generator dropout.3. No word loss or teacher word logits in this phase.

Before the full run, require a fresh two-update GPU0pilot using the first and
longest old TRAINsources; pad the latter to3,328, the observed maximum source
frame of the additional512-source preparation. Use real features and labels,
finite loss, gradients in every coarse head and source bridge, frozen words/
finest head/codec, peak allocation below8GB, exact model/optimizer/CPU-CUDA/
sampler/noise RNG reload and exact source-only code/word probe after reload.
The full run starts fresh from the same registered root checkpoint, never pilot.

Stop on nonfinite values, invalid gradient routing, changed frozen state,
producer/input mutation or memory failure, preserving every artifact. Save
2,048-step recovery checkpoints, select only final16,384, and do not extend the
budget or select a checkpoint using losses or development outputs. All update
sample IDs, corruption histories, losses, rates and observed source frames are
retained. Recovery must preserve the total budget and exact optimizer/RNG state.

Fresh roots `20261023_source_hierarchy_bootstrap_{pilot,train,results}_v1` under
the paper-regime directory. Full hierarchy training is not the final candidate:
afterward, freeze a generated-history word-training and source-only complete
summary evaluation protocol using vetted expanded data. All930prior task
judgments and any new label judgments remain canonical. Both final cohorts
stay closed. No training fit, bootstrap loss or component score establishes
95%task accuracy, compression or speedup.
