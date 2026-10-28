# Source hierarchy training lifecycle

Systems qualification `ncm-source-tree-training-lifecycle-v1`, not quality
evidence. The completed real-checkpoint pilot already demonstrated exact initial
parity and gradients for every new source branch. This follow-up checks the
actual training path before any architecture comparison.

Use the same epoch-two initializer and all 1,376 existing admitted TRAIN pairs,
with their unchanged native target codes and source-feature cache. Install the
eight-way source hierarchy at all twelve NCM blocks. Preserve the frozen codec,
word-source path and Llama body; train the generator and concept-source modules.
The prototype retains token-level attention and adds dense attention to pooled
ancestors. No speedup or sparse routing is claimed.

Use physical GPU1 RTX2060, locked Pixi, deterministic FP32 master weights and
source features, FP16 autocast for training, FP32 CPU AdamW state, TF32 disabled.
Use the existing task-supervision 32-example effective batch, four-example
microbatches, two prefetched batches, exact sampler/noise seeds, and learning
rate schedule. Only the declared physical-GPU metadata differs. This is a
qualification for that existing schedule, not a change to its ongoing runs.

First reproduce initial source-only outputs on four existing TRAIN controls
(first, shortest source, longest source, longest target), before and after
installing the tree. Compare serial and batch-four FP32 logits and gradients
for clean and noisy code histories using the already fixed parity tolerances.
Require all 123,063,936 trainable parameters to be included in the optimizer,
with fresh FP32 masters matching the loaded weights.

Then run two full accumulated updates drawn from the complete admitted
population, save all model/optimizer/scaler/RNG state after each, restore step
one, and require exact replay of step two including source-only control outputs.
Require finite gradients, unchanged protected tensors, unchanged other worker
identities, no skipped optimizer updates and peak allocation below 4.9 GB.
The long-source parity controls also exercise worst-case memory. A failure is
preserved; there is no silent batch, precision, seed or tolerance fallback.

Use root `20261027_source_tree_training_lifecycle_v1`; checkpoints go to
`/data2/structure_net/ncm_source_tree_20261027_v1/lifecycle`. No DEV or final
examples, semantic judges, or quality-based checkpoint selection are used.
Successful qualification permits planning a matched quality comparison; it
does not establish that the hierarchy improves summaries. That comparison must
separate three-source-block, twelve-source-block dense, and twelve-block tree
attention under equal data/exposure and an explicit hardware policy.
