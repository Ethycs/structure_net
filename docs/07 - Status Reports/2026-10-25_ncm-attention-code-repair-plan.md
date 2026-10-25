# Attention repair of actual NCM concept prediction errors

2026-10-25. Exploratory mechanism
`ncm-source-attention-code-repair-lifecycle-v1`. The user proposed using
attention to correct codes. The preceding two-case oracle-prefix diagnostic
showed root-only correction was insufficient, while all-oracle codes restored
substantially coherent decoding. This motivates a separate repair pass over
the complete hierarchy, not another decoder-only training run.

Freeze the parent, existing NCM generator, codec and native codebook geometry.
Add two width-256, four-head attention blocks. Queries represent all 641 actual
predicted codes, with native codebook embeddings, position and level identity.
Each block attends bidirectionally across the full code hierarchy and reads
all original source features through shared source keys/values. A residual
query head scores each native level against its own frozen codebook by
Euclidean distance. A zero residual and positive copy preference must reproduce
every input code exactly at initialization, including duplicate book vectors.

Prediction receives predicted codes, source features/mask and frozen books
only. Clean target codes enter the loss only. Use equal-weight per-level
cross entropy, plus 0.05 times MSE between the soft reconstructed full native
latent and the oracle latent. The latent term couples coarse/fine residual
choices; it does not guarantee semantic quality. Train on actual frozen
generator mistakes rather than uniform independent noise. Do not train on
component, permanent development or sealed final cases.

The first CUDA lifecycle uses exactly the first and longest-source cases
among the 16 already archived TRAIN diagnostics of the new final checkpoint.
Their source-only predicted codes must be bound to the audited checkpoint and
unaltered source features. No target-derived selection. Use two AdamW updates
at 1e-4, FP32, TF32 disabled, no dropout for reload checks; record the complete
configuration, seeds, losses, gradient norms, model/optimizer/RNG state and
strict reload. Require finite nonzero gradients in the residual head and
source attention by the second update; codebooks and source features must
receive no gradients or updates. Maximum allocated memory 3 GB on GPU2.
The parent and NCM generator need not be loaded for this cached-code lifecycle.

CPU tests first check exact identity initialization, genuine source influence,
masked-source isolation, gradients into attention, no gradients into inputs or
books, and rejection of incomplete codes. All passed/failing executions and
original predictions remain append-only. The two-update CUDA result is only
a systems check, not evidence that correction improves unseen summaries.

Root: `20261025_attention_code_repair_lifecycle_v1` under
`data/experiments/ncm_paper_regime/`. A successful lifecycle is followed by a
separate prospectively fixed TRAIN learnability comparison and then a larger
source-disjoint quality experiment. Include unchanged generator output and
source-disabled or shuffled-source repair controls. Never select a checkpoint
using sealed final data. The semantic lineage guards remain intact.

An extra attention pass adds computation to the current NCM. No speedup is
implied. After quality qualifies, measure total source-to-summary latency and
memory on matched inputs and hardware, including the repair pass and frozen
parent. A possible smaller generator plus repair is a future comparison,
not an established result.
