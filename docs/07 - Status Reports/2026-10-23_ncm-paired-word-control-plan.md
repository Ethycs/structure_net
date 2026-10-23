# Paired clean/noisy word training before further scaling

Date: 2026-10-23. Exploratory hypothesis `ncm-paired-word-control-v1`.
The frozen native model failed 0/64 development summaries. The crossed probe
found that correct target-derived coarse plans barely improve its four-pass
word decoder, although true previous words restore strong training recovery.
Do not change those completed results or open either sealed final cohort.

Test whether the existing word interface can learn at all under a closer
JacobiForcing objective. Start from the exact failed 2048-update checkpoint,
not random initialization. Keep original embeddings, all 28 body blocks,
MLPs, word head and concept books frozen. Train existing new query adapters,
hint and biases only; concept heads have no supervised loss in this control.
This is a combined training intervention, not a single-factor attribution or
replication of the upstream full-model trainer.

Use only the first four fixed summary TRAIN cases from the original immutable
CPU cache. Run exactly 128 optimizer updates, batch two, accumulation one,
AdamW learning rate 1e-4, weight decay .01, clip norm 1. Seeds are 7042 for
case sampling and 8042 for trajectory pass selection. The first 16 updates
use clean target-derived encoded coarse plans; later updates alternate
target-derived and source-predicted plans. Gold plans are legal training
conditions and explicit oracle controls, never source-only inference.

Every update includes both branches at identical weights and the same plan:

1. Clean native previous words: hard target CE multiplied by 10.
2. Native noisy words: initialize by rounding the selected plan, then run
   zero to three current-model parallel passes. Train to the detached full
   native-vocabulary distribution from the current clean branch, temperature
   one. Exclude padding and positions whose entire previous-word prefix is
   identical to the clean branch. No sparse top-K approximation.

Backpropagate the two terms sequentially before one clipping/optimizer step
to avoid retaining two body activation graphs. This has the same gradient as
their summed objective; test that equivalence on CPU. Word uncertainty is
from actual imperfect word trajectories. No Gaussian noise is added to the
clean teacher, no teacher API or stored offline teacher logits are used.
This differs from upstream trajectory packing, full-model training and
iterative inference; it borrows the paired objective, not published results.

First test mask offsets, full-distribution soft CE, detached teacher gradients
and sequential-versus-joint gradients on CPU. Then run a separate four-update
CUDA lifecycle pilot alternating summary/language tasks and both plan types,
using the first two cached training cases of each kind. Require finite losses,
nonzero new-parameter gradients, no original-weight gradients, unchanged
original body/books/cache, checkpoint/optimizer/RNG exact reload and finite
free outputs. Use physical GPU0, BF16 body/FP32 new weights, TF32 disabled.
Require observed terminal success before the 128-update control.

Retain before/after A (source plan) and B (oracle plan) outputs on all four
training summaries, using the same fixed four parallel word refinements.
The provisional capacity gate is at least 90% valid-position token recovery
under oracle plans after 128 updates, with all four examples retained. This
is a memorization/decoding control, not summary task accuracy or evidence of
generalization. Failure stops scaling this recipe; success admits a separate
source-disjoint training/evaluation study, not a 95% claim. Do not grade or
select checkpoints, extend the update budget, run development/final evaluation,
or test speed within this control. Save the final weights, optimizer and RNG
state plus every update's losses, plan source, sampled IDs and gradient norm.

Roots: `20261023_paired_word_control_pilot_v1` and
`20261023_paired_word_control_train_v1`. Preserve producer snapshots, input
hashes, raw outputs, all failures and actual worker terminal receipts.
