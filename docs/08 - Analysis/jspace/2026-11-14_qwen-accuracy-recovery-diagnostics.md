# Qwen recovery: residual compression fails before the head

Status: completed development diagnostics; recovery training remains in progress.
Study: `JSPACE-QWEN-ACCURACY-RECOVERY-01`, exploratory, `confirmed=false`.
Primary diagnostic aggregate:
`data/experiments/jspace_accuracy_recovery_diagnostics/20261114_v1/result.json`.
SHA256: `30733a8ad6a6fdc0befd195062a9f567a26c568358cd8f6c0eb7f3d9579b548f`.

These experiments respond to the failed all-block adaptation: native396/512,
compressed softmax211/512 and quadratic220/512. Those exposed evaluation cases
are diagnostic-only. A separate512-case confirmation population remains
unscored; its identities are excluded from further training.

## First-block intervention

Only block0 changes. Blocks1–35, embedding and readout remain native. Every arm
uses the same64 development prompts and FP32 arithmetic from BF16-rounded
native weights. Compressed arms preserve the full input residual and compute
compact branch updates; actual omitted-residual energy enters RMS denominators.
No recovery weights, learned attention metric, or polynomial attention are used.

| First block | Correct /64 | Wins/losses against native |
|---|---:|---:|
| Native |50|0/0|
| Full residual rank, full head dimensions, new wrapper |50|0/0|
| Full residual rank, half head dimensions |45|2/7|
| Half residual rank, full head dimensions |27|7/30|
| Half residual rank with mean preserved, full heads |30|7/27|

The wrapper control changes no predictions; its maximum absolute label-logit
difference from the native run is0.0035934. The residual-rank intervention causes
the larger failure. Repairing one global mean direction is insufficient. This
does not prove that every half-rank basis fails, or that recovery is impossible.

The first block's original1024-column basis contains334 numerically resolved
task-Jacobian directions and690 weak completion directions. The construction
uses the block's input Jacobian for both its input projections and output update
range. High retained input-Jacobian energy did not ensure preservation of the
actual native update. Training-only input/update enrichment is therefore a
specific next intervention, rather than another unchanged full-stack run.

## Other completed controls

An oracle using cached native final hidden states scores48/64. Projecting those
states into the final compressed basis and using either the original folded
readout or either fitted readout scores49/64. Actual student states score24–26/64.
The final basis/readout can represent the decisions on these development cases;
the upstream student states are damaged. These cached BF16-state results are
distinct from the newly executed matched-FP32 native50/64 baseline.

All36 residual-carry blocks score20/64 with native-projected initialization and
29/64 with the prior fitted compact weights. Preserving the global mean in all
36 fitted bases, without retraining, scores23/64. A first-block-only version of
that repair with half-size heads scores25/64. None passes the accuracy target.

Compressed-prefix/native-suffix probes score27/64 after replacing one block,
22/64 after12, and30/64 after30. These locate early damage; the nonmonotonic
counts are not evidence that adding damaged blocks restores the native mechanism.

## Recovery objective and its failure mode

Joint recovery now supervises all36 compressed blocks from the original frozen
teacher, with full-vocabulary KL at four positions and answer-token cross
entropy. The original tied vocabulary readout remains frozen. Checkpoint
selection prioritizes development accuracy, with KL only breaking ties.

After the first epoch, development loss falls27.5173→7.8148, but accuracy falls
29/64→20/64. Initial predictions are B62 times and A twice; epoch1 predicts A59
times and B five times. Neither predicts C or D. The development truth contains
A20/B30/C8/D6. The apparent baseline competence is largely answer-letter bias.

Full-vocabulary answer cross entropy decomposes into conditional choice cross
entropy plus a penalty for assigning little total probability to valid answer
letters. Conditional choice cross entropy worsens1.2370→1.2660, while mean total
answer-letter probability rises from6.19e-7 to0.4301. The loss reduction is
principally a format/calibration repair, not improved choice accuracy. Training
continues under the fixed three-epoch budget; no accuracy-recovery claim follows
from these intermediate losses.

All GPU runs use experiment reservations. Every intervention has a separate
artifact root, source/input hashes and retained predictions. CPU tests cover
native-width parity, causal masks, geometry, gradients and serialization. These
diagnostics do not establish generation quality, a size gate, or a speedup.
The interrupted earlier CPU benchmark supplies no usable timing result.

Source roots: `jspace_accuracy_oracles/20261114_v1`,
`jspace_accuracy_recovery/20261114_*`, `jspace_first_block_ablation/20261114_*`,
`jspace_prefix_localization/20261114_v1`, and
`jspace_answer_mass/20261114_joint_v1`, all under `data/experiments/`.

## Same-width reachable-state repair

A subsequent, separately frozen intervention keeps the334 strong J columns
bit-identical, adds global-input and position-zero input/update mean directions,
and fills the remaining slots from equal trace-normalized input/update second
moments. It uses128 training cases for fitting and a disjoint128 training cases
for projection diagnostics. The total residual rank remains1024. Head dimensions
remain native128 in this first-block test; there is no recovery training.

Development accuracy rises from27/64 to46/64:27 gains and8 losses against the
old half-rank basis. Against native50/64, it has0 gains and4 losses. Per-family
counts are ARC15, BoolQ13, HellaSwag12 and WinoGrande6 out of16 each. It therefore
shows a substantial local repair but still exceeds the allowed five-point
accuracy drop. It is not an all-block or held-out confirmation result.

On the separate training projection check, native-update energy lost falls
43.64%→8.32%, and the output-J-weighted update error falls22.87%→3.29%. Retained
input-Gram trace remains99.70%. These energy measurements include position0;
its large update can dominate aggregate energy. Further all-block diagnostics
must also report non-position0 errors. The fit does not explicitly preserve the
intermediate attention-to-MLP state, so these projection metrics cannot replace
the observed task test.

Fit: `data/experiments/jspace_first_block_enrich/20261114_v1`, fingerprint
`c433057e20d736af26a09f4697b7b4d5c67ceb0f4caca4ac297445b90a3182ec`.
Probe: `data/experiments/jspace_first_block_enrich/20261114_probe_v1/result.json`,
SHA256 `fcb05c30869551204623fb0c8f081584de2f8858bee10ebfe2f678dd35b43d7e`.

The distinction between input statistics and output sensitivity has relevant
precedent: [IO-SVD](https://arxiv.org/abs/2605.15626) combines activation whitening
with an output-side KL sensitivity metric. Our fixed-J plus reachable-state
enrichment is a different intervention; that paper does not validate this run.
[SVD-LLM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3104e1ab39875cf54fe1eb4473e7c5a1-Abstract-Conference.html)
also couples compression with sequential parameter updates. These sources
motivate checking both the representation and its recovered computation rather
than treating input-Jacobian energy alone as a quality certificate.
