# Joint output distillation of the recovered Qwen assembly

Study: `JSPACE-QWEN-RECOVERED-JOINT-DISTILLATION-01`, exploratory.

## Evidence and question fixed before execution

The user's successful two-block example included staged compression followed
by joint output distillation and supervised recovery. Its student recomputed
its own complete forward path, with gradients through both compressed blocks.
That example used a0.7 teacher-KL /0.3 label-CE objective at temperature2.
The supplied account is evidence of that recipe; its external sandbox artifacts
are not locally reproduced by this study.

Our current Qwen2.5-3B assembly has separately recovered blocks, full2048-wide
residual carry, rank1024 updates, native128-dimensional heads, softmax, and
11008-wide MLP hidden activations. The native FP32 development control scores
50/64; accepted compact blocks0/1 score49/64. All36 selected compact blocks
score27/64, while retaining native12/15 scores24/64. All34 independent fits
completed, but those local fits did not establish composition.

An earlier whole-stack Qwen trial used different bases, reduced heads and
BF16 parameters. Its full-vocabulary loss largely improved answer-letter mass
without improving the conditional answer decision. It is not a test of joint
recovery from the current selected FP32 checkpoints with an explicit decision
objective. The previously prepared semantic runner was never launched.

Question: can joint output supervision from the original frozen model recover
the accuracy of the CURRENT all36 compact assembly, without increasing its
rank, replacing its geometry, or resetting any recovered block?

## Initialization and original teacher

Authenticate the completed four-arm assembly and use its frozen selection for
every compact block. Block0 retains its first-block fit; blocks1..35 retain
their corresponding all-layer fits. Strictly load the recovered FP32 states
before any update. Preserve every weight value, the full residual path and
fixed geometry. The initialization must reproduce the all-compact27/64
predictions and logits within absolute5e-4 plus relative1e-5 tolerance.

Use the same512 training and64 development prompts and original token IDs.
Cache new targets from the ORIGINAL frozen model, whose stored BF16 weights
are promoted to FP32 for its forward pass. Save full-vocabulary FP32 logits
at the existing midpoint and final-answer positions, slots1 and3 of each
case's four loss positions. Do not cache the two universal formatting slots
for this objective. The original FP32 teacher must reproduce50/64 and the
authenticated native development predictions with the same logit tolerance.
This avoids inheriting the old BF16-forward teacher's48/64 decision context.

Cache identities, order, positions, model/source hashes and precision are
mandatory. Targets are detached and immutable. No teacher optimizer exists.
The fresh512 confirmation set remains unopened and excluded from all fitting.

## Student and objective

Every minibatch begins with raw prompt tokens and recomputes all36 student
blocks. No cached teacher hidden state may replace a student intermediate.
Train all compact parameter tensors and the final RMS gain together. Keep
the original tied embedding/readout matrix, selected bases, anchors, RoPE and
other geometric buffers fixed. Preserve FP32 parameters, gradients, residual
arithmetic and readout. Use nonreentrant activation checkpointing per block.

At temperature T=2, optimize the fixed prospective objective

`L = .2 T^2 KL_full + .5 T^2 KL_choice + .3 CE_choice`.

`KL_full` averages the teacher-to-student full-vocabulary KL at midpoint and
answer. `KL_choice` uses only the answer position, normalized over that case's
valid A/B or A/B/C/D tokens. `CE_choice` supervises the gold choice using that
same conditional distribution. Padding and unavailable choices are excluded.
The two KD terms retain total weight0.7 and supervised CE weight0.3, while
explicit decision losses address the observed formatting failure. These weights
are a declared first test, not an empirically established optimum. The full
vocabulary term is an anchor, not a guarantee of general generation quality.

Report actual task accuracy, teacher agreement, conditional-choice CE/KL,
full-vocabulary KL, answer-letter probability mass and choice margins
separately. No hidden-state MSE is optimized in this continuation.

## Fixed training and selection

Run3 epochs,32 optimizer updates per epoch,96 total: microbatch1, accumulation16,
AdamW learning rate1e-5, zero weight decay, global gradient clip1, seed27865.
CPU optimizer masters and moments are FP32, with32MiB transfer chunks and
a24GiB state cap. This is a bounded initial joint-recovery test; changing its
duration or objective requires a separately recorded continuation.

Evaluate initialization and each completed epoch. Select highest development
correct count, breaking ties with lower conditional-choice CE. Keep the
initializer eligible. Preserve all epoch metrics and predictions; retain only
the currently selected model checkpoint during training, using an atomic best
checkpoint replacement. Final selected weights and their hashes become immutable
at completion. The scientific gate is at least47/64 against native50/64.
Improvement over27/64 is intermediate progress, not a pass of that gate.

Reload the selected checkpoint strictly, then verify it in a fresh process.
Require exact predictions and existing numerical tolerances. Separately report
packing parity before any deployment benchmark. If the selected state misses
47/64, preserve the failure and do not open confirmation or claim speed success.

## Resource and implementation gates

Use the experiment reservation system for every GPU process. Planned allocation:
RTX3060 GPU-0370fdb0 at11264MiB; RTX2060 SUPER GPU-7b1db3cb at7168MiB.
Resolve physical UUIDs from the lease instead of assuming CUDA enumeration.
Blocks0..23 and the single tied embedding/readout live on3060; blocks24..35
live on2060 SUPER. Transfer the full hidden state at the partition boundary
and only selected output rows back for the full-vocabulary head. No gradients
are detached at the boundary. Native teacher capture uses the same placement
without gradients. This allocation is for correctness and memory capacity,
not an inference-speed claim.

Cap each Torch allocator at its lease minus256MiB and check both live
reservations and peaks. Retain failed receipts; no silent CPU fallback,
automatic scientific retry, or precision change. Keep locked offline Pixi,
the ncm environment for model execution, and bounded CPU optimizer storage.

Before training, pass focused CPU math/gradient/lifecycle tests and a fresh
longest-context GPU pilot. The pilot first checks untouched initialization
against27/64, then executes one real accumulated optimizer update. Verify
finite gradients in every trainable tensor, gradient flow through both cards,
updates in every block, and unchanged frozen state. Pilot weights are discarded;
training starts again from the authenticated selected assembly.

New roots, preserving prior completed evidence:

- `data/experiments/jspace_recovered_teacher/20261114_v1`
- `data/experiments/jspace_recovered_joint/20261114_v1`

Freeze and archive outcome-relevant code, tests and this protocol before GPU
execution. Record commands, allocations, checkpoint/cache hashes, completed
updates, peak memory and terminal status. All results remain development-only,
confirmed:false. A successful joint recovery would still need fresh confirmation
and a separate matched whole-model deployment benchmark.
