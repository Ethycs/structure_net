# Layer-wise distillation after benchmark-derived Gröbner compression

Status: prospective exploratory execution protocol. Date: 2026-11-13.
Study: `JSPACE-QWEN-LAYERKD-01`.
Root: `data/experiments/jspace_qwen_layerkd/20261113_v1`.

The user proposes distillation between each compressed block and the original
model. This extends the benchmark-derived compression experiment; it does not
replace its controls or move distillation ahead of the compression step.

The subsequently supplied successful-method account clarifies that its
"layerwise" procedure meant staged compression with output-level distillation,
followed by joint training of both compressed blocks. Its second-stage teacher
was the previously recovered compact model. It did not use hidden-state or
attention-map matching. The reported 97.44% test accuracy belongs to that
small-image experiment and has not been independently verified here. Source:
[user-provided schedule](</home/rabbit/.codex/attachments/dabdb122-3aa4-4259-abcf-1d46b7d56a3d/Pasted text.txt>).

The user explicitly requests retaining the new distillation angle. Therefore
the output-only recipe remains the baseline and local Euclidean/Jacobian losses
remain controlled additions. This first one-block comparison measures those
additions; it does not reproduce the subsequent two-block joint recovery. That
next stage must use a common recovered compact teacher, newly captured teacher
outputs, a fixed direct coordinate bridge and fresh student activations through
both trainable blocks on every minibatch. All experimental arms must share
those choices. The source's learning rate 4e-4, batch 256 and validation-CE
selection are not the settings of this Qwen comparison; its explicitly declared
settings below are exploratory adaptations, not a literal hyperparameter
replication.

## Rationale and literature

[FitNets](https://arxiv.org/abs/1412.6550) supplies intermediate teacher hints
to train thinner students. [TinyBERT](https://aclanthology.org/2020.findings-emnlp.372/)
applies intermediate and prediction-level supervision to Transformer students.
[Task-aware Layer-wise Distillation](https://proceedings.mlr.press/v202/liang23j.html)
argues that matching all teacher hidden information can overburden a smaller
student and instead learns task-aware filters. These support investigating
intermediate supervision. They do not establish that our Qwen/Gröbner method
will succeed. Using a benchmark-derived task-Jacobian metric as the filter is
our proposed adaptation and requires its own comparison.

The working hypothesis is that a local teaching signal can limit accumulated
compression error before multiple compiled blocks interact. The teacher and
distillation losses are training-only. Deployment contains the compiled blocks
and their retained native residual computation, with no added original-model
blocks between them.

## Preserve the order

1. Run the task benchmark through the frozen original model; estimate J-space,
   select functional anchors, and fit shared Gröbner graph relations.
2. Compile the selected channels and attention coordinates, retaining the
   uncovered native computation. Save this exact initialized checkpoint.
3. Distill that compiled block against the original block before adding the
   next compressed block. Preserve the initialized control.
4. Build a small contiguous stack, refresh inputs from the actual compressed
   prefix, and apply joint final-output distillation after local adaptation.
5. Pack, reload, and measure final task accuracy, tensor bytes and actual speed.

The one-block experiment is already being prepared. It can establish local
loss behavior but cannot establish prevention of accumulated multi-block drift.
A follow-up contiguous two-block test is required before a longer stack.

## Supervision and controls

First compare identical compiled initialization, data order, optimizer, update
budget and selection objective for three losses: final-output KL+CE alone;
KL+CE plus normalized layer-boundary matching; KL+CE plus task-Jacobian-weighted
boundary matching. Choose checkpoints using the same development KL+CE metric,
then compare untouched task accuracy. Loss weights, scales and the extension
evaluation split are frozen before running this separate study, using the
concrete settings below.

For the multi-block experiment, let `s` be the input actually produced by the
current student prefix, `T_i` the frozen original block, and `S_i` its compiled
replacement. Supervise `S_i(s)` against `stop_gradient(T_i(stop_gradient(s)))`.
This isolates local functional error at the same input. Separately measure
distance to the original full-model trajectory and preserve final teacher-logit
supervision on the original benchmark prompt. Same-input agreement alone does
not guarantee agreement with the original complete model.

Compare states in the same native coordinate system using the existing affine
reconstruction `a + U z`. Report retained-coordinate error and discarded-space
error separately. The affine discarded component creates an error floor; do
not interpret inability to remove that floor as failed optimization.

For functional matching, use downstream VJPs defined at the **output boundary
being compared**: mean over probes of `dot(g_k, e)^2`, where
`e = S_i(s) - T_i(s)`. Take each dot product over the complete token-by-channel
boundary before squaring, preserving cross-token terms through downstream
attention. The current pooled input-boundary Gramian cannot simply be reused
as this output-boundary metric. A tokenwise pooled PSD approximation would be
a separately labeled control. This remains a local sensitivity approximation,
not a finite-perturbation guarantee or a replacement for final evaluation.

For Euclidean loss, projecting the teacher into the student's affine subspace
removes an irreducible constant without changing its gradient. For the
Jacobian-weighted loss, retain the full native-coordinate difference: projecting
the teacher first generally changes the weighted optimum because accessible
and discarded directions can interact through the downstream Jacobian.

Use training-derived fixed normalizers rather than dividing each example by a
possibly tiny teacher residual update. Record individual loss values and
gradient norms/cosines, including how local and task gradients interact on
shared trainable parameters. A large loss value does not itself demonstrate
gradient dominance. A concrete first candidate is to calibrate the auxiliary
coefficient once to 0.1 times the endpoint-gradient norm on training calibration
minibatches, then freeze it and report its subsequent ratio. Finalizing this
rule and any alternatives precedes the separate experiment; the KL+CE-only
control and identical training budgets remain essential.

## What the earlier experiment did and did not show

The earlier full-model run already included local reconstruction and propagated
student-prefix caches. Its local target came from the original teacher
trajectory, while the student received its compressed-prefix input. Thus local
error combined input drift and block approximation error. Its per-example
normalizer was the teacher residual-update energy, which can be small.

The reported 63.05% statistic concerned weighted local **loss values** exceeding
the functional loss; it did not measure actual optimizer-gradient dominance.
The new comparison must measure that directly. Intermediate supervision is
therefore a refinement to test, not an untried explanation that erases the
previous negative result.

## Execution boundary

The current `JSPACE-QWEN-AFFINE-01` controls remain KL+CE-only. This note does
not silently change them. Any additional training uses a distinct frozen run,
the experiment reservation system and locked Pixi. The first decision is
whether intermediate supervision improves accuracy at the same compression
and training budget. Increasing the number of compressed blocks follows only
after the one-block validity checks and a measured two-block comparison.

## Concrete first comparison, authorized 2026-11-13

Use the already compiled final-block `r1536_h64_gb` architecture and the exact
benchmark-derived basis, anchor tables, head selectors and Gröbner relations
from `jspace_qwen_affine/20261113_v2`. Reconstruct and save one shared FP32
trainable initializer before fitting loss weights. Do not initialize from one
of the parent's adapted checkpoints. Layers 0–34 and the tied vocabulary stay
frozen. Train the compressed final block and final RMSNorm gain in all arms.

The three arms are `endpoint`, `euclidean` and `jacobian`. Each runs exactly
12 epochs over the same 512 training examples, with the same seed-861 order,
AdamW learning rate 3e-5, zero weight decay, cosine decay, accumulation four and
gradient clipping one: 1,536 optimizer steps per arm. No early stopping.
Endpoint loss remains `0.7 * 2^2 * KL(teacher_2 || student_2) + 0.3 * CE`.
Select the strictly lowest packed development endpoint loss, including epoch
zero, from the same 64 development examples for every arm. This shared
criterion deliberately does not include auxiliary losses.

Both auxiliary arms use the same four cached output positions, so their
comparison changes the metric rather than the amount of token supervision.
The Euclidean target is the original teacher boundary projected into the fixed
student affine space; report the discarded teacher floor separately. The
Jacobian arm uses the full native-coordinate teacher difference, without that
projection.

For this final-block comparison, compute the full-vocabulary first-order metric
directly. The frozen suffix is only RMSNorm and a linear readout, so an exact
analytic differential is available and random probes are unnecessary. For
teacher state `t`, error `e`, original norm gain `gamma`, width `d` and
`r = sqrt(mean(t^2) + epsilon)`, use
`v = gamma * (e/r - t*dot(t,e)/(d*r^3))` and
`L_J = sum_positions v^T (E^T E) v / (4 * vocabulary_size)`.
Compute the readout Gramian once in FP32 chunks of 4,096 vocabulary rows, with
TF32 disabled. Its 2,048-square matrix occupies 16 MiB. Store the original gain
and epsilon with it. Compute the differential in FP32 from the original BF16
teacher states and weights. This is exact in real arithmetic for the local
Jacobian, with ordinary floating-point error in execution; it is not an exact
finite-perturbation or BF16-rounding model. Unit tests compare both its value
and gradient to an explicitly materialized tiny full Jacobian, including
nonzero epsilon. This metric is defined at the original block output and
cannot be replaced by the parent's input-boundary Gramian.

Freeze each auxiliary weight using eight training examples per task family
(32 total), all evaluated at the identical compiled initialization. Compute
per-example endpoint and raw auxiliary gradients over the shared block
parameters, excluding the final RMSNorm because the local loss does not depend
on it. Set raw coefficient to `0.1 * sqrt(sum ||g_endpoint||² /
sum ||g_aux||²)`. Reject zero/nonfinite calibration norms. Use the mean initial
raw auxiliary loss over those 32 examples as a fixed normalizer and multiply
the coefficient by that same normalizer, preserving the raw-weighted objective.
No per-example residual-energy denominator is used. This specifies initial
gradient balance, not a guarantee about Adam update magnitudes. Record later
gradient norms and cosine on one fixed training diagnostic case each epoch.

Endpoint KL and actual held-out accuracy remain the outcome measures. The
analytic metric removes random-bank fitting error in this one-block test.
Earlier-layer suffixes will require different Jacobian machinery, preserving
cross-token terms; this closed-form simplification does not extend through
arbitrary attention layers.

Freeze 512 further evaluation questions, 128 per task family, disjoint from
training, development, the old 256 evaluation cases and the parent's new 512.
Use the same supported two-to-four-choice and prompt-length filters, recording
exclusions. Inspect final results only after all three training arms finish.
Report per-case paired changes against the endpoint arm and native model,
development divergence, actual deployed tensor bytes and identical runtime
measurements. All three compressed models must have the same tensor footprint.
Repeat native timing last and reload all selected checkpoints in a separate
reserved process, requiring exact held-out records.

Use a separate systems root. Its setup computes the output metric and the 32
coefficient-calibration cases; its longest-prompt shakedown performs two
optimizer updates per arm, packing/reload and CUDA-graph generation. This is
systems evidence only. Then freeze the main root and run the fixed campaign through `inference-ctl lease run`
using locked `ncm` Pixi on the RTX 3060 with 11,264 MiB reserved. Never modify
the parent's producing sources while either campaign uses them.

This first comparison tests whether intermediate supervision helps an already
compressed block. A useful result means higher fresh task accuracy at the same
deployed architecture and compute; endpoint divergence is supporting evidence.
Tiny differences on one seed are exploratory, not confirmation. The next
progressive-stack experiment must measure actual student-prefix drift and
refresh same-input teacher targets; it is not established by a one-block result.
