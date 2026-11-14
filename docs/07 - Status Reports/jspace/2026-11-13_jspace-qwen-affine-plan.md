# Qwen functional-coordinate compression with necessary architectural adaptations

Study: `JSPACE-QWEN-AFFINE-01`. Date: 2026-11-13.
Prospective exploratory follow-up to the completed full-model compact campaign.
Root: `data/experiments/jspace_qwen_affine/20261113_v2`.

Systems amendment before quality measurement: the v1 two-case calibration
completed, then relation diagnostics attempted CUDA int64 matrix multiplication,
which PyTorch does not implement. V2 computes the same exact eligibility overlap
counts using Boolean intersection and integer reduction. No scientific setting
changes. Preserve the failed shakedown and the prepared, unexecuted v1 main root.

## Question and known evidence

Can a locally calibrated affine-coordinate attention-plus-MLP block in
Qwen2.5-3B retain useful task accuracy while reducing stored state and measured
execution cost? Qwen is the experimental target. No MNIST replication is planned.

Known before this protocol: the previous all-layer experiment scored 88/256,
versus 210/256 native; MLP-only scored 205/256. The previous implementation used
one answer-margin derivative, a pooled basis, aggressive head/input cuts and an
additional local loss. Those results do not measure the proposed intervention.
The supplied recipe's 97.12% classifier result is not a Qwen result.

## Method and justified adaptations

Start with decoder layer 35, preserving the other 35 layers. This retains the
recipe's final-block compression boundary on the actual Qwen model and isolates
its effect before considering stacking. Run the ARC, BoolQ, HellaSwag and
WinoGrande benchmark training splits through the original model first. The
order is original-model task-Jacobian calibration, functional-anchor selection
and shared polynomial-relation fitting, channel projection and affine folding,
head projection, compilation of covered MLP channels with a retained native
residual, joint adaptation, inference packing, and measurement. Compare native, full-rank
compiled, rank-1536/head-128 channel-only control, rank-1536/head-64 and
rank-1024/head-64 blocks, plus a rank-1536/head-64 block with benchmark-derived
Gröbner compilation. All four trained arms receive the same adaptation budget
and objective. The matched rank-1536/head-64 pair isolates the incremental
effect of the polynomial compilation. No extra reconstruction training loss
is introduced.

Estimate a layer-local full-vocabulary output-Jacobian Gramian at the four
teacher-cache output positions per training example. These are three prompt
next-token positions and the final answer-letter position. Different examples,
positions and probes receive independent Rademacher signs over all 151,936
vocabulary logits. Pull each sign vector through the unchanged vocabulary
matrix and differentiate through the actual final RMSNorm and last block.
Average over probes and sum over every input token. Signs are scaled by
`1/sqrt(4*vocabulary_size)`, so the expected operator is the mean-output
`J^T J`, not an answer-margin operator. This is an unbiased stochastic estimator,
not an exact materialized vocabulary Jacobian. Use 32 training cases per family
and two independent groups of eight probes per case. Report independent-group
subspace overlap and cross-group captured energy. Fit the final basis to their
average; do not select probes using evaluation accuracy. The unbiased identity
holds mathematically; BF16 forward/backward and FP32 accumulation introduce
numerical approximation. Report the rank/width random-subspace overlap reference
alongside independent-group overlap, since large subspaces overlap by chance.

At the original MLP intermediate activations, retain all full-vocabulary VJP
rows at the same four output positions: 128 cases × 16 probes × 4 positions =
8,192 rows. Use deterministic randomized low-rank SVD over this entire derivative
bank, after multiplying columns by training activation scales (derivatives with
respect to standardized coordinates), to select nested functional anchor sets
of 16, 24 and 32 channels. Record
the sketch dimensions, seed and captured derivative energy. Use 64 singular
components, 16 oversampling directions and one power iteration; stream 256 rows
at a time. This replaces the
previous scalar answer-margin anchor bank; it does not invent a task-free basis.

Fit shared degree-two graph relations on the original model's cached 128
training cases, sampled at 16 positions per case (2,048 activation rows). Mean
and scale are fitted on these rows only. Features are constant, normalized
anchors and their pairwise products. Ridge is 0.01 with an unpenalized intercept;
round standardized coefficients to a grid with denominator `2^20` for the
rational graph-ideal certificate. Each candidate removes
exactly 1,024 nonanchor nonlinear channels, selected by pooled training relation
error weighted by the projected output-column norm. All remaining channels,
including anchors, keep their native nonlinear computation. Compare the three
widths using development projected-output reconstruction error at the actual
rank-1536/head-64 interface; polynomial coefficients remain fitted to native
training activations. Record native-development and compiled-development errors
separately, so changes caused by preceding compression remain visible.

The graph ideal has generators `h_j - p_j(anchors)` with the eliminated
channels ordered before the anchors. Distinct linear leading monomials make
these generators a Gröbner basis and give direct polynomial normal forms.
This formal statement certifies the fitted surrogate, not exact identities in
the pretrained model. Measure relation defects, per-task and cross-task errors,
shared eligibility counts and overlap of task-specific candidate sets. The
diagnostic eligibility threshold is normalized RMSE ≤ 0.5; it does not turn all
1,024 forced-budget removals into proven covered channels. Report any removals
that exceed this threshold. No accuracy claim follows from relation fit alone.

Compile the selected normal forms into the MLP output matrix and bias, physically
remove the corresponding gate/up rows, and retain the rest of the MLP. During
adaptation train the retained path and compiled polynomial output coefficients.
Pack their output columns into one dense multiplication for inference. Report
the added feature computation and buffers in both timing and size accounting.

For `U^T U=I`, set `a_t=(I-U U^T) mu_t`. Compute position means from all 512
training inputs only. Positions absent from those inputs use the training global
mean, explicitly recorded with their counts. Tables cover positions 0–415;
unsupported positions are errors. The experiment's prompt cap remains 384 and
timing generation cap 32. This does not establish long-context quality.

Qwen uses RMSNorm, not LayerNorm. Compile it with the original width:
`s_t(z)=sqrt((||a_t||^2+||z||^2)/2048+eps)` and
`W RMSNorm(a_t+Uz)+b=(Fz+g_t)/s_t(z)+b`.
Keep original GQA grouping, RoPE, causal masking and SwiGLU hidden width.
The full-head arm isolates residual-channel compression. For the complete arms,
reduce each head from 128 to 64 coordinates. Arbitrary head rotations do not
commute with Qwen's distinct-frequency RoPE; for a fixed incoming projection
that preserves that action, select 32 complete frequency pairs per KV group.
Use the same selection for its key and all eight queries. This is the declared
positional adaptation for this implementation, not a claim that all other head
compression methods are impossible. Preserve the original `1/sqrt(128)` scale.

Rank pairs using the full-output derivative of a shared multiplicative gate on
each Q/K frequency pair: sum activation-times-gradient over both pair members,
all tokens, and the group's queries and key, then square and average independent
probes/examples. Do not count the shared-key derivative eight times. This is a
local sensitivity criterion, not a guarantee for finite joint removal. Record
the pair-gradient Gramian and the removed-mask quadratic form. Fit the shared
per-KV-group 128-to-64 value basis to uncentered value second moments, including
mean energy. Record total and mean energy lost; do not assume exact mean
preservation. Fold these projections into incoming and outgoing matrices.

Attention input/output projections and both MLP projections operate on the
smaller residual coordinates. Project residual updates after attention and
after MLP, carrying the reduced state throughout the block. Compact attention
matmuls actually use width 64. The existing static-cache container has 128 lanes;
pad only cache storage and slice back before attention. Dynamic cache can store
64 lanes. Report both footprints; do not claim a static KV-cache reduction.

Precomputed position shifts remain fixed. Learn a shared zero-initialized
numerator offset for each incoming projection, alongside the compiled weights
and existing biases; fold that offset into the tables for inference. Fixed
geometry uses FP32, inference linear weights and shift tables use BF16, with
FP32 norm arithmetic. Record the actual cost of every table and basis buffer.

Reconstruct the native-width state once at block exit, preserving the tied
input embedding/vocabulary matrix. Folding a second vocabulary matrix would
add hundreds of megabytes while the prefix still needs the original embedding.
Adapt the final RMSNorm gain as well as the compressed block, but keep the tied
vocabulary weights and prefix frozen. No input/output vocabulary duplication
is hidden from accounting.

## Data and adaptation

Reuse the exact BF16 teacher cache: 512 training and 64 development examples,
78,115 total prompt tokens. Cache identity and tensor hashes must match the
completed capture. The teacher is fixed; cached-prefix training must reproduce
native logits before any intervention.

Use loss `0.7*T^2*KL(teacher_T || student_T)+0.3*CE`, `T=2`, averaged over the
four output positions. Use AdamW, learning rate 3e-5, weight decay zero, effective
batch four through accumulation, and gradient clipping at one. Run up to 12
complete training passes, with cosine learning-rate decay. Evaluate the same
KL+CE objective and answer accuracy on a packed deployment-precision copy after
each pass, leaving the live optimizer parameters untouched. Select the strictly
lowest development objective, including the initialized checkpoint. Stop after
four consecutive passes without an absolute objective improvement of 1e-4,
after at least four passes. Preserve every epoch's metrics and the selected
checkpoint; minimum improvement affects patience only, not best-checkpoint
selection. Selection must not use evaluation results. No claim that the source
classifier used this unspecified optimizer/budget is made.

Primary evaluation is 512 newly selected, length-filtered questions, 128 per
family, disjoint from training, development and the previously inspected 256
questions. Preserve the existing two-to-four-option interface; count exclusions
for unsupported choice formats (the ARC test file contains three five-option
items), duplicates and excess prompt length. The old 256 are a secondary continuity result. Freeze identities
before calibration and inspect primary accuracy only after all trained arms have
finished adaptation. Include actual answer accuracy, paired changes and
uncertainty; reconstruction or gradient-energy percentages are not accuracy.
Neither multiple-choice set establishes summarization or general-generation
quality. This is one seed and one checkpoint, not independent confirmation.

## Controls, validity and execution

First run CPU tests for affine RMS algebra, full-rank identity, projected
reference equivalence at reduced rank, cache semantics, gradients, packing and
the sketch estimator, polynomial substitution algebra, actual row removal,
task-shared fit and strict checkpoint reload. The explicit reduced-rank reference must project after
both residual updates. Test the full-output estimator against an explicit tiny
Jacobian using exhaustive signs.

A separate GPU shakedown uses two cases for the expensive Jacobian bank and
the full cached activation pool to exercise the fitting code. It is systems
evidence only. It exercises the longest cached prompt, backward,
optimizer updates, BF16 packing, checkpoint reload and cached generation. Check
that the prefix/vocabulary remain frozen. The full-rank compiled control must
have development choice agreement of at least 63/64 and mean full-vocabulary
KL no greater than 0.01; otherwise stop the quality campaign and diagnose the
implementation. Numerical failure, source/hash mismatch, invalid reservation
or exceeding reserved memory stops execution. Poor reduced-rank accuracy is a
reported outcome and does not invalidate a technically sound experiment.

All GPU work uses root locked `ncm` Pixi through inference-ctl leases on the
RTX 3060, requesting at most 11,264 MiB. CPU preparation uses locked Pixi with
CUDA hidden. Freeze producing source and configuration in a new append-only
root before each GPU stage. Changed producing code requires a new root/version.

Measure native and all selected compressed models with the same eager and
CUDA-graph generation implementations. Use eight fixed development prompts,
batch one, 32 generated tokens including prefill. Repeat graph measurements
three times and measure native again at the end to expose drift. Report setup
time separately. Measure isolated final-block prefill latency separately from
whole-model tok/s. Count learned tensors, all buffers, checkpoint bytes and
unchanged KV cache. Reload selected compact checkpoints in a fresh reserved
process and compare evaluation logits exactly.

The report must separate a working compiled implementation, useful quality/
size tradeoffs, and demonstrated speed gains. No hypothesis confirmation or
whole-model scaling claim follows automatically. Preserve source snapshots,
failed attempts, per-case outputs and all timings, then register the scoped
result with fresh-process metadata readback.
