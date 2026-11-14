# Full model compact coordinates and polynomial residual experiment

Date: 2026-11-13. Study: `JSPACE-FULL-COMPACT-01`.
Prospective exploratory extension, not independent confirmation of the preceding
four-task pilot. Root: `data/experiments/jspace_full_compact/20261113_v1`.

## Objective and scope

Apply the compression pass to all 36 Qwen2.5-3B-Instruct decoder layers. Combine
the supplied affine-coordinate compilation and adaptation technique with the
shared polynomial core, while retaining native computation for the uncovered
component. Measure actual whole-model accuracy, tensor bytes and generation
throughput. The preceding 209/256 result and 41.46 tok/s runtime are known; neither
the pasted classifier's 97.12% nor its speedup is a Qwen result.

The source recipe is the user attachment at
`/home/rabbit/.codex/attachments/3a9e1366-cc2e-4086-92d0-193f8890637b/Pasted text.txt`.
Its linked library implementation is not accessible through the public page;
the provided equations are sufficient to implement and test the transformation.

## Coordinates and retained residual

Build a common channel basis from task-Jacobian Gramians at the input to every
layer. Average each layer's Gramian over calibration examples, normalize each
layer by its trace, and sum across layers. Its orthonormal eigenbasis is
`Q=[U,V]`, with 1,024 functional coordinates and 1,024 retained residual
coordinates. Offline eigendecomposition supplies the full Ritz space; there is
no eigensolver or Krylov iteration during inference.

Carry `[z,r]=Q^T h` throughout the body. Fold RMSNorm gains into the incoming
linear projections and Q into their weights; fold Q^T into output projections.
The RMS denominator uses `(||z||²+||r||²)/2048 + epsilon`, preserving the
uncovered component's normalization contribution. This extends the supplied
affine formula to a retained residual, rather than silently treating that
residual as zero. Embeddings are rotated once on entry and the final state once
on exit, preserving the tied embedding/readout weights and avoiding a duplicate
vocabulary matrix. No full-width reconstruction occurs inside decoder layers.

First evaluate the complete-coordinate, uncompressed rotation control. Its
double-precision algebra must match an original block before CUDA work. BF16
rounding changes are measured on the model.

## Attention and polynomial compilation

For every layer, retain one native GQA group (eight queries, one key/value head)
and compile the other group. Select the group with lower calibration Jacobian
energy. The compact group's Q/K/V projections read the 1,024 functional
coordinates; the discarded input-coordinate contribution is anchored at its
calibration mean. Both groups use the full residual-aware RMS denominator.

Keep 48 of the original 64 RoPE frequency pairs, giving 96-dimensional compact
queries and keys. Pair selection uses calibration Q/K functional sensitivity;
it commutes with the original rotary action. Use a 96-dimensional value basis
from calibration covariance and fold it into V and O. Preserve the original
attention scale 1/sqrt(128) and every causal token position. Native and compact
groups perform ordinary softmax attention in 128 and 96 dimensions respectively.
The initial cache adapter pads compact cache rows to the HF 128-dimensional
container; therefore this version does not claim a KV-cache byte reduction.

At each MLP, fit a common degree-two graph basis in 16 task-Jacobian-selected
anchor activations. Remove 1,024 nonanchor gate/up channels, ranked by worst-task
development reconstruction NRMSE times calibration Jacobian sensitivity. Keep
all other nonlinear channels. Compile the removed contributions into 153 normal
form features (constant, linear and packed quadratic terms). Stack those features
with the native residual channels for a dense output multiplication. Report
cross-task reuse with the previous NRMSE <=0.5 definition, separately from this
larger engineering removal budget. No reuse threshold is a stop rule here.

## Data and progressive adaptation

Use the same pinned public ARC, BoolQ, HellaSwag and WinoGrande files, prompt
format and 384-token limit. Take 128 training and 16 development items per family
from official training data, and preserve the original 64 evaluation items per
family. The new splits must be disjoint; evaluation reuse is explicit. Calibration
uses the first 32 training items per family, with 16 token positions per item.

Capture original teacher states at all 37 block boundaries and selected full
vocabulary logits. Stream BF16 state bits from disk; do not retain the full
training corpus in Python memory. A common Jacobian basis and every polynomial
fit use training/calibration data, with channel-budget ranking on development.
No evaluation item selects a basis, rank, channel budget or checkpoint.

Adapt one compiled layer at a time, leaving native residual weights and basis Q
fixed. Use the progressively transformed student-prefix states as inputs and
the original teacher states/logits as targets. Future layers provide a frozen
full-coordinate teacher continuation until their own conversion. Train the
current compact attention coefficients and polynomial output coefficients for
96 steps, batch one, AdamW learning rate 1e-4, gradient norm cap 1.0, seed 861.

The task/language loss is `0.7*T²*KL(teacher_T || student_T) + 0.3*CE`, T=2,
at three prompt positions plus the final correct-option prediction. Add 0.1
times normalized local block-output MSE for neighborhood correction. Prompt
positions use their actual next token as the CE target. Adapted coefficients
are learned parameters and need not remain the exact original folding.

After adaptation, propagate the student's layer outputs to the next on-disk
boundary. Preserve each layer's optimizer/loss summary and coefficients. Evaluate
development and the same 256 held-out questions after layers 12, 24 and 36;
these are trajectory measurements, not checkpoint selection or stopping gates.

## Execution, controls and measurements

Run CPU tests for RMS folding with a retained residual, complete-coordinate
block equivalence, RoPE pair selection, graph substitution and gradient routing.
Then run a separate representative CUDA shakedown with a real backward and
checkpoint reload. Freeze all producing source and inputs before expensive work.

Capture, decomposition, conversion, adaptation and timing use root locked `ncm`
Pixi and inference-ctl reservations on the RTX 3060. Reserve up to 11,264 MiB;
check the lease before CUDA and monitor actual peak use. Invalid numerics,
broken fingerprints, incorrect exact controls or memory outside the reservation
stop execution. Poor task accuracy is a measured outcome, not grounds to abandon
this exploratory pass or relabel it a failed research direction.

Final controls are original Qwen, complete-coordinate rotation, initialized
compact model and progressively adapted compact model. Record task correct
counts, per-family accuracy, original-label agreement, choice KL, full-vocabulary
distillation losses, changed parameter/tensor bytes, and basis reuse by layer.
Measure standard generation and a matched CUDA-graph path on both original and
adapted models, using the previous eight prompts and 32 generated tokens with
prefill included. Report graph setup separately. Preserve any unavailable timing
as an explicit limitation; do not compare an optimized student to an unoptimized
teacher as evidence of a compression speedup.

The result remains exploratory and `confirmed=false`. Archive source, protocol,
data identities, all outcomes and amendments; write a measured report and scoped
NAL record with fresh-process readback. Do not overwrite the completed pilot.
