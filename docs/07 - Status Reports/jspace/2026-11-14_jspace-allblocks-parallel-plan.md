# All-block Qwen compression with local parallel fitting

Status: prospective exploratory implementation and systems validation.
Study: JSPACE-QWEN-ALLBLOCKS-01. Created 2026-11-14 UTC.

The target is at least 38% fewer deployed tensor bytes and a 1.4× whole-model
speedup, with no more than one percentage point of aggregate benchmark accuracy
loss. These are the user's current targets. The available MNIST notes support
26.8% tensor reduction and 1.44× final-block/readout speedup; they do not establish
that the historical example achieved the same whole-model targets.

## Method and scope

Use the pinned Qwen2.5-3B-Instruct checkpoint from the completed affine study,
all 36 decoder blocks, and the four-task ARC/BoolQ/HellaSwag/WinoGrande benchmark.
Retain the original embedding/readout matrix and vocabulary, tied and frozen.
Retain native width 2048 for RMS normalization and native attention scale
1/sqrt(128), while executing 1024-dimensional residual coordinates and
64-dimensional attention heads. Preserve the unmodeled MLP residual: replace
1024 of 11008 channels with the quadratic normal forms from exactly 24 anchors.
The polynomial feature count is 325. Pad its merged output to width 10368.

Process six groups in reverse depth order: 30–35, 24–29, 18–23, 12–17, 6–11,
0–5. Each group's frozen teacher is the previously selected recovered suffix
with its original native prefix; the first teacher is the original model.
Collect one shared teacher forward per case and one VJP per scalar output
probe for all six requested blocks. Recompute Jacobians against each selected
teacher. Never calibrate using a partially modified live teacher.

For each layer, first obtain the functional basis from the benchmark's
full-vocabulary randomized task-Jacobian Gram, then derive head projections,
then fit and compile the polynomial relations. Calibration is not performed
after compression. Polynomial candidates are assessed at the actual compiled
interface using the development set. Store every MLP-context derivative,
including zero rows, as the backward computation's exact BF16 bit pattern;
stream it as FP32 for decomposition. This is a storage encoding, not additional
lossy quantization.

Six independent polynomial fits are distributed over the local RTX3060,
RTX2060 and RTX2060SUPER, one process per GPU and two fits per worker. Their
decompositions use FP32. BF16 forward/backward recovery and fused deployment
use the RTX3060. Every GPU process is supervised by an inference-ctl lease.
No external model APIs, cloud workers, or model downloads are part of this run.

After fitting the group, assemble a single contiguous compact suffix with
fixed affine coordinate bridges. Restore each earlier layer's original FP32
fit geometry before constructing these bridges; retain its learned weights.
The training path encodes once, recomputes every trainable suffix layer, and
reconstructs once. The inference path folds fixed offsets, pads kernels,
uses the previously verified Triton pointwise kernels, and removes unused
intermediate native bases and anchor tables. Size includes every deployed
parameter and buffer, including bridges, norms, and the tied vocabulary once.

## Recovery and selection

The shared source recipe is staged teacher/student output distillation with
joint recovery, rather than independent local layer MSE. Train all compact
suffix parameters and the final RMSNorm for 12 epochs on the same 512 training
cases; all fitted bases, anchors, geometry, bridges and vocabulary remain
fixed. Use 0.7*T²*KL(teacher||student)+0.3*CE, T=2, over four fixed next-token
positions per case. The current selected teacher supplies KL targets; retain
the original teacher's targets separately for cumulative-drift diagnostics.

Microbatch 4, accumulation 4, effective batch 16, AdamW learning rate 1.2e-4,
weight decay zero, global gradient clipping at one. This scales the preceding
batch-4 / 3e-5 recovery setup to reduce optimizer transfer overhead. Use a cosine
learning-rate schedule over the fixed optimizer-update budget. Model parameters
and gradients are BF16; the CPU holds persistent FP32 masters and Adam moments.
The master state is updated without retaining a full CPU gradient copy. This
is an explicit resource adaptation, not bitwise equivalence to CUDA AdamW.
Block activation checkpointing is enabled. Frozen native prefix and teacher
weights are unloaded during recovery; prefix states and targets are disk
memmaps. No stale intermediate training activations are reused.

Select the lowest packed-development cross entropy among epoch zero and the
12 completed epochs. Development uses 64 cases and cannot select the final
test outcomes. Retain selected unpacked recovery weights for the next group
and packed inference weights, per-epoch metrics, and fitting artifacts. Keep
the best snapshot in CPU memory during training, avoiding twelve full disk
checkpoints per group. Failed quality gates remain reported; a poorer group
does not silently revert to an uncompressed block. Nonfinite arithmetic,
reservation violations, changed frozen producers, invalid calibration,
checkpoint reload failures, or insufficient storage stop execution.

## Systems gates and measurements

Before expensive recovery: focused CPU contracts; a tiny real grouped GPU
capture; shape-equivalent full-36-block forward/backward and CPU-master update;
strict save/reload and fused-versus-reference checks; and concurrent fits on
three cards. A repeated-layer systems model may be used to establish memory
and throughput, with any residual scaling explicitly recorded. It supplies no
task-quality evidence. Systems artifacts use a separate dated root and the
label `systems_lifecycle_only_not_quality_evidence`.

The final evaluation contains 512 newly selected cases, disjoint from the
training/development cases and all previously exposed evaluation pools in
the affine, local-distillation, and fast-recovery studies. Freeze their IDs
before training, score only at the end, and report native and compressed
accuracy, per-family accuracy, paired wins/losses, answer agreement, and the
exact paired binomial test. This task-restricted, single-checkpoint experiment
does not establish general language-model competence or exact equivalence.

The joint success gate is: valid execution/reload; at least 38% total deployed
tensor-byte reduction; at least 1.4× whole-model CUDA-graph decode throughput
on the same local RTX3060 and fixed eight prompts; no more than 1pp aggregate
accuracy loss and no family loss greater than 4/128. Report dynamic prefill,
eager generation and graph generation separately; local block timings
cannot satisfy the whole-model gate. Use the established three-repeat runtime
method and 32 generated tokens, preserving the 416-position supported range.
The primary whole-model rate includes dynamic prompt prefill and all 32 output
tokens; graph construction time is reported separately and amortized.

Partial size, speed or quality results cannot pass the joint gate. Preserve
all failure stages, finite but degraded trajectories, and allocation records.
Freeze every outcome-relevant source and input hash before launching primary
evidence. New producing code requires a new root or explicit dated amendment.
Finalize the measured report, conservative registry record/read-back, and
source/tests/protocol commit after the campaign closes.
