# Qwen3B progressive continuation toward all 36 blocks

Date: 2026-11-15. ID: `JSPACE-QWEN3B-PROGRESSIVE36-EXPLORATORY-20261115`.
Status at writing: prospective continuation; the selected 15-block parent is
already observed. This plan targets blocks 20 down through 0, one additional
compressed block per stage. Existing all-at-once training on the 6 GB RTX2060
continues independently and is excluded from this controller's allocations.

Before launch, the user redirected the RTX3060 to Phi-4-14B compression. The
generalized Qwen implementation is retained and tested, but this ladder has
not been prepared or launched. The saved 15-block parent is unchanged.

## Question and scope

Can the established progressive interface-fit/joint-recovery recipe extend
the verified 15-block suffix to all 36 blocks while retaining its original
development accuracy limits? Every intermediate stage and failure remains
visible; reaching a complete structural model does not certify general quality.

This explicitly continues the existing sparse multiple-choice experiment at
the user's request. It keeps that experiment's objective, data and controls.
It does not silently substitute the newly implemented complete-response
distillation standard, or claim compliance with its generation-recovery gates.
Whole-response recovery, generation evaluation and final deployment measurements
remain separate qualifications of any resulting model.

The parent at
`data/experiments/jspace_3b_broad_recovery/20261115_v2/progressive15`
completed 288 joint updates and fresh verification. Its selected checkpoint is
`train/update_288.pt`, SHA-256
`22adc8b9870b294b8a3405fb7b3be511ea2f0bed6bdef0ae7fb74dc479c2b1be`.
Development is 189/256 (73.828125%) versus original native 195/256 (76.171875%).
The earlier 14-block parent's failed 128-case audit and six-prompt generation
collapse remain attached to the lineage. No new held-out or generation result
is implied by the 15-block improvement.

## Fixed data and selection

Use exactly the existing 4,608 TRAIN and 256 DEV cases, tokenizer/model revision,
native content-four teacher targets, choice-row scoring and training order.
The four equal-size development families are ARC-Challenge, BoolQ, HellaSwag
and WinoGrande. They cover varied short multiple-choice tasks but are reused
for selection; their scores are not untouched full-benchmark estimates.

Every stage compares with the **original native** reference, never a rolling
parent tolerance. Overall accuracy may fall by at most five percentage points:
at least 183/256. Each family may lose at most six of its 64 cases from native.
No audit/final-test examples enter calibration, training or checkpoint selection.
Reusing DEV across up to 21 stages increases selection pressure; final claims
need an independently specified confirmation and generation protocol.

Select eligible candidates by highest DEV correct count, then earliest update;
the raw initializer precedes an equal-scoring warm initializer. Preserve raw,
warm, update-72/144/216/288, diagnostic-best and selected checkpoint receipts.
The parent is the explicit fallback. Strict fresh reload must reproduce the
selected metrics and records before that checkpoint may initialize the next
**exploratory** stage. Passing DEV is continuation eligibility, not promotion
to confirmed quality.

## One stage, repeated without changing its recipe

For parent cut `c+1`, compile native block `c` and inherit the entire recovered
suffix, normalization, fixed bases and existing bridges unchanged. Cuts descend
20, 19, ..., 0. Cut 0 means all 36 transformer blocks are structurally treated;
the tied embedding/readout remains native and must count in eventual storage.

The phases remain `boundary`, `capture`, `fit`, `parentfeatures`, `pilot`,
`prealign`, `train`, `verify`:

1. Capture original native pre-`c` states for all 4,864 cases. Recompute the
   adjacent pre-`c+1` states and require bit-exact agreement with the authenticated
   parent's native boundary bank. Stop the native forward at that boundary
   after both states are captured; no later block is needed. Hook cleanup and
   the cut-0 embedding boundary receive CPU tests. Preserve the established
   native versus compact RoPE constructor distinction.
2. Recompute functional/Jacobian probes against the actual frozen recovered
   parent: 128 fixed TRAIN cases, 32 per family, two independent groups of eight
   full-vocabulary VJPs, four content positions, and all-training means.
3. Fit rank-1,024 residual and 64-dimensional head projections. Compile the
   non-polynomial interface first, then fit the existing corrected-rank
   Gröbner/polynomial relations with 24 anchors and 1,024 removed MLP channels.
   Do not substitute the independent all-at-once run's previously fitted layers.
4. Store actual parent post-block-35 answer coordinates for TRAIN only. Fit
   their normalization energy on TRAIN; use feature coefficient
   `0.4192096950932534`. Preserve original native content-four logits as the
   output distillation teacher.
5. Run discarded representative entrance and joint-update pilots. Restore the
   exact raw initializer and start fresh optimizer state for science.
6. Prealign only the new block for one epoch: microbatch 4, accumulation 4,
   peak LR `2e-5`, existing cosine schedule and seed 20862. Select between
   initial/end interface features by DEV normalized MSE, earliest tie. Freeze
   inherited blocks, norm, geometry and bridges during this phase.
7. Jointly train all compressed blocks and final norm for 288 updates: one
   complete TRAIN pass, effective batch 16, LR `5e-6`, clip 1, zero weight decay,
   seed 20861. Use `0.7*T²*KL + 0.3*CE`, T=2, plus the fixed parent-feature term.
   Parameters are BF16; optimizer masters/moments use the existing FP32 resident
   and CPU-spill implementation. There is no adaptive epoch extension.
8. Freshly verify selected and distinct diagnostic checkpoints, complete fixed
   state, layer/bridge coverage, and exact development predictions/metrics.

All outcome-relevant producer code and dependencies are archived and hashed.
Use new generalized stage/capture modules; do not patch the frozen 15-block or
active all-36 producer. Admit large inputs once with hashes, then use immutable
file identity checks during work; do not repeatedly scan giant checkpoints per
update. Each consumed model checkpoint still receives its declared digest check.

## Reservations, storage and lifecycle

The CPU controller runs through locked/offline Pixi with CUDA hidden. It invokes
`inference-ctl lease run` before each worker process; it never assigns visibility
inside an already initialized CUDA process. Single-card phases request
`GPU-0370fdb0=11264` MiB (RTX3060). Joint pilot/train additionally request
`GPU-7b1db3cb=7168` MiB (RTX2060 SUPER). Allocator caps remain 11,008/6,912 MiB,
and the SUPER's optimizer resident budget remains 6,144 MiB; overflow uses the
existing bounded CPU spill. Each phase has a 90-minute lease.

The native full-model boundary forward and pilot establish capacity before
training at every new depth. An OOM, changed receipt, expired lease, numerical
failure or failed pilot stops execution and preserves the attempt. No batch,
precision, optimizer or scientific policy changes occur automatically.

The campaign lives on `/data`, preserving all original checkpoints. Preparation
checks a conservative 40 GiB per remaining stage plus 64 GiB free headroom;
before each next stage it rechecks that forecast. It does not delete intermediate
models to make room. Current storage is sufficient for the forecast, subject to
other writers. Concurrent training timing is never reported as inference speed.

The controller uses a file lock to prevent duplicate launches, logs every
subprocess/lease command, and advances only after a completed authenticated
phase. Completed matching stages are reusable; incomplete phases are preserved
for diagnosis rather than blindly rerun. Credential/prelaunch failures may be
retried only when no worker artifacts were created, under a bounded declared
retry policy. A stage's failed DEV gate stops descent and retains its parent;
further recovery is a separately specified follow-up, not a relaxed threshold.

Campaign records distinguish `running`, `execution_failed`,
`stopped_development_gate`, and `structurally_full`. Every state has
`confirmed:false`. Final structural completion must show all original layer
indices 0–35, a selected checkpoint and exact fresh verification. Accuracy on
diverse generation tasks, actual exported bytes and isolated prefill/decode
timing remain necessary before reporting a recovered, faster complete model.
