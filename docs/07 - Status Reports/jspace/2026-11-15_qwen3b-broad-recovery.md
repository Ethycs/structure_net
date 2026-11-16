# Broad recovery of the fourteen-block Qwen model

Status: prospective exploratory experiment, specified after the failed independent
sanity screen and before this experiment's model outputs. Study ID:
`jspace-qwen3b-broad-recovery-v1`. The user authorized execution of the recovery
plan. This replaces the unlaunched three-epoch continuation on the old 512 cases;
it does not alter that producer or any completed experiment.

## Question and fixed model

Does broader recovery data plus an objective anchored to the original model
recover the quality lost by the current compression? The separate option-order
diagnostic examines the WinoGrande failure without changing this trial's arms
or selecting its data after seeing diagnostic outputs.

Both arms start from the same verified epoch-1 checkpoint, SHA256
`814563a4de04d91c17bb02d670ed2d5a7609d3bb3460ed92073d721deac7e401`,
with compressed blocks 22–35. Native prefix blocks 0–21, the original tied
embedding/readout, learned bases and other fixed geometry remain frozen.
All fourteen compressed blocks' learned parameters and the final norm train
jointly. No additional block is compressed and no rank is changed.

Native model: the pinned Qwen2.5-3B-Instruct revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. Parent-feature teacher: the accepted
thirteen-block parent used by the cut22 campaign, checkpoint
`abc72e97e8f800517bc1b106a86bd740013647d90537fd68bcb7576bb0de9eec`.

## Data and evaluation separation

Use all 4,096 previously frozen additional training examples plus the existing
512, exactly once per arm: 4,608 distinct source questions. Family counts are
ARC 1,101, BoolQ 1,169, HellaSwag 1,169 and WinoGrande 1,169. Shuffle within
families deterministically, then interleave active families. Each case has the
same weight; batches are not claimed to remain exactly balanced after a family
exhausts. Both arms use the same fixed order and the same effective batch size.

Permute training choices deterministically, balancing correct option positions
within each family and choice-count group where possible. Remap the correct
answer and retain the original source identity and permutation. Preserve the
original order for detected position-dependent choices. Never reuse teacher
targets from a different option order. Tokenize with the pinned prompt/template
and reject silent truncation beyond the 384-token input cap.

Freeze a new development set of 256 examples, 64 per family, seed 582464; then
a separate recovery audit of 128 examples, 32 per family, seed 582465. Both
use original option order and exclude all historical source identities,
training variants, each other and the reserved final 512. Authenticate final
exclusions through the existing identity-only manifest. Neither final prompts
nor final labels may be opened. The recovery audit content is unavailable to
training and checkpoint selection; it is opened only after the global candidate
is fixed. The old 64 development cases and the exposed sanity128 are diagnostic
history, not the new selection population.

## Compared objectives

Arm `parent_legacy` uses the current objective: mean full-vocabulary
`0.7 * T^2 * KL(parent || student) + 0.3 * CE`, at temperature T=2 for KL,
at the original four positions `[0, floor((n-2)/2), n-2, n-1]`, plus the parent
final-coordinate feature loss. The first three CE targets are next prompt
tokens and the final target is the gold answer letter.

Arm `native_content` uses the same coefficients and temperature, with the
original uncompressed model as logit teacher, on three evenly spaced lexical
next-token targets within the user content plus the final gold answer position.
Exclude role delimiters, newline-only tokens and option-label tokens from those
three positions. Teacher/student losses are averaged over four positions in
both arms. Retain the same parent final-coordinate feature anchor as an
auxiliary term. This compares two recovery packages (teacher plus prediction
positions); it cannot isolate those changes from one another.

The auxiliary coefficient remains 0.4192096950932534 in both arms. Recompute
its target normalization energy from the common 4,608 TRAIN targets only;
do not calibrate against development or audit. Targets are the parent's actual
post-block35 compact coordinates at the last valid input token, before
reconstruction. No surrogate projection is substituted.

Capture fresh native pre22 and pre23 residuals from singleton unpadded forwards.
Cache native logits at both sets of positions, preserving BF16 bits. The parent
teacher consumes the native pre23 residuals and produces its legacy-position
logits and TRAIN-only final coordinates. Record its execution batch shape.
Both teachers remain frozen. Validation uses singleton, unpadded, choice-row
scoring in both arms and the native reference, matching the sanity evaluator.

## Training, checkpoints and selection

Use one complete pass, 288 optimizer updates at effective batch size 16,
constant learning rate 5e-6, and the existing microbatch/accumulation, clipping,
BF16 forward/backward and qualified resident FP32 Adam settings. Each arm
reconstructs FP32 masters from the same saved BF16 weights and resets optimizer
moments, counters and RNG. This is a weights-only warm start, not an optimizer
resume. Use the same fixed data order for both arms.

Run a discarded systems update first, verifying finite losses/gradients,
updates to every intended block, unchanged geometry/readout and correct
restore. Scientific training restarts from the untouched source. Save actual
weights and full development records at updates 0, 72, 144, 216 and 288. Both
arms complete the same budget unless validity, finiteness or reservation fails;
do not extend a run because its observed result is disappointing.

A checkpoint qualifies on the new development set only if its overall loss
relative to native is at most five percentage points (12 net lost out of 256)
and each family's loss is at most ten points (6 net lost out of 64). Rank
qualifying checkpoints by correct count, then fewer updates. Preserve the
source and a separately named diagnostic best even if none qualifies. There is
no CE/KL veto. An initially unqualified source is allowed to recover.

After both arms finish and fresh-process strict reload verifies each selected
checkpoint, freeze the global choice by the same development ranking. Exact
cross-arm ties favor `parent_legacy`; identical update-zero source states count
as one candidate. If none qualifies, freeze the diagnostic best but label the
development gate failed. Audit outcomes cannot rescue that failure.

## Independent audit and ordinary generation

Only after global selection is frozen, score original native and the chosen
candidate on the sealed recovery128. Report overall and per-family correct
counts, paired gains/losses and option frequencies. The descriptive audit gate
is at most 6 net lost overall and at most 3 per family. This small sample is
not a statistical five-point noninferiority proof. Do not choose a different
checkpoint or resample questions based on audit performance.

Run the six already written explanation, numeric summary, JSON, editing,
arithmetic and Python-function prompts as a separately scoped partial-model
generation diagnostic. Preserve the original full36 canary configuration and
admission rule; copy its prompts/requirements into this study's own frozen
configuration before generating. Compare native, the starting checkpoint and
the globally selected checkpoint. Use identical greedy settings, at most 64
new tokens, original EOS handling and total capacity 416, with cache disabled
in every arm to avoid introducing a new cache implementation. This is quality
diagnosis, not a throughput benchmark. Grade against the written semantic
requirements; preserve native failures. Do not select weights using these
canaries or train on their outputs.

## Operations and stopping

Use repository modules through locked/offline Pixi. GPU capture, evaluation and
generation use normal 3060 reservations. Training and its systems pilot reserve
the 3060 and 2060 SUPER for the qualified resident optimizer. Store large
artifacts on the data volume. Freeze source, configuration, inputs, teachers
and outputs with hashes; completed artifacts are append-only. Preserve all
failed phases and stop on invalid provenance, changed fixed state, nonfinite
values, failed reload or lost reservation. CPU preparation hides CUDA.

This experiment does not automatically advance compression depth. A failure
leads to a separately specified sensitivity/capacity investigation, not more
epochs or a silent relaxation of the quality budget. Report actual results,
including any disagreement between development, audit and generation.

Planned roots: `jspace_3b_broad_recovery_data/20261115_v1`,
`jspace_3b_broad_recovery/20261115_v1` and
`jspace_3b_broad_generation/20261115_v1`, under `data/experiments/`.
