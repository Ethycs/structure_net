# Incremental expansion of the recovered Qwen suffix

Study **JSPACE-QWEN-INCREMENTAL-SUFFIX-01**, exploratory, dated 2026-11-15 UTC.
Artifact root: `data/experiments/jspace_incremental_suffix/20261115_v2`.
This is one prospectively specified seven-block experiment, not an automatic
campaign extending to all 36 blocks. All results remain `confirmed: false`.

## Question and prior evidence

Can adding only block 29 to the previously recovered compact blocks 30–35,
then jointly recovering all seven, retain task accuracy within five percentage
points of both the accepted six-block model and the matched native model?

The historical group-0 checkpoint in
`data/experiments/jspace_qwen_parallel/20261114_v2` improved from 43/64 to 51/64
after output distillation. It completed 12 epochs and selected epoch 2 by packed
development cross-entropy. Its recovered checkpoint SHA256 is
`81026565492400de386d56f5765e6a4e8335e609af96dc96197e67ed0a4ca675`.
The next historical stage added six blocks at once: its initializer scored
35/64 and its selected twelve-block model scored 43/64. Subsequent expansion
continued with degraded teachers. These are reused development observations,
not independent confirmation.

The new intervention adds one block, jointly adapts the connected suffix, and
selects by actual task accuracy before cross-entropy. This combines several
changes from the historical expansion; success would demonstrate a viable
incremental step, not isolate which changed choice caused it.

## Model and immutable initialization

Use the same local Qwen2.5-3B-Instruct revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`.
Retain the historical suffix architecture: 1,024 residual coordinates,
64-dimensional attention heads with the original attention scale, softmax,
24 polynomial anchors replacing 1,024 MLP channels, fixed affine bridges,
BF16 trainable weights/forward computation, and the existing FP32 geometric
tables where required. This is different from the later FP32 full-residual,
full-head all-block experiment.

Start blocks 30–35 and the final norm from the exact group-0 recovered state.
Add only the original group-1 block-29 initializer: its functional basis,
interface and polynomial relation were fitted against the accepted group-0
teacher before group-1 recovery. Do not import any of the unsuccessful
twelve-block recovered weights. Preserve every inherited trainable tensor and
the six-block internal bridge buffers at initialization; construct only the
new block and its new interface from authenticated original fit geometry.

The original native prefix 0–28, tied embedding/readout, all bases, anchors,
bridge geometry, rotary tables and polynomial feature definitions remain
fixed. All seven blocks' existing compact parameters and the final norm are
trainable together. There is no new residual adapter or readout matrix.

## Data and teacher reuse

Reuse the original 512 training and 64 development prompts (16 per family on
development: ARC, BoolQ, HellaSwag and WinoGrande). Authenticate identities,
token IDs, targets, four output positions, offsets and train/development
separation. No held-out question enters fitting or checkpoint selection.

Use the group-1 block-29 BF16 native-prefix boundary bank and accepted-teacher
logits. Its capture metadata must identify the exact group-0 recovered teacher.
Map original native teacher logits explicitly by identity. Group-0 native
boundary-30 data is used for replay of the unchanged six-block control.

These prefix caches are valid because embeddings and all preceding native
blocks remain frozen. Every training minibatch recomputes all seven student
blocks on their current inputs, with gradients across every internal bridge.
No teacher hidden state replaces an intermediate trainable student state.
Report cached-prefix execution accurately; this study does not recompute the
entire original prefix from raw tokens during each optimization step.

Before launch, hash all consumed arrays, checkpoints, fit/interface/relation
artifacts and their producing manifests, verify their recorded provenance,
and archive the producing source closure. Preserve parent artifacts unchanged.
Subsequent stages recheck immutable file identities and the recorded critical
hashes. Changed sources or inputs stop execution.

## Controls and validity

Before a scientific update, execute:

1. Matched native BF16 blocks 29–35 plus native norm/readout, starting from the
   same authenticated native-prefix-29 bank. Compare its 64 predictions with
   the original native BF16 reference, expected 48/64. Do not substitute the
   later FP32 native result of 50/64. Execute this control one unpadded case at
   a time, matching the original reference capture. All other scoring paths
   retain microbatch four.
2. Exact accepted six-block packed model from its native boundary-30 bank.
   Require all 64 historical predictions to replay, including 51 correct.
3. The seven-block initializer, with exact inherited weights and geometry
   verified before it receives any optimizer update.

For packed-reference comparisons, require exact predictions and elementwise
choice-logit agreement within `0.03125 + (1/128)*abs(reference)`; this is a
prospectively fixed BF16 tolerance, not the FP32 tolerance from other studies.
Selected-checkpoint fresh verification must strictly reload its state, preserve
all predictions and satisfy the same logit tolerance. Report numerical deltas.

The historical accepted-teacher cache was produced through a differentiable
Torch path; historical selected scoring used packed/fused execution. Before
this experiment, CPU inspection found identical 64 decisions but a maximum
choice-logit difference of 0.25. Record this cache/packed difference separately;
do not require these distinct paths' logits to be bitwise equal or silently
rewrite either reference.

The preserved `20261115_v1` pilot stopped before any optimizer update because
native scoring used padded batches of four against the singleton native
reference. All 64 predictions still matched (48 correct), but three of 192
choice logits differed by 0.5 and exceeded the fixed tolerance. The parent
replay remained exact at 51/64; the seam replay also retained all 64 decisions.
Read-only CPU comparison found the two native-prefix-29 banks identical across
all 18,401,280 development values, excluding changed cached inputs. A separately
reserved diagnostic replayed the native suffix with singleton batches and
reproduced every reference logit exactly (maximum difference zero), recorded in
`20261115_v1/native_singleton_diagnostic.json`. Version 2 corrects only the
native-control execution shape. It preserves the failed attempt and its source
snapshots, retains the original tolerance, and leaves parent/seam scoring,
scientific initialization, training, selection and fresh verification unchanged.

Run a discarded one-update pilot on training examples covering the longest
contexts. Verify finite gradients and actual FP32 optimizer-master changes in
all seven blocks and norm, unchanged fixed buffers/readout, valid allocator
peaks and checkpoint lifecycle. Record stored BF16 changes separately: a valid
small master update may round to the same BF16 value. CPU inspection before
launch found that adding or subtracting the initial learning rate from any of
the parent's 2,048 final norm gains rounds back to its original BF16 value.
That is not evidence of missing gradients. Scientific training restarts from the untouched
seven-block initializer, never the pilot weights.

## Fixed recovery and selection

Reuse the successful suffix's output-only objective:

`L = 0.7 * T^2 * KL(accepted_six_teacher || student) + 0.3 * CE(student, gold)`

with `T=2`, full vocabulary and the four existing next-token positions. Keep
original-native divergence and answer-choice accuracy as separate diagnostics.
No hidden-state auxiliary, gradient-balancing sweep or new data is introduced.

Run 12 epochs, 32 updates per epoch, 384 total updates. Use microbatch 4,
accumulation 4, AdamW learning rate 1.2e-4 with the existing cosine schedule,
zero weight decay, global gradient clipping at one, seed 20861, nonreentrant
block activation checkpointing and FP32 CPU optimizer masters/moments with
32 MiB transfer chunks. Retain the existing 32 GiB CPU-state safety cap.

Evaluate initialization and each epoch through packed execution. Select the
highest development correct count, then lowest full-vocabulary development CE;
retain the earlier checkpoint for exact ties. Initialization remains eligible.
Record every epoch's metrics and individual predictions, all optimizer updates,
fixed-state evidence and selected unpacked/packed checkpoint provenance. If
initialization wins, do not attribute its accuracy to recovery training.

The required number correct is
`max(ceil(native_correct - 0.05*64), ceil(parent_correct - 0.05*64))`.
For the expected controls 48 and 51 this is **48/64**. The five-point allowance
applies to the model comparison, not individually to each compressed block.
Complete the fixed schedule even if an intermediate checkpoint misses quality.
Do not advance to eight blocks within this experiment, regardless of outcome.

## Outcomes and boundaries

Primary: selected seven-block actual development accuracy and its two quality
comparisons. Report initializer versus selected, family counts, paired gains
and losses against native/parent/initializer, teacher agreement, selected epoch,
complete learning curve, and actual tensor storage including all frozen native
prefix/readout tensors. Parameter bytes alone do not establish throughput.

Classify separately: valid expansion within budget; partial recovery below
budget; no recovery below budget; or invalid controls/execution. A passing
initializer establishes tolerated expansion without demonstrated training gain.
A quality miss remains a completed negative experiment. Invalid controls,
nonfinite updates, changed provenance, failed reload, reservation violation or
memory-cap violation stop dependent scientific work and preserve failure data.
Resource retries require a recorded new attempt; there is no automatic retry
of an accuracy miss or retrospective schedule change.

No untouched 512-case confirmation or speed benchmark is part of this study.
The reused 64-case development set cannot establish generalization. Further
expansion, held-out evaluation and deployment timing require separately fixed
comparisons. This experiment does not establish the 38%/1.4x whole-model goal.

## Execution and closeout

Use locked offline Pixi `-e ncm` from the repository root. CPU preparation and
focused tests hide CUDA. Every model process runs through `inference-ctl lease
run`, using the RTX 3060 (`GPU-0370fdb0`) with 8,192 MiB reserved and an allocator
limit no greater than 7,936 MiB. Check the lease and memory during execution;
release each lease on terminal completion. There is no concurrent GPU worker
in this experiment; independent agents prepare and review code on CPU.

Freeze protocol, producer, dependency closure and tests before the reserved
pilot. Require focused CPU numerical, provenance, gradient, selection and
reload tests, then the representative GPU pilot, before scientific training.
Verify the selected packed model in a separate reserved process. Generate the
typed NAL result and measured report, commit source/report, store the exact
registry result, and verify it by independent-process readback. Append the
outcome and learnings to the research journal. Preserve unrelated worktree
changes and all earlier completed experiments.
