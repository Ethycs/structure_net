# Phi-4 last-block compression and recovery pilot

Date: 2026-11-15. Study: `JSPACE-PHI4-BLOCK39-PILOT-20261115`.
Status: prospective; native capture, fitting and GPU recovery have not run at
the time this protocol is written. The eventual objective is a recovered,
structurally compressed whole Phi model. This first study treats only block 39
of 40; it cannot establish whole-model recovery or speed.

## Source and numerical reference

Use the existing local `/data/models/Phi-4-GPTQ-int4` checkpoint, with exact
weight-shard, config, tokenizer and chat-template hashes before execution.
This is a GPTQ-v1 int4 source, not the original unquantized Microsoft weights.
The bounded loader decodes group 128 weights with no act-order, including the
actual signed stored scales. Products round to the stored FP16 scale precision
before the declared BF16 execution cast. Retain the scale/zero/packing reference
and CPU equivalence tests in the producing archive.

The primary native reference is this decoded-source model executing BF16 with
eager attention. The compressed candidate uses the same source and execution
precision. FP32 full-rank controls and algebraic tests qualify implementation;
they do not turn a quantized source into an unquantized teacher. Keep native
embedding and untied output head distinct and include both in storage counts.
BF16 decoded or compact weights must not be reported as a file-size reduction
against the existing int4 files. Report source-file bytes, decoded precision,
fixed geometry and eventual matched-precision model bytes separately.

## Data and supervision

Retokenize the authenticated existing broad-recovery TRAIN/DEV raw prompts
using the actual local Phi chat template. Do not reinterpret Qwen token IDs.
Retain identities, semantic/source group IDs, labels and existing option
permutations. The full available population is 4,608 TRAIN and 256 DEV; CPU
retokenization found all 4,864 eligible without truncation. A frozen pilot takes
the first 32 eligible TRAIN and 16 DEV cases per family in existing source order:
ARC-Challenge, BoolQ, HellaSwag and WinoGrande. Thus 128 TRAIN and 64 DEV are used.
The source contains two-, three- and four-choice questions; score only the
actual choices. The four answer token IDs are 32, 33, 34, 35.

Prompt cap 384, complete sequence cap 416; reject excess length and record it.
The observed pilot contains 24,935 prompt tokens, maximum 331. No sealed audit or
final-confirmation content enters this preparation. These reused public task
subsets are a small multiple-choice pilot, not an independent broad benchmark.

`pilot_items.json` supplies prompt-only tokens, offsets and four diagnostic
positions: three lexical next-token locations within user content and the
answer location. `pilot_responses.jsonl` has identical case order and independently
verified assistant-prefix tokens. It contains the gold answer letter followed
by the actual template end token 100265, which the model generation config
declares as EOS. Both observed response tokens receive dense response loss.
No explanation, teacher-generated continuation or synthetic EOS is invented.
This exercises the response-boundary/masking standard, but such two-token MC
responses cannot qualify general generation recovery.

## Native phase and compression

Run the original decoded model one layer at a time on the reserved RTX3060.
Cache prompt-only and complete-response trajectories separately when required
by the producer; preserve their actual lengths, causal masks and token offsets.
No teacher state replaces a trainable student layer. The native prefix is
blocks 0–38; store its pre-block 39 boundary, original post-block 39 hidden state
before final RMSNorm, and required native output targets. The final native norm
and untied head define the reference. Bind every bank to the exact sequence
kind, source identity and decoder fingerprint. Prompt-only DEV scoring defines
the original native 64-case control; no Qwen score is assumed.

Fit block 39 using TRAIN only: grouped full-vocabulary Jacobian probes on the
actual fused Phi QKV output, residual rank 2,560, head dimension 128→64 while
preserving 40 query/10 KV heads and the original attention scale. Use the
established functional basis/head/value procedure and softmax attention.
Fit 24 polynomial anchors and replace 1,664 of 17,920 MLP channels, preserving the
existing relation-fit numerical admission rather than automatically relaxing
its cutoff. Save the fitted basis under `fit/basis/block_39`. All geometry and
relation choices are frozen before recovery outcomes.

## Recovery, selection and evidence

The budget is three early epochs, microbatch 1, accumulation 4 (effective batch
4), constant learning rate `2e-5`, zero weight decay, gradient clip 1 and seed
20861. Each epoch shuffles TRAIN deterministically with seed `20861 + epoch`.
The 128 cases give 32 updates per epoch and 96 total. Train only the compiled
block's learned BF16 parameters. Fixed geometry, native final RMSNorm and
untied output head remain frozen. Use the bounded streaming CPU AdamW with
FP32 master weights/moments, 32 MiB chunks and a 4 GiB CPU-state limit. A
discarded update pilot must establish gradients, master changes, frozen-state
parity and memory feasibility before the scientific initializer is restored.

The objective is feature coefficient 1 plus `0.7*T²*KL + 0.3*CE`, T=2. Compute
full-vocabulary output terms at both actual response-predicting positions
`P-1,P`, where `P` is prompt length. Process one vocabulary row at a time with
exact hidden-gradient accumulation. Teacher weights and cached targets remain
fixed. The feature term compares actual post-block compact coordinates with
`block.encode(native_post39_before_norm)`: the BF16 coordinate representation
of the native hidden state projected through the fixed FP32 basis. There is
**no mean subtraction** in this coordinate convention; the affine anchor is
the omitted-space component at reconstruction. Use FP32 mean squared error
over every valid full-sequence token and coordinate, including the gold letter
and observed EOS. Normalize by the fixed, equal-case TRAIN mean squared target
coordinate energy (floor `1e-8`; reject zero/nonfinite energy). DEV does not fit
that normalization. Original prompt-only DEV forward determines task accuracy;
full-sequence response loss is a separate diagnostic.

Save actual initializer and epoch 1/2/3 checkpoints and their DEV records.
Choose highest DEV correct count, then earliest epoch, including initializer;
CE is reported as a secondary measure and does not override the tie rule.
The exploratory quality screen allows at most five percentage points loss
against this study's original native 64: no more than three net lost answers.
Report all family scores and paired gains/losses; there is no hidden rolling
parent or per-family veto. Fresh strict reload must reproduce the selected
checkpoint's predictions and numerical metrics under the declared precision
controls before any later compression stage is admitted.

Systems success, small DEV quality and confirmed recovery are separate. No
generalization, generation, all 40-block completion or performance claim follows
from a passed one-block pilot. A later whole-model program needs a separately
frozen progression, response corpus, independent task/generation assessment,
export parity, actual storage and isolated prefill/decode measurements.

## Reservations and append-only execution

Preparation is CPU-only with `CUDA_VISIBLE_DEVICES=''`. Freeze code, tests,
this protocol, inputs and archives before native GPU work. Use one controller
allocation value `GPU-0370fdb0=11264` MiB, then launch worker Python under the
controller-created visibility; the single card is logical `cuda:0`. Cap the
allocator at 11,008 MiB and check the live lease before CUDA access and during
streaming. The independent Qwen all36 experiment on RTX2060 is untouched.
Controller `tainted` state alone does not invalidate an otherwise ready lease.

The initial thin launcher runs only native capture, with a 90-minute lease,
one immutable command/log/exit receipt and no automatic retry. Preserve failed
or partial worker directories; never relaunch an ambiguous attempt or change
batch/precision/thresholds silently. A zero process exit is not a scientific
receipt: authenticate the completed native manifest separately. No GPU work
is authorized for the drafting/review agents.
