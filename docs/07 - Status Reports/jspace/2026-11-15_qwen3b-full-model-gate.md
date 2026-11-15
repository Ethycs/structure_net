# Full-model validation of progressive Qwen2.5-3B compression

Study **JSPACE-QWEN3B-FULLMODEL-GATE-01**, specified before full-model held-out
evaluation. This is a separate gate after progressive recovery, not permission
to label a partially compressed suffix as a completed model.

## Admission and frozen inputs

Require all 29 one-block additions from the accepted seven-block seed through
cut zero to have completed the progressive protocol. Authenticate every
selected checkpoint, immediate-parent connection, both five-percentage-point
development gates, and fresh packed verification. Do not evaluate a failed
cut-zero model or select among candidates on the held-out outcomes.

Prospective continuation amendment, before any confirmation was opened:
the original campaign stopped at ten blocks, scoring 47/64 against native
48/64 and its immediate parent 52/64. It passed the user's original-native
budget but failed our additional parent-relative gate. Preserve that original
stop. The separate `jspace.qwen3b-native-budget.v1` campaign authenticates its
ten-block seed and records the policy change explicitly, then adds blocks
25 through zero under the fixed original-native minimum of 45/64. For this
campaign, admit its authenticated seed receipt and all 26 new transitions;
parent-relative drops remain diagnostics. Do not describe the rejected
historical transition as having passed the original dual-gate protocol.
The held-out accuracy, size and speed criteria below remain unchanged.

Prepare a new immutable root at
`data/experiments/jspace_3b_fullmodel/20261115_v1` only after admission. Pin
the final selected checkpoint, all fitted geometry/relation artifacts, native
Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`,
the final evaluator source closure, locked environment and the task manifest.
No 8B model is part of this comparison.

The native-budget continuation uses a separate final root,
`data/experiments/jspace_3b_fullmodel/20261115_v2`. Its final plan retains the
campaign schema and original seed-admission receipt alongside the new chain.

The historical parallel experiment's 512 evaluation examples were subsequently
used by the quadratic-attention experiment. They are not an untouched test.
Use the separate 512 `confirmation` examples in
`data/experiments/jspace_accuracy_recovery_data/20261114_v1`, after authenticating that
producer's manifest and checking their identities against all recorded
training, calibration, development and previously evaluated cases. This bank
was recorded as frozen and unscored. If any scoring consumer or overlap is
found before launch, stop preparation and specify a fresh bank prospectively.
The actual frozen list is authoritative; do not resample after outcomes.

Use 128 examples per family (ARC, BoolQ, HellaSwag and WinoGrande), native
tokenization, label logits and exact original evaluation prompts. Report
correct counts, percentage accuracy, paired gains/losses and family results.
No semantic judge replaces these task labels. Do not tune on confirmation.

## Deployable model and correctness

Assemble the fully compressed model from the accepted cut-zero stack, final
norm and native tied embedding/readout. The Hugging Face wrapper retains the
original 36 cache indices, tokenizer and generation configuration. Assert
that no native decoder block or meta tensor remains and that the embedding
and readout share storage. All learned weights and forward computation are
BF16; fixed geometry retains the existing deployed representation.

First verify exact agreement of raw-token embedding lookup with the original
boundary-zero cache, with no learned prefix. Export the packed deployment,
reload it in a fresh process and compare selected-development records with
the accepted receipt before accessing confirmation. Recompute all candidate
confirmation predictions after strict reload and require exact agreement.

On prespecified development prompts, compare full-prefill logits with
token-by-token dynamic-cache and static-cache logits. Require finite logits
and BF16 agreement within `0.03125 + abs(reference)/128`; report maxima.
Graph and ungraphed static execution must produce exactly the same greedy
tokens. Static cache retains native 128-channel storage where required,
although compact attention reads 64 channels; report cache/peak memory
separately from model tensor storage.

## Accuracy, size and performance comparisons

The primary accuracy reference is native BF16 eager attention, matching the
training teacher convention. Also score native BF16 SDPA, since backend
rounding can change decisions. Aggregate accuracy loss must be at most five
percentage points: at most 25 net fewer correct answers out of 512. Per-family
results and the SDPA accuracy comparison are diagnostic, not additional gates.

Count actual deployed tensor storage including fixed bases, polynomial
coefficients, bridges, positions, norm and the tied embedding/readout once.
Require at least 38% reduction relative to the same native BF16 model. Do not
count activations or KV cache as model compression or report optimizer state
as deployment size.

Measure native eager, optimized native SDPA and packed compact execution on
the same RTX 3060, BF16 precision, batch one, eight frozen development prompts
and exactly 32 generated tokens per prompt. Use five prespecified repetitions
after warmup. Exclude load/compilation/graph setup from warm inference time,
but report them separately where measured. End-to-end generation time includes
prefill, cache reset/copy, full-vocabulary readout and the remaining 31 decode
steps. Report prefill separately; label decode-only timing separately if used.
Do not substitute cached-hidden-state or training throughput for this metric.

Run optimized SDPA timing before and after compact timing to expose temporal
drift. Its two measurements use identical settings and benchmark inputs.
Require at least 1.4× end-to-end generation speedup against their pooled
baseline. Report the matched-eager ratio too. Capture failures or numerical
invalidity stop speed claims; do not silently fall back to a slower baseline.

## Execution and reporting

CPU preparation, validation and aggregation hide CUDA and use locked Pixi.
Each GPU phase obtains its own inference-controller lease on RTX 3060:
11264 MiB reserved, 11008 MiB allocator cap, finite duration, live-lease guards.
Models run sequentially to avoid combined residency. Preserve all manifests,
raw task records, timings, configuration, source hashes, failed phases and
resource peaks. No new phase may mutate the progressive training evidence.

Report the three gates separately: held-out accuracy, model storage and warm
inference throughput. Passing development recovery alone is not passing this
gate. A size or speed miss remains a useful measured result. Any further
optimization or recovery after viewing confirmation starts a new prospective
attempt and needs new untouched confirmation data for an independent claim.
