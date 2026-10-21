# Correct the NCM tokenizer validation bottleneck

Exploratory hypothesis `llama32-3b-ncm-host-io-v1`, before corrected timings.
Known: all six literal-copy conditions completed1800 exact native-output cells,
76.52–78.85 aggregate tokens/s; first batch9 pilot passed27 native-ID checks.
Rank256 factorization reduced model state to533987792 bytes but field checks
passed only7/64 before repair and39/64 after576 local updates; promotion fails.
Full source reviews and the bounded diagnostic remain required.

CPU diagnosis:100 calls to len(GPT2TokenizerFast) took1.1021s on this host.
Existing decode_ids calls len(tokenizer) for each actual output token. The
fixed vocabulary has50258 entries after adding PAD. Repeated backend vocabulary
construction is a host bottleneck, so the previous flat replica curve cannot
establish that GPU compute itself is saturated. Preserve all completed timings.

Interrupt the uncompleted legacy batch1 run and its coordinator, retaining the
passed batch9 pilot, partial rows/logs and explicit interruption receipts. Do
not silently substitute corrected data into its v1 roots. Interrupt the
uncompleted CPU replica audit (no aggregate had been written), then rerun its
unchanged output validation using the fixed vocabulary view; this changes no
measured cells or decisions. These are intentional engineering interruptions,
not scientific threshold retries. The legacy three-condition batching primary
campaign is unexecuted/incomplete; it is superseded, not passed.

Intervention: an inference-only tokenizer wrapper takes len(tokenizer) once
after all special tokens are added; subsequent length queries return that
fixed value. Delegate unchanged decode and all other tokenizer functions. Do
not mutate vocabulary after wrapping. Keep every original input-ID validation,
EOS/PAD rule and text decoder behavior. No model/precision/arithmetic changes.

CPU tests require identical valid/invalid-ID, EOS/PAD and canvas behavior and
one underlying length lookup. Separate CUDA pilot on eight development sources
requires all48 original/fast native IDs and complete texts identical, plus
batch9 reset/parity on nine development sources. Record request timings broken
into H2D, CUDA-event graph execution, D2H/native-ID validation/text decode and
full wall time, with slow/fast wrappers on identical outputs. Systems numbers
are diagnostic, not full quality evidence. Stop on any parity/finite failure.

Then compare literal replicas1,8,9 and shared batches1,9, fresh process per
condition, after both prior GPU producers terminate. Use the same complete
100 sources, three repetitions, one excluded development warmup wave, original
GraphRuntime/ReplicaPool and BatchRuntime arithmetic. Count actual Llama text
tokens; dummy static batch-tail computations stay timed but never count as
requests/output tokens. Include CPU IDs to decoded CPU text, exclude loading,
capture, counting, evidence writes. Preserve all1500 fresh request cells and
require exact parity to original saved outputs. Memory budgets remain separate
model-state6425499904, allocated6667318272, reserved6794772480 bytes.

Run a fresh isolated original Llama3B teacher control on all100 same sources,
three repeats and excluded per-case warmups, using its unchanged parent
graph/FlashAttention/current-prompt lookup runtime. Require native text/token
parity to its parent to inherit the100/100 source review; otherwise review all
new outputs. Primary engineering success: an eligible optimized NCM condition
retains all outputs, and is >=2x fresh original Llama common token throughput
and request throughput, using total actual output tokens / total timed wall
seconds for all conditions. Also report each repetition, per-copy effective
rate and wave latency. This is aggregate throughput, not a single-request
latency gain for large batches. Original Llama batching remains untested.

Use new `data/experiments/ncm_host_io/20261021_*`, fingerprint all imported
runtime/helper/official source and checkpoints, separate pilot and campaign,
strict audit/aggregate and source snapshots. Record all original interruptions,
completed legacy controls, new gates, schema and source fingerprints. Report,
conservatively register/read back and commit. All model/quality boundaries
remain: one trained model, used synthetic bounded facts,512/48 token shape,
UNDERPOWERED and confirmed:false; no production, mobile or general claim.
