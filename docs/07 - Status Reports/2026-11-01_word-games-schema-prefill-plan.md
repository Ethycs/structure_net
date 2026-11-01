# WORD_GAMES public schema prefill

Exploratory hypothesis `word-games-schema-prefill-v1`: restoring precomputed
public sender and receiver instruction prefixes reduces whole-request latency
on the deployed three-GPU model without changing the checked task behavior.
This is a new systems study, not a rescue of the earlier 70 tok/s hypothesis.

Use the deployed `free_whole1024` Q4_K_M model, Q8 KV, Flash Attention,
sampling-acquisition-patched b11960 runtime, context 8192, temperature 0, seed
73, batch 2048/microbatch 1024, and unchanged GPU placement. No new weights,
speculation, CPU expert fallback, context checkpoints or precision changes.

Snapshot only the fixed public sender/receiver contracts and their static
delimiters, inside the model's user chat framing. Never save event data,
retrieval results, authorized packets, private workspace or generated output.
Use native sequence save/restore, including recurrent and attention state.
Build snapshots anew on each model load, with zero generated tokens. Trim the
last prefix token to avoid boundary merges and check the entire token prefix
against every matching full prompt. A mismatch runs uncached; restoration
failure errors out. Restore before EVERY matching request; unknown prompts
run with reuse disabled. Keep one serialized native slot. Snapshot files are
local per-load runtime artifacts and are removed on unload.

First run a systems-only pilot containing sender, receiver, repeated sender
and receiver calls separated by unrelated input. Require valid JSON, actual
cached-token counts equal to the saved public prefix, no generated prefix
tokens, no memory guard failure, and no private text in snapshot metadata.
Do not interpret this pilot as quality evidence.

Then replay the frozen 36-request corpus in order control, cached, cached,
control (two timing repetitions each, not independent training seeds). Warm
the runtime before each pass. Selector requests remain uncached. Time the
entire Triton inference request, including formatting, tokenization, restore,
prefill and generation. Report pooled output tok/s, total time, prefill and
restore time, cache hits/tokens, snapshot sizes and startup construction cost.
Report startup amortization separately; do not omit restore overhead from
the primary rate. Compare all parsed outputs, receiver decision/slots and
selector validity against contemporaneous control and the previous baseline.
Retain failures and output differences. No retries for missed thresholds.

Stop on HTTP errors, malformed/truncated outputs, changed source/input/model
identity, incorrect cache lengths, GPU free memory below 384 MiB, or model
failure. Restore the original prepared Triton repository and load it in a
finally block. Snapshot/hash source, runtime manifest, inputs, plan and all
responses in a new dated artifact root. Keep old Qwen3-8B/OCR services stopped
as in the current deployment; do not overwrite their configurations.

A measured speed improvement requires cached pooled tok/s above control.
The 70 tok/s target is a separate endpoint. Automatic serving adoption also
requires all 72 parsed cached outputs to match the corresponding control
and all existing receiver/selector checks to pass. Otherwise leave baseline
serving, inspect differences and report the conservative result. Exact output
agreement is a regression check, not population semantic accuracy. No new
semantic judge is included; existing baseline limitations remain.

Use Pixi from repository root, focused prefix-isolation/lifecycle tests, real
CUDA validation, append-only evidence, a measured report, conservative registry
read-back and a durable source commit. Initial evidence roots:
`data/experiments/word_games/20261101_schema_prefill_pilot_v1` and
`data/experiments/word_games/20261101_schema_prefill_full_v1`.
