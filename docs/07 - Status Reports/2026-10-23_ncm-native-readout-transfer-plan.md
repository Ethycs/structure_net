# Check native readout transfer before further NCM claims

Date: 2026-10-23. EXPLORATORY `ncm-native-readout-transfer-v1`.
Known: the fixed native AR head completed1,024 updates and reproduced32/32
training targets,995/995 tokens; source-shuffled recovery was25/995. Native
base weights stayed frozen. This is memorization, not held-out task accuracy.
The native concept fit and CPU parallel-interface checks also passed; no native
coarse-to-fine student has been trained yet. Sonnet remains unavailable before
2026-10-23T09:50Z. Final task256 cases stay sealed.

Evaluate the final fixed AR checkpoint on the same64 existing source-disjoint
development passages. No additional updates, prompt/checkpoint/seed selection,
output rewriting or per-case exclusion. Keep the native source prompt, full
per-token features1,14,28, five frozen native blocks0,7,14,21,27, adapters,
tokenizer, BF16/SDPA, batch2, greedy BOS/EOS rollout and128-native-token cap.
Apply the unchanged whole-summary rubric and128-cl100k storage cap to actual
outputs, with all failures retained. This baseline is not NCM success, even
if it passes61/64. Report readable, faithful and salient counts separately.

The original capacity runner retained free outputs but only scalar shuffled
metrics. Preserve it unchanged. A new no-training replay in the transfer pilot
must reproduce all32 original free token sequences and both free/shuffled
metric dictionaries, and save the previously missing raw shuffled outputs.
If replay differs, preserve the discrepancy; do not overwrite the original
result or admit a downstream transfer claim from that checkpoint. This adds
audit evidence without changing the original capacity gate.

Pilot: after the capacity worker is observed terminal, verify all original
source/checkpoint/feature/positive-label identities. Extract the first four dev
cases using the unchanged full original Llama body. Record body hash before
and after, then retain the five readout blocks and restore the final adapter
checkpoint. Run the32-case capacity replay, four-case greedy repeat, frozen
weight and exact output save/reload checks. Require7.5GB free GPU0 memory before
the full-body extraction and stop on numerical, memory, identity or replay
failure. The64-case generation requires matching passed terminal pilot,
identical producer/input identities and exact first-four output agreement.

Grade the first four fixed full-run outputs with Terra as a systems pilot,
requiring four valid decisions, not four semantic passes. Seed first exact
packet judgments from the completed native-parent cache. Main grading reuses
all pilot outcomes, including negatives, then completes64 cases with four
qualified workers. Preserve the first valid/negative or exhausted invalid
decision for each exact packet/role; existing transport/schema attempt cap3.
No concurrent teacher requests. If valid Terra passes at least61/64, independent
Sonnet grading is required in a separate fresh stage after the documented
provider reset. Otherwise omit that stage and classify the development gate
as failed. Invalid decisions retain bounds and cannot pass the gate.

Roots `20261023_native_readout_transfer_{pilot,outputs,judge_pilot,grading,primary}_v1`.
Freeze exact source, runtime, inputs and checkpoint hashes. Preserve all raw
source features, training replay outputs, development generations, decisions
and terminal receipts; finalize/register conservatively. Passing a small
native-language baseline or training capacity cannot complete the95%NCM goal.
