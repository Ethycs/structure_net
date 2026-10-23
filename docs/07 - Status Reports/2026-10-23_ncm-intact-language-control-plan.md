# Keep an intact pretrained language path

Date: 2026-10-23. EXPLORATORY `ncm-intact-language-control-v1`.
New evidence: the native five-block AR readout has32/32 exact training outputs
but0/64 source-disjoint development summaries pass. All64 fail readability,
faithfulness and salience. Copying its vocabulary/base weights did not preserve
usable language generation in this trained configuration. This does not isolate
pruning, bridge initialization and limited training data as separate causes.

The actual native-parallel NCM pilot passed systems checks after two updates,
including frozen bases/books, cache consistency and reload. Its planned1,024-
update capacity run is **deferred, not executed**, because its borrowed language
component just failed all unseen cases. This is an explicit amendment based on
new external evidence, not a preregistered successful early stop or a rescue of
the five-block model. Preserve the unexecuted protocol and passed pilot unchanged.

Test an intact pretrained Llama-3.2-1B-Instruct as a candidate language component.
The parent encoder stays the original frozen3B. This control runs the1B model
alone; no NCM/coupler is yet attached and a pass cannot establish NCM success.
Keep all16 native blocks, native norm, embeddings and tied word head unchanged.
No training, layer removal, prompt search or output repair. A later source/
concept correction path must start as an exact no-op around this intact native
path, instead of replacing its input with an untrained representation.

Use cached revision9213176726f574b556790deb65791e0c5aa438b6. Before loading,
verify every cached file against official pinned HF metadata: LFS SHA256 for
weights, Git blob identity for regular files. Preserve cache unchanged; link
verified files into a fresh workspace input root. Record vocabulary/added-token
identity and native prompt-ID agreement with3B across64cases, without assuming
they match. This compatibility record does not measure language quality.

Use the unchanged64 development sources, native-parent source-faithfulness
SYSTEM prompt,768-token input ceiling without truncation, greedy seed42,
max128 native output tokens and the same128-cl100k storage cap. FP16/SDPA,
physicalGPU2, batch2; require3GB free before startup. This differs in model,
precision and hardware from the five-block BF16/GPU0 control, so no matched
speed or isolated causal effect of layer count is claimed.

Four-case real-memory pilot must repeat exact greedy outputs and serialization,
retain tied weight identity and full model hash. Main64 requires matching
producer/input hashes and exact first-four agreement. Stop on OOM, invalid
scores, cache/input/weight drift or inconsistent reload. Preserve every output,
including oversize or unfinished text; never clip or exclude a task case.

Reuse the native AR transfer's grading implementation and first-judgment cache
through a process-local wrapper, hashing both implementations. First four fixed
outputs have a four-valid-judgment systems pilot; main reuses all pilot positives,
negatives and exhausted invalids. Qualified Terra, four workers, unchanged
rubric and three-attempt transport/schema cap; no semantic retry. Primary Sonnet
only if valid whole-summary Terra>=61/64, in a fresh stage after09:50UTC.
No teacher-generation overlap with judging.

Necessary language-component gate: all64 valid and at least61/64 readable.
Report full faithful/salient/readable and whole-summary numerators, even if only
the language gate passes. This is not a95%summary-task pass unless the full
criterion passes, and is never NCM success. A separate NCM architecture/training
protocol must preserve the intact base path, qualify teacher data, and test
actual source-disjoint final summaries before any speed claim.

Roots `20261023_intact_decoder_inputs_v1/verified` and
`20261023_intact_decoder_{pilot,outputs,judge_pilot,grading,primary}_v1`.
All failures, original source/input splits and final256sealed cases remain.
