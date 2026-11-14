# Qwen compression: evidence, lessons and decisions

Logging audit and synthesis, 2026-11-14. This note consolidates completed
development evidence and explains the decisions behind the running parallel
campaign. It does not amend any frozen experiment protocol.

## What we have learned

| Finding | Evidence | Decision and limit |
|---|---|---|
| Basis construction mattered before distillation. | With only block0 changed, the old half-rank/full-head basis scored 27/64. Training-only input/update enrichment at the same rank scored 46/64 before training. Separate training diagnostics reduced lost update energy from 43.64% to 8.32%. | Check both input sensitivity and actual update directions. Attribute 27→46 to the basis intervention, not to distillation. This does not prove every layer has the same repair. [Diagnostic report](2026-11-14_qwen-accuracy-recovery-diagnostics.md). |
| One compressed block can recover useful task accuracy. | Isolated block0 distillation improved 46/64→48/64, versus native 50/64. All 12 epochs and 384 updates completed; a fresh process reproduced packed task logits exactly. | Retain the tested one-block recipe and qualify task outputs after compression. This establishes a local result, not that training order alone caused every earlier failure. [Registered result](2026-11-14_qwen-single-layer-recovery-result.md). |
| Falling training loss is not enough. | Earlier joint recovery reduced loss while development answers fell 29→20; much of the loss improvement increased answer-letter probability without better choices. In the two-block follow-up, initialization scored 49/64 and every trained epoch scored 46–48/64. | Select by task accuracy first, retain the initializer as a candidate, and distinguish formatting/calibration from semantic performance. The two-block result is not a further distillation gain. [Joint diagnosis](2026-11-14_qwen-accuracy-recovery-diagnostics.md); [two-block result](../../../data/experiments/jspace_next_layer_distill/20261114_block01_v1/train/result.json). |
| Remaining feature error has a substantial fixed-subspace component. | For selected block0, retained-subspace error fell 85.44%. The outside-subspace term is about 90.16% of remaining normalized error. Predictions match the projected-teacher control on all 64 cases, at 48 correct. | Use projection controls to distinguish fitting error from representational restrictions. The closest Euclidean projection is **not** a task-accuracy ceiling. [Result and oracle accounting](2026-11-14_qwen-single-layer-recovery-result.md). |
| Successful isolated blocks still need an assembly test. | The enriched first block reached 46/64 before recovery, while an untrained all-36 enriched assembly scored 24/64. The accepted two-block assembly now scores 49/64. | Fit independent native-input/output pairs concurrently, then test the assembled model. Repair interactions using actual student-prefix inputs where needed. Apply the five-percentage-point allowance once to the whole model. [Parallel protocol](../../07%20-%20Status%20Reports/jspace/2026-11-14_qwen-parallel-single-layer-recovery.md). |
| Precision and execution context must match the comparison. | Training pairs are original BF16-forward teacher states. Independent task checks use a separate FP32 native-prefix cache, which reproduces native 50/64. All three GPU pilots passed native prediction and logit checks. | Keep the training/evaluation distribution difference explicit, especially for deeper layers. The matched FP32 cache prevents accidentally treating a rounded cached prefix as the exact native FP32 context. [Native cache result](../../../data/experiments/jspace_native_dev_cache/20261114_v1/cache/result.json). |
| A short memory pilot did not predict the long-run allocator peak. | The first block0 attempt stopped when reserved CUDA memory exceeded its 4 GiB lease, despite a much smaller pilot peak. The identical scientific retry completed with a larger lease and an allocator cap below it. | Record this as a resource failure, preserve both attempts, and cap the allocator beneath the live reservation. Do not classify it as an accuracy failure. [Resource lineage](2026-11-14_qwen-single-layer-recovery-result.md). |
| Local storage savings are not whole-model savings or speed measurements. | The recovered block's packed state is 47.24% smaller than the native block at matched FP32, including geometry. With one block changed, whole-model FP32 savings are 1.18%. Replacing one native BF16 block with this FP32 block increases source-precision bytes by 0.138%. | Report matched precision, all saved buffers, the full model denominator and actual runtime separately. The current recipe has no measured whole-model speedup. [Storage accounting](2026-11-14_qwen-single-layer-recovery-result.md). |

The current recovered architecture retains softmax, native 128-dimensional
heads and the full residual carry. It is not yet the half-head quadratic
attention recipe. These measurements concern functional subspaces and recovered
execution; causal claims about polynomial relations or a reusable cross-task
basis library remain separate research questions.

## What is recorded, and where

| Evidence | Logging status |
|---|---|
| First-block diagnostic interventions | Completed aggregate, report, typed record and registry read-back. Registry result `28984bec138dac08`. |
| First-block distillation | Completed artifacts, report, typed record and registry read-back. Registry result `c756b51d60e74e9b`, 48/64, selected epoch7, `confirmed:false`. The exact registry entry was read again during this audit and matched. |
| Two-block follow-up | Completed training and fresh-verification artifacts, with hashes in the research log. Selected epoch0 scored 49/64. It is **not separately registered**; the existing single-layer registry result remains 48/64. |
| Parallel blocks2..35 | Frozen protocol and producers, shared native cache, three passed concurrent pilots, live per-worker events, per-epoch predictions, checkpoint hashes and lease receipts. Campaign still running at this audit; no final aggregate or registry result is claimed. |
| Failures and changes of direction | Preserved artifacts and append-only [research log](../../07%20-%20Status%20Reports/jspace/research-log.jsonl), including the all-block accuracy failure, failed joint recovery and memory-limit interruption. |

The latest two research-log entries omitted the normal schema, stable ID and
recording timestamp fields. Timestamped metadata supplements now identify their
original contents by canonical SHA-256. The original entries remain unchanged;
supplement timestamps denote this audit, not a reconstructed event time.

The [audit artifact](../../../data/experiments/jspace_logging_audit/20261114_v1/audit.json)
pins the source reports, completed result and verification files, registry
receipts, and frozen parallel plan/pilot gate. The current
[run record](../../07%20-%20Status%20Reports/jspace/2026-11-14_parallel-distillation-running.md)
points to live progress. Historical reports retain their original as-of status.

## Boundaries and next required evidence

All recovery numbers above use 64 repeatedly inspected development questions.
The fresh 512-case confirmation set remains unscored. Neither a development
pass nor a registry entry establishes generalization or semantic equivalence.

At parallel-campaign closeout, record every selected block, initializer wins,
quality misses and execution failures; authenticate fresh verification; then
measure the assembled model. Publish and register the stabilized aggregate with
its actual scope. Only after selecting the final assembly should we open fresh
confirmation and benchmark matched whole-model storage and inference speed.

Separate input and output bases, a wider basis for a failing block, and
task-sensitive recovery losses remain candidate follow-ups. They have not been
tested by this parallel campaign and are not recorded as established repairs.
