# Packed response-checkpoint evaluation

Status: preparing a separate packed inference evaluation of both active
Qwen2.5-3B-Instruct recovery arms. No packed quality or speed result is claimed
by this initial entry.

The user requested packing before testing. The new
[protocol](../../07%20-%20Status%20Reports/jspace/2026-11-16_qwen3b-packed-response-evaluation.md)
requires export, reload, and actual packed response/task evaluation. Existing
complete-response training on the3060 and2060SUPER continues with its frozen
unpacked producer. Packing strips training state and folds inference weights;
calling it on the live trainable model would invalidate that training path.

The spare6GB2060 is the intended evaluation device, with a separate5632MiB
reservation and5376MiB allocator cap. Its Turing architecture uses the packed
Torch executor; Ampere-only BF16 Triton kernels remain disabled. A future
3060 fused-kernel benchmark needs separate scheduling and cannot be inferred
from these results.

Each evaluation binds a durable saved checkpoint to the original campaign,
native teacher cache, exported state and frozen producer. It evaluates full
response DEV128, MC DEV256, six canaries, and16 generation cases spanning eight
families, including blinded semantic judgments and executable/numerical checks.
These are exploratory DEV screens. The old reserved final512 is not accessed;
eventual confirmation still requires a new untouched set.

The CPU observer waits for the training writer's post-save event before
consuming a checkpoint: the writer creates its checkpoint file before it
finishes writing, so file existence alone is insufficient. Scheduled snapshots
are updates0,46,90,152,228. The observer requests one evaluation lease at a time
and never changes the training workers or retries failed cells silently.

Prior packed/unpacked numerical discrepancies remain recorded. The relevant
question here is actual packed task quality within the user's5pp budget,
alongside the existing response and generation gates. Exact packed-state
export/reload and output replay remain implementation checks.

Runtime measurements must state whether they cover the full model, cached
teacher-forced suffix, or cache-free prefix-recomputed generation. Concurrent
training may add CPU/PCIe contention. No matched speedup, deployment readiness,
full-model size reduction, or recovery claim follows from preparing this path.
