# Packed response-checkpoint evaluation

Status: launched a separate packed inference evaluation of both active
Qwen2.5-3B-Instruct recovery arms. No completed packed quality or speed result
is claimed by this launch entry.

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

## Launch evidence

Producer commit `d8e7653`;19 focused checks passed across the adapter,
evaluation, and observer. An extra complete-response integration fixture found
an export digest bug: shared empty buffers were deduplicated by tensor object
identity before export but independently reconstructed after reload. Hashing
every registered name, as the serialized state does, fixed the inventory
mismatch without changing arithmetic. Both polynomial and non-polynomial
fixtures now replay exactly. Independent static review found no remaining
launch blocker; actual GPU execution remains the next check.

Observer started2026-11-16T08:20:36Z, PID692439, under locked/offline Pixi.
Artifact root:
`data/experiments/jspace_3b_packed_response_evaluation/20261116_v1`.
It observes scheduled durable checkpoints from both parent campaigns and
requests leases serially. The first cell is progressive16 update0, resident
prefix, fingerprint
`d06cc157e03c2f13d1d83456c43adc17e9b22b5c6a82482bb335212386183c38`.
The admitted saved checkpoint digest is
`2914a25998476a46fba1960332712320c4dcdfd555742a202dbf6465b6d0eb90`.
GPU worker PID695203 received lease `83921e8189cc`,5632MiB on the plain2060,
at08:21:33Z. Existing3060/SUPER training workers and leases were preserved.

The source archive pins108 local files. The sidecar uses one semantic judge
worker: the parent initializers had incomplete service responses (progressive
12/16 valid, all3611/16 valid), with HTTP429 and connection failures. Successful
responses matched the expected judge/model/schema. Missing semantic evidence
is not a measured zero or an accuracy drop. No past receipt or parent producer
was altered.

While packed construction ran, the parent progressive recovery reached its
first new checkpoint, update46. Its **unpacked** MC DEV fell from186/256
(72.66%) at initialization to164/256(64.06%): an8.59pp loss. HellaSwag fell
47to28 correct, ARC51to47, BoolQ53to51, while WinoGrande rose35to38. This fails
the existing retention gate. Full-response DEV NLL improved6.96739to4.47751,
and teacher KL5.98475to3.49189; these improvements do not rescue lost task
accuracy. This is a post-training regression on the same exposed MC benchmark,
not a packed-model result. The original initializer remains preserved and the
prospectively bounded recovery run continues; no failed candidate is promoted.

## First actual packed systems result

The progressive16 update0 cell completed packing/export and fresh structural
reload on the reserved2060. Packed-state SHA256:
`120e38f3760eb5dc63844e4395d6162429cc8edbace3e4331a35a19946f5b9f8`.
The2,046,607,289-byte snapshot includes the compact suffix and one original
embedding/readout copy; the20-layer native prefix remains an additional pinned
dependency. This byte count is not the total deployed model size.

Reload replay was exactly equal for all151,936 logits at66,67,68-token prefixes,
including identical greedy choices. The native prefix fit resident within the
5376MiB allocator cap; CPU-prefix fallback was not used. Warm full-prefix
next-logit latency was205.994ms at66tokens and206.625ms at67tokens, two warmups
and five measured repeats, including prefix/readout/guard work. These are
cache-free Torch measurements on Turing under concurrent training, not cached
decode throughput or a matched speedup.

Packed full-response DEV evaluation completed; packed MC DEV began at
2026-11-16T08:26:57Z. Generation, semantic grading and the completed quality
decision remain pending. The observer continues with the other arm and later
durable recovery checkpoints under separate reservations.
