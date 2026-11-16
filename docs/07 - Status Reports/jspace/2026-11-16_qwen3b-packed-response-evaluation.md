# Packed complete-response checkpoints

Study `JSPACE-QWEN3B-PACKED-RESPONSE-01`, exploratory deployment evaluation.
The user requested packing before testing on 2026-11-16. This is a separate
evaluation of exported checkpoints from `JSPACE-QWEN3B-RESPONSE-RECOVERY-01`;
it does not change either running recovery objective or its frozen producers.

## Question and known evidence

Does a saved recovery checkpoint retain useful task performance after folding
and packing its compact blocks, and what memory and runtime does that exported
model actually use? Evaluate the progressive16 and all36 arms independently.
Their different compression depths and training histories are not controlled
architectural comparisons. At preparation, both runs have saved update0000;
subsequent scheduled checkpoints are updates46,90,152,228. Evaluate each durable
checkpoint once per frozen execution configuration. Never read an actively
written temporary checkpoint or substitute live training weights.

Earlier packed-versus-unpacked probes failed a strict full-logit tolerance,
while all three sampled greedy decisions agreed for each arm. Those results
do not establish task-quality loss. This study deliberately tests the actual
packed outputs under the user's five-percentage-point task budget. Do not
silently loosen the earlier probe's result or label numerical equality proven.
Require exact packed-state export/reload identity and same-path output replay;
packed-versus-unpacked logit equality is a diagnostic, not this quality gate.

## Execution and provenance

Authenticate the parent campaign, original model, cache, exact saved recovery
checkpoint, and local producer closure. Construct the original student's
geometry, load the saved response checkpoint strictly, then fold and pack a
separate inference model. Save a standalone compact snapshot, release the first
instance, reload it, and test the reloaded packed instance. Pin every export's
tensor identities, shape/configuration, artifact bytes, and source checkpoint.
For progressive16, record the dependency on the original native prefix and
tied embedding/readout; compact snapshot bytes alone are not full-model size.

Use BF16 weights, FP32 reductions where already specified by the executor,
TF32 disabled, and the packed Torch path on Turing. Do not invoke Ampere-only
BF16 Triton kernels on a 2060. Generation projects the last hidden row through
the full vocabulary; response losses retain every supervised response position.
Teacher-forced evaluations may use authenticated native boundary caches, but
generation must recompute the original native prefix on the generated history.

Run under a separate reservation on the spare RTX2060: allocation5632MiB,
allocator cap5376MiB. Keep existing3060/SUPER recovery leases unchanged.
The default progressive prefix mode is resident. If this cannot fit, preserve
the failed receipt and prepare a separate CPU-resident-prefix configuration,
streaming frozen layer weights to the GPU and explicitly including transfers
and orchestration in latency. It is not comparable to fully resident GPU
execution. Use locked offline Pixi and bounded CPU threads. A resource
failure is not a model-quality result. No automatic training changes follow.

## Evaluation and decision

Reuse the parent study's admitted DEV128 across eight task families, full
response-token NLL/KL, historical MC DEV256, six executable generation canaries,
and two generation cases per family. Keep original native controls and raw
packed generations. Semantic judging, MBPP sandbox execution, GSM8K numerical
grading, and context/provenance remain the parent protocol's implementation.
At preparation the unpacked initializer judgments were incomplete: progressive
had12/16 valid replies, two HTTP429 and two connection errors; all36 had11/16
valid replies and five HTTP429. Successful replies matched the pinned judge
model and schema. Use one judge worker in this packed sidecar, instead of the
parent's two, to reduce concurrent service load. Retain the same model, rubric,
temperature, request timeout and single-attempt policy. Missing judgments remain
unavailable evidence, never a numerical accuracy score or a successful gate.
Do not access the old reserved final512 contents. Independent confirmation
still requires a new untouched set because of the previously recorded access
incident. No final TEST access occurs in this study.

Keep the parent eligibility rules: MC accuracy within5pp overall and10pp per
family of native, complete-response NLL at most1.10times native, and at least50%
success with no more than5pp native loss on canaries, semantic generation, MBPP,
and GSM8K screens. Missing evidence fails eligibility. These small selected DEV
screens are selection/engineering checks, not official benchmark estimates.
Retain every failure; never let improved speed rescue failed quality.

Among eligible packed checkpoints prefer higher semantic task success, lower
response NLL, then the earlier checkpoint. Selection is separate from the
unpacked training run's decision. A packed pass is exploratory, `confirmed:false`.
Do not declare recovery, deployment readiness, or a speedup before measurements.

Record warm inference timings, input/generated token counts, synchronization,
device/runtime, prefix execution, artifact bytes, peak GPU memory, and total
elapsed time. Cache-free generation recomputes prefixes and must be labeled as
such; it is not a KV-cached service decode benchmark. Teacher-forced suffix
timing excludes prefix work and cannot be presented as full-model latency.
Concurrent recovery may contend for CPU/PCIe, so timings under that load are
descriptive. A speedup claim needs a matched hardware/workload control; later
3060 fused-kernel measurements need their own valid reservation.

Artifacts live under a new dated
`data/experiments/jspace_3b_packed_response_evaluation/20261116_v1` root, one
immutable child per arm/checkpoint/execution mode. Archive source before GPU
execution, preserve NAL-compatible results and failures, and record launch and
measured findings in a separate analysis report. A CPU watcher may discover
durable scheduled checkpoints and request each GPU lease serially; it must
not bypass the reservation system or rewrite completed cells.
