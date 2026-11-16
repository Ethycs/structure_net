# Broad-response progressive compression rebuild

Study `JSPACE-QWEN3B-PROGRESSIVE-RESPONSE-REBUILD-01`, exploratory.
The user authorized the proposed small rebuild on2026-11-16 and explicitly
required keeping the all-at-once recovery running. Follow
[NAL-STD-DISTILLATION v1](../../05%20-%20Standards/NAL-STD-DISTILLATION-v1.md).

## Question, evidence and scope

Can fresh functional-subspace compression, calibrated and recovered on complete
responses from eight task families, preserve the native model's behavior when
one compressed block is extended to two? The original frozen teacher is
Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
This starts from the native model, not the old16-block compressed geometry.

The previous progressive procedure used four multiple-choice families,128
balanced TRAIN calibration cases and four supervised positions per case. It
preserved inherited bases and relations. The current complete-response recovery
already uses the broader corpus but holds that geometry fixed. Its first new
checkpoint reduced response DEV NLL6.96739to4.47751 while MC DEV fell186to164
correct out of256. This motivates testing broader geometry; it does not prove
geometry caused the regression. The new branch is not a matched causal
comparison with the older16-block model, which differs in depth and history.

Initially test native layer35 alone, then layers34–35. Advancing to the second
stage requires a fully evaluated eligible first-stage checkpoint. Deeper
three/four-block tests are conditional future work; do not automatically start
a36-block run. Keep the old16-block initializer and recovery checkpoints as
contextual controls. Let its current recovery reach durable update90 and finish
that checkpoint's GPU evaluation, then interrupt only that named worker and
release its3060 lease. Preserve its interruption receipt and all completed
results. The all36 worker and2060SUPER reservation remain unchanged. Existing
packed sidecar work may finish on its own2060 reservation.

## Corpus, separation and native targets

Reuse the authenticated `jspace_3b_response_data/20261116_v3` corpus:4,441 TRAIN
examples and128 DEV examples across Dolly, XSum, ARC, BoolQ, HellaSwag,
WinoGrande, GSM8K and MBPP. Preserve its source groups, duplicates policy,
complete responses and real EOS. The fixed envelope remains prompt<=288,
response<=128 includingEOS, total<=416; no truncation or invented EOS. This
does not qualify the application's long input range. No new TEST is opened.
The earlier reserved-final access incident remains recorded; eventual
confirmation requires a new untouched set.

Choose128 balanced TRAIN calibration examples(16 per family),32 separate
TRAIN relation-diagnostic examples(4 per family), and256 local-prealignment
TRAIN examples(32 per family). Keep example identities disjoint between these
three sets, with source-group separation where feasible and recorded. All DEV
examples remain excluded from every fit and training target selection. Freeze
the exact indices/seeds in the data plan before GPU execution. Joint recovery
may use the whole admitted TRAIN population, including calibration examples.

Capture original native residual boundaries34,35 and36 for the admitted corpus.
Boundary36 is the actual residual before final RMSNorm. Store exact BF16 bits
in mmap banks, with token/record identity, offsets, model and source provenance.
Validate normalization of boundary36 against the existing exact native final
hidden cache. Reuse its original teacher readout/distributions, generation
controls and response DEV reference. Capture separate native34/35 input banks
for the already exposed256-case MC DEV set; these rows are evaluation-only and
must never enter calibration or training. Never open the old audit/final task
contents to construct this retention set.

## Compression order and fixed arithmetic

For each new entrance block, use the native entrance plus any accepted compact
suffix as the differentiable task pathway. Generate two independent groups of
eight full-vocabulary logit probes over every response-prediction position,
including EOS, with derivatives through the entire causal input context.
Record actual token/probe counts; do not relabel four sampled positions as
complete response supervision. Reuse the existing functional Jacobian Gramian
eigenspace construction. Describe the executed eigendecomposition accurately;
do not claim an explicit truncated Krylov algorithm if none ran.
Compute position means from the complete admitted TRAIN native-boundary bank,
retaining the prior mean-filling rule for unobserved positions and recording
counts. Functional probes and head moments use the balanced128-case calibration
subset. Cached TRAIN means avoid additional full-model forward passes.

Keep the established geometric recipe: residual rank1024 from native2048,
head dimension64 with original head count, native softmax and original attention
scale, and the same native residual/compressed execution structure. The
polynomial substitution is in the MLP, not a new polynomial-attention change.
Preserve the original sequence: functional basis/head fitting, compiled
interface, activation capture in that compiled interface, then quadratic
relation fitting and removal of1024 MLP channels using the existing relation
compiler. Keep retained channels and the explicit residual path. Use the32
TRAIN-only diagnostic cases for fitting diagnostics; DEV must not select
bases, removed channels, anchors or polynomial coefficients. Pin the existing
fit controls and random seeds in each phase plan.

When prepending layer34, strictly inherit the accepted layer35 parameters,
geometry and existing interfaces. Introduce only the declared new entrance
and its bridge. Record inheritance and fixed-state checks before recovery.

## Sequential recovery

Save the raw compiled initializer. First fit only the newly added block on the
256 local TRAIN cases, one pass, microbatch1 and accumulation16, LR2e-5.
Freeze the accepted suffix, norm, native prefix, original readout and geometry.
At every response-prediction position, match the new block's reconstructed
full2048-dimensional residual to the original native next-boundary residual.
Use mean squared error divided by the positive mean squared native-target
energy over those same TRAIN positions and dimensions, computed in FP32 and
fixed for the stage. Keep the final local state; no DEV-based local epoch hunt.

Then jointly recover all compact blocks and final norm using exact full-vocabulary
forward KL to the original native teacher at every response/EOS position,
temperature1, LR5e-6, BF16 model/gradients, CPU FP32 Adam masters/moments,
microbatch1, accumulation16, clipping1, weight decay0. Geometry, residual route,
native prefix and tied readout remain fixed. Use a deterministic balanced
one-pass TRAIN order through the first complete window reaching25,000 response
tokens. Freeze actual examples/tokens/update boundaries prospectively.
Checkpoint after local fitting and at first windows reaching5k,10k,25k. No
automatic extra epochs or larger token rung. Retain the raw initializer and
all early checkpoints. Discard systems-pilot updates before scientific fitting.

## Packed evaluation and advancement

Before measuring a candidate, pack and export a separate snapshot, destroy
the first inference instance and structurally reload the saved artifact.
Require identical packed tensor state and same-packed-path output replay.
Do not require packed-versus-unpacked full-logit equality as a quality gate.
Generation must execute the real native prefix on its generated history;
teacher-forced boundary caches cannot substitute for this. Track native prefix
and shared embedding dependencies when reporting bytes or latency.

Use the original native-relative MC retention margin: at most5pp overall and
10pp per family on exposed MC DEV256. Require full-response DEV128 NLL at most
1.10times native. Run these checks first. Candidates failing either remain
diagnostic; expensive generation/judge results cannot rescue them and may be
skipped with an explicit reason. For candidates passing these checks, run six
canaries and two selected DEV generation cases per family, original EOS and
greedy128-token cap. Execute MBPP in the existing bounded sandbox, grade GSM
numeric answers, and use blinded paired semantic grading with the pinned
`codex_sdk/gpt-5.6-sol` route/rubric. Use one judge worker; preserve missing or
failed requests as unavailable evidence, never numerical zero accuracy.

For this new pilot only, use paired native-relative retention within5pp on
canaries, semantic generation/task success, MBPP and GSM8K; report absolute
rates separately. Do not impose the older50% absolute floor: the native teacher
itself passed only2/6 strict canaries, so that floor would reject unchanged
teacher behavior. This prospective calibration does not change any previous
run's classification. Native-relative retention is not an application-quality
or95%-accuracy claim. Missing judgments or executable evidence fail advancement.

Among fully eligible checkpoints prefer higher semantic task success, then
lower response NLL, then earlier update. If none qualifies, preserve the
diagnostic best and stop before the next compression stage. Do not replace
failed metrics with different successful endpoints. A pass is exploratory
relative retention only, `confirmed:false`, not deployment readiness.

## Execution and records

Use new modules and append-only artifacts under
`data/experiments/jspace_3b_progressive_response_rebuild/20261116_v1`.
Freeze source closure and inputs separately for native capture, fitting and
recovery phases. Reuse NAL data/loss/result contracts and existing compilers;
do not mutate the all36 producer or its source dependencies. Run focused CPU
tests, tiny lifecycle/serialization checks and representative discarded CUDA
pilots. Every GPU phase uses `inference-ctl lease run`, normally11264MiB on3060
with11008MiB allocator cap; CUDA remains hidden for preparation/orchestration.
No new phase may assume apparently free memory inside another experiment's
reservation. Record actual phase timing, samples/tokens, CPU/GPU storage,
failures, lease IDs and checkpoint provenance. Keep Phi paused.
