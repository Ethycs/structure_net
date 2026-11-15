# Qwen3-8B BF16 incremental compression and recovery

Study **JSPACE-QWEN3-8B-PROGRESSIVE-01**, exploratory. Written November 14,
2026 (America/Los_Angeles), before any Qwen3-8B model execution or outcome.
The user selected the official, exactly-8B-family Qwen3 checkpoint in BF16;
the earlier request for FP32 was explicitly superseded. This study does not
switch to Qwen3.5-9B or a quantized community checkpoint.

Status amendment, November 15 UTC: **deferred at the user's request to finish
the 3B model first**. The official checkpoint is downloaded and CPU-tested
port helpers are preserved. No 8B GPU calibration, training or benchmark has
been launched; this is preparation, not an 8B compression result.

## Question and intervention

Can the functional-subspace, polynomial-MLP, and staged joint-recovery method
be carried to Qwen3-8B, extending a contiguous compact suffix one block at a
time while keeping actual task accuracy within five percentage points of both
the original model and the previously accepted compact parent?

Pin `Qwen/Qwen3-8B` at
`b968826d9c46dd6066d109eabc6255188de91218`. The published BF16 checkpoint has
8,190,735,360 parameters, 36 blocks, residual width 4096, intermediate width
12288, 32 query heads, eight KV heads, head width 128, and **untied** input and
output embeddings. Authenticate all five safetensors shards, configuration,
tokenizer, and producing source. Keep earlier completed studies unchanged.

For each new block, reduce residual coordinates 4096 to 2048, retain all heads
but select 32 rotary pairs per KV group (128 to 64 attention coordinates), and
project values 128 to 64. Retain native attention scaling and softmax. Qwen3's
original 128-dimensional Q/K RMSNorm must run **before** selecting coordinates;
therefore this first port keeps full-width Q/K projections. It must not silently
replace native normalization with 64-dimensional normalization. Fold residual
projections into matrices and preserve the learned Q/K norm gains. Keep 24
polynomial anchors and replace 1024 MLP channels through the existing quadratic
graph-relation compiler, retaining the other native channels. These are
approximate fitted graph relations, not exact identities of the neural network.

Use BF16 trainable weights and activations, FP32 geometric construction,
anchors, bridge arithmetic, loss calculations and optimizer masters/moments.
Disable autocast and TF32. Preserve the original untied output head and input
embedding. Do not reuse the Qwen2 fused attention path, which omits Q/K norms.
The adapted packed model must independently reproduce the selected predictions.

## Data, task-derived bases, and leakage boundaries

Reuse the earlier fixed 512 training and 64 development prompt identities,
balanced over ARC, BoolQ, HellaSwag and WinoGrande. Retokenize raw prompts with
the pinned Qwen3 tokenizer and `enable_thinking=False`. Rebuild offsets and the
four causal supervision positions, including the final answer-letter token.
No old Qwen2 token IDs, activations, teacher logits or fitted bases are reused.
All 576 prompts fit the prospective 416-token capacity: measured tokenization
maximum 382, total 80,419 valid tokens, with no truncation or dropped example.
These are constrained next-token multiple-choice measurements, not unrestricted
thinking-mode generation or summarization quality measurements.

Stream the original frozen prefix through one native block at a time. Preserve
all needed boundary banks and full-vocabulary logits in FP32 disk arrays that
exactly represent the BF16 computation. Captured states can be reconstructed
losslessly in BF16. Every training minibatch recomputes the complete trainable
compact suffix on current student activations. Only the unchanged prefix is
cached. Place large artifacts on the data volume via the repository's dated
experiment-root link; do not place Python environments under experiment data.

For each block, deterministically select 32 training examples per family.
Differentiate the correct answer logit minus the mean competing-answer logits
through the native new block and the accepted compact suffix. Accumulate the
input task-gradient Gramian over valid positions. Construct an explicit rank
2048 block Krylov/Rayleigh–Ritz basis, block width 2048 and one expansion, seed
20861 plus the block index. Report resolved functional directions separately
from numerical-null seed completion. Use position-specific training means up
to 416 positions and a declared global training fallback for unobserved rows.

Capture head-mask sensitivity at **post-Q/K-RMSNorm** outputs, before RoPE;
combine the shared KV group's query/key contributions before squaring. Value
coordinates use an uncentered second moment. Polynomial normalization,
anchors, coefficients and removal ordering use training activations/Jacobians
only. Fix one anchor-width candidate, 24, so development diagnostics cannot
select a relation width. Use SVD rank 64, oversampling 16, one power iteration,
ridge .01, and chunks of 256 rows. Keep per-family reconstruction, output-error
and redundant-relation diagnostics. This margin-derived calibration is
explicitly different from the previous full-vocabulary random-probe capture.

## Recovery and prospective selection

Begin at block 35 with the original model as teacher. After an accepted stage,
add exactly one preceding block initialized from its newly calibrated fit and
carry the previous selected blocks/norm forward exactly. Rebuild only the new
interface; preserve inherited interior bridges. Jointly train every current
compact block and final norm, keeping geometry, embeddings and readout fixed.

Each stage runs 12 epochs, microbatch four, accumulation four, 384 optimizer
updates over 512 cases. AdamW uses learning rate 1.2e-4 with cosine decay, zero
weight decay, global gradient clipping at one, and seed 20861. Use output-only
`.7 * T² * KL(parent || student) + .3 * CE(student, gold)`, T=2, at the four
positions. The first parent is the native BF16 model; subsequent parents are
the prior accepted checkpoint. Native-teacher divergence remains diagnostic.

Use same-parameter-dtype CPU gradient accumulation after each completed leaf
gradient and clear its GPU gradient immediately. FP32 CPU masters/moments
perform the update in bounded chunks. Apply this rule from the first stage;
do not change the optimizer's accumulation precision as the suffix grows.
Record actual finite gradients and changes in optimizer masters, separating
BF16 stored-value changes that can round away. Account for CPU gradient banks,
masters, moments, checkpoint copies and live-model offload during packed
evaluation; a resource failure is not an accuracy failure.

Evaluate initialization and each completed epoch. Select highest development
correct count, then lowest full-vocabulary CE, then the earlier checkpoint.
Initialization remains eligible. Preserve every epoch's individual decisions
and the selected checkpoint. The next stage is permitted only if fresh selected
verification passes and the selected count is at least
`max(ceil(native_correct - .05*64), ceil(parent_correct - .05*64))`.
This is a whole-model comparison, not five points per block. Stop expansion at
the first accuracy miss and retain the last accepted version; complete and
report the failing stage without automatically changing its schedule. The
maximum requested expansion is all 36 blocks. A partially compressed model is
reported with its actual layer count, never as a fully compressed model.

## Validity, execution and reporting

Run focused CPU contracts, tiny actual forward/backward and reload tests, then
a separate reserved real-model lifecycle pilot before scientific capture or
training. Replay unchanged native suffixes against the authenticated boundary
and logit banks with the same batching. Match controls and comparisons to BF16;
do not mix old Qwen2 FP32 counts into the new baseline. Fresh packed verification
requires identical answer decisions and recorded numerical logit agreement.
Before outcomes, fix tolerance at `.0625 + abs(reference)/128`; report maximum
deltas rather than claiming bitwise equivalence where it is not observed.

Every CUDA process must run through `inference-ctl lease run`. Start with the
RTX 3060, UUID prefix `GPU-0370fdb0`, reserving 11264 MiB and capping the native
allocator at 11008 MiB. Check the live reservation and memory throughout.
Do not initialize another GPU or silently change execution precision. Validate
longest-context accumulated updates and packed-evaluation memory; increase
execution capacity only through a documented resource attempt, preserving the
failed attempt and unchanged scientific recipe. No unreserved GPU work.

Freeze each producing source version before its pilot and scientific work;
changed producers require a new versioned root and explicit evidence lineage.
Stage manifests authenticate all artifacts and the parent checkpoint. Resume
only completed fingerprint-matched stages; do not retry an accuracy miss.

Primary outcomes are actual correct/count, paired errors, last accepted layer
count, and exact deployed tensor inventory including **both** untied endpoint
matrices and all fixed geometry. Record calibration redundancy and recovered
rank as diagnostics. A speed claim requires a separate matched native/packed
deployment benchmark; streamed teacher/capture wall time is not inference
throughput. This exploratory development-selected campaign remains
`confirmed: false`. Untouched evaluation and generation quality require their
own fixed comparison, with no subsequent fitting on those answers.

Close completed stages with a measured report, authenticated machine-readable
result, conservative NAL registration/readback, and research-journal entry.
Preserve partial/failed/invalid stages explicitly and release reservations on
terminal execution.
