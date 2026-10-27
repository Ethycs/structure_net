# Test additional reviewed supervision at a fixed training budget

Exploratory hypothesis `ncm-task-supervision-expansion-v1`: adding the newly
collected task-specific summaries improves unseen complete-summary quality
over further training on the existing 1,376 examples. One paired seed; this is
not an independently replicated result. Current application DEV is 0/128 for
the public epoch 2 initializer, and no new-arm student outcome exists. Teacher
collection is running. The 95% source-disjoint summary goal remains fixed.

Both arms start from the exact completed public epoch 2 checkpoint, SHA256
`c59051faea7d9005c2d91b1c88113d627d7545c9ae1a9c3dd7ac0a2e8f96f620`.
Load its model strictly and initialize fresh identical optimizers/scalers.
Train the same 105,323,136 generator/concept-source parameters. Keep the original
Llama body, repaired codec and word-source reconstruction path fixed. Keep
the existing inference path. Do not select a different initializer after
inspecting the live epoch 4 result.

Existing arm: all original 1,376 reviewed pairs, unmodified. Expanded arm:
the same pairs plus every admitted new first from the complete 1,024-outcome
collection. Require the independently audited expanded catalog and targets,
with at least 900 admitted new pairs. Keep exclusions and failed firsts.
The new batch changes topic and teacher mix as well as data volume; any
improvement applies to adding this batch, not uniquely to dataset size.

Fix 2,048 optimizer updates, effective batch32, microbatch4 and 65,536 example
presentations per arm. Cycle through complete shuffled case permutations before
reuse. The larger arm therefore gets fewer visits per individual source.
Use exactly half noisy presentation positions with the same flag mask and
per-position noise seeds in both arms. Retain the existing same-level uniform
replacement schedule, 0.25 at the first level declining to 0.05 at the last;
targets are always unmodified oracle codes. Equal example and level weights.
Source lengths differ, so equal update/presentation budgets do not establish
equal token counts, FLOPs or elapsed time. Report those measured quantities.

Sampler seed251142, corruption seed252142, dropout seed253142. CPU FP32 AdamW,
betas0.9/0.95, eps1e-8, weight decay0.01, clip norm1. Linear warmup128 updates
to3e-5, then cosine decay to3e-6 at update2,048. FP32 model/features, FP16
forward/backward, GradScaler128 with growth interval2,000. A skipped update or
nonfinite/protected gradient is a failure. TF32 off, strict deterministic
algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Run arms sequentially on physical GPU0 RTX3060 while the independent public
curriculum continues unchanged on the RTX2060. Require <=4.9GB peak GPU
allocation, a 256MiB feature cache, one I/O owner and at most two prefetched
microbatches. Record complete-update checkpoints with model/optimizer/scaler/
RNG state every256 updates. Sampling/noise depend on completed positions, not
prefetched progress. No inference-speed comparison is authorized by this plan.

Before each full arm, qualify a real two-update lifecycle including the longest
source in its complete admitted population, original-reference gradient checks,
strict save/reload and exact replay of update2, identical source-only outputs,
and unchanged protected tensors. Both datasets and targets must be fully
admitted before full arm training. Lifecycle outputs are not quality evidence.

At the fixed final update evaluate both arms on the same unchanged 128 DEV
sources, using source-only generation and the existing complete-summary rubric
that accepts paraphrases. Preserve all previous first judgments before any new
request, including every completed intervening public milestone/probe. Do not
overlap new judging jobs. Terra grades all128; Sol is admitted only if Terra
has all valid decisions and passes both >=61/64 cohort gates. New judgments
must pass their established pilot and independent replay requirements.

Report each arm's semantic pass count, all component scores, paired decisions
and the difference. A descriptive additional-supervision benefit requires
all Terra decisions valid and at least seven additional whole-summary passes
for the expanded arm; this does not qualify the 95% goal. DEV qualification
still requires both judges' existing >=61/64 per-cohort gates. If both qualify,
choose the higher same-case joint pass count, breaking ties in favor of the
existing arm. Preserve both failures if neither qualifies.

Final success remains the separately guarded sealed320 protocol: extend the
full source-overlap audit to all actual training inputs, then require all valid
decisions and the unchanged >=244/256 short and >=61/64 long whole-summary
gates under both qualified judges. Do not open final320 for an unqualified
model. No code/token metric or teacher-label validity score substitutes for
semantic accuracy. Later training changes require new prospective versions.

The schedule implementation is `ncm_task_supervision_schedule`. The trainer,
lifecycle, evaluation, audit and durable launch entry points must still be
implemented and qualified; this plan itself is not a training result.
