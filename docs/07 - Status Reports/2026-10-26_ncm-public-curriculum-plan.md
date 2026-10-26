# Train the frozen-body NCM on complete public epochs with application anchors

Exploratory hypothesis `ncm-public-curriculum-v1`: broader paired conditional
training improves complete-summary application quality from the measured
BROAD initializer. This does not replicate the paper's architecture, training
budget or endpoint. Its XSum regime uses 350,000 updates at batch512; this
first larger-data continuation is a substantially smaller experiment. The
95% complete-summary objective is unchanged and cannot be replaced by code
accuracy, token agreement or a public-data score.

Use all 162,353 admitted public TRAIN pairs and all 1,376 reviewed application
TRAIN pairs. Public references have not been verified as factual gold. Keep
the 320 public codec diagnostic sources reserved; keep application DEV128,
reserved64 and final320 outside TRAIN. The complete public catalog, source
features, fixed codec targets and their audits must finish before training.
Require the actual successful complete training-bundle receipt. Do not train
on a convenient subset while feature extraction continues.

Start from the exact BROAD checkpoint `74cdce0eb14bb99c801b6ec0a025c092f95249663becc05561af8dfbad6040ff`.
Train its 105,323,136 generator and concept-source parameters only. Keep the
original Llama body, repaired codec and word-source decoder fixed. The failed
adapted decoder that scored 92.65% reserved content reconstruction is not
the BROAD decoder. Keep the current inference path for this study.

Budget eight complete public epochs. Each epoch shuffles every public case
once and inserts one reviewed application example after each group of three
public cases, including the last group of two. Application sampling cycles
through shuffled complete permutations of all 1,376 cases. This yields
54,118 application presentations plus 162,353 public presentations per epoch:
216,471 total, 1,692 optimizer updates at effective batch128, and a final
23-example update. Across eight epochs: 1,731,768 presentations and 13,536
updates. The larger reviewed fraction is an explicit application retention
choice, not a claim that public references are equivalent to reviewed labels.

Microbatch4, FP32 weights and full source features, FP16 forward/backward,
TF32 disabled, physical GPU1 RTX2060. Use the corrected frozen-VQ helper whose
learned root retains gradients. Require original-reference gradient parity
and the corrected batch4 optimizer/restart lifecycle before qualification.
CPU FP32 AdamW: betas0.9/0.95, eps1e-8, weight decay0.01, clip norm1. Linear
warmup256 updates to lr1e-4 followed by cosine decay to1e-5 at update13,536.
GradScaler starts128, growth interval2000; any skipped update is a failure.

At each epoch, choose exactly floor(216471/2) noisy presentations using an
independent seeded permutation. For each noisy example, retain the existing
uniform same-level replacement schedule, 0.25 at the first level declining
to0.05 at the final level. Labels remain the unmodified oracle codes. Every
example and every level has equal weight; tail-update loss divides by its
actual example count. Sampler seed231142, corruption seed232142, dropout
seed233142. Corruption RNG is derived from epoch and presentation position,
independent of I/O scheduling. No adversarial or contrastive loss is added.

Load complete source matrices from the immutable disk store. One background
I/O thread owns mutable JSONL seek handles, with two prefetched microbatches;
reads within a public batch may follow shard order but return requested order.
Keep only indexes, the bounded reviewed-feature cache and current/prefetched
matrices in RAM. Never clip inputs or quantize stored features. Checkpoint
completed-update positions rather than prefetched sampler progress. Preserve
full model/optimizer/scaler/CPU-CUDA RNG and curriculum identity, with atomic
recovery checkpoints every256 updates and at every epoch boundary. Resuming
must produce the same next update, including histories and dropout.

Before full training, run two effective128 updates on real mixed batches,
save/reload after update1 and require exact replay of update2 and source-only
outputs. Compare serial and prefetched reads. Record disk-wait time separately
from training time. Require peak GPU allocation <=4.9GB, all protected tensors
unchanged and all trainable gradients finite, including nonzero root gradients.
This pipeline pilot is systems evidence only, not quality or application speed.

Evaluate source-only complete summaries at epochs1,2,4,8. At every milestone,
preserve the complete first-judgment lineage and use the existing DEV128
rubric; require all128 valid Terra judgments. Sol runs only when Terra passes
both >=61/64 cohort gates. Select the earliest milestone that passes both
judges' gates; otherwise continue the fixed budget and preserve each failure.
DEV selects a candidate and is not the final success claim. Before final use,
extend source-disjointness checks to all public and reviewed training data;
the full protection policy and sealed final evaluation still apply. Final
success requires the same >=244/256 short and >=61/64 long complete-summary
passes under both qualified judges, with all judgments valid. Never open
final320 for an unqualified model. No speed claim precedes measured quality.

Numerical, data-binding, gradient, checkpoint, resource or judge-validity
failures stop the dependent stage and are retained. No tolerance widening,
silent batch-size fallback, changed sampler, changed target or repeated judge
attempt may make a failed stage pass. Later remedies require fresh versions.

This plan fixes the curriculum before its implementation is qualified. Run
`ncm_public_pipeline_pilot run` only after its upstream prerequisites pass.
The full trainer is `ncm_public_training --end-epoch 1 --attempt 1`; later
milestones use the same full eight-epoch learning-rate schedule and resume
the exact saved checkpoint. Recovery uses a new attempt root and an existing
hash-bound checkpoint sidecar; completed artifacts are never overwritten.
Milestone checkpoints remain on `/data2`, with a symlink for the established
evaluation loader. Every later milestone requires valid preceding DEV results
and stops if that model already qualifies for the final protocol.

Run `ncm_public_development {preflight,inference_pilot,outputs,judge_pilot,grade}`
with the selected `--epoch` and training `--attempt`. Its source guard binds
the existing exhaustive public overlap audit to the exact current DEV file
and checks every admitted public identity; it also retains the reviewed-source
guard. Reuse the established semantic packets, inference, first-judgment
policy and judge gates. Preserve all4,179 historical first judgments and each
subsequent milestone's firsts. This does not complete the separate sealed-final
source comparison. Successful CPU checks or creation of these entry points
does not mean that full training started or that task quality improved.

Execution amendment, before any mixed-data training: the first batch4 GPU
lifecycle passed gradient parity but failed exact optimizer replay. The fresh
deterministic lifecycle passed all original gates, including identical model,
optimizer/scaler/RNG states and source-only outputs; peak allocation was
4,057,702,912bytes. Preserve both roots. Require the completed
`20261026_public_batch4_lifecycle_deterministic_v2` qualification and its
unchanged source/input hashes. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the
environment for the mixed pipeline, full trainer and DEV entry points; they
enable strict deterministic algorithms and deterministic cuDNN with benchmarking
disabled. The frozen feature producer continues under its original configuration.
See the [failure and intervention](2026-10-26_ncm-public-batch4-determinism.md).
The effective128 mixed-data lifecycle still must pass before full training.
