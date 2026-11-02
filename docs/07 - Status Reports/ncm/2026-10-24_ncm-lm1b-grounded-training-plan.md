# Full690 source-grounded NCM: exploratory neighborhood training

Date: 2026-10-24. Hypothesis `ncm-lm1b-full690-source-grounded-neighborhood-v1`.
EXPLORATORY, underpowered, non-rescuing new architecture/training trial. The full
objective stays >=95% complete-summary accuracy on the appropriately source-
disjoint task; representation fit or training loss cannot establish success.

Known before this trial: source-only old student0/128; task-domain codec52/64
below61/64gate. First capacity-label proposals were rejected,32/35fit; critique
revision's35 latest candidates all fit, but at least one already has a valid
semantic failure. The strict label-qualification study continues unchanged.
Neither codec nor label gates are retroactively passed. This new trial explicitly
allows potentially noisy TRAIN targets, a normal supervised-learning condition;
all semantic assessments/failures remain recorded. It does not require or claim
perfect TRAIN labels and cannot rescue either parent experiment.

Use all690 original source cases, with the latest genuine capacity35 candidates
(23 first passes retained,12 critique-guided revisions) and655 unchanged original
targets. Pick these targets independently of any new student outcome; all cases
receive equal loss weight. Require every candidate schema-valid and every full
target to fit126BERT lexical positions,128storage tokens before/after decode.
No truncation, exclusion, label-free held-out relabeling, best-of output selection
or fallback. Record that semantic qualification is not claimed. Remove stale
GPT2word/XSumconcept arrays. All source/context/alignment fields must equal the
original690 dataset, SHAac3c56f7..., so its prior final-split exclusions remain
applicable. Independently check all690 source/session/group/hash IDs against
both permanent64-case task DEV cohorts; all128 must remain outside gradients.
Never open/hash final320or old final source files.

Prepare all native128word frames with genuine initial/terminalCLS101 andPAD0.
Run a representative first/longest source target-encoding pilot on physicalGPU2
then encode all690 fresh641-position hierarchies with the exact released frozen
encoder/books. Reproduce pilot targets after fresh load; all655 unchanged label
codes must equal the earlier native hierarchies. No target latent optimization.

Architecture is the tested source-grounded native97M NCP plus measured111M domain
codec, both fully frozen/eval. Start all11,796,480 NEW concept/word source-attention
weights fresh with zero output projections, constructor seeds68142and90142;
never continue a two-update pilot checkpoint. Original3B Llama body and MLPs are
never loaded/updated. Reuse exact FP32 full-source frozen-body features, date23
prompt hashes preserved. Each feature file must match ID/source/prompt/hash,
dtypeFP32 and full[N,3072]shape. Host-cache at most8GB and require12GBavailable;
current690 feature payload is6.046GB. Pad only inside a microbatch and mask it;
never truncate sources. Full sources can reach2757 aligned positions.

The complete clean/noisy/self-history objective already passed a representative
real BF16 CUDA lifecycle: microbatch4, both attention paths, exact model/optimizer/
allRNG and label-free generation reload, frozen states unchanged, peak2.731GB.
Run the new complete-data-path two-update pilot next, force schedule1024/maximal
noise and source-only self-history, first/longest full sources, real save/reload.
Stop on nonfinite loss/gradients, wrong shapes, drift or peak>7.5GB.

Full training: physicalGPU0 RTX3060 nativeBF16, FP32weights/BF16autocast losses,
TF32off, microbatch4, accumulation4,4096fixed updates. AdamW(.9,.95),eps1e-8,
weight_decay.01,clip1. Warm128updates to1e-3, cosine to1e-4. Independent sampler
95142,neighborhood96142,dropout94142; preserve all RNG states. Uniform sampling
over all690 with replacement. Save recovery1024/2048/3072, choose only final4096.

Concept CE averages within each of12levels then equally across levels. Clean
loss.5concept+.7word; corrupted-history.5concept+.3word, wordPAD0ignored. Uniform
replacement among8192codes with per-case rate0to(.25*min(step/1024,1)), root
protected. Backpropagate branches separately with1/accumulation scale. On first
microbatch every4th update after256, generate student concepts from features/
mask ONLY in FP32/eval; restore train mode and additionally backpropagate
.25*(conceptCE+wordCE), without accumulation scaling. Preserve complete predicted
histories, corruption proofs and every actual sampled ID/loss/gradient norm.

After pilot and full training, require exact fresh attention/optimizer/sampler/
neighborhood/CPU+CUDA RNG restoration and source-only concept+word probes.
Pretrained state hashes must remain unchanged. Bind source, target, feature,
checkpoint and environment hashes and actual child/whole-tool exits. Fresh
`20261024_lm1b_grounded_training_{data,cache_pilot,cache,pilot,train,workflow}_v1`
roots; producing source/plan immutable after launch. The workflow ends trained
with task accuracy PENDING, not qualified. All old failures are preserved.

Next mandatory stage is source-only prediction of ALL128 unchanged permanent
DEV cases with NO target words/concepts in prediction inputs, then both qualified
judges under the unchanged full-summary/128storage rubric and same-case cohort
gates. Only if both64-case cohorts pass61same cases under BOTH judges, all valid,
may the original sealed320final campaign proceed. No quality or speed claim from
this training campaign; no goal completion until actual final task evidence.
