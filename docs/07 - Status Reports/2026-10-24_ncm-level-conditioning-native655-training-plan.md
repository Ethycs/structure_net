# Fresh FP16native655 candidate with level-dependent conditioning noise

2026-10-24. EXPLORATORY hypothesis
`ncm-level-conditioning-FP16-native655-source-training-v1`.
Known baseline: original full-native655model trained1024updates on655original
targets and failed0/128DEV. No prior result is replaced. Final320remain closed.
While the10000source cache runs onGPU0 and teacher generation has timeouts,
use availableGPU2 to test a fresh corrective candidate rather than wait idle.

The actual two-update FP16GPU2 systems pilot completed and strictly reloaded
allmodel/optimizer/scaler/RNG/source-only outputs, peak3,614,384,128bytes. Its
checkpoint is NOT an initializer. Eleven core CPUunit checks passed; two
padding checks requireCUDA and passed withCUDAavailable (a prior CPU-only
invocation failed because the fixture still constructs CUDA target tensors).
Preserve that observation; it is not model-quality evidence.

Keep the SAME655unchanged original targets, excluding35revised targets,
nativeencoder/VQhashes and original28layer Llama body/MLPs. Use full original
FP32cached source features, detached, every position. A new512MiB raw-feature
LRU bounds host cache residency; no corpus-wide5.6GBtensor loading. This is
training data access, not application inference reuse. No new teacher labels
are consumed. Reapply identity/session/group/hash TRAIN/DEVguards.

RESEToriginalb4b901f... initializer for BOTHpilot/full, never native16/native655
or any new pilot weights. Train127,732,736nativegenerator+worddecoder+source
attention parameters. FP32weights and FP16training with FP32hard source-only
history/inference on SAMEphysicalGPU2 RTX2060SUPER. TF32off. Require5GBfree,
peak<=4.9GB. AdamW.9/.95,eps1e-8,wd.01,unscaledclip1,GradScalerinitial128 with
growthinterval2000. Nonfinite unscaled norm fails without skipped/repeated
optimizer updates, scale fallback or precision changes.

Fixed1024updates, micro1accum16, global16,16,384presentations. Separate sampler
126142/neighbor127142/dropout128142streams. Sample all655with replacement,
no length/quality filtering. Warm64updates to5e-5, cosine to5e-6at1024. Only
fixedfinal1024checkpoint is selected; no DEV-based checkpoint selection.

New objective keeps clean.5C+.7W and noisy.5C+.3W divided by accumulation and
once-per-update first-micro source-only self-history.25C+1W, with hard-forward
word values and biased soft-codebook backward. Native concept losses weight
levels equally. New conditioning history uses fixed linearly decreasing
same-codebook rates.75->.25including FIRSTGENERATEDcode; clean targets are
never modified. BOS/nullprefix is separate. Endpoints are our chosen candidate,
not released exact author LM1Btraining settings.

First require the completed actualFP16systems pilot. Then fresh data-path
pilot2updates with micro1accum2using first/longest real paired TRAINsources;
full run resets the initializer. Require actual fullstate/optimizer/scaler/allRNG
and16fixed TRAINsource-only hard outputs to reload exactly, protected hashes
unchanged, root/nativegenerator/decoder/source attention weights updated. Record
every sampled ID/source length, LR, unscalednorm, scaler and all per-level
corruption selected/changed counts/hashes in append-only trace. CPUaudit replays
all sampling and corruption and strictly loads full state after actualwholeexit0.

Fresh roots: `20261024_level_conditioning_FP16_native655_training_{workflow,data,pilot,train}_v1`
underdata/experiments/ncm_paper_regime. Stop and preserve on error/invalidity/OOM;
no automatic retry or mutation of an earlier producer. Bigger corpus preparation
and its failures continue independently; this candidate does not consume it.

After completed audited training, run complete128source-only DEV under the
unchanged semantic rubric and latestcanonical3222firsts. Full128denominator,
no output repair/truncation/oraclecodes. Terra first; Sonnet only if BOTHcohorts
reach61/64valid complete passes, then require SAME61/64cases percohort onboth.
DEVfailure cannot openfinal320. Future successfulDEVmust have its own verified
complete-checkpoint final protocol. Quality before speed; systems/training
success alone is insufficient.

This candidate changes noise, precision, microbatch and seed relative to the
failed baseline. A quality gain is useful exploratory evidence but does not
identify the causal contribution of noise alone. Do not claim matched fullpaper
replication or a throughput gain without the appropriate separate measurements.
