# Test sample scaled EMA for learned NCM codec transfer

Date: 2026-10-23. Exploratory hypothesis
`ncm-sample-scaled-codec-ema-v1`. The preceding broad codec transfer failed:
on252excluded complete summaries, original oracle token recovery was95.179%
and adapted recovery67.167%; changed-number cases increased53to105. These
are codec reconstruction measurements, not summary task accuracy. Preserve
the complete failed run and registry result before this follow-up starts.

The preceding2048optimizer updates caused16384EMA updates on microbatches of
four windows. The released checkpoint's mean cluster counts equal each level
width, consistent with4096windows per EMA update. Accumulation eight aggregates
gradients, not EMA sufficient statistics. The new centroids moved by relative
L2 distances1.46–2.07; no near-zero-centroid collapse was observed. This does
not prove that EMA alone caused the regression.

Change only the EMA transfer rule, holding corpus, microbatch four,
accumulation eight,2048updates, optimizer, learning-rate schedule, seeds,
dropout configuration, complete diagnostic targets and final-only selection
fixed. Start fresh from the released checkpoint. Let s=4/4096. Before any
training forward, scale EMA counts, EMA vector sums and smoothing epsilon by s,
without changing codebook vectors or encoder/decoder weights. Set per-forward
EMA decay to .99**s. Counts and sums now use actual-microbatch units, and1024
constant-statistics micro-updates have the decay of one4096-window update.
This is a controlled engineering adaptation, not a released upstream trainer.

Focused CPU tests prove unchanged initial codebook vectors, covariance of the
smoothed centroid update under unit scaling, and the decay composition rule.
Use the same real two-update GPU2 pilot and exact checkpoint/optimizer/RNG/
dropout-step reload controls as before. Require initial probe codes, raw words
and text to match the unscaled original-checkpoint pilot exactly. Full training
requires the pilot's observed terminal success and starts fresh. Require every
sampled window index and learning rate to match the failed2048-update run.
No old failed or intermediate checkpoint is selected for continuation.

A separate output-preserving optimization caches tokenizer vocabulary size
once per reconstruction call. The measured installed tokenizer took1.073s
for100length lookups. Differential CPU tests must retain identical text,
invalid-ID counts and all other fields. This changes validation overhead,
not model arithmetic or a tokens/s claim.

After terminal training, use the same paired252-target oracle reconstruction
protocol on GPU2, four48-token blocks per complete target. Require all252
original-arm outputs to reproduce the preceding original arm exactly. Retain
the previous99.5%token,99%first-EOS,zero-invalid-ID and non-increasing changed-
number representation gate. No full task-accuracy claim or conditional NCM
generator admission follows without the subsequent semantic and source-only
task evaluations. The clean native teacher remains56/64(87.5%), not95%.
Original parent weights stay frozen and absent from codec training. Both final
task cohorts remain sealed. Stop on systems failure or drift; preserve it.

New roots `20261023_sample_scaled_codec_ema_{pilot,train,reconstruction}_v1`.
Capture this wrapper and the unchanged core implementation in each producer,
with the exact EMA formula and runtime overrides recorded. Reproduction:
`python -m experiments.structure_net.ncm_sample_scaled_codec_ema
{pilot|train|measure}` in the existing moba environment with physicalGPU2,
FP32,TF32off. Aggregate, report and register under the new hypothesis identity.
