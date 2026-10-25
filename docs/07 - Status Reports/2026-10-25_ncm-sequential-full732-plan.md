# Full732 next-scale training after decoder repair

This conditional exploratory stage follows the user-authorized decoder → generator
→ semantic evaluation sequence. Launch only after the full732 decoder stage passes
its previously fixed reconstruction gate and strict reload. Earlier failed studies
remain failed; no held-out labels or sealed final sources enter training.

Initialize from the completed 16-case sequential generator and the fixed full732
decoder checkpoint. The generator has already seen the 16 control TRAIN examples;
disclose this warm start. Freeze the repaired decoder, encoder, codebooks, word
source attention and cached full-source Llama features. Train only the native
generator and its source attention (105,323,136 parameters). All 732 admitted
engineering TRAIN summaries participate. Use fixed final 2,048 updates, microbatch
1, accumulation 4, shuffled without replacement each epoch; 8,192 presentations.

Maintain native next-scale conditioning: each level reads only the cumulative
reconstruction of strictly earlier levels. Alternate clean and noisy histories,
using the prior .25→.05 per-level uniform corruption schedule, and always clean
code targets. Weight each of the 12 level cross-entropies equally. Source-only
inference remains the explicit 12-step sequential path.

For training throughput, use the author's native block-diagonal teacher-forced
forward to batch the independent levels. This changes execution order, not the
earlier-level dependency or loss. Before launch require actual CUDA comparisons
against the explicit per-level FP32 loop on first/longest sources, clean/noisy
histories: logits allclose rtol/atol 1e-4, zero argmax differences, all native
trainable gradient relative L2 error <=5e-5 and max absolute error <=1e-4.
Dropout is disabled only for this mathematical parity check. Training uses the
original dropout distribution; no bitwise identity of optimizer trajectories is
claimed. If parity or memory fails, preserve failure and do not launch full training.

Run focused CPU objective tests and a two-update actual CUDA lifecycle pilot.
Train FP32 weights with FP16 autocast/scaling; FP32 inference; no TF32; physical
GPU2; allocation cap 4.9GB. AdamW .9/.95, epsilon 1e-8, decay .01, clip 1; linear
64-update warmup to 1e-4 then cosine to 1e-5. Sampler/noise/dropout seeds are
161142/162142/163142. Reserve a bounded 24GiB host feature cache: all 17.44GiB of
unchanged FP32 TRAIN features fit in available RAM. This cache accelerates training
only and is excluded from application throughput claims. No source clipping.

Preserve source/input fingerprints, actual terminal receipts, every training loss,
checkpoint/optimizer/scaler/RNG states, and sampler remainder. Verify frozen codec
and adapters, exact full state reload and hard outputs. Report complete TRAIN
source-only rollout plus shuffled-source controls as diagnostics. Do not treat
code agreement or training reconstruction as task accuracy.

Following successful lifecycle completion, evaluate the original source-disjoint
64 short +64 long development sources using unchanged 128-token semantic rubric
and qualified Terra/Sol judges. Retain all 3,539 inherited first judgments and
append new candidate judgments, never retry valid unfavorable decisions. Judge
Terra first; require at least61/64 passes in each cohort before Sol, then at least
61/64 same-case joint passes per cohort. Record actual denominators, invalids and
confidence intervals. If quality fails, preserve the failure, identify whether
root prediction, fine codes or rendering is responsible, and do not benchmark or
open the sealed320 final evaluation. Quality, final test and speed are separate gates.
