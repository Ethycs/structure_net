# Expanded raw-text decoder-only continuation

2026-10-24. Hypothesis `ncm-expanded9936-FP16-decoder-only-continuation-v1`.
Prospective component repair; not source-only summary accuracy. Broaden the
word decoder that scored60/64 on the unchanged component cases. Do not change
the running FP16source-only trial or its fixed1024budget/checkpoint.

Require actual completed9936-source corpus preparation and independent CPU
replay. Exclude all64component source families and both permanent DEV cohorts;
sealed final320files remain unopened/unhashed. Start at the exact original
domain decoder SHA74d6bb3f... from checkpoint0a6b7d4d..., not a source-trained
or pilot decoder. Preserve the released encoder, native8192books/all641codes
and all other codec state. Parent body, parent MLPs and NCM generator are absent.
Only the existing18.87Mword decoder trains; no new random trainable weights.

Fixed1024updates, microbatch4, accumulation8, global32; AdamW(.9,.95),eps1e-8,
weight decay.01, norm clip1. Linear64update warmup to2e-5 then cosine to2e-6.
Sampler137142, augmentation138142, dropout139142. Each example chooses20percent
author TRAIN replay,80percent uniform raw source then uniform complete window.
As in prior generic repair, retain20percent original full frames; otherwise
take a uniformly sized lexical prefix and restore native CLS/PAD framing.
This is declared TRAIN augmentation, never output-summary clipping or repair.
All original raw-source lexical positions remain in the stored candidate pool.
Use unweighted word CE ignoring PAD0; encode complete target frames in FP32,
with the frozen encoder/VQ; decoder training uses FP16autocast and FP32CE.
No code corruption, fine-level dropout, new source features or teacher calls.

PhysicalGPU1RTX2060 only, TF32off, require2GBfree and cap peak allocated2GB.
FP32weights, scaler128/growthinterval2000, no skipped-update fallback; hard
readouts/reload in FP32. Two updates at the SAME full accumulation form the
systems pilot. Require finite gradients, unchanged protected state and exact
decoder/optimizer/scaler/allRNG/hard-output reload before full training. Reset
to original domain weights and original seeds for full1024, never pilot state.
Save all sample/augmentation descriptors, losses, gradients, scaler and LR.
Independent CPU replay binds every sample, step, protected state and checkpoint.

Fresh roots `20261024_expanded9936_codec_training_*_v1`. Select only the final
1024checkpoint. Whole training and CPU audit must actually exit0 before any
component quality claim. A separate frozen evaluation must use all64 unchanged
oracle-code packets, the latest completed judgment lineage, the existing
61/64same-case/both-judge gate and no semantic rerolls or repairs. Even a passing
codec component does not satisfy the95percent source-only summarizer goal.
