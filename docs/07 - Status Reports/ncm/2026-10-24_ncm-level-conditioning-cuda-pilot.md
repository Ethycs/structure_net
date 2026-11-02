# Revised level-conditioning objective: spare-GPU systems pilot

2026-10-24. Original native655checkpoint failed0/128held-out complete summaries.
The new noise helper is untrained. This is a TWOUPDATEsystems check, not another
quality experiment or permission to claim95%accuracy. All prior failures and
canonical3222firsts remain unchanged. Final320sources stay closed/unhashed.

Use physicalGPU2 RTX2060SUPER, leaving GPU0's frozen-body feature-cache job
running. FP32weights/FP16autocast training, FP32deterministic source-only hard
generation,TF32off. This is a separate precision/hardware systems variant; it
does not substitute for a representative GPU0/BF16pilot before a larger quality
run. Require5GBfree and peak allocated<=4.9GB. No silent precision, scale,
microbatch, dataset or memory-limit changes after a failure.

Reset the unchanged original b4b901f... native127,732,736parameter initializer
(released generator, domain word decoder, original warm source attention),
never the native16overfit, native655failure or pilot weights. Original Llama
body stays unloaded; cached original full FP32source features are detached.
Native encoder/VQ remain frozen with their established hashes.

Use the original first/longest distinct native655TRAIN sources and their real
paired128word/641concept targets. Microbatch1,accumulation2, one source each.
Right-pad to3777source positions with invalid padding masked; never truncate
the actual source. This matches the new corpus's maximum source tensor length
as a memory systems check without fabricating a new source/target pair.

Two AdamWsteps1e-5,.9/.95,eps1e-8,wd.01,unscaledclip1. FP16GradScalerinitial128,
growthinterval2000; nonfinite unscaled gradients fail without retries. Separate
neighbor124142 and dropout125142streams. Clean.5C+.7W and noisy.5C+.3W divided
by accumulation. Self-history once per update on first microbatch: FP32source-only
rollout then.25C+1W through hard-forward/soft-backward word surrogate, unscaled
by accumulation, as in native655. The ONLYnoise helper here is the new fixed
per-level.75->.25same-codebook replacement, including the first GENERATEDcode;
the clean targets never change and BOS/nullprefix is not in that tensor.
These endpoints are our explicit candidate choice inspired by the paper's
ablation, not a released exact LM1Btraining recipe.

Require finite active gradients in native root/generator/word decoder and all
source-attention groups, actual parameter updates, no protected codec gradients
or hash changes. Save and strictly reload the entire FP32model, AdamWstate,
FP16scaler and allCPU/CUDA/neighbor RNGstates. Two raw source-only hard outputs
must reload exactly on SAMEGPU2. Record actual tool exit, full trace and all
hashes in fresh append-only root
`data/experiments/ncm_paper_regime/20261024_level_conditioning_FP16_GPU2_pilot_v1`.
This pilot is never reused as a future quality-training initializer. No semantic
judging, source-disjoint task measurement, speed measurement or full-paper
replication claim is admitted by these two updates.
