# Native NCM fine-tuning: real CUDA lifecycle, not task qualification

2026-10-24. Hypothesis `ncm-lm1b-native-finetuning-lifecycle-v1`.
Run only after the current fixed1024 model's complete128 source-only generation
worker has actually exited0, freeing physicalGPU0. The unchanged semantic review
continues on CPU/gateway. No current-study file or inference policy changes.

Warm the fixed final1024 checkpoint c1f1b7c3..., complete model682f0fe4... and
attention da4c0878.... No development scores select an initializer. Load the
pretrained native generator and task-domain decoder, then enable the separate
e64959d per-instance mechanism. Initialize no weights. Train127,732,736distinct
generator/word-decoder/source-attention parameters. Keep original Llama body
and MLPs unloaded/unchanged; native encoder80d53f9d... and VQ34e1825b... frozen.
Target-code geometry and cached690 labels remain exact.

Use only the two original first/longest TRAIN sources and their complete FP32
cached features. Require initial bitwise FP32 full concept/word-logit and hard
source-only generation parity before/after enabling. Save both full tensors.
Before any native update, also compare joint teacher logits with the native
stepwise generator while forcing the SAME gold TRAIN histories at every level.
Save all logits and forced codes; require allclose(rtol1e-4,atol1e-4), report
maximum absolute difference and every argmax disagreement. Always restore the
sampler override, including on failure. This is a TRAIN-only consistency probe,
not source-only prediction or task evidence. Abort updates if it fails.
Then two real BF16 updates, microbatch4 alternating these sources, AdamW1e-5,
betas.9/.95,eps1e-8,wd.01,clip1. Dropout112142, neighborhood111142. TF32off.
Use the complete clean/noisy/source-only-history word-gradient objective, accum1:
clean.5C+.7W,noisy.5C+.3W,self.25C+1W. Noise schedule5120+step, max.25,
per-case uniform, root protected; temperature1. Native generator and word decoder
train modes are active, while encoder/VQ stay eval. No teacher enters history
generation. Save full second-update gradients and both trace rows.

Require finite nonzero generator, learned-root-query, decoder and all four source
attention gradient groups; protected gradients absent, encoder/VQ hashes unchanged,
actual root weights changed and all protected target-code states unchanged.
Save full model/optimizer/CPU+CUDA+neighborRNG and complete source-only probes.
Construct a fresh author/domain model, re-enable mechanism, strict-load fullstate,
restore optimizer/allRNG and require bitwise hard-generation and state parity.
Peak allocated CUDA<=8GB with the planned nativeBF16 RTX3060. Preserve any failure.

Freeze producing source/plan/independent auditor before GPU launch. Require actual
tool exit, independent CPU replay of full saved parity/gradients/noise/optimizer,
registered systems result and backend exact readback. No semantic judgments or
canonical-cache changes, no final source access and no95%or speed claim. This
two-update pilot checkpoint is not a full-training initializer. Larger-data or
TRAIN-fit diagnostics need separate fixed protocols and all real task gates.
