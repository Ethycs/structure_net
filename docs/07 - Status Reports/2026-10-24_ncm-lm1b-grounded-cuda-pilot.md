# Source attention in both native NCM stages: representative CUDA lifecycle

Date: 2026-10-24. Hypothesis `ncm-lm1b-dual-source-attention-BF16-lifecycle-v1`.
EXPLORATORY SYSTEMS ONLY. The best task-domain codec passed52/64, below its61/64
gate; this extension does not rescue that experiment or authorize a quality claim.

Add learned source cross-attention to all four existing384-wide word-decoder
blocks, alongside the tested768-wide source attention in the concept generator.
Use full detached3072-wide frozen-Llama source features. New word-memory K/V and
four Q/output adapters have3,538,944 weights; combined new attention has11,796,480.
Word output projections start at zero, seed90142; concept adapters start fresh
from their zero-output seed68142. The pretrained97M native generator and measured
domain codec569bf1ad... remain frozen/eval. Original Llama and MLPs are absent.

The learned word attention reads source evidence while decoding concepts; it is
not output repair, a vocabulary mask, retrieval lookup or a fallback. VQVAE.decode
is decorated no_grad, so only frozen code-to-latent conversion is under no_grad;
direct native decoder execution must retain gradients for NEW source attention.

Before a real run require the three focused CPU tests: exact zero-output parity,
new-only word gradients despite the codec API, padding invariance, attention-only
reload and label-free generation interface. Then one CUDA worker on physicalGPU0
RTX3060 with8GBfree, TF32off. Check exact FP32 native concept AND word-logit parity
on the same first/longest single-frame TRAIN systems probes as the earlier pilot.

Use a representative padded microbatch4 made by repeating the two complete
source cases (486 and2744 aligned feature positions). No source truncation. Run
two BF16-autocast updates with FP32 weights, AdamW1e-3 beta(.9,.95),weight_decay0,
clip1,seed91142. Loss is clean concept CE plus word CE ignoring PAD0. Require
finite gradients in both new attention branches and both source K/V projections
by update2, zero pretrained gradients and exact frozen-state hashes. The later
full training loss, including neighborhood corruption/self-history, needs its
own prospective protocol and lifecycle before a full campaign.

Save new attention/optimizer/CPU+CUDA RNG state only. A fresh strict pretrained
load plus attention restore must reproduce FP32 teacher logits and deterministic
label-free source-only concept and word generation exactly. Both source-memory
contexts must clear. Peak at most7.5GB. Store complete source/input hashes, trace,
checkpoint and actual terminal receipt in fresh
`20261024_lm1b_grounded_cuda_pilot_v1`; preserve any failure and use a new root for
changed producing code. This is no semantic,95% task, or speed claim. Full690
TRAIN and permanent128 DEV/320 sealed final scopes remain unchanged.
