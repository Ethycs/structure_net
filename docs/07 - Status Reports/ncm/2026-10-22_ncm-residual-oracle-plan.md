# Can finer NCM codes repair a wrong coarse prediction?

Date:2026-10-22. EXPLORATORY `ncm-residual-oracle-v1`, a representation diagnostic.
The matched decoding policies all scored0/64under the secondary judge. The
revised full-data generator is still training. The NCM paper keeps clean code
labels under conditioning corruption, and identifies model-error-conditioned
corruption as future work (Section3.4andAppendixF.3):
https://arxiv.org/html/2610.14162v1#A6.SS3.

Question: after changing early codes, can the frozen quantizer choose different
finer residual codes that restore the original text? This tests whether a
future generator could learn to compensate for early errors. The oracle has
the actual target text/latent, so its reconstruction is emphatically not model
task accuracy or proof of source-to-summary generalization. No training occurs.
Do not call this dynamic reencoding the paper's published training recipe.

Use the polished codec and all1,999existing training targets. Preserve every
codec weight/buffer. Encode target latents inFP32. Compare these paired arms:

1. Clean original hierarchy.
2. Replace one uniformly chosen coarse position with a uniformly chosen member
   of its8nearest distinct, non-identical codebook vectors; replay clean fines.
3. Same changed coarse codes, then greedily requantize all finer residuals from
   the actual target latent while holding those coarse codes fixed.
4. Replace all4coarse positions with independently sampled8-neighbor codes;
   replay clean fines.
5. Same4changed coarse codes, then requantize finer residuals as in3.

Compute neighborhoods in the original coarse embedding's squared Euclidean
metric usingFP64neighbor search to avoid cancellation; exclude self and
distance<=1e-8. Quantizer assignment remainsFP32. FixedCPUaugmentation seed3042. Never
choose the perturbation or outcomes by quality. Each pair shares identical
changed coarse codes. Match the vendor quantizer's FP32subtraction, resampling
and residual-projection order. First require clean assignment and latent parity
with the vendor on every evaluated batch. Reencoding must preserve every forced
coarse code. Decode all arms with identicalFP16forward/rawargmax and local
EOS-or-PADrendering; the paired comparison holds decoding fixed.

Use one worker on physicalGPU2, preserving resident services and the active
GPU0training run. FP32codec parameters/quantization, FP16decoder, batch32.
A separate representative32-case lifecycle pilot must pass assignment/latent
parity, unchanged codec, finite output and reload before the full diagnostic.
Stop on mismatch, OOM, changed dependency or nonfinite output; no CPU fallback.

Report integer non-PADtoken counts(includingEOS), exact text counts and every
output/code assignment for all5arms. Clean reconstruction is a same-profile
control, not an assumed reproduction of the earlierBF16integer counts. Report
per-arm and paired error recovery. This study is measurement-only; no model
quality/95%gate is awarded. A useful result must still lead to a separately
specified source-conditioned training experiment and held-out whole-summary
evaluation. All known semantic label errors and earlier failures stay visible.

Artifacts:`20261022_residual_oracle_pilot_v1` and
`20261022_residual_oracle_v1` under`data/experiments/ncm_paper_regime`.
Freeze sources/config/input hashes and retain checkpoints, sampled perturbations,
raw outputs, code arrays, memory and exact-code/latent parity receipts. The
final256test remains sealed; there is no inference-speed claim.
