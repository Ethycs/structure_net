# Train codec text decoding with the released representation frozen

Date: 2026-10-23. Exploratory hypothesis
`ncm-frozen-representation-decoder-transfer-v1`. The earlier joint codec
transfer failed: 67.167% oracle token recovery, versus released 95.179%.
Sample-scaled EMA restored 94.262% but still failed its 99.5% representation
gate. Its completed registry result is required before this follow-up.
The source-grounded semantic audit is running separately; its first four
adapted-codec reconstructions all failed, while the corrected uncompressed
control has 63/64 Terra passes. No final semantic aggregate is assumed here.

Question: can a decoder trained against the exact frozen full-code inference
distribution improve reconstruction without changing what is encoded? Start
from a fresh copy of the released original XSum codec, not either failed
adaptation. Freeze its encoder, normalization, codebooks, EMA buffers and
resampling modules. Keep these modules in evaluation mode. Train only the new
copy's text decoder, including its new feed-forward and output weights. Parent
Llama weights are absent and unchanged. Preserve all previous checkpoints.

For each 48-token fit window, compute the original target-derived codes and
their full nine-level latent reconstruction under no-grad, then apply the
trainable decoder. Optimize token CE ignoring PAD. Do not run the joint VQ
forward: no EMA update, pre-quantization dropout, fine-level dropout, VQ loss,
Gaussian corruption or source-conditioned prediction. Retain the decoder's
released .1 training dropout. This tests a distinct training recipe, not a
single-variable causal explanation of every difference from joint training.

Use the immutable 31,109-window corpus from 7,088 unpaired training texts and
1,024 groups. All 252 complete diagnostic targets and their source identities
remain excluded. Keep microbatch four, accumulation eight, exactly 2,048 updates,
model/dropout seed8142, independent sampler seed9142, AdamW beta=(.9,.95),
weight decay .01 and clip norm one. Warm up128updates to2e-5 and cosine decay
to zero at2048. Require every sampled window list and learning rate to match
the earlier 2048-update runs. Only the final checkpoint is selected. Save
recovery checkpoints at512/1024/1536, with no extra training budget.

Focused CPU tests must demonstrate real decoder gradients and changes,
unchanged encoder/quantizer/EMA state, unchanged codes and initial logits,
and absence of frozen parameters in AdamW. Then run the real two-update CUDA
pilot on physical GPU2 in FP32 with TF32 disabled, microbatch four and
accumulation one on fixed first/last windows. Probe the first and longest
complete target. Require finite loss, nonzero decoder gradients, no frozen
gradients, exact model/optimizer/CPU-CUDA-sampler RNG reload, unchanged frozen
representation and fine-dropout step, initial inference equal to the released
reference, and peak allocation below4.5GB. Full training starts fresh after
observed pilot terminal success. Stop on OOM, nonfiniteness or invariant drift.

After observed training terminal success, reconstruct all252 complete targets
with both original and final codecs, four48-token blocks each, without clipping
or per-block EOS insertion. Require original outputs to match the preceding
original reference exactly and every trained-arm code to equal its original
counterpart. Retain token recovery, exact texts, first EOS, invalid IDs and
changed-number counts. Keep the existing representation gate: token recovery
at least99.5%, first-EOS agreement at least99%, zero active invalid IDs and no
increase in changed-number cases. Exact text matching is descriptive. Any
semantic audit of this new producer requires its own prospective follow-up;
do not substitute token recovery for summary task accuracy.

Roots `20261023_frozen_representation_decoder_{pilot,train,reconstruction,results}_v1`
under `data/experiments/ncm_paper_regime`. Entry point
`python -m experiments.structure_net.ncm_frozen_representation_decoder
{pilot|train|measure}`. Hash source, plan, inputs, released vendor and all runtime
overrides. Aggregate, report and register with actual terminal receipts and
read-back. The HTTP semantic audit may run concurrently; no speed measurement
is made under contention. Both final task cohorts remain closed. The actual
source-only teacher remains56/64(87.5%); no NCM success is established here.
