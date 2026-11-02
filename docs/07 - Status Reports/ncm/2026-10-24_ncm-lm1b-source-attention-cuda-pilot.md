# Native released NCM source-attention CUDA lifecycle

Date: 2026-10-24. Hypothesis `ncm-lm1b-source-attention-native-CUDA-lifecycle-v1`.
SYSTEMS ONLY; explicitly exploratory and not a downstream quality claim. The
native codec representation has not qualified, so this does not authorize a
source-summary accuracy claim or a full conditional training campaign.

The existing zero-output source-attention module passed three toy CPU tests but
has not executed a real native forward/backward. Test the exact released LM1B
97M NCP and 111M codec, preserving native CLS prefix, learned level-zero query,
post-upsampling convolutions and shared output head. Keep both modules frozen
and in eval mode. The original Llama body is absent; use SHA-verified existing
full 3072-wide frozen-body TRAIN features.

Use physical GPU0 RTX3060, FP32 with TF32 off, one worker, at least 8 GB free,
peak at most 8 GB. GPU2 may independently train the task-domain decoder; shared
wall time is not a benchmark. New parameters are exactly 8,257,536: shared source
K/V projection and query/output attention at native blocks 8,10,11. Constructor
seed 68142; output projections start at zero. Baseline native full641 concept
logits must match attached zero-output logits exactly for first and longest
source TRAIN cases that fit one native frame. This capacity selection is only
for the systems pilot and cannot change later task evaluation denominators.

Execute exactly two AdamW1e-3 updates, beta(.9,.95), weight decay0, clip1, one
case per update. Dropout seed88142. Require finite loss/gradients, nonzero new
output gradients and shared source K/V gradients by update2, no frozen-module
gradients or state changes. Save only new adapter/optimizer/RNG state. A fresh
released-model reload must exactly reproduce teacher logits and deterministic
source-only code generation. Source memory must clear after every request.

Use fresh `20261024_lm1b_source_attention_cuda_pilot_v1`, snapshot source/data/
checkpoint hashes and actual tool terminal result. Stop and preserve any error;
fixes require new wrappers/roots. This is not semantic evaluation, not a fitted
summarizer, not a 95% or speed result. The original source-only DEV0/128 and all
codec failures remain visible. All final sources remain closed.
