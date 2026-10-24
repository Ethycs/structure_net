# Full690 word-gradient continuation from the fixed4096 model

Date: 2026-10-24. Hypothesis `ncm-lm1b-full690-word-gradient-neighborhood-v1`.
Exploratory continuation after both greedy and native-word-readout policies
scored0/128 under the qualified semantic judge. Those failures remain unchanged.
The systems pilot independently proved bitwise hard-forward equality, new
word-loss gradients into concept attention, two BF16 updates and exact reload;
registered systems campaign SHA18bdccb3.... This is not task qualification.

Use all690 unchanged targets, native641-code caches and full FP32 source features
from the completed study. Keep all source/session/group/hash exclusions against
both permanent64-case DEV cohorts. The final320 sources and old final selections
stay closed and unhashed. No target clipping, exclusions, reweighting or new labels.
Potential label noise remains acknowledged; no retroactive label/codec gate pass.

Warm-start the complete attention state from the original final4096checkpoint
SHA bb239a48..., attention SHA38af3e58.... Never load a two-update pilot as the
training initializer. No new trainable parameters are initialized. Original
Llama/body/MLP weights remain unloaded, and the released generator and domain
codec stay frozen. Only the existing11,796,480 source-attention parameters update.
Reset optimizer and random streams. All hard inference code stays unchanged.

Before full training, run a fresh complete-data-path two-update pilot using the
first/longest TRAIN sources at microbatch4, maximal noise and source-only history.
Require exact attention/optimizer/allRNG/source-only hard-generation reload,
all expected gradient groups active, frozen states unchanged and peak<=7.5GB.
The main run resets again to the fixed4096base, not that pilot's updated weights.

Full run: physicalGPU0 RTX3060, FP32weights/BF16training, TF32off, microbatch4,
accumulation4, fixed1024updates. AdamW(.9,.95),eps1e-8,weight_decay.01,clip1.
Warm64updates to1e-4, cosine to2e-5. Sampler106142, neighborhood107142,
dropout108142. Uniform sampling with replacement over all690cases. Select only
final1024; no early stopping, recovery/model selection or DEV look during training.

Clean/noisy losses remain .5concept+.7word and .5concept+.3word, divided by
accumulation. Corruption uses schedule4096+currentstep, hence maximum.25;
per-case rate uniform0to.25, root protected. On first microbatch EVERY update,
generate own history from features/mask only in FP32/eval, then restore train.
Add .25self-conceptCE plus1.0self-wordCE, unscaled by accumulation. Word loss
uses hard+(soft-soft.detach()) through the native codebooks at temperature1.
This combines changed routing, word weight and history frequency; do not attribute
an eventual quality change to just one factor. Keep every sampled ID, generated
history, noise proof, finite loss and gradient norm. Cached source payload remains
bounded8GB with12GBhost memory available; no source truncation.

Fresh `20261024_lm1b_word_gradient_training_{pilot,train,workflow}_v1` roots.
Freeze source/tests/plan before launch. Require exact fresh final checkpoint and
optimizer/allRNG/hard-source-only reload plus actual worker/whole-tool exits.
Training ends accuracy PENDING. Mandatory next stage generates all128 unchanged
source-only DEV outputs, FP32, with no teacher words/codes, then the qualified
128-token semantic rubric. Inherit all latest2728 canonical first judgments,
including both failed policies; no forks, rerolls or retries of exhausted invalids.
Terra scores all128; Sonnet only if each64 has64valid and>=61passes. Need>=61of
the SAME64 passing BOTH judges in EACH cohort. Only an audited/registered full
DEV pass can admit sealed final320. No overall goal completion without actual
final95%task evidence, and no speed claim from training or systems checks.
