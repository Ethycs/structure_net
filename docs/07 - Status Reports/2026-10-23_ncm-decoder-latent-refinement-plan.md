# Refining teacher latents through the frozen NCM decoder

Date: 2026-10-23. Prospective exploratory hypothesis
`ncm-decoder-latent-refinement-v1`. This control asks whether optimizing input
latents can make the unchanged released decoder recover known TRAIN targets.
It tests a possible way to construct dense distillation labels while preserving
the learned decoder. It does not train a source-to-summary model.

The completed continuous-target control measured 3133/3197 correct tokens
(98.00%) without vector quantization, versus 2989/3197 (93.49%) with quantization,
on the same 32 TRAIN summaries. Both miss the existing 99.5% fitting gate.
The clean-word semantic diagnosis passed 2/32 with correct concepts and 1/32
with actual concepts. The separate paired dropout experiment is running on
physical GPU 0; leave its code, configuration and artifacts unchanged.

Use exactly the same 32 targets and original 16/additional 16 groups as those
controls. Each complete target includes EOS 50256 and right PAD 50257 in a
192-position frame. Retain four independent 48-position codec blocks, the
original encoder normalization, and the same frozen neural decoder. Decode
with vocabulary IDs 0..50256, excluding PAD and unused rows. Use the first
global EOS for text and retain all 192 raw IDs. No sealed source is accessed.
Bind the 32 target identities and complete words directly to the registered
continuous-control producer, measurements and 690-case TRAIN JSON. No source
features or forecaster checkpoint is used or rehashed for this target-only
control; their previous checks do not substitute for the actual target binding.

The paired control is the continuous encoder latent. For the intervention,
copy that latent into a trainable per-example tensor and optimize ONLY that
tensor, not model weights. Use 32 AdamW updates with learning rate .02,
betas (.9, .99), epsilon 1e-8, weight decay 0, and gradient clip 1. The objective
is mean nonpadding word cross entropy plus .001 times mean squared displacement
from the initial continuous latent across the whole 192-by-384 frame. Include EOS in
CE and ignore PAD. Use every fixed update; select only the final latent, with
no best-step selection, per-case reroll, clipping of text or decoded correction.
Reset latent and optimizer independently for each case. Model construction seed
43142 identifies the restored original weights; there is no random model
initialization, sampling or minibatch selection in this control.

Use physical GPU 2, logical cuda:0, FP32 weights/latents/forward/loss/AdamW, no
autocast, both TF32 modes off and math SDPA only. The original codec stays
eval with all parameter gradients disabled and no dropout. Require its full
state hash 50fc5680ce0fb3c82764d865243a2e509d8f16dd8c051113b16080b5d763d680
before and after computation. No parent Llama body or conditional generator is
loaded. Physical GPU 2 currently has more than 5 GB free; this control must peak
below 2,000,000,000 allocated bytes and never overlap another new control on that
GPU. Its concurrency pilot runs with existing resident processes and the
independent physical GPU 0 dropout study. No wall-time performance claim is made.

Before the full 32 cases, pass focused CPU tests of the frozen gradient path,
objective, active-token counting, raw output/latent replay, exact optimizer
reload, fixed budget and subgroup gates. The actual CUDA lifecycle/memory pilot
uses the first and longest selected TRAIN sources, two latent updates each at
the fixed maximum LR. Save and reload exact latents and optimizer states;
reload a fresh original codec and require identical raw logits readout digests,
IDs, texts and losses. The pilot is systems evidence only. Full cases start
again at the original encoder latents with fresh optimizers and 32 updates.
Their unoptimized baseline outputs must exactly retain the pilot baselines;
their optimized outputs need not match the two-update pilot.

The primary control gate is at least 199/200 correct active word tokens,
inclusive, separately for original 16, additional 16 and all 32. Report both arms,
exact complete target text counts, CE and latent displacement. Classify the
fixed final result as oracle_refined_latent_fit_passed or failed. A valid miss
is retained, not retried. Passing establishes teacher-target fitting on these
known cases only; it does not show that a student can predict those latents.
Any larger label construction or dense distillation needs a fresh prospective
experiment. This result cannot admit final evaluation or satisfy the 95% goal.

Use new roots 20261023_decoder_latent_refinement_{workflow,pilot,measure,results}_v1
under data/experiments/ncm_paper_regime. Commit source, tests and this plan before
launch. A supervisor owns the sequential pilot, measure, CPU audit and registry
stages and records actual subprocess.wait receipts. Preserve every raw output,
initial/final tensor, optimizer state, 32-update trace and failure. The audit
replays fixed identities, complete text/ID bindings, target hashes, rates,
optimizer steps/moments, latent digests/displacements, group counts/gates and
pilot baseline retention, with an immutable aggregate and exact registry
read-back. Keep task_accuracy null, confirmed false and both final cohorts
unopened. The full goal remains 95% complete-summary semantic accuracy on the
original source-disjoint evaluation, before any speed claim.
