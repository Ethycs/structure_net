# Word-loss gradient through native NCM codebooks: systems pilot

Date: 2026-10-24. Hypothesis `ncm-lm1b-word-gradient-lifecycle-v1`.
Systems validation only; task accuracy is unmeasured and the95%goal unchanged.
The completed fixed4096source-only DEV run scored0/128, all valid. The native
word-readout control is independently running and its result stays preserved.

The old generated-word loss updates word-source attention but cannot update
concept-source attention through hard code IDs. Test a targeted surrogate:
compute the code-probability-weighted embeddings at each native level, using
the unchanged native upsampling/residual operations. Form hard.detach() plus
(soft-soft.detach()). Parentheses make the forward correction exactly zero;
only backward follows the soft path. Temperature1. This is a biased training
surrogate, not a claim of exact gradients or a replication of paper training.

Use only the original first/longest TRAIN cases, already fixed in the training
selection. All690 TRAIN cases remain excluded from the permanent128DEV by the
existing source-family proof. DEV source metadata is read only to replay that
exclusion; no DEV/final example enters model inputs or gradients, and no teacher
or judge is called. Load the exact final4096attention checkpoint bb239a48... over the
released frozen generator60316... and task-domain codec569bf.... All original
Llama/body/MLP weights remain unloaded and unchanged. Only the existing
11,796,480 new attention parameters may receive optimizer updates.

First verify on a real FP32 eval-mode batch2 of both full sources that hard
word loss has no concept-attention gradient, then that the surrogate produces
BITWISE IDENTICAL complete word logits while supplying finite nonzero gradients
to both concept-source memory/adapters and word-source memory/adapters. No
optimizer update in this diagnostic. Save both complete logit tensors and
gradient evidence for an independent CPU audit. No source truncation.

Then run two actual BF16 updates with FP32 parameters on physical GPU0 RTX3060,
TF32off, microbatch4 alternating those two sources. AdamW1e-4,(.9,.95),eps1e-8,
weight_decay.01,clip1. Use the prior clean/noisy objective with maximal1024
corruption schedule, and source-only FP32/eval generated history on each update.
The added branch is .25conceptCE plus1.0wordCE through the surrogate, unscaled
by accumulation. No teacher codes enter history generation. This combined
recipe changes auxiliary weight/frequency as well as gradient routing; an
eventual quality gain cannot be attributed to one component without controls.

Separate dropout104142 and neighborhood105142 RNGs. Verify exact fresh
attention/optimizer/allRNG/source-only hard-generation reload after two updates.
Require all pretrained states unchanged, all expected gradient groups active,
finite losses/norms, clean contexts, and peak allocated<=7.5GB. Fresh root
`20261024_lm1b_word_gradient_cuda_pilot_v1`. Freeze source/tests/plan before CUDA.
Save actual exit receipts and independently audit/report/register conservatively.
Never use the two-update pilot checkpoint as the next full-training initializer;
any later training starts again from the fixed4096base under its own protocol.
