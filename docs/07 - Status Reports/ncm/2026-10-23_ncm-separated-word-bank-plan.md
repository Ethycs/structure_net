# Isolate the native language adapter before training it again

Date: 2026-10-23. Exploratory systems hypothesis:
`ncm-separated-word-bank-identity-v1`.

The converged native adaptation scored 41/64 complete summaries, including
42/64 faithful, 62/64 salient and 62/64 readable. The unchanged 3B baseline
previously scored 52/64. Wrong list counts, cross-item attribute binding,
broadened scope and lost modality dominate the failed claims. Convergence
does not fix that regression. The earlier 128-update paired-loss control
also failed; do not extend or select it.

The current adapter is shared by concept classification, clean native-word
training and noisy native-word correction. Test a bank split without changing
any original parameter: retain the trained concept bank, concept heads, hint
and level biases exactly, freeze them, and add an independent rank-eight word
Q/K/V/O bank. Its A matrices are initialized with seed 5142; B matrices are
zero. All original 28 body blocks, MLPs, embeddings and native head remain
frozen and shared. This is a preparation control for a quality teacher, not
a new summary-quality result or a claim that parameter sharing caused every
failure.

The word-quality path uses native previous words and the original source-only
cache, with the independent word bank. It adds no concept hint or level bias.
Before training, require exact logits against the original cached native
language path with all adapters disabled, on the first two fixed summary TRAIN
cases. Require exact preservation of all five concept predictions before and
after replacing wrappers. Decoder-bank context must restore even on exceptions.

First run CPU tests for exact old-bank preservation, zero-word identity,
context restoration, and gradient routing. Then run a real CUDA lifecycle
pilot on those same two training cases: two paired-case clean hard-target CE
updates, AdamW 1e-4, weight decay .01, clip norm one, batch two, no accumulation,
no noise, teacher API, development cases or test cases. Only new word-bank
parameters may receive gradients. Require nonzero word gradients, finite
outputs, unchanged old bank/body/heads/books and CPU cache, unchanged concept
predictions, and exact word-parameter/optimizer/RNG/output reload after
deliberate perturbation. Use physical GPU0, BF16 body/FP32 new parameters,
TF32 disabled. Preserve failures and use a fresh version for any correction.

Root `20261023_separated_word_bank_pilot_v1`. Freeze the original 2048-update
checkpoint, source/cache entries, helper sources and this plan. This does not
admit a larger training run or replace the 41/64 endpoint. A subsequent clean
summary-quality training study needs a fixed protocol and full unseen-summary
grading before any NCM acceleration distillation. Such distillation must use
the quality teacher's own native fixed-point trajectories, rather than assume
arbitrary external reference wording is already that model's fixed point.
Both final cohorts remain sealed; no speed or compression benchmark here.
