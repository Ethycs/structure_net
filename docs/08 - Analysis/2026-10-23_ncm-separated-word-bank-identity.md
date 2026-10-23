# Independent word adapters preserve the native parent path

Date: 2026-10-23. Hypothesis `ncm-separated-word-bank-identity-v1`.
Classification `separated_word_bank_identity_passed_quality_unmeasured`.
Aggregate `data/experiments/ncm_paper_regime/20261023_separated_word_bank_results_v1/campaign_results.json`, SHA256 `8e0ac5d37c42d5001dc56c63ad3960319067842585d043cf7c77d27ff8b0ca4d`.

The separate word bank passed an exact identity and training-lifecycle control.
Before training, its zero-output rank-eight adapters produced exactly the same
logits as the original cached native language path with all adapters disabled.
Replacing wrappers preserved the trained five-level concept predictions and
their prior vectors exactly. This comparison uses the cached query path; it
does not claim identity to every full-prompt or incremental numerical kernel.

Two clean hard-target updates on the first two fixed summary TRAIN examples
trained only 4,587,520 new word-adapter parameters. All original 28 blocks,
MLPs, embeddings, native head, trained concept adapters/heads, old hint and
biases remained frozen. Word queries intentionally omit the old concept hint
and level bias. The shared original body is retained, with no extra full model
copy. Only word-bank A/B tensors appear in the saved trainable checkpoint.

CPU tests checked old-bank preservation, zero-word identity, gradient routing
and context restoration after exceptions. The real CUDA pilot passed finite
loss and nonzero word-gradient checks, unchanged old concept predictions,
body/books/cache hashes, and exact parameter/optimizer/RNG/output reload after
deliberate perturbation. Peak allocated GPU memory was
8,789,184,000 bytes. The worker exited normally.

This prepares an isolated language-quality teacher; it does not demonstrate
new summary accuracy, use of NCM concepts in word generation, compression or
speed. The measured converged hybrid remains **41/64 whole-summary passes**,
below the unchanged 3B baseline's 52/64. The original nine-pass model remains
0/64. No development or sealed cases were used by this two-case pilot.

The observed regression motivates separating clean language-quality learning
from concept/noisy-state gradients, but this control does not identify sharing
as the sole cause. A new fixed training protocol and representative long-input
memory pilot are still required. Learn and measure a faithful quality teacher
before distilling its own native fixed-point trajectories for acceleration.
All prior results remain unchanged; confirmed=false.
