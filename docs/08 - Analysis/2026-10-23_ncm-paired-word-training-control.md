# Paired word training: capacity control, not summary accuracy

Date: 2026-10-23. Hypothesis `ncm-paired-word-control-v1`.
Classification `oracle_word_training_capacity_failed`. Aggregate `data/experiments/ncm_paper_regime/20261023_paired_word_control_results_v1/campaign_results.json`,
SHA256 `ebe5e3bfb43e4b5fd15a422a6a06b0a0e6141b600dad89afed68cd7f5a147540`.

This control continued the failed 2,048-update checkpoint for exactly 128 word
updates on four fixed TRAIN summaries. It retained the frozen original
28-block Llama 3B body, MLPs, embeddings, word head and concept books.
Every update paired clean hard-target CE×10 with full-vocabulary consistency
on imperfect word states against the detached current clean branch. Correct
coarse plans were used during 16 warmup updates and half of later updates;
remaining updates used source-predicted plans. No new labels or API requests.

All results below are **training token recovery, not task accuracy**. Both
conditions use the same four parallel word refinements. Arm B reveals target
information and is unavailable to source-only inference.

| Plan condition | Before | After 128 updates |
| --- | --- | --- |
| A: source-predicted plan | 32/346 (9.25%) | 113/346 (32.66%) |
| B: oracle target-derived plan | 37/346 (10.69%) | 105/346 (30.35%) |

The fixed 90% oracle capacity gate **failed**.
Do not scale this recipe on the strength of lower losses; its decoding control failed.
The complete source-disjoint student task endpoint remains 0/64 whole-summary
passes. No development or final summaries were generated in this control,
and neither target-conditioned recovery nor a four-case fit can establish 95%.

Focused CPU checks verified causal divergence masking, full soft CE, detached
teacher gradients and sequential/joint gradient equivalence. The separate
four-update CUDA pilot covered summary/language cases and both plan types.
Both workers terminated successfully. Original body, books and selected cache
remained unchanged. Reload checks deliberately cleared optimizer state and
perturbed weights and RNGs before restoration; parameter values, optimizer
moments, RNG state and all probe outputs matched exactly. Peak allocated GPU
memory was 8,903,119,360 bytes. Sampling and trajectory RNG
streams and raw output token matches were replayed offline at finalization.

This is a combined training change inspired by JacobiForcing, not its original
full-model recipe or a replication of the unreleased NCM trainer. New query
adapters are shared with coarse prediction, so word training can also move
source-predicted plans; this control does not isolate those effects. No
generalization, compression, latency or throughput advantage is established.
