# Dense NCM word training on all approved summaries

Date: 2026-10-24. Exploratory hypothesis `ncm-full-population-dense-word-v1`.
Classification **full690_dense_word_TRAIN_fit_passed_semantics_unmeasured**. Campaign `data/experiments/ncm_paper_regime/20261024_full_population_dense_word_results_v1/campaign_results.json`;
SHA256 `af020507480aa1dc20735fcb69a6650121db916c58ca803a8252ea5568248fab`.

| Fixed TRAIN probes | Correct active words | Exact target texts |
| --- | --- | --- |
| original_16 | 1628/1635 | 10/16 |
| additional_16 | 1562/1562 | 16/16 |
| all_32 | 3190/3197 | 26/32 |

The word branch continued the exact learned2048-update dense initializer for
32768 fixed updates on all690 approved TRAIN cases. Source-only histories came
from the new final matched forecaster; target words and optimized latent labels
entered only latent MSE plus0.1 nonpadding word CE. The original Llama body and
MLPs, forecaster, codec, decoder and unused heads stayed frozen. FP32 masters,
optimizer, decoder and losses combined with BF16 source/transformer computation,
zero dropout and disabled TF32. Only the fixed final checkpoint was selected.

Maximum-frame/maximum-LR gradients, actual native cache parity and exact model,
optimizer, sampler and RNG reload preceded full training. Frozen bytes and the
8GB allocation limit held. CPU replay checked every selected source, actual
history, target word and teacher latent hash, frame, learning rate, weighted
objective, optimizer step and complete sampler coverage. All fixed recovery
checkpoints and raw192-word diagnostic predictions remain available.

The original32-case parent fitting and semantic failures remain unchanged.
This continuation changes both training population and forecaster; it does not
isolate either cause. The99.5% token gate is a TRAIN diagnostic, with no unseen
accuracy interpretation. Fresh complete-summary semantic evaluation on all128
development cases remains required for this fixed candidate. All sealed final
sources stay closed until eligible. The95% task goal remains unmet, and no speed
or compression gain is established by this study.
