# Fixed longer optimization of NCM teacher latents

Date: 2026-10-23. Hypothesis `ncm-decoder-latent-convergence-v1`.
Classification **oracle_latent_convergence_fit_passed**. Campaign `data/experiments/ncm_paper_regime/20261023_decoder_latent_convergence_results_v1/campaign_results.json`;
SHA256 `1a5587221af8048f3052296bb17d6c6ab40afdf7e62d689088b7b0061070a834`.

| Same TRAIN targets | Continuous encoder | 32 latent updates | 256 total latent updates |
| --- | --- | --- | --- |
| original_16 | 1592/1635 = 97.37% | 1616/1635 = 98.84% | 1635/1635 = 100.00% |
| additional_16 | 1541/1562 = 98.66% | 1554/1562 = 99.49% | 1562/1562 = 100.00% |
| all_32 | 3133/3197 = 98.00% | 3170/3197 = 99.16% | 3197/3197 = 100.00% |

This exploratory follow-up was specified after the32-step fitting miss.
It continues each exact saved latent and AdamW state for224 further updates,
without resetting moments or changing the learning rate, objective, targets or
frozen original decoder. Only the fixed final256-step tensors are selected.
The original32-step failure remains unchanged. The two-case systems pilot adds
two updates and does not supply a quality estimate or initialize the full run.

The audit replayed the complete inherited32-step traces and every added target
hash, rate, finite loss and gradient, full latent and optimizer state, raw
192-word decoding/counts and all subgroup gates. Fresh decoder/optimizer reloads
were exact and the same-GPU baseline and32-step readouts stayed identical.

These are target-aware TRAIN reconstruction measurements. They cannot establish
source-only summary accuracy or a speed gain. Passing the99.5% fitting gate in
each group supports a new dense-target distillation experiment; it does not
admit any student training automatically. The95% source-disjoint complete-summary
goal remains unmet and both final cohorts remain unopened.
