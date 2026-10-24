# Source only NCM fitting with dense latent targets

Date: 2026-10-23. Hypothesis `ncm-dense-latent-source-word-fit-v1`.
Classification **dense_source_TRAIN_fit_failed**. Campaign `data/experiments/ncm_paper_regime/20261023_dense_word_fit_results_v1/campaign_results.json`;
SHA256 `1561dd9359a17438993d3b6700f69b3619771525ec465815a896bd1b0eb77d67`.

| TRAIN group | Actual history token agreement | Exact target texts |
| --- | --- | --- |
| original_16 | 1626/1635 = 99.45% | 10/16 |
| additional_16 | 1560/1562 = 99.87% | 14/16 |
| all_32 | 3186/3197 = 99.66% | 24/32 |

The student trained on these same32 TRAIN cases for2048 fixed updates. Predictions
receive complete frozen-Llama source features and source-only concept histories
from the unchanged forecaster. Correct target concepts, target words and refined
teacher latents never enter prediction inputs. Labels supervise full-frame
latent MSE plus0.1 nonpadding word CE. The original decoder was restored and
frozen, source/transformer/projection dropout was zero, and only the existing
source norm/projection, NCM word transformer and word projection trained.

The initializer retains learned clean-word weights outside the restored decoder.
The two-update maximum-LR/maximum-frame CUDA pilot was not retained in full
training. FP32 masters, optimizer, decoder and loss combine with BF16 source and
transformer computation; both TF32 modes are off. Body/MLPs, codec, forecaster,
original decoder and unused coarse heads stay frozen. Native cache parity,
exact save/reload, all2048 sample/target/teacher/rate/optimizer records, original
frozen parameter bytes and raw192-word outputs replayed.

This is a one-seed training-fit diagnostic. The unchanged99.5% token gate applies
separately to each16-case group and their32-case union, using actual predicted
histories. Passing cannot establish unseen summary accuracy or generalization.
Any full690-case latent construction/training needs a separate prospective plan.
The95% complete-summary goal remains unmet; both final cohorts remain unopened.
No speed or compression improvement is established by these measurements.
