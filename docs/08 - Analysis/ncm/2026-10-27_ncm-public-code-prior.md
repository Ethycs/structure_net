# Public code agreement against a source-blind baseline

Classification: `source_blind_mode_matches_or_exceeds_epoch2_code_agreement`. Hypothesis: `ncm-public-code-prior-v1`.
Aggregate: `data/experiments/ncm_paper_regime/20261027_public_code_prior_results_v1/campaign_results.json`; SHA256 `21e78374e6c3d2e5bd99e4ff3b02eb6cd3017bf3edaffdc0eae71598b4447e5f`.

| Predictor | Code agreement | Agreement on nonmodal reference codes | Nonmodal correct / positions |
|---|---:|---:|---:|
| positional_mode | 34.60% | 0.00% | 0 / 13415 |
| BROAD_matched | 11.04% | 6.16% | 826 / 13415 |
| BROAD_shuffled | 11.00% | 6.34% | 851 / 13415 |
| epoch2_matched | 32.85% | 18.53% | 2486 / 13415 |
| epoch2_shuffled | 37.25% | 21.94% | 2943 / 13415 |

| Level | Code positions | Mean positional entropy (bits) | Mean occupied codes | Mean training mode share |
|---:|---:|---:|---:|---:|
| 0 | 1 | 3.184 | 950.0 | 63.90% |
| 1 | 8 | 3.385 | 1352.0 | 42.05% |
| 2 | 16 | 4.472 | 2192.9 | 29.20% |
| 3 | 24 | 3.222 | 2246.8 | 68.77% |
| 4 | 32 | 3.918 | 2193.4 | 39.96% |
| 5 | 40 | 4.006 | 2090.8 | 32.34% |
| 6 | 52 | 4.231 | 1951.1 | 31.33% |
| 7 | 64 | 4.143 | 1847.3 | 29.87% |
| 8 | 76 | 4.363 | 1700.8 | 26.96% |
| 9 | 92 | 4.865 | 1619.4 | 25.41% |
| 10 | 108 | 4.396 | 1654.9 | 30.87% |
| 11 | 128 | 3.600 | 1596.0 | 44.77% |

The baseline uses the most frequent code separately at each of 641 positions, breaking ties by the lowest code ID. It fits all 162,321 admitted public TRAIN examples outside the fixed 32-case probe. Source content is not an input to its predictions, and it runs no neural model. The model predictions are the exact saved BROAD and epoch2 matched/shuffled outputs from that probe.
Nonmodal agreement counts only reference positions whose code differs from the fitted positional mode. Its denominator is fixed across predictors. These are exact-code diagnostics, not semantic summary scores. A modal hierarchy assembled from different examples need not decode to coherent text. Frequent codes cannot automatically be identified with padding, and positional entropy does not measure semantic information.
This exploratory follow-up was specified after observing epoch2 matched/shuffled agreement of 32.85%/37.25%. All labels and TRAIN index bindings were checked. The 256-record lifecycle pilot reproduced in the full pass. An independent full pass using Python Counters reproduced every NumPy count, then replayed all reported measurements. It reuses the existing 32-case TRAIN artifact but opens no new source corpus, source feature matrices, DEV or final-test payloads.
The ongoing public curriculum and matched application-data training studies remained unchanged. No GPU, model training, new semantic judgments or speed measurement was involved. This result cannot establish that more data or a particular new loss will solve the task. Application semantic DEV remains 0/128; the 95% complete-summary goal is unqualified.
