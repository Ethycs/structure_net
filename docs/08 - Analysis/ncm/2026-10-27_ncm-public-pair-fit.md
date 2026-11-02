# Fit on public summarization training pairs

Classification: `public_TRAIN_rollout_improved`. Hypothesis: `ncm-public-pair-fit-probe-v1`.
Aggregate: `data/experiments/ncm_paper_regime/20261027_public_pair_fit_results_v1/campaign_results.json`; SHA256 `7bc97c74054179c0c134d958a34e80f7ac42fb790d3fce181ed6bbdd1f8ad286`.

| Checkpoint | Source | Teacher code agreement | Generated code agreement | Native word agreement |
|---|---|---:|---:|---:|
| BROAD | matched | 76.78% | 11.04% | 4.76% |
| BROAD | shuffled | 76.23% | 11.00% | 4.43% |
| epoch2 | matched | 87.18% | 32.85% | 7.68% |
| epoch2 | shuffled | 86.87% | 37.25% | 6.49% |

At epoch2, matched minus shuffled source agreement is -4.40 percentage points for generated codes and +1.19 points for native words. These reference comparisons do not establish reliable source-grounded summarization or identify insufficient data as the cause of failure.

The sample contains 32 public TRAIN pairs selected by a fixed hash from all 162,353 admitted pairs before inspecting outputs. Each public pair had two presentations by epoch2. The earlier 16-case TRAIN result instead measured frequently repeated application anchors. These populations differ, so their score difference does not isolate repetition as a cause.
Each checkpoint generated from complete source features alone. Teacher forcing was a separate diagnostic. Shuffled conditions cyclically permuted sources while retaining targets. The two-case pilot included the longest selected source, repeated exactly under both checkpoints, and reproduced in the full run. CPU replay checked selection, source pairing, raw counts and rendered words.
Exact reference code or word agreement is not semantic summary accuracy: valid alternative summaries may disagree, and public references are not individually verified factual gold. This small training sample cannot establish generalization, prove that additional training will solve the failure, or qualify the 95% goal. Application semantic DEV remains 0/128 at the last completed evaluation.
Locked Pixi, physical GPU2 RTX2060SUPER, FP32 inference, TF32 off; one source feature matrix streamed at a time. Model tensors and both ongoing training service identities remained unchanged. No gradients, model updates, new judges, final-test access or speed measurement occurred.
