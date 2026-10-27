# Correct early codes, then generate the remaining hierarchy

Classification: `early_prefix_rescue_not_established`. This intervention supplies known target codes; it does not measure source-only summarization accuracy.

| Correct codes supplied | Remaining-code agreement | Content-token agreement | Exact native texts | Digit sequences preserved |
|---:|---:|---:|---:|---:|
| 0 | 11.89% | 5.85% | 0/64 | 0/18 |
| 1 | 15.92% | 5.96% | 0/64 | 0/18 |
| 9 | 18.37% | 6.62% | 0/64 | 0/18 |
| 25 | 24.22% | 8.15% | 0/64 | 0/18 |
| 641 | No generation remains | 99.15% | 44/64 | 16/18 |

Changes versus the uncorrected baseline on the same remaining positions:

| Correct codes supplied | Remaining-code gain | Content-token gain | Joint descriptive gate |
|---:|---:|---:|---|
| 1 | 4.02 pp | 0.11 pp | False |
| 9 | 6.74 pp | 0.78 pp | False |
| 25 | 12.78 pp | 2.31 pp | False |

The fixed joint gate requires at least10 percentage points of remaining-code improvement and20 percentage points of content-token improvement with at most25 injected codes. Injected positions never count as generated successes. The all641-code reconstruction control cannot pass this gate.
All64 previously reserved and inspected component sources were used in their original order, with the same epoch2 checkpoint and full original Llama features. Correct prefixes contain1,9 or25 codes at complete level boundaries; later target codes never enter that generation call. Different valid summaries can have different codes, so these exact comparisons are component diagnostics.
Every no-injection output exactly reproduced the prior source-only codes, words and text. Every all641-code output exactly reproduced the repaired-parent source-attention codec control. The two-case pilot repeated all conditions exactly and matched the full run. An independent CPU audit re-rendered raw words and replayed all320 outputs, metrics, controls and gates.
The run used locked Pixi ncm, physicalGPU0 RTX3060, FP32, TF32 off and a256MiB feature cache within4.9GB peak allocation. Full model hashes remained unchanged; no gradients, training, new semantic judgments, final-test access or speed measurement occurred. The epoch4 trainer continued unchanged on the RTX2060.
Application semantic DEV remains0/128. Supplying correct codes does not show that a learned model can predict them, and this experiment cannot qualify the95% goal.
Raw evidence: `data/experiments/ncm_paper_regime/20261027_oracle_prefix_run_v1`.
