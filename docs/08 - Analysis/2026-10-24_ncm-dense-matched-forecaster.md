# Concept forecasting trained on the complete dense target population

Date: 2026-10-24. Exploratory hypothesis `ncm-dense-matched-forecaster-v2`.
Classification **matched_dense_forecaster_coarse_fit_passed_word_quality_unmeasured**. Campaign `data/experiments/ncm_paper_regime/20261024_dense_matched_forecaster_results_v2/campaign_results.json`;
SHA256 `5424f6c185ab3ebd2f113be242b17d14ba5f77746a3ec4a3f471f655a11dfa77`.

| Fixed TRAIN probes | Coarse level | Correct active codes |
| --- | --- | --- |
| all_32 | 0 | 336/336 |
| all_32 | 1 | 504/504 |
| all_32 | 2 | 840/840 |
| all_32 | 3 | 1175/1176 |
| all_32 | 4 | 1677/1680 |
| all_32 | 5 | 2181/2184 |
| all_32 | 6 | 2679/2688 |
| all_32 | 7 | 3351/3360 |
| original_16 | 0 | 176/176 |
| original_16 | 1 | 264/264 |
| original_16 | 2 | 440/440 |
| original_16 | 3 | 616/616 |
| original_16 | 4 | 879/880 |
| original_16 | 5 | 1143/1144 |
| original_16 | 6 | 1405/1408 |
| original_16 | 7 | 1759/1760 |
| additional_16 | 0 | 160/160 |
| additional_16 | 1 | 240/240 |
| additional_16 | 2 | 400/400 |
| additional_16 | 3 | 559/560 |
| additional_16 | 4 | 798/800 |
| additional_16 | 5 | 1038/1040 |
| additional_16 | 6 | 1274/1280 |
| additional_16 | 7 | 1592/1600 |

The new forecaster continued the exact learned252-case initializer for32768
fixed updates on all690 approved TRAIN summaries, with a fresh optimizer.
The earlier conditional forecaster remains declined. This exploratory study
instead followed qualified full690 dense target construction. The original
Llama body/MLPs, codec, word modules and finest head remained frozen. Source
features retained every token. FP32 masters/optimizer/loss and reconstruction,
BF16 forward, original dropout and disabled TF32 remained fixed. Uniform history
corruption and weighted active/padded concept losses follow the existing recipe.
Only the fixed final checkpoint was selected.

Maximum-frame/max-LR gradients, exact model/optimizer/RNG/native reload and
source-only native-generation pilots preceded full training. The CPU audit
replayed all samples, histories, target hashes, masks, rates, weighted losses,
optimizer steps and sampler/noise states, preserved snapshots and frozen bytes.
All690 fresh native-frame source-only concept histories were cached; the fixed
TRAIN probes agree exactly with separate fitting outputs.

The99% active-code gate per level and TRAIN group is a component diagnostic.
It establishes no summary accuracy and does not automatically admit downstream
training. Dense word adaptation needs a separate fixed protocol using these
actual histories and qualified targets, followed by fresh source-disjoint
complete-summary evaluation. The last student's raw development scores were
0/64 short and1/64 long, with the lone pass from a TRAIN-overlapping passage.
The95% task goal remains unmet; sealed final sources stay closed. No speed
improvement is established by this component.
