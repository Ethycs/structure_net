# Group-invariant dictionary readout failed to improve the native codec

Date: 2026-10-24. Hypothesis `ncm-lm1b-group-invariant-readout-v1`.
Classification **group_invariant_dictionary_readout_worse_than_native_decoder**. Campaign `data/experiments/ncm_paper_regime/20261024_lm1b_dictionary_readout_results_v1/campaign_results.json`;
SHA256 `bc3aee90e59e7da074b9190cb408cc48dcdb919c76f8d9562f548fd945de829e`. Exploratory reconstruction control, no training.

| Same saved native codes, all690 TRAIN targets | Native token agreement | Exact native sequences |
| --- | ---: | ---: |
| Learned native decoder | 63872/65063 = 98.1695% | 271/690 |
| Group-invariant dictionary | 62941/65063 = 96.7385% | 86/690 |

The fixed dictionary rule loses931 correct tokens.
The independent continuous-encoder control recovered all91520
raw positions, including padding. Thus the normalization invariance held for
clean encoder vectors, but the fixed readout did not sufficiently tolerate the
quantized vectors. This does not prove an information-theoretic loss bound.
The learned decoder remains the better measured readout. No normalization
variant was selected afterward, and no weights were changed.

Prediction received only fixed-code latents and pretrained embeddings, with no
correct token IDs or true group statistics. Both pilot cases reloaded exactly;
all690 comparisons and raw predictions were retained and independently replayed.
FP32, TF32 off, physical GPU2; peak805916672 bytes.

These are target-code reconstruction measurements, not source-only summaries.
The frozen-body student's separate date-controlled development score is0/128.
No95% task success, final-test result or speed gain is established. All2051
canonical semantic first decisions and every sealed final source remain intact.
