# Matched whole-TRAIN training-dropout diagnostic

Date:2026-10-23. Hypothesis ncm-clean-word-dropout-fit-v1.
Classification **both_dropout_fit_gates_failed**. Campaign data/experiments/ncm_paper_regime/20261023_clean_word_dropout_results_v1/campaign_results.json; SHA256 43783fe3dec0e5968c0655222699994e63d83b4d003a97a7069fc92843cfb302.

| Fixed TRAIN probes | Upstream final | Continued original dropout | Continued zero dropout |
| --- | --- | --- | --- |
| original_16 | 1395/1635 = 85.32% | 1404/1635 = 85.87% | 1435/1635 = 87.77% |
| additional_16 | 1314/1562 = 84.12% | 1327/1562 = 84.96% | 1381/1562 = 88.41% |
| all_32 | 2709/3197 = 84.74% | 2731/3197 = 85.42% | 2816/3197 = 88.08% |

These are correct-history word-token fitting measurements including EOS and
excluding padding, not semantic summary accuracy. Both arms used the same690
immutable TRAIN examples, the exact registered final initializer
c90ee2ae272c298022d50d7a9b4182f06d8d995fadf8ecceb6f091e35b8b5223, identical sampled IDs/frames/history/word-target
hashes and rates,8192updates each, and fresh matched AdamW optimizers. Only
training dropout differed: original .3 transformer/.1 decoder versus zero,
covering both nn.Dropout and numeric functional attention dropout. Both were
evaluated in eval mode. One model/dropout seed and one independent sampler seed
were used; changed dropout RNG consumption is part of the intervention.

The unchanged199/200 fitting gate applies separately to original16, added16
and all32 probes. If both continuations pass, extra training cannot be confused
with an isolated benefit from removing dropout. No arm is automatically promoted
to the old queued forecaster, and none establishes unseen generation quality.

Both CUDA pilots, frozen-state/native-cache/reload controls, full paired sampler
and optimizer audits, exact raw fitting replay and conservative registry checks
are retained. Original Llama body/MLPs, codec, old252-case concept forecaster and
unused heads stayed frozen. Predictions received no finest labels or words.

The last completed semantic candidate remains0/64short+0/64long,128valid Terra
judgments. The full95% complete-summary source-disjoint semantic goal remains
unmet. Both sealed final source cohorts stayed unopened. No speed/compression
benefit is measured by this diagnostic.
