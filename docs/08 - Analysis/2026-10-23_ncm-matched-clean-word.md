# Correct-history word fitting on the complete TRAIN population

Date: 2026-10-23. Component hypothesis ncm-matched-population-clean-word-v1.
Classification: **clean_word_fit_gate_failed**. Campaign data/experiments/ncm_paper_regime/20261023_matched_clean_word_results_v1/campaign_results.json.
SHA256: 4459a4bf7dbb2af444768a8af3868199e13e35e596a4476030a74ed7cf31a6ab.

| Fixed TRAIN probes | Correct-history token agreement | Actual-history token agreement |
| --- | --- | --- |
| original_16 | 1395/1635 = 85.32% | 1396/1635 = 85.38% |
| additional_16 | 1314/1562 = 84.12% | 22/1562 = 1.41% |
| all_32 | 2709/3197 = 84.74% | 1418/3197 = 44.35% |

The component gate requires at least 199/200 active-token agreement in each
sixteen-case probe group and in their complete union. This is TRAIN fitting,
including EOS and excluding padding. It is not semantic summary accuracy.
The preceding model's measured semantic development score remains 0/64 short
and 0/64 long, with 128 valid Terra judgments. This new word branch has not
received a semantic evaluation.

The word branch started from registered checkpoint e608f416e2915204a56d5f892d27087dc5cc2ff0e7b0b4cc24dbc44fe272acb3; its
parameters were learned previously, while AdamW and RNG streams started fresh.
It trained exactly32,768updates on all690 immutable TRAIN examples, with
correct coarse histories only and no finest target codes or target words in
prediction inputs. FP32 parameters and loss, BF16 forward computation, disabled
TF32, original dropout, complete native source frames and all192 output
positions were preserved. Only the fixed final checkpoint was selected.

Maximum-frame gradient/optimizer/RNG/composed-reload and fresh-model fit-path
pilots completed before full training. All sampled sources, complete frames,
correct-history hashes, learning rates, optimizer steps, sampler state, unused
word-branch parameters and raw fitting counts replayed. The original Llama
body/MLPs, original codec and 252-case forecaster stayed frozen.

A qualifying component result admits further matched-population forecaster
training and later predicted-history adaptation; neither has been established
by this component. Both sealed final source cohorts remain unopened. The goal
still requires at least95% complete-summary semantic quality on the original
source-disjoint final evaluation, after qualifying development results.
No speed or compression benefit is measured here.
