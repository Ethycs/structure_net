# Sample-scaled EMA for learned NCM codec transfer

Date: 2026-10-23. Hypothesis `ncm-sample-scaled-codec-ema-v1`.
Classification **codec_reconstruction_fidelity_failed**. Aggregate `data/experiments/ncm_paper_regime/20261023_sample_scaled_codec_ema_results_v1/campaign_results.json`,
SHA256 `db22e43d20d7fc23ffa198785b7b0a12eb590f76562fdbe1936f6f74df4593b5`.

| Codec | Correct non-PAD tokens | Exact texts | Correct first EOS | Changed number sequences |
| --- | --- | --- | --- | --- |
| Released reference | 20849/21905 (95.179%) | 23/252 | 251/252 | 53 |
| Unscaled EMA transfer | 14713/21905 (67.167%) | 0/252 | 238/252 | 105 |
| Sample-scaled EMA transfer | 20648/21905 (94.262%) | 23/252 | 252/252 | 46 |

The matched change recovered 27.094 percentage points
of token recovery, but still trails the released reference and fails the
99.5% representation gate. All arms have zero active invalid vocabulary IDs.
The other gates require 99% first-EOS agreement and no increase in changed-number
cases. Changed number sequences are diagnostics, not independently judged factual
errors. The original failure remains recorded.

Both evaluations supply correct target-derived codes to each codec. These are
oracle reconstruction measurements, **not complete-summary model accuracy**.
The latest measured clean native teacher is 56/64 (87.5%); the previous NCM
hybrid is 41/64. Neither establishes the 95% goal. A new semantic reconstruction
audit can assess preserved meaning without retrospectively passing this study.

The released checkpoint uses EMA statistics consistent with 4,096-window updates;
our microbatch is four. The follow-up scales counts, vector sums and epsilon by
4/4096 and uses decay .99**(4/4096), leaving initial embedding vectors unchanged.
All 2,048 update sample lists and learning rates match the failed transfer.
Initial probe inference and all 252 original-reference outputs match exactly.
The two-update CUDA pilot, full checkpoint/optimizer/RNG/dropout reload, source
exclusion and replay of raw codes, words, texts and counts passed.

The controls hold data, seeds, gradient accumulation, precision, dropout and
optimizer schedule fixed. This isolates the EMA-unit intervention in this run;
it does not establish the missing upstream training recipe or eliminate other
transfer mismatches. Parent weights are absent from codec training and remain
frozen. No source-conditioned generator, final-test result or speedup is claimed.
Both final cohorts remain sealed. Cached tokenizer length reduces validation
overhead while retaining identical outputs; it is not model throughput evidence.
