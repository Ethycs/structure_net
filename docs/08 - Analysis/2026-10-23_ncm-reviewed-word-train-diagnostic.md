# TRAIN mechanism diagnosis after zero-percent summary quality

Date: 2026-10-23. Diagnostic `ncm-reviewed-word-train-mechanism-v1`.
The current model's measured semantic development accuracy remains **0/64 short
and 0/64 long**, with all 128 Terra decisions valid. These additional TRAIN
measurements explain component behavior and establish no semantic accuracy.

Both the repeated two-case CUDA pilot and the complete 32-case measurement
finished with actual exit zero. Model checkpoint SHA256 remains
`e608f416e2915204a56d5f892d27087dc5cc2ff0e7b0b4cc24dbc44fe272acb3`.
The complete raw result is
`data/experiments/ncm_paper_regime/20261023_reviewed_word_train_diagnostic_v1/result.json`,
SHA256 `7e17101af06d30d4ef490cc740225aaada14128e315c47870d33263c7aed837f`.

| TRAIN probe group | Actual-history token agreement | Correct-history token agreement |
| --- | --- | --- |
| Same original 16 | 1,331/1,635 = 81.41% | 1,331/1,635 = 81.41% |
| Additional 16 | 1,181/1,562 = 75.61% | 1,106/1,562 = 70.81% |
| All 32 | 2,512/3,197 = 78.57% | 2,437/3,197 = 76.23% |

These are exact active-token fitting diagnostics, including EOS and excluding
padding. Neither arm reproduced a complete target text. This is separate from
the semantic evaluation, which permits paraphrases and measured zero percent.

On the original sixteen, active coarse-code agreement is 6,683/6,688 = 99.93%.
Actual- and correct-history word predictions agree at every active position.
The decoder therefore still fails to fit these TRAIN targets despite nearly
correct plans. On the additional sixteen, active coarse-code agreement is
100/6,080 = 1.64%, with individual levels between 1.17% and 3.75%. The fixed
forecaster had trained on the original 252 cases, while the word branch trained
on all 690; these additional cases expose its poor transfer to that population.

Every fresh forecast matched its actual TRAIN cache exactly. Complete GPT2
targets decoded exactly; pilot results survived a fresh model load unchanged.
Model parameters and the original codec stayed byte-identical; peak diagnostic
allocation was 2,023,462,912 bytes. A padding-loss check found the correct ignore
ID, 50257. These observations weaken a cache or token-ID mismatch explanation.

The earlier word-model diagnostic achieved 95.84% token fitting on these same
original targets with correct codes and 1.28% with predicted codes. The new
recipe removed that gap on the original probes while losing correct-code
fitting. Because the model and training recipe both changed, this comparison
does not isolate a single causal intervention.

The next training design should align the forecaster and word branch on the
same complete TRAIN population and establish correct-history word fitting
before further adaptation to predicted histories. Whether that improves unseen
summary quality remains unproven. Preserve both failures and both unopened final
cohorts; retain the original 95% complete-summary target and make no speed claim.
