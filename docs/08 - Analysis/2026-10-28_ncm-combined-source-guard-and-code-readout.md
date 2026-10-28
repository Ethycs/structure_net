# Combined source guard and fixed-code readout findings

The combined 208,631-pair inventory yields **208,603 pairs passing the new source
guard**. On the four newly inspected recovered TRAIN controls, the improved
standalone codec reconstructs reference summaries exactly, but the same codec
still produces qualitatively poor summaries from the predictor's saved codes.
Neither finding establishes the 95 percent application goal.

The completed source-guard result is
`data/experiments/ncm_paper_regime/20261028_combined_prepared_source_guard_v1/result.json`,
SHA256 `e9b48a02b8a35b05fc73aa7adf3c20ea1dc3e57e0b46f4dfe95d3ada5ef823d2`.
Its actual execution receipt binds session43836, terminal chunk663851, exit zero.

| Population | Indexed pairs | Pass source guard | Excluded |
|---|---:|---:|---:|
| XSum | 162,353 | 162,353 | 0 |
| Recovered conversations | 43,894 | 43,894 | 0 |
| Reviewed anchors | 2,384 | 2,356 | 28 |
| Total | 208,631 | **208,603** | **28** |

The audit checked 205,627 original source/context records, covering all 202,340
distinct source texts, against 640 open held-out sources and the existing
512-component reserved metadata. It joined exact source, component, group,
family and turn aliases and propagated exclusions across whole connected groups.
There are 171,091 resulting TRAIN source components.

All 28 exclusions are associated with the recovered-label diagnostic cohort:
24 anchors have a direct metadata alias match, 11 have a direct text match,
and their union covers all 28. The four without direct alias matches are among
the 11 text matches. No direct conflict was found with the main application DEV,
public development or original 64 reserved component cohorts. The breakdown is
in `20261028_combined_source_guard_conflict_breakdown_v1`; the text-only closure
was also checked by actual terminal chunk7b826d, exit zero.

These exclusions apply to a future mixture. They do not remove prior exposure
from existing model weights: the recovered-label diagnostic cohort must not be
represented as source-disjoint for predictors already trained on those anchors.
Original corpus artifacts and all prior evaluation decisions remain unchanged.
Final-test source text was neither opened nor hashed. This is the declared
five-gram/metadata guard, not an exhaustive semantic duplicate search or a
comparison of every TRAIN pair against every other TRAIN pair. New mixture
training remains unadmitted pending its protocol and the current candidate
evaluations.

The fixed-code readout result is
`20261028_saved_recovered_codes_new_readout_v1/result.json`, SHA256
`022905b5ceca9de977bb9288f607eee1fa3ba3c4045390a1c84a15c0f059aaea`.
The run used the fixed step6212 standalone decoder in FP32 on GPU1, with no
generator calls or parameter updates. Twelve outputs were decoded twice each;
all repetitions matched, codec tensors stayed unchanged, and peak GPU allocation
was 799,133,184 bytes. Actual session2933 ended at chunk913d4a with exit zero.

For all four references, re-encoding reproduced the stored singleton target
codes. Decoding those codes gave **4/4 exact native-rendered summaries, zero
lexical-token errors and preserved digit sequences in all four**. These are
TRAIN reconstruction controls, not source-only task accuracy. The independent
rendering replay is `20261028_saved_recovered_codes_new_readout_replay_v1`.

With the source-predicted codes, the improved codec retains severe repetition,
omissions and unrelated content. For example, Houston sports history still
becomes repeated generic football/club text, and sunscreen recommendations still
become unrelated lists. Exact reference-code agreement changed from146/2564 to
156/2564 through training; alternative summaries can use different codes, so this
number is only a mechanism diagnostic. The controlled reconstruction comparison
places a bottleneck upstream of the standalone word decoder on these examples.
It does not identify which predictor objective, data exposure or conditioning
change will fix it.

Separately, the older existing1376-pair candidate's complete semantic audit
finished with0/128 valid Terra passes, preserving5775 canonical first judgments.
That result is not an accuracy measurement of the latest43894-pair candidate.
The expanded candidate's source-only inference pilot has now been launched;
hierarchy and recovered-candidate DEV evaluations retain their original order.
