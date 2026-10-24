# Fixed word-gradient continuation: complete source-only task evaluation

Hypothesis `ncm-lm1b-word-gradient-source-only-DEV-v1`, 2026-10-24.
Freeze this implementation and independent auditor before observing new DEV
outputs. Evaluate only the final1024 continuation from original4096. Require
both actual child exits, actual whole-tool0, complete training trace replay,
strict checkpoint/optimizer/allRNG/hard-generation reload and unchanged native
states. Training alone is not success. No two-update pilot initialization.

Independently replay all1024 updates, 16384 uniform690-case presentations,
4096 neighborhood batches using schedule4096+step and1024 first-microbatch
source-only histories. Require exactly the planned auxiliary flags/temperature/
weights and finite losses. Verify source lengths, sampler and neighborhood RNG,
all11,796,480 parameters, actual optimizer1024 and checkpoint attention hash.
The old Llama body remains unloaded and unchanged.

Retain the original two ordered64-case DEV cohorts and pinned date23 FP32 full
features. Validate source/session/group/hash exclusions against all690 TRAIN
cases. Prediction accepts source-only metadata and uses unchanged hard native
12-level generation and128-position full-vocabulary word argmax. No targets,
teacher codes, truncation, correction, reranking, masking or AR fallback. Retain
all641 raw codes,128 raw word IDs, predicted boundaries and unmapped IDs.
Run first/longest inference pilot and require deterministic fresh full parity.

Inherit the latest2728 canonical first judgments from the registered native
word-readout campaign c8164656..., cache d27baaf9.... Never fork, reroll valid
negatives or retry exhausted invalids. Build exact public-source packets. Use
the existing qualified Terra and Sonnet judges and semantic summary rubric:
supported claims, salient content, readable, nonempty and<=128cl100k tokens.
Paraphrases are accepted. Exact matching is not a task-quality metric.

The four-case judgment pilot checks validity only. Score all128 with Terra.
Only if BOTH cohorts have64valid and>=61passes admit Sonnet on all128. DEV
qualifies only if>=61of the SAME64 pass BOTH judges in EACH cohort, allvalid.
Independently replay outputs, packets, judgments, decisions and cache lineage;
register the campaign and verify backend readback. Preserve every failure.
The sealed final320 and older final selections remain closed and unhashed.
Only audited registered DEV success may admit the final test; no overall95%
claim until that final task evaluation passes. No speed claim in this study.
