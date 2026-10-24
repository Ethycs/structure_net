# Pretrained, source-grounded NCM: complete source-only development evaluation

Date: 2026-10-24. Hypothesis `ncm-lm1b-full690-grounded-source-only-DEV-v1`.
EXPLORATORY development, not a replication of unconditional LM1B generation.
The objective remains >=95% complete-summary task accuracy on source-disjoint
evaluation; neither training loss nor oracle reconstruction qualifies it.

This design is fixed while the full690 training run is still running and before
any new student DEV output is inspected. Use only its fixed final4096 checkpoint,
after actual successful child and whole-tool termination and exact reload. Never
select a recovery checkpoint. Start from the authors' hash-pinned LM1B generator
and the already measured task-domain codec; load the new 11,796,480 attention
parameters strictly. Check all frozen pretrained hashes and model state before
and after inference. Full FP32 inference, TF32 off, physical GPU0, batch1, eval.
Do not run concurrently with the original training job.

Use all unchanged permanent DEV cases: 64 short_development and 64
long_diagnostic. Source metadata SHA1395a35a... and feature-manifest
SHA637be63f... fix case order, the complete source, alignments and date23 prompts.
Replay source/session/group/hash exclusion against all690 training cases. Full
FP32 cached source features have width3072; verify each file and identity. No
source truncation. Both sealed final cohorts and older final files remain closed
and unhashed. Reading the shared public corpus for exact-source proof is allowed.

Prediction accepts source-only metadata and features; forbid target_text,
word_ids, native_word_ids and concept_ids. Call generate(features,mask) only.
Keep all641 concept IDs at all12 native levels and all128 raw full-vocabulary
word IDs. Use the existing native CLS-frame renderer unchanged: consume a
predicted leading CLS101 if present, stop at the next predicted CLS, and BERT
decode with skip_special_tokens. Native tokenizer decoding can omit the six
unused IDs30522..30527; retain and count those positions explicitly. Do not mask
or repair model outputs. Empty text, overflow, missing BOS/EOS and all raw IDs
remain in the evidence. No teacher codes, reranking, retrieval, lookup or AR
fallback. Contexts clear after every request. Inference pilot uses the first
short case and longest full source, checks deterministic repeated generation;
fresh full inference must reproduce both pilots exactly. All128 stay scored.

Grade source evidence with the unchanged 128-storage-token semantic rubric:
faithful, salient, readable, nonempty, within cap; paraphrases accepted. Retain
all2472 canonical first judgments from the registered critique-label campaign
SHAeaeb64b..., cache SHA41e96b97.... No forks, rerolls of valid negatives or
retries of exhausted invalids. One cache spans cohorts and roles. Both exact
judge aliases retain their existing50/50 qualification. Four-case validity-only
pilot (first2 per cohort), then all128 Terra secondary. Admit Sonnet primary
only if BOTH complete64-case cohorts have64valid and >=61passes under Terra.
Success requires >=61 of the SAME64cases passing BOTH judges in EACH cohort,
all64valid for each judge. Invalid judgments stay in denominators with bounds;
do not silently discard any case. A secondary miss is a measured scientific
failure; primary then remains unmeasured.

Use fresh `20261024_lm1b_grounded_development_*_v1` roots. Freeze source, tests
and this plan before inference. Retain actual child exits and a whole-tool exit
receipt. An independent CPU finalizer replays decoding, source identities,
packets, every first semantic decision, role admission and same-case gates;
also audit the training sampler/noise histories and strict checkpoint contract.
Report and register conservatively: DEV only, final accuracy pending, no speed
claim. Passing DEV merely authorizes a separately audited sealed final stage;
this workflow does not open final sources or mark the overall goal complete.
