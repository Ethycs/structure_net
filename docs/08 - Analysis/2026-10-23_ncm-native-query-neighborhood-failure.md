# The full frozen-native NCM failed summary generation

Date: 2026-10-23. Hypothesis `ncm-native-query-neighborhood-v1`.
Classification: **NCM_summary_development_failed**. Aggregate:
`data/experiments/ncm_paper_regime/20261023_native_query_neighborhood_results_v1/campaign_results.json`, SHA256 `1b796ee714ea3b4e5667f55f5c9289efb92aa1bc04e2f956571825f971f06812`.

The completed 2048-update model scored **0/64 whole-summary
passes (0.0%)**, with 64/64 valid
first judgments from qualified Terra. Component passes: faithful
10/64, salient 4/64, readable
0/64. Stored summaries exceeded128tokens in
0/64 cases. The Wilson95% interval for whole-summary
accuracy is0–5.66%. Zero readable summaries is a catastrophic generation
failure. Training loss, numerical controls and correct process execution do
not qualify it as a successful model.

All28original Llama3.2-3B blocks and native embedding/output weights remained
frozen. The model trained8,637,696new query-attention, concept-head and hint
parameters in BF16-native/FP32-new precision, using1024language updates and
1024summary updates with language replay. Its five concept levels and four
parallel word refinements were fixed in advance. Every development source
was complete, untruncated, in the original64-case order, and disjoint from
training groups. No AR fallback or external verifier generated these outputs.

All three workers terminated successfully. Frozen body, books and CPU cache
integrity and exact checkpoint/generator reload passed. These are systems
checks. Final generated texts and all raw first-judgment requests were retained;
the offline finalizer replayed packet identities, token lengths, raw response
parsing, first-valid decisions and whole-summary gates. All128earlier baseline
judgments, including negatives, remain exact in the grading cache. The generic
endpoint's inherited `NCM_trained:false` annotation is stale: the actual model
completed2048updates. This correction is explicit here and in the aggregate;
the completed raw endpoint is unchanged.

The required61/64Terra gate failed, so Sonnet was correctly not run. Neither
the original256sealed final cases nor the new64longer-source final cases were
opened. No throughput, compression or speed advantage was measured. Under the
same128-token rubric, unchanged3B and1B AR baselines scored52/64 and47/64;
neither establishes95%, but both substantially exceed this adaptation.

This is another failed custom adaptation, not a faithful replication of an
author-provided NCM trainer: the inspected official NCM repository releases
inference/checkpoints but no training loop. The pinned JacobiForcing trainer
retains a clean AR loss plus a noisy-token consistency loss, uses different
training capacity and decoding, and has not been run locally. Comparing its
code does not transfer its reported results to this model.

The next registered exploratory diagnostic crosses predicted versus oracle
coarse plans with parallel versus teacher-forced word inputs on four training
summaries and four raw-language diagnostic cases. Its recovery rates will be
mechanism evidence only. The0/64primary result stays fixed. Wider vetted
teacher labels are likewise training data quality, never student accuracy.
