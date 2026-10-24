# Semantic development accuracy of the dense NCM student

Date: 2026-10-23. Hypothesis `ncm-dense-source-semantic-development-v1`.
Classification **dense_semantic_development_quality_failed**. Campaign `data/experiments/ncm_paper_regime/20261023_dense_semantic_dev_results_v1/campaign_results.json`;
SHA256 `38604600a716a0e94b14d1caaa9bc1e1ad9209b51ab78b18d24f2b33fb53d0b2`.

| Reused development cohort | Judge | Complete summary passes | Valid | Semantic task accuracy |
| --- | --- | --- | --- | --- |
| short_development | secondary | 0/64 | 64/64 | 0.00% |
| long_diagnostic | secondary | 1/64 | 64/64 | 1.56% |

The fixed final2048 student is evaluated on the unchanged64 short and64 long
source-disjoint development cases. Each summary must be faithful, salient,
readable, nonempty and at most128 stored cl100k tokens. Judges inspect source
claims and evidence, allowing paraphrases; exact target-text matching is absent.
Qualified Terra is secondary and Sonnet primary. Sonnet is admitted separately
for a cohort only after64 valid Terra decisions and at least61 passes.
The new joint gate additionally requires at least61 of the SAME64 cases to pass
both judges, with all decisions valid, in both cohorts. Judges are models;
these are not independent human ratings or a95% population lower-bound claim.

The student's32-case TRAIN token agreement was99.66%, with one subgroup missing
its literal99.5% fitting gate by one token. That failed study stays failed.
This separately specified exploratory semantic evaluation measures the actual
summary task and does not retroactively pass the parent numerical gate.

The original body/MLPs and decoder stay frozen. A fixed source-only forecaster
runs eight concept levels, then the dense word branch predicts latents and
decodes all192 positions, using one global EOS. Full public sources and frozen
feature bytes are retained. No oracle labels, output repair, clipping, lookup,
autoregressive fallback or best-of choice enters prediction. Constructor parity
against the registered TRAIN outputs, fresh reload, retained development pilot,
model/codec identity and memory checks passed. All1669 prior first judgments,
pilot decisions, raw responses, claim citations, budget counts and Wilson
intervals replayed, with no semantic rerolls or excluded cases.

This reused development evaluation does not establish the final95% goal. Both
sealed final cohorts remain unopened; passing supports a separately specified
final evaluation. No inference speed or compression improvement is claimed.
