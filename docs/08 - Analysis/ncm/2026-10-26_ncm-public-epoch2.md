# Public NCM curriculum: epoch2

Classification: `DEV_semantic_quality_failed`. Final320 remains sealed.

| Epoch | Terra semantic passes | Valid judgments | Joint Terra/Sol accuracy | DEV qualified |
|---|---:|---:|---:|---|
| 1 | 0/128 (0.00%) | 128/128 | unmeasured or unresolved | False |
| 2 | 0/128 (0.00%) | 128/128 | unmeasured or unresolved | False |

Accuracy measures complete readable, faithful, salient summaries within128 stored tokens; paraphrases are accepted.
Training loss, code agreement and codec reconstruction are separate diagnostics and are not task accuracy.
Each epoch visits all162,353 public pairs and54,118 reviewed-anchor presentations. The original Llama body and reconstruction path remain fixed.
This is one continuing training trajectory evaluated at fixed milestones, not independent model replications.
Earliest candidate for the separate sealed-final protocol: `None`. DEV qualification does not establish the final95% goal.
Inference speed remains unmeasured.

Detailed cohort decisions and artifact hashes: `data/experiments/ncm_paper_regime/20261026_public_curriculum_epoch2_results_v1/campaign_results.json`.
