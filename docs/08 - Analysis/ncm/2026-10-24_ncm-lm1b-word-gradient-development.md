# Word-gradient NCM source-only development result

Date: 2026-10-24. Hypothesis `ncm-lm1b-word-gradient-source-only-DEV-v1`. Classification **full128_source_only_DEV_quality_failed**.
Campaign `data/experiments/ncm_paper_regime/20261024_lm1b_word_gradient_development_results_v1/campaign_results.json`, SHA256 `067ade9b2c24680c9564dec3a2847e02cdbb805ca7053fb60b7f08d6a6edcfe2`.

The authors' released LM1B generator and task-adapted codec were frozen while
11,796,480 source-attention parameters continued for1024fixed updates on690
examples from the original4096 checkpoint, without new random weights. The
independent CPU auditor replayed all16,384 sampled examples,
all neighborhood corruption and1024source-only self-history microbatches.
The auxiliary word loss used an exact hard forward and biased soft-codebook
surrogate backward; generation stayed hard. No single-factor attribution.
Full precision source-only inference used every unchanged permanent DEV source,
without target words or codes. This is conditional task adaptation, not a paper replication.

| Cohort | Judge | Complete-summary passes | Valid |
| --- | --- | ---: | ---: |
| short_development | secondary | 0/64 | 64/64 |
| long_diagnostic | secondary | 0/64 | 64/64 |

Both-cohort same-case DEV qualified: **False**. Primary admitted: **False**. Canonical first judgments retained: 2856.

The semantic judges assess supported claims, salience and readability; exact
summary matches are not required. Nonempty output and128stored-token cap
remain mandatory. Invalids are retained. Every raw output, failure and prior
first judgment is preserved. Previous old-student0/128 and codec failures
remain separate evidence. The final320sources remain closed and unhashed;
final task accuracy and speed are unmeasured. This DEV result does not
complete the95%task objective.
