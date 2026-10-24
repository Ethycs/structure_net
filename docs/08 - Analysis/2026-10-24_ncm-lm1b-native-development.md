# Full-native655 NCM source-only development result

Date: 2026-10-24. Hypothesis `ncm-lm1b-native655-source-only-DEV-v1`. Classification **full128_source_only_DEV_quality_failed**.
Campaign `data/experiments/ncm_paper_regime/20261024_lm1b_native655_development_results_v1/campaign_results.json`, SHA256 `4f8776030d08a77ae88c47bbfdf340e6da5ade083b170ba00cf0f25adc3e8d13`.

The released97Mconcept generator, domain word decoder and source attention
were adapted together:127,732,736parameters,1024fixed updates on all655
unchanged original TRAIN targets. The35revised targets were excluded.
Both pilot and full run reset the same warm initializer; neither pilot
weights nor the16-case overfit checkpoint were reused. Native encoder/VQ
geometry and the original Llama body/MLPs remained frozen.
The actual independent CPU training audit replayed all16,384presentations,
1024source-only history microbatches, complete noise and checkpoint state.
The hard-forward/biased-soft-backward word objective is an experimental
conditional adaptation, not a matched paper replication. Full FP32
source-only inference loaded the ENTIRE trained native checkpoint, retained
every source position, and received no target words or codes.

| Cohort | Judge | Complete-summary passes | Valid |
| --- | --- | ---: | ---: |
| short_development | secondary | 0/64 | 64/64 |
| long_diagnostic | secondary | 0/64 | 64/64 |

Both-cohort same-case DEV qualified: **False**. Primary admitted: **False**. Canonical first judgments retained: 3222.

The semantic judges assess supported claims, salience and readability; exact
summary matches are not required. Nonempty output and128stored-token cap
remain mandatory. Invalids are retained. Every raw output, failure and prior
first judgment is preserved. Previous old-student0/128 and codec failures
remain separate evidence. The final320sources remain closed and unhashed;
final task accuracy and speed are unmeasured. This DEV result does not
complete the95%task objective.
