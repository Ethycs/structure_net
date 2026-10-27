# Semantic quality of the frozen XSum adapter probe

Classification: `public_probe_semantic_improvement_not_established`. These are semantic summary judgments, not exact-token matches.

| Model | Terra passes | Semantic pass rate | Valid judgments | Faithful | Salient | Readable |
|---|---:|---:|---:|---:|---:|---:|
| native | 3/128 | 2.34% | 128/128 | 4 | 26 | 32 |
| initial | 0/128 | 0.00% | 128/128 | 1 | 14 | 44 |
| mse | 0/128 | 0.00% | 128/128 | 1 | 14 | 42 |
| distill | 1/128 | 0.78% | 128/128 | 1 | 14 | 42 |

Paired distill-versus-MSE outcomes: `{"accuracy_difference": 0.0078125, "both_fail": 127, "both_pass": 0, "cases": 128, "directional_gate_passed": false, "distill_only": 1, "extra_passes": 1, "invalid": 0, "mse_only": 0}`.
Joint Terra/Sol passes, where admitted and valid: `{"distill": null, "initial": null, "mse": null, "native": null}`.
A pass requires readable, faithful, salient, nonempty text within128 stored tokens. Paraphrases and minor grammar defects are accepted; all material facts must be supported by the complete source.
Every arm used the same128 sources and frozen saved outputs. Each request contained one anonymous candidate. Existing qualified Terra judged every case; Sol was conditional on at least122 valid Terra passes for an arm. All first valid judgments and exhausted invalid attempts were preserved.
Canonical first judgments retained: 5190 (including all4691 inherited firsts).
These public XSum TRAIN sources were excluded from the512-case adapter fit but may have been seen by the released teacher, and this probe was reused in development. These scores are not application final accuracy or full input-range validation. They cannot fulfill the95% goal.
The separate main hybrid still has0/128 on its latest application DEV evaluation, a different population and model. No direct score comparison between these populations is warranted. No new model training, final-test access or speed measurement occurred in this diagnostic.
