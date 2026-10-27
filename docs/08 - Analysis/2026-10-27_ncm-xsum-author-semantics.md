# Semantic quality with the author text renderer

Classification: `author_semantic_improvement_not_established`. Application final accuracy remains unmeasured.

| Model | Old renderer passes | Author renderer passes | Author pass rate | Valid | Faithful | Salient | Readable |
|---|---:|---:|---:|---:|---:|---:|---:|
| native | 3/128 | 3/128 | 2.34% | 128/128 | 3 | 27 | 33 |
| initial | 0/128 | 0/128 | 0.00% | 128/128 | 0 | 14 | 46 |
| mse | 0/128 | 0/128 | 0.00% | 128/128 | 0 | 14 | 40 |
| distill | 1/128 | 1/128 | 0.78% | 128/128 | 1 | 15 | 44 |

Only text rendering changed: stop at EOS, remove PAD tokens, and strip outer whitespace. All raw generated IDs and complete sources stayed fixed. Every unchanged packet reused its first judgment, including failures.
A pass requires faithful, salient, readable, nonempty text within 128 stored tokens. Paraphrases are accepted. Terra judged all cases; Sol was conditional on 122/128 valid Terra passes.
Retained 5190 inherited firsts and added 73. No original result was replaced.
Paired distill versus MSE: `{"accuracy_difference": 0.0078125, "both_fail": 127, "both_pass": 0, "cases": 128, "directional_gate_passed": false, "distill_only": 1, "extra_passes": 1, "invalid": 0, "mse_only": 0}`.
Joint Terra/Sol passes, where admitted and valid: `{"distill": null, "initial": null, "mse": null, "native": null}`.
These reused public XSum TRAIN sources were excluded from adapter fitting but may have appeared in the released teacher training. They cannot establish application DEV, source-disjoint final accuracy, or the 95% goal. No model was trained, no final320 source was opened and no speed was measured.
