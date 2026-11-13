# J-space relation propagation: first intact-model run

Date: 2026-11-13. Study: `JSPACE-RELATION-PROPAGATION-01`.  
Status: completed, exploratory; unconfirmed.  
Classification: `heldout_relations_without_validated_propagation`.

Seven development-selected J-derived cross-layer relations survived held-out
checks. None of the four-dimensional within-layer banks supplied an accepted
source constraint, so the neural Gröbner-propagation stage had no eligible
source ideal. PCA supplied more surviving transition relations than J; random
directions supplied none. This run establishes some predictable cross-layer
structure, not a validated neural Gröbner basis, a J-specific advantage, or a
compression result.

The immutable aggregate is
[`campaign_results.json`](../../../data/experiments/jspace_relation_propagation/20261113_neural_v1/campaign_results.json),
SHA-256 `d92d042d55c712704093ae7a8f52b9bf8171f16fbdd650017dabd65127a3ee93`.
The scientific fingerprint is
`e7f0a06871bcf9bd480e45e4e71343cf385cd2ffb06fd03a2fa0b628ba1a2d76`.
The [prospective protocol](../../07%20-%20Status%20Reports/jspace/2026-11-13_jspace-relation-neural-protocol.md)
and archived producer determine the gates; this report does not retune them.

## What actually ran

The frozen Qwen2.5-3B-Instruct checkpoint ran intact in BF16, with 512 training,
128 development, and 128 evaluation examples. Facts in each prompt map a person
to a country and then to a colour. Names and prompt families differ between
splits; anonymous mapping patterns recur, as recorded in the dataset manifest.
No model weights were trained. Dictionary estimation and statistical relation
fitting are separate from neural model training.

Four normalized dictionary rows, ordered China/France/Germany/Russia, were
measured at 12 block outputs. Six adjacent edges were tested. The rows are
selected Jacobian-lens observables; their labels alone do not establish that
they are causally effective concept activation vectors. PCA and random controls
also have four rows. All observations are continuous, final-prompt-position
measurements. No Boolean or one-hot relations were imposed on them.

All 768 natural and 384 intervention captures completed. A separate 16-case
shakedown passed finite-value and save/reload checks. The lens used 16 independent
calibration paragraphs. Capture took 243.36 seconds and analysis 35.22 seconds.
Peak allocated/reserved Torch memory was 6.62/6.77 GiB. These are experiment
timings, not a model throughput benchmark.

Reservation `8908ce8fa8c6` allocated 10240 MiB on the RTX 3060. The controller
confirmed `released`, with end reason `command exited 0`; all cards returned
to their prior free state. The [release receipt](../../../data/experiments/jspace_relation_propagation/20261113_neural_v1/reservation_release_receipt.json)
preserves that check. Runtime versions were Torch 2.4.1+cu121, Transformers
4.53.3, and SymPy 1.14.0, through locked root Pixi. No inference service was
brought online for this experiment.

## Relation results

Selection required normalized RMS ≤0.10 and at least threefold improvement over
32 coordinate shuffles. Mixed relations also had to beat source/destination
pairing shuffles. All fitting and development selection were frozen to disk
before evaluation scoring.

| Bank | Within-layer relations accepted on development | Joint relations accepted on development | Joint relations passing evaluation | Supported eliminated constraints |
| --- | ---: | ---: | ---: | ---: |
| J-derived | 0 | 8 | 7 | 0 |
| PCA | 0 | 22 | 13 | 0 |
| Random | 0 | 0 | 0 | 0 |

Counts span correlated layers and fitted polynomials, not independent model
replications. They are descriptive. In particular, 13 versus 7 does not establish
a statistically quantified advantage for PCA.

| Edge, zero-based blocks | J passing evaluation | PCA passing evaluation | Random passing evaluation |
| --- | ---: | ---: | ---: |
| 5→6 | 0 | 2 | 0 |
| 11→12 | 0 | 1 | 0 |
| 17→18 | 0 | 3 | 0 |
| 23→24 | 0 | 1 | 0 |
| 29→30 | 4 | 3 | 0 |
| 34→35 | 3 | 3 | 0 |

The four surviving J candidates at 29→30 came from the degree-two search; their
held-out normalized RMS values were 0.05585, 0.04722, 0.03011, and 0.01614.
Three linear candidates survived at 34→35, with RMS 0.05510, 0.06098, and
0.08072. The fourth selected linear candidate failed at 0.34055. These are
approximate joint relations, not source-state constraints or proven independent
nonlinear generators.

For example, one surviving 34→35 relation is approximately

```text
  0.21546 s_China - 0.50500 s_France + 0.43273 s_Germany - 0.03496 s_Russia
- 0.20378 t_China + 0.52049 t_France - 0.44264 t_Germany + 0.04149 t_Russia ≈ 0
```

Here `s` and `t` are separately training-standardized source and destination
scores. The exact floating coefficients and scales are in the frozen artifact;
the displayed rounded equation is not an exact identity. Its small residual
shows a predictable combination across the last two blocks, without showing
that the four source variables themselves obey a low-degree constraint.

The smallest J within-layer development RMS was 0.2089, above the frozen 0.10
threshold. Thus every one of the 18 bank/edge entries returned
`insufficient_accepted_relations` before symbolic elimination. There were no
neural Gröbner timeouts, unit ideals, or failed rationalizations to interpret:
that symbolic stage was not reached. The independent exact Boolean control
passed positive, negative, and exhaustive-state checks, which validates the
algebra implementation only.

Train-only regression baselines are saved separately. For J at 29→30, held-out
normalized RMSE was 0.5032 for linear and 0.4461 for quadratic regression; at
34→35 it was 0.2383 and 0.2722. Small residuals of selected joint polynomials
therefore do not imply accurate reconstruction of every destination coordinate.

## Task accuracy and interventions

The frozen endpoint is a correct colour as the first complete lexical answer.
All answers remain in the denominator.

| Split | Correct | Strict task accuracy |
| --- | ---: | ---: |
| Training | 505/512 | 98.63% |
| Development | 85/128 | 66.41% |
| Evaluation | 61/128 | 47.66% |

A post-hoc template audit explains much of the evaluation drop: the JSON
dictionary family scored 61/64 (95.31%), whereas the reverse-roster family
scored 0/64. In that second family the model began answers with names or
explanatory text, for example `Flora's country` or `Hazel is`, despite the
colour-only instruction. The first-word decoder stopped there. This is a
measured output-contract failure; the semantic correctness of a longer
explanation was not measured. Keep the overall 61/128 endpoint unchanged.
Development also varied by family: records 62/64, checklist 23/64.

The causal subset was selected before capture, balanced over 16 country–colour
cells. Thirty of its 64 natural answers were correct; one already matched its
counterfactual label. Per site, J interventions yielded 30/64 original-correct
and 1/64 counterfactual-correct answers. All 30 originally correct cases retained
their original answer; none switched to its counterfactual label. Random and
PCA controls also achieved zero counterfactual successes among those same 30
originally correct cases.

The J intervention changed one parsed answer at each site. Its median realized
perturbation was about 1.96% of the original residual norm at block 11 and 1.49%
at block 23. Controls matched those norms. Thus these pair-coefficient swaps
did not demonstrate usable causal control of the intermediate concept; this
does not show that stronger or differently fitted interventions cannot work.

The seven late J joint relations still passed the all-row intervention checks
at both sites. Because the source relation banks were empty, every
source-compatible conditional subset was explicitly undefined. Stability under
these small interventions is not evidence of propagated source ideals or
successful task steering.

## What remains outside the vector bank

After the user asked about the remainder, a separate, post-hoc CPU diagnostic
measured orthogonal projection energy from the already captured activations.
This did not change any campaign gate. Its [result](../../../data/experiments/jspace_relation_propagation/20261113_neural_v1/analysis/posthoc_span_energy.json)
has SHA-256
`7012f1a4eddc3593661a237a665b9954a835bfcb624bac1e11f3553a72ea852f`.

For vector rows `B`, compute their rank-four orthonormal span `Q`, then
`h = Q.T Q h + r`. Use squared norms pooled across the 128 evaluation examples
separately at each layer. The nonorthogonal concept scores are not themselves
an orthogonal projection. The energy accounting error was below 3×10⁻¹⁶.

| Bank | Raw activation energy captured, range across 12 layers | Energy captured around the fixed training mean |
| --- | ---: | ---: |
| Four J-derived directions | 0.042–1.659% | 0.139–1.216% |
| Four PCA directions | 1.466–14.893% | 2.315–20.405% |
| Four random directions | 0.011–0.242% | 0.190–0.564% |

The four J directions leave 98.341–99.958% of raw activation energy in the
orthogonal residual. These are measurements of a tiny selected bank, not of
the full Jacobian dictionary. Energy is also not task information, accuracy,
parameter count, or computational cost. A low-energy direction may be causally
important. The centered column uses the mean from training activations, never
a refitted evaluation mean; it measures held-out energy around that reference.

For a Gröbner basis `G={g_i}`, the algebraic remainder has a different meaning:
`f = sum(q_i g_i) + NF_G(f)`. That normal form contains the terms not eliminated
by the ideal under the fixed monomial order. It is not a percentage of activation
energy. This experiment did not establish a neural Gröbner basis that accounts
for half the model. The 47.66% number was strict task accuracy.

## Interpretation and next experiment

This bounded run found late cross-layer relationships, but no low-degree
within-layer constraints from which to propagate an ideal. It also did not
establish that the four named dictionary rows were task-controlling CAVs.
Those are specific gaps, not a numerical crash or a failed compression gate.

The next useful experiment is to fit task-aligned CAV candidates inside a
larger J-derived dictionary, then validate intermediate-country discrimination
and counterfactual swaps before searching for their algebraic relations.
Separate new-world tests using a stable output format from harder prompt-family
transfer tests. Any such follow-up needs new frozen data and a prospective
protocol; it must not quietly rescore the 0/64 family or tune this held-out set.

Even if an eliminated constraint later passes development and evaluation, its
source and mixed generators must be checked separately. A surviving output
constraint can be a destination regularity when its antecedents fail. Exact
elimination proves a rational-surrogate identity, and small approximation
errors can amplify during elimination. Conditional intervention subsets cannot
replace unconditional causal effects. No exact-neural or compression claim
follows from the present evidence.

Validation comprised 100 capture/algebra/runner tests, 41 registry tests, and
four post-hoc energy-accounting tests. The tests include actual tiny-Qwen
forwards, exact algebra controls, causal hook isolation, artifact mutation
checks, and a durable fit-before-evaluation boundary. Reproduction commands
are in the prospective protocol; completed artifacts require a new output root
for any rerun.
