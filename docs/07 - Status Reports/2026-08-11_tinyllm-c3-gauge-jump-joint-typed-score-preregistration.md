# TinyLLM C3 gauge-jump joint typed-score preregistration

**Status:** FROZEN BEFORE FRESH CONFIRMATORY COHORT GENERATION

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `FRESH PROSPECTIVE NO-TRAINING REPLICATION`

**Hypothesis:** `tinyllm-c3-gauge-jump-joint-typed-score-v1`

## Decision question

The valid hidden gauge-jump primary established three facts on fresh data:

```text
fixed invariant physical-chart selector:     5/5
oracle charged connection:                   5/5
joint charged 288-way selector:              1/5
```

The registered result licenses one compact typed gauge/physical chart-mixture
comparison, but not unrestricted TinyLLM training. Before fitting that model,
test the lowest-cost decomposition suggested by the two successful controls:

> Does one physically normalized joint score close the rare charged-connection
> tail by letting the invariant carrier score the physical chart and the
> charged carrier score the gauge connection?

No checkpoint, model, optimizer, reusable fitted parameter, target-using fit,
or learned selector is permitted.

## Frozen typed score

Retain the exact 288 connection/switch/deletion candidates from the primary.
For candidate

```text
c = (omitted frame, physical switch, jump time, jump element),
```

compute:

```text
R_charged(c)  = mean squared first-character phase residual
R_invariant(c)= mean squared cubic-character phase residual
                for the candidate's omitted frame and physical switch

R_joint(c) = R_charged(c) + R_invariant(c) / 9.
```

Select the minimum `R_joint` candidate and use its exact charged forecast,
cubing only at the output. There are no weights to tune. The factor `1/9` is
fixed by local error propagation: cubing triples phase error, so its squared
residual has nine times the charged phase variance.

The score is globally `C3` invariant because both constituent residuals are
invariant. Exact Eisenstein relative and anchor-cube operations remain integer
valued until the same final floating phase calculation used in the sealed
primary.

## Fresh population

Use five seeds absent from every predecessor, pilot, and corrective cohort:

```text
773, 821, 1003, 1031, 1039
```

Generate `4,096` examples per seed and shift.

| Stream | Composition | Extrapolation |
| --- | ---: | ---: |
| base data and physical switch | `1075107 + seed` | `1077107 + seed` |
| donor derangement | `1079107 + seed` | `1081107 + seed` |
| corrupted frame | `1083107 + seed` | `1085107 + seed` |
| gauge-jump time | `1087107 + seed` | `1089107 + seed` |
| gauge-jump element | `1091107 + seed` | `1093107 + seed` |
| target derangement | `1095107 + seed` | `1097107 + seed` |

Lifecycle tests may use `64` examples from seed `1117` and base-data streams
`1099107 + seed` and `1101107 + seed`. Lifecycle outcomes are not evidence.

Retain the primary coverage floors: switch count `>=1200`, corrupted-frame
count `>=400`, jump-time count `>=580`, jump-element count `>=1800`, and every
jump-time/element count `>=280`.

## Frozen arms

1. `oracle_invariant_switch_drop`;
2. `fixed_invariant_switch_drop`;
3. `fixed_charged_no_connection`;
4. `oracle_charged_connection`;
5. `fixed_charged_connection`, the original charged-only 288-way selector;
6. `fixed_joint_typed_connection`, the new `charged + invariant/9` selector.

The original fixed charged selector is rerun unchanged. It is a prospective
replication comparator, not assumed to fail on the new sample.

## Primary endpoints

Retain the strong ceiling:

```text
scalar RMSE <= .020
exact-bin accuracy >= .90
complete physical task gate passes.
```

The complete task gate remains:

| Endpoint | Composition | Extrapolation |
| --- | --- | --- |
| posterior-mean correlation | `>=.90` | `>=.90` |
| exact-bin accuracy | `>=.50` | `>=.35` |
| target cross-entropy | `<=1.80` | `<=2.20` |
| predicted-bin coverage | `>=14` | `>=12` |

Require each population endpoint jointly on composition and extrapolation in
at least `4/5` seeds:

- both invariant and charged oracles recover;
- the joint typed score closes;
- the joint typed score matches the charged oracle within RMSE excess `.002`,
  accuracy delta `-.01`, and cross-entropy excess `.005`;
- the charged no-connection arm fails;
- every action, continuous, shuffle, and validity contract passes.

The fixed invariant and original fixed charged pass counts are locked
comparators. Tail repair is supported only if the original fixed charged
selector passes `<4/5`, the typed score passes `>=4/5`, and at least `4/5`
seeds have original failure plus typed success in both shifts.

## Validity contracts

Retain every primary contract:

- exact regeneration, zero saturation, and zero derangement fixed points;
- exact suffix-jump construction/inverse and global-action commutation;
- zero exact integer connection-action errors;
- exact charged final action and all other action errors `<=2e-12`;
- continuous oracle, original fixed, and joint typed forecasts `<=1e-10`;
- stabilization displacement `<=1e-12` and both chart margins `>=.20`;
- every shuffled arm has absolute correlation `<=.10`, RMSE `>=.80`, and
  fails the task gate;
- strict finite JSON and zero model/checkpoint/optimizer/parameter/fitted-state
  accounting.

The implementation must validate the sealed primary preregistration, runner,
result, and report before generating a fresh example.

## Locked classifications

Use the first matching valid row:

| Outcome | Classification | Decision |
| --- | --- | --- |
| oracles, typed closure/fidelity, and typed material repair each `>=4/5`; original fixed `<4/5` | `joint_typed_score_closes_time_varying_gauge_connection_tail` | close compact learned comparison and TinyLLM training |
| oracles and both fixed charged scores pass `>=4/5` | `both_fixed_connection_scores_close_fresh_scope` | do not claim unique repair; close learned work |
| charged oracle `>=4/5`, but both fixed charged scores `<4/5` | `time_varying_gauge_requires_compact_typed_chart_mixture` | license exactly one compact typed mixture, not TinyLLM |
| either oracle `<4/5` | `fresh_time_varying_gauge_not_oracle_recoverable` | repair observation precision; do not train |
| any other valid combination | `inconclusive_joint_typed_connection_score` | inspect joint failures without tuning |
| any validity failure | `invalid_joint_typed_connection_score` | infrastructure repair only |

`tinyllm_training_licensed=false` in every row.
`compact_typed_chart_mixture_licensed=true` only for
`time_varying_gauge_requires_compact_typed_chart_mixture`.

## Disclosed outcome-known diagnostic

After the valid primary was classified, the frozen score above was evaluated
on all ten sealed primary cells. It passed all ten and produced scalar RMSE
between `.004067` and `.004595`. The original fixed charged selector failed
four cells with RMSE `.024107-.026572`.

This diagnostic selected the score and therefore contributes zero evidence to
the fresh result. No primary example, aggregate, residual, or candidate choice
may be pooled into the new population.

## Frozen source lineage

| Source | SHA-256 |
| --- | --- |
| gauge-jump primary preregistration | `caab4704bb5c11a3f7353c31e20a80da33f14e1a925402df055b652916de10d9` |
| gauge-jump primary runner | `5b35658103481645aba809f5575d38159dcddc9dc7330ebfa6764ad65ba170a4` |
| gauge-jump primary result | `16f98f5c3cbf09fedfc18f12eca24a5fe69da46411d587c48c5d9072c912aca7` |
| gauge-jump primary report | `2c2fb44afaf9bd35a63cd82d4331943bdbbfa1a2c366080f50c5a5e8fe0bb6b1` |

## Expected artifact and accounting

```text
data/experiments/tinyllm_c3_gauge_jump_joint_typed_score/
  20260811_preregistered/result.json
```

The fresh run contains `40,960` base examples and `40,960` corrupted
evaluations. The same candidate banks serve both charged fixed scores, so the
primary and continuous candidate-fit counts remain `13,762,560` and
`12,779,520`. Models, checkpoints, optimizer steps, changed parameters,
reusable fits, and target-using fits remain zero.

