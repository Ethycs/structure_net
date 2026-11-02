# TinyLLM C3 temporal sensor mechanism decomposition preregistration

**Status:** FROZEN BEFORE CHECKPOINT ANALYSIS

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `REGISTERED POST-OUTCOME / ARTIFACT-ONLY CAUSAL DIAGNOSTIC`

**Hypothesis:** `tinyllm-c3-temporal-sensor-mechanism-v1`

## Decision question

The five-seed sensor-only campaign confirmed that task loss recovers a nearly
analytic exact-`C3` carrier when the temporal operator and interval decoder are
fixed. Before changing scope or training anything else, ask:

> Did the learned sensors solve the task through the affine identity response
> already exhibited by the closed-form GELU witness, or do they require
> nonlinear shared-response harmonics?

This diagnostic was selected after the positive campaign outcome. It is
prospective only with respect to the untouched checkpoint internals. It loads
and decomposes the ten saved sensors but performs no optimization and changes
no parameter.

## Frozen sources

| Source | SHA-256 |
| --- | --- |
| sensor-only campaign | `4d023d9f9d77f64b1fe75970acd63461cfd5a64aa1de60c7954a2c671c427012` |
| five-result manifest | `a23d19892f112645a7d3b5401d1528a0eb612a5d6db26520fe3514246c4c6d1a` |
| twenty-checkpoint manifest | `e83832872f29d072d710859e022f4e17d1b6da6a9e16b63049f41d4ea2eb01a0` |
| producing runner | `7f4a5990f2f9a56bcaad0032d7cf9eca20f74b599a0684337c48dcdf9593b3ed` |
| function-class capacity result | `6a01db25ebc2ed15d202884c39f16db685d5218647b0bb209e2e5a737696a383` |

Validate the complete campaign, all result files, all checkpoints, source
identities, per-arm exact resume, and population gates before reading weights.
Any mismatch is `invalid_source_contract`.

## Exact response decomposition

For an encoder with shared features `f_k(x)` and complex mixer
`m_k = m_k^R + i m_k^I`, define its scalar complex response

```text
g(x) = sum_k f_k(x) m_k.
```

Character projection and mixer application commute, so the pre-normalization
carrier at each time is exactly

```text
z(theta) = sum_c g(x_c(theta)) exp(-2 pi i c / 3).
```

For each seed and arm, reconstruct the registered 4,096-example training
cohort from its sealed seed and verify its tensor hash. Fit one target-free
complex affine response on all corrected scalar observations:

```text
g_aff(x) = alpha x + beta
```

by deterministic float64 least squares. Define

```text
g_res(x) = g(x) - g_aff(x).
```

The constant `beta` cancels under the nontrivial character in exact arithmetic.
No target, phase, analytic carrier, held-out metric, or seed outcome enters the
fit.

## Causal response patches

On the unchanged disjoint reference, composition, and extrapolation cohorts,
evaluate three response arms through the original normalization, cubing,
frozen temporal operator, and fixed interval decoder:

| Arm | Pre-character response | Meaning |
| --- | --- | --- |
| `full_replay` | `g` | source checkpoint replay |
| `affine_only` | `g_aff` | identity-character causal projection |
| `nonlinear_residual_only` | `g_res` | all source-fitted nonlinear response after removing the affine component |

At every cohort verify before normalization:

```text
z_full = z_aff + z_res
```

to maximum error `1e-6`. Report coefficient magnitude and the fraction below
`1e-6` for every arm; the existing encoder clamp remains unchanged.

Fit one global `O(2)` carrier gauge per response arm on the disjoint reference
cohort and apply it unchanged to both primary shifts. Retain the sensor-only
carrier gates:

```text
mean aligned dot >= .90
aligned coordinate RMSE <= .35.
```

Retain the complete fixed task gate on both shifts:

| Endpoint | Composition | Extrapolation |
| --- | ---: | ---: |
| posterior-mean correlation | `>= .90` | `>= .90` |
| exact-bin accuracy | `>= .50` | `>= .35` |
| target cross-entropy | `<= 1.80` | `<= 2.20` |
| predicted-bin coverage | `>= 14` | `>= 12` |

A response arm passes a seed only when carrier and complete task gates pass on
both shifts. Structural exact-`C3` invariance is inherited from character
projection and cubing; report its numerical error but do not count it as
mechanistic specificity.

## Replay and controls

- `full_replay` posterior error relative to a direct call through the reloaded
  source encoder must be at most `2e-6` for both shifts in all ten checkpoints;
- every recomputed full-replay task metric must match the sealed result within
  `2e-6` (integer coverage must match exactly); the source campaign stores
  metrics and replay hashes rather than posterior arrays, so a cross-file
  posterior comparison is unavailable;
- each encoder state digest must remain unchanged before and after analysis;
- the source-fitted affine slope magnitude must be at least `1e-6`;
- analyze both `learned_true` and the matched
  `learned_target_shuffled` checkpoints;
- no response coefficient, gauge, or threshold may be refit by shift.

The target-shuffled checkpoints are the primary specificity control. No new
random direction is needed because the question concerns the source-trained
identity response versus its exact nonlinear complement.

## Population decision

The affine-identity mechanism is supported only if:

```text
true affine_only joint passes >= 4/5
true nonlinear_residual_only joint passes <= 1/5
shuffled affine_only joint passes <= 1/5
full replay valid in 10/10 checkpoints.
```

## Locked classifications

| Outcome | Classification | Decision |
| --- | --- | --- |
| affine identity passes the primary rule | `affine_identity_character_carries_learned_solution` | the 184-parameter sensors converged functionally to the five-parameter analytic mechanism; do not add sensor capacity |
| true affine `<4/5`, true residual `>=4/5` | `nonlinear_shared_response_required` | retain nonlinear sensor capacity and localize allowed harmonics before compression |
| true affine and residual both pass `>=4/5` | `affine_and_nonlinear_paths_redundant` | test coefficient-level necessity with a separately registered interpolation, not post hoc |
| shuffled affine passes `>1/5` | `affine_mechanism_specificity_failed` | do not interpret true affine success as target-specific |
| replay, source, reconstruction, state, or finiteness fails | `invalid_source_contract` | repair analysis only; draw no mechanistic conclusion |
| none of the above | `mixed_sensor_mechanisms` | report seedwise heterogeneity and stop universal compression claims |

Passing an affine patch does not prove the learned parameters equal the
closed-form witness. It establishes functional causal sufficiency of the same
identity-character response after a source-only projection.

Expected artifact:

```text
data/experiments/tinyllm_c3_temporal_sensor_mechanism/
  20260811_preregistered/campaign_results.json
```
