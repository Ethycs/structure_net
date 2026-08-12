# TinyLLM observable C3 temporal-quotient preflight

**Status:** FROZEN BEFORE EXECUTION

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `DESIGN / NO-TRAINING GENERATOR PREFLIGHT`

**Hypothesis:** `tinyllm-c3-temporal-quotient-preflight-v1`

## Decision question

The calibrated `C2` future-cosine task is exhausted: exact physical typing
reduces its public scalar to the analytic answer. The next scope must retain an
exact observable symmetry without making the invariant front end itself solve
the task.

This preflight asks:

> Can an exactly token-observable `C3` action admit a nondegenerate invariant
> temporal carrier that is sufficient for a future task, while one instantaneous
> carrier state remains insufficient?

No TinyLLM checkpoint is loaded and no parameter is optimized. Training is
licensed only if every observation, group, identifiability, sufficiency, and
specificity contract below passes.

## Generator

For time `t=0,...,7` and channels `m=0,1,2`, generate

```text
y[t,m] = A cos(theta_0 + t v + 2 pi m / 3) + o + t d.
```

The target is one-step future triple-angle cosine:

```text
T = cos(3(theta_0 + 8 v)).
```

The nuisance deck element `j in C3` cyclically permutes the three channels at
every time. It changes the observed sheet but preserves `T`. Additional
target-free calibration reports only amplitude `A`, common offset `o`, and
common drift `d`; it does not report phase, speed, deck element, or target.

Each scalar channel is uniformly quantized into 1,024 bins on `[-4,4]`.
Tokenization is channel-separable, so a deck action is an exact permutation of
the token tensor rather than a latent continuous rotation followed by
requantization. This explicitly avoids the hidden pre-quantization action defect
that invalidated the historical observable `C3` constructor.

Use 4,096 fresh examples in each of two regimes:

| Regime | `|v|` | amplitude | offset | drift |
| --- | --- | --- | --- | --- |
| composition | `[.04,.12]` | `[.7,1.8]` | `[-.4,.4]` | `[-.06,.06]` |
| extrapolation | `[.13,.20]` | `[.5,2.2]` | `[-.7,.7]` | `[-.10,.10]` |

Signs, phases, calibration variables, and deck elements are independently
sampled from fixed preflight seeds. Require zero quantizer saturation.

## Exact group and target contracts

For every example and all `j,k in C3`, require:

- token action composition `g_j(g_k(x)) = g_(j+k mod 3)(x)` exactly;
- `g_0` identity and `g_1^3` identity exactly;
- latent generation at deck `j` equals the token permutation of deck zero;
- target change under any deck element at most `1e-12`;
- calibration packet unchanged under the deck action.

Any failure is `invalid_observable_group_action` and stops the study before an
encoder or model is considered.

## Analytic invariant carrier

After target-free offset/drift subtraction and amplitude normalization, form the
first channel Fourier coefficient

```text
c_1(t) = sum_m y[t,m] exp(-2 pi i m / 3).
```

Normalize it and expose the cubic carrier

```text
q(t) = (c_1(t) / |c_1(t)|)^3.
```

A channel shift multiplies `c_1` by a cubic root of unity, so `q` must be
exactly invariant. On both regimes require:

- maximum deck-invariance error `<=1e-6`;
- minimum carrier magnitude before normalization `>=.25`;
- both real and imaginary carrier variance `>=.20`;
- the centered raw Reynolds average over all channel permutations has maximum
  norm `<=1e-6`.

The final condition proves that the useful invariant is nonlinear/character-
neutral rather than present in the raw linear orbit mean.

## Temporal sufficiency and instantaneous insufficiency

Estimate the observed triple-angle step from the last two carrier states:

```text
r = q(7) conjugate(q(6))
q_future = q(7) r
T_hat = Re(q_future).
```

This uses only the observed invariant sequence. Require on each regime:

```text
corr(T_hat,T) >= .99
RMSE(T_hat,T) <= .08.
```

For specificity, compare the same predictions with a fixed derangement of
targets; require absolute correlation `<=.10` and RMSE `>=.80` in both regimes.

The front end must not already be the answer. Construct two noiseless histories
with the same final `q(7)` and speeds `+.15` and `-.15`. Require identical final
carrier within `1e-12` but future targets separated by at least `.25`. Their
full carrier histories must differ. This is a deductive witness that no function
of the instantaneous invariant state alone defines the future task.

## Locked classifications

| Outcome | Classification | Decision |
| --- | --- | --- |
| all contracts pass | `c3_temporal_quotient_preflight_passed` | freeze a prospective raw/analytic/learned `C3` training design; do not train in this preflight |
| action, tokenization, target invariance, calibration, or saturation fails | `invalid_observable_group_action` | repair the generator only |
| carrier invariance, nondegeneracy, or raw-Reynolds null fails | `c3_invariant_carrier_contract_failed` | reject this representation before training |
| temporal predictor misses either shift | `c3_invariant_carrier_not_sufficient` | reject this task/front-end pairing |
| one-frame witness fails | `c3_frontend_collapses_to_answer` | reject the design as another analytic-answer spine |
| shuffled specificity fails | `c3_temporal_specificity_failed` | reject the endpoint |

No threshold, regime, sample count, quantizer, or control may be changed after
execution to rescue the preflight.

## Conditional prospective campaign

Only `c3_temporal_quotient_preflight_passed` licenses a later training design.
That design must compare matched raw, fixed analytic-invariant, and learned
`C3`-equivariant/invariant sequence encoders across d6 and d10, five seeds,
composition and extrapolation. It must retain:

- exact observable action and target-changing controls;
- representation, natural-task, and frozen causal-continuation gates;
- carrier sequence cuts before TinyLLM and at post-attention/post-MLP/full
  depth;
- a declared character-coupling diagnostic without assuming cubic Taylor
  dominance merely because the group has degree three.

The preflight cannot count as evidence for that future trained-model claim.
