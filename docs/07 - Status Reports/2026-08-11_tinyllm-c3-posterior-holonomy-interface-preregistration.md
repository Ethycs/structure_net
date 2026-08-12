# TinyLLM C3 posterior-holonomy interface preregistration

**Status:** PREREGISTERED — DETERMINISTIC NO-TRAINING AUDIT

**Date:** 2026-08-11

**Hypothesis ID:** `tinyllm-c3-posterior-holonomy-interface-v1`

**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`

## Question

For an arbitrary posterior over C3 total-connection error, does all information
needed by the endpoint cosine task collapse to one complex character moment,
and can that soft interface be inserted exactly into every frozen relational
module without retraining?

The directional prediction is:

> Conditional on a charged endpoint relation `z`, the posterior
> `q_e = P(E=e | O)` is represented without loss by
> `m(O)=sum_e q_e exp(-2pi i e/3)`. The Bayes cosine prediction is
> `Re(z m)`, its ideal conditional risk is `(1-|m|^2)/2`, and a frozen linear
> neutral head consumes `z m` exactly as the posterior average of its three hard
> connection counterfactuals.

If supported, this does not license TinyLLM. It limits a future learned job to
estimating a two-real-number typed posterior moment and requires comparison
against fixed and low-dimensional adaptive estimators.

## Sources

The audit is downstream of the completed connection-observation identifiability
result:

```text
data/experiments/tinyllm_c3_connection_observation_identifiability/
  20260811_preregistered/result.json
SHA-256: 23a8989e820d73d1b72c8abaf3f5b4fde0664b854fb03a17ff6df3c5e2d24c7c
```

That result established that total holonomy is sufficient, every single-edge
erasure is non-identifiable in the current generator, and known independent
symmetric noise is analytically closed. This audit reuses its five frozen
`learned_true` source checkpoints and its deterministic fresh cohort streams
only to validate the soft interface against actual module code. The exhaustive
posterior-simplex algebra is the primary evidence.

## Declared algebra

Let

```text
omega = exp(2pi i / 3),
E in {0,1,2},
q_e = P(E=e | O),
m(q) = q_0 + q_1 omega^-1 + q_2 omega^-2.
```

For an observed noisy neutral character `z`, the three hard-error candidate
targets are

```text
y_e(z) = Re(z omega^-e).
```

The declared factorization is

```text
sum_e q_e y_e(z) = Re(z m(q)).
```

For C3, the first character moment is invertible over probability vectors:

```text
q_e = (1 + 2 Re(m omega^e)) / 3.
```

Under a uniform physical endpoint angle, the ideal squared-error Bayes risk is

```text
R_soft(q) = (1 - |m(q)|^2) / 2.
```

For any hard error choice `h`, its risk decomposes as

```text
R_hard(q,h) = R_soft(q) + |omega^-h - m(q)|^2 / 2.
```

Thus a posterior mean is never worse than hard connection selection and is
strictly better for every non-vertex posterior.

## Exhaustive deterministic grid

Enumerate the complete rational C3 posterior simplex with denominator `63`:

```text
q = (i/63, j/63, (63-i-j)/63),
i >= 0, j >= 0, i+j <= 63.
```

This gives `2,080` posterior points and includes all vertices plus the exact
barycenter `(1/3,1/3,1/3)`. Evaluate each posterior on a uniform `2,048`-point
phase grid over `[0,2pi)`.

Also test all three error-coordinate relabelings. If `E' = E+k` and
`z' = z omega^k`, then `m' = omega^-k m` and `z'm'=zm` must remain unchanged.

## Frozen-module replay

For source seeds `(1453,1471,1483,1531,1543)` and both previously declared
fresh shifts, regenerate the exact `1,024`-example cohorts and verify their
stored hashes. For each example, evaluate the three counterfactual total
connections `H-e` and the following six fixed posterior vectors:

```text
(1,0,0), (0,1,0), (0,0,1),
(1/3,1/3,1/3), (0.6,0.3,0.1), (0.1,0.2,0.7).
```

Compare the posterior-weighted average of the three hard module outputs with
the module's unchanged linear head applied to the soft neutral carrier `z m`.
No module parameter may change.

## Primary endpoints

The hypothesis passes only if all endpoints pass:

1. all `2,080` simplex points reconstruct from `m` with maximum probability
   error `<=1e-12`, sum error `<=1e-12`, and minimum reconstructed probability
   `>=-1e-12`;
2. conditional-mean factorization over all simplex/phase cells has maximum
   error `<=1e-12`;
3. the direct phase-averaged soft and hard risks match both closed forms within
   `1e-12`; soft regret is nonnegative everywhere and strictly positive at every
   non-vertex simplex point for its MAP hard choice;
4. all three coordinate relabelings preserve `z m` within `1e-12`;
5. all five frozen checkpoints on both shifts and all six registered posteriors
   match hard-output averaging to soft-carrier injection within `1e-6`, with
   exact source checkpoint, dataset, and predecessor hashes;
6. all values are finite, optimizer steps and probe fits are zero, no checkpoint
   state changes, and zero TinyLLM models are instantiated.

## Outcome meanings

| Outcome | Interpretation | Program action |
| --- | --- | --- |
| all gates pass | one complex posterior moment is the complete C3 soft-connection interface for the declared task and frozen linear neutral heads | permit only a compact posterior estimator after a new identifiable uncertainty-law preflight; require fixed/global/adaptive controls |
| simplex inverse fails | one moment is not sufficient even for C3 | stop and repair the representation claim |
| Bayes factorization or risk fails | the proposed uncertainty calculus is wrong | stop; do not train |
| frozen replay fails | the algebra does not match the implemented module path | diagnose head/transport behavior before designing an estimator |
| integrity fails | sources are not comparable | quarantine the result |

## Boundaries

The connection-error factorization conditions on the charged neutral character
`z`. Joint uncertainty in physical phase and connection may require estimating
the posterior moment of the physical relation directly. The first character
moment is probability-invertible specifically for C3; this statement is not
automatically true for larger cyclic groups. The audit does not train or test
an uncertainty estimator and does not establish TinyLLM utility.

## Planned artifacts

```text
experiments/structure_net/tinyllm_c3_posterior_holonomy_interface.py
tests/structure_net/test_tinyllm_c3_posterior_holonomy_interface.py
data/experiments/tinyllm_c3_posterior_holonomy_interface/
  20260811_preregistered/result.json
docs/08 - Analysis/2026-08-11_tinyllm-c3-posterior-holonomy-interface.md
```

Planned command:

```bash
pixi run python -m \
  experiments.structure_net.tinyllm_c3_posterior_holonomy_interface
```
