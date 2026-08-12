# TinyLLM C3 posterior-holonomy interface

**Status:** VALID NO-TRAINING RESULT — EXACT MINIMAL SOFT CONNECTION INTERFACE

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`

**Hypothesis:** `tinyllm-c3-posterior-holonomy-interface-v1`

**Classification:** `posterior_holonomy_moment_exact_soft_interface`

**Preregistration:** [posterior-holonomy interface protocol](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-posterior-holonomy-interface-preregistration.md)

**Primary artifact:**
`data/experiments/tinyllm_c3_posterior_holonomy_interface/20260811_preregistered/result.json`

## Verdict

Arbitrary uncertainty about C3 total-connection error has an exact
two-real-number interface. If

```text
q_e = P(E=e | O),
m(q) = sum_e q_e exp(-2pi i e/3),
```

then, conditional on the observed charged neutral relation `z`, the Bayes
cosine prediction is exactly

```text
Re(z m(q)).
```

For C3, `m` is not merely a summary: it reconstructs all three posterior
probabilities. The entire 2,080-point rational posterior simplex reconstructed
within `3.16e-16`, and conditional means and squared-error risks matched their
closed forms over 4,259,840 posterior/phase cells within `3.34e-16`.

The interface also matches the implemented model path. Posterior-averaging the
three hard connection counterfactual outputs agreed with applying each frozen
module's unchanged linear head to the soft neutral carrier `z m` within
`2.38e-7` in all ten source-seed/shift cells.

This supports a compact uncertainty-interface design. It does not license an
estimator or TinyLLM training because no identifiable unknown uncertainty law
has yet been declared.

## Integrity

| Item | Result |
| --- | ---: |
| posterior simplex points | `2,080` |
| phase points | `2,048` |
| simplex/phase cells | `4,259,840` |
| frozen source checkpoints | `5/5` hash-verified |
| frozen replay cells | `10/10` |
| fixed posterior vectors per replay cell | `6` |
| checkpoint states changed | `0` |
| optimizer steps | `0` |
| learned probe fits | `0` |
| TinyLLM models instantiated | `0` |

The exact fresh datasets from the predecessor observation audit were
regenerated and matched their stored hashes. Their reuse validates the new
interface against the frozen implementation; it is not presented as an
independent task replication.

## Preregistered endpoints

| Endpoint | Observed maximum/minimum | Gate | Verdict |
| --- | ---: | ---: | --- |
| posterior reconstruction error | `3.16e-16` | `<=1e-12` | pass |
| posterior sum error | `4.44e-16` | `<=1e-12` | pass |
| minimum reconstructed probability | `-2.96e-16` | `>=-1e-12` | pass |
| conditional-mean factorization error | `3.33e-16` | `<=1e-12` | pass |
| soft-risk formula error | `2.22e-16` | `<=1e-12` | pass |
| hard-risk formula error | `2.78e-16` | `<=1e-12` | pass |
| minimum MAP-hard regret, nonvertices | `.00037793` | `>0` | pass |
| coordinate-covariance product error | `1.05e-15` | `<=1e-12` | pass |
| frozen hard-average/soft-injection error | `2.38e-7` | `<=1e-6` | pass |
| source/data/state/accounting integrity | pass | required | pass |

Every primary gate passed.

## The posterior triangle is the soft connection

The three exact-error states map to the three unit roots of C3. Their posterior
mixtures fill the equilateral triangle spanned by those roots:

```text
exact error known       -> |m| = 1
partial uncertainty     -> 0 < |m| < 1
uniform missing error   -> m = 0
```

The inverse is exact:

```text
q_e = (1 + 2 Re(m exp(2pi i e/3))) / 3.
```

Thus two real coordinates are both sufficient and minimal for a general C3
error posterior. Passing a scalar confidence alone would discard asymmetric
error direction; passing three unconstrained probabilities is redundant by one
dimension.

This is also the correct way to soften a discrete connection. Averaging edge
integers has no coordinate-invariant meaning. Averaging their characters does.

## Bayes prediction and hard-selection regret

For candidate targets

```text
y_e(z) = Re(z exp(-2pi i e/3)),
```

linearity gives

```text
sum_e q_e y_e(z) = Re(z m(q)).
```

Under the uniform physical angle law, the ideal Bayes risk is

```text
R_soft(q) = (1 - |m|^2) / 2.
```

Selecting one hard error state `h` instead has

```text
R_hard(q,h)
  = R_soft(q) + |exp(-2pi i h/3) - m|^2 / 2.
```

Therefore posterior transport is never worse under squared loss and is
strictly better whenever uncertainty is non-degenerate. The minimum strict
advantage on the denominator-63 grid was `.00037793`; only the three exact
posterior vertices had zero regret.

The prior known symmetric-noise result is the real-axis special case
`m=lambda(p)`. A completely missing uniform edge is the barycenter `m=0`.
Asymmetric or observation-dependent errors generally require both the real and
imaginary components.

## Exact covariance

Relabeling the error coordinate by `E'=E+k` rotates the moment by
`exp(-2pi i k/3)`. The corresponding observed neutral character rotates in the
opposite direction, so `z m` is unchanged. All three coordinate shifts passed
within `1.05e-15`.

This supplies the type contract for a future estimator: its output must
transform as the inverse charge-one character. An unconstrained scalar
confidence head is not the general object.

## Frozen implementation result

For every source module, shift, example, and registered posterior, the audit:

1. evaluated hard total-holonomy counterfactuals `H-e` for `e=0,1,2`;
2. averaged their scalar outputs with `q`;
3. formed the soft neutral carrier `z m(q)`;
4. passed that carrier through the unchanged source head.

The two paths agreed within `2.38e-7`, and all checkpoint state digests were
unchanged. This holds even for source seeds that failed calibrated task
acquisition: it is a structural property of the typed neutral carrier and
linear head, not evidence that those checkpoints solve the task.

## Program decision

The uncertain-connection architecture is now determined up to the estimator:

```text
exact charged endpoint carrier z
  + compact estimator of q(E | observed reliability evidence)
  -> typed complex moment m(q)
  -> soft neutral carrier z m(q)
  -> existing linear task head
```

Do not retrain the 187-parameter sensor or TinyLLM continuation to rediscover
this algebra. A learned campaign is still unlicensed until a new preflight
defines an observation law under which `q(E|O)` is statistically identifiable
and cannot be closed by a known-law rule, one global posterior, or a simple
adaptive calibration estimator.

Any such campaign must include:

- an oracle posterior-moment arm;
- the fixed known-law moment when available;
- a global learned posterior control;
- a low-dimensional observation-adaptive estimator;
- shuffled reliability evidence and target-changing controls;
- a Bayes-relative gate rather than the impossible clean-connection gate.

Only a remaining gap after those controls could license a compact
sequence-dependent estimator. TinyLLM would require a further causal
continuation advantage beyond accurate `m(q)` estimation.

## Boundaries

The factorization conditions on the charged neutral character. If physical
phase uncertainty and connection uncertainty are coupled, the directly
sufficient object is the posterior first moment of the physical endpoint
relation. The probability-invertibility of one nontrivial character is special
to C3 and does not automatically extend to larger cyclic groups. No estimator,
uncertainty law, natural task, or deployment robustness was tested here.

## Reproduction and provenance

```bash
pixi run python -m \
  experiments.structure_net.tinyllm_c3_posterior_holonomy_interface
```

| Artifact | SHA-256 |
| --- | --- |
| preregistration | `c95670b166b7ba1368b2241e0a17ea73c84bd07ec51d232b71ee09e0a5321e02` |
| runner | `8556317b17637e784eae0746335bcb4de6cbafb8fa0285508b003e37bef7708c` |
| result | `102125d3c465a30be64a51b6a3b3a59ebb8c350dfb92a562f72684431b4601fc` |
| predecessor result | `23a8989e820d73d1b72c8abaf3f5b4fde0664b854fb03a17ff6df3c5e2d24c7c` |

Focused verification completed as `6 passed, 18 warnings`.
