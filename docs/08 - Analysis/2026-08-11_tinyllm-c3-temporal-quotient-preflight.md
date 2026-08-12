# TinyLLM observable C3 temporal-quotient preflight

**Status:** VALID NO-TRAINING PREFLIGHT — PROSPECTIVE C3 TRAINING DESIGN LICENSED

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `DESIGN / NO-TRAINING GENERATOR PREFLIGHT`

**Hypothesis:** `tinyllm-c3-temporal-quotient-preflight-v1`

**Frozen design:** [observable C3 temporal quotient](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-temporal-quotient-preflight.md)

## Verdict

The redesigned `C3` task passes every pre-model contract. Its deck action is an
exact token permutation, its triple-angle target is invariant, and its cubic
Fourier carrier is nondegenerate and invariant while the centered raw Reynolds
mean is null. The carrier sequence predicts the future target under composition
and speed/amplitude extrapolation, but one instantaneous carrier state is
provably insufficient.

The locked classification is:

```text
c3_temporal_quotient_preflight_passed
```

This licenses a prospective matched TinyLLM design. It is not trained-model
evidence: zero optimizer steps ran and zero checkpoints were loaded.

## Why this task is genuinely new

The historical observable `C3` action attempted to rotate a decoded continuous
sensor and requantize it. That transformation did not reproduce the separately
quantized latent generator, even with latent phase supplied, so exact causal
front comparisons were not identifiable from one observed sheet.

Here `C3` acts by cyclic permutation of three explicitly observed channels.
Channel-separable quantization commutes with that permutation exactly. The
semantic target `cos(3 theta_future)` descends through the action, and the
calibration packet contains only amplitude, common offset, and common drift—not
phase, speed, branch, or target.

The invariant front end also does not emit the answer. It emits an eight-step
sequence of cubic Fourier states. Temporal evolution remains necessary to
predict the ninth state.

## Exact observable group action

Each regime contains 4,096 fresh eight-step, three-channel examples quantized
into 1,024 bins. Composition and extrapolation both pass:

| Contract | Composition | Extrapolation |
| --- | ---: | ---: |
| identity token errors | `0` | `0` |
| group-composition token errors | `0` | `0` |
| order-three token errors | `0` | `0` |
| stored action token errors | `0` | `0` |
| independently generated sheet mismatches | `0` | `0` |
| quantizer saturation | `0` | `0` |
| maximum target-invariance error | `5.468e-15` | `6.606e-15` |

The observed action therefore has the exact semantics required for future
causal patching. It does not inherit the old pre-quantization ambiguity.

## Invariant carrier geometry

The analytic carrier is the normalized first channel character cubed:

```text
c1(t) = sum_m y[t,m] exp(-2 pi i m / 3)
q(t)  = (c1(t) / |c1(t)|)^3.
```

| Measure | Composition | Extrapolation |
| --- | ---: | ---: |
| maximum deck-invariance error | `1.601e-15` | `1.644e-15` |
| minimum pre-normalization magnitude | `1.4910` | `1.4867` |
| real variance | `.49952` | `.49919` |
| imaginary variance | `.50039` | `.50081` |
| maximum centered raw-Reynolds norm | `1.282e-16` | `1.282e-16` |

The nonlinear invariant is well conditioned and uses both coordinates. The raw
linear orbit average contains no centered semantic carrier. This is a clean
character-neutral synthesis problem rather than a coordinate already present
in the Reynolds mean.

## Temporal sufficiency

The no-fit observed predictor estimates the character rotation from the last
two invariant states and advances it once:

```text
r        = q(7) conjugate(q(6))
q_future = q(7) r
T_hat    = Re(q_future).
```

| Regime | target correlation | target RMSE | shuffled correlation | shuffled RMSE |
| --- | ---: | ---: | ---: | ---: |
| composition | `.999940` | `.007755` | `.004378` | `.999982` |
| extrapolation | `.999931` | `.008258` | `-.000426` | `.994557` |

Both true endpoints exceed the locked `.99/.08` gates by a wide margin. Both
fixed derangements pass the `|corr| <= .10`, RMSE `>= .80` specificity gates.

## The front end is not the answer

The deductive witness uses speeds `+.15` and `-.15` with initial phases chosen
so the two histories end at the same triple-angle state:

| Measure | Result |
| --- | ---: |
| final carrier difference | `0.0` |
| full-history RMS difference | `1.32124` |
| future-target difference | `.68144` |

Thus `q(7)` alone cannot determine the task. The full temporal carrier contains
the missing dynamics. This avoids the exact degeneracy found in the previous
physical-cosine construction, where freezing the analytic scalar also froze the
answer.

## Scientific accounting

### What this establishes

- An observable finite-group action can commute exactly with tokenization when
  the action is represented as a channel permutation.
- The target descends through the `C3` observation relation without an
  orientation reference.
- A nonlinear character-neutral carrier can be invariant, nondegenerate, and
  temporally task-sufficient while the raw Reynolds mean is null.
- Instantaneous invariance and temporal task sufficiency are distinct.
- The new scope survives the required analytic positive control and shuffled
  specificity test before training.

### What remains untested

- Whether raw TinyLLM discovers the `C3` quotient.
- Whether a learned equivariant/invariant encoder approaches the analytic
  carrier under both shifts.
- Whether either representation is causally sufficient through the frozen
  continuation.
- Whether learned quotient synthesis is localized, distributed, or compact.
- Whether any result replicates across d6/d10 seeds.

## Decision

Freeze a prospective training design with three matched arms:

```text
raw channel tokens
vs fixed analytic C3-invariant carrier sequence
vs learned C3-equivariant/invariant sequence encoder.
```

Use d6 and d10, five seeds, identical examples/minibatches/optimizer, and joint
composition/extrapolation gates. The analytic arm is the positive control. The
learned arm is the scientific result. Retain exact action, target-changing, and
shuffled controls plus frozen causal patching at front-end, post-attention,
post-MLP, and full-depth cuts.

Do not assume a cubic Taylor mechanism merely because the group is `C3`. The
earlier character experiment showed that conjugate `r=1,2` modes permit neutral
quadratic coupling. Measure exact Reynolds/Jensen defects first and treat Taylor
order as a secondary diagnostic.

## Reproduction

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_temporal_quotient_preflight \
  --output /tmp/tinyllm-c3-temporal-quotient-preflight.json

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
pixi run pytest -q \
  tests/structure_net/test_tinyllm_c3_temporal_quotient_preflight.py
```

The single-thread environment is a systems lifecycle requirement. The first
test process hit a host illegal-instruction fault inside a vectorized CPU
trigonometric kernel before producing any outcome; rerunning the unchanged
design under the repository's established single-thread BLAS constraint passed
all three tests.

| Artifact | SHA-256 |
| --- | --- |
| frozen design | `1da4d34892df7a988cee3e42e44a90b94e3e2f74af7f982ec4b515c0d6fbe8e0` |
| preflight implementation | `812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118` |

The JSON output remains a disposable `/tmp` design artifact. No model evidence
or DVC data object was created.
