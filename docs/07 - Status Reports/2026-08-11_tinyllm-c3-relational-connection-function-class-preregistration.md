# TinyLLM C3 relational connection function-class preregistration

**Status:** FROZEN BEFORE FRESH GRADIENT AND CUDA LIFECYCLE STREAMS

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `DESIGN / NO-TRAINING FUNCTION-CLASS AND LIFECYCLE PREFLIGHT`

**Hypothesis:** `tinyllm-c3-relational-connection-function-class-v1`

## Decision question

The relational preflight established that an observed `C3` edge connection is
necessary and sufficient for `cos(theta_7-theta_0)`, while every pointwise or
connection-free control fails. Before training any learned arm, ask the cheaper
constructive prerequisite:

> Does a minimal exact-equivariant connection-conditioned function class
> contain the analytic transport solution, remain invariant for every allowed
> parameter state by construction, expose a specific downhill true-task
> gradient, and survive checkpoint and CUDA lifecycle boundaries?

This preflight performs no optimizer step and instantiates no TinyLLM.

## Frozen module

Reuse the existing shared scalar sensor map and first-character construction,
but retain its charged character rather than cubing it. The encoder has:

```text
shared 1 -> 16 -> 8 GELU channel map
first C3 Fourier character per learned channel
learned complex mixing
unit-magnitude normalization
```

It has exactly `184` parameters. A three-parameter linear head consumes the
neutral transported pair `(real, imag)`, for `187` total parameters.

For corrected channels `x_t`, the shared map commutes with channel
permutations. Taking the first Fourier character therefore gives

```text
E(roll_g x_t) = rho_1(g) E(x_t)
```

for every parameter state. With observed connection `a`, form

```text
r = E(x_7) exp(+2*pi*i*sum(a)/3) conjugate(E(x_0)).
```

Under an arbitrary local action `h`, the endpoint characters and edge
connection transform oppositely, so `r` is unchanged. Any learned function of
`(Re r, Im r)` is consequently invariant. The trainable head cannot route
around this neutral product.

## Closed-form function-class witness

Use the exact identity

```text
GELU(x) - GELU(-x) = x
```

to assign, without fitting:

1. two first-layer weights `+1,-1`;
2. two second-layer weights `+1,-1` into character channel zero;
3. one real character mixer weight `1`;
4. the head's real neutral-product weight `1`.

Every other parameter is zero. The six-nonzero-parameter state implements the
analytic charged endpoint transport.

Regenerate the ten sealed relational-preflight cohorts and require their exact
dataset hashes to match the predecessor result. On every cell, the witness
must have:

```text
charged-character error          <= 2e-6
scalar correlation               >= .999
scalar RMSE                      <= .01
exact-bin accuracy               >= .98
target cross-entropy             <= 1.35
predicted-bin coverage           == 16
local-action prediction error    <= 2e-5
```

## Structural invariance states

On the fresh diagnostic stream below, evaluate:

1. deterministic random initialization;
2. that state plus deterministic nonzero parameter perturbations;
3. the closed-form analytic witness.

For all three states, the output change under an independently sampled local
action and covariantly transformed connection must be `<=2e-5`. Charged
encoder covariance must also be `<=2e-5`. For the random state, true-task loss
must change by at most `5e-6` and any parameter gradient by at most `2e-5`.

These numerical checks audit the implementation. The all-parameter claim is
the architectural Fourier/neutral-product identity above, not an inference
from three sampled parameter states.

## Fresh gradient and controls

Use a separate `512`-example composition-law diagnostic cohort:

```text
parameter seed:             62,117
dataset seed:               1,143,107
local-action seed:          1,145,107
connection-shuffle seed:    1,147,107
target-shuffle seed:        1,149,107
perturbation seed:          63,103
```

No seed or example from the unregistered pilot may enter the result. Use the
same phase, endpoint-relation, gauge, calibration, and quantization law as the
predecessor.

Differentiate the fixed sixteen-bin target cross-entropy through all `187`
parameters. The true-task route passes only if:

- all losses and gradients are finite;
- total gradient norm is at least `1e-6`;
- at least `.90` of scalar parameters have nonzero numerical gradient;
- one restored normalized negative-gradient perturbation of radius `1e-3`
  lowers true loss by at least `1e-4`.

For both a Sattolo-deranged connection and a separately Sattolo-deranged
target, require zero fixed points, gradient cosine with the true task at most
`.95`, and relative gradient-vector difference at least `.20`. These are
diagnostic specificity controls, not trained negative arms.

## Checkpoint and device lifecycle

Save the random module to a temporary checkpoint, load it with
`weights_only=True` into a new CPU module, and require:

- exact state-digest identity;
- tensor-exact CPU output replay;
- schema and parameter-count identity.

The host CUDA lifecycle is required. Load the same state on CUDA device zero
and require finite output, CPU/CUDA maximum output difference `<=5e-5`, local
action invariance `<=2e-5`, and state-digest identity after returning tensors
to CPU. The temporary checkpoint is a lifecycle artifact, not a historical or
trained checkpoint.

## Fixed source lineage

| Source | SHA-256 |
| --- | --- |
| relational preflight runner | `2ab21b7885fa93c49427973f67e5c57913b7f13c154ad2e970ef9636b2b90214` |
| relational preflight result | `ae0321b3c3d5b7e8c80ebbd57bfaea10c0522c4b2241082daab007a54e55984e` |
| relational preflight report | `4631f3ab2f99702e384d8b66c1dac4251cb63f4207ddbf0dca03cbe413a40aff` |
| existing exact-C3 sensor family | `dbc934190b5f725185ff9a99690409ac63fc4f16bd04686f870e7ef21cba03a6` |
| fixed interval likelihood | `b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6` |

## Locked classifications

| Outcome | Classification | Decision |
| --- | --- | --- |
| witness, structural invariance, gradient specificity, CPU replay, and CUDA lifecycle all pass | `connection_invariant_function_class_contains_transport_and_task_gradient` | license one matched sensor/readout acquisition campaign; TinyLLM remains excluded |
| mathematical/CPU gates pass but CUDA is unavailable or its lifecycle fails | `connection_function_class_valid_cuda_lifecycle_pending` | repair device lifecycle only; no campaign |
| witness fails with valid sources | `connection_function_class_missing_analytic_transport` | redesign before training |
| witness passes but true/control gradient gates fail | `connection_function_class_gradient_route_insufficient` | repair the differentiable interface before training |
| source, data-hash, shape, finiteness, or checkpoint contract fails | `invalid_connection_function_class_preflight` | infrastructure repair only |

`unrestricted_tinyllm_training_licensed=false` in every row. A positive result
licenses only a separately preregistered, five-seed learned sensor/readout arm
against fixed analytic, no-connection, connection-shuffled, and
target-shuffled controls.

## Scientific boundary

Passing would establish capacity, exact symmetry typing, local gradient
access, and executable lifecycle—not global trainability, sample efficiency,
robustness to missing/noisy connections, or utility beyond the fixed analytic
solution.

## Expected artifact

```text
data/experiments/tinyllm_c3_relational_connection_function_class/
  20260811_preregistered/result.json
```
