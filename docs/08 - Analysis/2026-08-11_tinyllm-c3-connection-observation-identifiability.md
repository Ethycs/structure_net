# TinyLLM C3 connection-observation identifiability audit

**Status:** VALID NO-TRAINING RESULT — TOTAL HOLONOMY IS MINIMAL; ERASURE AND KNOWN NOISE DO NOT LICENSE LEARNING

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`

**Hypothesis:** `tinyllm-c3-connection-observation-identifiability-v1`

**Classification:**
`total_holonomy_minimal_known_noise_analytic_no_training_scope`

**Preregistration:** [connection-observation identifiability protocol](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-connection-observation-identifiability-preregistration.md)

**Primary artifact:**
`data/experiments/tinyllm_c3_connection_observation_identifiability/20260811_preregistered/result.json`

## Verdict

Missing or known-noisy connection data does not create a useful learned
successor under the current independent-phase C3 generator.

The seven-edge connection is redundant: the analytic witness and all five
frozen learned modules use it only through total holonomy

```text
H(A) = sum_t A_t mod 3.
```

Replacing every connection with a canonical vector containing only `H(A)`
changed no prediction in any of ten fresh composition/extrapolation cells.
But `H(A)` is also minimal for point identification. Erasing any one edge
admits an exact same-observation collision whose targets are `1` and `-1/2`.

For independent symmetric known edge noise, exhaustive enumeration agrees
with the closed-form attenuation

```text
lambda(p) = (1 - 3p/2)^7.
```

The optimal ideal continuous conditional mean therefore has irreducible RMSE
`sqrt((1-lambda^2)/2)`. The current joint scalar gate tolerates per-edge error
probability only up to `9.5247e-6`. This is an information limit, not an
optimizer defect that TinyLLM can repair.

## Campaign integrity

| Item | Result |
| --- | ---: |
| frozen source model seeds | `5/5` loaded and hash-verified |
| fresh evaluation cells | `10/10` complete |
| fresh zero-saturation / 16-bin cells | `10/10` |
| source result manifest | exact match |
| source checkpoint manifest | exact match |
| optimizer steps | `0` |
| learned probe fits | `0` |
| TinyLLM models instantiated | `0` |

The source acquisition remains the locked negative result
`exact_function_class_but_population_acquisition_unreliable`. This audit does
not alter its `1/5` primary gate or promote its post-outcome `4/5` readout
diagnostic.

## Primary endpoints

| Endpoint | Observed | Gate | Verdict |
| --- | ---: | ---: | --- |
| analytic clean endpoint cells | `10/10` | `10/10` | pass |
| maximum analytic or learned full-to-total prediction change | `0` | `<= 1e-6` | pass |
| exact single-edge erasure collisions | `7/7` | `7/7` | pass |
| erasure-witness target separation | `1.5` at every edge | `>= 1.49` | pass |
| known-noise enumerations | `5/5` | `5/5` | pass |
| maximum attenuation/prediction enumeration error | `1.11e-15` | `<= 1e-12` | pass |
| declared `p >= 1e-4` cells failing a current scalar gate | `4/4` | `4/4` | pass |
| source/data/accounting integrity | pass | required | pass |

Every preregistered endpoint passed. The hypothesis is supported under the
declared generator and symmetric-noise law.

## Total holonomy is sufficient

The module's only use of the connection is the transport phase determined by
the sum of its seven C3 edge values. The audit replaced each full vector by

```text
(0, 0, 0, 0, 0, 0, H(A)).
```

The maximum change was exactly zero for both the six-weight analytic witness
and every frozen learned checkpoint on both fresh shifts. The analytic witness
also retained its scalar/task gate in all ten cells. Its fresh scalar
correlation ranged from `.9999967` to `.9999976`, RMSE from `.001538` to
`.001822`, and exact-bin accuracy from `.9873` to `.9951`.

The learned metrics reproduced the source failure structure on disjoint data:
seed `1471` retained the calibrated solution, seeds `1483`, `1531`, and `1543`
retained highly correlated but mis-scaled outputs, and seed `1453` remained in
the wrong representation sector. Total-holonomy equivalence held for all of
them; it is architectural, not contingent on task success.

## One missing edge destroys point identification

For each edge index, the audit started from zero phase and zero gauge, then
added `2pi/3` to every phase and one modulo three to every gauge after that
edge. This suffix transformation produced:

```text
quantized tokens:       identical
calibration:            identical
all visible edges:      identical
erased edge:            differs by one C3 element
target pair:            1 versus -1/2
absolute separation:    1.5
```

All seven constructions passed exactly. Thus observing six of seven individual
edges is not a degraded but still point-identifiable version of this task. The
unobserved edge leaves the total holonomy undetermined, and the current
generator contains no temporal law from which it can be inferred.

This also identifies the minimal interface: the task does not require the full
edge path, but it does require the total holonomy or equivalent information.

## Known symmetric noise has an analytic ceiling

Each edge error was declared to be zero with probability `1-p` and `+1` or
`-1` with probability `p/2`. Enumerating all `2,187` seven-edge error patterns
gave:

| Per-edge `p` | `lambda(p)` / ideal correlation | Ideal Bayes RMSE | Correlation gate | RMSE gate |
| ---: | ---: | ---: | ---: | ---: |
| `1e-5` | `.9998950` | `.010246` | pass | fail |
| `1e-4` | `.9989505` | `.032388` | fail | fail |
| `1e-3` | `.9895471` | `.101972` | fail | fail |
| `1e-2` | `.8996086` | `.308791` | fail | fail |
| `5e-2` | `.5794182` | `.576314` | fail | fail |

The exact per-edge tolerances implied by the current scalar gates are:

| Gate | Maximum `p` |
| --- | ---: |
| correlation `>= .999` | `9.5279e-5` |
| RMSE `<= .01` | `9.5247e-6` |
| joint scalar gate | `9.5247e-6` |

At and above this limit, failure is mandated even for the ideal continuous
Bayes predictor. Training a larger model against the same noisy observation
cannot overcome that conditional uncertainty.

## Program decision

Close two apparent scope changes:

1. **Missing or partial connection under the present generator:** no training.
   The point target is not identifiable.
2. **Independent symmetric connection noise with known rate:** no training.
   The conditional mean and irreducible error are analytic.

A successor must change more than the corruption rate. The remaining honest
opening is an observation-dependent or unknown uncertainty law with repeated
structure from which the law or missing holonomy can be inferred. Before any
model optimization, it must show all of the following:

- the target is statistically identifiable from the new observations;
- a learned estimate can in principle improve on a fixed analytic or
  low-dimensional adaptive estimator;
- the evaluation gate is relaxed to the Bayes ceiling implied by the declared
  noise, rather than retaining an impossible clean-data threshold;
- the source checkpoints are used only if their exact-connection interface
  remains scientifically matched.

Even then, TinyLLM is not the default. Estimating a single global symmetric
noise rate only changes `lambda(p)` and is a scalar calibration problem. A
learned architecture becomes plausible only when uncertainty depends on the
observed sequence in a way a fixed transport cannot express.

## Boundaries

This audit does not cover correlated or observation-dependent errors, unknown
dynamics, repeated views, a task invariant to connection error, or a generator
whose temporal law makes missing edges inferable. It proves neither that all
connection learning is futile nor that the existing learned module is robust.
It establishes the minimal sufficient statistic and the information ceiling
for the two declared observation models.

## Reproduction and provenance

```bash
pixi run python -m \
  experiments.structure_net.tinyllm_c3_connection_observation_identifiability
```

| Artifact | SHA-256 |
| --- | --- |
| preregistration | `e2c3a75b50403bfcdf86d89db8b3fdb144ac6b324a89221de326023ec003b938` |
| audit runner | `a7ecb704dad4f472d57ee94dbda432a632ee9d5a4657dcbfb49307e5beedb819` |
| audit result | `23a8989e820d73d1b72c8abaf3f5b4fde0664b854fb03a17ff6df3c5e2d24c7c` |
| source acquisition campaign | `b4b6e94392a866f7a94188d18935db6602fbc83b936e14324b1060e56ce61a4a` |

Focused verification completed as `11 passed, 18 warnings`.
