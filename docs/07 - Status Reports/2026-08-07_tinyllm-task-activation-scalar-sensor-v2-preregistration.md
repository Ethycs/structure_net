# TinyLLM task-activation scalar sensor v2 preregistration

**Status:** PREREGISTERED CORRECTIVE — FRESH-E OUTCOMES NOT GENERATED OR INSPECTED  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, frozen-checkpoint post-outcome mechanistic diagnostic  
**Hypothesis:** `tinyllm-c2-task-activation-scalar-sensor-v2`  
**Schema:** `nal.tinyllm-c2-task-activation-scalar-sensor.v2`

> Corrective sequencing note: v1 was superseded before outcome because it
> shared cohort D with `tinyllm-c2-observable-scalar-residual-v1`. That narrower
> protocol's systems-only shakedown was invalid before quality interpretation:
> its newly introduced observed-phase carrier violated source-known alignment
> and covector-replay contracts, confounding phase replacement with the scalar
> question. The combined protocol was consequently execution-locked before
> fresh-E generation. Version 2 supersedes both earlier protocols, keeps the
> proven oracle phase/covector fixed, changes only the scalar sensor, and
> reserves untouched cohort E (`630007/630008`). No
> cohort-E input, activation, fit, derivative, or outcome was generated or
> inspected before this document.

## Question and evidence boundary

The completed scalar task-covector campaign established a decomposition on a
fresh C cohort:

```text
portable phase-conditioned task covector g_hat(theta)
  + nonportable phase-only signed amplitude y_hat(theta)
  -> source-only correction fails 0/3 checkpoints.
```

The covector map achieved fresh zero-referenced `R2 0.989--0.996` and repaired
all six fresh cells when supplied the exact local signed error. The phase-only
signed-error map had near-zero fresh R2 and only `54.8--59.8%` sign agreement.

This experiment asks the shortest unresolved activation question:

> Do observable task-cut activations contain a source-portable estimate of
> the example-specific signed scalar multiplying the known task covector?

No TinyLLM checkpoint, front end, writer, decoder, carrier basis, or task
covector is trained or refit. Small scalar ridge maps and label-free PCA
summaries are fitted on source cohorts A/B only. Fresh-E outcomes have not been
generated or inspected.

## Locked sources

Freeze all three selected d6 degree-two checkpoints, source-fitted rank-three
carrier bases, order-four quotient writers, coordinate scales, and
phase-conditioned covector maps from:

```text
data/experiments/tinyllm_source_task_covector_portability/
    20260807_d6_preregistered_fresh_cohort/campaign_results.json
SHA-256 fe153abd0aa1749a862095e577e6e9cf8a0f7054bade2ef3c6833bf32d7f4ac5
implementation 6716b909d0c245059a1ed1310f20f4d9e56deb8c49a7d3a031972542fccb3046
```

The checkpoints `7`, `29`, and `53` are the replication units. Cohorts,
regimes, and orbits are repeated observations.

## Source and fresh cohorts

Fit scalar sensors on the predecessor's unchanged source cohorts:

```text
heldout_a x {composition, extrapolation}
heldout_b x {composition, extrapolation}
```

Use a new fresh-E cohort with seeds:

```text
composition   630007
extrapolation 630008
```

Each cell contains 64 exact `C2` orbits. These seeds have not appeared in the
alignment, A/B, fresh-C, exact-group, or preceding activation campaigns.

## Locked task decomposition

At the order-four predicted block-0 post-attention state, retain the validated
standardized finite-difference task covector `g` and signed circular output
error `y`. The minimum-norm scalar correction is

```text
delta c = g y / (g dot g).
```

The task-covector map `g_hat(theta)` is loaded byte-for-byte from the
predecessor result. It is not refit. Its oracle quotient-phase input remains a
method boundary; this experiment isolates the missing scalar rather than
claiming a deployable complete sidecar.

## Leakage boundary

Primary scalar features may use only quantities available at or before the
block-0 post-attention patch:

- observed calibration packet;
- the propagated Reynolds-barycenter state `F(b)` before the defect write;
- the order-four predicted post-attention state `F(b) + c4`.

The exact Reynolds defect, exact carrier coordinates, target posterior, target
bin, target phase, branch, exact post-attention state, exact post-MLP state,
and actual full-depth posterior are forbidden scalar inputs.

Two later-state arms are retained only as mechanistic look-ahead diagnostics:

- post-MLP state obtained by applying the frozen block-0 MLP to the order-four
  predicted state; and
- the frozen continuation posterior from the order-four predicted state.

Neither later arm may rescue the primary causal-availability claim. Success
there would motivate delayed feedback or a second pass, not a feed-forward
sensor at the patch cut.

## Source-fitted activation summaries

For each checkpoint and each activation cut, pool the 256 source A/B orbit
examples, center the query-token activation, and fit an eight-component PCA.
The PCA is label-free and source-only. Record orthogonality, explained energy,
source means, components, and hashes. Apply the fixed PCA transforms to fresh
E without refitting.

Use nested scalar feature arms:

| arm | inputs | role |
| --- | --- | --- |
| `phase_only` | order-four quotient Fourier chart | inherited negative comparator |
| `calibration` | observed calibration packet | acquisition-only comparator |
| `prewrite_activation` | eight PCA scores of `F(b)` | recipient-context test |
| `predicted_activation` | eight PCA scores of `F(b)+c4` | state-at-patch test |
| `causal_combined` | calibration + both primary activation summaries | **primary scalar sensor** |
| `post_mlp_lookahead` | primary features + eight predicted post-MLP scores | later block-0 diagnostic |
| `output_lookahead` | primary features + predicted posterior/task-confidence statistics | downstream diagnostic |
| `full_lookahead` | all declared summaries | descriptive upper diagnostic |

Every feature column is standardized using source A/B mean and scale only.
Fit one affine ridge scalar map per arm with ridge `1e-3`. No hyperparameter,
PCA rank, feature arm, or checkpoint is selected using fresh-E outcomes.

The predicted-posterior feature vector consists of the complete answer-token
posterior plus entropy, maximum probability, top-one/top-two margin, circular
moment cosine, sine, and radius. It contains no target comparison.

## Predictive endpoints

For every arm report source and fresh-E:

- zero-referenced R2;
- MAE and RMSE;
- sign agreement for examples with `|y| >= 0.01` bins; and
- prediction-to-target RMS ratio.

The primary `causal_combined` scalar sensor passes predictive portability only
when, in each checkpoint after pooling both fresh-E shifts:

```text
zero-referenced R2 >= 0.50
sign agreement >= 0.75
relative L2 <= sqrt(0.50).
```

The look-ahead arms are reported against the same thresholds but are never
part of the primary pass.

## Frozen causal endpoint

Patch these coordinate states into the same frozen continuation:

| state | correction | role |
| --- | --- | --- |
| `order4` | none | failed-writer reference |
| `direct_rank3` | exact rank-three defect | full-state control |
| `local_oracle` | local `g` with exact `y` | local positive control |
| `source_covector_oracle_error` | frozen `g_hat` with exact `y` | covector positive control |
| `phase_only` | frozen `g_hat` with phase-only scalar | inherited comparator |
| `causal_combined` | frozen `g_hat` with primary activation scalar | primary intervention |
| `post_mlp_lookahead` | frozen `g_hat` with post-MLP scalar | delayed-information diagnostic |
| `output_lookahead` | frozen `g_hat` with output scalar | full-continuation diagnostic |
| `full_lookahead` | frozen `g_hat` with all features | descriptive upper diagnostic |
| `causal_shuffled` | primary scalar permuted across fresh orbits | correspondence control |
| `causal_flipped` | sign-flipped primary correction | sign control |
| `causal_random_direction` | norm-matched isotropic direction | direction control |

Use the existing continuous endpoint unchanged: alignment loss at most
`0.005`, mean circular shift at most `0.125` bins, p95 shift at most `0.50`
bins, resolved sampling, and preserved winding degree.

The primary causal scalar gate requires, per checkpoint:

1. `zero` fails and `exact`, `direct_rank3`, `local_oracle`, and
   `source_covector_oracle_error` pass both fresh-E cells;
2. `causal_combined` passes composition and extrapolation;
3. each shuffled, flipped, and random-direction control fails at least one
   fresh-E cell; and
4. each control's two-cell aggregate mean shift trails `causal_combined` by at
   least `0.05` bins.

The full hypothesis requires both predictive and causal scalar gates in all
`3/3` selected checkpoints. The result remains underpowered regardless of
outcome.

## Numerical contracts

Require, before interpreting sensor outcomes:

- predecessor campaign, result, checkpoint, basis, writer, scale, and frozen
  covector hashes match;
- source and fresh fine/coarse task derivatives satisfy the predecessor's
  cosine and relative-L2 gates;
- every feature, PCA component, map, prediction, and intervention is finite;
- PCA orthogonality error is at most `1e-8`;
- each requested PCA rank is eight; and
- exact/direct controls pass while zero fails in both fresh cells.

Failure is `invalid`; no feature arm may rescue an invalid checkpoint.

## Fixed classifications

Apply the first matching rule:

1. `invalid` if any provenance, numerical, PCA, derivative, or target control
   fails;
2. `causal_activation_scalar_supported` if the primary predictive and causal
   gates both pass;
3. `causal_activation_predictive_not_causal` if only prediction passes;
4. `causal_activation_causal_not_predictive` if only causal patching passes;
5. `lookahead_scalar_only` if the primary fails but a declared look-ahead arm
   passes both its descriptive predictive and causal endpoints;
6. `observable_scalar_not_identified` otherwise.

No arm average, seed subset, source fit, or look-ahead result can rescue the
joint primary gate.

## Outcome meanings and stop rules

| Outcome | Meaning | Next action |
| --- | --- | --- |
| primary passes 3/3 | pre-write task activations carry a portable scalar error sensor | implement the smallest prospective sensor, then remove oracle phase |
| prediction passes, causal fails | scalar fit is statistically accurate but misses endpoint-critical tails | audit task-boundary weighting once; do not add capacity |
| only post-MLP/output passes | downstream computation exposes the error after the write | test one delayed-feedback/second-pass sensor, not a feed-forward sidecar |
| no arm passes | the signed amplitude is not linearly portable in these observable summaries | stop retrospective sidecar fitting and move to prospective equivariant training |
| invalid | digital/numerical contract failed | repair under a new root before interpretation |

## Artifacts and execution plan

- runner:
  `experiments/structure_net/tinyllm_task_activation_scalar_sensor_v2.py`
- tests:
  `tests/structure_net/test_tinyllm_task_activation_scalar_sensor_v2.py`
- systems-only root:
  `data/experiments/tinyllm_task_activation_scalar_sensor/20260807_v2_shakedown_cuda`
- primary root:
  `data/experiments/tinyllm_task_activation_scalar_sensor/20260807_d6_fresh_e`
- report:
  `docs/08 - Analysis/2026-08-07_tinyllm-task-activation-scalar-sensor-v2.md`
- meta hypothesis:
  `tinyllm-c2-task-activation-scalar-sensor-v2`

The one-checkpoint CUDA lifecycle uses disposable `systems_f` seeds
`730007/730008`, never cohort E, and is permanently excluded from scientific
gates. Focused CPU contracts, a real CUDA lifecycle, strict JSON, implementation
hashing, scientific fingerprints, immutable resume, DVC refresh, and lakeFS
backup are required.

## Method boundaries

The frozen covector still consumes an oracle quotient-phase chart. Source
scalar labels use exact diagnostic residuals. PCA and scalar maps are small
fitted observers, not natural network computations. Predicted post-MLP and
output arms are later than the patch and cannot support a one-pass causal
interface. Patches are local and off manifold. Three selected checkpoints do
not establish population prevalence.


## Corrective exclusions

The invalid observable-residual shakedown is systems-only and contributes no
quality evidence. Its cohort-D outcomes cannot set v2 features, thresholds,
PCA rank, ridge, controls, or endpoints. The superseded activation v1 runner
remains execution-locked and contributes no outcomes. Version 2 changes only
the cohort identity and removes the failed observed-phase substitution; every
scalar-sensor arm and gate above is otherwise locked before cohort E.
