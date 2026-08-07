# TinyLLM observable scalar-residual preregistration

**Status:** INVALIDATED BY SOURCE-ONLY SHAKEDOWN — SUPERSEDED BEFORE FRESH-E OUTCOME — DO NOT EXECUTE  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, sequential post-outcome mechanistic diagnostic  
**Hypothesis:** `tinyllm-c2-observable-scalar-residual-v1`  
**Schema:** `nal.tinyllm-c2-observable-scalar-residual.v1`

> Final lifecycle amendment: an obsolete one-checkpoint lifecycle run had already
> generated cohort-D files before the primary campaign. Its quality fields were
> not inspected during protocol selection, but D is nevertheless quarantined
> as non-fresh and excluded from every scientific gate. The primary cohort is
> `heldout_e` with previously unused seeds `630007/630008`. Before E was
> generated, the required source-only CUDA shakedown failed the observed-carrier
> and frozen-covector replay contracts. The combined protocol is therefore
> execution-locked and E remains untouched. The narrower
> `tinyllm-c2-task-activation-scalar-sensor-v2` corrective now owns E; it retains
> the validated oracle phase/covector and changes only the unresolved scalar
> sensor.

The source-only artifact is
`data/experiments/tinyllm_observable_scalar_residual/20260807_source_only_shakedown_cuda/campaign_results.json`
(SHA-256 `2742cf905e4a618bd009809d56de37f7f7369742be4b510257f630f55c5b8d63`).
Its carrier alignment was high (`0.99779`), but mean phase shift was `0.12828`
bins, maximum paired-sheet difference was `0.36106`, and covector replay
relative L2 was `0.06778`; the locked ceilings were `0.125`, `0.01`, and
`0.02`, respectively. These are systems/contract findings, not fresh-E quality
evidence.

## Question and prediction

The preceding fresh-cohort component test established that a source-fitted,
phase-conditioned task covector predicts the cohort-C continuation gradient at
`R2 0.989--0.996` and repairs all six cells when supplied the fresh signed
error. A phase-only signed-error map has near-zero fresh R2, so the fully
source-predicted correction passes only one of six cells.

This study freezes that successful covector and asks only:

```text
Can intrinsic continuation signals predict the missing one-dimensional
signed correction on a new cohort, without access to its exact carrier
coordinate, exact continuation, target posterior, target bin, or derivative?
```

The directional primary prediction is that adding intrinsic order-4 posterior
statistics to the existing observed phase chart produces a scalar correction
that passes fresh composition and extrapolation in all three checkpoints and
beats correspondence, sign, and random-sign controls. Calibration and local
activation contexts are declared secondary complexity rungs; they cannot
rescue a failed primary hypothesis.

No TinyLLM checkpoint, carrier basis, decoder, writer, or covector is trained
or refit. Three scalar ridge maps per checkpoint are fit on source A/B, plus a
phase-only replay map for comparison.

## Locked source and replication units

Freeze the source covector coefficients, checkpoint identities, and source
A/B split from:

```text
data/experiments/tinyllm_source_task_covector_portability/
    20260807_d6_preregistered_fresh_cohort/campaign_results.json
SHA-256 fe153abd0aa1749a862095e577e6e9cf8a0f7054bade2ef3c6833bf32d7f4ac5
implementation 6716b909d0c245059a1ed1310f20f4d9e56deb8c49a7d3a031972542fccb3046
```

Reuse the underlying order-4 writer campaign exactly as locked by that source.
The three d6 degree-two checkpoints `7`, `29`, and `53` are the replication
units. The campaign is underpowered and does not establish population
prevalence.

## Data and split contract

Each cell contains 64 exact paired C2 orbits from the unchanged generator.

| Role | Cohort | Composition seed | Extrapolation seed | Use |
| --- | --- | ---: | ---: | --- |
| writer alignment | `alignment_fit` | `130007` | `130008` | replay basis, writer, and scale only |
| scalar source fit | `heldout_a` | `230007` | `230008` | fit scalar maps and context standardizers |
| scalar source fit | `heldout_b` | `330007` | `330008` | fit scalar maps and context standardizers |
| prior fresh evidence | `heldout_c` | `430007` | `430008` | provenance only; never fit or evaluate this campaign |
| invalidated lifecycle | `heldout_d` | `530007` | `530008` | quarantined; never scientific evidence |
| new primary test | `heldout_e` | `630007` | `630008` | one fresh evaluation of all locked maps |

Seeds `630007/630008` were searched before generation and do not identify an
existing completed experimental cohort. Cohort-E exact residuals and derivatives may be
computed only for validity, oracle controls, and post-intervention evaluation.

## Observable phase and leakage boundary

Construct the neutral C2 carrier from the decoded sensor and observed
calibration packet using the existing analytic phase carrier and exact neutral
fusion:

```text
(x, y) -> (x^2 - y^2, 2xy, x^2 + y^2).
```

The two paired sheets must agree and the observed carrier must pass the
existing circular-alignment contract against the latent generator audit. Only
the observed carrier enters candidate features or the frozen covector map.

Candidate scalar features MUST NOT contain or derive from:

- the exact rank-three coordinate or its residual from order 4;
- the exact/full continuation posterior or angle;
- `target_posteriors`, `target_bins`, latent `phase`, latent `quotient_phase`,
  or branch identity;
- a cohort-E derivative; or
- a statistic selected after inspecting cohort E.

The scalar fit target on source A/B is the signed direct-rank-three minus
order-4 continuation-angle difference `y`. That value is unavailable to every
cohort-E candidate intervention.

## Nested scalar sensors

Let `q4` be the nine fixed Fourier channels through order four of the observed
C2 carrier. Let `p4` be the order-4 frozen-continuation posterior. Compute six
intrinsic posterior statistics without a target index:

1. normalized entropy;
2. maximum probability;
3. top-one minus top-two probability margin;
4. first circular-moment radius;
5. second circular-moment radius; and
6. distance of the first-moment angle from its nearest answer-bin boundary.

The calibration context is the existing observed eight-field packet averaged
over its exactly matched C2 sheets. It contains orientation cosine/sine,
signed speed, amplitude, two offsets, and two drifts.

For the activation rung, extract the query-token state at:

- the propagated Reynolds barycenter;
- the predicted order-4 post-attention state; and
- the deterministic post-MLP continuation of that predicted state.

Fit a rank-two PCA chart for each cut on pooled source A/B only. No exact
cohort-E activation participates in PCA or standardization.

Standardize each context on pooled source A/B and form a fixed tensor product
with `q4`. The nested feature widths are locked:

| Arm | Context beyond phase | Feature width | Evidence role |
| --- | --- | ---: | --- |
| `phase_only` | none | 9 | replay of failed scalar baseline |
| `posterior` | six intrinsic `p4` statistics | 63 | **primary** |
| `calibration` | posterior + eight observed calibration fields | 135 | secondary |
| `activation` | calibration + six source-PCA activation coordinates | 189 | secondary |

Fit each scalar map with deterministic ridge `1e-3`. These maps are nested but
not capacity matched; the scientific question is the least observable context
needed, not generic capacity. Report source fit and fresh prediction metrics
for every rung. A high-width secondary pass is not evidence that its named
context, rather than capacity, is uniquely necessary.

## Frozen cohort-E interventions

Evaluate the following coordinates at each fresh cell:

| State | Correction | Role |
| --- | --- | --- |
| `zero`, `exact`, `direct_rank3` | unchanged predecessor controls | target validity |
| `order4` | frozen failed coordinate | baseline |
| `local_oracle` | fresh local covector and fresh exact signed error | local replication |
| `frozen_covector_oracle_error` | frozen A/B covector and fresh exact signed error | covector portability replication |
| `<arm>` | frozen covector and source-fitted scalar from the named arm | candidate |
| `<arm>_shuffled` | within-cell permutation of the named scalar | correspondence control |
| `<arm>_flipped` | negative named scalar | sign control |
| `<arm>_random` | deterministic per-example norm-matched random sign | scalar-direction control |

Controls are evaluated for `posterior`, `calibration`, and `activation`.
Permutation and random streams are fixed from checkpoint and evaluation seed.
No cohort-E outcome chooses an arm, coefficient, context rank, threshold, or
control stream.

## Validity gates

A checkpoint is valid only if:

1. source campaign, results, implementation, checkpoint, basis, writer, and
   covector hashes replay;
2. observed-carrier alignment is at least `0.99`, mean shift at most `0.125`
   bins, p95 shift at most `0.50` bins, and maximum paired-sheet difference at
   most `0.01` in every source and fresh cell;
3. all fixed feature widths are exact, all source scales exceed `1e-8`, all
   PCA charts are rank two, and every value is finite;
4. source and fresh local-linearization gates reuse the established thresholds
   (`0.98` derivative cosine, `0.15` relative L2, `0.50` signed-error R2,
   `0.50` residual-MAE fraction, and `0.75` sign agreement above `0.01` bins);
5. the frozen observed-carrier covector differs from its latent-carrier replay
   by at most `0.02` relative L2 and retains at least `0.99` mean signed cosine;
6. `zero` fails while `exact` and `direct_rank3` pass in both fresh cells; and
7. `local_oracle` and `frozen_covector_oracle_error` pass both fresh cells.

A validity failure is not evidence against a scalar sensor.

## Primary endpoint and specificity

Use the unchanged joint continuous endpoint:

- circular alignment loss from exact at most `0.005`;
- mean circular-moment shift at most `0.125` output bins;
- p95 shift at most `0.50` bins;
- winding degree within `0.10` of degree two; and
- resolved sampling.

A checkpoint passes the **posterior scalar-sensor gate** only if:

1. all validity gates pass;
2. `posterior` passes both fresh composition and extrapolation cells;
3. each posterior shuffled, flipped, and random-sign control fails at least one
   fresh cell; and
4. posterior aggregate mean shift is at least `0.125` output bins lower than
   each of those controls.

The campaign supports the primary hypothesis only if all three checkpoints
pass. This deliberately retains the predecessor's conservative absolute
specificity margin. Secondary averages or larger-context arms cannot rescue a
failed primary gate.

For `calibration` and `activation`, report the identical arm-level endpoint
and specificity gate as planned secondary evidence. The phase-only arm reports
only its two-cell endpoint and fresh scalar prediction metrics.

## Fixed classifications

Apply the first matching row per checkpoint:

| Outcome | Classification |
| --- | --- |
| provenance, leakage, numerical, observed-carrier, local, or target controls fail | `invalid` |
| local or frozen-covector oracle fails | `fresh_local_mechanism_not_replicated` |
| posterior primary and specificity gates pass | `posterior_scalar_sensor_sufficient` |
| posterior endpoint passes but specificity fails | `posterior_sensor_nonspecific` |
| posterior fails and calibration secondary gate passes | `calibration_scalar_rescue` |
| posterior/calibration fail and activation secondary gate passes | `activation_scalar_rescue` |
| any secondary endpoint passes without its specificity gate | `secondary_sensor_nonspecific` |
| no observable candidate passes both fresh cells | `no_observable_scalar_sensor` |

If the nested outcomes do not match a preceding row, use
`mixed_scalar_sensor_geometry`.

## Outcome meanings and stopping rule

| Outcome | Meaning | Next action |
| --- | --- | --- |
| posterior primary `3/3` | the missing scalar is readable from the model's own continuation uncertainty | replace oracle phase with the existing analytic carrier in a complete sidecar test |
| calibration rescue | nuisance calibration determines error amplitude beyond posterior confidence | test a typed calibration-conditioned scalar head, not a new covector |
| activation rescue | the needed amplitude exists locally but is not output-observable | localize the smallest cut, then test one frozen scalar head |
| nonspecific pass | correction helps but the declared signal is not identified | prefer the simpler passing control and narrow the claim |
| no candidate passes | signed amplitude remains example-local under all declared observables | stop source-only sidecar recovery and retain the covector as an explanatory diagnostic |
| local mechanism fails | prior fresh-C effect is support-relative | audit cohort shift before any sensor or architecture work |

No new writer, richer covector, topology scan, residual penalty, or TinyLLM
training follows a negative result. A structured sidecar is justified only by
a specific fresh scalar-sensor pass.

## Artifacts and execution plan

- runner:
  `experiments/structure_net/tinyllm_observable_scalar_residual.py`
- tests:
  `tests/structure_net/test_tinyllm_observable_scalar_residual.py`
- primary root:
  `data/experiments/tinyllm_observable_scalar_residual/20260807_d6_preregistered_fresh_e`
- report:
  `docs/08 - Analysis/2026-08-07_tinyllm-observable-scalar-residual.md`
- meta hypothesis:
  `tinyllm-c2-observable-scalar-residual-v1`

A CUDA shakedown was source-only and did not generate or evaluate cohort E.
Because it failed prerequisite contracts, the fresh-E campaign is permanently
execution-locked. The preserved inert implementation must
preserve strict JSON, source and result hashes, producing-code digest,
scientific fingerprints, map coefficients, standardizers, PCA charts, feature
widths, control-stream digests, and immutable resume.

## Method boundaries

The frozen covector is decoder-conditioned and the intervention is local and
off manifold. Although its phase is computed from the observed sensor and
calibration packet, the propagated barycenter and exact C2 orbit chart require
paired counterfactual sheets; this is a mechanistic sensor test, not yet a
single-example deployment. The activation rung is not capacity matched to the
smaller rungs. Cohort E changes generator seeds within known shift families.
Source cohorts, checkpoints, cuts, and the covector were selected after prior
outcomes. Three checkpoints remain underpowered.
