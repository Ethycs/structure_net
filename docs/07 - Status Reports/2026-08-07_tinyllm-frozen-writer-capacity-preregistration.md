# TinyLLM frozen quotient-writer capacity preregistration

**Status:** PREREGISTERED POST-OUTCOME DIAGNOSTIC — WRITER-LADDER OUTCOMES NOT INSPECTED  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, post-outcome diagnostic  
**Hypothesis:** `tinyllm-c2-frozen-writer-capacity-v1`  
**Schema:** `nal.tinyllm-c2-frozen-writer-capacity.v1`

## Known result and remaining ambiguity

The locked fixed-gauge oracle diagnostic established that exact
`(cos(2 phi), sin(2 phi), 1)` coordinates do not rescue a single linear,
nuisance-blind writer. The oracle coordinate fit remains strong, direct
rank-three patches pass, and all three checkpoint writers nevertheless fail
their complete held-out causal gate.

The untested ambiguity is now narrow:

```text
curved quotient-only write
    versus
quotient write conditioned on the invariant state already present at the cut.
```

This campaign must resolve that ambiguity with frozen checkpoints before any
sidecar is trained.

## Locked sources

Reuse the exact three degree-two checkpoints, source-selected rank-three bases,
alignment-fit cohorts, four held-out cells, calibrated discrete readouts, and
continuous endpoints from the fixed-gauge sequence. The direct predecessor is:

```text
data/experiments/tinyllm_fixed_gauge_error_decomposition/
    20260806_d6_preregistered_diagnostic/campaign_results.json
SHA-256 be4cf87248550c8ace7fc474efff2ce168bce3c9002932ecf6063977c91f0fa6
```

The selected checkpoints remain seeds `7`, `29`, and `53`. This is a
post-outcome, three-checkpoint mechanistic diagnostic and cannot establish
population prevalence.

## Writer ladder

Let the exact quotient angle be `theta = 2 phi`. For Fourier order `m`, define

```text
Phi_m(theta) = [1,
                cos(theta), sin(theta),
                ...,
                cos(m theta), sin(m theta)].
```

Fit no-intercept ridge maps from these features to the locked target
rank-three Reynolds-defect coordinates. The low-order semantic ladder is
`m = 1, 2, 3, 4`. Order one exactly reproduces the capacity class of the
failed oracle writer.

For the conditional branch, flatten the propagated Reynolds barycenter
`F(mean h)` at the intervention cut. Fit a rank-three PCA basis using only the
two alignment-fit regimes, standardize its scores using only those regimes,
and call the resulting invariant local context `c in R^3`. It is invariant to
sheet permutation and is already present in the frozen computation; it is not
a target, latent phase, or held-out fit.

The context-conditioned feature map is the tensor product

```text
Psi_m(theta, c) = Phi_m(theta) tensor [1, c_1, c_2, c_3].
```

Evaluate it for `m = 1, 2, 3, 4`. To prevent raw feature count from
impersonating conditioning, compare each context writer against the nearest
parameter-matched quotient-only writer:

| Context order | Context features | Matched Fourier order | Fourier features |
| ---: | ---: | ---: | ---: |
| 1 | 12 | 6 | 13 |
| 2 | 20 | 10 | 21 |
| 3 | 28 | 14 | 29 |
| 4 | 36 | 18 | 37 |

Every design receives a regime-preserving shuffled-correspondence writer with
identical features and regularization. No TinyLLM, front end, decoder,
representation, probe, or predictive observer is trained.

## Unchanged causal endpoint

At each of the four held-out cells, patch the predicted rank-three defect into
the exact target propagated barycenter and run the unchanged frozen
continuation. A writer passes a cell only when its continuous endpoint passes:

- circular alignment loss at most `0.005`;
- mean phase shift at most `0.125` output bins;
- p95 phase shift at most `0.50` bins;
- winding degree within `0.10` of degree two; and
- resolved sampling.

A writer passes a checkpoint only if all four held-out cells pass. The zero,
exact, and direct-rank-three controls must retain their locked outcomes, and
the predecessor order-one oracle metrics must replay within `1e-6`.

For a selected passing writer, specificity additionally requires its matched
shuffled writer to fail at least one held-out cell and to have aggregate mean
phase shift at least `0.125` bins worse. Coordinate `R2`, exact-bin accuracy,
and Fisher effect are reported but cannot replace the continuous endpoint.

## Fixed classifications

Classify each checkpoint by the first applicable rule:

1. `invalid` if provenance, replay, target controls, or numerical context
   contracts fail.
2. `low_order_curvature_sufficient` if some quotient-only order `m <= 4`
   passes with specificity.
3. `high_order_quotient_capacity_sufficient` if no low-order writer passes but
   a capacity-control Fourier writer in `{6,10,14,18}` passes with
   specificity.
4. `invariant_context_required` if a context writer passes with specificity
   while its parameter-matched quotient-only writer fails.
5. `context_helpful_not_decisive` if context improves the locked aggregate
   mean shift by at least `0.125` bins without passing all cells.
6. `small_writer_insufficient` otherwise.

The campaign-level conclusion is the common non-invalid classification only
if all three checkpoints agree. Otherwise it is
`checkpoint_stratified_writer_mechanism` with the per-seed classifications
reported explicitly.

## Decisions fixed before execution

- Curvature sufficiency implies a small nonlinear neutral writer is the next
  architectural unit; do not add nuisance context.
- Context requirement implies the writer must consume a typed invariant local
  state as well as the quotient carrier.
- A stratified outcome implies there is no single writer mechanism across the
  selected checkpoints and the sidecar must first fix the training gauge.
- Failure of every small writer ends sidecar fitting and redirects the next
  frozen study to the downstream nonlinear continuation.

## Planned artifacts

- runner:
  `experiments/structure_net/tinyllm_frozen_writer_capacity.py`
- tests:
  `tests/structure_net/test_tinyllm_frozen_writer_capacity.py`
- root:
  `data/experiments/tinyllm_frozen_writer_capacity/20260807_d6_preregistered_diagnostic`
- report:
  `docs/08 - Analysis/2026-08-07_tinyllm-frozen-writer-capacity.md`
- meta hypothesis:
  `tinyllm-c2-frozen-writer-capacity-v1`

## Method boundaries

The exact quotient angle uses latent phase and is diagnostic only. PCA and all
writers have alignment-fit access, patches are off manifold, the four held-out
cells have appeared in earlier post-outcome studies, and model selection uses
the preregistered nested pass rule rather than a fresh confirmatory cohort.
The experiment does not test a learned sidecar, a new architecture, or
population prevalence.

## Amendment A — immutable completion root

**Recorded:** 2026-08-07, after inspecting the completed seed-7 record and
before inspecting or producing seed-29, seed-53, or aggregate outcomes.

The initial primary root contains one completed seed-7 record but no campaign
aggregate. The frozen runner validates complete aggregates but does not skip an
already completed per-seed record while an aggregate is absent. Resuming the
three-seed command in that root would therefore overwrite seed-7's terminal
evidence, contrary to the append-only artifact contract.

The incomplete root is preserved unchanged and excluded from the campaign:

```text
data/experiments/tinyllm_frozen_writer_capacity/
    20260807_d6_preregistered_diagnostic
```

The complete campaign will use a new root:

```text
data/experiments/tinyllm_frozen_writer_capacity/
    20260807_d6_preregistered_diagnostic_v2
```

No checkpoint, cohort, seed, feature, writer, ridge, threshold, control,
classification, implementation, or scientific fingerprint changes. The same
producer digest `d53edaedd49ae553af9f8393d92254664239e5100246ac0fd3a06cb420ca80ed`
must produce the replacement root. The observed seed-7 classification was
`small_writer_insufficient`; it remains visible in the preserved partial root
and cannot be presented as fresh evidence in the complete rerun. The original
per-seed record is not pooled with, copied into, or used to aggregate the v2
campaign.

## Amendment B — correction to Amendment A

**Recorded:** 2026-08-07, after both campaign roots completed and after their
classifications were inspected.

Amendment A was based on a directory snapshot taken while the original
three-seed job was still running. That job had written seed 7 and was finishing
seeds 29 and 53 in the background. It completed the preregistered root at
2026-08-07 00:12:35 local time, before Amendment A was recorded at 00:14:25.
No terminal result in the original root was overwritten.

Therefore Amendment A's premise that the original campaign was interrupted is
incorrect, and its proposed replacement of the primary root is canceled. The
original preregistered root remains the primary evidence:

```text
data/experiments/tinyllm_frozen_writer_capacity/
    20260807_d6_preregistered_diagnostic
```

The independently completed `..._v2` root used the identical configuration,
producer digest, scientific fingerprints, cohorts, and gates on a second GPU
after seed-7 and campaign outcomes were available. It is retained as a
post-outcome deterministic hardware replication and is not pooled with the
primary campaign. This correction changes no scientific endpoint or
classification. Both roots and the mistaken Amendment A remain visible.
