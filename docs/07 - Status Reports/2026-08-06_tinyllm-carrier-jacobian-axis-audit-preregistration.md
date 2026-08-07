# TinyLLM carrier-Jacobian axis audit preregistration

**Status:** PREREGISTERED — AXIS OUTCOMES NOT INSPECTED  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, post-outcome diagnostic  
**Hypothesis:** `tinyllm-c2-carrier-jacobian-axis-audit-v1`  
**Schema:** `nal.tinyllm-carrier-jacobian-axis-audit.v1`

## Evidence boundary

This study begins after the outcomes of the cross-seed carrier-transport
campaign were inspected. It reuses the same six directed checkpoint pairs and
24 held-out cells. The overall transport failure and the three canonical
correlations are therefore known. The local derivatives, axis-resolved causal
patches, attribution metrics, and axis classification declared below have not
been inspected.

The evidence role is
`preregistered_post_outcome_underpowered_mechanistic_diagnostic`. A passing
result can localize the known failure, but cannot independently confirm the
preceding transport result or establish population prevalence.

## Question

Is the failure of cross-seed causal-carrier transport concentrated in the weak,
checkpoint-local third canonical axis, or do the two statistically shared axes
remain causally misoriented?

## Fixed evidence and intervention

- frozen d6 calibrated degree-two TinyLLM checkpoints, seeds 7, 29, and 53;
- all six directed source-to-target maps from the authoritative transport
  campaign;
- the same held-out-A and held-out-B composition/extrapolation cells;
- the same source-fitted rank-three block-0 post-attention carrier bases;
- no training, probe, decoder, map refitting, threshold tuning, or held-out
  adaptation;
- the stored paired affine map is the transport intervention under audit.

For each directed pair, the alignment-fit coordinates are reconstructed only
to recover the target-side canonical-correlation frame. The recomputed affine
map and singular values must reproduce the stored map before any audit result
is accepted.

## Canonical-axis decomposition

Let `e = c_paired - c_target` be the held-out target-coordinate error. With the
target whitening matrix `W_t`, target coloring matrix `C_t`, and target
canonical vectors `V` from the alignment-fit SVD,

```text
e_can = e W_t V
e_j   = e_can[j] V[:, j]^T C_t.
```

The three raw-coordinate components must reconstruct `e` to relative error at
most `1e-8`. Axes 1--2 are the statistically shared subspace because their
previously observed canonical correlations exceed `0.987`; axis 3 is the weak
cross-seed direction (`0.128--0.281`).

## Frozen local derivative

At each target orbit and its direct target rank-three coordinate, estimate the
signed circular-moment derivative along each normalized target canonical axis
by centered finite differences. The primary step is `0.025` target standard
deviations; `0.05` is the convergence control. Derivatives are measured in
output-bin units per target standard deviation.

```text
J_j ~= wrap(theta(c + h a_j) - theta(c - h a_j)) / (2 h bin_width)
predicted transport error = sum_j J_j e_can[j].
```

The comparison is to the actual paired-versus-direct-rank-three signed moment
error. This isolates cross-seed coordinate error from the already measured
rank-three truncation error relative to the exact full defect.

## Causal axis controls

Each cell evaluates four frozen target continuations:

| State | Coordinate write | Meaning |
| --- | --- | --- |
| direct | `c_target` | zero transport-error reference |
| paired | `c_target + e1 + e2 + e3` | known transported write |
| shared-only | `c_target + e1 + e2` | remove weak-axis mismatch |
| third-only | `c_target + e3` | remove shared-axis mismatch |

Finite-difference plus/minus writes are derivative measurements, not learned
interventions. All continuations, frontends, and answer-token decoders remain
frozen.

## Pair-level primary endpoints

Metrics are pooled over the four held-out cells within each directed pair.

The local linearization is adequate only when all of the following hold:

1. coarse/fine derivative cosine is at least `0.98`;
2. coarse/fine derivative relative L2 difference is at most `0.15`;
3. zero-referenced signed-error R2 is at least `0.50`;
4. prediction residual MAE is at most `0.50` of observed paired-error MAE;
5. sign agreement is at least `0.75` on observations of magnitude at least
   `0.01` bins.

Within an adequate pair, axis 3 is classified as dominant only if:

1. its predicted RMS fraction is at least `0.60` of the sum of shared-axis and
   third-axis predicted RMS;
2. the actual third-only state retains at least `0.60` of paired mean absolute
   moment error; and
3. the actual shared-only state retains at most `0.40` of paired mean absolute
   moment error.

The shared axes are classified as dominant by the exact reciprocal rule:
shared predicted RMS fraction at least `0.60`, shared-only retention at least
`0.60`, and third-only retention at most `0.40`. All other adequate pairs are
classified `mixed`; inadequate local models are classified `nonlinear_or_unresolved`.

## Campaign decision

The **universal 2D base plus local scalar correction** hypothesis is supported
only if the local linearization is adequate in all six directions and axis 3
is dominant in at least five of six.

The **shared axes are causally misoriented** alternative is supported only if
the local linearization is adequate in all six directions and axes 1--2 are
dominant in at least five of six.

Any other result is reported as mixed or unresolved. Neither an average metric
nor one favorable checkpoint pair may rescue a failed campaign rule.

## Secondary measurements

- per-axis derivative RMS and predicted-effect RMS;
- paired, shared-only, and third-only signed-error MAE/RMS;
- canonical-coordinate residual RMS by axis;
- correlation and slope of predicted versus observed signed error;
- results separated by cohort and shift.

Secondary measurements cannot override the classification rule.

## Outcome interpretation

| Outcome | Interpretation | Next shortest action |
| --- | --- | --- |
| axis 3 dominates | a shared 2D semantic base coexists with a checkpoint-local high-sensitivity correction | test a typed 2D sidecar plus local scalar readout |
| axes 1--2 dominate | Euclidean/canonical alignment is semantically misoriented even in the shared subspace | test a group-anchored task-metric map |
| mixed across pairs | the carrier atlas is checkpoint-stratified | retain local charts and inspect pair-specific task metrics |
| local model inadequate | transport errors leave the local linear regime or the circular endpoint is too nonlinear | run a radius titration before fitting another map |

## Integrity, artifacts, and execution

Every result must record the producing implementation digest, source campaign
and pair-result hashes, checkpoint identities, scientific fingerprint,
canonical-frame reproduction error, strict JSON summaries, and zero trained
models/observers/maps. A shakedown is systems-only evidence and cannot enter the
scientific aggregate. Completed compatible records are byte-immutable.

- runner:
  `experiments/structure_net/tinyllm_carrier_jacobian_axis_audit.py`
- tests:
  `tests/structure_net/test_tinyllm_carrier_jacobian_axis_audit.py`
- primary root:
  `data/experiments/tinyllm_carrier_jacobian_axis_audit/20260806_d6_preregistered_diagnostic`
- report:
  `docs/08 - Analysis/2026-08-06_tinyllm-carrier-jacobian-axis-audit.md`
- meta hypothesis:
  `tinyllm-c2-carrier-jacobian-axis-audit-v1`

## Method boundaries

The task moment and frozen answer-token decoder define the Jacobian, so this is
a decoder-conditioned causal metric, not an intrinsic representation metric.
Canonical axes are source-pair dependent and may rotate under near-degenerate
singular values. Patches are off-manifold residual interventions. Reused
held-out cells make the study efficient and paired, but not an independent
replication. Three selected checkpoints remain an underpowered mechanistic
cohort.

## Amendment A — pre-outcome frame-reproduction tolerance

Recorded after the first CUDA systems shakedown failed its stored-map
reproduction check and before any axis derivative, counterfactual, or
classification result was written or inspected. Re-extracting the frozen
carrier and refitting the declared affine map differed from its stored JSON by
`1.05e-7`, reflecting numerical SVD/eigendecomposition replay rather than a
scientific mismatch. The frame-reproduction tolerance is fixed at `1e-6` for
the primary campaign. Coordinate-component reconstruction remains fixed at
`1e-8`, and no derivative, attribution, dominance, or campaign threshold
changes.
