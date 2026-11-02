# TinyLLM scalar groupoid-defect preregistration

**Status:** COMPLETED — WRITER-ONLY REDUCTION REJECTED; TWO-TERM DEFECT IN `3/3`  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, post-outcome frozen-checkpoint diagnostic  
**Hypothesis:** `tinyllm-c2-scalar-groupoid-defect-v1`  
**Schema:** `nal.tinyllm-c2-scalar-groupoid-defect.v1`

**Measured report:**
[`2026-08-07_tinyllm-scalar-groupoid-defect.md`](../08%20-%20Analysis/2026-08-07_tinyllm-scalar-groupoid-defect.md)

## Decision question

The completed nuisance-scalar transformation study rejected an invariant
correction amplitude in all three checkpoints and all 24 exact observed-group
cells. Before training an action-conditioned sidecar, make the shortest no-fit
attribution:

> Is the correction's action dependence almost entirely the negative
> symmetry defect of the frozen order-four writer, or does the direct
> rank-three task state contribute a material second term?

This distinction determines whether a prospective sidecar has one stable
defect to predict or must model the difference between two independently
nuisance-sensitive computations.

## Locked sources

Reuse the completed transformation-law campaign byte for byte:

```text
data/experiments/tinyllm_nuisance_scalar_transformation_law/
    20260807_d6_existing_group/campaign_results.json
SHA-256 e1e21cf08b736547d8e77de0f15b5ac34b0a8a92ccba8afa19fa8dfb8f22b633
implementation c7ee01f82a257e779b8fc8a656321d0daf9e3a05df418aa97b74598b04de008d
```

It in turn fixes the exact-group source campaign:

```text
data/experiments/tinyllm_local_metric_field_transport/
    20260807_d6_fresh_cohort/campaign_results.json
SHA-256 2aa80cff5882cb79754d732e9012c810c2624934bc5be9e03c9e77de4f525f55
```

Reuse checkpoints `7`, `29`, and `53`, rank-three bases, order-four writers,
group seeds `430007/430008`, 64 exact C2 orbits, and the four amplitude,
orientation, offset, and composed actions. No model, writer, encoder, probe,
or observer is fit.

## Exact decomposition

For the frozen continuation, let

```text
p(x) = task angle from the order-four predicted carrier state
d(x) = task angle from the direct rank-three carrier state
y(x) = wrap_bins(d(x) - p(x)).
```

For each exact observed action `g`, define the action defects

```text
delta_p(g,x) = wrap_bins(p(gx) - p(x))
delta_d(g,x) = wrap_bins(d(gx) - d(x))
delta_y(g,x) = wrap_bins(y(gx) - y(x)).
```

The groupoid identity is

```text
delta_y(g,x) = wrap_bins(delta_d(g,x) - delta_p(g,x)).
```

This identity is a numerical contract, not the scientific hypothesis. The
scientific writer-defect hypothesis is the stronger approximation

```text
delta_y(g,x) ~= -delta_p(g,x),
```

which holds only when the direct-state action defect is negligible.

## Metrics

For every checkpoint, shift, and action report:

- RMS and p95 absolute values of `delta_y`, `delta_d`, and `delta_p`;
- direct-to-total RMS ratio `RMS(delta_d) / RMS(delta_y)`;
- exact groupoid reconstruction maximum error and relative L2;
- zero-referenced R2, relative L2, sign agreement, MAE, and correlation for
  `-delta_p` predicting `delta_y`; and
- the same predictive metrics for the sign-flipped `+delta_p` and a
  phase-matched nuisance-shuffled `-delta_p`.

Use the existing four-cycle nuisance permutation within each quotient phase.
It is a valid correspondence control here because the claim concerns the
example-specific action defect, not nuisance equivalence of a semantic state.

## Validity contracts

A checkpoint is valid only if:

1. transformation-law campaign/result/array, group campaign/result/array,
   checkpoint, basis, writer, and implementation hashes match;
2. regenerated input summaries exactly match the producing group campaign;
3. all predecessor input-pair, input-replay, local-linearization, and target
   controls passed;
4. every recomputed task angle and defect is finite; and
5. the exact groupoid reconstruction has maximum absolute wrapped error at
   most `1e-6` bins and relative L2 at most `1e-6` in every cell.

Failure is `invalid`, not mechanistic evidence.

## Writer-defect dominance gate

The direct term is negligible in a cell only when both hold:

```text
RMS(delta_d) / (RMS(delta_y) + epsilon) <= 0.10
p95(abs(delta_d)) <= 0.05 bins.
```

The writer-only approximation passes only when:

```text
R2(-delta_p -> delta_y) >= 0.90
relative L2 <= sqrt(0.10)
sign agreement >= 0.90 above 0.01 bins.
```

Specificity additionally requires both sign-flipped and phase-shuffled
controls to fail at least one predictive threshold in the cell, and their R2
to trail the paired writer approximation by at least `0.10`.

A checkpoint passes the primary gate only if the exact reconstruction,
direct-negligibility, writer-only prediction, and specificity conditions pass
all eight action cells. The hypothesis requires `3/3` checkpoints.

## Fixed classifications

Apply the first matching rule:

1. `invalid` if any provenance, input replay, finite, target-control, or exact
   groupoid reconstruction contract fails;
2. `writer_symmetry_defect_dominant` if the complete primary gate passes;
3. `writer_defect_descriptive_not_specific` if direct-negligibility and
   writer-only prediction pass but specificity does not;
4. `two_term_groupoid_defect` if exact reconstruction passes but any direct
   term is non-negligible;
5. `writer_only_reduction_failed` otherwise.

No average, action subset, shift subset, or checkpoint subset can rescue the
joint gate.

## Outcome-directed decisions

| Outcome | Consequence |
| --- | --- |
| writer defect dominant `3/3` | preregister one typed action-conditioned writer-defect head versus a parameter-matched untyped head |
| descriptive but nonspecific | do not train; the attribution is too generic for a typed mechanism |
| two-term defect | stop the frozen-writer sidecar branch unless a separate observable direct-state defect is justified |
| writer-only reduction fails | stop sidecar recovery and use the already validated calibrated invariant front end |
| invalid | repair only the digital/numerical contract under a new root |

## Artifacts

- runner:
  `experiments/structure_net/tinyllm_scalar_groupoid_defect.py`
- tests:
  `tests/structure_net/test_tinyllm_scalar_groupoid_defect.py`
- result root:
  `data/experiments/tinyllm_scalar_groupoid_defect/20260807_d6_existing_group_gauge_replay`
- CPU cross-device replay (audit only):
  `data/experiments/tinyllm_scalar_groupoid_defect/20260807_d6_existing_group`
- systems-only lifecycle root:
  `data/experiments/tinyllm_scalar_groupoid_defect/20260807_shakedown_cpu`
- report:
  `docs/08 - Analysis/2026-08-07_tinyllm-scalar-groupoid-defect.md`
- meta hypothesis:
  `tinyllm-c2-scalar-groupoid-defect-v1`

The producing runner must retain strict JSON, deterministic fingerprints,
source hashes, result/NPZ hashes, exact resume, and the zero-training evidence
boundary.

## Post-registration source and lifecycle amendment

The locked transformation-law source was superseded by its carrier-basis
gauge-replay repair before this experiment executed. The authoritative input is

```text
data/experiments/tinyllm_nuisance_scalar_transformation_law/
    20260807_d6_existing_group_gauge_replay/campaign_results.json
SHA-256 1e1795c931335c739a38550b85def7c4b442be39afcea76af8cd8491b1ff6589
implementation e92c3894b3e8820c0242edeba29e1f893f321e7f1dc20d75583c898ce7b1526f
```

This repair changes only the arbitrary rank-three basis gauge and adds its
explicit replay gate; the scientific cohort, exact angles, hypothesis, metrics,
thresholds, controls, and classification rules are unchanged.

A one-seed CPU lifecycle subsequently exposed seed-7 quality fields before the
three-checkpoint launch. No protocol choice or implementation changed after
that exposure. The measured report therefore labels the primary run sequential
confirmation rather than claiming independence from the lifecycle result.

## Method boundaries

The groupoid identity is algebraic; only term dominance and specificity are
scientific. The direct rank-three and order-four states are diagnostic frozen
patches, not native deployable interfaces. The exact group covers only scale,
planar orientation, constant offset, and their composition. Reusing the
previously analyzed cohort makes this a derived mechanistic diagnostic, not
independent fresh confirmation. Three checkpoints do not establish population
prevalence.
