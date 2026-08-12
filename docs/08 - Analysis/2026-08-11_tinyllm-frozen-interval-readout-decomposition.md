# TinyLLM frozen interval-readout decomposition

**Status:** VALID PREREGISTERED PARTIAL REPAIR; ARCHITECTURE-FAMILY CLAIM REJECTED

**Date:** 2026-08-11

**Hypothesis ID:** `tinyllm-frozen-interval-readout-decomposition-v1`

**Evidence role:** `prospective_frozen_backbone_closed_form_interface_fit`

**Preregistration:** [frozen interval-readout decomposition](../07%20-%20Status%20Reports/2026-08-11_tinyllm-frozen-interval-readout-decomposition-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_frozen_interval_readout_decomposition/20260811_d6_d10_preregistered/campaign_results.json`

## Verdict

A physically gauged scalar readout with the exact ordered interval chart
partially repairs the frozen architecture population, but it is not
architecture-family stable. It passes all analytic strata and improves the d6
learned stratum from `1/5` source seeds to `4/5`. The d10 learned stratum
remains `1/5`, so the preregistered requirement of at least four seeds in all
four strata fails.

The locked classification is:

```text
partial_frozen_interface_repair
```

This is not an answer-row-only failure. A free sixteen-logit readout is less
stable than the typed scalar readout and also passes only `1/5` d10 learned
seeds. Nor is it merely an input affine-gauge failure: mapping the learned
front-end scalar to physical cosine and feeding it through the old continuation
passes no learned seed in either architecture.

The sharper failure is **support-dependent coordinate calibration through the
learned interface**. Every typed final scalar remains highly correlated with
cosine under both shifts, including all failed d10 seeds, but its extrapolation
scale and offset are inaccurate enough to cross the fixed interval decision
boundaries. The representation can carry the right ordering while the
coordinate chart remains unusable at the required absolute calibration.

## Primary result

A seed passes the typed arm only when it meets its inherited task-accuracy
floor and scalar correlation `>= 0.90` on both composition and extrapolation.
The same seed set must supply at least four passes in each stratum.

| Preset | Source condition | Source task pass | Input affine gauge | Untyped final readout | Typed interval readout | Front-end typed bypass |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| d6 | analytic | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| d6 | learned equivariant | 1/5 | 0/5 | 2/5 | **4/5** | 3/5 |
| d10 | analytic | 3/5 | 4/5 | 5/5 | 5/5 | 5/5 |
| d10 | learned equivariant | 1/5 | 0/5 | 1/5 | **1/5** | 0/5 |

Every target-shuffled arm passes `0/5` in every stratum. All true-arm
improvements are therefore specific to the matched supervised coordinate or
posterior target under the declared fits.

The typed arm repairs five of the ten source failures:

- d6 learned seeds `7`, `17`, and `53`;
- d10 analytic seeds `17` and `29`.

It repairs none of the four failed d10 learned checkpoints. The untyped arm
repairs three source failures, the front-end typed bypass repairs four, and the
input affine-gauge arm repairs one.

## Task performance

Mean exact-bin accuracies show that both fitted final readouts help composition
substantially. Their learned-front-end extrapolation improvement is too small
and too seed-dependent for the inherited joint floors.

| Preset/condition | Arm | Composition | Extrapolation |
| --- | --- | ---: | ---: |
| d6 analytic | source | 0.7523 | 0.6289 |
|  | untyped final | 0.7895 | 0.6551 |
|  | typed interval | **0.7863** | **0.6566** |
| d6 learned | source | 0.7666 | 0.4734 |
|  | untyped final | 0.8580 | 0.4873 |
|  | typed interval | **0.8553** | **0.4924** |
| d10 analytic | source | 0.7289 | 0.6158 |
|  | untyped final | 0.7895 | 0.6486 |
|  | typed interval | **0.7902** | **0.6518** |
| d10 learned | source | 0.6926 | 0.3480 |
|  | untyped final | 0.8289 | 0.3836 |
|  | typed interval | **0.8367** | **0.3832** |

For d10 learned, the typed arm passes seed `17`. The other four seeds meet the
composition floor but miss extrapolation. Their typed extrapolation accuracies
are `0.3115`, `0.4863`, `0.3057`, and `0.2842` for seeds `7`, `29`, `41`, and
`53`, respectively.

## The typed scalar is ordered but not calibrated

The primary scalar-correlation conjunct does not cause the d10 failures. It
passes in every one of the forty typed readout regime cells.

| Preset/condition | Composition correlation, minimum | Extrapolation correlation, minimum | Mean extrapolation RMSE |
| --- | ---: | ---: | ---: |
| d6 analytic | 0.9978 | 0.9918 | 0.0694 |
| d6 learned | 0.9987 | 0.9531 | 0.1013 |
| d10 analytic | 0.9979 | 0.9918 | 0.0695 |
| d10 learned | 0.9970 | 0.9697 | **0.1146** |

Thus high correlation is not a calibrated interface contract. The frozen
state preserves a nearly monotone cosine coordinate while shift-dependent
absolute error moves examples across ordered answer bins. This also explains
why a free sixteen-row ridge head does not solve the population: it learns the
same source-supported final-state chart and extrapolates similarly.

The front-end bypass localizes the problem further. The analytic sensor plus
fixed interval decoder passes `5/5` in both architectures. The learned sensor
plus a source-fitted affine physical gauge reaches only `3/5` d6 and `0/5`
d10. The unstable physical calibration therefore begins no later than the
learned sensor output; it is not created solely by the tied LM head.

## Input gauge is co-adapted with the continuation

Forcing the natural learned scalar through a source-fitted affine map to
physical cosine, then using the original scalar embedding, transformer, and
answer rows, produces mean exact accuracy of only `0.030--0.070` in every
learned stratum and shift. The same arm remains adequate in both analytic
strata.

This is consistent with the preceding frozen interventions. The learned
encoder and continuation have co-adapted a checkpoint-local sign, scale,
offset, and decision convention. Correcting the input coordinate alone moves
the continuation off its learned chart. Correcting the output alone recovers
some systems, but leaves support-relative calibration in the larger learned
architecture.

## Controls and campaign integrity

All validity controls pass:

- `20/20` source cells completed with no retry, exclusion, or failed fit;
- all source model and complete-system hashes are unchanged;
- direct source output versus natural-scalar reinjection has maximum posterior
  error exactly `0.0`;
- replay against stored full-cohort task metrics has maximum error
  `1.9531e-7`, below `2e-6`;
- the exact-cosine fixed decoder replays generator posteriors with maximum
  error `4.2434e-7`, below `2e-6`;
- every coefficient and prediction array reloads exactly from the saved NPZ;
- all values are finite;
- all shuffled-arm stratum counts are zero.

The fit matrices have full observed rank. Their regularized normal-equation
condition numbers range from `1.61e6` to `3.36e6`; all solves use float64 and
the locked `1e-4` ridge coefficient. This numerical conditioning is a method
boundary, but it does not explain the typed/untyped agreement, the exact
controls, or the systematic learned-d10 extrapolation failure.

The campaign performs 120 deterministic closed-form fits over twenty frozen
checkpoints. It trains zero models or front ends and uses no iterative
optimizer. Aggregate cell time is `273.30` seconds, per-cell time is
`8.86--19.49` seconds, and maximum allocated CUDA memory is `0.4392` GB.

The first d8 lifecycle is preserved as invalid systems evidence because it
compared a reduced 64-example cohort with stored 1,024-example metrics. The
corrected d8 lifecycle uses direct-versus-injected replay, passes every
validity check, and resumes byte-identically. Neither lifecycle contributes to
the primary result.

## What the hypothesis earned

The ordered interval chart was a materially better constructive hypothesis
than the earlier cyclic answer projection:

- it repairs all analytic checkpoints;
- it raises the d6 learned population to the registered `4/5` gate;
- it recovers five source failures with only one physical scalar;
- it outperforms a more flexible free-logit readout on the d6 learned gate;
- matched shuffled targets reproduce none of the result.

It did not earn the architecture-family claim. A one-scalar coordinate can be
highly correlated, causally sufficient in the source system, and still fail as
a portable physical chart because its absolute calibration moves under a new
nuisance support.

## Next action

The preregistered stop rule licenses the prospective joint-interface branch.
Do not fit another endpoint-only map or change the ridge coefficient. The next
study should retain this frozen typed readout as the comparator and train only
the minimum interface needed to enforce physical calibration through the
learned path:

1. require the learned equivariant sensor output to predict physical cosine,
   not an arbitrary invariant scalar;
2. fix or supervise the scalar embedding/continuation convention so physical
   scale is not re-gauged between the sensor and final state;
3. retain the exact ordered interval decoder with full bin coverage;
4. freeze the large transformer backbone in the first arm, adding a
   full-interface fine-tuning arm only if the frozen-backbone joint interface
   fails;
5. use the same five seeds and require `4/5` jointly on composition and
   extrapolation, with an analytic sensor/decoder positive control and
   target-shuffled specificity.

The decisive question is no longer whether an answer chart can be repaired in
isolation. It is whether a declared physical scalar can remain calibrated from
observed sensor through continuation under extrapolation.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mplconfig pixi run python -m \
  experiments.structure_net.tinyllm_frozen_interval_readout_decomposition \
  --device cuda:0 \
  --output data/experiments/tinyllm_frozen_interval_readout_decomposition/20260811_d6_d10_preregistered
```

Exact resume leaves the completed campaign bytes unchanged.

| Artifact | SHA-256 |
| --- | --- |
| campaign | `3f15245386d1fb41e797f0688ff512aadc7c9690a552c3d58c8ba92754ee9208` |
| result manifest | `8a567df1952634a0be782b0dbd52e8dd92bd8d5d39f3f2d4f23c4f6df2a04638` |
| diagnostics manifest | `7110c85b90c96581e3d65aaffba0a2c2d829e763a69f7f2dd7eac40403759454` |
| producing runner | `bb6a73c203fcf4e654295bf7567f205826a6f054153ad50bbdd36851297de926` |
| preregistration | `e5e1ee52bdccb420403643cf071adf3a49c80854c70adf5c632bba56e683d62f` |
| combined implementation | `67e225b46a52d5724b84689dc411db507bc3d51dc5b0f81db0f0f707765b0d79` |
| composition dataset | `b025623e23f534ca3670f49f1bafc1c3979d1ca834aec2223f9c3166be8f52a6` |
| extrapolation dataset | `f2f1e8c2c06b38fbecf856f0679e9861d9fab3e1c2a41358e55043f4d309a214` |

## Data and evidence backup

The complete data tree, including the valid primary, preserved invalid and
corrected d8 lifecycles, twenty-result meta-hypothesis record, and Chroma
readback, is tracked by DVC root
`c4af030c66b90b2d87347d8936fdc765.dir`: `49,080,305,013` logical bytes
across 3,509 files. DVC reports the local cache and configured `lakefs` remote
in sync.

The uploaded objects are sealed as immutable lakeFS commit
`9a3fd4f4c462fbc4e4dc9270d4defca53dd3eb3fb333d9e3340b81ec0d824c77`.
The branch has no uncommitted `structure-net` diff. Direct metadata readback at
that commit verifies the DVC root checksum
`c4af030c66b90b2d87347d8936fdc765`, campaign checksum
`f0e668259c15d279646e27377503b985`, and meta-hypothesis checksum
`58ec089388fa9a299077cc5a1153365d`. Chroma readback verifies the stable
hypothesis ID and all twenty direct experiment records.

## Method boundaries

- Source checkpoint outcomes were known; the closed-form interface fits were
  preregistered before their d6/d10 outcomes were inspected.
- Cosine is a supervised generator-derived training target, not an
  unsupervised discovery.
- The typed readout tests affine accessibility and does not exclude nonlinear
  calibrated readouts.
- Correlation measures ordering, not absolute coordinate calibration.
- The d6/d10 presets jointly vary depth, width, and head count.
- The result is limited to the declared synthetic task and nuisance shifts.
