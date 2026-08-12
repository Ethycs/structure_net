# TinyLLM frozen scalar-interface decomposition preregistration

**Status:** PREREGISTERED — SOURCE OUTCOMES KNOWN; NO INTERFACE-DIAGNOSTIC OUTCOME INSPECTED  
**Date:** 2026-08-11  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, registered post-outcome
frozen causal diagnostic  
**Hypothesis:** `tinyllm-frozen-scalar-interface-decomposition-v1`  
**Schema:** `nal.tinyllm-frozen-scalar-interface-decomposition.v1`

## Decision question

The prospective architecture-family replication passed every structured
representation and causal-closure population but failed only the same-seed
natural-task adequacy conjunct. This diagnostic asks where that failure lies:

1. does the front end estimate the wrong scalar;
2. is the exact cosine expressed in the wrong scalar coordinate for the frozen
   continuation; or
3. can the frozen continuation and answer rows not express the correct answer
   for any admissible scalar input?

No TinyLLM, front end, embedding, probe, observer, or answer head is trained or
fit. Exact cosine and target labels are used only as declared causal oracles;
they are not proposed deployment inputs.

## Locked source

| Artifact | Digest | Role |
| --- | --- | --- |
| architecture-family campaign | SHA-256 `656d9814a032d1899810e81d398adf935cea3e1116712460e2062da188a0c9e2` | source outcome and task floors |
| 20 structured-result manifest | SHA-256 `f87740e70062f5fecf5238f00dd00774246e4f3e155dceb87752b099ce4ca80a` | source-result integrity |
| 60 structured checkpoint/front-end/diagnostics manifest | SHA-256 `5c08c771d04aae513ad9605d9e4818867ab0a8b0303680337dabbf87dce352e0` | source-artifact integrity |
| architecture runner | SHA-256 `661384de2eac23d95dbc550e0ecf49b14ebfeb01ef47fc1a8164e7e3b2b0ca90` | producing implementation |
| architecture preregistration | SHA-256 `36c1a8c35823fda3076b6a73648facd7fc18513c3c969bfa297ca9c0b34c4c77` | frozen source protocol |
| DVC data root | MD5 `e02d4354a7463797d6e7881d571c298c.dir` | complete source tree |
| lakeFS commit | `40140ec85a29a638924c56f62da14c4cdf4e9d4955000f91da2c6e64462e3660` | immutable backup |

The runner must revalidate the campaign, all 20 structured result files, all
60 declared source artifacts, source state digests, and the dataset hashes
before promoting a diagnostic cell.

## Frozen population

Use all 20 fresh structured checkpoints:

```text
presets       d6, d10
arms          analytic_calibrated, learned_calibrated_equivariant
seeds         7, 17, 29, 41, 53
```

Ten source cells fail task adequacy and are the primary localization units:

| Preset/arm | Failed seeds | Passing controls |
| --- | --- | --- |
| d6 analytic | none | 7, 17, 29, 41, 53 |
| d6 learned equivariant | 7, 17, 41, 53 | 29 |
| d10 analytic | 17, 29 | 7, 41, 53 |
| d10 learned equivariant | 7, 29, 41, 53 | 17 |

The remaining ten cells are resolution and replay positive controls. The
source failure labels are outcome-known strata; no cell is selected using the
new diagnostic.

## Locked cohorts and task gate

Regenerate the exact 1,024-example source task cohorts:

| Regime | Seed | SHA-256 |
| --- | ---: | --- |
| composition | 1399 | `b025623e23f534ca3670f49f1bafc1c3979d1ca834aec2223f9c3166be8f52a6` |
| extrapolation | 2408 | `f2f1e8c2c06b38fbecf856f0679e9861d9fab3e1c2a41358e55043f4d309a214` |

For every structured checkpoint, use the already sealed same-condition,
same-seed d8 exact-accuracy floor. A scalar intervention repairs a cell only
when exact-bin accuracy meets that floor separately on composition and
extrapolation. Circular error and target cross-entropy remain descriptive and
cannot rescue an exact-accuracy miss.

## Nested scalar interventions

All interventions replace only the one-dimensional input to the checkpoint's
existing `scalar_embedding`. The BOS token, query token, positional embedding,
TinyLLM blocks, final normalization, answer rows, and every parameter remain
unchanged.

### 1. Natural scalar replay

Run the ordinary analytic or learned-equivariant scalar through an explicit
scalar-injection continuation. Its posterior must match the system's natural
forward pass to maximum absolute error `<= 2e-6`, and its three task metrics
must reproduce the stored source metrics to `<= 2e-6`.

### 2. Exact latent cosine substitution

Replace the front-end scalar by the generator's exact task coordinate
`cos(phi)`. This tests whether eliminating sensor-estimation error is sufficient
for the frozen scalar embedding and continuation.

Two target-changing controls use the same path:

- negative cosine, `-cos(phi)`;
- a locked cyclic permutation of exact cosine across examples.

Neither control is a realistic corruption model. Their purpose is to show
that any repair depends on the correct semantic coordinate rather than merely
placing an in-range number into the embedding.

### 3. One-dimensional reachability oracle

For each distinct observed `(BOS, query)` context, evaluate the frozen
continuation on a candidate scalar set consisting of:

- 4,097 evenly spaced values on `[-1, 1]`;
- every natural scalar observed in that context;
- every exact-cosine value observed in that context.

The latter two sets are label-free and guarantee that natural and exact
positive controls lie in the searched set. For each example, record:

- whether any candidate makes its target bin the posterior argmax;
- the scalar with minimum target cross-entropy;
- the nearest scalar whose argmax is the target bin;
- distance from the natural and exact scalars to that reachable region.

The oracle is label-using by design. It tests only expressivity/reachability of
the frozen one-dimensional interface; it does not demonstrate an observable
or learnable correction.

## Cell classification

Apply this table independently to each of the ten source-failed cells, using
both shifts jointly:

| Exact cosine | 1-D oracle | Classification |
| --- | --- | --- |
| passes | passes | `sensor_scalar_estimation_failure` |
| fails | passes | `scalar_coordinate_or_boundary_failure` |
| fails | fails | `continuation_or_answer_row_failure` |
| passes | fails | `invalid_oracle_resolution` |

The first two labels are nested causal localizations, not deployment repairs.
If a source-passing positive-control cell is not recovered by the oracle, the
campaign is invalid rather than evidence for continuation failure.

## Population hypotheses

The source failure is **uniformly upstream of the scalar embedding** only if
exact cosine repairs:

- both `2/2` failed d10 analytic cells;
- at least `3/4` failed d6 learned cells;
- at least `3/4` failed d10 learned cells; and
- at least `8/10` failed cells overall.

The one-dimensional frozen interface is **expressively sufficient** only if
the reachability oracle repairs all `10/10` failed cells and recovers all
`10/10` source-passing controls.

At most one of ten source-failed cells may pass both shifts under negative
cosine, and at most one may pass under shuffled cosine. Otherwise the semantic
specificity gate fails and the campaign is invalid.

## Validity and state controls

1. Source campaign, result, checkpoint, front-end, diagnostics, implementation,
   and preregistration hashes must match before model evaluation.
2. Cohort hashes must match the source campaign.
3. Natural scalar injection must reproduce the direct natural posterior and
   stored task metrics to `2e-6`.
4. All natural and exact scalars must lie in `[-1-1e-6, 1+1e-6]`.
5. Every source-passing cell must remain reachable at or above its source task
   floor under the candidate-set oracle.
6. Model and complete-system state hashes must match their source digests before
   and after every cell.
7. No gradient graph, optimizer, fitted probe, or parameter write is allowed.
8. All stored numerical values and arrays must be finite; missing nearest
   reachable scalars are stored with an explicit Boolean mask, not infinity.

Any failure makes the affected cell invalid. Any source, cohort, replay,
resolution-control, state, or population-specificity failure makes the full
campaign `invalid`.

## Evidence and interpretation boundaries

This is a registered post-outcome diagnostic. It may localize why the known
architecture-family task gate failed, but it is not independent confirmation
of the earlier representation or causal-closure hypotheses.

The presets still co-vary depth, width, and head count. An oracle repair cannot
be described as deployable, observable, learned, invariant, or architecture-
portable. A grid-reachable answer does not show that optimization can find the
required coordinate.

## Lifecycle and stopping rule

1. Freeze this registration hash in the diagnostic runner.
2. Unit-test source manifests, cohort hashes, scalar injection, oracle
   reachability, classification, controls, state immutability, and exact resume.
3. Run one reduced d6/analytic lifecycle cell outside the primary root.
4. Run all 20 frozen cells once; do not select only the ten failures after
   diagnostic inspection.
5. Stop at the locked cell and population classifications. Do not fit an
   affine map, codebook, observer, readout, or new scalar head to rescue a miss.

Primary output root:

```text
data/experiments/tinyllm_frozen_scalar_interface_decomposition/
    20260811_d6_d10_preregistered/
```

After reporting and meta-hypothesis storage, refresh the existing DVC root,
push new objects to the configured remote, commit them on lakeFS, and verify
the campaign and meta-ledger objects at the immutable commit.
