# TinyLLM affine gauge and inverse-embedding transport

**Status:** VALID REGISTERED POST-OUTCOME NEGATIVE — TRAINING-COHORT AFFINE GAUGE IS SUPPORT-RELATIVE

**Date:** 2026-08-11

**Conformance:** NAL-STD-EXPERIMENT `REGISTERED POST-OUTCOME ARTIFACT-ONLY DIAGNOSTIC`

**Hypothesis:** `tinyllm-affine-gauge-transport-v1`

**Preregistration:** [affine gauge transport](../07%20-%20Status%20Reports/2026-08-11_tinyllm-affine-gauge-transport-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_affine_gauge_transport/20260811_d6_d10_registered/campaign_results.json`

## Verdict

The learned sensor's private coordinate is almost affine in physical cosine on
the sealed training cohort, and its affine gauge can be transported through the
scalar embedding without changing the complete trained function. That chart is
not sufficiently stable to repair the registered task under both held-out
shifts. The physical arm passes the joint seed gate in only `2/5` d6 seeds and
`1/5` d10 seeds; all pair-shuffled controls pass `0/5`. The locked
classification is:

```text
support_relative_affine_gauge_insufficient
```

This closes the possibility that the failed learned physical interface is only
a harmless checkpoint-local sign, scale, and offset convention. Across all ten
physical checkpoints, the training affine fit has `R2` between `.9939` and
`.9982`, and canonical-front cosine correlation remains above `.95` under
extrapolation. Nevertheless, the fixed sixteen-bin task endpoint is
support-sensitive: all ten physical cells pass composition, but only two per
preset pass extrapolation.

The result is corrective mechanism evidence. Full-interface outcomes and an
exploratory composition self-fit were known before registration. Only the
sealed 4,096-example training-cohort fit, unchanged held-out transport, and
algebraic inverse-embedding identity are registered here. No model was trained,
selected, or changed.

## Campaign integrity

| Check | Result |
| --- | ---: |
| source cells requested / completed / failed | `10 / 10 / 0` |
| physical / pair-shuffled arms | `10 / 10` |
| optimizer steps / trained parameters | `0 / 0` |
| exact sealed training-cohort affine fits | `20/20` |
| valid slopes and finite records | `20/20` |
| inverse scalar-embedding transport | `20/20` |
| maximum float32 transport error | `2.384e-7` |
| maximum float64 transport error | `2.220e-16` |
| physical joint passes | d6 `2/5`; d10 `1/5` |
| pair-shuffled joint passes | d6 `0/5`; d10 `0/5` |
| aggregate cell wall time | `26.00 s` |
| primary artifact size / files | `2.9 MB / 21` |
| exact campaign resume | byte-stable |

The exact resume revalidated the complete parent campaign and all derived cell
artifacts, reported the campaign already complete, and left the primary JSON at
SHA-256 `d4ff8b69474d09a47a767a655e8d89610f3e09a0415c0334882bc8ec2babb017`
before and after execution.

## Population gates

A physical seed passes only when the unchanged training-cohort affine law makes
the canonical front pass composition and extrapolation and the byte-identical
parent full-depth endpoint also passes both shifts.

| preset | canonical front | unchanged parent full depth | joint | shuffled joint | required |
| --- | ---: | ---: | ---: | ---: | ---: |
| d6 | `2/5` | `2/5` | **`2/5`** | `0/5` | `4/5` |
| d10 | `2/5` | `1/5` | **`1/5`** | `0/5` | `4/5` |

Both physical population gates fail. Both specificity gates pass. No
composition-only result, high training `R2`, or unchanged downstream output can
promote the registered hypothesis.

## Continuous fit versus discrete task transport

Five-seed physical-arm means are:

| preset | shift | canonical corr | canonical RMSE | exact-bin accuracy | endpoint passes |
| --- | --- | ---: | ---: | ---: | ---: |
| d6 | composition | `.9979` | `.0343` | `.8111` | `5/5` |
| d6 | extrapolation | `.9777` | `.1119` | `.4371` | `2/5` |
| d10 | composition | `.9988` | `.0262` | `.8467` | `5/5` |
| d10 | extrapolation | `.9806` | `.1056` | `.3992` | `2/5` |

Correlation is not the failed gate. Conditional branch accuracy and log-loss
gain were already below their registered ceilings and are invariant under an
invertible scalar affine change. The failure is the inherited exact-bin task
floor under extrapolation. A small continuous calibration error can cross a
discrete interval boundary even while order and branch contraction remain
excellent.

Seed-level physical results show that distinction directly. Accuracies are
reported as `measured / inherited floor`.

| preset | seed | training alpha | training R2 | composition accuracy | extrapolation accuracy | front / parent / joint |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| d6 | 7 | `-.1215` | `.9975` | `.8555 / .6233` | `.3760 / .4583` | no / no / no |
| d6 | 17 | `.6677` | `.9970` | `.8408 / .7591` | `.4521 / .3069` | yes / yes / yes |
| d6 | 29 | `.7740` | `.9939` | `.7402 / .7014` | `.5225 / .5208` | yes / yes / yes |
| d6 | 41 | `.8064` | `.9949` | `.7939 / .6487` | `.3652 / .4954` | no / no / no |
| d6 | 53 | `-.1603` | `.9970` | `.8252 / .7014` | `.4697 / .5266` | no / no / no |
| d10 | 7 | `-.1223` | `.9976` | `.8330 / .6233` | `.3193 / .4583` | no / no / no |
| d10 | 17 | `-.1338` | `.9981` | `.8643 / .7591` | `.5039 / .3069` | yes / no / no |
| d10 | 29 | `-.1577` | `.9963` | `.8105 / .7014` | `.5312 / .5208` | yes / yes / yes |
| d10 | 41 | `-.1736` | `.9982` | `.8691 / .6487` | `.2676 / .4954` | no / no / no |
| d10 | 53 | `-.1511` | `.9978` | `.8564 / .7014` | `.3740 / .5266` | no / no / no |

D6 retains the full-interface campaign's three positive and two sign-reversed
sensor orientations. D10 retains five sign-reversed orientations. Affine
canonicalization removes those conventions descriptively, but it does not make
the residual error harmless to the task.

## Exact gauge transport

For the saved scalar embedding `E(s) = W s + b` and the training fit
`s = alpha y + beta`, the audit constructs

```text
y_hat = (s - beta) / alpha
W_new = alpha W
b_new = b + beta W.
```

Therefore `E_new(y_hat) = E(s)` algebraically. The measured maximum float32
error across training, composition, and extrapolation is `2.384e-7`, more than
eight times below the locked `2e-6` ceiling. Float64 closes to `2.220e-16`.

This establishes an important distinction. Sign, scale, and offset are genuine
coordinate gauges of the learned interface: they can be moved between the
sensor and embedding without changing the model. The remaining task failure is
not explained by those affine degrees of freedom.

## Pair-shuffled specificity

The shuffled training targets have negligible in-sample relation to the saved
sensor. Training `R2` ranges from `.000003` to `.001256` in d10 and `.000071`
to `.001779` in d6. Their inverse affine maps amplify noise, giving mean
canonical-front RMSEs of roughly `106.5--110.5` for d10 and `25.6--26.0` for
d6. No shuffled composition, extrapolation, front-joint, parent-joint, or
complete seed endpoint passes.

The successful embedding identity is therefore generic algebra, while the
high-quality physical affine fit is semantically specific. The failed physical
population gate cannot be attributed to a broken transport implementation.

## Scientific accounting

### What this result rejects

- The learned interface failure is not only a checkpoint-local affine gauge.
- A high training `R2` and held-out correlation do not establish an adequate
  physical task chart.
- Exact inverse transport through the scalar embedding does not repair
  support-relative sensor error; it deliberately leaves the full function
  unchanged.
- Post-hoc sign correction, affine calibration, endpoint-map fitting, loss
  tuning, clipping, warm starts, more seeds, and further unfreezing are closed
  as explanations or repairs of this registered branch.

### What remains supported

- The learned sensor contains a strongly ordered, nuisance-contracted physical
  coordinate rather than an arbitrary scalar.
- Affine sensor/embedding gauge freedom is real and can be moved exactly
  without changing the computation.
- The physical correspondence is specific: pair-shuffled targets do not
  reproduce it.
- The remaining defect is support-relative continuous calibration coupled to a
  sensitive discrete readout, not loss of cosine order or branch collapse.

The evidence boundary is now:

```text
ordered invariant coordinate       supported
affine coordinate gauge            supported
portable physical chart            contradicted for the learned construction
portable exact-bin interface       contradicted under extrapolation
```

## Decision

Close further post-hoc scalar-map and optimizer studies on these checkpoints.
The constructive successor must type orientation, scale, range, endpoints, and
the scalar-to-bin convention in its function class. It should begin with a
no-training contract test showing that the declared analytic positive control
is representable exactly and that every allowed learned correction preserves
the physical chart. Only then is a prospective five-seed d6/d10 training study
licensed.

The theory-level alternative remains a declared new scope—a richer calibrated
group or a different identifiable task family—not another repair of this `C2`
cosine interface.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mplconfig \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m \
  experiments.structure_net.tinyllm_affine_gauge_transport \
  --output \
  data/experiments/tinyllm_affine_gauge_transport/20260811_d6_d10_registered

pixi run pytest -q \
  tests/structure_net/test_tinyllm_affine_gauge_transport.py \
  tests/neural_architecture_lab/test_affine_gauge_transport_meta_hypothesis.py
```

| Artifact | SHA-256 |
| --- | --- |
| campaign | `d4ff8b69474d09a47a767a655e8d89610f3e09a0415c0334882bc8ec2babb017` |
| result manifest | `600e71c8506d4ce15643a5f9c7b99ecccd496c86980ddefd9518508919078833` |
| derived diagnostics manifest | `7304fa026ea3dcdf928ef53c1098f79645e5b36007ffdf23a140d6e3756f9027` |
| implementation digest | `dfa81f86af3e30c0b24eba4a66cc25b5aa9afda9794565100610ae8ec3a7f89b` |
| campaign fingerprint | `56907481c9c3fb14f8272948e197494dc236b1433322f3d0bc3ec4da15d887a3` |
| runner | `94f5068644f8ba5c84ab74796219bfa37e036da47fec8334539a975913fad040` |
| preregistration | `aa7624e98859822592552e0d2a645c8e90a2946fdd5bcf64a72ef37f938a4813` |
| parent full-interface campaign | `cf8f27e088f9022b78f36d285f2ddb49920bd6bc740d71e6efac7b04ab877cc1` |

## Data and evidence backup

The complete repository data tree is tracked by DVC root
`acb8c1fc7631703c03b56e4b64efe5c8.dir` (`53,853,226,687` logical bytes,
`3,980` files). DVC pushed 40 new objects and reports the cache and configured
`lakefs` remote in sync.

lakeFS commit
`ba836be9cdc42c378500a9c06eac4c50c7b48ed9befbe572259d1826dc8e7d79`
seals the object graph on `artifacts/main`, with parent
`af24f6b57ba10d83dd08d86c144c3177c522741cca1370d47761359b4388c996`.
The branch diff is empty after commit. Direct immutable-object checks recover
the DVC root checksum `acb8c1fc7631703c03b56e4b64efe5c8`, campaign MD5
`4ecf1128379ab7343cbe9d458b0021c9`, and meta-record MD5
`de68fd953131e970fdd6234d286f4f9e`.
