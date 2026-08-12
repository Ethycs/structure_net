# TinyLLM C3 temporal continuation/readout decomposition

**Status:** VALID PREREGISTERED NEGATIVE — AFFINE FROZEN INTERFACES INSUFFICIENT

**Date:** 2026-08-11

**Hypothesis:** `tinyllm-c3-temporal-continuation-readout-v1`

**Classification:** `analytic_sensor_valid_frozen_continuation_not_affinely_typed`

**Preregistration:** [C3 temporal continuation/readout](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-temporal-continuation-readout-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_c3_temporal_continuation_readout/20260811_d6_preregistered/campaign_results.json`

## Verdict

The frozen normalized final query does not expose a population-reliable future
physical cosine through the registered affine interface. The typed final
readout passes the complete composition-and-extrapolation gate in only `1/5`
seeds. A free sixteen-logit final readout and a scalar recalibration of the
inherited posterior mean each pass `0/5`. Every matched target-shuffled control
passes `0/5`, and all five source states remain unchanged.

The exact analytic temporal bypass passes every endpoint. The locked result is
therefore:

```text
analytic_sensor_valid_frozen_continuation_not_affinely_typed
```

This does not mean the final state lacks target information. The typed scalar
has extrapolation correlation `.912-.980`, and its exact-bin accuracy exceeds
the locked `.35` floor in all five seeds. Four seeds fail only the registered
cross-entropy endpoint. The result establishes that the frozen continuation
does not expose the physical coordinate with enough outside-support precision
for the complete fixed metric interface—not that cosine is absent or
nondecodable.

## Population gates

| Arm | True joint passes | Shuffled joint passes | Population gate |
| --- | ---: | ---: | --- |
| inherited source output | `2/5` | not fitted | known baseline |
| output scalar recalibration | `0/5` | `0/5` | fail |
| free sixteen-logit final readout | `0/5` | `0/5` | fail |
| typed scalar final readout | `1/5` | `0/5` | **fail** |
| exact analytic temporal bypass | complete gate passes | no fitted arm | positive control |

The primary hypothesis required typed success in at least four of five seeds
and at most one shuffled success. Specificity passes; the true population gate
does not.

Every fitted true arm passes its complete composition gate in all five seeds.
No output-recalibration or free-logit seed passes extrapolation. Only typed seed
7 passes extrapolation. The failure is therefore support-relative rather than
an inability to fit the supported map.

## Typed final readout by seed

| Seed | Source joint | Typed joint | Extrap corr | Extrap RMSE | Extrap acc | Extrap CE | Failed typed endpoint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 7 | pass | **pass** | `.97935` | `.14937` | `.52930` | `1.87269` | none |
| 17 | fail | fail | `.90781` | `.34251` | `.35840` | `4.26156` | cross-entropy |
| 29 | fail | fail | `.96871` | `.18913` | `.48535` | `2.24158` | cross-entropy by `.04158` |
| 41 | fail | fail | `.96622` | `.19225` | `.45117` | `2.22983` | cross-entropy by `.02983` |
| 53 | pass | fail | `.93937` | `.25837` | `.41016` | `3.06498` | cross-entropy |

All five typed extrapolation cells cover all sixteen bins and pass the locked
correlation and exact-accuracy floors. No threshold is relaxed for seeds 29 or
41. In particular, the two near-boundary cross-entropies remain failures under
the preregistered joint rule.

## Aggregate task measurements

Values are means over five seeds.

| Arm | Shift | Accuracy | Posterior-mean corr | Cross-entropy |
| --- | --- | ---: | ---: | ---: |
| inherited source | composition | `.81738` | `.99889` | `1.29525` |
| inherited source | extrapolation | `.38066` | `.94376` | `2.40134` |
| output recalibration | composition | `.79551` | `.99817` | `1.29633` |
| output recalibration | extrapolation | `.36738` | `.94209` | `2.97715` |
| free final readout | composition | `.89766` | `.99953` | `1.27503` |
| free final readout | extrapolation | `.39922` | `.94775` | `3.06175` |
| typed final readout | composition | `.93457` | `.99951` | `1.27309` |
| typed final readout | extrapolation | `.44688` | `.95229` | `2.73413` |

The typed arm materially improves mean exact accuracy over the inherited output
on both shifts. Its extrapolation cross-entropy is worse on average because a
small physical-coordinate error is amplified by the narrow fixed interval
posterior. This is precisely why correlation or argmax accuracy alone was not
allowed to certify a typed metric interface.

## What the intervention localizes

### The inherited answer rows are not the sole failure

Replacing them with an unrestricted affine sixteen-logit map passes no seed.
The source output also cannot be repaired by one source-fitted affine scalar
calibration. This rules out the shortest answer-row and sign/scale/offset
explanations under the registered families.

### The continuation retains an approximate physical coordinate

The typed affine readout produces high cosine correlation and useful exact-bin
accuracy in every seed. Thus the analytic quotient is not simply erased. The
missing property is stable metric precision under the outside-range shift.
The final state carries a support-relative approximation that is adequate for
composition and often for argmax classification, but not for the declared
probabilistic chart.

### The analytic temporal solution remains valid

The checkpoint-free rule

```text
q_next = q(7) * conjugate(q(6)) * q(7)
```

followed by the fixed interval decoder reaches extrapolation correlation above
`.9999`, exact accuracy about `.959`, and cross-entropy `1.27524`. The observed
sensor, target, and fixed output chart are therefore jointly sufficient. That
positive control bypasses TinyLLM and cannot rescue the failed frozen-interface
hypothesis.

## Scientific accounting

### Supported

- The exact observable `C3` quotient remains valid and causally sufficient in
  the source population.
- The frozen continuation retains a repeatable, approximately affine physical
  target coordinate under both shifts.
- Composition fit is easy for every registered readout family.
- Outside-support metric precision, rather than branch leakage or loss of
  ordering, is the decisive failure in this population.
- The negative result is specific: every target-shuffled population is `0/5`.

### Not supported

- A free answer-row replacement is not sufficient.
- One affine calibration of the inherited posterior mean is not sufficient.
- An affine physical scalar from the final state is not population-reliable
  under the complete extrapolation gate.
- The result does not evaluate the stopped raw or learned-`C3` arms.
- It does not prove that no nonlinear decoder can recover the target.

## Program decision

Do not fit another endpoint map, broaden the fixed interval posterior, or tune
the ridge coefficient on these outcomes. The affine ladder has already tested
the inherited scalar, free answer rows, and a physically typed final scalar.
Four typed failures are specifically precision/calibration failures under
extrapolation.

For engineering, the fixed analytic temporal operator plus fixed interval
decoder is already the reliable path. For the scientific program, any learned
successor must move the known temporal law into the function class before
testing sensor learning. The clean next design is:

```text
analytic C3 carrier sequence
    -> fixed q7 * conjugate(q6) * q7 temporal operator
    -> fixed physical interval decoder
```

as the positive control, followed conditionally by a learned exact-`C3` sensor
feeding that same frozen operator and decoder. This tests acquisition of the
invariant carrier without asking an unrestricted transformer continuation to
invent a support-stable metric chart. It is a new hypothesis and requires a
new preregistration; the ten stopped raw/learned cells from the predecessor
campaign remain stopped.

## Integrity and lifecycle

| Check | Result |
| --- | ---: |
| requested/completed/failed cells | `5 / 5 / 0` |
| model optimizer steps | `0` |
| changed model parameters | `0` |
| source-state identity | `5/5` |
| source replay maximum error | `2.384e-7` |
| exact decoder maximum error | `2.965e-8` |
| coefficient/diagnostic reload | `5/5` |
| target-shuffled passes | `0/15` arm populations |
| aggregate cell wall time | `23.91 s` |
| maximum CUDA allocation | `.27494 GB` |

Before primary execution, nine CPU contract tests passed. A separate reduced
seed-7 CUDA lifecycle passed source identity, replay, decoder, artifact, and
zero-training gates and remains under `/tmp`; none of its metrics enters this
report.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mpl-c3-readout pixi run python -m \
  experiments.structure_net.tinyllm_c3_temporal_continuation_readout \
  --execute-primary --device cuda:0 \
  --output \
  data/experiments/tinyllm_c3_temporal_continuation_readout/20260811_d6_preregistered
```

Exact resume reuses only cells whose scientific fingerprint and diagnostic
hash match.

| Artifact | SHA-256 |
| --- | --- |
| campaign | `0da3dcad91a7b9eac34a24a23f687b598b5315349e0b0302a5cc6a0bebab2dea` |
| five-result manifest | `3a61695e502813d0b134695edafc4139bad98e86474b8e03b402ce73d932660d` |
| five-diagnostic manifest | `71718d4730f1a41e1302acb01e194430a804520091a4547b0741facd2883d225` |
| implementation | `3ccb922b8e0fb5119cc8c327024f6b9e1e957bb34b959dcb9e523518ac41746e` |
| preregistration | `f4833ffcc7455fd7b72c23d49da1db92920cad5e3030928cccef6af53dbc20ae` |
| source C3 campaign | `e46710de358a3c9c5d30ddd4a19e36182b94a50cabc699014f79d58d3d3de7cc` |

The campaign's combined implementation digest is
`11db63d258e663eed8654e144cbbb3cd5c70284488962e7dab6a458107e46110`;
the scientific campaign fingerprint is
`813a234e5103c806a1d5c788ef9b44a4c84677c91ed761e33e3f8f625b68b210`.
