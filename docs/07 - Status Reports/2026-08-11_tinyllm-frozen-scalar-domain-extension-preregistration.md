# TinyLLM frozen scalar-domain extension preregistration

**Status:** FROZEN BEFORE NEW WIDER-DOMAIN OUTPUTS  
**Date:** 2026-08-11  
**Hypothesis ID:** `tinyllm-frozen-scalar-domain-extension-v1`  
**Schema:** `nal.tinyllm-frozen-scalar-domain-extension.v1`  
**Evidence role:** `registered_outcome_informed_frozen_scalar_domain_diagnostic`

## Evidence boundary

This is a deliberately outcome-informed localization study, not fresh
confirmation. The completed scalar-interface campaign showed that d10 learned
seed 29 omits answer bins `0`, `1`, `6`, and `15` on the admissible scalar
interval `[-1, 1]`. Its minimum-soft-target-cross-entropy scalar oracle misses
the composition task floor, although target-bin reachability remains above the
old exact-accuracy floor in both shifts.

The source evidence is frozen as follows:

| Artifact | Frozen identity |
| --- | --- |
| scalar-interface campaign | `f71b85bc51f9694346fa23482bc63e1631e0918d55f5cae4f3a0d5ce09a47ba2` |
| scalar-interface result manifest | `201f6bb780e5a92938df1596e61f9f75164bb0e3bfcd0554f6ad03ba74f177b4` |
| d10 learned seed-29 result | `16deb857f4449c66a22eac347c2c7e2b2cf23fbd5a8a1ad0daaf39660bf910f0` |
| d10 learned seed-29 diagnostics | `3bfbf496e1cb85d79548849b533b4139c0737e7079a53e4e5bb2f974de7f9476` |
| architecture source campaign | `656d9814a032d1899810e81d398adf935cea3e1116712460e2062da188a0c9e2` |
| model checkpoint | `94311bfeede8b1f87aad0bb417bf0d665fece541219646645e8be482489e01df` |
| front-end checkpoint | `0877630a8815356fd40c1eff96656b55869e9c36f95307eee2825e1dd13520bb` |
| DVC source root | `70142aadaf9ec5f5f45236f4f85e3472.dir` |
| lakeFS source commit | `53462c21f889423c467bad0a003555496ec987b9ef9f42bab3839a10bef7b5c3` |

Only behavior at `|s| > 1` is new scientific evidence. Recomputed behavior on
`[-1, 1]` is a validity replay.

## Question

Are the four missing answer bins absent because the learned front end bounds
its scalar output with `tanh`, or because the frozen scalar-to-posterior curve
never gives those answer rows a winning region even outside the encoder image?

No parameter will be trained, fit, or changed. The model, front end, scalar
embedding, transformer continuation, and answer rows remain frozen.

## Registered intervention

Load only the d10 learned-equivariant seed-29 checkpoint. Bypass the front-end
scalar output and inject a declared scalar grid through its existing scalar
embedding. Use one nested grid with spacing exactly `1/2048` over `[-8, 8]`,
for 32,769 fixed candidates. Report cumulative restrictions at radii
`1`, `2`, `4`, and `8`.

For source-compatible replay, add the stored natural and exact-cosine scalar
values to the candidate set. These values lie inside `[-1, 1]`; they cannot
create a wider-domain discovery. Evaluate composition and extrapolation
separately with their frozen 1,024-example cohorts and the one observed
`(BOS=1, query=2)` context.

For every radius and shift, record:

- answer bins attained as posterior argmax;
- first positive and negative scalar at which each bin wins;
- exact-bin reachability across examples;
- the posterior and accuracy selected by minimum full soft-target cross
  entropy;
- task-floor pass/fail;
- boundary winner bins and maximum answer probability.

## Primary endpoint and locked classification

The primary endpoint is whether each source-missing bin in `{0, 1, 6, 15}`
has a resolved winning scalar with `1 < |s| <= 8`.

| Outcome | Locked classification | Meaning |
| --- | --- | --- |
| all four bins appear outside `[-1,1]` | `bounded_encoder_range_hole` | the encoder's bounded scalar image excludes answer regions that exist in the frozen continuation |
| one to three bins appear | `mixed_range_and_answer_curve_hole` | saturation explains only part of the omission |
| none appears | `continuation_answer_curve_hole_persists_to_radius_8` | the omission persists in the frozen continuation over the declared resolved domain |

The minimum-cross-entropy oracle is secondary because the previous result
showed that posterior-shape selection and argmax reachability answer different
questions. If all bins appear but that oracle still misses a source floor, the
range diagnosis stands while posterior calibration remains separately failed.

This finite grid cannot prove mathematical absence outside `[-8,8]` or between
resolved points. Any negative conclusion is explicitly limited to the
registered radius and resolution.

## Validity gates

The cell is valid only if all of the following pass:

1. source campaign, result, diagnostic, checkpoint, and front-end hashes match;
2. a fresh replay at every stored source scalar reproduces the stored
   posterior curves within maximum absolute error `2e-6`;
3. the fixed `[-1,1]` grid reproduces the source reachable-bin set
   `{2,3,4,5,7,8,9,10,11,12,13,14}` in both shifts;
4. composition and extrapolation dataset hashes match their source records;
5. model and complete-system state digests are unchanged before and after;
6. all arrays and JSON metrics are finite;
7. CUDA execution is recorded and peak allocation is reported;
8. exact resume leaves completed output bytes unchanged.

Failure of a validity gate yields `invalid` and no scientific classification.

## Lifecycle and artifact contract

The runner must first pass focused unit tests and a reduced-sample lifecycle in
a separate root. The lifecycle result is systems evidence only. The primary
artifact root is:

```text
data/experiments/tinyllm_frozen_scalar_domain_extension/
  20260811_d10_learned_seed29_registered/
```

It must contain an atomic `result.json`, compressed diagnostic arrays, runner
and preregistration hashes, a scientific fingerprint, source identities, state
records, and the locked classification. After interpretation, write the dated
analysis report, store and read back a conservative meta-hypothesis record,
refresh DVC, push the configured remote, commit lakeFS, and verify immutable
root/result/meta checksums.

## Stop rule

Stop after this cell. Do not widen the interval again, change grid resolution,
fit a coordinate map, or retrain the encoder or TinyLLM in response to the
outcome. A further domain extension would be a separately registered study.
