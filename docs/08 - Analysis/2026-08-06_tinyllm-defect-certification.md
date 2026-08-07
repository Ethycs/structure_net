# TinyLLM d6 step-15 defect certification attempt

**Status:** FORMAL CERTIFICATE NOT OBTAINED — REGULAR POSITIVE NUMERICAL ROOT RETAINED  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`  
**Hypothesis:** `tinyllm-d6-step15-defect-certificate-v1`  
**Preregistration:** `../07 - Status Reports/2026-08-06_tinyllm-defect-certification-preregistration.md`

## Verdict

The d6 step-14→15 cylinder contains a stable floating-point root at phase
`4.3250336122` and path fraction `0.2038221007`. Its residual is `7.45e-9` and
the Jacobian determinant in the cylinder's `(path, phase)` orientation is
`+0.13654734`, consistent with the previously measured local charge `+1`.

This is not a formal actual-network certificate. The implementation does not
provide outward-rounded transformer evaluation, a Krawczyk uniqueness enclosure,
proof that all other boxes are root-free, or a rigorous network-to-surrogate
remainder bound. The preregistered formal hypothesis is therefore not confirmed.

A subsequent directed-rounding calculation does certify a unique positive-index
root of the **stored degree-8 floating Chebyshev polynomial**. It does not certify
transfer to the transformer, because the sampled network-to-surrogate error is
not a rigorous remainder enclosure. The actual-network verdict is unchanged.

## Campaign integrity and endpoints

The experiment deterministically replayed the retained d6 seed-7 trajectory and
verified that step 15 is a degree-changing interval in the source result. It then
audited 128 floating-point samples per expanded rectangle edge, fitted a degree-8
two-dimensional Chebyshev surrogate, tested it on a `25×25` grid, and measured
endpoint winding on 4,096 phase points.

| Measurement | Value |
| --- | ---: |
| root phase | 4.3250336122 |
| root path fraction | 0.2038221007 |
| root residual norm | 7.45e-9 |
| determinant in `(phase,path)` coordinates | -0.13654734 |
| determinant in charged-cell `(path,phase)` orientation | +0.13654734 |
| minimum sampled boundary magnitude | 0.00242179 |
| endpoint degree change | +1 |
| sampled surrogate max / RMS error | 4.34e-5 / 1.54e-5 |

The two determinant signs differ only because swapping Jacobian columns reverses
orientation. The positive index statement uses the same `(path, phase)` convention
as the original defect-charge implementation.

## Preregistered gates

| Gate | Result |
| --- | --- |
| endpoint winding change `+1` and resolved | pass |
| sampled boundary avoids zero | descriptive pass |
| outward-rounded actual-network boundary excludes zero | fail / unavailable |
| actual-network Krawczyk box proves a unique root | fail / unavailable |
| every other box is interval root-free | fail / unavailable |
| actual-network Jacobian determinant interval is positive | fail / unavailable |
| actual-network surrogate remainder rigorously enclosed | fail / unavailable |
| formal certificate | **not obtained** |

## Stored-surrogate interval certificate

The follow-up treats each stored binary64 Chebyshev coefficient as an exact real
constant and applies `nextafter`-directed outward rounding after every arithmetic
primitive. All 4,096 boundary segments exclude zero. Adaptive subdivision leaves
two adjacent unresolved boxes whose hull is mapped strictly into itself by a
Krawczyk operator, and the oriented determinant interval is strictly positive.

| Surrogate certificate measurement | Value |
| --- | --- |
| boundary boxes exclude zero | yes |
| excluded / unresolved leaves | 298 / 2 |
| physical phase hull | [4.325030883789062, 4.325033142089843] |
| physical path-fraction hull | [0.203763525390625, 0.2037748046875] |
| Krawczyk image strictly inside | yes |
| oriented determinant interval | [2.870054376030508e-5, 2.8709286473770056e-5] |
| stored-polynomial unique positive root | **certified** |
| actual-network transfer | **open** |

This is a formal statement about the serialized polynomial only. Its certified
root differs slightly from the direct-network numerical root because the fitted
surrogate has nonzero sampled approximation error.

## Interpretation and boundaries

The result upgrades the earlier charged grid cell to a precise, regular numerical
root and fixes its orientation convention. It does not upgrade sampled numerical
evidence into a theorem. A real certificate requires interval-aware evaluation of
the actual network or a surrogate whose value and derivative remainders are
rigorously bounded over the entire rectangle.

## Artifacts and reproduction

| Artifact | Path |
| --- | --- |
| result | `data/experiments/tinyllm_defect_certification/20260806_d6_step15/results.json` |
| arrays and surrogate | `data/experiments/tinyllm_defect_certification/20260806_d6_step15/certificate_arrays.npz` |
| source replay | `data/experiments/tinyllm_degree_defect_cobordism/20260805_d6_d8_seed7/results.json` |
| result SHA-256 | `bcf9b595958eabf0e3fa38d299972a0ddeb7d88839be8884d6596a1cc36766a8` |
| surrogate interval result | `data/experiments/tinyllm_defect_certification/20260806_d6_step15_surrogate_interval/results.json` |
| surrogate interval result SHA-256 | `ddf2b97527368a5b657d6ef852794eaf8117a0aa21f39348a9b5c01c15693df2` |

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
pixi run python -m experiments.structure_net.tinyllm_defect_certification \
  --device cuda:0 --boundary-samples 128 --endpoint-phase-points 4096 \
  --surrogate-degree 8 --surrogate-audit-points 25 \
  --output data/experiments/tinyllm_defect_certification/20260806_d6_step15

pixi run python -m experiments.structure_net.tinyllm_defect_surrogate_interval_certificate \
  --source data/experiments/tinyllm_defect_certification/20260806_d6_step15/results.json \
  --output data/experiments/tinyllm_defect_certification/20260806_d6_step15_surrogate_interval/results.json
```
