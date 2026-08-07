# TinyLLM calibration-degradation breakpoints

**Status:** SUPPORTED — BOUNDED STABLE BREAKPOINTS IN BOTH STRUCTURED ARMS  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, sequential frozen-checkpoint robustness diagnostic  
**Hypothesis:** `tinyllm-calibration-noise-breakpoint-v1`  
**Schema:** `nal.tinyllm-calibration-degradation.v1`  
**Preregistration:** [calibration-degradation breakpoint preregistration](../07%20-%20Status%20Reports/2026-08-07_tinyllm-calibration-degradation-preregistration.md)

## Verdict

Both previously successful gauge-repaired TinyLLM mechanisms have finite,
ordered calibration-noise breakpoints on the registered grid. The analytic
canonicalizer passes the complete quotient gate through `q=1` and first fails
at `q=2`, giving breakpoint interval `[1, 2]`. The learned calibrated-
equivariant encoder passes through `q=0.5` and first fails at `q=1`, giving
interval `[0.5, 1]`.

Both curves satisfy every preregistered condition:

- exact calibration passes `5/5` seeds;
- at least one nonzero level passes;
- the maximum level fails `0/5`;
- neither curve re-enters after failure; and
- shuffled calibration fails the joint gate in `5/5` seeds for both arms.

The hypothesis is therefore supported under its operational definition. The
later analytic breakpoint is descriptive: the protocol explicitly made no
superiority claim between the two model families.

This is deliberately a **representation-only** robustness result. The later
[complete-system degradation test](2026-08-07_tinyllm-calibration-degradation-causal.md)
and
[orientation-only titration](2026-08-07_tinyllm-calibration-orientation-noise.md)
also require frozen exact-bin task utility. Those stricter gates fail much
earlier even while this cosine/branch endpoint continues to pass. The results
are complementary: the internal quotient is more tolerant than its fixed
chart-to-bin readout.

## Campaign integrity

| Item | Value |
| --- | --- |
| source checkpoints | ten frozen d8 systems: two arms x five seeds |
| requested / completed / failed / excluded | `10 / 10 / 0 / 0` |
| reused on first launch / retries | `0 / 0` |
| trained TinyLLMs / fitted front ends | `0 / 0` |
| fitted held-out measurement probes | `170` |
| implementation SHA-256 | `175160f9ca3c51bffb713b8fc9251c3eb888d0f9121424772811566aee77a329` |
| campaign SHA-256 | `87a556e61db4a584b9cd423af9cb9d663f3f0225757a31a69a7158788137ef86` |
| source campaign SHA-256 | `80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501` |
| device | NVIDIA GeForce RTX 2060 SUPER, `cuda:1` |
| peak CUDA allocation | `327,117,824` bytes |
| PyTorch / Python | `2.5.1+cu121 / 3.11.13` |
| analysis time | `513.29` seconds |

Every source result, model checkpoint, front-end checkpoint, model state, and
complete system state validated before inference. All ten cells pass all five
provenance, exact-calibration, source-metric replay, and numerical contracts.
The zero-noise replay error is exactly `0` for every checkpoint. Exact resume
left the primary campaign and per-cell result bytes unchanged.

The systems-only seed-7 artifact is retained separately at
`data/experiments/tinyllm_calibration_degradation/20260807_shakedown_seed7/`.
Its campaign SHA-256 is
`8098a9bf04d8152496e6acf35e9652d8d6376d7d15dfa79311b3928da53d2ad6`;
it is not pooled with the primary evidence.

## Primary breakpoint gates

The table reports the number of seeds passing all four primary cells jointly:
front-end and full depth under both composition and extrapolation.

| Arm | `q=0` | `0.125` | `0.25` | `0.5` | `1` | `2` | `4` | Breakpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| analytic calibrated | 5 | 5 | 5 | 5 | **5** | **0** | 0 | `[1, 2]` |
| learned calibrated-equivariant | 5 | 5 | 5 | **5** | **3** | 0 | 0 | `[0.5, 1]` |

The learned `q=1` mean does not override the seedwise gate. Seeds 7, 41, and
53 pass jointly. Seed 17 fails extrapolation at both cuts, with cosine
correlations `0.868` and `0.888`. Seed 29 fails front-end composition and both
extrapolation cuts, with correlations `0.889`, `0.865`, and `0.899`.

No level re-enters after failure. The shuffled calibration control has zero
jointly passing seeds in both arms.

## Extrapolation geometry and task behavior

The following values are five-seed means. `front/full corr` are cosine
correlations; `branch` is full-depth conditional branch balanced accuracy; and
`task` is exact-bin accuracy. The primary gate uses per-seed values, not these
means.

| Arm | Level | Front corr | Full corr | Branch | Task |
| --- | ---: | ---: | ---: | ---: | ---: |
| analytic | 0 | 0.964 | 0.992 | 0.496 | 0.616 |
| analytic | 0.5 | 0.953 | 0.979 | 0.502 | 0.468 |
| analytic | 1 | 0.919 | 0.942 | 0.509 | 0.334 |
| analytic | 2 | 0.806 | 0.824 | 0.496 | 0.203 |
| learned | 0 | 0.960 | 0.987 | 0.510 | 0.492 |
| learned | 0.5 | 0.944 | 0.969 | 0.502 | 0.396 |
| learned | 1 | 0.892 | 0.916 | 0.499 | 0.281 |
| learned | 2 | 0.740 | 0.758 | 0.509 | 0.181 |

Branch accuracy remains near `0.50` throughout. Failure is therefore not a
return of conditionally decodable deck branch under the tested nonlinear
probe. It is loss of the semantic cosine coordinate.

Transformer depth continues to improve correlation after the front end, even
near and beyond the breakpoint, but cannot reconstruct a calibration-damaged
coordinate that falls too far below the threshold. At learned `q=1`, for
example, full-depth extrapolation improves the mean from `0.892` to `0.916`,
yet only three seeds satisfy every registered cell.

Task accuracy degrades earlier and more continuously than the representation
gate. The analytic arm still passes the quotient endpoint `5/5` at `q=1`, but
its extrapolation exact-bin accuracy has fallen from `0.616` to `0.334`.
Consequently, the quotient geometry gate is not an engineering task-accuracy
guarantee; decoder calibration and confidence remain separate requirements.

## Physical scale of the breakpoint grid

The same locked Gaussian field is multiplied by `q`, so levels form a nested
curve. On the extrapolation split, the realized RMS calibration errors are:

| Level | Orientation, rad | Log amplitude | Log speed | Offset | Drift |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.155 | 0.233 | 0.226 | 0.102 | 0.166 |
| 2 | 0.310 | 0.466 | 0.453 | 0.203 | 0.332 |
| 4 | 0.619 | 0.933 | 0.906 | 0.407 | 0.665 |

Orientation remains unit-normalized to maximum error `5.96e-8`; amplitude
stays positive and speed sign is preserved. These are joint multi-field error
levels, so the result does not identify which calibration field sets either
breakpoint.

## Interpretation

The earlier calibrated-front-end experiment established constructibility with
an exact gauge reference. This result adds a narrower engineering statement:

```text
exact reference
  -> stable quotient in both structured mechanisms
moderate joint reference error
  -> quotient retained, task accuracy declines
larger joint reference error
  -> semantic coordinate crosses a finite robustness boundary
```

The analytic mechanism tolerates one additional registered interval. A likely
reason is structural: it applies the declared inverse acquisition transform
directly, whereas the learned encoder was optimized only on exact calibration
and must extrapolate through its learned scalar map under metadata error. This
is an interpretation, not a separately randomized family comparison.

The shortest justified follow-up, only if a real calibration budget is needed,
is a preregistered field-wise ablation around `q=0.5--2` to distinguish
orientation, scale/speed, offset, and drift sensitivity. It should reuse these
same checkpoints. It should not reopen residual-sidecar optimization, topology
scans, link cobordism, or TinyLLM retraining.

## Artifacts and reproduction

- aggregate:
  `data/experiments/tinyllm_calibration_degradation/20260807_d8_preregistered/campaign_results.json`
- per-checkpoint records:
  `data/experiments/tinyllm_calibration_degradation/20260807_d8_preregistered/runs/*/seed_*/result.json`
- runner:
  `experiments/structure_net/tinyllm_calibration_degradation.py`
- tests:
  `tests/structure_net/test_tinyllm_calibration_degradation.py`
- typed evidence:
  `data/meta_hypotheses/tinyllm-calibration-noise-breakpoint-v1.json`
- source campaign:
  `data/experiments/tinyllm_calibrated_frontend_causal/20260806_d8_preregistered/`

The typed evidence JSON has SHA-256
`99753671da3256791870bbb58fbc19cd0b0c3521e91cf14787bc8b66548a7632`.
Its hypothesis and all ten checkpoint records were read back from the
meta-hypothesis store.

The complete data tree is tracked and backed up as:

```text
DVC root: b4ab9fb2c3f8b806cc0b576e51c1df24.dir
Files: 2,130
Logical bytes: 39,892,567,237
lakeFS commit: 5de8c30b4537cb55774fca68bc109270324dbfe0726beb578900b24f834cdc35
```

The exact directory object is present at
`lakefs://artifacts/5de8c30b4537cb55774fca68bc109270324dbfe0726beb578900b24f834cdc35/structure-net/files/md5/b4/ab9fb2c3f8b806cc0b576e51c1df24.dir`.
After commit, lakeFS reported no uncommitted object diff, and DVC reported both
the local tree and configured `lakefs` remote in sync.

```bash
MPLCONFIGDIR=/tmp/matplotlib-calibration-degradation \
pixi run python -m experiments.structure_net.tinyllm_calibration_degradation \
  --device cuda:1 \
  --output \
  data/experiments/tinyllm_calibration_degradation/20260807_d8_preregistered
```

## Method boundaries

These ten checkpoints were selected because their exact-calibration quotient
gate had already passed. This is conditional robustness evidence, not an
independent replication or a population-prevalence estimate. Corruption occurs
only at inference after exact-calibration training. The Gaussian joint-error
curve is synthetic and does not represent a specific instrument. It does not
test bias, adversarial corruption, temporally varying reference quality,
training-time calibration noise, or a learned pilot estimator. Conditional
branch results are recoverability under the declared held-out nonlinear probe,
not certified absence of information.
