# TinyLLM C3 temporal sensor mechanism decomposition

**Status:** REGISTERED POST-OUTCOME ARTIFACT-ONLY RESULT CONFIRMED

**Date:** 2026-08-11

**Hypothesis:** `tinyllm-c3-temporal-sensor-mechanism-v1`

**Classification:** `affine_identity_character_carries_learned_solution`

**Preregistration:** [C3 temporal sensor mechanism decomposition](../07%20-%20Status%20Reports/2026-08-11_tinyllm-c3-temporal-sensor-mechanism-preregistration.md)

**Primary artifact:** `data/experiments/tinyllm_c3_temporal_sensor_mechanism/20260811_preregistered/campaign_results.json`

## Verdict

The successful 184-parameter exact-`C3` sensors solved the fixed temporal task
through their affine identity response. Replacing each learned scalar response
with one source-fitted complex affine function preserves the complete carrier
and task gate in all five true seeds. Keeping only the complementary nonlinear
response preserves it in no seed. The same affine patch passes no matched
target-shuffled seed.

```text
true affine_only:                 5/5
true nonlinear_residual_only:     0/5
target-shuffled affine_only:       0/5
full checkpoint replay:          10/10

registered requirement: >=4/5, <=1/5, <=1/5, and 10/10
```

The learned parameters need not equal the closed-form five-nonzero-parameter
GELU witness. The causal result is functional: after fitting only the frozen
sensor response on its reconstructed training observations, the identity
character alone carries the learned solution. The nonlinear shared-response
harmonics are unnecessary for the registered task under both shifts.

No optimization was performed, no parameter changed, and no TinyLLM model was
instantiated.

## Causal construction

For each reloaded sensor, the effective complex scalar response was

```text
g(x) = sum_k f_k(x) m_k.
```

A target-free float64 least-squares fit on the sealed source training
observations produced

```text
g_aff(x) = alpha x + beta,
g_res(x) = g(x) - g_aff(x).
```

The three frozen response patches were then passed through the original
nontrivial `C3` character projection, normalization, cubing, analytic temporal
operator, and fixed sixteen-bin decoder. No target, phase, analytic carrier,
held-out observation, or shift-specific statistic entered the response fit.

The constant `beta` cancels under the nontrivial character. Consequently, the
affine patch is the same identity-character mechanism exhibited by the earlier
five-parameter analytic witness, up to the learned complex slope and the
resulting global carrier orientation.

## Primary per-seed result

| Seed | Affine comp acc | Affine comp corr | Affine extrap acc | Affine extrap corr | Affine extrap CE | Residual extrap acc | Residual extrap corr |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | `.9248` | `.99952` | `.9229` | `.99946` | `1.27760` | `.0957` | `.16521` |
| 17 | `.9502` | `.99959` | `.9590` | `.99955` | `1.27528` | `.1494` | `.48444` |
| 29 | `.9482` | `.99957` | `.9580` | `.99953` | `1.27574` | `.0605` | `-.27290` |
| 41 | `.9512` | `.99959` | `.9590` | `.99955` | `1.27528` | `.0400` | `-.41570` |
| 53 | `.9492` | `.99957` | `.9609` | `.99954` | `1.27566` | `.0771` | `.24187` |

Every affine-only true cell covers all sixteen output bins and passes the
registered carrier, correlation, accuracy, and cross-entropy gates on both
shifts. Every nonlinear-only true seed fails the simultaneous two-shift gate.

## Population measurements

Means over five seeds:

| Source arm | Response patch | Shift | Accuracy | Correlation | Cross-entropy | Carrier dot | Carrier RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| true | full replay | composition | `.94336` | `.999559` | `1.27248` | `.999998` | `.001413` |
| true | full replay | extrapolation | `.95039` | `.999513` | `1.27614` | `.999998` | `.001415` |
| true | affine only | composition | `.94473` | `.999568` | `1.27231` | `1.000000` | `<3e-7` |
| true | affine only | extrapolation | `.95195` | `.999526` | `1.27591` | `1.000000` | `<3e-7` |
| true | nonlinear residual only | composition | `.09531` | `.12301` | `14.1948` | `.43444` | `.75014` |
| true | nonlinear residual only | extrapolation | `.08457` | `.04059` | `14.8875` | `.43524` | `.74965` |
| target shuffled | affine only | composition | `.04922` | `.09597` | `15.3798` | `1.000000` | `<1e-6` |
| target shuffled | affine only | extrapolation | `.04395` | `.09071` | `15.5259` | `1.000000` | `<1e-6` |

The full learned response in the true population is already nearly affine:
its mean source-fit complex `R^2` is `.999503`, with individual values from
`.998359` to `.999895`. The shuffled population has a lower and heterogeneous
mean `R^2` of `.831907`.

## What the control establishes

Any nonzero complex affine response generates the same geometric carrier up to
a global `O(2)` gauge. Accordingly, even the shuffled affine patches have
carrier dot approximately `1.0` after a diagnostic gauge is fitted. They still
fail the frozen task because their unfitted carrier orientation is not aligned
with the temporal operator and physical decoder.

This is the useful specificity result. True-task training does not merely
create an affine-looking response; it selects the affine slope phase needed by
the fixed downstream computation. The target-shuffled checkpoints do not.

The result therefore supports:

```text
task loss
  -> task-aligned complex affine identity response
  -> exact C3 character projection and cubing
  -> support-stable temporal carrier and fixed task.
```

It does not support a claim that the nonlinear hidden response is a necessary
part of the solution.

## Numerical contracts

| Contract | Result | Registered limit |
| --- | ---: | ---: |
| source campaign and twenty-checkpoint provenance | pass | exact |
| full direct posterior replay | `0.0` maximum error | `<=2e-6` |
| sealed task-metric replay | `0.0` maximum error | `<=2e-6` |
| pre-normalization `z_full = z_aff + z_res` | `1.973e-7` maximum error | `<=1e-6` |
| state identity | `10/10` | exact |
| optimizer steps / parameters changed / TinyLLM instances | `0 / 0 / 0` | `0 / 0 / 0` |

The original GPU evaluation order and the algebraically commuted complex
response differ by at most `1.076e-5` after normalization in the weakest
shuffled cells. The authoritative full replay therefore uses the exact frozen
encoder evaluation order, while the registered affine/residual identity is
verified before normalization.

Finite-precision deck-action errors are reported but were prospectively
excluded from mechanistic specificity because invariance is imposed by the
character construction. The largest true affine-only action error is
`2.348e-6`, slightly above the inherited descriptive `2e-6` reference. The
largest nonlinear-only error is `.002248`; its character coefficient is often
near the existing `1e-6` normalization clamp, so tiny equivariant roundoff is
amplified. Neither numerical action diagnostic is used to rescue a task gate.

## What this settles

### Supported

- The successful sensors use the affine identity character as a causally
  sufficient solution in `5/5` seeds and under both registered shifts.
- Nonlinear shared-response harmonics are unnecessary under the registered
  intervention: their isolated complement passes `0/5` seeds.
- Task training selects the downstream-compatible carrier orientation;
  target-shuffled affine controls pass `0/5` complete task gates.
- The 184-parameter family is functionally overcomplete for this noiseless
  task. The earlier five-nonzero-parameter witness describes its retained
  mechanism.

### Not supported

- The learned tensor parameters are not proven equal or close to the analytic
  witness parameters; only functional causal sufficiency is established.
- The result does not show that nonlinear sensor capacity is useless with
  noisy, missing, nonlinear, or approximate observations.
- TinyLLM utility, transformer quotient learning, language behavior, and
  generalization to a new group or temporal law remain untested here.
- Because the mechanism question was registered after the sensor-only success,
  this is a registered post-outcome decomposition, not an independent
  prospective replication of the original training result.

## Program decision

Close the noiseless sensor-capacity branch. Do not enlarge the sensor, scan its
harmonics, or retrain it with another optimizer. For the current generator, the
analytic carrier remains the engineering baseline and the five-parameter
witness is the correct learned-mechanism compression target.

Further learning is justified only after changing the estimation problem so
the affine analytic rule is insufficient—for example, declared observation
noise, missing calibration, approximate group actions, or a second identifiable
group. A TinyLLM reintegration should be separately preregistered and use a
typed continuation that preserves carrier orientation and metric meaning; the
present result supplies its positive control but does not establish its value.

## Reproduction and provenance

```bash
MPLCONFIGDIR=/tmp/mpl-c3-sensor-mechanism \
pixi run python -m \
  experiments.structure_net.tinyllm_c3_temporal_sensor_mechanism \
  --device cuda:0
```

| Artifact | SHA-256 |
| --- | --- |
| primary result | `c3dbfecd7a6381c2129e4d99f135557f003ad8a225fa2b4d3f4fa0cb429f669b` |
| runner | `8ca2143d8f7262b192c23eeb83aa56a6cc7a55d9cf6670c32a68c765122c14d3` |
| preregistration | `b32dbd5fce221b7eec50a7845f41d2dd036275397d69837e293db25bfecb72d5` |
| source sensor-only campaign | `4d023d9f9d77f64b1fe75970acd63461cfd5a64aa1de60c7954a2c671c427012` |
| source checkpoint manifest | `e83832872f29d072d710859e022f4e17d1b6da6a9e16b63049f41d4ea2eb01a0` |

The focused function-class, sensor-only, and mechanism suites pass `17/17`
tests against the authoritative artifact.
