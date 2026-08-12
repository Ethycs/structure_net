# TinyLLM observed-twirl noise-law falsifier

**Status:** INVALID PREREGISTERED PRIMARY — ISOTROPIC POSITIVE CONTROL FAILED  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`; frozen-system,
no-fit sensor intervention  
**Hypothesis:** `tinyllm-noise-law-observed-twirl-v1`  
**Schema:** `nal.tinyllm-noise-law-observed-twirl.v1`  
**Preregistration:** [observed-twirl noise-law preregistration](../07%20-%20Status%20Reports/2026-08-10_tinyllm-noise-law-observed-twirl-preregistration.md)

## Verdict

The registered campaign does **not** establish whether observed quotient
closure is robust to reflection-asymmetric noise. Its isotropic positive
control failed before the asymmetric comparison could be interpreted.

At the fixed additional planar dose `sigma=0.05`, natural exact-bin utility in
the analytic arm passes `0/5` seeds under isotropic and lab-anisotropic noise
and only `1/5` under lab-biased noise. The preregistration required at least
`4/5` for every analytic law and explicitly classified failure of isotropic or
analytic positive controls as invalid. The locked classification is therefore:

```text
invalid_isotropic_positive_control
```

This is not an implementation failure. Source replay, cut replay, generator
contracts, finite-number checks, state identity, and target-changing controls
all pass. The added dose itself perturbs the natural task more than the strict
five-point accuracy ceiling, so the experiment cannot isolate distributional
symmetry from ordinary measurement sensitivity.

A narrower preregistered secondary is nevertheless exact and repeatable:
conditional on each noisy identity, the correct observed action and Reynolds
twirl remain task-sufficient in **5/5 seeds for every arm, law, cut, and
shift**, while every orthogonal target-changing control fails. This does not
rescue the primary gate.

## Primary gate

A seed had to pass natural noisy utility, correct-action sufficiency, and
correct-twirl sufficiency jointly on composition and extrapolation at both
`pre_block` and `full`. An arm/law required four of five seeds. The primary
required all six arm/law cells and control specificity.

| Frozen arm | Noise law | Natural utility | Joint primary cell | Correct action, both cuts | Correct twirl, both cuts | Orthogonal control |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| analytic | isotropic | **0/5** | **0/5** | 5/5 | 5/5 | 0/5 |
| analytic | lab-anisotropic | **0/5** | **0/5** | 5/5 | 5/5 | 0/5 |
| analytic | lab-biased | **1/5** | **1/5** | 5/5 | 5/5 | 0/5 |
| learned equivariant | isotropic | 4/5 | 4/5 | 5/5 | 5/5 | 0/5 |
| learned equivariant | lab-anisotropic | 3/5 | 3/5 | 5/5 | 5/5 | 0/5 |
| learned equivariant | lab-biased | **0/5** | **0/5** | 5/5 | 5/5 | 0/5 |

The joint and natural counts coincide because correct action and twirl pass in
every cell. Natural measurement robustness, not group intervention closure,
is the limiting gate at this dose.

## Natural utility at the registered dose

The table gives five-checkpoint medians. Accuracy loss is relative to the
same frozen system on the identical clean cohort. The locked ceiling is five
percentage points.

| Arm | Law | Composition clean | Composition noisy | Composition loss | Extrapolation clean | Extrapolation noisy | Extrapolation loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| analytic | isotropic | `0.7461` | `0.6602` | **8.98 pp** | `0.6152` | `0.5654` | **6.25 pp** |
| analytic | anisotropic | `0.7461` | `0.6504` | **10.25 pp** | `0.6152` | `0.5410` | **7.71 pp** |
| analytic | biased | `0.7461` | `0.6094` | **14.26 pp** | `0.6152` | `0.5596` | **6.45 pp** |
| learned | isotropic | `0.7314` | `0.6719` | `4.20 pp` | `0.5254` | `0.4961` | `2.15 pp` |
| learned | anisotropic | `0.7314` | `0.6758` | `4.79 pp` | `0.5254` | `0.4961` | `2.93 pp` |
| learned | biased | `0.7314` | `0.6514` | **8.01 pp** | `0.5254` | `0.4619` | `3.52 pp` |

The analytic isotropic composition loss ranges from `8.20` to `10.84` points;
its extrapolation loss ranges from `4.39` to `9.57` points. This is not a
single unlucky seed. The learned encoder is more robust to isotropic error,
passing four seeds, but the strong experiment required the analytic positive
control as well.

The likely architectural reason is declared rather than retrofitted as a new
endpoint: the analytic canonicalizer estimates its coordinate from the final
planar sample, whereas the learned equivariant encoder combines the full time
history. The current campaign did not preregister a denoising-capacity claim.

## Noise-law validity

All three laws share the same Gaussian draws and expected planar squared norm
`2 sigma^2 = 0.005`. Realized RMS differs from the expected `0.070711` by at
most `0.89%`.

| Regime | Isotropic covariance defect | Anisotropic covariance defect | Biased mean defect |
| --- | ---: | ---: | ---: |
| composition | `<1.8e-16` median | `0.1842` median | `0.1634` median |
| extrapolation | `<1.9e-16` median | `0.6615` median | `0.6144` median |

The registered floors were `0.10` for anisotropic covariance and `0.05` for
biased mean. Thus the laws are energy matched, the isotropic law is
reflection-compatible, and both asymmetric laws genuinely break the declared
population symmetry.

## Narrow structural secondary

Although the natural utility gate fails, the action mechanism itself is
stable conditional on the noisy observation:

- all `30/30` arm-by-law-by-seed cells pass correct action at both cuts and
  both shifts;
- all `30/30` pass correct twirl at both cuts and both shifts;
- all `30/30` matched orthogonal controls fail;
- maximum correct-action accuracy loss is `0.00879`;
- maximum correct-twirl accuracy loss is `0.00684`;
- maximum action posterior JS is `9.79e-4`;
- maximum twirl posterior JS is `2.45e-4`; and
- analytic feature action error is at most `2.39e-7`.

This supports an algebraic distinction:

```text
measurement robustness of the natural task
    !=
functional equivariance of the observed action and twirl.
```

The correct intervention remains essentially task-neutral even off the
natural support of an asymmetric law. But because the noisy identity itself
has already crossed the registered utility boundary, this cannot certify a
usable quotient under those laws.

## Integrity

- requested/completed/failed/excluded/retried cells: `10/10/0/0/0`;
- reused primary cells: `0`;
- trained or fitted parameters: `0`;
- source clean posterior replay error: exactly `0.0`;
- continuation replay error at both cuts: exactly `0.0`;
- analytic feature-invariance gate: `5/5` systems across all laws and shifts;
- every model and full-system state hash remains unchanged;
- every numeric record is finite;
- target-changing control passes: `0/30`;
- exact-resume primary tree:
  `1b010e4ac36669ca1e93d4a496840793ed5e3ccaa57938b88202af7f01b733ba`;
- exact-resume shakedown tree:
  `df2e79f0d15982d8178fc771125cd3d36715b369dd091c7c4cd7b9724a043374`.

The systems-only shakedown is not scientific evidence. It validated one
analytic checkpoint, 64 examples per shift, all three laws, artifact writing,
source/cut replay, state identity, and byte-identical resume.

## Program decision

Preserve the invalid primary. Do not describe the asymmetric laws as either
confirmed robust or causally harmful at `sigma=0.05`; the isotropic positive
control prevents that comparison.

Do not retrain TinyLLM or either front end. The shortest corrective diagnostic
is a new, explicitly outcome-informed **nested dose-localization** study using
scaled copies of the already frozen error arrays. It should first identify a
common nonzero scale at which isotropic natural utility passes `>=4/5` in both
arms. Only within that utility-valid window may anisotropic and biased laws be
compared. The original `sigma=0.05` result must remain visible and must not be
rewritten as a tuned primary.

If no common nonzero utility-valid window exists, stop: the current strict
five-point task endpoint is too sensitive for a shape-only noise-law test. If
a window exists and asymmetric laws fail there, noise-law symmetry is
causally material. If all laws pass there, close this additive-noise branch at
the certified dose and retain the exact canonicalizer/twirl as the structural
mechanism.

## Artifacts and reproduction

| Item | SHA-256 / value |
| --- | --- |
| campaign | `868ad0ffee546f157e701790c34a83f20bfb3116e78b2f8c5bc34dd7bfe660d7` |
| error arrays file | `d3771eac8e29f7940df7feaedebe74a5a78fb273cda2e70928c9be9e37ff3ba6` |
| error-array content | `93df61bc76ed073ea241c9450e7ec3523e7a98b5ac06e58d7e920a5df07d70aa` |
| result manifest | `7246968593214d5a91b9283e856472cf351b2e921d6712402f9fc128bb457d4d` |
| composite implementation | `d4a7e172b0cb9ed5da9a4508c812211882075fcb75db540a17ac6912a8330d6a` |
| runner | `7bed49c064e8a2148268d2a4ab3a42ec70847a15d83c7297cff3d9dccc7970d2` |
| preregistration | `8ff50bc47cb7c6223dbf234044a3d18fefd91b15c9d67578d674bb33029be26b` |
| predecessor campaign | `79c3e27374d8b6f4611552595de5852ace940204bda825e64cf80eff6ab2050d` |
| shakedown campaign | `95d86c02763f8c565b5f581ee1488554f81a01352c39ba399c89d9cffb0bee35` |
| meta-hypothesis record | `6a550fe17448db501a0cb0a871f610363be0e6e9fa978de8430bd49a8c6837c0` |
| DVC data root | `19f1fbbe86b6b9235eb211a88bb32aa2.dir` |
| lakeFS commit | `f3c895cdf8d5f25e8ae6a87b3f694d0bbacb24cdd14d4736d0c7dfa41399c130` |
| device | NVIDIA GeForce RTX 2060 SUPER (`cuda:1`) |
| peak CUDA allocation | `293,665,792` bytes |
| campaign analysis time | `201.81` seconds |

- primary:
  `data/experiments/tinyllm_noise_law_observed_twirl/20260810_d10_preregistered/`
- systems-only shakedown:
  `data/experiments/tinyllm_noise_law_observed_twirl/20260810_shakedown_analytic_cuda/`

```bash
MPLCONFIGDIR=/tmp/matplotlib-noise-law-primary \
pixi run python -m \
  experiments.structure_net.tinyllm_noise_law_observed_twirl \
  --device cuda:1 \
  --output \
  data/experiments/tinyllm_noise_law_observed_twirl/20260810_d10_preregistered
```

## Boundaries

This result covers one added error scale, three additive lab-frame planar
laws, two reused synthetic shifts, one calibrated `C2` task, and ten retained
d8/N3 systems. It does not test temporal correlation, calibration-packet
error, requantization, learned denoising, other doses, other groups,
architecture populations, natural language, or real sensors. The transformed
asymmetric observations can be off the natural support of their law; the
narrow action result is frozen functional robustness, not proof of a
distributional `C2` symmetry.
