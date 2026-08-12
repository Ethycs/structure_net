# TinyLLM noisy shared-bias pilot acquisition

**Status:** VALID PREREGISTERED RESULT — FINITE NOISY PILOT REPAIR CONFIRMED  
**Date:** 2026-08-10  
**Hypothesis:** `tinyllm-noisy-bias-pilot-acquisition-v1`  
**Evidence role:** `preregistered_frozen_reused_draw_bias_pilot_titration`  
**Preregistration:** [noisy shared-bias pilot acquisition](../07%20-%20Status%20Reports/2026-08-10_tinyllm-noisy-bias-pilot-acquisition-preregistration.md)

## Verdict

Four noisy observations of a known zero-signal pilot are sufficient at the
registered two-arm population endpoint. Both frozen structured TinyLLM
populations pass composition and extrapolation simultaneously on all **16/16**
independent draws at `m=4`. The preregistered ceiling `m=256` also passes all
16 draws and every one of its 160 checkpoint-by-draw cells.

One observation is not reliable: only **12/16** complete draws pass. The
registered classification is therefore
`finite_noisy_pilot_repair_reliable`, with `m=4` the smallest reliable count.

No model, front end, task head, bias estimator, denoiser, observer, probe, or
noise process was trained or fitted. The experiment reused a sealed array
generated independently of the evaluation sensor noise.

## Primary population endpoint

A checkpoint passes a draw/count only when both composition and extrapolation
meet all three natural-utility limits. An arm passes with at least four of five
checkpoints; a complete draw requires both arms.

| Pilot count | Analytic arm draws | Learned arm draws | Complete draws |
| ---: | ---: | ---: | ---: |
| `1` | `13/16` | `14/16` | **`12/16`** |
| `4` | `16/16` | `16/16` | **`16/16`** |
| `16` | `16/16` | `16/16` | **`16/16`** |
| `64` | `16/16` | `16/16` | **`16/16`** |
| `256` | `16/16` | `16/16` | **`16/16`** |

The `16/16` result has a two-sided 95% Wilson interval of approximately
`[0.806, 1.000]`. “Reliable” means the preregistered sampled-population gate,
not a universal guarantee.

## Checkpoint sensitivity

Each entry is the number of the sixteen draws on which one frozen checkpoint
passes both shifts.

| Front end | Seed | `m=1` | `m=4` | `m=16` | `m=64` | `m=256` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| analytic | 7 | 15 | 16 | 16 | 16 | 16 |
| analytic | 17 | 12 | 16 | 16 | 16 | 16 |
| analytic | 29 | 15 | 16 | 16 | 16 | 16 |
| analytic | 41 | 15 | 16 | 16 | 16 | 16 |
| analytic | 53 | 9 | 12 | 16 | 16 | 16 |
| learned equivariant | 7 | 14 | 16 | 16 | 15 | 16 |
| learned equivariant | 17 | 11 | 15 | 16 | 16 | 16 |
| learned equivariant | 29 | 15 | 16 | 16 | 16 | 16 |
| learned equivariant | 41 | 14 | 16 | 16 | 16 | 16 |
| learned equivariant | 53 | 16 | 16 | 16 | 16 | 16 |

Thus `m=4` is a population count, not a per-checkpoint ceiling. Analytic seed
53 fails four `m=4` draws and learned seed 17 fails one, but no draw loses more
than one checkpoint in either arm. At `m=16`, every fixed checkpoint passes
every draw. The nonmonotone learned-seed-7 `m=64` miss is permitted by the
frozen nested-prefix design and does not affect the population endpoint.

## Acquisition precision

The pilot estimator is

\[
\widehat\mu_{d,m}=\mu+
\frac{0.03125}{\sqrt{2}}\frac1m\sum_{t=1}^{m}z_{d,t}.
\]

Its planar estimation error contracts with repeated acquisition:

| Count | Mean error norm | 95th percentile | Maximum |
| ---: | ---: | ---: | ---: |
| `1` | `0.02809` | `0.04627` | `0.04868` |
| `4` | `0.01733` | `0.03213` | `0.04083` |
| `16` | `0.00763` | `0.01314` | `0.01364` |
| `64` | `0.00379` | `0.00739` | `0.00997` |
| `256` | `0.00193` | `0.00306` | `0.00388` |

At `m=4`, the 95th-percentile exact-bin accuracy loss across checkpoint/draw
cells is `4.81` percentage points for analytic composition and `4.11` points
for learned composition. Individual worst cells can exceed the five-point
gate (`9.28` and `10.16` points), which is why the seedwise population rule
must not be misreported as system-level robustness. At `m=256`, the worst
accuracy loss is `3.03` points in the analytic arm and `2.05` points in the
learned arm across both shifts.

## Controls and integrity

- The sealed exact-pilot source passes `5/5` seeds in both arms.
- The sealed uncorrected biased source replays at `1/5` analytic and `3/5`
  learned seeds.
- Wrong-sign draw-0 `m=256` passes one analytic seed and zero learned seeds,
  within the registered ceiling.
- The inherited target-changing exact-pilot control remains pinned at `0/5`
  in both arms through the exact source-campaign validation.
- All ten result cells are finite, replay clean and source metrics within
  `2e-6`, and leave model/system state unchanged.
- The complete primary tree is byte-identical on aggregate resume; its
  manifest-of-files hash is
  `bb16ba336727a3c796b7ac825ca26b54fe311bbc997a1ad46fa347914fb457b9`.
- The final-code shakedown is explicitly serialized as underpowered and cannot
  claim a primary outcome.
- The eight directly impacted runner/meta suites pass `65/65`, and the full
  active repository suite passes `1330`, skips one, and fails zero.
- DVC is clean at root `73d421913a00dfcd3efe8c5d66b88824.dir`, and
  lakeFS commit
  `ab65ecfef3f08620775705b96eb09f26b2e13e40380db60433edca20352f2d40`
  has no uncommitted branch diff. The root, campaign, pilot-array, and
  meta-hypothesis objects exist at that immutable commit. Direct object
  readback reproduces their local SHA-256 values exactly.

## Interpretation

The exact shared-bias repair is not merely an oracle construction. Under the
declared independent Gaussian pilot law, a small finite observed calibration
sample estimates the signed displacement accurately enough for the two
five-checkpoint populations. The result closes the practical positive-control
chain:

```text
persistent positive sensor bias
    -> frozen task failure
observed zero-signal pilot
    -> signed bias estimate
four independent pilot measurements
    -> 16/16 population recovery draws
wrong-sign correction
    -> control failure
```

This strengthens the observation-interface account without restoring a
whole-residual invariance theory. The repair occurs before TinyLLM and leaves
the already validated calibrated quotient computation frozen.

## Scope and decision

The finding covers one persistent bias vector, one independent unbiased
homoscedastic Gaussian pilot law, sixteen draws, five retained checkpoints per
front-end arm, and the declared composition/extrapolation datasets. It does
not cover drift, correlated or heavy-tailed pilot noise, example-dependent
bias, a different bias direction or magnitude, natural-language tasks, or a
new architecture population.

Close this exact-law pilot-count branch. More counts, new seeds, a learned
denoiser, or TinyLLM retraining would only refine the same sampled threshold.
A successor experiment requires a distinct deployment hypothesis, such as
between-pilot bias drift or temporally correlated acquisition error.

## Reproduction and provenance

Primary artifact:
`data/experiments/tinyllm_noisy_bias_pilot_acquisition/20260810_d16_preregistered/campaign_results.json`

```bash
pixi run python -m \
  experiments.structure_net.tinyllm_noisy_bias_pilot_acquisition \
  --output \
  data/experiments/tinyllm_noisy_bias_pilot_acquisition/20260810_d16_preregistered
```

| Item | SHA-256 / value |
| --- | --- |
| campaign | `29d34222764215edb654d2741a85394223999a591cd7e2b44272612e1db8ccdf` |
| result manifest | `648f53acac88b625c357447bb930e55ae3312ff2555c1ef33ea72341ae527130` |
| implementation | `e034f234c3d0406cc62920de0b4a5efe591e1ca30f2d8d1f06c4fa6a29ab50b3` |
| runner | `827ff5b6d083b1e872954b5eb9a95a848632c555ecece1a51af54e1945573ed2` |
| preregistration | `2e19e9c7bf3908c97e29fd614a1801ef9e3eaec568a3babf3a1fe24adfa9830b` |
| pilot contract | `303c19e3ef8d35476f479f267226b0b6d4a2bcf7fd7cfc7e628e5a3f1c883583` |
| pilot arrays | `bfab87e36131fcfda8acdc33d7b9cce6e59c592dcca49c75e6d37b41410dcbc7` |
| pilot-array content | `81cea5d30b15c90b4101c015880afc98111ac03b23704c1ff4b22b45567a054c` |
| source exact-repair campaign | `1996ac4c2534b62a25a2f52ceadfd21055a91bdadc81f38ccf01c6855da2b7d0` |
| source acquisition campaign | `968f85010129d761268b4816d85ddd2ab578bbc93307e8a936e58fa891e89d93` |
| source acquisition arrays | `57eca80cccf1b916a60d79d5982bdbffe3b515cee7dfbee7645830448779aace` |
| meta-hypothesis record | `971198c6f9c884a3e6e79518b95e5aaf4f7fc5d776945c56b447199908ae8adf` |
| meta-hypothesis implementation | `ade26b0ac947abafe6a8c17a060aa83901d0f0084fc174890904ff73d61f4906` |
| DVC data root | `73d421913a00dfcd3efe8c5d66b88824.dir` |
| lakeFS commit | `ab65ecfef3f08620775705b96eb09f26b2e13e40380db60433edca20352f2d40` |
