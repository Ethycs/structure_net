# TinyLLM acquisition-draw stability

**Status:** VALID PREREGISTERED RESULT — `m=256` STABILITY CONFIRMED  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, frozen-system,
no-fit acquisition replication  
**Hypothesis:** `tinyllm-acquisition-draw-stability-v1`  
**Schema:** `nal.tinyllm-acquisition-draw-stability.v1`  
**Preregistration:** [acquisition-draw stability preregistration](../07%20-%20Status%20Reports/2026-08-10_tinyllm-acquisition-draw-stability-preregistration.md)

## Verdict

Two hundred fifty-six independent observations of the orientation reference
form a stable population-level recovery ceiling under the declared synthetic
noise process. The unchanged analytic and learned-front-end TinyLLM
populations pass composition and extrapolation simultaneously on all
**16/16** fresh acquisition draws. Every one of the 160 checkpoint-by-draw
cells passes.

Sixty-four observations are useful but not a stable ceiling. The joint
two-arm population gate passes **14/16** draws. The analytic population passes
`14/16`; the learned-front-end population passes `16/16`, although individual
checkpoints in both arms remain draw-sensitive. The locked classification is:

```text
m64_broadly_stable_checkpoint_variable
```

The preregistered primary `m=256 >= 15/16` hypothesis passes. No model, front
end, readout, denoiser, observer, or probe was trained or fit.

## Primary endpoint

A checkpoint passes only when its exact-bin accuracy loss from the unchanged
exact-reference baseline is at most three percentage points on both held-out
composition and extrapolation. An arm passes a draw at four of five
checkpoints; a complete draw passes only when both arms pass.

| Repeats | Analytic population draws | Learned population draws | Complete joint draws | 95% Wilson interval |
| ---: | ---: | ---: | ---: | ---: |
| `64` | `14/16` | `16/16` | **`14/16` (`87.5%`)** | `[0.640, 0.965]` |
| `256` | `16/16` | `16/16` | **`16/16` (`100%`)** | `[0.806, 1.000]` |

The two complete `m=64` failures are draws 6 and 9. Both arise in the analytic
population, which reaches only three of five checkpoint passes on those
draws. The learned population never falls below four of five.

## Checkpoint-level draw sensitivity

Each cell is the number of fresh acquisition draws on which that fixed
checkpoint passes both held-out shifts.

| Front end | Checkpoint seed | `m=64` | `m=256` |
| --- | ---: | ---: | ---: |
| analytic | `7` | `14/16` | **`16/16`** |
| analytic | `17` | `12/16` | **`16/16`** |
| analytic | `29` | `15/16` | **`16/16`** |
| analytic | `41` | **`16/16`** | **`16/16`** |
| analytic | `53` | **`16/16`** | **`16/16`** |
| learned front end | `7` | `14/16` | **`16/16`** |
| learned front end | `17` | **`16/16`** | **`16/16`** |
| learned front end | `29` | `15/16` | **`16/16`** |
| learned front end | `41` | **`16/16`** | **`16/16`** |
| learned front end | `53` | `10/16` | **`16/16`** |

The arm-level result and checkpoint-level result must be kept distinct. The
learned-front-end population passes every `m=64` draw because four robust
checkpoints absorb seed 53's failures; `m=64` is not a per-checkpoint ceiling.
Conversely, the analytic population is less redundant even though its seeds
41 and 53 are individually perfect on this draw set.

## How sharp is the boundary?

At `m=64`, 16 of 160 checkpoint-by-draw cells fail: seven analytic and nine
learned-front-end cells. These are near-boundary task failures rather than a
collapse of the acquisition estimate.

| Front end | Repeats | Shift | Mean accuracy loss | 95th percentile | Worst loss | Failing cells |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| analytic | `64` | composition | `1.53` pp | `2.83` pp | `3.22` pp | `4/80` |
| analytic | `64` | extrapolation | `1.33` pp | `2.83` pp | `3.61` pp | `3/80` |
| learned | `64` | composition | `1.74` pp | `3.13` pp | `4.00` pp | `7/80` |
| learned | `64` | extrapolation | `0.73` pp | `2.54` pp | `4.49` pp | `2/80` |
| analytic | `256` | composition | `0.20` pp | `1.17` pp | `1.76` pp | `0/80` |
| analytic | `256` | extrapolation | `0.35` pp | `1.56` pp | `1.86` pp | `0/80` |
| learned | `256` | composition | `0.47` pp | `1.46` pp | `1.56` pp | `0/80` |
| learned | `256` | extrapolation | `0.28` pp | `1.17` pp | `2.54` pp | `0/80` |

The acquisition geometry follows the expected standard-error scale without a
fit. Mean angular RMSE is `0.021999`/`0.021789` radians at `m=64` and
`0.011103`/`0.010944` at `m=256` on composition/extrapolation, close to the
analytic values `0.175/sqrt(64)=0.021875` and
`0.175/sqrt(256)=0.0109375`.

## Causal interpretation

This prospective replication closes the uncertainty left by the two earlier
arrays:

```text
one noisy observed reference
    -> quotient representation survives but exact-bin task utility fails
independent circular averaging
    -> orientation uncertainty contracts as m^-1/2
m=64
    -> usually enough for the two five-checkpoint populations,
       but individual systems and two complete draws remain vulnerable
m=256
    -> all fixed systems recover on every fresh draw
```

The result strengthens the acquisition-precision explanation. Recovery is
caused by a coherent input-side measurement intervention and requires no
change to the internal representation or task computation. It also corrects
an overly crisp reading of the predecessor campaigns: 64 observations are a
broad population threshold, not a stable system-level constant.

The comparison between frozen front ends is descriptive. Both arms receive
the identical analytic circular mean. The learned population's `16/16`
`m=64` result therefore reflects how its five trained systems use the observed
reference, not an advantage from a learned acquisition estimator.

## Controls and integrity

- exact-reference metrics replay the locked source in all ten systems;
- the inherited one-observation negative baseline passes `0/5` checkpoints in
  each arm;
- the fresh draw-zero fiber-shuffled `m=256` control passes `0/5` in each arm;
- all sixteen streams are distinct and their maximum absolute inter-draw
  correlation is `0.00720`, below the locked `0.05` ceiling;
- paired-sheet angular acquisition errors agree within `4.63e-8`;
- maximum aggregate unit-norm error is `2.22e-16`;
- every system-state hash remains unchanged;
- all 10 requested systems complete with no retry, exclusion, or failure;
- fitted or trained parameters: **zero**; and
- exact resume leaves the result tree byte-identical.

The one-system shakedown used the disjoint seed root `81026999`, passed all
lifecycle contracts, and is not pooled. The primary arrays use root
`81027026`; their SHA-256 does not alias either outcome-exposed predecessor
array.

| Item | SHA-256 / value |
| --- | --- |
| campaign | `968f85010129d761268b4816d85ddd2ab578bbc93307e8a936e58fa891e89d93` |
| implementation | `a0eae3da0dfcf74328ff0f2fa264a8e712b61f901337608f9e23ef93657d0440` |
| ten-result manifest | `d13e52a07423e507cef034c78b734219b85abc8468feae6313b52148fa95b163` |
| acquisition arrays | `57eca80cccf1b916a60d79d5982bdbffe3b515cee7dfbee7645830448779aace` |
| exact-resume tree manifest | `de8dc71acdb704c23e8e12d5ab592bb59a4fba5d0e6bd187098d37a5a695062f` |
| meta-hypothesis record | `3d054fe33cc61b863216ba1c13e7882d744ca5248daa770eff74757a6adb379b` |
| source orientation campaign | `876af062f9d0bdd365e2f1ffbb959d9ff2e5e4e277321b510bbe6a956456877f` |
| source calibrated campaign | `80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501` |
| device | NVIDIA GeForce RTX 2060 SUPER (`cuda:1`) |
| peak CUDA allocation | `293,563,392` bytes |
| analysis time | `159.25` seconds |
| DVC data root | `36ec5a8382b3b89f0ac20a86dd53e0bb.dir` (`2,738` files; `40,097,909,308` bytes) |
| lakeFS commit | `29f78369b7eb1c5e944089caa17ddcecc86b72aca29a62c000dea7a87a3147d1` |

The DVC cache and configured `lakefs` remote are synchronized. The immutable
root object was verified at
`lakefs://artifacts/29f78369b7eb1c5e944089caa17ddcecc86b72aca29a62c000dea7a87a3147d1/structure-net/files/md5/36/ec5a8382b3b89f0ac20a86dd53e0bb.dir`;
the campaign, complete draw array, and meta-hypothesis blobs were also read
back at the same commit. The lakeFS branch has no uncommitted diff.

## Program decision

Close the independent-Gaussian sample-count branch. The shortest prescribed
replication has confirmed a stable `m=256` population ceiling and measured the
draw sensitivity at `m=64`; another seed sweep would refine a binomial
interval without changing the current mechanism.

This result licenses no model retraining, representation penalty, writer,
observer, denoiser, topology scan, or link-cobordism analysis. A future sensor
study would need a genuinely new deployment question—correlated errors,
systematic bias, or acquisition cost—rather than more optimization under the
solved independent Gaussian model.

## Artifacts and reproduction

- primary campaign:
  `data/experiments/tinyllm_acquisition_draw_stability/20260810_d16_preregistered/campaign_results.json`
- per-system records:
  `data/experiments/tinyllm_acquisition_draw_stability/20260810_d16_preregistered/runs/*/seed_*/result.json`
- complete fresh draw arrays:
  `data/experiments/tinyllm_acquisition_draw_stability/20260810_d16_preregistered/acquisition_draw_errors.npz`
- lifecycle-only disjoint shakedown:
  `data/experiments/tinyllm_acquisition_draw_stability/20260810_shakedown_cuda/`
- runner and tests:
  `experiments/structure_net/tinyllm_acquisition_draw_stability.py`,
  `tests/structure_net/test_tinyllm_acquisition_draw_stability.py`

```bash
MPLCONFIGDIR=/tmp/matplotlib-acquisition-draw-primary \
pixi run python -m \
  experiments.structure_net.tinyllm_acquisition_draw_stability \
  --device cuda:1 \
  --output \
  data/experiments/tinyllm_acquisition_draw_stability/20260810_d16_preregistered
```

## Scope boundary

The result covers sixteen fresh draws from one declared independent,
unbiased, homoscedastic Gaussian orientation-error process at `sigma=0.175`.
It covers five retained d8/N3 checkpoints in each of two structured front-end
arms. The 95% interval for a perfect 16/16 result still has a lower bound of
`0.806`; “stable” means the preregistered sampled-population gate, not a
universal guarantee. The experiment does not establish behavior under
correlated or biased noise, other noise scales, real sensing cost, natural
language, or an architecture population.
