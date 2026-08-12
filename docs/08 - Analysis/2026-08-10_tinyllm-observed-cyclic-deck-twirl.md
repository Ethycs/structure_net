# TinyLLM observed cyclic-deck twirl and causal-front replication

**Status:** VALID PREREGISTERED RESULT — FULL FRONT-REPLICATION HYPOTHESIS NOT CONFIRMED  
**Date:** 2026-08-10  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, frozen-checkpoint
causal replication  
**Hypothesis:** `tinyllm-observed-cyclic-deck-twirl-front-v1`  
**Schema:** `nal.tinyllm-observed-cyclic-deck-twirl-front.v1`  
**Preregistration:** [observed cyclic-deck twirl preregistration](../07%20-%20Status%20Reports/2026-08-10_tinyllm-observed-cyclic-deck-twirl-preregistration.md)

## Verdict

The full hypothesis is **not confirmed**. A one-observation calibrated action
reproduces the earlier generator-defined causal front within one cut for `C2`
in **4/5** checkpoints, but only **2/5** for `C3`, below the locked four-seed
gate.

The failed endpoint is front location, not mature quotient sufficiency. For
both `C2` and `C3`:

- the observed twirl destroys the task at the analytic carrier and block-0
  pre-attention in **5/5** seeds;
- the same twirl preserves the frozen task at full depth in **5/5** seeds on
  composition and extrapolation;
- the half-task-turn control remains valid for its own target in **5/5** seeds;
  and
- that control preserves the original source target in **0/5** seeds.

The conservative conclusion is:

> Mature finite-cyclic quotient sufficiency is constructible from one observed
> calibrated input for both `C2` and `C3`, but the precise causal-front cut is
> not invariant to replacing separately quantized generator fibers with a
> continuous within-example action. The sensitivity is substantially stronger
> for `C3`, especially under extrapolation.

The raw runner classification is `c2_only_observed_front`: `C2` passes every
registered population gate and `C3` does not. Under the preregistered outcome
table, the more informative failure branch is “mature twirls pass but front
replication fails.” No threshold or endpoint was changed after outcome.

## What was made observable

Starting from one decoded calibrated observation, the action subtracts its
observed planar offset and drift, rotates the corrected planar history by
`2 pi j/k`, and restores the same offset and drift. Calibration, signed speed,
amplitude, and the third channel remain unchanged. The action is continuous
after token decoding and is not re-quantized.

The constructor reads only:

```text
decoded sensor, observed offset/drift, fixed time grid,
declared cyclic group element.
```

It does not read latent phase, target posterior/bin, branch, quotient phase,
fiber ID, or another generator row. A contract test changed every non-anchor
generator sensor row by `+100`; the constructed orbit remained byte-identical.

The matched control uses the same cyclic orbit after an offset of `pi/k`. Its
degree-`k` target is therefore antipodal to the source target. This gives equal
group cardinality, spacing, nuisance, calibration, and norm while changing only
the semantic target.

## Campaign integrity

All ten requested frozen cells completed without retry, exclusion, training, or
fitting.

| Item | Value |
| --- | --- |
| checkpoint population | d6 analytic-carrier TinyLLMs, `k=2,3`, seeds `7,17,29,41,53` |
| held-out cohorts | 256 source anchors per checkpoint and shift |
| map cohort | 192 action-generated points per checkpoint and shift |
| cuts | carrier, block-0 pre-attention, each attention/MLP cut through block 2, full depth |
| requested / completed / failed | `10 / 10 / 0` |
| trained or fitted parameters | `0` |
| device | NVIDIA GeForce RTX 2060 SUPER (`cuda:1`) |
| peak CUDA allocation | `210,228,736` bytes |
| analysis time | `123.85` seconds |
| exact-resume primary tree | `e66ffd2a6c2c01366909f0221680150df3fe5f3ad207b75d8fe22c9ee30293a5` |
| exact-resume shakedown tree | `dcb315da6638ad2593b2d8781726de3f2956723a3600994e45bd31a06a711675` |

Every checkpoint and front-end file was loaded from the locked degree-ladder
campaign and state-validated. Continuation from every unpatched cut replayed
its captured posterior within the registered tolerance. Model and system state
digests were unchanged after analysis, all numeric records are finite, and a
second primary invocation left every artifact byte-identical.

## Preregistered gates

Counts require the same seed to pass composition and extrapolation jointly.

| Degree | Mature full-depth twirl | Pre-attention cover | Front within one cut | Control valid for shifted target | Control preserves source target | Full degree gate |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `C2` | **5/5** | **5/5** | **4/5** | **5/5** | **0/5** | **pass** |
| `C3` | **5/5** | **5/5** | **2/5** | **5/5** | **0/5** | **fail** |

The full hypothesis required every gate for both groups and therefore fails.
The `C3` result cannot be promoted by its positive mature endpoint.

## Front locations

The table reports `(observed distance from reference)` in cut positions for
composition / extrapolation. Distances at most one pass.

| Seed | `C2` | `C3` |
| ---: | ---: | ---: |
| 7 | `0 / 0` | `0 / 2` |
| 17 | `2 / 0` | `1 / 1` |
| 29 | `0 / 0` | `2 / 2` |
| 41 | `0 / 0` | `0 / 0` |
| 53 | `0 / 1` | `0 / 4` |

The population pattern is more stable than the individual fronts:

| Degree and shift | First cut reaching at least 4/5 observed preservation | Generator-defined reference at that cut |
| --- | --- | ---: |
| `C2` composition | block-1 post-attention | `4/5` |
| `C2` extrapolation | block-1 post-attention | `4/5` |
| `C3` composition | block-2 post-attention | `4/5` |
| `C3` extrapolation | block-2 post-attention | `5/5` reference versus `4/5` observed |

At full depth, both observed and reference twirls preserve all ten
checkpoint-shift populations. Thus the front mismatch is a depth-localization
failure, not disappearance of the mature quotient.

## Task effects and specificity

At full depth, accuracy change from correct twirling ranges from `-0.59` to
`+3.32` percentage points for `C2` and from `-0.26` to `+7.68` points for `C3`.
Every loss remains below the locked three-point ceiling. The control twirl has
similar performance for its own shifted target, but loses `62.30--83.01`
points for the original `C2` target and `51.30--70.83` points for the original
`C3` target.

Individual action elements are only approximately output-invariant. At full
depth, their mean posterior JS from the source sheet ranges over
`0.00084--0.00465` for `C2` and `0.00562--0.01314` for `C3`. This larger `C3`
dispersion is consistent with, but does not alone explain, its later observed
front.

## Action contract and quantization boundary

All 40 task/map, checkpoint, and shift contracts pass:

| Contract | Observed range |
| --- | ---: |
| applying the generator `k` times | maximum error `1.49e-7--4.77e-7` |
| analytic carrier equivariance | maximum error `1.19e-7--2.38e-7` |
| correct degree-`k` character preservation | maximum error `3.37e-7--8.99e-7` |
| control antipodal character | maximum error `2.67e-7--8.81e-7` |
| transformed planar support | maximum absolute value `0.775--1.993` |

The observed orbit is not numerically identical to the older
generator-defined orbit. The latter separately rotates the latent signal and
then quantizes every sheet; the present action rotates one already decoded
sheet without re-quantization. Relative sensor RMS between them is
`0.045--0.337` for `C2` (median `0.160`) and `0.057--0.454` for `C3` (median
`0.335`).

This gap supplies a concrete explanation for the localization result. It does
not invalidate the observed action: carrier equivariance and the task
characters are exact to numerical precision. It shows that the early nonlinear
continuation is sensitive to quantization-scale realization even when the
declared semantic action is exact, while the mature continuation is robust to
that distinction.

## Interpretation and decision

The experiment closes the mature oracle-membership question for the tested
finite cyclic groups. A generator does not need to enumerate the other sheets
for the frozen full-depth computation to use their Reynolds average; a single
calibrated observation suffices to construct them.

It simultaneously rejects a stronger universal statement:

```text
causal quotient front = architecture-only cut independent of
observation discretization and orbit realization.
```

The front is an intervention-relative transition. `C2` is stable enough to
replicate under the one-cut criterion, while `C3` shifts later under the
continuous observed action, most clearly on extrapolation. Full-depth quotient
sufficiency is the robust fact; exact first-preserved depth is not.

Do not retrain a group front end, add a representation loss, or tune the front
threshold from this result. The shortest same-scope finite-cyclic membership
checks are exhausted. Any next study must change a real assumption—such as
quantization-aware versus continuous action semantics, an unknown group frame,
a non-cyclic group, or a sampled architecture population—and preregister which
notion of action is causal.

## Artifacts and reproduction

| Item | SHA-256 / value |
| --- | --- |
| campaign | `7a1b099495f7ecb6c3eeea7c9b836411a5baee709eb6b51ab8103f88927e8a86` |
| implementation | `0c377e7a928e1926e916a981fec7464f0d4575e7f0e229bbbf1b7fc5298a0e56` |
| ten-result manifest | `52648e21fc8c3c47d62d588449fcc9cd1d36ac63302d836c5785eeb130f20184` |
| preregistration | `4ea38b10c77eede7a67d07c9d253a80ad058cd879ac49995cf762ad680f1dd59` |
| source degree ladder | `cf12b76691da41b7bc15e47570bce324f6aaefc7c9f670ef68db1fa4d9421046` |
| source deck intervention | `a3c14ce7022b7301344beaca876e0d454445c972a57de69c9cd4cd89098036b3` |
| meta-hypothesis record | `16a5e151ff89e02a73363a938fc4f6177a4d252c7ac86026e55e2c531309e9cb` |
| DVC data root | `2a706563f637fc52a746ee47c91357c4.dir` |
| lakeFS commit | `52348108433c33ce1842c7efa1dad8343c8bcca29e3f3122940405624d900a3c` |

- primary campaign:
  `data/experiments/tinyllm_observed_cyclic_deck_twirl/20260810_d6_preregistered/`
- per-checkpoint causal arrays:
  `data/experiments/tinyllm_observed_cyclic_deck_twirl/20260810_d6_preregistered/runs/k*/seed_*/observed_cyclic_deck_diagnostics.npz`
- systems-only shakedown:
  `data/experiments/tinyllm_observed_cyclic_deck_twirl/20260810_shakedown_k2_cuda/`

```bash
MPLCONFIGDIR=/tmp/matplotlib-cyclic-primary \
pixi run python -m \
  experiments.structure_net.tinyllm_observed_cyclic_deck_twirl \
  --device cuda:1 \
  --output \
  data/experiments/tinyllm_observed_cyclic_deck_twirl/20260810_d6_preregistered
```

## Boundaries

The result covers analytic calibrated carriers, ten retained d6 six-block
TinyLLMs, cuts through the first three blocks plus full depth, synthetic
`C2`/`C3` degree tasks, and two held-out N3 nuisance regimes. It does not
establish learned/raw front-end behavior, unknown
calibration, token-space group actions, anisotropic acquisition laws,
non-cyclic groups, real sensors, natural language, or prevalence across an
architecture distribution.
