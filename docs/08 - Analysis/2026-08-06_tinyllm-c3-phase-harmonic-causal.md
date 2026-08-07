# TinyLLM `C3` phase-harmonic causal decomposition

**Status:** CONFIRMED WITHIN THE FROZEN PHASE-SENSITIVE STRATUM — THE FIRST `3 THETA` HARMONIC IS CAUSALLY SUFFICIENT  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`  
**Hypothesis:** `tinyllm-c3-phase-harmonic-fusion-v1`  
**Preregistration:** [`2026-08-06_tinyllm-c3-phase-harmonic-causal-preregistration.md`](../07%20-%20Status%20Reports/2026-08-06_tinyllm-c3-phase-harmonic-causal-preregistration.md)

## Verdict

The first symmetry-allowed `C3` phase harmonic is the causal phase channel in
all three checkpoints previously classified as phase-sensitive. Under both
composition and extrapolation, continuously twirling the charged carrier made
the frozen quotient continuation fail. Restoring only the `3 theta` Fourier
pair made it pass and explained `0.864`--`0.995` of the finite-phase downstream
Fisher effect. The minimal sufficient prefix was therefore exactly
`Q0+Q3+Q-3` in all six selected seed/shift cells.

The two prespecified mixed comparators reproduced their earlier support
dependence: phase twirling was sufficient under composition but failed under
extrapolation, where the same first harmonic restored the task. This
comparison was descriptive and could not promote the primary result.

The result turns the symmetry recommendation into a concrete architectural
constraint:

> A degree-three equivariant module should retain the charged real carrier and
> expose its first neutral discrete-phase channel to the invariant readout.
> A purely radial `O(2)` norm deletes a causally required computation in the
> phase-sensitive stratum.

The result identifies phase frequency, not unique polynomial order. A
`3 theta` response is consistent with cubic invariants such as `c1^3+c2^3`,
but higher nonlinear terms can contribute at the same frequency.

## Campaign integrity

This checkpoint-only study reused five retained d6 `k=3` models, the same 64
exact orbits as the predecessor irrep campaign, and its frozen first synthesis
transition under both shifts. It did not train, fit a probe, select a new cut,
or change the primary phase-sensitive stratum after observing outcomes.

Seeds 17, 29, and 53 were frozen as the primary stratum because the predecessor
classified all three as phase-sensitive under both shifts. Seeds 7 and 41 were
frozen descriptive comparators. Reusing the predecessor cohort makes this a
causal decomposition of an established event, not an independent replication
of its prevalence.

| Item | Value |
| --- | --- |
| requested / completed / failed | 5 / 5 / 0 |
| primary mechanistic stratum | seeds 17, 29, 53 |
| descriptive comparators | seeds 7, 41 |
| exact orbits | 64 per shift and seed; predecessor cohort reused |
| phase grid | 24 points on `[0,2 pi)` |
| allowed phase frequencies | `0, 3, 6, 9, 12` |
| training or fitted observers | none |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| total analysis time | 18.4 seconds |
| implementation SHA-256 | `073d2e65ca11664e6d50ef005e3aaba8a501a773ca2229d444eafa14086c2c3d` |
| campaign SHA-256 | `c6522d55e7cb5dd02434e23bcea40d125320b9dd13476dc8df5ad5d2dfee76a9` |
| DVC data root | `33e174317905dfe64832c047ed08135d.dir` |
| lakeFS commit | `58f19b9670fcae28b9f9ef93448ff1e40f139899990f04f21852e956a33a3018` |

A disposable eight-orbit seed-17 CUDA lifecycle and three exact spectral unit
tests completed before the primary run. The under-resolved lifecycle did not
enter evidence.

## Preregistered gates

The spectral contract applied to all five seeds. Mechanistic gates required
all three preselected seeds jointly across both shifts.

| Gate | Result | Required |
| --- | ---: | ---: |
| spectral `C3` group contract | **5/5** | 5/5 |
| eligible exact endpoint, selected stratum | **3/3** | 3/3 |
| finite-phase twirl is causally insufficient | **3/3** | 3/3 |
| first `3 theta` harmonic is sufficient | **3/3** | 3/3 |
| minimal sufficient prefix is shift-stable | **3/3** | 3/3 |

All gates pass under their frozen scope.

## Causal harmonic endpoints

Effect explained is relative to the exact natural-phase posterior versus the
continuous phase-twirl posterior. Values are composition / extrapolation.

| Seed | Target cut | Twirl passes | First harmonic passes | First-harmonic effect explained | Minimal prefix |
| ---: | --- | --- | --- | --- | --- |
| **17** | block-1 attention / block-0 attention | no / no | yes / yes | `0.864 / 0.995` | `3 / 3` |
| **29** | block-1 attention / block-1 attention | no / no | yes / yes | `0.908 / 0.953` | `3 / 3` |
| **53** | block-1 attention / block-1 attention | no / no | yes / yes | `0.897 / 0.887` | `3 / 3` |
| 7 | block-3 attention / block-1 attention | yes / no | yes / yes | `0.892 / 0.970` | `twirl / 3` |
| 41 | block-0 MLP / block-0 MLP | yes / no | yes / yes | `0.997 / 0.994` | `twirl / 3` |

The selected seed-17 composition effect was small in absolute Fisher geometry
(`0.00451`) but nondegenerate and causally decisive under the frozen conjunction:
the twirl missed the task gate and the first harmonic restored it. The other
selected finite-phase effects were `0.217`--`5.510`, so the stratum result is
not driven solely by near-threshold cells.

The comparator pattern matters. Under composition, their radial/phase-twirled
state already remained quotient-sufficient even though adding `3 theta`
changed the smooth posterior. Under extrapolation, the twirl failed and the
first harmonic became necessary. Phase dependence is therefore support-
relative in exactly the two models previously labeled mixed or shift-dependent.

## Spectral selection rule

The exact state response obeyed the predicted `C3` spectrum. Across all ten
seed/shift cells:

- maximum deck-periodicity relative error was `5.28e-8`;
- maximum forbidden-frequency state-energy fraction was `2.10e-13`;
- maximum forbidden-frequency posterior-energy fraction was `1.60e-11`;
- maximum full-DFT reconstruction error at natural phase was below `8e-17`.

The first harmonic dominated residual variation, carrying `0.876`--`0.982` of
nonconstant state-spectrum energy. It carried `0.532`--`0.940` of posterior-
spectrum energy. The lower posterior fraction in seed-17 extrapolation is
especially informative: higher harmonics visibly move the posterior, yet the
first harmonic alone explains `0.995` of the task-relevant exact-versus-twirl
effect and passes the full causal gate. Spectral energy and causal task effect
are not interchangeable.

| Seed | State `3 theta` energy, composition / extrapolation | Posterior `3 theta` energy | Causal effect explained |
| ---: | --- | --- | --- |
| 17 | `0.941 / 0.982` | `0.935 / 0.532` | `0.864 / 0.995` |
| 29 | `0.928 / 0.929` | `0.684 / 0.684` | `0.908 / 0.953` |
| 53 | `0.876 / 0.879` | `0.778 / 0.754` | `0.897 / 0.887` |

Higher allowed frequencies are real but secondary. State-space energy in
`6 theta` was `0.015`--`0.090` in the primary stratum; `9 theta` and `12 theta`
were smaller. None was required to reach the declared task-sufficiency gate.

## Interpretation

The shortest supported mechanism for phase-sensitive degree-three synthesis is:

```text
charged C3 carrier (c1,c2)
    -> first neutral discrete-phase response at frequency 3 theta
    -> quotient-sufficient invariant state
    -> higher allowed harmonics refine, but do not create, task sufficiency.
```

This resolves the ambiguity left by the earlier local Taylor failure. The
quadratic neutral interaction `c1*c2` is radial and survives continuous phase
twirling; it cannot explain why the twirl fails. The causally necessary piece
must depend on a finite-`C3` phase invariant. Its lowest representation-theory
frequency is exactly the observed `3 theta` pair.

The architectural implication is more precise than "use symmetry groups":

```text
carry:       c0, c1, c2
radial fuse: c1 x c2 -> c0
phase fuse:  expose a real 3-theta invariant channel
readout:     invariant c0 channels only
```

An implementation may realize the phase channel with explicit tensor-product
fusion, a complex carrier with a cubic invariant, or a finite-group
equivariant nonlinear basis. What the experiment rules out is projecting to
`c0` before fusion or replacing exact `C3` with continuous rotational
invariance.

## Boundaries

This is a post-stratified causal decomposition: the three primary seeds were
selected by a frozen predecessor outcome and evaluated on the same orbit
cohort. It confirms the harmonic mechanism inside that stratum, not that three
of five is a population-stable prevalence estimate. Continuous phase rotations
remain off natural orbit support. A `3 theta` response does not uniquely prove
a cubic Taylor term, and the result remains conditioned on the frozen decoder
and task gate.

## Artifacts and reproduction

- Aggregate:
  `data/experiments/tinyllm_c3_phase_harmonic/20260806_d6_preregistered/campaign_results.json`
- Per-seed records:
  `data/experiments/tinyllm_c3_phase_harmonic/20260806_d6_preregistered/runs/seed_*/result.json`
- Disposable lifecycle:
  `data/experiments/tinyllm_c3_phase_harmonic/shakedown_20260806/`
- Runner: `experiments/structure_net/tinyllm_c3_phase_harmonic.py`
- Tests: `tests/structure_net/test_tinyllm_c3_phase_harmonic.py`

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m experiments.structure_net.tinyllm_c3_phase_harmonic \
  --output data/experiments/tinyllm_c3_phase_harmonic/20260806_d6_preregistered \
  --device cuda:0
```
