# TinyLLM degree-two character-fusion radius

**Status:** NOT CONFIRMED — EXACT CHARACTER RESPONSE IS SPECIFIC BUT NOT
DEPTH-LOCAL OR SHIFT-STABLE  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`  
**Hypothesis:** `tinyllm-c2-character-fusion-radius-v1`  
**Preregistration:**
[`2026-08-06_tinyllm-c2-fusion-radius-preregistration.md`](../07%20-%20Status%20Reports/2026-08-06_tinyllm-c2-fusion-radius-preregistration.md)

## Verdict

The preregistered early-versus-later character-fusion radius prediction did
**not** replicate on fresh exact orbits. All three early block-0-attention seeds
required radius `0.75`, rather than at most `0.50`. Neither later-front seed met
its joint finite-radius prediction: seed 41 turned on at `1.00` under
composition but `0.50` under extrapolation, while seed 17's frozen synthesis
cut failed at every primary composition radius and was already sufficient at
radius zero under extrapolation.

Only three of five onsets were shift-stable, below the required four. The full
hypothesis therefore fails.

The negative result leaves a narrower causal finding. Exact-character radial
responses were monotone through the observed radius in four of five seeds,
neither matched control reproduced their downstream effect in any seed, and
sheet exchange was exactly invariant. The degree-two character direction is
specific and causal, but its task-valid amplitude is not predicted by front
depth and is not uniformly stable across fresh cohorts and shifts.

## Campaign integrity

All five requested frozen-checkpoint cells completed with no failures or
retries and no retraining. Every cell validated the degree-two source
checkpoint digest and the corresponding frozen Reynolds character-coupling
record before analysis. Each regime used 64 newly generated exact
nuisance-matched orbits, distinct from the predecessor campaign.

| Item | Value |
| --- | --- |
| requested / completed / failed / reused | 5 / 5 / 0 / 0 |
| independently trained checkpoints | seeds 7, 17, 29, 41, 53 |
| shifts | composition; outside-range extrapolation |
| exact orbits | 64 per seed and shift |
| primary radii | 0.125, 0.25, 0.375, 0.50, 0.75, 1.00 |
| secondary radius | 1.25 |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| implementation SHA-256 | `9626d53d157cd27360001627b45695652ab3dc040187dc9174e82388c08d9724` |
| aggregate SHA-256 | `c98e34e3ff51cacec6bce9177d03fa07c66bed97cbeec66afc15af02bdcda8e9` |

Thirteen focused experiment and predecessor tests passed before launch. A
separate eight-orbit CUDA lifecycle was labeled
`systems_lifecycle_only_not_quality_evidence` and was not pooled.

## Fixed intervention

At the residual cut immediately before the previously measured synthesis
sublayer, each exact `C2` fiber was written as `b +/- delta`. For frozen
sublayer `F`, the campaign evaluated

```text
chi(s) = [F(b + s delta) + F(b - s delta)] / 2 - F(b)
```

and patched `F(b) + chi(s)` into the unchanged continuation. The primary onset
was the smallest positive radius through `1.00` passing the frozen causal
conjunction: circular alignment at least `0.90`, resolved sampling, winding
degree within `0.10` of two, and exact-bin accuracy loss no more than `0.03`
from the untouched checkpoint.

Controls replaced `delta` by a norm-matched cross-orbit character transplant
or a norm-matched random symmetric direction. They were scored against the
exact radius-one posterior rather than a control-specific target.

## Primary endpoints

| Seed | Frozen target cut | Composition onset | Extrapolation onset | Cohort prediction | Monotone through 1.0 |
| ---: | --- | ---: | ---: | --- | --- |
| 7 | block-0 post-attention | 0.75 | 0.75 | fail: early | pass |
| 17 | block-2 post-attention | missing | 0.125 | fail: later | fail |
| 29 | block-0 post-attention | 0.75 | 0.75 | fail: early | pass |
| 41 | block-1 post-attention | 1.00 | 0.50 | fail: later | pass |
| 53 | block-0 post-attention | 0.75 | 0.75 | fail: early | pass |

The seed-17 extrapolation onset needs a qualification: its radius-zero
propagated barycenter already passed on the fresh extrapolation cohort. The
frozen target sublayer therefore was not a synthesis event for that cohort.
Under composition, its exact radius-one patch missed only the accuracy-loss
component: untouched accuracy was `0.7578`, patched accuracy was `0.7188`, and
the loss `0.0391` exceeded the frozen `0.03` ceiling. Radius `1.25` passed, but
that off-manifold secondary intervention cannot define primary onset.

## Independent-cohort reconciliation

A contemporaneous preregistered
[orbit-radius titration](2026-08-06_tinyllm-orbit-radius-titration.md) evaluated
the same checkpoints and frozen cuts on a second independent 64-orbit cohort
with a finer primary grid. Its within-campaign degree-two gate passed: all five
zero/full endpoints replicated, all curves crossed once, and four of five
shift thresholds were within `0.125`.

That pass does not rescue this campaign's failed gates. Across both cohorts,
the three block-0-attention seeds 7, 29, and 53 had monotone, shift-stable
onsets. The later seeds did not: seed 17 changed from `0.50 / 0.25` to
`missing / 0.125`, and seed 41 changed from `0.75 / 0.75` to `1.00 / 0.50`.
Thus front depth did not predict the magnitude of onset as preregistered here,
but it did separate the three cross-cohort-stable early fronts from two
cohort-sensitive later fronts in these retained checkpoints.

## Preregistered gates

| Gate | Result | Required |
| --- | ---: | ---: |
| early-front onset at most 0.50 | **fail, 0/3** | all seeds 7, 29, 53 |
| later-front onset at least 0.75 | **fail, 0/2** | both seeds 17, 41 |
| shift-stable onset | **fail, 3/5** | at least 4/5 |
| monotone causal response through radius 1.0 | **pass, 4/5** | at least 4/5 |
| control specificity | **pass, 5/5** | at least 4/5 |
| sheet-exchange invariance | **pass, 5/5** | 5/5, error at most `1e-6` |

All measured exchange errors were exactly zero. Across every seed, shift, and
primary radius, the largest control Fisher-effect explained fraction was
`0.531`, below the preregistered `0.70` reproduction threshold.

## Mechanistic measurements

The three early-front seeds show why local Taylor sufficiency and causal onset
are different measurements. At radius `0.50`, their exact response already
explained `0.929`--`0.988` of the full radius-one downstream Fisher effect and
had circular alignment `0.970`--`0.988`. Their winding degree was already two.
Nevertheless, all six seed/shift cells failed the hard exact-bin accuracy-loss
gate until radius `0.75`.

Their response direction was recognizable still earlier. At radius `0.125`,
the residual cosine to the radius-one defect was `0.937`--`0.964`. The radial
norm then saturated: adjacent log slopes generally fell from approximately
`1.7`--`1.8` near radius `0.25` to `0.55`--`0.78` by radius `1.00`. The effect is
an aligned, finite-amplitude response rather than a constant quadratic law over
the full observed ray.

Seed 41 behaved differently. Its small-radius response cosine to the full
defect was only `0.294` under composition and `0.305` under extrapolation at
radius `0.125`; it rose above `0.93` only at radius `0.75`. This supports a
genuinely rotating or nonlinear character response in that checkpoint, but the
causal onset still differed by `0.50` across shifts.

The secondary radius exposed two additional nonuniformities. Seed 29
composition lost the hard causal pass again at radius `1.25`, while seed 17
composition first passed there. These outcomes do not alter the primary
monotonicity gate, which ended at the observed radius `1.00`.

## Interpretation and boundaries

This campaign falsifies a simple inference from the predecessor result:
strong quadratic Fisher-effect approximation does not imply that a small
physical character amplitude is task-sufficient. The earlier quadratic patch
rescaled a local finite difference to predict the full defect; the present
intervention instead asked how much exact character amplitude the frozen
sublayer needs. Those are distinct causal questions.

For group-structured architecture design, the evidence favors preserving typed
character carriers and learning an amplitude-aware invariant fusion or radial
gate. It does not support replacing the observed computation with a universal
small-radius quadratic contraction. Exact orbit direction matters—both
controls failed—but the needed amplitude depends on the checkpoint and shift.
The independent cohort narrows that further: early block-0 fusion is the
repeatable regime, while later fusion should not be treated as a fixed
checkpoint property without cohort replication.

The binary onset is sensitive to the hard exact-bin accuracy ceiling. With 64
orbits, one or two changed predictions can move a cell across the `0.03`
boundary even when smooth posterior geometry changes little. The report
therefore preserves the preregistered verdict while separately reporting the
Fisher and alignment curves; those secondary measurements cannot rescue the
failed gates.

The intervention follows one observed residual-space ray. It does not recover
a global group representation, identify unique isotypic coordinates, or prove
that arbitrary states lack sheet information. Repeating the patch across each
fiber makes within-orbit identity constant by construction. Fisher--Rao
measurements remain conditioned on the frozen task decoder.

## Artifacts and reproduction

- Aggregate:
  `data/experiments/tinyllm_c2_fusion_radius/20260806_d6_preregistered/campaign_results.json`
- Per-seed records:
  `data/experiments/tinyllm_c2_fusion_radius/20260806_d6_preregistered/runs/seed_*/result.json`
- Frozen checkpoints:
  `data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered/runs/k2/seed_*/model.pt`
- Frozen source comparator:
  `data/experiments/tinyllm_reynolds_character_coupling/20260806_d6_preregistered`
- Independent cohort aggregate:
  `data/experiments/tinyllm_orbit_radius_titration/20260806_d6_preregistered/campaign_results.json`
- Runner: `experiments/structure_net/tinyllm_c2_fusion_radius.py`
- Tests: `tests/structure_net/test_tinyllm_c2_fusion_radius.py`
- Meta-hypothesis:
  `data/meta_hypotheses/tinyllm-c2-character-fusion-radius-v1.json`

The conservative failed verdict and five direct experiment records were read
back from the configured ChromaDB collections under the stable hypothesis ID.
Telemetry warnings from the legacy Chroma client did not affect that
authoritative read-back.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m experiments.structure_net.tinyllm_c2_fusion_radius \
  --device cuda:0 \
  --output data/experiments/tinyllm_c2_fusion_radius/20260806_d6_preregistered
```
