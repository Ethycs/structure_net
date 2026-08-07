# TinyLLM nuisance-scalar transformation law

**Status:** REJECTED — THE CORRECTION SCALAR IS ACTION-DEPENDENT, `0/3`  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, existing-artifact mechanistic diagnostic  
**Hypothesis:** `tinyllm-c2-nuisance-scalar-transformation-law-v1`  
**Schema:** `nal.tinyllm-c2-nuisance-scalar-transformation-law.v1`  
**Preregistration:** [nuisance-scalar transformation-law preregistration](../07%20-%20Status%20Reports/2026-08-07_tinyllm-nuisance-scalar-transformation-law-preregistration.md)

## Verdict

The signed correction required by the frozen order-four writer is **not** an
invariant scalar under the declared target-preserving observed similarity
actions. Seeds `7`, `29`, and `53` are all valid and all receive the locked
classification `scalar_action_dependent`. The complete invariance gate passes
`0/3` checkpoints and `0/24` action cells.

This falsifies the proposed output type before prospective training. An
invariant head satisfying `s(gx)=s(x)` cannot predict a correction target for
which `y(gx) != y(x)`. The next architecture must retain the observed action or
calibration context and represent a symmetry defect, or avoid the frozen
writer-relative correction by constructing the invariant semantic estimate
directly.

The failure is far from the thresholds. Across all cells, paired
zero-referenced R2 ranges from `-2.886` to `-0.341`, relative L2 from `1.158`
to `1.971`, and sign agreement from `36.5%` to `65.6%`. The locked requirements
were respectively at least `0.90`, at most `0.316`, and at least `90%`.

## Why this is a valid negative

Every provenance, input-pair, deterministic replay, carrier-basis gauge replay,
local-linearization, and target-control contract passes in all three
checkpoints. Across the recomputed reference and transformed cells:

- minimum fine/coarse derivative cosine is `0.99999943`;
- maximum fine/coarse derivative relative L2 is `0.00115`;
- minimum signed-error linearization R2 is `0.99731`;
- sign agreement for the exact local linearization is `100%`; and
- the zero state fails while exact and direct-rank-three controls pass.

The corrected replay aligns each regenerated rank-three basis to the stored
group-campaign coordinates before signed quantities are compared. Maximum
all-cell coordinate replay error is `3.14e-6`, below the locked `1e-5`
tolerance. This contract was added after the first root exposed an immutable
resume mismatch; that root is retained as superseded provenance, while the
gauge-replay root below is authoritative. The correction changes none of the
reported scientific measurements or classifications.

The action law therefore fails while its diagnostic target is finite, locally
linear, and causally meaningful. No model, writer, encoder, or observer was
trained or fit for this study.

## Primary measurements

The table pools three checkpoints within each shift and action. Entries are
means of `(paired R2, relative L2, sign agreement)` followed by the number of
cells passing the complete invariance gate.

| Shift | Action | Mean paired metrics | Passes |
| --- | --- | --- | ---: |
| composition | amplitude | `(-1.510, 1.584, 53.9%)` | `0/3` |
| composition | orientation | `(-1.578, 1.605, 41.9%)` | `0/3` |
| composition | offset | `(-2.654, 1.911, 43.3%)` | `0/3` |
| composition | composed | `(-1.629, 1.621, 46.9%)` | `0/3` |
| extrapolation | amplitude | `(-0.698, 1.302, 51.4%)` | `0/3` |
| extrapolation | orientation | `(-0.516, 1.231, 62.3%)` | `0/3` |
| extrapolation | offset | `(-1.051, 1.431, 55.5%)` | `0/3` |
| extrapolation | composed | `(-0.406, 1.185, 47.8%)` | `0/3` |

Pure amplitude, orientation, and offset actions each invalidate invariance, so
the result is not driven only by a difficult composed transformation.
Correlation between paired scalars ranges from `-0.259` to `0.278`; the
reference correction is not merely rescaled by a consistent action law.

The optional phase-matched nuisance shuffle is correspondence-descriptive, not
part of the mathematical invariance gate. Its `0.10` R2 margin passes only
`5/24` cells and cannot rescue the failed primary law.

## Mechanistic interpretation

The semantic task is invariant under the declared similarity actions, but the
measured scalar is not the semantic target. It is the signed output correction
from a checkpoint-local frozen order-four prediction to its direct rank-three
state. It therefore inherits the frozen model's symmetry defect.

Writing the two frozen task angles as `d(x)` for the direct state and `p(x)`
for the order-four prediction gives

```text
y(x) = wrap(d(x) - p(x))

y(gx) - y(x)
  = wrap((d(gx) - d(x)) - (p(gx) - p(x))).
```

There is no contradiction with the earlier calibrated-front-end result. The
calibrated analytic and learned encoders estimated an identifiable invariant
semantic quantity directly and passed in `5/5` seeds. This experiment asks a
different question: whether one invariant scalar can repair a nuisance-sensitive
frozen internal writer. The answer is no for all three tested checkpoints.

The component split is now:

```text
phase-conditioned task covector: portable on the tested fresh cohorts
invariant correction amplitude: structurally incompatible with the target
exact example-local correction: causally sufficient
```

## Decision and completed shortest test

Do **not** train the previously proposed invariant scalar head. Do not interpret
its prospective failure as an optimization result, because its target type is
already wrong.

The registered no-fit [scalar groupoid-defect decomposition](2026-08-07_tinyllm-scalar-groupoid-defect.md)
then used the same stored group cells to:

1. measure the transformed direct-state task-angle change;
2. measure the transformed order-four task-angle change;
3. verify their signed difference reconstructs `y(gx)-y(x)`;
4. test whether the direct-state term is negligible enough that the correction
   reduces to the negative frozen-writer symmetry defect; and
5. retain calibration/action context explicitly if both terms are required.

The exact two-term identity closed in all 24 cells, but the direct-state term
was material in every cell and the writer-only approximation passed none. That
closes the frozen-writer sidecar branch in the tested scope and selects the
already successful calibrated invariant front end as the constructive
architecture.

Link cobordism is not the next tool for this failure. There is no canonical
codimension-two defect locus to recover: the decisive obstruction is the
measured transformation type of the scalar target.

## Campaign integrity

| Item | Value |
| --- | --- |
| checkpoints requested/completed | `3/3` (`7`, `29`, `53`) |
| action cells | `24` (`4` actions x `2` shifts x `3` checkpoints) |
| trained models / fitted modules | `0 / 0` |
| source group cohort | seeds `430007/430008`, `64` exact C2 orbits per regime |
| device | NVIDIA GeForce RTX 2060 SUPER (`cuda:1`) |
| peak CUDA memory | `275,850,240` bytes |
| analysis time | `19.61` seconds |
| implementation SHA-256 | `e92c3894b3e8820c0242edeba29e1f893f321e7f1dc20d75583c898ce7b1526f` |
| campaign SHA-256 | `1e1795c931335c739a38550b85def7c4b442be39afcea76af8cd8491b1ff6589` |
| source group campaign SHA-256 | `2aa80cff5882cb79754d732e9012c810c2624934bc5be9e03c9e77de4f525f55` |
| DVC root | `68567037e5d128c412329a3843fea67f.dir` (`1,967` files; `39,818,433,620` bytes) |
| lakeFS snapshot | `6d7ac9ecb76b68fb1b38e3cfe7bcaf80b9272e3bc5eaebe72d1c296960740117` |

The checkpoints are replication units; the 24 action cells are repeated
measurements. The evidence remains underpowered for checkpoint-population
prevalence. Exact resume verified the completed aggregate and left its bytes
unchanged.

## Artifacts and reproduction

- campaign:
  `data/experiments/tinyllm_nuisance_scalar_transformation_law/20260807_d6_existing_group_gauge_replay/campaign_results.json`
- per-checkpoint records and arrays:
  `data/experiments/tinyllm_nuisance_scalar_transformation_law/20260807_d6_existing_group_gauge_replay/runs/seed_*/`
- superseded pre-gauge-contract root (audit only):
  `data/experiments/tinyllm_nuisance_scalar_transformation_law/20260807_d6_existing_group/`
- runner:
  `experiments/structure_net/tinyllm_nuisance_scalar_transformation_law.py`
- tests:
  `tests/structure_net/test_tinyllm_nuisance_scalar_transformation_law.py`

```bash
MPLCONFIGDIR=/tmp/matplotlib-nuisance-scalar \
pixi run python -m \
  experiments.structure_net.tinyllm_nuisance_scalar_transformation_law \
  --device cuda:1 \
  --output \
  data/experiments/tinyllm_nuisance_scalar_transformation_law/20260807_d6_existing_group_gauge_replay
```

The exact DVC directory object is present at
`lakefs://artifacts/6d7ac9ecb76b68fb1b38e3cfe7bcaf80b9272e3bc5eaebe72d1c296960740117/structure-net/files/md5/68/567037e5d128c412329a3843fea67f.dir`
in the cited clean snapshot. The meta-hypothesis readback verified all three
checkpoint records.
