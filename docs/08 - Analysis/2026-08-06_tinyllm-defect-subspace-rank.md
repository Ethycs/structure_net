# TinyLLM stable C2 defect-subspace rank titration

**Status:** NOT CONFIRMED — THE STABLE QUOTIENT WRITE IS NOT SCALAR  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`  
**Hypothesis:** `tinyllm-c2-defect-subspace-rank-v1`  
**Preregistration:** [`2026-08-06_tinyllm-defect-subspace-rank-preregistration.md`](../07%20-%20Status%20Reports/2026-08-06_tinyllm-defect-subspace-rank-preregistration.md)

## Verdict

The stable degree-two quotient front is distributed not only over attention
heads but over more than one activation direction. A source-cohort rank-one
SVD projection failed the joint causal and `0.90` Fisher-preservation endpoint
in all three frozen checkpoints and all 12 held-out cohort/shift cells. Its
orthogonal complement was weak everywhere in only one of three checkpoints.
The preregistered scalar-write hypothesis is therefore **not confirmed**.

The narrower constructive result is strong. One checkpoint-local geometric
basis, fitted without task labels to source composition and extrapolation,
transferred to two disjoint held-out cohorts. The smallest sufficient ranks in
the preregistered dyadic grid were 2, 8, and 4 for seeds 7, 29, and 53. Rank 8 passed every held-out
cell in all three checkpoints. The stable quotient write is consequently a
low-dimensional vector-valued carrier under this frozen-decoder intervention,
not an arbitrary 1,152-dimensional residual and not a scalar line.

## Campaign integrity

The campaign reused three independently trained d6/N3 checkpoints whose
block-0 attention synthesis fronts were already cross-cohort stable. It fitted
only a checkpoint-local SVD basis from 128 source defects (64 exact `C2`
orbits under each shift), then froze that basis for four held-out cells. No
model, observer, decoder, or rank selector was trained.

| Item | Value |
| --- | --- |
| requested / completed / failed / excluded | 3 / 3 / 0 / 0 |
| retries / reused during initial execution | 0 / 0 |
| trained models / predictive observers | 0 / 0 |
| source / held-out cohorts | 1 / 2 disjoint cohorts |
| evaluation shifts | composition; outside-range extrapolation |
| exact orbits | 64 per cohort, shift, and checkpoint |
| residual features | 1,152 flattened token-channel coordinates |
| tested ranks | 1, 2, 4, 8, 16, 32, 64, complete source span |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| summed per-seed analysis time | 10.8 seconds |
| implementation SHA-256 | `f64803475e289782e575daa7f59642a0fdaba20ab19c3ce85a3c20ef9904f71e` |
| campaign SHA-256 | `b6586abd878a70819b8c7c921126c9cb86319f414886f7fa322d93535a05a324` |

All three source matrices had numerical rank 128 at the preregistered
`1e-10` relative tolerance. Maximum basis orthogonality error was `1.65e-13`;
maximum exact head-defect reconstruction error was `2.34e-7`, below `1e-6`.
Thirteen focused subspace and predecessor tests passed before primary launch.

The NAL draft recommends five seeds. This deliberately reused the three stable
early-front checkpoints fixed by prior evidence, so the report is marked
`UNDERPOWERED` and makes a checkpoint-cohort mechanism claim only.

## Preregistered gates

Every gate required all three frozen checkpoints.

| Gate | Result | Required | Verdict |
| --- | ---: | ---: | --- |
| numerical-rank and orthogonality contract | **3/3** | 3/3 | pass |
| zero fails and exact full defect passes in all held-out cells | **3/3** | 3/3 | pass |
| fixed rank-one projection is sufficient in all held-out cells | 0/3 | 3/3 | **fail** |
| rank-one complement preserves at most `0.50` everywhere | 1/3 | 3/3 | **fail** |
| deterministic random source-span line is insufficient | **3/3** | 3/3 | pass |

Controls validate the intervention but cannot rescue the two failed mechanism
gates. The full hypothesis remains false.

## Rank titration

Source energy is cumulative singular-value-squared energy. Held-out
preservation is the worst decoder-conditioned Fisher fraction over two cohorts
and both shifts. A minimum rank additionally had to pass the frozen causal task
conjunction in all four cells.

| Seed | Source energy r1 / r2 / r4 | Rank-1 held-out preservation | Minimum sufficient tested dyadic rank | Preservation at minimum rank | Causal cells at preceding rank |
| ---: | --- | --- | ---: | ---: | ---: |
| 7 | `0.573 / 0.959 / 0.9991` | `0.460–0.513` | **2** | `>=0.998999` | rank 1: 0/4 |
| 29 | `0.564 / 0.882 / 0.9988` | `0.306–0.458` | **8** | `>=0.999999` | rank 4: 3/4 |
| 53 | `0.463 / 0.863 / 0.9976` | `0.427–0.512` | **4** | `>=0.999931` | rank 2: 2/4 |

Rank one never crossed the causal gate. The random source-span line preserved
between `-0.0009` and `0.0013` and also failed every task cell, so the leading
line is meaningful but incomplete rather than an arbitrary one-dimensional
projection.

The rank-one complements never passed the causal task gate, yet their smooth
Fisher preservation ranged `0.453–0.479` in seed 7, `0.553–0.682` in seed 29,
and `0.464–0.613` in seed 53. Thus the first mode and its remainder each carry
substantial decoder-visible variation in two checkpoints; neither alone
implements the quotient.

## Hard task boundary versus smooth effective rank

Seed 29's declared rank 8 is conservative. Rank 4 already preserved at least
`0.999866` of the exact Fisher effect and passed three of four causal cells.
On held-out-B extrapolation its exact-bin accuracy was `0.671875` versus an
untouched baseline of `0.710938`, a `0.039063` loss that narrowly exceeded the
frozen `0.03` ceiling. Rank 8 restored `0.6875` and passed.

Seed 53's rank-2 projection preserved `0.963–0.972`, but failed both
composition task cells; rank 4 passed all four. Seed 7 was unambiguous: rank 2
preserved at least `0.998999` and passed all four cells.

The preregistered dyadic-grid answer remains ranks 2/8/4. The registered
boundary follow-up then tested intervening singular directions individually
and refined the stable ranks to 2/5/3. Seed 29's fifth direction corrects one
decoder-boundary example; seed 53's third direction restores a broadly
distorted continuous map. See
[`2026-08-06_tinyllm-defect-boundary-audit.md`](2026-08-06_tinyllm-defect-boundary-audit.md).

A subsequent source-only boundary-normal basis also repaired every checkpoint
with one added direction, but the next ordinary geometric direction and a
source-pair-shuffled normal basis did the same. That follow-up independently
confirms ranks `2/5/3` while rejecting boundary-normal fitting as a specific
mechanism. See
[`2026-08-06_tinyllm-defect-boundary-basis.md`](2026-08-06_tinyllm-defect-boundary-basis.md).

As a secondary
smooth-geometry description, the dominant source defect is approximately
four-dimensional: the top four modes explain `99.76–99.91%` of source energy
and reproduce nearly all posterior Fisher effect. Exact-bin thresholds make
that statement insufficient for the seed-29 hard endpoint.

## Mechanistic synthesis

Together with the exact six-head decomposition, the supported mechanism is:

```text
distributed contributions from four to five attention heads
  -> shared block-0 Reynolds defect
  -> two to three semantic modes plus an occasional margin-correction tail
  -> degree-two invariant posterior behavior.
```

This narrows the architecture decision. A one-dimensional symmetry adapter is
too small, but a compact multi-head sidecar carrying a three-dimensional
neutral carrier remains plausible. Exact-bin margin calibration should be
treated separately from semantic carrier dimension.

## Boundaries and anomaly record

The projection uses each held-out cell's exact defect, so it establishes
representational sufficiency, not independent computability from raw input.
The SVD is geometric, checkpoint-local, and not task-weighted. Fisher effects
and causal gates are frozen-decoder-conditioned, and projected residuals can
be off the natural activation manifold. Three previously selected stable
checkpoints do not establish prevalence across arbitrary training seeds.

The fingerprinted resume audit reused all three immutable per-run records but
the runner attempted to rewrite the aggregate timestamp and execution counters.
The original byte-identical campaign artifact was restored and the resume
event preserved separately in `resume_audit_20260806.json`; no scientific
outcome changed. Future runner revisions should make aggregate resume records
append-only.

## Artifacts and reproduction

- Aggregate:
  `data/experiments/tinyllm_defect_subspace_rank/20260806_d6_preregistered/campaign_results.json`
- Per-seed evidence:
  `data/experiments/tinyllm_defect_subspace_rank/20260806_d6_preregistered/runs/seed_*/result.json`
- Resume audit:
  `data/experiments/tinyllm_defect_subspace_rank/20260806_d6_preregistered/resume_audit_20260806.json`
- Disposable systems-only shakedown:
  `data/experiments/tinyllm_defect_subspace_rank/shakedown_20260806/`
- Runner: `experiments/structure_net/tinyllm_defect_subspace_rank.py`
- Tests: `tests/structure_net/test_tinyllm_defect_subspace_rank.py`
- Meta-hypothesis:
  `data/meta_hypotheses/tinyllm-c2-defect-subspace-rank-v1.json`

The meta-hypothesis write completed with authoritative persistent-store
readback of the failed verdict, exact hypothesis ID, and all three experiment
IDs. Legacy Chroma telemetry and NumPy compatibility warnings were non-fatal.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m experiments.structure_net.tinyllm_defect_subspace_rank \
  --device cuda:0 \
  --output data/experiments/tinyllm_defect_subspace_rank/20260806_d6_preregistered
```
