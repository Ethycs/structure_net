# TinyLLM cyclic writer-intertwiner audit

**Status:** NOT CONFIRMED — HARMONIC-MIXED WRITER IMAGE 3/3  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, algebraic frozen-artifact diagnostic  
**Hypothesis:** `tinyllm-c16-writer-intertwiner-v1`  
**Preregistration:** [cyclic writer-intertwiner preregistration](../07%20-%20Status%20Reports/2026-08-07_tinyllm-cyclic-writer-intertwiner-preregistration.md)

## Verdict

The current three-coordinate order-four writer does **not** carry the exact
cyclic task-phase action. All provenance, analytic-group, rank, and random-
subspace specificity controls pass, but the writer image leaves its own
three-dimensional subspace under `C16` rotation in all three checkpoints.
Every checkpoint is `harmonic_mixed_writer_image`; the registered
writer-intertwiner gate passes `0/3`.

The failure is structured rather than arbitrary. Each writer image has
`0.988--0.997` normalized overlap with the canonical first-harmonic-plus-
constant subspace and is dramatically more invariant than a random rank-three
subspace. Small contributions from higher harmonic charges nevertheless
produce maximum orbit obstructions of `0.089--0.201`, above the locked `0.05`
ceiling.

```text
almost first-harmonic rank-three writer
  + small mixed-charge improvement
  -> no closed three-dimensional C16 representation
  -> contragredient C16 task-phase covector transport is not well defined on
     this interface.
```

This is the shortest answer to “apply symmetry groups” for the existing
writer: the group is useful, and it exposes a necessary architectural defect
before another model or activation campaign is run.

## Primary gates

| Gate | Result | Required | Verdict |
| --- | ---: | ---: | --- |
| source provenance and analytic group controls | **3/3** | 3/3 | pass |
| finite rank-three writer image | **3/3** | 3/3 | pass |
| maximum `C16` orbit obstruction at most `0.05` | **0/3** | 3/3 | **fail** |
| induced action closure at most `0.05` | 2/3 | 3/3 | fail |
| lower obstruction than random fifth percentile | **3/3** | 3/3 | pass |
| complete writer-intertwiner gate | **0/3** | 3/3 | **fail** |

The exact nine-dimensional Fourier feature action closes to
`1.29e-15`. Each of the four canonical
`span(cos(m theta), sin(m theta), 1)` controls has zero measured subspace
obstruction and induced closure below `1.46e-15`. The negative result is
therefore not a matrix-convention or group-generator failure.

## Checkpoint evidence

| seed | rank | condition number | max obstruction | RMS obstruction | induced closure | random p05 | classification |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 7 | 3 | 4.55 | **0.2011** | 0.1421 | **0.1042** | 0.8252 | harmonic mixed |
| 29 | 3 | 2.16 | **0.1350** | 0.1055 | 0.0489 | 0.8248 | harmonic mixed |
| 53 | 3 | 2.66 | **0.0890** | 0.0689 | 0.0234 | 0.8238 | harmonic mixed |

All writer images are well conditioned. Seeds 29 and 53 happen to produce
least-squares three-dimensional generators whose sixteenth powers are near
identity, but this cannot rescue them: the primary subspace equation

```text
A W = W R
```

already fails. A closing fitted `R` does not mean that `A W` lies in the
writer image.

The 768 deterministic random rank-three controls have median maximum
obstructions near `0.893--0.894`; their fifth percentiles are
`0.824--0.825`. The actual writer values are much smaller, so training did
select a strongly phase-structured subspace. It did not select an invariant
one.

## Harmonic content

Normalized projector overlaps with the canonical invariant subspaces are:

| seed | harmonic 1 | harmonic 2 | harmonic 3 | harmonic 4 |
| ---: | ---: | ---: | ---: | ---: |
| 7 | **0.9877** | 0.3393 | 0.3294 | 0.3293 |
| 29 | **0.9935** | 0.3336 | 0.3323 | 0.3300 |
| 53 | **0.9967** | 0.3337 | 0.3315 | 0.3311 |

Different canonical subspaces share the constant direction, so their
baseline mutual overlap is `1/3`. The writers are therefore almost pure
first-harmonic-plus-constant interfaces, with small charged leakage. That
small leakage is consistent with the preceding capacity result: order four
improves mean causal error over order one, but neither writer passes a full
checkpoint.

Simply projecting back to the exact first-harmonic subspace is not a new
solution—it recovers the already tested order-one capacity branch. Exact
symmetry and causal sufficiency must be achieved together.

## Architectural consequence

Do not transport the measured local covector through the current rank-three
writer with a post-hoc `3 x 3` action. The necessary representation does not
exist at the preregistered tolerance.

The smallest constructive alternative is a typed direct sum:

```text
(cos theta, sin theta)       charge 1
oplus (cos 2theta, sin 2theta) charge 2
oplus ...
oplus constant              charge 0
  -> symmetry-allowed nonlinear fusion
  -> three-dimensional task-effective write.
```

Each harmonic block carries its exact known `C16` representation. The fusion
may combine charges only through declared neutral/equivariant products. This
retains the higher-harmonic corrections that helped the failed writer without
mixing incompatible charges into one ordinary three-vector.

**Superseded program instruction (2026-08-10):** this report originally
recommended comparing the order-four write with an exactly typed
harmonic-fusion write of matched capacity after the cheaper signed-residual
sensor test. That sensor, its action law, the exact groupoid decomposition, and
the posterior-coordinate rank ladder are now complete. Together they close the
post-hoc fitted-writer/sidecar branch: another Fourier or harmonic writer is no
longer the next experiment. Preserve the typed direct-sum construction only as
a candidate for a prospectively trained architecture whose sign, basis,
normalization, metric, fusion, and downstream embedding are fixed by design.
See the
[current frontier audit](../07%20-%20Status%20Reports/2026-08-10_tinyllm-interpretability-frontier.md).

## Campaign integrity

| Item | Value |
| --- | --- |
| requested / completed / failed / excluded | 3 / 3 / 0 / 0 |
| trained models / fitted writers / fitted observers | 0 / 0 / 0 |
| stored writer matrices | three `9 x 3` order-four maps |
| evaluated group elements | 16 per writer |
| random rank-three controls | 768 |
| device | CPU |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| analysis time | 2.22 seconds |
| implementation SHA-256 | `acec5b9dd0291ddf603c1b65bf2f526f57ef9ae274a39b4249219a80be894702` |
| campaign SHA-256 | `75cd3fc0c41a673fde883798bb9a6bfe9e86a65c7e8e155e50b20e09935f3eed` |
| final DVC data root | `9f9077c17fbbc668805088bf604deafc.dir` (`1,904` files, `39,816,811,567` bytes) |
| lakeFS snapshot | `8eccad2c763ea0230fde1e484b2d8c631dbe91524799c21920686bd23d704872` |

Fingerprint-matched resume returned the existing aggregate and preserved all
result hashes.
The final DVC root is synchronized to
`lakefs://artifacts/main/structure-net/`; the exact directory object exists in
the cited clean lakeFS commit and the branch reports no uncommitted objects.

## Artifacts and reproduction

- aggregate:
  `data/experiments/tinyllm_cyclic_writer_intertwiner/20260807_preregistered_diagnostic/campaign_results.json`
- per-checkpoint records:
  `data/experiments/tinyllm_cyclic_writer_intertwiner/20260807_preregistered_diagnostic/runs/seed_*/result.json`
- runner:
  `experiments/structure_net/tinyllm_cyclic_writer_intertwiner.py`
- tests:
  `tests/structure_net/test_tinyllm_cyclic_writer_intertwiner.py`
- meta-hypothesis:
  `data/meta_hypotheses/tinyllm-c16-writer-intertwiner-v1.json`

```bash
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
pixi run python -m \
  experiments.structure_net.tinyllm_cyclic_writer_intertwiner
```

## Method boundaries

`C16` is the exact synthetic task-output phase group, not the full acquisition
nuisance group. The audit tests the selected quotient-only order-four writer,
not every possible carrier or nonlinear equivariant architecture. Approximate
subspace invariance does not establish causal sufficiency. Random subspaces
are specificity controls rather than a population null over trained models,
and three selected checkpoints remain underpowered.
