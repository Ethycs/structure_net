# TinyLLM scalar groupoid-defect decomposition

**Status:** NOT CONFIRMED — WRITER-ONLY REDUCTION REJECTED; TWO-TERM DEFECT, `3/3`  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, no-fit existing-artifact diagnostic  
**Hypothesis:** `tinyllm-c2-scalar-groupoid-defect-v1`  
**Schema:** `nal.tinyllm-c2-scalar-groupoid-defect.v1`  
**Preregistration:** [scalar groupoid-defect preregistration](../07%20-%20Status%20Reports/2026-08-07_tinyllm-scalar-groupoid-defect-preregistration.md)

## Verdict

The action-dependent correction is **not** primarily the negative symmetry
defect of the frozen order-four writer. All three checkpoints receive the
locked classification `two_term_groupoid_defect`; the writer-dominance gate
passes `0/3` checkpoints and `0/24` action cells.

The direct-rank-three state contributes a material—and usually dominant—action
term in every cell. Its RMS is `0.808--1.004` times the total scalar-defect RMS,
far above the registered `0.10` ceiling, and its p95 magnitude is
`0.359--0.616` bins, far above the `0.05`-bin ceiling. The writer-only
approximation reaches zero-referenced R2 only `-0.008--0.348`, relative L2
`0.808--1.004`, and sign agreement `48.3--74.6%`; the required gates were
respectively `>=0.90`, `<=0.316`, and `>=90%`.

This rejects a sidecar that models only the frozen writer's nuisance defect.
It does not contradict the separately completed first-order decomposition:
that study shows the *coordinate residual* changes while the local task
covector remains stable. Together they imply that the coordinate residual
contains a required direct-state term that is absent from the deployed frozen
writer interface. The residual-coordinate result is supported; its reduction
to an observable writer-only defect is rejected.

## Exact decomposition

For direct-state angle `d(x)`, writer angle `p(x)`, and correction
`y(x)=wrap(d(x)-p(x))`, every observed action obeys

```text
delta_y(g,x) = wrap(delta_d(g,x) - delta_p(g,x)).
```

The identity is a validity contract, not a scientific result. It confirms that
the measured components exhaust the action defect: maximum reconstruction
error is `2.22e-15` bins and maximum relative L2 is `9.79e-16` across all 24
cells.

Every action defect is nondegenerate. Scalar-defect RMS ranges from `0.194` to
`0.303` bins and p95 magnitude from `0.368` to `0.661` bins. The direct-state
term has RMS `0.187--0.290` bins; the writer term is smaller at
`0.033--0.144` bins but is not zero. Thus `two_term_groupoid_defect` is the
correct registered label even though the direct term dominates descriptively.

## Primary measurements

The table pools the three checkpoints within each action and shift.

| Shift | Action | Mean scalar RMS | Mean direct/total RMS | Mean writer-only R2 |
| --- | --- | ---: | ---: | ---: |
| composition | amplitude | `0.260` | `0.962` | `0.074` |
| composition | orientation | `0.298` | `0.967` | `0.065` |
| composition | offset | `0.270` | `0.910` | `0.169` |
| composition | composed | `0.250` | `0.929` | `0.134` |
| extrapolation | amplitude | `0.284` | `0.910` | `0.170` |
| extrapolation | orientation | `0.203` | `0.942` | `0.109` |
| extrapolation | offset | `0.217` | `0.913` | `0.160` |
| extrapolation | composed | `0.280` | `0.926` | `0.141` |

The result is not driven by one checkpoint. Mean direct/total ratios are
`0.984`, `0.879`, and `0.934` in seeds `7`, `29`, and `53`. The best individual
writer-only R2 is only `0.348`, in seed 29; no action cell approaches the
registered prediction gate.

## Relation to the first-order result

The concurrent [scalar action-defect decomposition](2026-08-07_tinyllm-scalar-action-defect-decomposition.md)
uses the local factorization `J z`. It finds that changes in standardized
rank-three coordinate residual `z` explain the observed scalar defect at R2
`0.998--1.000`, while changes in task covector `J` explain essentially none.

The two studies answer different questions:

```text
first-order factorization:
  Which local factor changes?             residual coordinates, not covector

exact task-angle groupoid:
  Is the writer's own action defect enough?  no; direct-state term is required
```

The joint mechanistic statement is therefore

```text
portable phase-conditioned task covector
  x action-conditioned coordinate residual
  -> exact local correction,

but the coordinate residual cannot be reduced to the writer's action defect.
```

## Validity and lifecycle

All inherited and recomputed contracts pass in `3/3` checkpoints:

- transformation-campaign provenance and target controls;
- exact regenerated input identity;
- carrier-basis gauge and all-cell target-coordinate replay;
- signed-scalar replay, with maximum error `7.83e-7` bins;
- all-cell target-coordinate gauge replay, with maximum error `3.14e-6`;
- finite direct and writer task angles; and
- exact groupoid identity in all cells.

No model, writer, encoder, observer, or regression was trained or fit. A CPU
three-checkpoint replay and a one-checkpoint systems lifecycle are retained for
audit but are not pooled with the authoritative `cuda:1` campaign. The CPU
replay used the same repaired gauge source and reached the same classification
and every gate count. Exact resume left the authoritative campaign, result,
and NPZ bytes unchanged.

## Decision

Do not train either an invariant scalar head or a writer-defect-only sidecar.
The frozen sidecar branch is closed in the tested scope: recovering its missing
term would require a separately justified observable estimator of the direct
semantic state, at which point the already validated calibrated invariant front
end solves the constructive problem more directly.

Use that calibrated front end as the positive-control architecture. The next
prospective study, if needed, should degrade calibration quality or compare a
learned pilot estimator with the analytic control. It should not reopen
residual-sidecar optimization without a new observability argument.

No topology scan, wider retrospective probe, or representation penalty is
justified by this result. Link cobordism is irrelevant to this observability
failure because no singular or codimension-two defect locus is implicated.

## Campaign integrity

| Item | Value |
| --- | --- |
| checkpoints requested/completed | `3/3` (`7`, `29`, `53`) |
| exact action cells | `24` |
| trained or fitted modules | `0` |
| device | NVIDIA GeForce RTX 2060 SUPER (`cuda:1`) |
| peak CUDA memory | `275,856,896` bytes |
| PyTorch / Python | `2.5.1+cu121` / `3.11.13` |
| analysis time | `12.17` seconds |
| implementation SHA-256 | `81faf6321c87b452f22b8f155a437bbb0d4511f8826c3950f75768e0aad630e4` |
| campaign SHA-256 | `aeda0f8d90057a5983c75e81398000bfec4a5c133c5e2a476dc91cda97fc89d4` |
| source transformation campaign | `1e1795c931335c739a38550b85def7c4b442be39afcea76af8cd8491b1ff6589` |
| meta-hypothesis SHA-256 | `cb9dd3815f9840cceab300a407bb04124b3fcffc5a4681917e03280af6353dd6` |
| DVC root | `3f4a33e1f5698241a714c4337dc6dc3f.dir` (`2,015` files; `39,819,591,104` bytes) |
| lakeFS snapshot | `d162fdf4c3e31e55f41a2b3e52e801e3d1bb7e47c1f2c3bf093c6407ee4efcd4` |

The checkpoints are the replication units. This is a derived, post-outcome
mechanistic diagnostic on an already analyzed exact-group cohort and remains
underpowered for checkpoint-population prevalence.

The one-checkpoint CPU lifecycle exposed seed-7 quality fields before the
three-checkpoint launch. No hypothesis, metric, threshold, control,
classification, implementation, or source identity changed afterward. The
campaign is therefore sequential confirmation, not an independent replication
of that lifecycle result.

## Artifacts and reproduction

- campaign:
  `data/experiments/tinyllm_scalar_groupoid_defect/20260807_d6_existing_group_gauge_replay/campaign_results.json`
- per-checkpoint records and arrays:
  `data/experiments/tinyllm_scalar_groupoid_defect/20260807_d6_existing_group_gauge_replay/runs/seed_*/`
- CPU cross-device replay (audit only):
  `data/experiments/tinyllm_scalar_groupoid_defect/20260807_d6_existing_group/`
- systems-only lifecycle root:
  `data/experiments/tinyllm_scalar_groupoid_defect/20260807_shakedown_cpu/`
- runner:
  `experiments/structure_net/tinyllm_scalar_groupoid_defect.py`
- tests:
  `tests/structure_net/test_tinyllm_scalar_groupoid_defect.py`

```bash
MPLCONFIGDIR=/tmp/matplotlib-scalar-groupoid \
pixi run python -m experiments.structure_net.tinyllm_scalar_groupoid_defect \
  --device cuda:1 \
  --output \
  data/experiments/tinyllm_scalar_groupoid_defect/20260807_d6_existing_group_gauge_replay
```

The exact DVC directory object is present at
`lakefs://artifacts/d162fdf4c3e31e55f41a2b3e52e801e3d1bb7e47c1f2c3bf093c6407ee4efcd4/structure-net/files/md5/3f/4a33e1f5698241a714c4337dc6dc3f.dir`.
The lakeFS branch was clean after commit. ChromaDB readback verified the
meta-hypothesis and all three checkpoint records with result hashes
`19d1be56091746ad`, `1457bb015e45a346`, and `6e10f46c889709dc`.
