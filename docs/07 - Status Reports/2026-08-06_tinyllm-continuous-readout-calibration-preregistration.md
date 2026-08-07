# TinyLLM continuous-readout and bin-calibration preregistration

> **Execution integrity note (2026-08-06):** The authoritative campaign root is
> `data/experiments/tinyllm_continuous_readout_calibration/20260806_d6_preregistered_v2`.
> The first root is excluded because a shared source file changed during its
> sequential execution and produced mixed implementation digests. No endpoint,
> rank rule, calibration rule, or threshold changed. The runner was hardened to
> pin one implementation digest and abort on drift before the clean full rerun.

**Status:** PREREGISTERED — PRIMARY OUTCOMES NOT INSPECTED  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`  
**Hypothesis:** `tinyllm-c2-semantic-carrier-readout-separation-v1`  
**Initial schema:** `nal.tinyllm-continuous-readout-calibration.v1`  
**Authoritative corrective schema:** `nal.tinyllm-continuous-readout-calibration.v1.1`

## Question

Can the semantic dimension of the stable degree-two defect be separated from
the frozen decoder's 16-bin margin sensitivity using only a continuous circle
endpoint and one fixed scalar boundary calibration?

## Frozen scope

Reuse the three cross-cohort-stable block-0 checkpoints, seeds `7,29,53`, and
the source-fitted geometric bases from the rank campaign. Train no transformer,
adapter, hidden-state observer, or nonlinear readout. Retain one source cohort
and two untouched held-out cohorts under composition and extrapolation.

This selected three-checkpoint mechanism study is `UNDERPOWERED`; it does not
make a five-seed prevalence claim.

The three generator cohorts are disjoint by their fixed seed maps:
`source_selection=(2501,101)`, `heldout_a=(4501,101)`, and
`heldout_b=(9101,307)`, where the pair is `(offset, regime stride)` and the
evaluation seed is `checkpoint seed + offset + stride * regime index`.

## Source-only semantic-rank selection

Evaluate geometric ranks `1,2,3,4,5`. For posterior `p`, use its first circular
moment

```text
mu(p) = arg sum_j p_j exp(2 pi i j / 16).
```

A rank preserves the continuous map in a cell only when:

1. circular-alignment loss relative to the exact-defect state is at most
   `0.005`;
2. mean absolute moment displacement is at most `0.125` bin widths;
3. 95th-percentile displacement is at most `0.50` bin widths;
4. winding remains within `0.10` of degree two and sampling remains resolved.

For each checkpoint, select the smallest rank passing all four conditions on
both source shifts. Freeze it before held-out evaluation. The semantic-core
hypothesis requires selected rank at most three and the same continuous gate
on all four held-out cells.

## One-parameter discrete calibration

For the selected source rank, fit one global circular boundary rotation
`delta in [-0.5,0.5]` bin widths on a fixed 4,097-point grid. It maximizes
source exact-bin accuracy jointly across composition and extrapolation. Ties
select minimum absolute rotation, then the smaller signed value. This is an
explicit one-scalar supervised readout calibration; no held-out outcome enters
the fit.

Apply the frozen rotation to held-out moment angles and quantize to the nearest
of 16 bins. Fit the same one-parameter calibrator to exact-defect source states
as a positive-control ceiling.

The calibrated selected-rank readout passes a cell only when exact-bin
accuracy falls no more than `0.03` below the untouched-model baseline. Both
selected-rank and exact-state calibrators must pass all four held-out cells.

## Primary gates

The separation hypothesis requires all three checkpoints to pass:

1. source-selected continuous rank is at most three;
2. that rank passes the continuous endpoint in all held-out cells;
3. the exact-state scalar calibrator passes all held-out discrete endpoints;
4. the selected-rank scalar calibrator passes all held-out discrete endpoints.

Additionally, seed 29's held-out-B extrapolation must improve by at least one
of 64 examples relative to its uncalibrated selected-rank moment readout. A
control failure invalidates the corresponding interpretation and cannot be
rescued by another checkpoint.

## Interpretation

- Continuous ranks `<=3` and calibrated discrete pass: semantic carrier and
  decoder discretization separate cleanly.
- Continuous ranks `<=3`, calibration fails: semantic dimension is small, but
  a single global margin parameter cannot replace the residual tail.
- Rank above three: the proposed common three-dimensional carrier is false.
- Exact calibrator fails: moment quantization is an inadequate readout family;
  do not interpret selected-rank calibration failure.

## Frozen provenance

The primary runner validates the source checkpoint hash and all available
per-seed predecessor identities. The frozen campaign records are:

| Predecessor | Campaign SHA-256 |
| --- | --- |
| defect subspace rank | `b6586abd878a70819b8c7c921126c9cb86319f414886f7fa322d93535a05a324` |
| authoritative decoder-boundary audit V3 | `869cd0bd6160164e2a83810e7088a4232a278767d1e76bec1fae59247ada8490` |
| source-fitted boundary basis | `c4cbcec89a0f64a94da29c4add90f61bdbc4b6a618caa70779663c1afbdd1a39` |
| Reynolds character coupling | `a7ccda0d8a36a5c96de96045a32400deaf3cdbdb0856d969164df8d6a455495b` |

The boundary audit only contains results for seeds `29,53`; its campaign
identity remains a declared provenance input, while the rank and boundary-basis
predecessors are required per seed. No boundary-audit measurement enters rank
selection or calibration fitting.

## Artifacts and execution plan

- disposable CUDA lifecycle:
  `data/experiments/tinyllm_continuous_readout_calibration/shakedown_20260806/`;
- primary aggregate:
  `data/experiments/tinyllm_continuous_readout_calibration/20260806_d6_preregistered/campaign_results.json`;
- primary per-seed records: the corresponding `runs/seed_*/result.json`;
- runner: `experiments/structure_net/tinyllm_continuous_readout_calibration.py`;
- focused tests:
  `tests/structure_net/test_tinyllm_continuous_readout_calibration.py`.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run python -m \
  experiments.structure_net.tinyllm_continuous_readout_calibration \
  --device cuda:0 \
  --output \
  data/experiments/tinyllm_continuous_readout_calibration/20260806_d6_preregistered
```

## Method boundaries

This is a decoder-conditioned causal sufficiency study, not an intrinsic
dimension theorem. Held-out states use exact held-out defects projected into a
source-fitted basis, so the experiment does not show that the low-rank state is
computable from raw observations. The supervised scalar rotation is explicitly
a readout calibration. All three checkpoints were selected for stable early
fronts, and the resulting three-seed claim is mechanistic and underpowered.

## Amendment A — pre-outcome evidence-pedigree and resume repair

Recorded before this investigation inspected any primary outcome. A concurrent
producer was later found to have been active during the amendment window, so
this amendment must not be described as certainly predating every artifact.
The initial runner default named the excluded original decoder-boundary audit
root. It now binds to the authoritative schema-v1.1 corrective root,
`20260806_d6_preregistered_v3`, and requires the declared evidence role and
producing implementation identity for every available predecessor record.

The same amendment adds a byte-immutable completed-aggregate resume guard and
the standard scheduled/retry/exclusion, result-reference, and execution
fields. These changes affect evidence pedigree and lifecycle integrity only;
no rank, threshold, cohort, seed, calibration grid, endpoint, or gate changed.

## Amendment B — post-outcome corrective replication

Recorded after the concurrent primary root and its outcomes became visible.
That root is preserved at `20260806_d6_preregistered`; it used the corrected
predecessor paths and scientific protocol but was produced by implementation
SHA-256 `b6b768445ae1f963415779f6d966a52792a30718f824d37e9bf23b274ce7ac26`
while the lifecycle and evidence-pedigree guards were still changing. Its
campaign SHA-256 is
`6e15fee43bcc218f2248975ecc40e1e826baeb915f7364690cbe2a0a090b8f27`.

The current runner uses schema
`nal.tinyllm-continuous-readout-calibration.v1.1` and can produce an explicitly
labelled `post_outcome_corrective_replication_evidence` campaign under
`20260806_d6_preregistered_v2`. The correction changes no scientific setting
or endpoint. Because all primary outcomes are now known, agreement may verify
reproduction and mechanism description but is not fresh confirmatory evidence.

## Amendment C — stale-bytecode exclusion and pinned corrective root

Recorded after `_v2` completed. That root is excluded: its records declare
schema v1 and `preregistered_underpowered_mechanistic_evidence`, while its
stored implementation SHA-256 identifies the contemporaneous raw v1.1 source.
The only plausible producer account is a stale Python bytecode image executing
while the source changed. It reproduced the numerical outcomes but cannot
serve as identified evidence.

The authoritative correction is therefore pinned to
`20260806_d6_corrective_v3_6a88480c`, schema v1.1, implementation SHA-256
`6a88480cf37e0f03819a73d03685d0bb5ab7cd0c6554b84194d2fac9b9a127d6`,
and evidence role `post_outcome_corrective_replication_evidence`. It must be
launched with an empty alternate `PYTHONPYCACHEPREFIX`, and the loaded schema,
role, and digest were checked before execution. As in Amendment B, this can
verify reproduction but cannot restore fresh confirmatory status.
