# TinyLLM fixed semantic-gauge writer

**Status:** NOT CONFIRMED — COORDINATES TRANSFER, CAUSAL WRITE FAILS 3/3  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`, `UNDERPOWERED`, post-outcome positive control  
**Hypothesis:** `tinyllm-c2-fixed-semantic-gauge-writer-v1`  
**Preregistration:** [`2026-08-06_tinyllm-fixed-semantic-gauge-writer-preregistration.md`](../07%20-%20Status%20Reports/2026-08-06_tinyllm-fixed-semantic-gauge-writer-preregistration.md)

## Verdict

An observation-derived, analytically gauge-fixed `C2` carrier predicts each
checkpoint's rank-three defect coordinates extremely well, but its linear
writer does not reproduce the frozen continuation's causal behavior. The
fixed-gauge writer passes the continuous endpoint in only `3/12` held-out
cells and in no complete checkpoint. The full hypothesis is not confirmed.

This is a useful separation:

```text
stable semantic coordinates (held-out R2 >= 0.9737)
    do not imply
a decoder-compatible causal write (0/3 checkpoint gates).
```

The shuffled writer fails by several output bins, so the good coordinate fit
and partial seed-53 result are not explained by marginal distributions. The
failure is also not a task-control failure: zero, exact, and direct-rank-three
controls behave as required in every checkpoint.

## Intervention

For each observed calibrated sensor packet, the fixed front end recovers a
charged phase vector `(x, y)` without using latent phase or target labels and
forms the exact neutral fusion

```text
z = (x^2 - y^2, 2xy, x^2 + y^2).
```

A checkpoint-local, no-intercept ridge writer maps `z` into the previously
measured rank-three defect basis. It is fitted on the two alignment-fit
regimes and patched into the frozen block-0 continuation on two untouched
cohorts under composition and extrapolation. A regime-preserving shuffled
writer is the specificity control. No TinyLLM, encoder, decoder, observer, or
calibration parameter is trained.

## Campaign integrity

| Item | Value |
| --- | --- |
| requested / completed / failed / excluded | 3 / 3 / 0 / 0 checkpoints |
| trained models / predictive observers | 0 / 0 |
| fitted writers | 6: fixed plus shuffled per checkpoint |
| held-out cells | 12: 2 cohorts x 2 shifts x 3 checkpoints |
| exact orbits per cell | 64 |
| device | NVIDIA GeForce RTX 2060 SUPER, CUDA |
| PyTorch / Python | 2.5.1+cu121 / 3.11.13 |
| implementation SHA-256 | `4508847ca4fa85c2220e3691d8e9922d714e768b0f4891ce5237a4a74089c808` |
| campaign SHA-256 | `de80e30c23e06801c75d6fae899c67d0da82b86fdaff9158d94270597df8379c` |
| final DVC data root | `f29e1f0e920aff74661e2a64d7ec56c1.dir` (`1,796` files, `39,812,097,258` bytes) |
| lakeFS snapshot | `71cda38c5b84bfa364c136a0741dd4ff6e77040395f4e24b5d50d8419c11a648` |

All stored result hashes and frozen-checkpoint identities validate. The
campaign reuses the preregistered cross-seed carrier cohort and its locked
predecessor campaign. Runtime was `11.63` seconds because all models remain
frozen. The final DVC root is current locally, was pushed to the configured
`lakefs://artifacts/main/structure-net/` remote, and is contained in the cited
clean lakeFS commit.

## Primary gates

| Gate | Result | Required | Verdict |
| --- | ---: | ---: | --- |
| observation contract | **0/3** checkpoints | 3/3 | fail |
| continuous target controls | **3/3** | 3/3 | pass |
| fixed-gauge causal writer | **0/3** | 3/3 | fail |
| shuffled specificity | **3/3** | 3/3 | pass |
| complete hypothesis | **not confirmed** | every gate | fail |

The observation contract fails narrowly rather than catastrophically. Across
the four unique held-out cells, circular alignment is at least `0.99759` and
p95 phase error is at most `0.3712` output bins, but the maximum mean error is
`0.1412` against the preregistered `0.125` ceiling. Only heldout-B composition
passes that complete sensor contract.

## Checkpoint results

| Seed | Worst held-out coordinate R2 | Mean causal shift, bins | Passing cells | Checkpoint verdict |
| ---: | ---: | ---: | ---: | --- |
| 7 | 0.98534 | 0.20570 | 0/4 | fail |
| 29 | 0.97372 | 0.18517 | 0/4 | fail |
| 53 | 0.98868 | 0.09691 | 3/4 | fail |

Seed 53 is the informative near miss. Its first three cells have mean shifts
`0.0824`, `0.0995`, and `0.0799` bins. Heldout-B extrapolation reaches
`0.125917`, missing the mean ceiling by `0.000917` bins. Seed 7 fails uniformly;
seed 29 is substantially worse on both extrapolation cohorts.

Across all cells, the fixed writer has mean coordinate `R2 = 0.98786` but mean
causal shift `0.16259` bins. The shuffled control's mean shift is `3.9848`
bins. The correspondence is real, but coordinate-space residual size is not
the task metric.

## Interpretation and decision

Because the sensor contract failed narrowly, this campaign alone cannot say
whether better phase recovery would rescue the writer. It does establish that
an analytic neutral carrier plus a linear checkpoint-local write is not, as
implemented, a sufficient causal interface.

The preregistered oracle decomposition therefore substitutes exact latent
phase while keeping every frozen target, fit cohort, and held-out endpoint
unchanged. Its result is reported separately in
[`2026-08-06_tinyllm-fixed-gauge-error-decomposition.md`](2026-08-06_tinyllm-fixed-gauge-error-decomposition.md).

## Artifacts and reproduction

- Aggregate:
  `data/experiments/tinyllm_fixed_semantic_gauge_writer/20260806_d6_preregistered_diagnostic/campaign_results.json`
- Per-checkpoint records:
  `data/experiments/tinyllm_fixed_semantic_gauge_writer/20260806_d6_preregistered_diagnostic/runs/seed_*/result.json`
- Runner:
  `experiments/structure_net/tinyllm_fixed_semantic_gauge_writer.py`
- Tests:
  `tests/structure_net/test_tinyllm_fixed_semantic_gauge_writer.py`
- Meta-hypothesis:
  `data/meta_hypotheses/tinyllm-c2-fixed-semantic-gauge-writer-v1.json`

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
MPLCONFIGDIR=/tmp/matplotlib-structure-net \
pixi run python -m \
  experiments.structure_net.tinyllm_fixed_semantic_gauge_writer \
  --device cuda:0 \
  --output \
  data/experiments/tinyllm_fixed_semantic_gauge_writer/20260806_d6_preregistered_diagnostic
```

## Method boundaries

This is an analytic positive control, not a learned encoder. It requires the
observed calibration packet and exact analytic sensor decoder. Writers have
checkpoint-local target access on alignment-fit orbits, patches are
off-manifold interventions, and only three selected stable checkpoints are
tested. High coordinate `R2` is descriptive evidence, not causal success.
