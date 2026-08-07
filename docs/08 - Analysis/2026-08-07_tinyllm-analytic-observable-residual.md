# TinyLLM analytic observable-residual validity shakedown

**Status:** INVALID — STOPPED BEFORE PRIMARY EXECUTION  
**Date:** 2026-08-07  
**Conformance:** NAL-STD-EXPERIMENT `UNDERPOWERED`, source-only lifecycle evidence  
**Hypothesis:** `tinyllm-c2-analytic-observable-residual-v1`

## Verdict

The required one-checkpoint CUDA shakedown is invalid and supplies no evidence
for or against analytic observable-residual sufficiency. Provenance, input,
basis gauge, scalar replay, local linearization, target controls, source
covector replay, and the frozen-covector exact-scalar oracle all passed. The
registered observed-carrier contract failed, so the three-checkpoint campaign
is execution-forbidden.

Across the ten action cells, the worst observed-carrier values were alignment
`0.99678`, mean phase shift `0.15165` output bins, p95 shift `0.39155` bins,
and maximum paired-sheet neutral-carrier difference `0.58817`. The locked
ceilings were alignment at least `0.99`, mean shift at most `0.125`, p95 at
most `0.50`, and paired-sheet difference at most `0.01`.

The failed gate concerns the digital observation interface. The sensor is
decoded from 32 value tokens spanning `[-2, 2]`, giving a value-bin width of
`0.125`; an endpoint-derived paired-sheet tolerance of `0.01` was not achieved
after quantization. Candidate endpoint values generated downstream of that
failure are quarantined and must not be interpreted.

## Stop decision

Do not relax the post-run carrier threshold, fit another retrospective
observer, or run the primary campaign. The exact scalar groupoid decomposition
already established that a direct-state term is indispensable in all 24
action cells and closed the frozen-writer sidecar branch in the tested scope.

Retain the independently validated calibrated invariant front end. The next
prospective test should perturb its observed orientation reference along a
fixed noise curve, reuse the existing five checkpoints without TinyLLM
training, and retain the joint base-retention and conditional branch-leakage
gate.

## Integrity and artifacts

| Item | Value |
| --- | --- |
| checkpoint | seed `7` only |
| device | NVIDIA GeForce RTX 2060 SUPER (`cuda:1`) |
| TinyLLM models / writers / covectors / observers fit | `0 / 0 / 0 / 0` |
| generated but quarantined cells | `10` |
| analysis time | `7.08` seconds |
| peak CUDA allocation | `222,172,672` bytes |
| campaign SHA-256 | `88d684550e71d321837a52da83e202744364d06e29d460f3771f15356f1b7bf0` |
| implementation SHA-256 | `760a16d6926a03ce5afde63e50f0af8940a50da37315b628f19eaa8f5b0deb6b` |

- campaign:
  `data/experiments/tinyllm_analytic_observable_residual/20260807_shakedown_seed7/campaign_results.json`
- preregistration and lifecycle amendment:
  `docs/07 - Status Reports/2026-08-07_tinyllm-analytic-observable-residual-preregistration.md`
- runner:
  `experiments/structure_net/tinyllm_analytic_observable_residual.py`
- tests:
  `tests/structure_net/test_tinyllm_analytic_observable_residual.py`

```bash
MPLCONFIGDIR=/tmp/matplotlib-analytic-observable \
pixi run python -m \
  experiments.structure_net.tinyllm_analytic_observable_residual \
  --device cuda:1 --seeds 7 --allow-underpowered \
  --output \
  data/experiments/tinyllm_analytic_observable_residual/20260807_shakedown_seed7
```
