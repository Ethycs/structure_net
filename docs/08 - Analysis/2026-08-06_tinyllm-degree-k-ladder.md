# TinyLLM degree-k finite-quotient ladder

**Status:** NOT CONFIRMED — MAP DEGREE GENERALIZES, FIBERS REMAIN DECODABLE  
**Date:** 2026-08-06  
**Conformance:** NAL-STD-EXPERIMENT `PREREGISTERED`  
**Hypothesis:** `tinyllm-degree-k-finite-quotient-ladder-v1`  
**Preregistration:** `../07 - Status Reports/2026-08-06_tinyllm-degree-k-ladder-preregistration.md`

## Verdict

The analytic carrier cleanly supports degree `k=1,2,3`: every one of the 15 d6
models had the target winding on both composition and extrapolation, with strong
alignment and persistent `H1`. But the intended finite quotients did not form for
`k=2` or `k=3`. Conditional sheet identity remained almost perfectly decodable
after block-1 MLP and at full depth, so both rungs passed 0/5 joint seeds.

This separates two facts that an output-only topology scan would conflate. The
models learn robust degree-k maps, but they retain the k-sheet fiber rather than
contracting it internally.

## Campaign integrity

All 15 requested arms completed for `k=1,2,3` and seeds `7,17,29,41,53`; none
failed. Every model used the same d6 architecture, 600-step budget, analytic
observed calibration carrier, examples, optimizer family, and evaluation splits.
Seed 7 additionally retained a stepwise degree trace and transition charge audit.

## Primary endpoints

| k | Composition: alignment / degree / normalized H1 | Extrapolation: alignment / degree / normalized H1 | Joint seeds |
| ---: | --- | --- | ---: |
| 1 | 0.9992 / 1.0 / 0.8497 | 0.9943 / 1.0 / 0.8225 | **5/5** |
| 2 | 0.9963 / 2.0 / 0.7835 | 0.9766 / 2.0 / 0.7789 | **0/5** |
| 3 | 0.9898 / 3.0 / 0.7464 | 0.9476 / 3.0 / 0.7470 | **0/5** |

The map gates passed for every row. The failures at `k>1` came from the independent
conditional branch endpoint:

| k | Post-MLP branch accuracy comp. / extra. | Full-depth branch accuracy comp. / extra. | Required ceiling |
| ---: | --- | --- | ---: |
| 2 | 0.9875 / 0.9715 | 0.9879 / 0.9709 | 0.55 |
| 3 | 0.9834 / 0.9701 | 0.9824 / 0.9693 | 0.3834 |

Task accuracy was `0.7957/0.6627` for `k=2` and `0.6623/0.5055` for `k=3` on
composition/extrapolation. The failure is therefore not merely failure to learn
the supervised map.

## Defect accounting

Seed 7 accumulated net indexed charge `1`, `2`, and `2` for target degrees `1`,
`2`, and `3`. The charge identity therefore matched `k` for the first two rungs
but not the third. These are finite-grid cells on the declared continuous input
lift, not interval-certified roots.

## Preregistered gates

| Gate | k=1 | k=2 | k=3 |
| --- | --- | --- | --- |
| map degree/alignment/resolution/H1 on both shifts | pass 5/5 | pass 5/5 | pass 5/5 |
| hidden finite-sheet contraction | not applicable | fail 0/5 | fail 0/5 |
| seed-7 net charge equals k | pass | pass | fail |
| full rung | **pass** | **fail** | **fail** |

The preregistered full ladder is not confirmed because all three rungs were
required.

## Interpretation and boundaries

Robust topological degree is easier here than internal quotient formation. A
fixed analytic front end can make the output map globally correct while an
unrestricted TinyLLM routes sheet identity through the residual stream. This is
direct evidence that map topology is not a sufficient proxy for fiber contraction.

The branch probe is one declared nonlinear estimator, and the defect audit is a
finite mesh. No claim is made about all decoders or an interval-certified defect
count.

## Artifacts and reproduction

| Artifact | Path |
| --- | --- |
| aggregate | `data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered/campaign_results.json` |
| weights, maps, probes, traces | `data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered/runs/` |
| aggregate SHA-256 | `cf12b76691da41b7bc15e47570bce324f6aaefc7c9f670ef68db1fa4d9421046` |

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
pixi run python -m experiments.structure_net.tinyllm_degree_k_ladder \
  --device cuda:0 \
  --output data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered
```
