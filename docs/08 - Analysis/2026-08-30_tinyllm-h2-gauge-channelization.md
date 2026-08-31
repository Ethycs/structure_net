# TinyLLM A8 H2 gauge channelization

**Status:** COMPLETED
**Hypothesis:** `tinyllm-h2-gauge-channelization-v1`
**Classification:** `gauge_channelization_sparsity_accuracy_tradeoff`
**Campaign artifact:** `data/experiments/tinyllm_h2_gauge_channelization/20260830_registered/campaign_results.json`
**Campaign SHA-256:** `cbd5f01d218c10ae22e8193a06ae34f058b011d3f41dffdb2cae1fd3d78dfa32`

## Result

Internal gauge synchronization substantially reduced the factor-only off-block
objective, and every fixed block arm passed the frozen A7 storage and operation
gates at 256 tokens. Hard projection nevertheless destroyed the A7 attention
certificate. The result is therefore a clean sparsity/accuracy tradeoff, not a
constructive compression pass.

All five evaluation seeds completed, covering 120 layer/head/input cells and
360 cell-lengths. Every parent hash, partition, exact-gauge, orthogonality,
causality, and explicit-versus-contraction validity check passed.

## Frozen primary results at 256 tokens

| Arm | Passing cells | Storage median / p90 | Operations median / p90 | Kernel-row error median / p90 | Compression |
| --- | ---: | ---: | ---: | ---: | --- |
| identity diagonal | 10/120 | 0.4947 / 0.5022 | 0.4363 / 0.4438 | 0.7837 / 2.7175 | pass |
| identity block 2 | 10/120 | 0.5350 / 0.5457 | 0.4766 / 0.4873 | 0.5225 / 2.2889 | pass |
| identity block 4 | 10/120 | 0.6158 / 0.6327 | 0.5574 / 0.5743 | 0.2868 / 1.7631 | pass |
| optimized diagonal | 0/120 | 0.4947 / 0.5022 | 0.4363 / 0.4438 | 0.6944 / 3.0289 | pass |
| optimized block 2 | 1/120 | 0.5350 / 0.5457 | 0.4766 / 0.4873 | 0.5377 / 2.4301 | pass |
| optimized block 4 | 1/120 | 0.6158 / 0.6327 | 0.5574 / 0.5743 | 0.3554 / 1.8847 | pass |

The representation requirement was at least 96/120 passing cells and at least
50% passing cells in every layer. No optimized arm approached either gate.
Kernel-row and denominator errors were the tightest failures: at 256 tokens the
optimized diagonal, block-2, and block-4 arms passed the kernel-row gate in
0, 1, and 1 cells and the denominator gate in 0, 2, and 2 cells, respectively.

## Factor objective across lengths

| Length | Diagonal final / reduction | Block 2 final / reduction | Block 4 final / reduction |
| ---: | ---: | ---: | ---: |
| 64 | 0.0133 / 77.2% | 0.00564 / 74.0% | 0.00184 / 69.0% |
| 128 | 0.3160 / 52.7% | 0.3067 / 48.6% | 0.2937 / 40.8% |
| 256 | 0.3675 / 51.4% | 0.3567 / 48.0% | 0.3445 / 43.0% |

The fixed local gauge rule therefore finds nearly separated factor channels at
64 tokens, but the retained off-block energy rises sharply by 128 and remains
large at 256. More importantly, minimizing unweighted Frobenius energy over
transfers and couplings is not aligned with the row-relative normalized-attention
certificate. Optimized gauges sometimes reconstruct less faithfully after hard
projection than the original SVD coordinates even while their factor objective
is lower.

## Validity and implementation

- preregistration SHA-256:
  `59451a7cc0e61de73335f5c455954b355ebb9611c5c5cb4133ec1d27acd4a53b`;
- implementation SHA-256:
  `5d2476077220e8f06ee49ea8b0e00bebd3410491cc5c767958e87470a6549758`;
- maximum exact-gauge assembly error: `2.05e-15`;
- maximum exact-gauge contraction error: `5.61e-16`;
- maximum gauge orthogonality residual: `5.43e-15`;
- maximum pruned explicit/contraction discrepancy: `4.26e-16`;
- completed seeds: `5/5`; failed seeds: `0`.

The explicit pruned operator was assembled recursively from leaf bases and the
stored pruned transfer paths. This is the operator actually applied by the H2
contraction; using A7's shortcut internal bases after pruning would incorrectly
compare two different operators. This correction was made and validated during
non-evidentiary shakedown, before the primary campaign.

The six width/restart candidates were executed as batched Cayley solves and
batched factor contractions. Seed-level workers ran concurrently on the RTX 3060
and RTX 2060 SUPER. This changed scheduling only: every candidate retained its
own parameters, Adam state, fixed 96 updates, and factor-only objective. Per-seed
analysis times were 1,290–1,407 seconds; parallel wall time was about 23 minutes.

## Interpretation

The A6 nested subspaces have genuine gauge freedom, but under the preregistered
scalar, 2-channel, and 4-channel masks they do not split into accuracy-preserving
independent channels. A7's quadratic-looking factor overhead cannot simply be
removed by coordinate rotation followed by hard block deletion.

This does not prove every gauge-aware compression is impossible. It rules out
the tested unweighted, operator-specific, fixed-block mechanism. A materially
different successor would need an error-aware or path-weighted sparsification
objective, larger/adaptive blocks, low-rank residuals, or a jointly learned
hierarchy. Such a successor must charge its metadata and cannot rescue A8.

No online construction cost, sparse-kernel wall-clock speed, cross-checkpoint
shared gauge, A5 rank reduction, or token-space causal descrambler is claimed.
