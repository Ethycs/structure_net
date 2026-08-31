# Experiment 11 — Competence-Wavelet Router Follow-up

**Verdict:** COMPLETE; ACCURACY IMPROVED, COST DID NOT  
**Evidence class:** held-out development follow-up  
**Date:** 2026-08-16

The leakage-controlled construction produced finite 128-dimensional coordinates
for all 7,983 development groups. The 512-landmark graph was connected without
repair bridges, and basis orthogonality error was `1.52e-14`. The serialized
transform contains semantic normalization, the partition-1 competence atlas,
landmark carriers and competence distributions, and the 128-coordinate basis;
an unseen partition-3 prompt was successfully mapped using no label or success
input.

Partition-2 selection again chose unweighted 15-NN at `τ=0.80`. On the untouched
partition-3 routing evaluation:

| Router | Accuracy | Mean cost | Escalation | ECE |
| --- | ---: | ---: | ---: | ---: |
| Generic SmolLM/PCA | 77.02% | 150.94 | 99.61% | 0.046 |
| Competence wavelet | 77.66% | 152.73 | 99.95% | 0.070 |

The wavelet embedding improved held-out accuracy by 0.64 percentage points but
increased proxy cost by 1.79 and worsened calibration. It routed `A/B/C =
1/55/1,976`, compared with `8/72/1,952` for the generic router. Thus the missing
component works technically and changes the competence geometry in the expected
direction for accuracy, but it does not solve the economic-routing limitation:
the weak A/B success surfaces still force almost universal C usage.

This result does not reuse PAWS test and does not revise Experiment 10. Canonical
artifacts are under
`data/experiments/paws_abc_routing/2026-08-16_experiment_11/`.
