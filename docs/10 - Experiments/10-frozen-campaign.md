# Experiment 10 — Frozen End-to-End Campaign

**Verdict:** COMPLETE; ACCURACY FLOOR MET, ROUTER MOSTLY ESCALATES  
**Date:** 2026-08-16

All Experiments 01–09 gates were true before the process opened test labels.
The canonical evaluation contains 7,965 unique eligible PAWS test groups.

| Measure | Result |
| --- | ---: |
| Routed accuracy | 78.29% |
| Routed balanced accuracy | 78.18% |
| A standalone accuracy | 55.92% |
| B standalone accuracy | 66.06% |
| C standalone accuracy | 78.83% |
| Mean relative cost | 150.74 / 156.86 |
| Cost reduction vs C | 3.91% |
| Escalation rate | 99.75% |
| OOD fallback rate | 1.98% |
| ECE | 0.071 |

Routed accuracy was 0.54 percentage points below C, satisfying the frozen
one-point accuracy floor. Subgroup accuracy was 74.89% for low lexical overlap,
83.33% for high overlap, and 82.93% for high word-order displacement. The
router works conservatively but provides little practical savings with the
current A/B heads; that negative engineering result is part of the conclusion.

Raw predictions, Qwen outputs, frozen features, NAL result, summary, and hashes
are under `data/experiments/paws_abc_routing/2026-08-16_experiment_10/`.
