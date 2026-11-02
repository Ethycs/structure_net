# Fisher-Wavelet Competence Router Synthetic Preflight

**Verdict:** PASS — the task-aware router reduced the declared synthetic
routing loss in 5/5 seeds and satisfied every geometry and fallback gate. This
is implementation evidence only; no language model was evaluated.  
**Date:** 2026-08-16  
**Hypothesis ID:** `fisher-wavelet-competence-router-synthetic-v1`

## Result

The Fisher-weighted wavelet router achieved mean penalized loss `2.71875`,
compared with `6.432421875` for the nuisance-only baseline. The mean relative
reduction was `0.5772423955`, exceeding the preregistered `0.20` threshold, and
the wavelet router improved on the baseline in all five fixed seeds.

| Primary gate | Result | Verdict |
| --- | ---: | --- |
| improving seeds | 5/5 | pass |
| mean relative loss reduction | 57.72% | pass |
| non-nested signatures represented | all five seeds | pass |
| declared OOD prompts routed to C | 64/64 | pass |
| eigenbasis orthonormality error | below `1e-10` | pass |

The eight-state Boolean cube retained all 12 structural edges under positive
Fisher weighting. The router estimated A, B, and C independently and therefore
represented complementary signatures (`100`, `101`, `010`, and `110`) rather
than forcing a scalar capacity hierarchy.

## Configuration

- Seeds: `7, 17, 29, 41, 53`
- Samples per seed: 640 calibration, 512 test
- Neighbors: 21, inverse-distance weighted
- Acceptance threshold: 0.70
- Costs: A=1, B=2, C=4; failure penalty=8
- Representation: first four heat-scaled eigenvectors of the
  Fisher-weighted normalized task Laplacian plus nuisance coordinates weighted
  by 0.08
- Baseline: the same nuisance coordinates without task coordinates
- Label noise: 0.03

## Execution and artifacts

Executed with:

```text
pixi run python -m pytest tests/structure_net/test_fisher_wavelet_competence_router.py -q
pixi run python experiments/structure_net/fisher_wavelet_competence_router.py
```

Focused tests passed `3/3`. The campaign aggregate is at
`data/experiments/fisher_wavelet_competence_router/2026-08-16_preflight/campaign_results.json`;
fingerprint-matched seed records are under its `runs/seed_<seed>/result.json`
subdirectories.

## Interpretation

The experiment establishes that the proposed components are jointly
executable and that exact task topology can protect routing from an embedding
containing only nuisance variation. It also demonstrates the required
non-nested competence representation and conservative unknown-state fallback.

It does not establish that Fisher weighting is better than an unweighted task
graph, that a cheap text encoder preserves real task states, that nearest
neighbors are calibrated probabilities, or that the method saves latency or
cost on real A/B/C models. The baseline intentionally lacks task information,
so the large effect is a positive-control systems result rather than a
competitive routing benchmark.

## Next licensed experiment

Use held-out prompts and objective success labels from three actual models.
Compare nuisance-only, unweighted task-graph, teacher-Fisher, and
competence-Fisher representations under distribution shift. Calibrate lower
confidence bounds and the support radius on a validation split before reading
test routing regret.
