# Fisher-Wavelet Competence Router Synthetic Preflight

**Status:** PREREGISTERED BEFORE EXECUTION  
**Date:** 2026-08-16  
**Hypothesis ID:** `fisher-wavelet-competence-router-synthetic-v1`

## Question and prediction

On a finite three-bit task with non-nested A/B/C competence regions, does a
Fisher-weighted task-wavelet representation enable lower routing loss than a
nuisance-only embedding while preserving a conservative OOD fallback?

Prediction: the wavelet router has lower penalized routing loss in at least
four of five seeds and the mean reduction is at least 20%.

## Fixed design

- Replication unit: one independently generated calibration/test split.
- Seeds: `7, 17, 29, 41, 53`.
- Task states: the eight vertices of the three-dimensional Boolean cube.
- State adjacency: Hamming distance one.
- Teacher signal: a fixed four-class soft distribution per state.
- Competence labels: fixed, explicitly non-nested Boolean rules for A/B/C with
  independent label noise.
- Calibration/test samples per seed: 640/512.
- Baseline: three-dimensional nuisance coordinates only.
- Intervention: four Fisher-weighted heat-kernel state coordinates plus the
  same nuisance coordinates at small weight.
- Estimator: inverse-distance 21-nearest-neighbor success estimate per model.
- Route: cheapest model with estimated success at least 0.70; otherwise C.
- Costs: A=1, B=2, C=4.
- Routing loss: selected-model cost plus 8 when the selected model fails.
- OOD check: synthetic states with a fourth active bit are always classified
  outside support and routed to C.

## Primary gates

The preflight passes only if all are true:

1. wavelet penalized routing loss is lower than baseline in at least 4/5 seeds;
2. mean relative loss reduction is at least 0.20;
3. every deliberately non-nested success signature occurs in calibration data;
4. 100% of declared OOD examples route to C;
5. the wavelet basis is orthonormal to numerical tolerance `1e-10` and all
   persisted values are finite.

## Secondary measurements

Report selected-model success, average cost, oracle routing loss, state-graph
connectivity, Laplacian eigenvalues, and signature counts. Secondary results
cannot rescue a failed primary gate.

## Outcome meanings

| Outcome | Meaning |
| --- | --- |
| all gates pass | synthetic mechanism is implemented and merits a real-model calibration study |
| geometry passes, routing fails | basis is valid but does not expose the competence seams adequately |
| routing passes, OOD fails | useful interpolation but unsafe fallback implementation |
| contract failure | implementation defect; no scientific interpretation |

## Artifacts and command

- Experiment: `experiments/structure_net/fisher_wavelet_competence_router.py`
- Tests: `tests/structure_net/test_fisher_wavelet_competence_router.py`
- Output root: `data/experiments/fisher_wavelet_competence_router/2026-08-16_preflight/`
- Command: `python experiments/structure_net/fisher_wavelet_competence_router.py`

This is a no-training synthetic preflight, not model-quality evidence.

