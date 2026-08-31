# Experiment 07 — Embedding-Proximity Router Preregistration

**Status:** LOCKED BEFORE ROUTER OUTCOMES  
**Date:** 2026-08-16

Use Experiment 06 partition 1 as the immutable neighbor atlas and partition 2
for selection. Fit PCA with 32 components to the frozen SmolLM prompt
representation using partition 1 only, then standardize there. Compare
unweighted and inverse-distance k-nearest-neighbor competence estimators for
`k ∈ {5, 15, 31, 63}` and reliability thresholds
`τ ∈ {0.80, 0.85, 0.90, 0.95}`.

For each model, route on its 95% normal lower confidence bound, with effective
sample size computed from neighbor weights. Select the cheapest qualifying
model in A/B/C order; if none qualifies, fall back to C. Relative costs are
parameter-count proxies `(1, 360/51, 8000/51)` and are not latency claims.

The fixed lexical comparator uses character-token Jaccard, token-count ratio,
absolute length difference, shared-token fraction, and normalized word-order
displacement with three independent logistic correctness models.

Select among configurations whose partition-2 realized accuracy is at least
the always-C accuracy minus one percentage point, minimizing mean cost and then
maximizing accuracy. If none is feasible, maximize accuracy then minimize cost.
Partition 3 and PAWS test are unavailable to selection. Persist the chosen
router, all candidate metrics, calibration error, escalation rate, and routing
regret against the cheapest actually-correct model.
