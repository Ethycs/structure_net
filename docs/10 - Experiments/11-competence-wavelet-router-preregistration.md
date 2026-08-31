# Experiment 11 — Competence-Wavelet Router Follow-up

**Status:** LOCKED BEFORE FOLLOW-UP ROUTER OUTCOMES  
**Date:** 2026-08-16  
**Evidence class:** held-out development follow-up; PAWS test is unavailable

## Finding that motivates this experiment

Experiments 07 and 10 routed in a generic frozen SmolLM/PCA space. Experiment
09 subsequently constructed the requested task/competence wavelets, but those
coordinates were not connected back to the router. Therefore the 3.91% test
cost reduction is evidence about the generic semantic router, not a direct test
of the intended competence-wavelet router. This sequencing gap is an
experimental finding and an implementation omission, not evidence against the
original task-embedding hypothesis.

## Leakage-controlled construction

Rebuild the 24-dimensional carrier using the frozen semantic/lexical features
and three competence probabilities learned from partition 1. For partition-1
points, exclude the query itself from its 31-neighbor estimate. For partitions
2 and 3, use partition 1 normally. Thus no coordinate receives its own success
bit as a zero-distance feature.

Using partitions 1–2 only, select 512 deterministic stratum-proportional
landmarks and rebuild the Experiment 09 Fisher-weighted connected diffusion
basis. Define a genuine out-of-sample map without labels or success bits:

1. compute a new prompt's frozen carrier using predicted competence only;
2. connect it to its 12 nearest landmarks with carrier/Fisher affinity;
3. barycentrically extend the first 128 landmark wavelet coordinates.

True label and A/B/C success are allowed to construct calibration landmark
strata, but are forbidden inputs to the query extension.

## Selection and evaluation

Train the success-labeled neighbor atlas on partition 1. Search the same
`k ∈ {5,15,31,63}` and `τ ∈ {0.80,0.85,0.90,0.95}` grid on partition 2, with
the same one-point always-C accuracy floor and cost ordering as Experiment 07.
Evaluate the selected wavelet router exactly once on partition 3. Evaluate the
already-frozen generic Experiment 07 router on that same partition as the
comparator. Do not read or rerun PAWS test.

Completion requires a connected graph, orthogonality error at most `1e-5`, no
self-neighbor leakage, finite 128-dimensional coordinates for all 7,983
development groups, a serialized out-of-sample transform, and canonical NAL
selection/evaluation results. This is a development follow-up, not a new
confirmatory test claim.
