# Experiment 10 — Frozen End-to-End Campaign Preregistration

**Status:** LOCKED; TEST LABEL ACCESS FORBIDDEN UNTIL GATES 01–09 PASS  
**Date:** 2026-08-16

The executable must verify the canonical completion artifacts for Experiments
01–09 before it opens the PAWS test file. It then removes the seven conflicting
normalized groups registered by Experiment 01, evaluates each frozen model
once, applies the selected Experiment 07 router without refitting, and retains
every route and prediction. No threshold, checkpoint, PCA, lexical model,
stratum, graph, or wavelet parameter may change after test outcomes appear.

The router falls back to C when its nearest atlas distance exceeds the frozen
partition-1 99th-percentile support threshold. Report routed accuracy,
balanced accuracy, A/B/C standalone accuracy, mean relative cost, escalation,
OOD fallback, calibration error, routing regret against the cheapest
actually-correct model, and lexical-overlap/word-order subgroup results. The
relative costs are compute proxies, not measured billing.

The headline gate is descriptive rather than a promise of improvement: report
whether routed accuracy is within one percentage point of always-C and its
cost reduction. Failure does not permit re-selection on test. All raw Qwen
generations and artifact hashes remain auditable. WiC is outside the selected
PAWS campaign and may only be a separately preregistered transfer experiment;
its absence cannot be represented as PAWS evidence.
