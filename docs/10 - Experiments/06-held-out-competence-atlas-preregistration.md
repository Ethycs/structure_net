# Experiment 06 — Held-out Competence Atlas Preregistration

**Status:** LOCKED BEFORE ATLAS OUTCOMES  
**Date:** 2026-08-16

Freeze, for each local model, the Experiment 04/05 checkpoint with highest
partition-0 balanced accuracy (ties: accuracy, then lexical experiment ID).
Evaluate the frozen A/B/C task interfaces on all remaining development groups
once and retain their partition assignments. Partition 1 is the competence
atlas, partition 2 is router selection/calibration, and partition 3 is the
strata/wavelet audit. The official test split remains unread.

For every group retain human label, exact prediction, correctness bit, local
class probabilities, frozen SmolLM pair representation, and all artifact
hashes. Qwen generation is resumable and uses the frozen Experiment 01 request.

Report all eight A/B/C success signatures on partition 1 and the non-nesting
rate

`P(A succeeds and B fails, or B succeeds and C fails)`.

No signature is merged because it is rare or contradicts model size. Completion
requires exact group alignment across both feature caches, selected loadable
checkpoints, terminal Qwen records, and one competence record per eligible
development group.
