# Experiment 02 — Undistilled Baseline Preregistration

**Status:** LOCKED BEFORE OUTCOME INSPECTION  
**Date:** 2026-08-16

## Question

What semantic paraphrase competence is present before task-specific training in
TinyLLM d8 (A), SmolLM2-360M-Instruct (B), and served Qwen3-8B (C)? This is a
descriptive baseline, not a test that competence is nested by model size.

## Frozen sample and interfaces

Select 256 eligible PAWS development pair-groups, balanced 128/128 by human
label. Within each label, order the immutable group IDs by
`SHA256("17:" + group_id)` and take the first 128. The official test split is
not read. The dataset and ordered selection hashes are saved with every run.

All models receive the Experiment 01 prompt and use greedy exact generation.
Anything except `PARAPHRASE` or `DIFFERENT` is malformed and incorrect. No
candidate ranking, label extraction, or post-hoc repair is allowed. TinyLLM's
raw pretrained checkpoint is expected to have poor instruction compliance;
that is part of the undistilled baseline rather than a reason to substitute a
different task.

## Measures and decision rule

Report accuracy, balanced accuracy, malformed-output rate, runtime, every
prediction, and raw generation. There is no superiority gate and
no model is removed because of this experiment. Completion requires three
successful canonical NAL runs, exact sample/hash agreement, and a result file
for each model. A smaller run may validate lifecycle only and cannot replace
the 256-example result.

No routing, threshold, distillation, or wavelet choice may use PAWS test labels.
