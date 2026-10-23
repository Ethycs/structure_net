# Measure source-disjoint accuracy after the fixed recipe update

Date: 2026-10-22, during the fixed matched training run, before any new final
training-summary or test outputs/grades have been inspected. The original study
does not estimate held-out accuracy. Add a separately identified, exploratory
test follow-up: `ncm-recipe-update-test-v1`.

After both fixed final checkpoints are saved, generate once on all 64 original
frozen-body test cases, with their original cached body features, no source or
target truncation, greedy code generation and unchanged local argmax renderer.
The 32 new training pairs come only from the original train partition; verify
disjoint source IDs and source hashes. Reuse the exact original full-Llama parent
outputs after checking their campaign hash. These 64 cases were previously
inspected in diagnostics: this is a reused source-disjoint test cohort, not an
untouched benchmark or an independent confirmation.

Judge legacy, updated and parent summaries anonymously with the unchanged
qualified source-grounded rubric/parser, four workers, at most three attempts
for transport/schema failure only. Verify public LongMemEval source spans
before sending only public passages and generated summaries. Preserve all raw
replies. All 64 judgments must be valid; no output regeneration or checkpoint
selection based on test results. Report count/N and Wilson 95% intervals. A
95% point target requires 61/64; this is not a 95% lower-confidence-bound claim.

Run this as a diagnostic regardless of the training-competence result; it cannot
rescue a failed training gate or authorize further training. No new optimizer
updates, speed measurement, human-gold claim or production claim. Separate root:
`data/experiments/ncm_recipe_update/20261022_test_v1`, separate report/registry.
