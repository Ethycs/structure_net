# Test accuracy after the NCM recipe update

**Updated summary task accuracy: 0/64.**
Classification: `updated_test_accuracy_invalid`. Date: 2026-10-22.
ID: `ncm-recipe-update-test-v1`. EXPLORATORY; one training seed.
[Fixed follow-up](../07%20-%20Status%20Reports/2026-10-22_ncm-recipe-test-amendment.md).
Aggregate: `data/experiments/ncm_recipe_update/20261022_test_v1/campaign_results.json`.
SHA256: `04a654e1e95a24521d235b749b6e347c3c8bd4b93b4dcf48f0a01b98703e6500`.

| Candidate | Passing complete summaries | Task accuracy estimate |
| --- | ---: | ---: |
| legacy | invalid | invalid |
| updated | invalid | invalid |
| parent | invalid | invalid |

The test source conversations and exact source texts are disjoint from the 32
quality-checked training pairs. These are the original 64 test cases, previously
inspected during diagnosis; this is not an untouched benchmark or independent
confirmation. No checkpoint selection, training, regeneration or output repair
used the test scores. The updated and matched legacy recipes share initialization,
data, steps and frozen body features. The full-parent output was reused after
checking its immutable campaign hash. The parent used the original 32-word
instruction; regenerated training targets requested 24 words to fit the codec.

The judge is `claude_code/claude-sonnet-4-6`, with unchanged source-grounded faithfulness,
salience, readability, nonempty and <=128-token checks. It previously accepted
24/24 good controls and rejected 24/24 bad controls, authored/source-reviewed by
Codex rather than independent humans. Candidates were anonymous and shuffled.
Public LongMemEval source file SHA and exact spans were verified before every
grading stage. Only public passages and generated summaries were uploaded.
All raw replies and claim evidence are retained. Valid low grades were not
retried. All 64 packets valid: `False`. Transport/schema retries: `3`.

Point target95% requires 61/64. Wilson intervals for every model are in the
aggregate; they exclude judge error and model-seed variation. This small,
LLM-judged test does not establish human-rated production accuracy. It cannot
rescue a failed training-competence gate. No throughput or full application
input/output-range claim is made: source prefix512, codec canvas48 GPT2 tokens.

Reproduction uses the moba Python, offline model cache, physical GPU0 selected
before CUDA, `LD_LIBRARY_PATH='' OMP_NUM_THREADS=2 PYTHONPATH=src:.`:

```text
python -m experiments.structure_net.ncm_recipe_test generate
python -m experiments.structure_net.ncm_recipe_test grade --report NEW_REPORT
python -m experiments.structure_net.ncm_recipe_test register --report NEW_REPORT
```

Add the memory dependencies and tiktoken cache for CPU/network grading. Pinned
artifact roots are explicit in source; new runs require a new producing version.

## Validity clarification and separate candidate recovery

The headline denominator64 is the requested cohort, not a valid complete-packet
accuracy denominator: only63 original comparison packets validated. The original
comparison remains invalid and its task-accuracy fields remain null. A separate
[post-hoc candidate recovery](2026-10-22_ncm-recipe-update-test-recovery.md)
validates both NCM decisions in the remaining packet without changing or
requesting another NCM judgment. The new parent-only schema repair and all
source/validation changes are disclosed there. Do not treat that secondary
record as a rescued primary comparison.
