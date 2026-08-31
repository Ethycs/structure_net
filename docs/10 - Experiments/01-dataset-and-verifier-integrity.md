# Experiment 01 — Dataset and Verifier Integrity

**Verdict:** PASS  
**Runtime:** canonical NAL CPU experiment  
**Date:** 2026-08-16

All ten contract gates passed. The official 49,401/8,000/8,000 PAWS splits
match their registered hashes, have no cross-split normalized sentence or pair
reuse, and fit both local model contexts without truncation.

| Interface | Maximum prompt length | Context | Result |
| --- | ---: | ---: | --- |
| TinyLLM BPE | 251 | 256 | pass |
| SmolLM tokenizer | 214 | 8,192 | pass |
| Qwen serving tokenizer | server-side | 8,192 | longest prompt accepted |

Normalized unordered grouping found 30 conflicting groups (63 rows) in train,
none in development, and 7 groups (14 rows) in test. Raw evidence is preserved,
but these rows are marked `eligible=false` in immutable manifests. They cannot
enter training, calibration, or headline evaluation.

Focused tests passed `4/4`. The producing implementation is
`experiments/structure_net/paws_dataset_contract.py`; the strict contract and
NAL result are under
`data/experiments/paws_abc_routing/2026-08-16_experiment_01/`.

Experiment 02 is licensed: measure the three frozen, undistilled model
baselines on an explicitly selected development subset. No test outcome may be
read during baseline or model selection.
