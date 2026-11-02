# NCM training failure diagnosis

Date: 2026-10-22. ID: `ncm-frozen-body-diagnosis-v1`.
Exploratory, selected after the completed frozen-body quality failure.
This diagnostic cannot amend or rescue that study's 0/64 result.

Question: where do saved-target reconstruction and conditional code generation
break down in the completed checkpoints? No optimizer updates or external
judgments are authorized by this diagnostic. The parent study and its artifacts
remain unchanged.

Use the exact saved codec, hybrid/native generators, cached frozen-body features,
and selection from `data/experiments/ncm_frozen_body/20261022_train_v1`.
Verify the parent campaign SHA256 and all consumed input/source hashes; snapshot
producing code into a new root before GPU forwards. One worker on physical GPU 0,
FP32 parameters and BF16 autocast, batch one, deterministic greedy code sampling.
Stop on hash mismatch, nonfinite tensors, missing inputs or CUDA failure.

First pass focused CPU metric/decoder tests, then a separate CUDA pilot on the
first two test and two train records. Run the fixed diagnostic on all 64 test
records and the first 32 train records only after that pilot succeeds. No sampling
by output quality, checkpoint selection, further seeds, or adaptive training.
Roots: `data/experiments/ncm_frozen_body/20261022_diagnosis_pilot_v1` and
`data/experiments/ncm_frozen_body/20261022_diagnosis_v1`.

Measurements, none of which are whole-summary task accuracy:

- Oracle codec reconstruction: content-token accuracy excluding EOS/PAD,
  non-PAD accuracy including EOS, all-canvas accuracy, exact decoded text.
- Teacher-forced code cross entropy and argmax accuracy, overall and per level;
  train/test comparison. Targets remain the original saved summary snippets.
- On test cases, free-running code accuracy and code CE conditioned on the
  generated earlier codes. Compare the resulting forward argmax codes against
  the actual generation codes as a train/inference alignment diagnostic.
- Generate again with the correct first-level codes supplied as an oracle;
  compare later code accuracy and decoded outputs. This leaks target information
  deliberately and is not a deployable model or task-quality estimate.
- Decode oracle, teacher-forced, generated and oracle-first-level codes with raw
  token argmax and the exact released frequency/adjacency penalty formula and
  released XSum token-unigram allowance. This allowance is not fitted to the new
  data. Report outputs and repetition counts; do not label reduced repetition as
  validated summary quality. Also expose the released EOS/PAD rendering policy.

Retain per-case outputs, metric numerators/denominators, per-level measurements,
training-target length distribution and trace summaries. Interpretation must
separate measured defects from untested causal explanations. In particular, no
corruption-training ablation is run here, and a short warm-start adaptation is
not comparable to training the paper's models from scratch.
