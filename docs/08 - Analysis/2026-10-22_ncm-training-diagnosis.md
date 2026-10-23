# Why the NCM summarizer fails after training

**The saved model poorly predicts the first concept level, then loses much more
accuracy when conditioning on its own earlier predictions. The supplied targets
also train snippet extraction more often than useful whole-passage summarization.**
The optimization ran; its loss is an inadequate measure of the requested task.

Date: 2026-10-22. ID: `ncm-frozen-body-diagnosis-v1`. Conformance: EXPLORATORY,
post-hoc, one reused training seed. Classification:
`exploratory_code_generation_failure_diagnosed`.
[Plan](../07%20-%20Status%20Reports/2026-10-22_ncm-training-diagnosis-plan.md),
[pilot correction](../07%20-%20Status%20Reports/2026-10-22_ncm-diagnosis-cache-amendment.md).
No weights were trained or changed and no new LLM grades were requested.

Aggregate: `data/experiments/ncm_frozen_body/20261022_diagnosis_v1/campaign_results.json`.
SHA256: `6d5d5d637eccaa4168cd787277e1bad6ef1a5fd91909b14df1c80e499eb63e6c`.
Per-case raw outputs, IDs and denominators: `records.json`, SHA256
`3609aea58886700bb417b47765b76c9bdb0dbe8740cd21a0b4c850577540d1ea`.

## Measured localization

The exact completed checkpoints were evaluated on all 64 previously selected
test cases and the first 32 selected train cases. Sources, weights, features,
original campaign and producing dependencies were hash-checked. Both arms
reproduced the original saved argmax output IDs on all 64 cases.

| Diagnostic on 64 test cases | Frozen-body hybrid | Matched native NCM |
| --- | ---: | ---: |
| Code accuracy with correct earlier target codes | 75.44% | 81.83% |
| Code accuracy during actual free generation | 24.37% | 27.37% |
| First-level target codes correct | 3/256 = 1.17% | 3/256 = 1.17% |
| Later-code accuracy during free generation | 24.84% | 27.91% |
| Later-code accuracy with oracle first level | 56.54% | 66.41% |
| CE with correct earlier codes | 2.117 | 1.278 |
| CE conditioned on generated earlier codes | 8.671 | 9.311 |

These are latent-code diagnostics, **not summary task accuracy**. Oracle
conditioning intentionally supplies answer information unavailable in deployment.
Exact code recovery is not required for every valid paraphrase. Nevertheless,
the gap and the oracle-first-level intervention localize a substantial failure
to early concept prediction and subsequent conditioning. Even supplying the
correct first level does not solve later generation.

The first level contains four of 200 code positions: only 2% of the equally
weighted CE objective. On test cases, fine-level teacher-forced accuracy reaches
88.83% hybrid / 91.28% native while the first level remains at 1.17%. A falling
average loss can therefore hide failure to initiate the hierarchy. A causal
loss-weighting ablation has not been run; upweighting the coarse level is a
candidate, not an established fix.

The 32 train-case teacher-forced CEs are 1.351 hybrid and 0.820 native, versus
2.117 and 1.278 on test. First-level train code recovery is also poor: 0/128 and
4/128. This is not a successful memorization check or evidence that simply
adding more updates will reach 95% task accuracy.

Recomputing forward logits conditioned on generated codes agrees with actual
generation codes at 99.68% of positions for the hybrid and 99.52% for native.
This is high agreement, not bitwise equivalence; the differently shaped BF16
forward paths may differ numerically. It does not account for the much larger
teacher-forcing/free-generation gap. No precision ablation was performed.

## Targets and the actual objective

In the fixed training selection, median target length is 17 GPT-2 tokens;
426/2,048 targets have at most eight tokens and 1,010 have at most 16. Short
targets are not inherently wrong, but actual selected examples include:

- `fc4c2bdb22005fced165afee7de4abcbb6bebdb168bd4a5026c815664887b736`:
  a source explaining a complete Python XYZ-file reader, target `import numpy as np`.
- `dc7c0650f487b1f1d69048e90949a633a49891fe9cb79572d996570d55b2f859`:
  a source giving scheduling advice, target `deposit the check`.

The earlier, separate [256-case target audit](2026-10-22_ncm-memory-task-accuracy.md)
found 238/256 targets source-supported, but only 58/256 passed the complete-summary
rubric. This is a task mismatch rather than evidence that most labels invent
facts. That audit is LLM judged and is not an independently human-labeled
estimate of all training targets.

The head objective reproduces saved target codes; it does not distill the full
parent's good summary outputs or its neighborhood behavior. The last adaptation
used 512 updates of batch eight, 4,096 example presentations from 2,048 selected
pairs. The generator was warm-started, with a newly initialized adapter for the
hybrid. It was not a randomly initialized full model or a full reproduction of
the paper's training budget.

## Codec and released decoder behavior

Even with all correct target codes, raw argmax reconstructs only 965/1,030
content tokens = **93.69%**, or 1,029/1,094 = 94.06% when EOS is included and PAD
excluded. Exact decoded target text is recovered on 28/64 cases. Exact text
recovery is not the application success rule and is not a hard upper bound on
semantic summary quality. It still demonstrates actual codec corruption:
one target's `literacy` becomes `promoting`; another target's `utilize` becomes
`country`. The earlier 95.46% non-PAD reconstruction result came from a different
256-case cohort; neither figure is 95% whole-summary accuracy.

The original inference path omitted the released XSum decoder's frequency and
adjacency penalties. Applying that formula and its unchanged XSum token-unigram
allowance to the same oracle-code logits **reduces** content-token recovery to
866/1,030 = 84.08%, with only 3/64 exact decoded targets. It reduces adjacent
repeated generated token IDs from 53 to nine for hybrid and 58 to 21 for native
under the released rendering policy. Reduced repetition is not established
summary quality, and applying these penalties without domain validation is not
a demonstrated repair.

The released renderer ignores PAD entries before EOS; the original local
renderer stops at the first EOS **or PAD**. Both rendered strings and raw IDs are
saved. The aggregate's immediate-repetition fields use the local stopping rule,
so they can be shortened by an inserted PAD and must not be read as the released
renderer metric. `static_audit.json` explicitly recomputes the released-policy
counts above and verifies that packed targets exactly equal re-encoded selected
target text on 32 train and 64 test cases. All underlying IDs are in `records.json`.
The static-audit receipt SHA256 is
`c960ad22ae179f1c7a135b9c26fba18a648770b734917217d4d807f894d48741`.

## Recipe omissions and next experiment

The [NCM paper, section 3.4 and appendix F.3](https://arxiv.org/html/2610.14162v1#S3.SS4)
trains with corrupted conditioning codes, retaining clean target codes, to
increase tolerance to generation errors. Our adaptation used clean conditioning
only. The observed gap is consistent with that concern, but this diagnosis
contains no corruption-training ablation and does not prove noise alone fixes
it. Corruption cannot substitute for learning the first level from the source.

The paper also reports 99.79% token reconstruction on XSum; our different-domain
codec and task estimates are not directly comparable benchmark scores.

The next useful training experiment should keep the body frozen, repair or
filter summary targets for the intended task, validate meaningful codec
reconstruction, and test coarse-level learning and conditioning corruption.
First require a small training subset to produce faithful, readable summaries
without oracle codes; then evaluate on a separate development cohort. Keep the
already inspected test cohort for diagnosis, not checkpoint selection. Validate
decoder changes against semantic quality rather than repetition alone.

This setup still covers only <=512 source tokens and <=47 GPT-2 target tokens
plus EOS. It does not establish quality over the application's 2,048-token
fragments, 6,300-token batches, or full 128-token summary allowance.

## Integrity, limitations and reproduction

Sixteen focused CPU tests passed. The first diagnostic pilot was invalidated
because its broad autocast scope retained cached parameter casts across loading
different checkpoints. Its artifacts and invalidity receipt remain intact.
The corrected pilot explicitly clears that cache after each checkpoint load,
reproduced all saved outputs, and preceded the full diagnostic. The original
training/evaluation used separate autocast scopes and is not affected by this
diagnostic bug. There was one full diagnostic run, no new training, no seed or
checkpoint selection, and no new task grading.

The diagnostic GPU stage took 111.6 seconds and peaked at 1,018,953,216 allocated
bytes on physical GPU 0, an RTX 3060. This is diagnostic execution time, not a
throughput benchmark. One worker, FP32 parameters, BF16 autocast, greedy code
generation, unchanged checkpoints and cached body features were used.

The [parent quality result](2026-10-22_ncm-frozen-body.md) remains **0/64 hybrid,
0/64 native and 59/64 full Llama**, under the existing LLM judge. These findings
do not establish that a frozen transformer body is intrinsically unsuitable:
the standalone control also fails. No task score has been revised here.

From the repository root, use the moba environment Python with
`LD_LIBRARY_PATH='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=src:.` and set
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0` before GPU execution:

```text
python -m experiments.structure_net.diagnose_ncm_training pilot --output NEW_PILOT_ROOT
python -m experiments.structure_net.diagnose_ncm_training run --pilot NEW_PILOT_ROOT --output NEW_DIAGNOSTIC_ROOT
python -m experiments.neural_architecture_lab.finalize_ncm_training_diagnosis seal --root NEW_DIAGNOSTIC_ROOT
python -m experiments.neural_architecture_lab.finalize_ncm_training_diagnosis register --root NEW_DIAGNOSTIC_ROOT --report NEW_REPORT
```

The canonical JSON registry and semantic record must retain this diagnosis's
exploratory classification and `confirmed: false`; it is not a successful
summary-quality or efficiency claim.
