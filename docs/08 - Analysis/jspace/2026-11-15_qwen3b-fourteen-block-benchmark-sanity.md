# Fourteen-block Qwen fails the independent benchmark sanity screen

Status: completed exploratory comparison; `confirmed=false`. The selected
Qwen2.5-3B-Instruct model with 14 of 36 blocks compressed scored **88/128
(68.75%)**, versus the original model's **101/128 (78.90625%)**. The observed
loss is **10.15625 percentage points**, exceeding the prospectively specified
five-point descriptive screen. This does not pass the current quality budget.

Study: `jspace-qwen3b-fourteen-block-sanity-v1`.
Protocol: [prospective sanity design](../../07%20-%20Status%20Reports/jspace/2026-11-15_qwen3b-fourteen-block-benchmark-sanity.md).
Evidence: [paired_result.json](../../../data/experiments/jspace_3b_sanity_evaluate/20261115_v1/paired_result.json),
SHA256 `4df96ae0d0f72d5197a8fab8204129e2f58d71ba7b8e9a06d8b49a1b1208b849`.

| Benchmark | Original /32 | Compressed /32 | Accuracy change |
|---|---:|---:|---:|
| ARC-Challenge |25|25|0.00 pp|
| BoolQ |30|27|−9.375 pp|
| HellaSwag |24|25|+3.125 pp|
| WinoGrande |22|11|−34.375 pp|
| **Total /128** |**101**|**88**|**−10.15625 pp**|

The compressed model loses 22 answers that the original gets right and gains 9
that the original gets wrong. The two-sided exact paired sign/McNemar test is
`p=0.029449373483657837`. Prediction agreement is 73.4375%. WinoGrande accounts
for 11 of the 13 net lost correct answers. Its family result is exploratory;
family p-values are unadjusted and are not separate confirmatory claims.

An independent read-only audit reproduced all paired counts, predictions,
exact-test values and result hashes. WinoGrande's sample has 8 A answers and
24 B answers. The original predicts A 12 times; the compressed model predicts
A 25 times. Correct B answers fall from 17/24 to 5/24, while correct A answers
rise from 5/8 to 6/8. The sample's label imbalance amplifies this option bias;
do not generalize its 34.375-point family loss to the complete benchmark.
The fixed sample and original aggregate remain unchanged. Across the other
three families together, the count changes from 79/96 to 77/96.

The median WinoGrande top-choice logit margin falls from 13.875 to 1.0.
The native margins for regressed WinoGrande cases range from 5.25 to 24.5,
so these were not native near-ties. Logit margins depend on scale and are
diagnostics, not calibrated confidence or a proven mechanism.

The earlier 50/64 (78.125%) development result was reproduced exactly from the
saved epoch 1 weights. It remains a valid development measurement, but it did
not establish generalization. That 64-case set has been reused for checkpoint
selection, and it uses batch 4/full-vocabulary scoring rather than this matched
singleton choice-row protocol. The new result is consistent with the concern
that a small, repeatedly selected development set overstates retained quality;
it does not isolate overfitting as the sole cause. This experiment also does
not isolate which compressed block or recovery loss caused the regression.

The new 128 cases were fixed before scoring, 32 per task family, seed 582463.
They are disjoint from 11,142 conservatively collected historical identity
candidates, including all known training, development and prior evaluation
examples and the reserved 512 final examples. The final questions and labels
were not opened; their IDs came from an authenticated identity-only manifest.
Source evaluation splits, the pinned Qwen tokenizer/prompt and the 384-token
input cap were preserved. Actual maximum input length is 302 tokens. These
are restricted multiple-choice samples, not the full official leaderboard
protocol or a general generation-quality test.

Both arms use BF16, eager attention, TF32 disabled, unpadded singleton prompts
and the same answer-token rows from the original tied embedding/readout.
The native arm captures its pre-layer 22 residual while scoring raw prompts.
The compressed arm then executes the strictly restored and packed blocks 22–35
plus their trained final norm on those lossless residuals. This is the native
prefix of the hybrid model, not an answer or a final teacher representation.
A pre-existing development boundary control matched bit-for-bit with maximum
absolute difference 0. The compact checkpoint and its trained-norm provenance
are authenticated. All 128 paired outputs are finite and aligned in both arms.

The descriptive screen permits at most 6 net lost answers on 128; 13 were lost.
The sample is too small to prove a population five-point noninferiority bound,
and this failure is an observed screen result rather than a claim that the
population drop is known to exceed five points. There is nevertheless a clear
task-specific failure worth fixing before extending compression. This set
becomes diagnostic evidence once its results inform subsequent changes; keep
the final 512 sealed and use separate training examples for recovery.

No weights changed, no checkpoint was reselected, no questions were resampled,
and no failing examples were excluded after scoring. The next recovery design
should examine the WinoGrande regression and use a broader development signal;
the previous 50/64 count alone should not admit further compression. The planned
short early-checkpoint recovery and automatic depth expansion have not started.

Provenance:

- Native revision: `Qwen/Qwen2.5-3B-Instruct`,
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Accepted compressed checkpoint:
  `814563a4de04d91c17bb02d670ed2d5a7609d3bb3460ed92073d721deac7e401`.
- Evaluation fingerprint:
  `31ef3488ca0df751365c7ec01d95bf2b9ac7b6fed0e15c68a87ba31518dae363`.
- Data fingerprint:
  `be51dd6e74951a62169218c77d3533a73ee1f6656545de93b67ab8c632e998bc`.
- Protocol/data/evaluator commits: `4868452`, `084a31c`, `b9dbe4e`.
- Six data tests and 27 evaluator tests passed; the latter include real tiny
  BF16 Qwen boundary, trained norm, tied readout and reservation checks.
- Native reservation: `41effecf22f9`; compact reservation: `d24cc6ebaeb6`.
  Both ran on the 3060 under the experiment controller and were released.
- Completion manifest SHA256:
  `97405d4507e62ae9e37cfff92a38fd2edcd9c925c2cbdcbdc1f8a6a450e6ea3e`.

This cached-prefix decomposition measures quality. Its phase durations and GPU
memory do not measure deployed whole-model speed or memory: the compact phase
does not recompute the prefix. No throughput or full 36-block success is claimed.
