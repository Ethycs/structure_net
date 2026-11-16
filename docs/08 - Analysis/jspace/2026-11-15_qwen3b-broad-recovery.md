# Qwen 3B broad recovery: development improvement, failed independent audit and generation

Study `jspace-qwen3b-broad-recovery-v1` is completed and **not confirmed**.
The development-selected fourteen-block checkpoint improved answer-choice
accuracy, but missed the independent five-percentage-point accuracy screen and
failed all six ordinary-generation canaries. It is not a recovered usable
general-purpose model. Only14 of36 blocks are compressed.

Evidence:

- Recovery: `data/experiments/jspace_3b_broad_recovery/20261115_v2`.
- Assessment: `data/experiments/jspace_3b_broad_assess/20261115_v1`.
- Assessment result SHA256: `945bafce2c2e77c88789dfd16b1f2a7ce7c8f0b7d66c7668b567053e58e4d046`.
- Appended semantic review: `semantic_review_v1/result.json`, SHA256
  `cb44d7d2d4132cf4302108ccf10d936ef2cd844976acabf60c457644fbf1bc75`.

## Matched training and selection

Both arms used the same initialization, 4,608 training examples, order, one
pass of288 updates, and constant learning rate5e-6. Checkpoints0/72/144/216/288
were retained. Selection used256 development examples and the prespecified
overall/family accuracy limits; earlier checkpoints won accuracy ties.
The parent-feature auxiliary objective was shared.

| Model/checkpoint | Development correct | Accuracy |
| --- | ---: | ---: |
| Original native |195/256|76.17%|
| Starting compressed14 |173/256|67.58%|
| Parent-legacy eligible candidate, update288 |183/256|71.48%|
| Native-content selected candidate, update72 |189/256|73.83%|

The selected checkpoint is
`native_content/train/update_072.pt`, SHA256
`d4780204782ba26304456f5c27d4455de05dbdf92651a3d66b86b95e322f70a2`.
It gained6.25 points on development and remained2.34 points below native.
The parent-legacy diagnostic best at update144 also scored183, but failed the
family gate and was not substituted for the eligible update288 candidate.

## Independent audit

The128 audit cases were opened only after global selection was frozen.

| Family | Native | Selected compressed14 |
| --- | ---: | ---: |
| ARC |27/32|28/32|
| BoolQ |28/32|22/32|
| HellaSwag |30/32|28/32|
| WinoGrande |18/32|15/32|
| Overall |103/128 (80.47%)|93/128 (72.66%)|

The7.81-point overall loss fails the five-point screen. BoolQ also fails the
ten-point family screen. These are descriptive sample gates, not population
noninferiority proofs. This audit is now exposed; it cannot be called a fresh
sealed test for subsequent experiments. The separate final512 remains sealed.

## Ordinary generation

The native model, starting compressed14 and selected compressed14 received the
same six frozen chat prompts and64-token greedy generation budget, without a
KV cache. Semantic review inspected requirements and raw responses; it did not
exact-match summary wording or use another external judge.

Both compressed checkpoints produced repetitive labels or incoherent text on
all six prompts and fulfilled none. The problem therefore predates this
recovery run. Native produced no comparable collapse. Its strict fulfillment
was only1/6: two outputs were truncated by the64-token cap, and three missed
formatting/detail instructions. Those native failures remain in the record.
These six checks diagnose gross failure, not overall model accuracy.

The sparse target audit found622,447 TRAIN input tokens but18,432 supervised
positions (2.96%): three interior question/content positions and one answer
letter per example. All prompts request a multiple-choice letter. There are
no multi-token assistant continuation or EOS training targets. This is a
concrete mismatch with preserving ordinary generation, but it does not by
itself prove the causal contribution of training versus compression geometry.
Full-vocabulary generation-path parity remains a separate diagnostic.

## Integrity, limitations and learning

Two execution issues were separately diagnosed without changing the scientific
training recipe: restoring the active CUDA device before Triton execution, and
reconstructing nonpersistent rotary buffers on their original GPU rather than
on CPU. Failed receipts were retained. Fresh verification reproduced selected
development records and the complete fixed-state fingerprint exactly.

Those controls establish reproducibility, not broad capability. Earlier
statements that recovery had worked were too broad. Generation should have
been a cheap early gate, and the target distribution should have been audited
before extending compression. Numerical bookkeeping must not displace tests
of the user's intended behavior.

No inference throughput gain, full36 recovery, or deployment readiness is
established by this study. New progressive15 and independent all36 preparations
are exploratory; neither can repair this study's failed result retrospectively.
