# NCM generator depth: 12 versus 24 blocks

Prospective exploratory experiment, 2026-10-26. Requested by the user: try more
NCM layers. This is a capacity comparison, not a paper replication or a speed
claim. The preceding BROAD and both on-policy candidates scored 0/128 semantic
DEV passes. That result is the motivation, not evidence that depth will help.

## Intervention and control

Both arms start from the completed BROAD checkpoint (SHA256
74cdce0eb14bb99c801b6ec0a025c092f95249663becc05561af8dfbad6040ff).
The control retains 12 generator transformer blocks. The intervention inserts
one independently copied block after each existing block, producing 24 blocks.
Both residual output projections and their biases in inserted blocks start at
zero. Original blocks, source-attention hooks, codec and word readout retain
their tensors and behavior. Inserted blocks have no copied source hooks. The
12 quantization levels and 641 code positions do not change.

The codec, word-source attention and original 28-layer Llama body remain frozen.
Complete FP32 source features stream from the existing reviewed corpus. This
experiment uses all 1,376 reviewed TRAIN pairs; it does not select a subset of
the public extraction that is still running.

## Fixed training budget

Each arm receives 1,024 optimizer updates of four examples, microbatch two.
Shuffle without replacement through all 1,376 rows, repeated as necessary.
Identical sampler/corruption seeds and sample order in both arms. Two examples
per update use clean oracle histories and two use the existing level-dependent
0.25-to-0.05 corruption. Every example and code level has equal objective weight.
FP32 weights, FP16 training autocast, FP32 greedy generation; CPU FP32 AdamW
(0.9, 0.95), epsilon 1e-8, decay 0.01, gradient norm clip 1. Learning rate warms
over 64 updates to 1e-4 and cosine decays to 1e-5. Seeds: sampler 241142,
corruption 242142, dropout 243142. Same dropout seed does not mean identical
dropout masks across architectures. Strict deterministic CUDA algorithms;
TF32 off. RTX 2060 physical GPU 1; peak allocation must remain below
5,200,000,000 bytes. GPU 0's public feature extraction is independent.

## Gates and measurements

Before full training, each arm runs two real updates on four TRAIN examples
including the longest source. The 24-block arm must preserve initial FP32
teacher logits and source-only generated codes/words exactly. Added projections
and learned root queries must receive finite, nonzero gradients. Frozen modules
must retain their hashes. Pilot save/reload/replay must reproduce the second
update, optimizer/scaler/RNG and source-only outputs exactly. Preserve failed
attempts; no automatic reduced batch or precision fallback.

After both complete training runs, generate all 128 existing source-disjoint DEV
summaries and grade using the unchanged whole-summary semantic rubric: readable,
faithful, salient, within the stored 128-token cap; paraphrases are accepted.
Qualified Terra is the first judge. Preserve all 4,179 prior first judgments and
then the control arm's first judgments before evaluating the deeper arm. The
existing Sol confirmation threshold (61/64 in each cohort) remains unchanged.
The 320 final examples stay sealed. Report numerator and denominator for task
accuracy, per-cohort counts, parameter count, memory and training time. Training
code agreement and word agreement are diagnostics, never task accuracy. More
DEV passes than the matched control supports further depth investigation;
95% remains the actual application goal and is not established by this test.

Artifact roots: `20261026_ncm_depth_{12,24}_{pilot,train}_v1` under
`data/experiments/ncm_paper_regime`. Preserve sources, input hashes, sample/noise
traces, full resumable checkpoints and actual tool exit receipts. No adaptive
early stopping or extra updates based on DEV. If either lifecycle gate fails,
stop that arm and document the failure before a separately identified recovery.
