# NCM decoding failure and residual-code capacity

Date: 2026-10-22. Two separate exploratory studies. Neither reaches the user's
95% complete-summary target. The final256-case test remains sealed; no speedup
has been established.

| Matched generated decoding policy | Complete summaries passing Terra |
| --- | --- |
| Local raw argmax |0/64=0%|
| Published frequency/adjacency penalties and released prior |0/64=0%|
| Published penalties and domain summary-position prior |0/64=0%|

Every policy used identical generated codes and text logits for each source.
All192 judgments were valid with no retries. Each policy's Wilson95% interval
is[0%,5.6624%]. Under the pre-judgment ordered-gates amendment, none qualified
for primary Sonnet grading. Sonnet was therefore not run for these policies;
no joint-judge accuracy is estimated. This is a diagnostic failure on the
development cohort, not a final-test result.

The candidate pairs the old native noisy-pair generator with the polished
decoder and uses FP32 parameters/FP16 forward on GPU2. This differs from older
BF16/old-decoder results and is not a controlled precision or decoder ablation.
Within this study, codes, logits and precision are matched. A CPU test compared
the new published-policy implementation with the actual released evaluator's
`pick_tokens` function. The published penalties did not repair invented words
and names from this generator. [Official NCM implementation](https://github.com/HazyResearch/ncm).

The separate oracle asks whether frozen finer codebooks have enough capacity
to compensate for a nearby wrong coarse code. For1,999 known training targets,
it substitutes one or all four coarse codes with one of the eight nearest
non-identical codebook vectors. It either replays the original finer codes or
recomputes them against the known desired target latent.

| Oracle condition | Correct non-PAD tokens /69,682 | Exact target texts /1,999 |
| --- | --- | --- |
| Clean hierarchy |69,517=99.7632%|1,858|
| One changed coarse code, old finer codes |68,355=98.0956%|988|
| One changed coarse code, recomputed finer codes |69,432=99.6412%|1,785|
| All four changed, old finer codes |64,898=93.1345%|117|
| All four changed, recomputed finer codes |69,162=99.2538%|1,576|

Recomputing finer codes restores1,077/1,162=92.6850% and4,264/4,619=92.3144%
of token losses relative to the clean baseline. Vendor quantization parity,
preservation of forced coarse roots, frozen codec identity and exact checkpoint
reload passed. Neighbors exclude identical vectors; minimum squared distance
was1.30769. Own peak allocation was862,561,792bytes on GPU2.

**This oracle sees the correct target text. It is not a summarizer, model task
accuracy, distillation success or evidence of generalization.** It supports
testing a new training objective that teaches finer codes to repair actual
coarse-code mistakes. The published training regime uses clean finer labels
with corrupted conditioning; dynamic residual relabeling would be an explicitly
new adaptation, not a reproduction claim. The original frozen Llama body and
all codec parameters remain unchanged in this oracle.

Artifacts, producer hashes, preserved raw judgments and oracle predictions:
`data/experiments/ncm_paper_regime/20261022_decoding_residual_results_v1/campaign_results.json` (SHA256
`29aa70a5948599af28a6c131be83d0748bc1a3be954dbc7272c9c7f5cbaf862c`). Reproduce with
`ncm_decode_policy_campaign`, `ncm_decode_grade_campaign` and
`run_ncm_residual_oracle` under `experiments.structure_net`, following their
immutable plans and recorded environments. Both registry records remain
tested=true, confirmed=false and underpowered=true.
