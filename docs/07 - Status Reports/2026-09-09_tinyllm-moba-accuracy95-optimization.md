# TinyLLM MoBA: final downstream accuracy toward 95%

Date: 2026-09-09. Study ID: `tinyllm-moba-accuracy95-v1`.
Profile: EXPLORATORY task optimization, with sealed final evaluation.

The user clarified that “get to 95” refers to final accuracy. This means the
BLiMP grammatical-acceptability accuracy column in the previous report. The
earlier routing-recall optimization draft is withdrawn; no routing candidate
was implemented or run. This document defines the active target.

## Goal and scope

Reach accuracy >=0.95 on both the original 128 BLiMP pairs and a fresh 400-pair
test set, for each of the five adaptation seeds (7,17,29,41,53). Keep the same
four agreement suites, tokenizer/BOS, summed sentence log-likelihood scoring,
tie score 0.5, native-256 d8 model, 32-token blocks and k=2 MoBA attention.
Final natural-text NLL should stay <=1.05 times the matched original full
teacher NLL; this is an additional quality guard and will be reported even if
it prevents declaring full success. No teacher/oracle is used at inference.

This is supervised adaptation to a grammatical-acceptability task. It is not
a zero-shot BLiMP comparison or a pure full-attention-to-MoBA conversion study.
The original teacher scores 75.78%; the parent KD students average 77.97%.
The task requires learning behavior beyond that teacher's existing decisions.
These four suites are not the complete 67-suite BLiMP benchmark. All original
128 examples are short enough that k=2 does not prune attention; accuracy here
does not establish long-context routing quality or sparse-kernel acceleration.

The completed original campaign remains unchanged and negative. Its aggregate
SHA-256 is `abcb16c37bc37ffc62afe8f1060075944f86225ae9af38fc82acd8c47f86f6d6` at
`data/experiments/tinyllm_moba_distillation/20260909_registered/campaign_results.json`.

## Data and protection against leakage

Use the same four official hashed BLiMP JSONL files and tokenizer as the parent.
Per suite, pair IDs 0–31 are the historical test, 32–799 training, 800–899
validation, and 900–999 fresh test. Compare strip/casefold-normalized complete
good and bad sentences. Exclude any validation pair sharing either sentence
with either test set; exclude any training pair sharing either sentence with
validation or either test set. This deterministic check currently retains
3072/400/400/128 pairs, with no overlap. Record exact pair IDs and data hashes.
Do not evaluate either test set during candidate selection.

Natural-text replay uses the BabyLM prefix excluding the final 262144 tokens.
NLL monitoring uses 16 nonoverlapping 257-token bins in the first half of the
suffix. Fresh final text windows use 32 bins in the second half, excluding
every bin used by the parent campaign's final evaluation. Save schedules before
training. RNG salts: 95117 for pair batches, 95217 for replay starts, 95317 for
text validation, and 95417 for fresh text tests. Arms are paired by seed.
Prior parent results are known, and the original 128-pair score is historical;
the fresh 400-pair test must remain unobserved during search.

## Optimization

Start each seed from its matching parent's KD student, verifying the parent's
result/checkpoint hash. The original full-attention pretrained teacher remains
frozen for natural-text replay distillation. All student parameters are trainable.

One update consumes eight grammatical pairs and one 256-token natural-text
sequence. For each pair, compute d = summed log p(good) - summed log p(bad).
The supervised objective is mean softplus(1-d), encouraging a one-nat margin.
Add 0.1 times token-mean LM loss over good sentences. Backpropagate this batch,
then accumulate natural replay LM and KL(teacher || student) gradients before
one optimizer update. Padding is ignored in scores and LM loss. No truncation,
length normalization, test-set training, or inference-time auxiliary head.

AdamW, betas (0.9,0.95), epsilon 1e-8, weight decay .01, gradient clipping 1,
FP32, deterministic algorithms, no AMP/TF32. Fixed candidate order:

| Candidate | LR | Replay LM weight | Replay logit KL weight | Max updates |
| --- | ---: | ---: | ---: | ---: |
| pair_replay | 2e-5 | .25 | .5 | 1600 |
| pair_replay_slow | 1e-5 | .25 | .5 | 3200 |
| pair_replay_strong | 2e-5 | .5 | 1.0 | 1600 |

Search on seed 7. Evaluate validation pairs and text NLL every 100 updates,
including update 0, preserving every evaluated checkpoint. The selection gate
is validation accuracy >=.98 and validation text NLL <=1.05 times teacher NLL
at two consecutive scheduled checkpoints. Select the first passing candidate
and the second passing checkpoint's update count; unneeded candidates do not
run. If none passes, preserve results and write a new dated search amendment
before extending the method. A systems-only shakedown cannot select a candidate.

After selection, freeze candidate and update count, train that configuration
from the matching parent KD checkpoints for all five seeds, and evaluate both
held-out grammatical sets and fresh natural-text windows once. Seed 7's exact
selected checkpoint may be reused with verified identity. No test-dependent
checkpoint extension or threshold change. Every failure remains reported; a
further optimization round requires new artifacts and a new untouched final
test partition or an explicit disclosure that the old test is now tuning data.

## Execution and evidence

Entry point: `experiments/structure_net/tinyllm_moba_accuracy95.py`.
Initial root: `data/experiments/tinyllm_moba_accuracy95/20260909_search_v1/`.
CPU split/scoring/gradient/checkpoint tests and a four-update real CUDA
lifecycle/memory pilot precede search. One worker, currently physical GPU 0;
check capacity before launch. Preserve producer sources and protocol, hash all
outcome-relevant dependencies, parent checkpoints, tokenizer/data and exact
configurations. Keep strict JSON, per-checkpoint metrics, terminal status,
optimizer steps, data volume, completion counts, failures and source identity.
Completed evidence is append-only; resume requires exact fingerprint/artifact
matches. CUDA training only. The result report must distinguish task-supervised
accuracy improvements from the original zero-shot conversion findings.
