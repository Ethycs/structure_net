# TinyLLM MoBA: attention-only final accuracy optimization

Date: 2026-09-09. Hypothesis: `tinyllm-moba-attention-accuracy95-v1`.
Profile: EXPLORATORY_TASK_SUPERVISED, with sealed final evaluation.

This is the execution protocol following the
[attention-only scope correction](2026-09-09_tinyllm-moba-attention-only-amendment.md).
The prior all-weight search reached 98% validation at isolated checkpoints,
missed its consecutive-check selection rule for candidate one, and was stopped
during candidate two. It provides no eligible initialization or final evidence.

## Fixed architecture and training scope

Start every stream from the original native-256 d8 pretrained checkpoint:
`data/experiments/tinyllm_babylm_pretrain/20260812_d8_seed7/checkpoint_step12000.pt`,
SHA-256 `5a7491b4231c30feaaabda7babf0a831dd4d72fbb4e7f3e757114cadf098df09`.
Freeze all parameters, then train only each attention module's `c_attn` (QKV)
and `c_proj` (attention output), including their biases. MLPs, embeddings,
normalization and the output head must remain tensor-exact. Check frozen-state
hashes at every validation checkpoint and reject any frozen parameter gradient.
Save trainable parameter names/counts and before/after frozen hashes.

MoBA uses native length 256, block size 32, k=2 including the local block.
Use the unchanged dense-mask reference helper from the original completed
campaign. This accuracy study establishes no kernel or end-to-end speedup.

## Data, objective, selection, and final gate

Retain the exact four official hashed BLiMP files, tokenizer and BOS from the
previous protocol. Per suite, historical IDs 0–31, train 32–799, validation
800–899, and fresh test 900–999. Exclude complete-sentence overlap between
partitions as in the existing deterministic splitter. The resulting counts are
128 historical, 3072 train, 400 validation and 400 fresh pairs. No final test
has been scored in either search. Previous validation findings are known, so
this remains exploratory supervised task adaptation, not zero-shot evaluation.

Use summed sentence log likelihood, no length normalization or truncation,
and half credit for ties. Per update: eight grammatical pairs with
mean softplus(1 − [log p(good) − log p(bad)]) plus 0.1 good-sentence token-mean
LM loss; then one 256-token natural-text replay sequence with LM and frozen
full-teacher logit KL, accumulating before one optimizer update.

Natural-text replay uses the BabyLM prefix excluding the final 262144 tokens.
Validation uses 16 nonoverlapping 257-token bins in the first half of that
suffix; final evaluation uses 32 bins in the second half excluding all parent
final windows. Schedules are saved before training. Pair/replay/validation/test
RNG salts are 95117/95217/95317/95417, paired by seed.

AdamW: betas (.9,.95), epsilon 1e-8, weight decay .01, gradient clipping 1.
FP32, deterministic CUDA, no AMP/TF32. The optimizer receives only the explicit
attention parameter list. Fixed candidate order:

| Candidate | LR | Replay LM | Replay KL | Max updates |
| --- | ---: | ---: | ---: | ---: |
| attention_pair_replay | 5e-5 | .25 | .5 | 2400 |
| attention_pair_replay_slow | 2e-5 | .25 | .5 | 3200 |
| attention_pair_replay_strong | 5e-5 | .5 | 1.0 | 2400 |

Search on seed 7, checking all 400 validation pairs and text NLL every 100
updates including zero. Select the first candidate whose validation accuracy
is >=98% and NLL <=1.05 times the matched full teacher at two consecutive
scheduled checks. Freeze that candidate and the second check's update count.
Do not execute later candidates after selection. If none qualifies, preserve
all evidence and amend before extending the search.

Then train exactly the selected budget on streams 7,17,29,41,53, reusing seed 7's
exact selected checkpoint, and evaluate both final grammar sets and fresh text
once per seed. Success requires >=95% accuracy on both historical and fresh
sets plus NLL <=1.05 times teacher for every seed. No test-dependent checkpoint
extension. New optimization after test exposure requires an untouched test
partition or explicit treatment of the old test as tuning data.

## Execution contracts

Entry point: `experiments/structure_net/tinyllm_moba_attention_accuracy95.py`.
New root: `data/experiments/tinyllm_moba_attention_accuracy95/20260909_search_v1/`.
Run CPU objective/split/frozen-gradient contracts and a separate four-update
real-model CUDA lifecycle/memory pilot before search. One sequential worker on
physical GPU 0; preserve other GPU workloads. No CPU training fallback.
Hash and copy producing sources/protocol, inputs, checkpoints and schedules;
strict JSON and append-only completed evidence. Report scope violations,
stops, missing cells, all gates and limitations explicitly.

This tests attention-only supervised agreement adaptation on one pretrained
TinyLLM and four agreement suites. It does not establish broad BLiMP accuracy,
95% population accuracy with confidence, long-context routing quality, a Qwen
result, or acceleration. Qwen work requires a separately tested efficient
kernel, frozen-backbone checks and matched optimized-attention speed baselines.
