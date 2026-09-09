# TinyLLM MoBA conversion by teacher distillation

Date: 2026-09-09. Hypothesis: `tinyllm-moba-distillation-v1`.
Profile: PREREGISTERED, conditional on one pretrained checkpoint.
Written before conversion outcomes. This is a new study, independent of the
closed native-2048 competence failure.

## Question and prediction

At a fixed small adaptation budget, does block-routing and behavioral
distillation preserve the native-256 teacher better than LM-only adaptation?
The primary comparison is combined attention-mass distillation versus LM-only.
Projected contribution is a prespecified secondary target, not an alternative
way to pass the primary hypothesis.

## Fixed model, data, and controls

Teacher: frozen d8 TinyLLM (8 layers, 8 heads, width 512, vocabulary 50257,
native context 256), checkpoint
`data/experiments/tinyllm_babylm_pretrain/20260812_d8_seed7/checkpoint_step12000.pt`,
SHA-256 `5a7491b4231c30feaaabda7babf0a831dd4d72fbb4e7f3e757114cadf098df09`.
All students start with exactly these weights. No model pretraining is repeated.
Adaptation seeds: 7, 17, 29, 41, 53; these independently resample training
minibatches, not pretrained initialization. All arms within each seed use the
same saved input schedule. The teacher is evaluated on each seed's inputs.

Token stream: `data/corpora/babylm_10M_bpe16k.tokens.npy`, SHA-256
`f339655453b970ae6cc1cbdc7c78f8a5234c42437bc7bba8fb31a1dee5c9d765`.
Adaptation uses the prefix excluding the final 262144 tokens. Monitoring uses
the first half of that suffix; final evaluation uses the second half. Each
seed has 4 monitoring and 16 final windows, each with 257 tokens. Windows are
selected without replacement from nonoverlapping bins, with separate RNG salts
1103 (training), 2203 (monitoring), 3301 (test). Evaluation windows may coincide
across seeds; they are paired input sets, not independent corpus samples.
The parent used this suffix for validation; it is held out from optimization,
not historically unseen model-selection data. No outcome-based checkpoint or
hyperparameter selection is permitted here.

| Arm | Objective during stages 1–3, in addition to LM |
| --- | --- |
| F | Frozen full-attention teacher; no optimization |
| M0 | Frozen zero-shot conversion; no optimization |
| LM | None |
| KD | Logit KL |
| route_mass | Attention-mass routing KL |
| route_contribution | Projected-value-contribution routing KL |
| combined_mass | Mass routing KL + logit KL + hidden MSE; attention MSE in stage 2 |
| combined_contribution | Contribution routing KL + logit KL + hidden MSE; attention MSE in stage 2 |

The staged unfreezing, sparsity schedule, LM anchoring, data, update count,
optimizer and learning rate are matched across all six trained arms. The
combined arm tests a package; route-only versus LM isolates the routing loss.

## Attention and training protocol

Use an explicit dense-mask reference implementation of causal MoBA semantics.
Blocks contain 32 tokens, giving 8 blocks at length 256. Top-k includes the
mandatory local block, so target k=2 selects at most one completed past block.
Only completed past blocks compete in routing. Future and local block means
are excluded from the soft routing objective, preventing future-key leakage
and a trivial local-block recall advantage. First-block queries have no router
target and contribute zero routing loss. Top-k recall excludes the forced
local block and uses min(k-1, number of past blocks) as denominator.

Router logits are raw Q dot mean(K), temperature 1. Teacher block mass is the
sum of full attention probabilities within each completed past block,
renormalized over eligible past blocks. Contribution targets are the L2 norm
of each head's block sum of A*V after that head's slice of W_O, normalized over
eligible past blocks; zero-total contributions fall back to mass. Projection
bias is excluded. This norm is a contribution proxy, not an ablation-based
causal effect. No straight-through estimator is applied to hard top-k.

128 AdamW updates, one sequence/update (32768 target tokens per trained arm).
Learning rate 3e-5, betas (0.9, 0.95), epsilon 1e-8, weight decay 0.01, global
gradient clipping 1.0; FP32 parameters/operations, no AMP or TF32. Four equal
32-update stages, each with a fresh optimizer:

1. Q/K only, k=8. The fused V rows receive neither gradients nor weight decay;
   the whole fused QKV tensor has zero decay in this stage.
2. Q/K/V/O only, k=4.
3. All parameters, k=2.
4. All parameters, k=2, LM-only for every trained arm (teacher-free finish).

LM and logit losses are token means; KD is KL(teacher || student), temperature
1. Routing KL is averaged over eligible query/head pairs, then layers. Hidden
MSE averages the post-block residual streams over elements then layers;
attention MSE averages projected attention outputs before residual addition.
Weights: LM=1, logits=1, route=1, hidden=0.1, attention=0.1.
Monitor at updates 0,32,64,96,128 at target k=2. Save final model, complete
configuration, schedules, stage losses, and monitoring curve. Evaluate final
test at k=1,2,4,8 for a conversion curve; k=2 is primary. No adaptive k stopping
or quality-based early stopping; the entire small fixed budget is the test.

## Endpoints and decision order

On 4096 held-out target tokens per seed, measure token-mean NLL/perplexity,
next-token top-1 accuracy, teacher/student token-mean logit KL, projected
attention-output MSE, post-block hidden MSE, routing KL, mass/contribution
recall@k, teacher mass captured, and actual selected causal edge fraction.

Validity precedes training: input hashes and architecture must match; frozen
teacher must be finite; full-budget conversion must reproduce original SDPA
logits within max absolute error 1e-4 on one paired window per seed. A mismatch
stops the campaign as invalid. The zero-shot conversion must have logit KL
at least 0.001 in at least four seeds; otherwise classify
`no_material_conversion_gap` and stop before adaptation.

Primary joint gate for each seed at final k=2:

- combined_mass logit KL <= 0.80 times matched LM logit KL;
- combined_mass mass recall >= M0 mass recall + 0.05;
- combined_mass NLL <= 1.05 times F NLL.

At least 4/5 joint passes supports `routing_distillation_supported`. Otherwise,
at least 4/5 KL passes yields `behavioral_gain_without_full_joint_gate`;
remaining complete valid campaigns yield `distillation_not_confirmed`.
Secondary targets, BLiMP, intermediate updates, and other k values cannot
rescue the gate. Threshold inclusivity is exact. No p-value or pretrained
population generalization is claimed from these five adaptation streams.

## Downstream and mechanistic measurements

Zero-shot BLiMP subset: first 32 records in each of
`regular_plural_subject_verb_agreement_1`,
`regular_plural_subject_verb_agreement_2`, `anaphor_number_agreement`, and
`determiner_noun_agreement_1` (128 paired examples, fixed across seeds).
Download official JSONL data to `data/corpora/blimp_moba_20260909/` before
launch; hash files and tokenizer in the producer manifest. Score complete
sentences by summed next-token log probability with the corpus BOS token,
without length normalization; ties receive 0.5. Do not truncate examples.
Report overall and per-suite accuracy at final k=2, plus fraction of examples
long enough for sparsity to activate. No fine-tuning on this data. This is a
small grammatical-acceptability subset, not a full BLiMP or long-context
benchmark; corpus overlap is not audited at the text level.

Also report parameter deformation, routing target comparison, teacher-free
stage retention, and fixed-budget curves. Shared-GPU elapsed time is an
operational measure, not an acceleration benchmark. The dense-mask reference
does not measure sparse-kernel speedups or memory savings. One budget cannot
establish orders-of-magnitude adaptation sample-complexity improvements.

## Execution, provenance, and recovery

Entry point: `experiments/structure_net/tinyllm_moba_distillation.py`.
New artifact root: `data/experiments/tinyllm_moba_distillation/20260909_registered/`.
CPU unit tests and a tiny random-model lifecycle precede a real-architecture
CUDA shakedown under separate labeled roots. One worker on physical GPU 2;
the representative pilot at this single-worker concurrency also validates
the intended scheduling. CPU is allowed only for systems tests.

Freeze and copy all producer Python sources and this preregistration, record
their hashes, data/checkpoint identities, full immutable configuration,
environment and command before primary execution. Per-arm/seed artifacts and
aggregate use strict finite JSON with scientific fingerprints. Resume only
successful exact fingerprint matches with required checkpoint/schedule hashes.
Incomplete jobs stop and retain an error record; an infrastructure retry uses
a new root or an explicit incident record and cannot retry a scientific miss.
All completed evidence is append-only. No concurrent producers per root.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 pixi run python -m \
  experiments.structure_net.tinyllm_moba_distillation --device cuda:0 \
  --output data/experiments/tinyllm_moba_distillation/20260909_registered
```

## Sources

[MoBA repository](https://github.com/MoonshotAI/MoBA),
[MoBA paper](https://arxiv.org/abs/2502.13189), and
[BLiMP dataset and scoring protocol](https://github.com/alexwarstadt/blimp).
The user's attached proposal motivates this study; the numeric protocol above
operationalizes its otherwise unspecified budget, weights, controls and gates.
