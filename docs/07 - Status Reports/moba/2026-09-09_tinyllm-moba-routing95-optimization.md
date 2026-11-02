# MoBA routing-recall optimization toward 95%

**Withdrawn before implementation or execution.** The user clarified that the
95 target refers to final downstream accuracy, not routing recall. No candidate
in this document was run, and it cannot define success for the active goal.

Date: 2026-09-09. Study ID: `tinyllm-moba-routing95-v1`.
Profile: EXPLORATORY optimization followed by a sealed five-seed evaluation.

The user requested a goal of “95” after the completed distillation comparison.
Pending clarification, this protocol interprets that as at least 95% teacher
top-block recall at the existing k=2 budget: one mandatory local 32-token block
and one selected completed past block. The previous recall estimator is fixed,
including its exclusion of the forced local block. Final NLL must remain no
more than 1.05 times the matched frozen full-attention teacher's NLL. The aim
is to meet both thresholds in all five adaptation streams, not just on average.

The original campaign remains completed and negative. Its k=4 control already
achieved 97.78% recall at 68.87% of causal attention edges, versus 34.63% at
k=2. Increasing k is not an accepted way to meet this optimization target.
No inference-time teacher calls or full-attention oracle routing are allowed.

## Evidence available before this protocol

Parent aggregate:
`data/experiments/tinyllm_moba_distillation/20260909_registered/campaign_results.json`,
SHA-256 `abcb16c37bc37ffc62afe8f1060075944f86225ae9af38fc82acd8c47f86f6d6`.
Mean k=2 recall: zero-shot 0.890278, KD 0.912149, combined mass 0.887080.
Combined mass reduced soft routing KL by 58.26% but failed every primary gate.
All those parent outcomes are known and may motivate exploratory choices.

The parent KD student for each adaptation seed (7,17,29,41,53) is the starting
checkpoint. The teacher remains the original pretrained native-256 d8 model.
Verify parent per-run and checkpoint hashes before use. Preserve parameterless
mean-key MoBA routing and the unchanged architecture in the first search.

## First fixed candidate search

Optimize seed 7 first, in this order:

1. `rank_qk`: Q/K only; LM + logit KL + hard-block cross entropy.
2. `rank_all`: all parameters; the same three losses.
3. `kd_all`: all parameters; LM + logit KL, as a longer-adaptation control.
4. `rank3_qk`: Q/K only; LM + logit KL + 3 times hard-block cross entropy.

The hard routing target is the argmax teacher attention mass over completed
past blocks, per query/head/layer. Scores are student Q dot mean(K) divided by
sqrt(head dimension) before cross entropy. First-block queries are excluded;
single-candidate queries contribute zero CE. Average over eligible queries and
heads, then layers. This directly supervises the discrete top-1 block decision.
All other weights are 1. Logit KL is KL(teacher || student), temperature 1.

Each candidate starts from the same saved parent KD checkpoint and the same
new token schedule for its seed. Every optimization forward uses k=2. AdamW
LR=3e-5, betas=(0.9,0.95), epsilon=1e-8, clipping=1, weight decay=0.01 for
all-parameter arms. Q/K-only arms have zero weight decay and mask fused V-row
gradients; all other parameters are frozen. FP32, deterministic algorithms,
no AMP/TF32, one 256-token sequence per update. Teacher losses remain active
through the final step; inference remains teacher-free.

Up to 1024 updates/candidate, validation every 128 updates including update 0.
Keep all evaluated checkpoints. Select the first candidate to achieve recall
>=0.955 and NLL ratio <=1.05 at two successive validation checkpoints; retain
the second checkpoint and its update count. Later candidates need not run if
this gate passes. This is an explicitly adaptive engineering search, not an
unbiased hypothesis test. If none passes, preserve the search result and write
a new dated search amendment before changing candidates or objectives.

## Data and seed roles

Use the same hashed BabyLM token stream and frozen original teacher as the
parent. Training uses the prefix excluding the final 262144 tokens. Training
schedule RNG salt 951103; candidate arms are paired by seed and exact schedule.
Validation uses 16 nonoverlapping 257-token bins from the first half of that
suffix, RNG salt 952203. Prior monitoring outcomes from this partition are
known; it is tuning data.

The final evaluation uses 32 bins per seed from the second half of the suffix,
excluding every bin used by any parent final-test seed. Use RNG salt 953301
and seed-specific sampling without replacement. Seeds may share evaluation
bins; they are adaptation replicates, not independent pretrained models.
Save all schedules before optimization, but do not evaluate the fresh final
windows during candidate search. Hash and retain the schedules.

After selecting a candidate and update count, run exactly that configuration
from the matching KD parent checkpoint for all five seeds. Do not select or
extend checkpoints based on final-test results. Report every seed's recall,
teacher-relative NLL, KL, and mask edge fraction. Success requires recall >=.95
and NLL ratio <=1.05 in every seed. If validation or final evaluation misses,
preserve it as a failed attempt and define a new experiment for further work;
do not reuse exposed final windows as a fresh test.

## Execution and artifacts

Entry point: `experiments/structure_net/tinyllm_moba_routing95.py`.
First root: `data/experiments/tinyllm_moba_routing95/20260909_search_v1/`.
CPU contract tests and a separate four-step CUDA lifecycle/memory pilot precede
the search. One worker, physical GPU selected before Python starts, currently
GPU 0 because GPU 2 is occupied. A systems-only pilot cannot qualify a candidate.

Content-hash and preserve this protocol, the new producer, the frozen parent
helper, model dependencies, environment and parent identities. Strict JSON;
append-only completed results and checkpoints; source/configuration changes
require a new root. Completed-result reuse requires matching fingerprints and
artifact hashes. OOM/infrastructure errors remain visible and do not count as
scientific misses. No speedup claim is made for the dense-mask reference.
