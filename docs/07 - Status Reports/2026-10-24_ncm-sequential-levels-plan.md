# Sequential native NCM learnability control

Date: 2026-10-24 (America/Los_Angeles). Profile: EXPLORATORY, one seed.
Hypothesis: `ncm-sequential-native-engineering16-v1`.

User requested sequential training after the parallel repair control. Test
whether the native generator can learn the same 16 engineering TRAIN examples
when explicitly processing one level at a time. Earlier native655 training
already used next-scale generation; this is not its first implementation.

Use the unchanged 16 IDs in engineering repair v3's selection.json. Preserve
all 732 labels and source-disjoint reservations. No DEV/final targets or sources
are opened. Start from the previously audited full checkpoint SHA
3911eab74db6ad13104f87765196e4212893a3e179ba0283e103e65e78273bec.
Train native generator plus concept source-attention parameters. Freeze the
complete codec and word-source attention, and keep them in eval mode. Llama and
its MLPs remain frozen; use audited full FP32 source features. This changes
capacity and objective from the 3.8M repair control: not a matched causal speed
or sequential-versus-parallel comparison, and not a complete paper replication.

Twelve levels: 1,8,16,24,32,40,52,64,76,92,108,128. At level k, input is the
native cumulative reconstruction of levels strictly below k. Compute native
categorical CE against clean level-k codes and backpropagate before advancing.
One optimizer update follows two microbatches; first has clean history, second
has uniform code replacement with rates decreasing linearly .25 to .05.
These endpoints are an exploratory choice, not an exact author recipe.
No word-gradient surrogate, joint repair head, or gradient through code choice.
Generation uses only source features and its own previously generated codes.

Fixed final checkpoint at 512 updates, AdamW betas .9/.95, eps1e-8, wd.01,
clip1, warmup32 then cosine 1e-4 to1e-5. Microbatch1, accumulation2. FP32 weights,
FP16 training autocast with GradScaler initial128; FP32 generation, TF32 off.
Independent seeds: sampler141142, corruption142142, dropout143142. Repeated
random permutations of16 cases, no target-dependent sampling. Physical GPU2,
one worker, peak allocated budget4.9GB. Never retry a numerical/scientific miss
to obtain a pass. Two-update pilot includes longest selected source, verifies
native/explicit step logits and greedy outputs, real gradients, codec freezing,
checkpoint/optimizer/scaler/RNG and exact output reload before capacity launch.

Fresh roots `20261024_sequential_engineering16_{pilot,capacity}_v1` under
data/experiments/ncm_paper_regime. Pilot weights are not reused. Hash all local
Python producers, vendor source, inputs and this plan. Preserve checkpoints
at128,256,512; only final512 selected. Record per-level TRAIN teacher-history
and source-only rollout code agreement; word agreement and decoded text;
frozen-codec oracle; source-shuffle control; loss trace, memory, terminal exits.

Learnability gate: final mean TRAIN clean-history CE <= .5 of initial;
source-only active word agreement >=95%; no empty outputs; inspect every
decoded source-only/target pair for material entity, numeric and relation
errors. This is a TRAIN diagnostic, never 95% task accuracy. If any check fails,
do not scale to732 on this experiment. If all pass, prepare a separately scoped
full-data experiment and semantic evaluation; this module has no full-data path.
Semantic judge availability remains unresolved and no automatic task-accuracy
claim is permitted. Previously sealed final evaluation remains closed.
