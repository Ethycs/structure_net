# Broader complete-summary training with the reconstruction path frozen

Hypothesis `ncm-broad-generator-v1`; exploratory, one training seed and
underpowered for architecture claims. The parent scored 0/128 complete summaries
on source-disjoint DEV. Its codec-only reserved reconstruction was 99.29% of
content tokens, but that is not semantic summary accuracy. The last decoder-only
adaptation degraded reserved oracle reconstruction and still scored 0/128.

Test whether additional sequential code-generator training on the verified
1,376 complete-summary teacher pairs improves source-only semantic accuracy.
Start from the earlier generator parent, checkpoint SHA256
`37d8d70f14d81a63ecd63ca6b9a3f5bf1ac9301d866af1135e24404224e795ad`, full model hash
`9eb6e2c41a3a4e496c588a2ec0666c4cbad6f93ba0fc16c6c88b0ea2df2d339b`.
Train only the existing generator, concept source memory and concept attention
adapters. Freeze the entire codec and word source memory/adapters, preserving
clean oracle reconstruction by tensor identity. This is not the proposed
contrastive-loss experiment: no discriminator, altered factual target or new
architecture is introduced. The baseline is reused, not a compute-matched arm;
broader data and additional updates cannot be causally separated here.

Use every admitted row in `20261025_broad_teacher_catalog_v1/cases.json`, SHA256
`cd3d2a8a1fe0d398d1ca951e0df382d05c47cefe4d07892107b23d6e1c7e5914`.
All 1,449 observed first candidates remain accounted for and the 73 invalid
firsts remain excluded. These 1,376 reviewed targets are not independently
verified semantic gold. Keep the previously rejected atomic-label pool out.
The catalog's 512 protected source components, 192 open source text guards,
64 reserved component cases, and sealed final320 are unchanged. Some sources
may have appeared in historical native TRAIN; these are not all novel sources
to the whole model lineage. No source truncation or target clipping is allowed.

Prepare all 641 target codes in FP32 using the fixed parent encoder/VQ, retaining
source/target/frame identity. Record codec-only and source-bound clean oracle
reconstructions for every TRAIN row. Verify the first, longest-source and
longest-target rows by repeat encoding. These are diagnostics, not admission
filters. Select 16 TRAIN diagnostics before looking at model outputs: the first
15 IDs in lexical order plus the longest-source ID (fill to 16 if duplicated).

After focused CPU tests, run a two-update CUDA pilot on first and longest-source
rows with packed-versus-explicit gradient/logit parity for clean and corrupted
history. Check actual optimizer updates, absence of frozen gradients, finite
loss/gradients, strict model/optimizer/scaler/RNG/sample-state reload, preserved
frozen hashes and identical oracle outputs. Single worker on physical GPU1
RTX2060, locked Pixi ncm from repository root. The pilot does not feed training.

The fixed full run is 4,096 optimizer updates, microbatch1, accumulation4,
without-replacement shuffled sampling (seed181142), code-corruption seed182142,
dropout seed183142. This gives about 11.9 presentations per example, comparable
to the earlier 732-example generator run's exposure count. Half of presentations
have corrupted earlier-level conditioning; keep the existing 25% early-level
to 5% late-level replacement schedule and unchanged gold labels. Use the existing
packed block-diagonal equal-level code CE and explicit 12-level source-only
inference. FP32 weights, FP16 training, FP32 evaluation, TF32off. CPU FP32 AdamW,
betas0.9/0.95, epsilon1e-8, weight decay0.01, clip1; warm128 updates to1e-4 and
cosine decay to1e-5. Full FP32 disk features with a256MiB LRU cache. Peak CUDA
allocation must not exceed4.9GB at any lifecycle phase. Optimizer CPU memory is
separate from the bounded feature cache. No other GPU service is modified.

Stop on provenance, finite-numeric, gradient, frozen-state, memory or lifecycle
failure; preserve failure artifacts. Never retry a score miss to obtain a pass.
Only the fixed final checkpoint is eligible; no score-based early stopping or
checkpoint selection. Save checkpoint and optimizer/RNG/sample state before
final diagnostics so evaluation failure does not require repeating updates.
Evaluate all1,376 TRAIN rollouts and16 shuffled-source controls diagnostically.

After valid training, score all128 unchanged DEV cases using the existing
complete-summary semantic protocol, regardless of TRAIN agreement. Terra first:
at least61/64 in both short and long cohorts permits Sol; at least61 same-case
joint passes in each cohort qualifies DEV. Preserve all3,795 existing canonical
first judgments and all historical uncertain-request reservations. Invalid
transport is unresolved evidence, not model inaccuracy. A valid positive but
subthreshold score is partial progress; zero passes fails the intervention.
No token/code agreement threshold substitutes for semantic grading. The sealed
final320 remains closed unless DEV qualifies; DEV success alone cannot complete
the95% task goal. No speed claim or timing benchmark is part of this study.

Fresh roots: `data/experiments/ncm_paper_regime/20261025_broad_generator_{prepare,pilot,train}_v1`.
Entry point: `pixi run --locked -e ncm python -m experiments.structure_net.ncm_broad_generator <stage>`
with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`
set before Python. Freeze producer/input/environment identities and preserve all
raw training traces, outputs, actual process receipts and previous failures.
Audit, aggregate, report and conservatively register after semantic grading.
