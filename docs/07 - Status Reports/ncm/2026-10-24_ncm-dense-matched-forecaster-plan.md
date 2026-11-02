# Train concept forecasting on the full dense target population

Date: 2026-10-24. Exploratory hypothesis `ncm-dense-matched-forecaster-v2`.
The word branch trained on690 cases while its concept forecaster saw252.
Full690 dense latent construction now reconstructs62960/62965 active tokens
and685/690 target texts through the unchanged decoder. This qualifies labels,
not student quality. Train a new matched-population forecaster before dense
word adaptation. The earlier conditional forecaster remains declined; neither
its failed clean-word prerequisite nor any prior semantic failure is rescued.

Require the complete audited/registered dense-target archive
618ce743575e8ac962ac57150546d7dc905201a62ca35bd0305027fa56edc575,
with all four upstream child exits0. Use the same ordered690 TRAIN cases,
cases SHA ac3c56f7d03cf9160de9e719ef042b44a0bf12dececd778949b46e7288aac983,
and original-body feature manifest
dc831896063be90fcdaaca71a2bfb08a17bb417123ae0c7c87361c7e0167a5c7.
No newly reserved development/final sources enter training. Source reservation
may run concurrently using the already fixed known-source exclusions; no
outputs or labels on reserved cases influence its selection.

Start from the exact learned forecaster checkpoint
27f835d1391d7e14439716a0514a86d5b301bee79341199ab9dcfe88748e454d,
complete parameter hash5357dde0aa21c3c67cc281dcdb09ce74c380f6d29027eaa2853fe9a1aa863bf2.
Use a fresh optimizer, not random learned weights. Train only the existing new
source norm/projection, concept transformer/root queries and eight coarse heads.
Keep original Llama weights/MLPs, original codec, word projection/decoder and
finest head frozen. The parent is not loaded; cached original-body features
retain all source tokens. Use the existing numerical trainer in a new-process
wrapper, preserving the earlier producing files byte-for-byte.

Run32768 fixed updates, batch1, uniform TRAIN sampling, warmup256, maximum LR
3e-4 and cosine floor3e-5. AdamW betas0.9/0.95, epsilon1e-8, weight decay0.01,
clip norm1. Seeds52142 model,53142 sampler and54142 noise. Preserve original
dropout. Corrupt half the conditioning histories uniformly with rates descending
0.35 to0 over the eight coarse levels. Supply constant zeros for the unused
finest history; current/future-level labels cannot enter their predictions.
Root loss weight4 and padded-block weight0.25 remain. Word IDs determine only
active-block masks, never prediction inputs or word reconstruction losses.

Use physical GPU0 selected before startup, FP32 master weights/optimizer/loss
and reconstruction, BF16 forward, both TF32 modes off and less than8GB peak
head allocation. Require the actual device/memory preflight and focused CPU
tests. First pass a two-update maximum-LR/maximum-frame3328 gradient and exact
model/optimizer/sampler/noise/CPU-CUDA-RNG/native reload pilot. A separate fresh
32-TRAIN-case pilot checks finite, repeatable source-only eight-level rollouts
against native generation. Full training starts fresh from the same initializer;
pilot weights are not retained.

Save recovery checkpoints at8192,16384 and24576 and select only final32768.
Recompute the unchanged32 TRAIN diagnostics (original16/additional16). Report
the existing99/100 active-code threshold separately at each level/group as a
component diagnostic. No code-fit result establishes95% summary quality or
automatically admits downstream training.

After full training, cache fresh source-only608-code histories for all690 native
source frames. Strip target/candidate label fields before input construction.
Repeat first/longest probes exactly and bind every complete history to source
and final model hashes. All32 cached probe histories must match the separately
measured fitting results. Keep target-aware refined latents as future labels
only; they do not enter this forecaster or its predictions.

Freeze source/tests/plan/auditor before launch. One supervisor owns train pilot,
fit pilot, full training, fitting, full cache, CPU audit and registry/read-back,
with actual exits. Use fresh20261024 dense_matched_forecaster workflow/pilot/
fit_pilot/train/fit/cache/results v2 roots. Independently replay every sample,
frame, target/history hash, rate, weighted objective, optimizer/RNG state,
frozen parameter byte, raw fitting count, recovery state and cache identity.
Preserve every failure. Dense word training needs a separate prospective plan,
then fresh source-disjoint semantic development and closed final evaluation.
The95% complete-summary objective remains active; no speed claim is admitted.
