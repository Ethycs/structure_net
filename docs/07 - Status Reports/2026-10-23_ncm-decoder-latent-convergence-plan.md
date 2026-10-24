# Fixed longer optimization of NCM teacher latents

Date: 2026-10-23. Prospective exploratory hypothesis
`ncm-decoder-latent-convergence-v1`. The completed 32-step control improved
target-aware reconstruction from 3133/3197 to 3170/3197 correct tokens, and from
12/32 to 19/32 exact texts, but missed the fixed 99.5% gate. There are 19 errors
on original 16 and eight on additional 16. Known residual errors include altered
technical terms and names. The prior failed result remains registered and
unchanged; this follow-up asks whether further latent optimization reduces them.

Use the same 32 complete TRAIN targets, four independent 48-position blocks,
FP32 original codec, vocabulary, EOS/PAD policy, objective and optimizer settings
as the committed prior control. Continue each exact saved latent and AdamW
state from step 32 through step 256, for 224 additional updates. Keep learning
rate .02, betas (.9, .99), epsilon 1e-8, weight decay 0, clip 1, and nonpadding
word CE plus .001 times MSE from the ORIGINAL continuous encoder latent. Do not
reset optimizer moments or move the anchor to the 32-step latent. Save the
inherited and added traces together. Select only step 256, with no per-case
budget extension, reroll or best-checkpoint selection.

Use physical GPU 2 with the same FP32, math SDPA and disabled TF32 configuration.
The codec stays frozen and eval, with state hash unchanged and no parameter
gradients. Require allocation below 2,000,000,000 bytes. The prior supervisor
has actually exited zero; no previous control remains live on this GPU. The
independent paired dropout study on GPU 0 remains unchanged. No source body,
conditional generator or judge is used in this target-aware control.

Before full computation, pass focused tests of exact continuation equivalence
to uninterrupted optimization, immutable input states, fixed budgets and
frozen decoder gradients. Run a separate two-case lifecycle/memory pilot from
the actual saved step-32 states, adding two steps each. Reload exact latent and
optimizer state and a fresh codec, requiring identical full readouts. Verify
the initial continuous and 32-step readouts match the earlier GPU results
bit-for-bit. The pilot is systems evidence only; the full run starts again
from the saved step-32 states and adds all 224 updates independently.

The gate remains at least 199/200 correct nonpadding target tokens, including
EOS, separately on original 16, additional 16 and all 32. Report the initial,
32-step and 256-step counts and exact target text counts. Classify the result
as oracle_latent_convergence_fit_passed or failed. A valid miss is preserved.
Passing establishes a label construction control on known targets, not a
source-only model; larger label construction and student training require a
new prospective plan. Both final cohorts remain unopened, task_accuracy stays
null, and no speed claim is admitted.

Use fresh roots 20261023_decoder_latent_convergence_{workflow,pilot,measure,results}_v1
under data/experiments/ncm_paper_regime. Commit source, tests and plan before
launch. A single supervisor owns pilot, full measurement, CPU finalization
and exact registry stages with actual child exit receipts. Freeze the prior
canonical archive SHA e3984e88a4d8844bab6fc680209855e115c9182ec8060dbdd2afa49141a6ca32,
raw measurements, tensor checkpoint and their receipts. Replay every inherited
trace, original latent, complete target binding, optimizer final step/moments,
new target hash, rate, finite objective/gradient, raw word/text and subgroup
gate. Preserve failures and the original fixed 95% semantic summary goal.
