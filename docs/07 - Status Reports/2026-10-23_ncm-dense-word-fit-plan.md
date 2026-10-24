# Source only NCM word fitting with dense latent targets

Date: 2026-10-23. Prospective exploratory hypothesis
`ncm-dense-latent-source-word-fit-v1`. The fixed 256-step oracle control recovered
all 3197 active tokens and all 32 TRAIN target texts with the original codec
unchanged. The student has not learned these latents. This study asks whether
dense supervision can repair its fitting failure before another larger run.
It remains subordinate to the full 95% source-disjoint semantic summary goal.

Admit GPU work only after the running original/zero-dropout comparison completes
all eight child stages with zero exits, is audited and registered, and its
original supervisor PID/start identity is terminal. Require both fitting gates
to fail. If either dropout arm passes, decline this study without GPU stages;
that simpler result should inform the next whole-population experiment. An
upstream infrastructure/provenance failure stops this workflow. Never restart
or modify the running dropout experiment.

Use precisely the same 32 known TRAIN cases: original 16 and additional 16, with
complete public sources and no sealed evaluation access. Bind the dense labels
to the registered 256-step teacher archive SHA
1a5587221af8048f3052296bb17d6c6ab40afdf7e62d689088b7b0061070a834.
Keep all 192 word positions, four independent 48-position decoder blocks, EOS
50256 and right PAD 50257. The dense target is the final optimized 192-by-384
FP32 tensor for each case. Do not generate labels for other sources here.

Initialize the learned word branch from the exact final clean-word checkpoint
c90ee2ae272c298022d50d7a9b4182f06d8d995fadf8ecceb6f091e35b8b5223.
Restore its separate word decoder to the original released codec weights, then
freeze that decoder and keep it eval. Set all module and numeric attention
dropout to zero in the word transformer and copied decoder. Preserve original
Llama body/MLPs, feature caches, codec, old forecaster, unused root queries and
coarse/fine heads. Train only the existing source norm/projection, word
transformer and word projection. No new parameters or random weights are added.

Prediction receives detached complete frozen-body source features, source IDs
and mask, plus actual source-only coarse histories from the unchanged cached
forecaster. Never feed target words, refined latents or correct coarse targets
into prediction. Load source batches through the source-view guard without
label fields; construct labels separately for the loss. Fresh composed inference
must reproduce cached concept histories at native source frames.

The objective is mean squared error over the full latent frame plus 0.1 times
word cross entropy. Include EOS and ignore target PAD in CE. Dense supervision
includes padding latents because decoder attention can read every position.
Use 2048 updates, batch 1, no accumulation, AdamW betas (.9, .95), epsilon 1e-8,
weight decay .01 and gradient clip 1. Warm up for 64 steps to 1e-4, then cosine
decay to 1e-5 at step 2048. Model seed 45142 and independent sampler seed 46142
are fixed. Require every selected case to be sampled. Save a recovery snapshot
at 1024; select only the final 2048 checkpoint. No output selection or reroll.

Use physical GPU 0 after the previous supervisor is terminal. Use FP32 master
weights, optimizer, decoder, latent residual addition and loss, with BF16 source
projection/word transformer/word projection under autocast. Both TF32 modes stay
off. Keep the original body unloaded during this cached-feature fitting study;
actual application inference will still require its complete source pass.
Own peak allocation must stay below 8 GB.

Before full training, pass focused numerical/gradient/scope/label-separation,
schedule/admission/raw-replay tests and an actual-data CLI check. The two-update
CUDA lifecycle and memory pilot uses the first and longest selected sources at
maximum LR, padding the second to frame 3328. Require finite objectives and
positive source/word projection gradients, no frozen gradients, unchanged frozen
states, and exact model/optimizer/sampler/CPU-CUDA RNG and native composed-output
reload. Pilot updates do not initialize full training. Run a fresh full 32-case
fitting worker afterward using actual forecaster rollouts only.

The fitting gate is at least 199/200 correct nonpadding word tokens, inclusive,
in EACH original 16, additional 16 and union 32 group. Report exact texts as a
secondary diagnostic. Classify dense_source_TRAIN_fit_passed or failed. Passing
is a fitting result on seen sources, not unseen semantic accuracy. A larger
690-case latent bank or student run needs a new prospective study; no stage is
automatically admitted by this result.

Use fresh roots 20261023_dense_word_fit_{workflow,pilot,train,measure,results}_v1
under data/experiments/ncm_paper_regime. Commit producing source, tests, auditor
and plan before launch. The supervisor owns pilot, train, fit, CPU finalization
and exact registry stages. Preserve full states, all actual teacher/word/history
hashes, sampler/rate/optimizer traces, raw outputs, receipts and failures. Audit
initializer restoration, every frozen parameter, full selected-population
coverage, subgroup gates and native cache/reload parity. Keep task_accuracy null,
confirmed false and both final cohorts unopened; do not claim speed gains.
