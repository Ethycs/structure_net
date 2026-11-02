# Dense word training on the full approved summary population

Date: 2026-10-24. Prospective exploratory hypothesis
`ncm-full-population-dense-word-v1`.

Train the source-only dense NCM word branch on all690 approved summaries using
the full690 matched forecaster and qualified optimized latent labels. The last
dense student fitted32 TRAIN examples at99.66% token agreement but failed
semantic development evaluation: raw0/64 short and1/64 long, with the lone pass
from an overlapping TRAIN source. That failure remains unchanged. This study
expands word training and uses the separately updated forecaster; it cannot
attribute any change to either intervention alone.

Use all690 unchanged TRAIN cases and all690 optimized256-step latent labels from
the registered target campaign
`618ce743575e8ac962ac57150546d7dc905201a62ca35bd0305027fa56edc575`.
Bind each actual source, complete192-token EOS/PAD target and label-state file.
Keep the same full source feature bytes. Teacher words and latents enter only
the loss. Prediction receives complete source features, source GPT2 IDs, masks
and608 earlier codes generated from that source by the fixed final32768-update
matched forecaster. No correct target codes enter prediction.

Initialize the word branch from the complete learned2048-update dense checkpoint
`0b8afde259c1165ddc4ebb14110568b0695228850dd09eaeadf645095550e266`.
Preserve the original decoder and zero dropout. Train only its source norm and
projection, word-level NCM transformer and word projection. The original Llama
body and MLPs, codec, decoder, forecaster, coarse heads and root queries remain
frozen. Keep latent MSE weight1 and nonpadding word CE weight0.1. The original
32-case fitting failure is not retroactively passed by this continuation.

Run32768 fixed updates, batch1, uniform sampling with model seed55142 and sampler
seed56142. Use256 warmup updates, maximum LR1e-4, cosine floor1e-5,
AdamW betas0.9/0.95, epsilon1e-8, weight decay0.01 and gradient clipping1.
Use FP32 masters, optimizer, decoder, reconstruction and losses with BF16
source/transformer/projection computation. Disable matmul and cuDNN TF32.
Retain recovery snapshots at8192,16384 and24576; select only final32768.

Before model training, run focused CPU tests, full690 label/source checks and an
actual CPU constructor check of every selected word parameter and frozen codec.
The CPU constructor may use the historical forecaster for systems checks only;
GPU training must use the new registered final matched forecaster.

The two-update GPU0 pilot uses the first TRAIN case and longest GPT2 source,
with the latter padded to3328 source positions. Both updates use maximum LR.
Require positive source and word-projection gradients, finite gradients and
losses, unchanged frozen weights, exact model/optimizer/sampler/RNG reload and
bit-exact native source-only predictions before/after reload. Their actual
forecaster histories must match the new cache. Restart the untouched learned
initializer and a fresh optimizer for full training. Peak allocation must stay
at or below8GB. Never change a launched source to repair a failed stage.

The observer may wait while both existing supervisors remain authoritatively
live. Capture their PIDs and Linux start ticks. Begin numerical work only after
the matched forecaster and fresh128-case DEV feature workflows both finish with
their full observed zero-exit lifecycles, exact registry readbacks and terminal
original process identities. This serializes GPU0 feature extraction and word
training. An upstream component fitting miss is retained as a diagnostic; an
incomplete lifecycle or infrastructure failure stops this workflow.

After training, measure the same original16 and additional16 TRAIN probes with
fresh native histories. Preserve raw192-word outputs, global EOS decoding and
the99.5% per-group token gate as diagnostics. A gate miss does not substitute
for complete-summary semantic evaluation or block a separately specified
exploratory evaluation of this fixed candidate. No source filtering, output
repair, clipping, autoregressive fallback, lookup or best-of selection is allowed.

The CPU auditor replays every sampled source/history/word/latent hash, source
frame, learning rate, weighted loss, optimizer step, sampler state, retained
snapshot, actual child receipt and frozen parameter byte, then the raw TRAIN
readouts and exact registry record. Fresh DEV evaluation must use all64 short
and64 long cases, qualified semantic judges and the unchanged joint61/64 gate
per cohort. Final320 evaluation stays sealed until eligible. Neither TRAIN
fitting nor this study alone establishes the95% summary goal or a speed gain.

Use fresh roots under
`data/experiments/ncm_paper_regime/20261024_full_population_dense_word_*_v1`.
Run with the existing NCM environment and
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=src:. python -m experiments.structure_net.ncm_full_population_dense_word workflow`.
