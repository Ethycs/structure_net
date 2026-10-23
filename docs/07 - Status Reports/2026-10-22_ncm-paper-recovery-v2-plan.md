# NCM recovery v2: train the codec and curate the paired labels

Date: 2026-10-22. EXPLORATORY, new hypothesis `ncm-paper-regime-recovery-v2`.
The v1 conditional-training path stopped: independent label audit30/32 (<31),
released codec96.3118% non-PAD reconstruction (<98%). Its 0/64 released-native
development baseline, failed gates, raw replies and pilot remain unchanged.
No full generator adaptation ran. The precision diagnostic reproduced the
96.3118% cell and found FP32/FP32 only96.3247%; precision is not the main cause.

Two independent repair stages may run concurrently (one GPU worker plus CPU/API):

1. Screen all 1,999 generated training labels with the qualified secondary judge.
   Reuse the first valid original32 audit judgments exactly, including both
   failures. Other labels get one valid judgment, at most3 transport/schema
   attempts; no semantic retry. Check eight-worker concurrency on8 already
   audited cases as a systems-only probe, never using those repeat decisions
   for curation. Keep all passing labels in original order, minimum1,024; this
   new curation threshold is fixed before observing whole-pool pass counts.
   Sample32 retained labels by a new fixed ID hash for the qualified primary
   judge; require>=31/32. That judge authored the teacher labels, so this is
   independent of the selector but not the author. Do not claim human gold or
   independent ground truth. Preserve rejected/invalid labels. The final model
   evaluation still requires both judges on untouched sources.
2. Train a genuine target-text VQ-VAE, starting from the released codec. Train
   encoder, decoder, learned resampling and residual projections; update frozen
   embedding weights by EMA, not AdamW. At each update choose the full hierarchy
   with probability.5, otherwise uniform cutoff1..8. Compute reconstruction CE,
   commitment (.25) and latent quantization MSE at that actual cutoff, with a
   straight-through reconstruction gradient. Quantization and nearest-code
   assignment stay FP32; encoder/decoder use BF16. This corrects the earlier
   decoder-only approximation. Data: all1,999 new labels plus complete47-token
   public training-source windows with EOS. Codec fitting does not claim those
   source windows or imperfect summaries are good supervised summary targets.

Codec config: microbatch32,1,024updates, AdamW lr5e-5, betas(.9,.95),WD.01,
clip1,warmup64/cosine floor.1. Model42,dropout52,sampler2042,augmentation3042.
EMA base.95 at reference batch4,096; use decay .95**(32/4096) per update.
Before training, normalize EMA cluster mass to batch32*level_width and reset EMA
weighted sums consistently with current embeddings, preserving the starting
book vectors. No dead-code restart. Save all EMA/optimizer/RNG state.
Half the codec samples are summaries and half source windows, with independently
seeded sampling. Evaluate fixed original1,999-label reconstruction after training;
require>=98% non-PAD plus >=31/32 source-grounded oracle passes on the curated
32-label audit cohort before conditional training. Those are training controls,
not held-out task accuracy. A two-update CUDA lifecycle/memory pilot and CPU
gradient/EMA/cutoff tests precede the full codec run. If it fails, preserve and
design the next iteration; never claim the generator passed through that gate.

After accepting a codec, freeze it, rebuild all code labels, and construct the
generator with head buffers derived from these exact books. Then run the
unpaired/bridge/conditional stages of v1 with new source versions, matched
native/hybrid data, and new roots. A fresh matching generator lifecycle pilot
must use the accepted codec. Final test admission still requires>=61/64 fresh
development passes. The same256 final-test sources remain sealed; success still
requires>=244/256 independently under both judges, with complete valid grading.
No threshold changes, human-accuracy claim or speed claim. New artifacts under
`data/experiments/ncm_paper_regime/20261022_*_v2`.
