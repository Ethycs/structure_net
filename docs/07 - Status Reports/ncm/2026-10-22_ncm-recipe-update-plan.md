# NCM recipe update and bounded validation

Date: 2026-10-22. ID: `ncm-recipe-update-v1`. EXPLORATORY; earlier failures known.
Preserve all completed studies. New files and dated artifact roots only.

Implement and test decoder adaptation on randomly truncated cumulative code
hierarchies, generator conditioning-code corruption with unchanged clean labels,
and explicit per-level CE weighting. Encoder/codebooks and parent body stay
frozen. This is adaptation of released weights, not full VQ-VAE training or an
exact reproduction of an unpublished training loop. Codec adaptation does not
move code assignments. Re-verify frozen identities and save augmentation RNG.

Fixed updated recipe: nested dropout probability .5; uniformly keep 1..K-1
levels on dropped batches, all K otherwise. Uniform same-level conditioning-code
replacement rates decrease linearly from .30 at the first level to zero at the
last. Labels stay clean. Equal level loss weights (alpha=1) are an explicit
experimental choice, not a claim that the paper requires it. Separate model,
minibatch and augmentation RNGs; no corruption in evaluation.

Controls, before larger adaptation:

1. Focused CPU tests: actual latent truncation, full-latent parity, gradient
   routing, corruption bounds/vocab/label integrity, level weights and RNG.
2. Download the authors' public XSum test inputs with published SHA256 checks.
   Run the unmodified released evaluation shard on the first 128 inputs, batch
   eight, and score stemming ROUGE. This is a limited sanity check, not full
   paper reproduction or 95% task accuracy. Require nonempty finite output,
   mean ROUGE-L >=15 (0..100), and >=98% non-PAD codec reconstruction on this
   cohort before any new quality-training claim. A tiny two-case CUDA lifecycle
   check precedes this fixed control. No selection among evaluation samples.
3. Quality-check 128 candidate training pairs from the existing 2,048 shared-data
   training selection, using fixed ID-hash order, 20..40 words and native canvas
   eligibility. Send only verified public LongMemEval passages and candidate
   summaries to the already-qualified source-grounded judge. Use the same rubric
   and parser. Never retry a valid failure. Select the first 32 accepted pairs;
   if fewer qualify, stop a quality-training claim rather than use bad labels.
   This is training-data selection, not held-out evaluation.
4. A separate CUDA pilot validates real decoder and generator updates, nonzero
   level-zero/adapter gradients, save/reload, frozen encoder/codebooks, clean
   evaluation and continuation state. Physical GPU 0, one worker, no service
   changes. Stop on hash, nonfinite, OOM or parity failure.
5. After controls, a bounded 32-example memorization comparison may run: old
   versus updated recipe, same saved hybrid initialization and frozen body
   features. Each arm: 128 decoder updates (batch eight, lr1e-4) and 512
   generator+adapter updates (batch eight, lr1e-4), AdamW (.9,.95), WD .01,
   clip1, warmup32/cosine floor .1. Seeds model42, sampler2042,
   augmentation3042, dropout52. No validation-based checkpoint selection.

The bounded run measures training-set competence, free-generation outputs and
source-grounded train summary pass counts under blinded judging; these are not
held-out task accuracy. Require >=31/32 updated train summaries to pass before
proposing a larger generalization run. Report every failure and both arms.
Never use code accuracy, token reconstruction, ROUGE or train passes as the
95% application accuracy endpoint. No new throughput claim.

Artifacts: `data/experiments/ncm_recipe_update/20261022_*_v1`.
Freeze sources, config, input hashes and copies before model jobs. Keep producer,
raw replies, traces, checkpoints, report, aggregate and conservative registry
read-back. Baseline or data-quality gate failure stops the bounded run; systems
pilot may still validate the implemented update without a quality claim.
