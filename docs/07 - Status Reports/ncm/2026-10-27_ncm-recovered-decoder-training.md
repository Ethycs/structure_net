# Broad recovered-text decoder adaptation

Exploratory hypothesis `ncm-recovered-decoder-domain-v1`: two complete passes
over the newly recovered native text frames improve reconstruction of the fixed
128 held-out diagnostic targets. Known parent results are 101/128 exact native
renderings, 5264/5306 lexical positions and 26/27 digit-bearing cases preserved.
The diagnostic source components and duplicate native frames were excluded from
the 99,367-frame corpus before this proposal. These diagnostics are development
data, not a new confirmatory test. The final application cohort stays sealed.

Initialize from the repaired codec tensor hash
`eb962706f10c693871f9c6df007e9fd3e23004c91c6258f435cdb09abf7fb366`.
Train only its decoder. Freeze the encoder, VQ books and all other buffers;
the source generator and Llama body are not loaded. Use the completed frozen
code cache, with exact encoder identity and parity checks. Native input is
128 BERT positions including both CLS boundaries and PAD0; code width is 641.

Two separately seeded shuffled epochs each cover all 99,367 unique frames once,
including the seven-frame tail. There are 3,106 updates per epoch, 6,212 total.
Each update accumulates up to 32 domain examples in microbatches of four, plus
eight author LM1B TRAIN retention examples in two microbatches of four. Domain
and retention losses have fixed weights 0.8 and 0.2. Each is the mean of per-text
non-PAD cross-entropies, so short and long texts have equal example weight.
Retention uses the prior 16,384 TRAIN frames and its existing 20% native / 80%
random-prefix augmentation. No code corruption is introduced in this experiment.

Use deterministic FP32, TF32 off, AdamW (betas 0.9/0.95, epsilon 1e-8, weight
decay 0.01), gradient clipping at 1, warmup 128 updates to 2e-5, cosine decay
to 2e-6 at the fixed final update. Seeds: domain 93142 plus epoch index,
retention 94142, augmentation 95142, stochastic model 96142. No early stopping,
best-checkpoint selection or retrospective changes to data/loss/budget.

Read domain frames and codes through read-only memory maps. Retention is the
fixed small TRAIN tensor. Use locked Pixi, physical GPU2 RTX 2060 SUPER, with
a 4.5 GB own allocation ceiling and at least that much initially free memory.
Do not alter other GPU workers. Save full decoder, optimizer and RNG state to
append-only checkpoint files every 1,024 updates and at each epoch boundary.

Before full training, run focused CPU contracts, then a representative CUDA
lifecycle pilot with the same accumulation, optimizer and precision. Use one
ordinary shuffled update and one update containing the longest and shortest
TRAIN frames. Save after two updates; run a third update uninterrupted, reload
the two-update state into a fresh model, and require the same third-update
losses, parameters, optimizer state and RNG exactly. Require finite nonzero
decoder gradients, no protected gradients, unchanged encoder/VQ, and the memory
cap. The full run resets to the parent, never the pilot weights. A failed
contract stops that attempt and is preserved; no silent precision/batch fallback.

After actual successful terminal execution and checkpoint replay, evaluate all
128 held-out targets once, using the fixed original codes and native renderer.
Report exact complete text, lexical agreement, digit preservation, all paired
regressions and improvements. Directional improvement means more than 101 exact
texts without worse digit-case count; the stronger reconstruction target is at
least 122/128 exact and all 27 digit-bearing cases preserved. Neither result
constitutes source-to-summary task accuracy. Separately report training time,
memory, coverage and the lack of a measured inference speed gain.

Roots: `20261027_recovered_decoder_{pilot,train}_v1` under the NCM experiment
directory; checkpoints under `/data2/structure_net/ncm_recovered_codec_20261027_v1`.
CPU tests and their actual exit receipt are required before the CUDA pilot;
the actual pilot result/receipt is required before the full run.
