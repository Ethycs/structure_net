# Fixed-assignment NCM decoder polish

Date: 2026-10-22. EXPLORATORY `ncm-decoder-polish-v1`.
The genuine nested-cutoff codec run achieved98.2047% non-PAD reconstruction on
1,999 training labels but only1,169 exact texts. The subsequent32-example
coarse-curriculum diagnostic reached98.17% free code accuracy at step768 while
decoded target-token accuracy remained98.52% and19/32texts were exact. These are
training diagnostics, not task accuracy. The full-source final test stays sealed.

Question: can a final full-cutoff reconstruction phase reduce decoder errors
without changing the discrete language that existing generators learned? Start
from `20261022_codec_training_v2/model.pt`. Freeze the encoder, codebooks, all
resampling/residual projections and every other non-decoder parameter/buffer.
Train only the text decoder. Preserve the original code assignments and latent
reconstruction exactly; verify both state hashes and all summary code labels.
This phase follows genuine nested-cutoff tokenizer training; it does not claim
to reproduce the paper's original published training budget.

Fixed data: all1,999 training summaries and complete47-token windows of their
public training sources, with EOS and PAD. Each batch mixes16summary and16source
windows. No development/test source or reference label is used. Precompute clean
FP32-assigned codes, reconstruct the full9-level latent, use CE ignoring PAD.
No nested dropout in this final phase, no quantizer/EMA update. Train1,024steps,
AdamW(.9,.95), LR1e-4, WD.01, warmup64, cosine floor.1, clip1, BF16 autocast,
FP32 parameters. Seeds42/52/2042 for model/dropout/sampling. One worker on GPU0,
after the preceding training job exits. No resident-service changes or CPU
fallback. A fresh matching two-update gradient/frozen-state/reload CUDA pilot
must pass. Stop on changed encoder/books/projections, nonfinite values or OOM.

Compare the new decoder with the immutable original decoder on the same complete
summary hierarchies. Save all raw text/token outputs and integer counts. The
diagnostic gate is >=99.5% non-PAD reconstruction and >=90% exact training text,
with every non-decoder state byte and all target codes unchanged. Report failure
if either threshold is missed. Reconstruction is not95% summary task accuracy;
even a passing training gate still requires independent source-grounded output
grading and fresh held-out assessment.

This decoder is compatible with an existing generator only after an explicit
compatibility receipt verifies identical non-decoder state/code labels and binds
both original/new codec SHA plus generator SHA. Do not bypass an old generator's
pairing check or overwrite its metadata silently. A new composite artifact and
producer are required for a changed codec/generator pair.

Artifacts: `data/experiments/ncm_paper_regime/20261022_decoder_{pilot,train}_v1`.
Preserve baseline, pilot, checkpoint, optimizer/RNGs, raw outputs, source/config
hash copies, actual peak allocation and conservative final result. This one-seed
training fit is underpowered and cannot claim speed or final task success.
