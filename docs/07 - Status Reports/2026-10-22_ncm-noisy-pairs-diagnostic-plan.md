# Paper-regime training with measured label noise

Date: 2026-10-22. New EXPLORATORY hypothesis `ncm-paper-noisy-pairs-v1`.
This is a non-rescuing follow-up to the failed v1 competence controls and the
interrupted v2 screening infrastructure. Neither earlier gate is reclassified.
Known evidence: released native NCM0/64 on new development sources; label
audit30/32; new genuine codec98.2047% non-PAD reconstruction. Higher judge
concurrency caused read timeouts. No new full generator has yet been trained.

Question: does the complete staged NCM training recipe improve actual
source-disjoint summary quality with the existing1,999 generated pairs,
despite their measured label noise? A model may learn from noisy labels; the
teacher's sample pass rate is not logically an upper bound on the student's
task accuracy. Run this as an explicitly exploratory training diagnostic while
independent data screening continues, rather than call it a passing extension
of the earlier quality-gated studies. Label noise and codec errors are known
limitations, not hidden exclusions. Do not claim95% from reconstruction or CE.

Use all1,999 public teacher pairs without semantic filtering, the exact new
codec `20261022_codec_training_v2/model.pt`, and the same source-disjoint
2,048/64/256 selection (some train labels were canvas-ineligible). No new data
generation, test selection or private uploads. The codec's encoder, codebooks,
resamplers and decoder are frozen during generator training. Rebuild all code
labels and tied-head buffers from the accepted codec. Full Llama body stays
frozen. Initialize the generator from released XSum weights, bridge from seed42.

Use the unchanged numerical configuration in the v1 training addendum:
512unpaired updates on public training-source windows; then native-prefix and
frozen-Llama-context arms from the same checkpoint. Hybrid512bridge-only
updates, both arms2,048conditional updates. Batch8 x accumulation2, AdamW
(.9,.95), WD.1, generator LR1e-4/bridge3e-4, warmups128/32, cosine floor.1,
clip1, published dropout.3, BF16 autocast/FP32 codes and parameters. Uniform
same-level conditioning corruption.30→0, clean labels, alpha0. Separate seeds
42/52/2042/3042. Same conditional minibatch and augmentation streams across
arms. Right-pack frozen context. A fresh exact-codec CUDA lifecycle pilot must
pass before the full diagnostic. One worker on physical GPU0, no service changes.

The `--diagnostic` flag intentionally bypasses semantic label/oracle admission
gates for this new question; it never bypasses data identity, >=98% codec
reconstruction, checkpoint pairing, finiteness, gradient, memory or frozen-state
checks. Save the failed label-audit result as an input. Record unqualified-label
status in producers/results. Main quality-gated v2 remains a separate study.

Preserve development outputs at steps512/1024/1536/2048 and final32train outputs.
Measure complete-summary accuracy with unchanged qualified source-grounded
judges when the endpoint is responsive. No accuracy claim while grading is
missing/invalid. Improvement over the released0/64 baseline is diagnostic;
the user's target remains>=95%. Final test remains sealed. Admission to a new
final-test study requires >=61/64 development passes independently under both
judges, complete valid grading and unchanged judge calibration. Final success
still requires>=244/256 under both judges; report Wilson intervals and every
failure. No speed or human-quality claim from this diagnostic.

Artifacts: `data/experiments/ncm_paper_regime/20261022_noisy_*_v1`.
Freeze sources/config/input hashes and copies, preserve raw outputs/checkpoints,
and report/aggregate/register this new study separately. The previous failed
studies stay failed even if this diagnostic later produces a useful model.
