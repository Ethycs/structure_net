# Codec precision diagnostic before tokenizer adaptation

Date: 2026-10-22. Exploratory child of `ncm-paper-regime-recovery-v1`.
Known outcome: the released codec, FP32 target-code assignment with BF16
reconstruction/decoder, recovered 67,112/69,682 non-PAD tokens (96.3118%) across
1,999 new teacher summaries. Only 676 decoded texts matched exactly. The
original full-data reconstruction gate failed; do not start conditional training.
The independent training-label audit is still running.

Before choosing a tokenizer training intervention, test a fixed four-cell
precision diagnostic on these same 1,999 labels. Encode once in FP32; compare
FP32 versus BF16 cumulative latent construction crossed with FP32 versus BF16
decoder computation. No optimizer, data selection, semantic grading or test
access. Store per-cell non-PAD/content counts, exact text counts, all decoded
outputs and maximum logit/latent differences. The BF16/BF16 cell must reproduce
the original integer count before interpreting the diagnostic. If it does not,
stop on a provenance/numerics discrepancy.

This is a new explanatory experiment, not a retrospective passing result for
the old codec control. If precision alone restores >=98% reconstruction, a new
explicitly defined mixed-precision protocol and matching systems pilot plus
oracle semantic control are needed before training. Otherwise proceed to genuine
tokenizer adaptation, with a new numerical training configuration written before
that job. Preserve the original failed control and all diagnostic cells.

One CUDA worker on physical GPU0, batch32, released tokenizer checkpoint,
FP32 model parameters. Artifact root:
`data/experiments/ncm_paper_regime/20261022_codec_precision_v1`.
