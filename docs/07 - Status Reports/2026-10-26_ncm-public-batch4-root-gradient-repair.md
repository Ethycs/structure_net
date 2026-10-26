# Preserve learned-root gradients in the new batching path

Code review before the public batch4 lifecycle found that
`ncm_example_batching_codec_probe.serial_codec_logits` puts the learned root
query and concatenation inside `torch.no_grad()`. This repeats the old
attention-only path's gradient routing. The existing native fine-tuning
reference deliberately puts both outside that context. The previous batching
probe replaced the helper for both its serial and batched comparisons, so
agreement there did not verify the original root gradient. No optimizer steps
were performed in that batching study.

Preserve the completed batching study and its producing code. Its per-example
codec numerical observation remains evidence, but its helper is not admitted
for training. Add a new helper that retains per-example frozen codec
preparation while restoring learned-root gradients. The previous BROAD and
matched on-policy training used the correct native or sequential path; this
new helper regression does not explain their recorded semantic failures.

For the not-yet-run public batch4 lifecycle, compare the new batched helper
against four calls to the original `ncm_lm1b_native_finetuning.teacher_logits`,
not against itself. Require a nonzero root gradient in the reference and
compare every trainable gradient under the unchanged numerical tolerances.
Use the corrected helper for the subsequent optimizer/resume checks. The
initializer, four selected cases, update count and memory gates are unchanged.
The lifecycle also requires the completed exhaustive CPU code-target audit
and its actual successful receipt, honoring the target-preparation admission
rule before optimizer use. It re-encodes the four selected labels with the
initializer and compares them to the completed full target corpus.

A CPU regression test must demonstrate the old helper's missing root
gradient and the corrected helper's equality to the native reference. This
is a training-route repair, not new source-only accuracy evidence.
