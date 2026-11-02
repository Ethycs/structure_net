# Check whether the failed word decoder was stopped before convergence

Date: 2026-10-23. Exploratory hypothesis `ncm-word-convergence-control-v1`.
The primary nine-pass summarizer remains a 0/64 failure. The paired-loss
capacity control is a separate training study and cannot change this endpoint.

JacobiForcing uses iterative causal block decoding. Our adaptation instead
stopped after four whole-frame word updates. Test this decoding difference
without changing any weights: restore the original failed 2048-update model,
use its source-only five-level coarse predictions, and keep the resulting
plan fixed throughout word refinement. Use only the same first four fixed
summary TRAIN cases already used by the crossed mechanism probe. No new
examples, targets at inference, teacher calls, development or sealed cases.

Record all 128 successive whole-frame native-word states, plus one additional
verification pass. Retain snapshots at 4, 8, 16, 32, 64 and 128 updates, first
stop and decoded text, valid-position training token matches, consecutive
matching-prefix lengths, and exact full-frame equality on the verification
pass. Use no oracle plan, clean previous words, external verifier, AR fallback,
early stopping, checkpoint selection or inference optimization. Compare the
four-pass raw IDs and coarse codes exactly against the original reload probe.

For a deterministic causal update where position j consumes only words before
j, iteration t settles at least the first t positions. Test that property on a
CPU toy recurrence. The 129th pass should therefore leave the 128-position
frame unchanged; a failure is a systems/causality issue to preserve, not excuse.
This is a convergence diagnosis. More iterations can trivially remove a
parallel speed advantage; no quality or speed success follows from convergence
or training-target recovery. Do not retroactively change the primary budget.

Reuse the completed, terminal-success native-word CUDA pilot: identical body,
precision, batch two, cache shape and 128-position word kernel, with no retained
activation graphs. The new loop only retains CPU integer states. Run physical
GPU0, BF16 body/FP32 new parameters, TF32 disabled, after the active training
worker terminates. Require unchanged body, books, new parameters and selected
cache, finite logits, original four-pass replay and exact final fixed point.
Root `20261023_native_word_convergence_v1`; preserve every state and failure.
