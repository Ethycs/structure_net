# Final-feature recovery: prospective controlled comparison

Status: prepared protocol; no quality or speed result yet. This is a new branch of the Qwen2.5-3B experiment, not a change to the frozen staged campaign or its hidden-feature diagnostic.

## Question and fixed control

Does retaining the accepted parent's final compact answer representation during joint recovery prevent quality loss when a thirteenth compressed block is added? The original joint objective matches parent logits and next-token targets but has no final-feature term. This arm adds that term without changing the architecture, data, teacher, optimizer, or protected selection rule.

Use the authenticated raw and prealigned initializers from cut23 of `jspace_3b_staged_campaign/20261115_v1`. The accepted twelve-block teacher is checkpoint `db53db32597ebb4947e38f7324e31e6eb9c7686f491a9e8b9745b0c43bca8292`. Its targets are actual post-block35 compact coordinates before the output bridge, captured by the separate read-only diagnostic. Do not reconstruct and reproject these coordinates. Do not substitute native-model features in this arm.

Keep the same 512 training cases, 64 reused development cases, order, BF16 learned parameters and forward computation, frozen geometry, CPU FP32 Adam masters, 12 epochs, 384 updates, learning rate 2e-5 with the existing schedule, accumulation, clipping, and temperature2. Preserve the raw protected candidate and warm initializer exactly. The untouched confirmation set remains unopened.

## Added objective and train-only calibration

The endpoint objective remains `0.7 * T^2 * KL(parent || student) + 0.3 * CE`. Add `alpha * mean((z_student - z_parent)^2) / E_train` at the last valid input token, where `E_train` is the mean squared coordinate energy over all512 TRAIN parent targets. Require a finite positive energy; normalization is never fit on development examples.

Fix alpha in a disposable pilot using the first eight training cases from each of the four families in the frozen source order. Compute gradients separately for the endpoint objective and the normalized feature objective at the same prealigned initial weights. Over those32 cases, use the same case/batch grouping in both measurements and accumulate squared gradient norms of shared compressed-block parameters. Exclude final normalization parameters from this calibration because the feature term bypasses them. Set `alpha = 0.1 * sqrt(sum(endpoint_gradient_norm_squared) / sum(feature_gradient_norm_squared))`. Require finite positive calibration sums; do not tune alpha using development results. Save calibration identities, grouping, norms, normalization and alpha in an authenticated receipt. Restore/discard pilot updates before training.

The differentiable feature must be the actual block35 output before the outgoing bridge. A functional forward may expose it, but must preserve existing block order, bridges, checkpointing, masks, rotary embeddings, and numerical execution. Verify its outputs and gradients against the existing forward on CPU fixtures and verify the real raw/warm packed development controls before training.

## Evidence and selection

Keep the existing eligible winner and original-native continuation gate unchanged for the controlled comparison. Also save a separate highest-development-accuracy checkpoint, breaking ties by lower CE then earlier epoch, even when that checkpoint is ineligible under the existing CE/KL rule. Label its eligibility explicitly; it must not silently replace `selected.pt`. The added feature loss does not introduce a new quality gate.

Log endpoint and normalized feature losses separately, the fixed alpha, gradient/update controls, every epoch's full existing development metrics, selected checkpoint hashes, fixed-state identity and fresh strict reload. Compare against the completed cut23 baseline: retained46/64, best observed49/64 at epoch5 whose weights were not saved. Neither reused development accuracy nor a synthetic optimizer speed measurement is a held-out quality or inference-speed claim.

A subsequent native-teacher arm may use original-native projected features plus original-native logits, but that changes the teacher as well as the feature objective and requires its own prospective protocol. Resident optimizer adoption is a separate controlled experiment and is excluded from this initial feature comparison.
