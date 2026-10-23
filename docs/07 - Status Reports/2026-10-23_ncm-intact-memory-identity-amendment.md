# Preserve the exact inference path for a zero-output correction

Date: 2026-10-23. EXPLORATORY systems recovery, v2.
The v1 intact-memory GPU pilot exited1 at the initial exact logits/greedy check,
before any optimizer updates. CPU identity tests had passed. Preserve v1 source,
producer, traceback and failed gate. The original check combined logits and
greedy equality and did not save their individual differences, so the cause is
not established. No task-quality conclusion follows from this systems failure.

An FP16 linear operation can select different arithmetic for a copied tensor
with different strides. That is a hypothesis, not the observed cause. Make the
zero-output inference case an explicit identity returning the original hidden
tensor, preserving its strides and storage. Only when autograd is disabled and
the output matrix is exactly zero may this bypass the correction arithmetic.
When gradients are enabled, run the original correction even at zero so its
output projection can learn. After a nonzero update, always run the correction.
Do not bypass nonzero trained weights, repair outputs or alter native weights.

Use a new subclass/producer and `20261023_intact_memory_pilot_v2`, retaining the
original two-update/full128-target/FP16/GPU2/batch2 lifecycle protocol otherwise.
Require the original initial exact logits and32-token greedy checks, all frozen-
base/gradient/feature checks and exact reload. Add CPU tests of identity object/
strides, initial gradients and nonzero-output branch. Stop if this new pilot
fails; no repeated roots or threshold relaxation. Synthetic concept vectors
remain target-independent; no actual NCM or task-quality/speed claim.
