# The earlier batching helper must not be reused for training

Review before the public batch4 lifecycle found a gradient-routing regression
in `ncm_example_batching_codec_probe.serial_codec_logits`: its `no_grad`
context includes the learned root query and the concatenation feeding the
generator. The original native fine-tuning implementation places those
operations outside that context. Only fixed codec computations should be
detached.

The original saved GPU records corroborate this finding. In the default
batching probe, `concepts.generator.l0_queries` has a nonzero reference
squared gradient norm of `3.539212061313042e-9`. In both clean and noisy
comparisons of the serial-codec probe, the same parameter is recorded as
`expected_present: false` and `actual_present: false`. That probe patched the
helper on both sides, so its passing gradient comparison did not establish
equivalence to the original trainable-root path.

The [completed batching report](2026-10-26_ncm-example-batching.md), raw
results and producing code remain unchanged. Its per-example codec numerical
observation and resource measurements remain recorded. The detached-root
helper is **not admitted for training**, and the resource-only batch4 result
still does not qualify an optimizer run. No optimizer steps occurred in that
batching study. Earlier BROAD and matched on-policy training used the correct
native or explicit sequential path; this newly introduced regression does
not explain their 0/128 semantic results.

The replacement `ncm_trainable_example_batching.teacher_logits` retains
per-example frozen codec preparation while restoring the root-query graph.
A focused CPU test demonstrates the old missing gradient and verifies the
replacement against the original native reference, including a nonzero root
gradient and unchanged frozen embedding gradients. Ten focused tests passed
across this regression, disk training joins, checkpoint serialization and
existing batching helpers. They do not establish real-model CUDA parity.

The pending public batch4 lifecycle now compares four serial calls to the
original native fine-tuning reference against the corrected batched helper.
It explicitly requires a nonzero reference root gradient, unchanged gradient
tolerances, complete optimizer/scaler/RNG resume equality, protected tensors
and the existing 4.9 GB allocation bound. That GPU run has not started; it
waits for the live RTX2060 codec-target preparation to finish.

No new task-quality or application-throughput result is established by this
repair. The [dated amendment](../07%20-%20Status%20Reports/2026-10-26_ncm-public-batch4-root-gradient-repair.md)
preserves the original comparison and states the corrected qualification.
