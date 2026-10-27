# Expand source access through the existing NCM generator

Exploratory systems hypothesis `ncm-source-coverage-pilot-v1`. The current model
injects source attention after blocks 8, 10 and 11 of twelve generator blocks.
The earlier depth experiment kept those three attention modules on the original
blocks while inserting new residual blocks. It did not test source attention
in every block. This observation is not proof of the cause of semantic failure.

Add nine source-attention modules after the remaining existing blocks. Copy
each query projection from the nearest original source-attention module, with
ties resolved to the lower block index. Initialize every added output projection
to zero. Keep the shared source K/V projection, existing modules, generator
depth, word codec, word-source path and original frozen Llama body unchanged.
No random initialization or new RNG draws are needed. The intended architecture
has 115,939,968 trainable generator/concept-source parameters instead of
105,323,136 when the same training scope is enabled.

This pilot checks only exact initial-function preservation and gradient routing.
It uses the fixed epoch2 checkpoint and two existing TRAIN controls: the first
of the original sixteen diagnostic cases and the longest of those cases. If
they coincide, retain that case and add the next original diagnostic case.
Before and after expansion, compare all FP32 teacher logits, source-only code
hierarchies and raw decoded word IDs exactly. Reproduce the baseline's saved
epoch2 outputs. Check every original tensor is unchanged. For each control,
backpropagate equal-level code CE without an optimizer step and require finite,
nonzero gradients on all nine added output projections; the new query gradients
must be zero at zero-output initialization. Frozen codec and word-source
parameters must receive no gradients. Clear all gradients and recheck tensors.

Run focused CPU tests first, then the real pilot under locked Pixi ncm on
physical GPU2 RTX2060SUPER, FP32 with TF32 off and deterministic algorithms.
Require at least 3.5 GB free before loading, and at most 3.5 GB peak own CUDA
allocation. Stream one complete original FP32 feature matrix per case. Preserve
both running GPU0/GPU1 service identities and do not touch the unrelated GPU2
resident. Use the fresh append-only root
`20261027_source_coverage_pilot_v1`; no fallback precision or tolerance is allowed.

A passing pilot is not trained quality, a full training lifecycle qualification,
or permission to skip representative microbatch/optimizer/restart checks for a
later fixed-budget comparison. It does not change the two current training jobs,
their evaluation order, the semantic rubric or sealed final test. No optimizer
update, new teacher/judge request or speed measurement occurs. The 95% semantic
goal remains unchanged; latest completed application DEV remains 0/128.
