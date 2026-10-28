# Source-tree activation checkpointing amendment

The first mixed-precision lifecycle attempt failed before optimization, during
the four-example FP32 gradient comparison on the 6 GB RTX2060. CUDA reported
4.66 GiB allocated and 639 MiB reserved but unallocated, with only 14.94 MiB
device memory free. Preserve that root, failure, and actual exit-one receipt.
No task-accuracy measurement or optimizer update occurred.

Repeat the exact source-tree lifecycle with nonreentrant activation checkpointing
around each block's complete source-attention branch (token leaves and pooled
ancestors). Recompute activations in backward, preserving random state. Preserve
parameter objects, leaf/ancestor attention, level routing, all source positions,
the four-example microbatch and 32-example accumulated update, FP32 comparison
and FP16 training, initializer, sampler, noise, optimizer and learning-rate
schedule. This amendment changes memory/computation tradeoff only. No source
file bound to an existing training producer may be changed.

First test exact CPU forward/parameter/input-gradient parity with nonzero tree
outputs, both zero and nonzero attention dropout. Test that recomputation leaves
the post-forward random state unchanged. Existing mask/pooling/initialization
contracts remain required. Then repeat original-versus-tree source-only output
parity, clean/noisy batch-four FP32 gradient parity, two accumulated updates,
and exact resumed second update on physical GPU1 RTX2060, with the same 4.9 GB
allocation ceiling. A second failure stops that attempt; no tolerance, batch,
precision, label or hardware change is implicit in this amendment.

Use `20261027_source_tree_training_lifecycle_v2` and checkpoint directory
`/data2/structure_net/ncm_source_tree_20261027_v1/lifecycle_checkpointed`.
Hash and retain the failed attempt and new CPU qualification as input evidence.
This remains systems qualification, not summary-quality or speed evidence.
