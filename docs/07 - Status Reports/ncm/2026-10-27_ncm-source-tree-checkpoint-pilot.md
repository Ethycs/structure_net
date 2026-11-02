# Source-tree checkpoint pilot

Exploratory systems check `ncm-source-tree-checkpoint-pilot-v1`, following the
successful four synthetic CPU contracts in `20261027_source_tree_CPU_qualification_v2`.
This is a source hierarchy shared by all twelve NCM blocks. It does not connect
the outputs of different transformer layers into a tree.

Load the unchanged epoch2 checkpoint and the same two existing TRAIN controls
as the source-coverage pilot: the first diagnostic case and the longest source,
with the next diagnostic case breaking a duplicate. Use FP32, deterministic
execution, TF32 off, locked Pixi, physical GPU2 RTX2060SUPER. Require 3.5 GB free
at entry and own peak CUDA allocation below 3.5 GB. Do not stop other residents.
Record the public and expanded-task workers' invocation identities and require
them unchanged at completion; natural successful completion is permitted.

Reproduce archived epoch2 source-only codes and raw word IDs. Install the fixed
eight-way source tree, preserving all original parameters and RNG state. Require
exact equality of every teacher logit, generated code and raw word ID before
and after installation. Rename bindings only by the wrapper's `.leaf` path;
the actual original parameter objects must be identical. Backpropagate the
same full-hierarchy code loss on both cases without taking an optimizer step.
Require finite, nonzero gradients on all twelve new tree-output projections
and all nine new leaf-output projections. Zero-output initialization must leave
the new level-router and newly copied query gradients zero. Frozen codec and
word-source parameters must receive no gradients. Clear gradients and verify
that every tensor still matches its pre-backward binding.

Freeze producer sources, inputs and this plan before model loading; save exact
outputs, tensor bindings, gradient measurements and actual terminal receipts
in `20261027_source_tree_checkpoint_pilot_v1`. Preserve any failure without
reusing its root. There are no optimizer updates, judge requests or final-test
accesses. Passing does not establish trained accuracy, latency, mixed-precision
training, optimizer/restart behavior or full training qualification. A later
matched comparison still needs baseline, twelve-layer dense attention and
twelve-layer source-tree arms on the same data and training exposure.
