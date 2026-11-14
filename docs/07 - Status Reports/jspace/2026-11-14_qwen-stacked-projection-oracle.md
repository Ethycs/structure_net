# Stacked teacher-state projection diagnostic

Study: `JSPACE-QWEN-STACKED-PROJECTION-ORACLE-01`, exploratory diagnostic.

Before this protocol, the recovered all36-block assembly scored27/64 against
native50/64. Its fixed three-epoch joint-distillation continuation is running;
epoch1 scored24/64. This diagnostic runs independently and will not change
that continuation's objective, duration, selection rule or source code.

The question is whether a greedy teacher-state correction through the existing
sequence of fixed update subspaces can preserve the native decisions on the
same64 development prompts. Use the authenticated native FP32 development
cache and the current assembly's exact per-layer bases: the separate first
fit for block0 and the corresponding all-layer fits for blocks1..35.

Initialize h'_0 to native h_0. For each layer l=0..35, compute

`h'_(l+1) = h'_l + U_l U_l^T (h_native_(l+1) - h'_l)`.

The correction is relative to the ACTUAL accumulated oracle state, not the
native layer input. Each correction lies in the fixed rank1024 subspace;
the remaining full2048-wide residual carries forward. The oracle does not
execute a learned compressed attention or MLP. It receives the teacher's
full internal target at every layer and is therefore not deployable.

At the final boundary, use original native RMS normalization and tied vocabulary
readout. Compare oracle choices against the original labels and native choices.
First require the cached native h36 control to reproduce50/64, all reference
predictions, and logits within5e-4 absolute plus1e-5 relative tolerance.
Preserve original case order, identities, lengths, offsets and answer positions.

For every layer report relative RMS error versus native h_(l+1), for all valid
tokens and separately the last answer row. Use equal case weighting, equal
valid-token weighting within each case, and per-row squared error divided by
native row squared norm, clamped below at1e-20, followed by the square root
of the average. Also report paired gains/losses and per-family final accuracy.

The existing47/64 gate is a reference for whether this oracle preserves task
quality on these cases. A pass does not prove the compressed block can learn
the oracle correction. A miss does not prove the architecture cannot perform
the task: greedy Euclidean projection is not an upper bound on task accuracy.
This diagnostic separates a concrete projected state trajectory from its
learned implementation; it neither replaces the running training experiment
nor establishes generalization, compression quality, or inference speed.

New root: `data/experiments/jspace_stacked_projection_oracle/20261114_v1`.
Hash the assembly metadata, all36 fits, all37 native boundary files, items,
reference predictions, code, tests and protocol. CPU preparation precedes
GPU use. Run through locked offline Pixi ncm and inference-ctl on the otherwise
idle RTX2060, GPU-8cf440df, with3072MiB reserved and allocator capped256MiB
below the lease. Check the live lease and memory peak; no gradients or training
updates. Keep failed receipts and never overwrite completed evidence. The
fresh512-case confirmation set remains unopened; confirmed:false.
