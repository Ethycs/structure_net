# Affine value correction and packed attention follow-up

Study `JSPACE-FULL-COMPACT-01`, 2026-11-13. Preserve all v1 artifacts.

The v1 complete model finished at 103/256 after adaptation, versus native
210/256, with 36.35 versus 40.90 graph tokens/s. Development-only decomposition
gave MLP-only 46/64, native 48/64, and attention variants 18–26/64. The isolated
split-attention double-precision identity tests pass for both KV groups and head
widths 4 and 128.

An implementation omission was found while auditing the supplied affine trick:
the value PCA uses centered covariance, but v1 reconstructed `S S^T v` without
restoring `(I-S S^T) mean(v)`. Its selected value subspaces lose 70.87% and 81.89%
of value-mean squared energy in layers zero and one. That fraction describes the
mean, not total value energy, and is not itself proof of the accuracy cause.

The new version adds the fixed `sum_h O_h (I-S S^T) mean(v)` contribution to the
attention output. Since attention probabilities sum to one, this is the exact
affine correction for the retained mean. A regression test uses constant values
and demonstrates that the uncorrected implementation loses the mean, while the
corrected implementation preserves it before and after packing.

It also combines the native and padded compact heads into one attention call,
then one output projection. The fixed output offset is represented by two BF16
constant columns inside the GEMM accumulator, with padding to a multiple of 16.
This removes separate output rounding and reduces kernel count. The compact
queries still have 96 informative coordinates; padding does not restore the
discarded Q/K directions. No KV-cache byte saving is claimed.

New roots: `20261113_affine_shakedown_v2` and `20261113_affine_v2`. Repeat the
real backward/cache shakedown, the native/rotation/initialized controls, and the
same 96-step-per-layer progressive adaptation. All model/data/rank/anchor/loss/
optimizer settings remain fixed. Reuse of the evaluation pool remains explicit;
neither version is an independent confirmation. The v1 result is evidence about
an incomplete affine implementation, not a clean test of the supplied method.
