# Q4 kernel layout revision before model evaluation

Exploratory amendment to `llama32-3b-q4-kernel-runtime-v1`. The first operator
producer completed 98 cases and twelve irregular-mask controls with numerical
agreement. Most fused layouts were slower than the old kernel; transient
prefill was faster at long widths, while fused decode was substantially slower.
No model quality/development output has been observed. Preserve the entire
`20261014_operators_v1` producer, rows and selection as the first tuning attempt.

Revise the implementation in a new operator root `20261014_operators_v2`:
use direct packed loads instead of join/permute unpacking; retain the old fast
FP32 decode kernel and fuse only its reduction/core/add epilogue; admit K=32
prefill tiles; and write transient dequantization in contiguous K-by-N order,
passing its transpose to cuBLAS. Test the new finite layout list recorded in
the v2 producer. Decode uses FP32 accumulation for every family. Share its
selected layout across families; select the unchanged old decode instead if
its summed operator time is no larger. This fallback changes no model weights.

Prefill still compares full FP32 and staged FP16/FP32 arithmetic. The transient
family still holds no dense tail across calls and includes its per-call dense
temporary in peak memory. Numerical thresholds, mask controls, model pilots,
development/final seeds, quality/speed gates and all primary endpoints remain
unchanged. The first version's timings remain reported rather than pooled
with the revised operator selection. Fresh 26000-series inputs were already
generated with a preserved producer; its generator logic is unchanged.
