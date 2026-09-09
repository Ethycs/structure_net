# Qwen MoBA idle-GPU latency diagnostic

The user challenged the approximately two-second prefix latency. GPU 0 is now
idle with 11,618 MiB free; the unrelated 5 GiB workload has ended. Preserve the
completed shared-GPU study. This is a systems diagnostic on its already-selected
block1024/top2/all-six-layers configuration, not a new quality selection.

Use the same public frozen Qwen3-8B prefix, FP16, and Wikipedia validation
window at 8192 and 16384 tokens. Compare full attention with 2048-token MLP
chunks, full attention with one complete-sequence MLP call, MoBA with four-head
chunks, MoBA with all 32 heads, and MoBA with all 32 heads plus the complete-
sequence MLP. Every path executes the same five MLPs and sixth-layer readout.
This changes execution scheduling only. Record readout differences between
scheduling variants and the original full-attention reference.

Use one warmup followed by five alternating/rotating CUDA-event measurements
per path. Record peak memory and GPU/process state before and after every cell;
reject uncontended interpretation if another GPU process appears or an existing
external process grows materially. Preserve OOMs as capacity failures.

Separately instrument attention calls, the five MLPs and attention projections
with CUDA events for a component-cost diagnostic at 16K. Do not include those
instrumented runs in the primary timing sample. Event ranges are approximate
component attribution and include launch gaps within each range.

All loaded weights remain frozen and are hashed before/after. Snapshot outcome-
relevant sources and this protocol under a new
`data/experiments/qwen_moba_prefix/20260909_idle_latency_v1` root. Report measured
latency, scheduling effects, approximation metrics, limitations and exact
provenance. Do not relabel reused validation windows as new held-out evidence or
infer application accuracy. The previous shared-GPU registry remains intact.
