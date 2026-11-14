# Compact shared basis and dense execution notes

Date: 2026-11-13, America/Los_Angeles.
Study: `JSPACE-BENCH-TENSOR-01`. Model: Qwen2.5-3B-Instruct, BF16, RTX 3060.

Stacking the compact polynomial component with the retained nonlinear residual
made the affected layers approximately 2.5× faster than the separate polynomial
branch. CUDA-graph decoding then raised whole-model throughput from 12.41 to
41.46 tokens/s on the same packed model, with no further loss on the measured
multiple-choice task accuracy.

| Measurement | Recorded result |
| --- | --- |
| Affected MLP latency, one token | About 1.04 ms → 0.413 ms, approximately 2.52× faster |
| Packed model, standard generation → CUDA graph | 12.4146 → 41.4632 tok/s, 3.3399× faster |
| Original model accuracy | 210/256, 82.03% |
| Packed model, before and after runtime optimization | 209/256, 81.64%; all 256 task predictions and choice logits unchanged by the final runtime |
| Covered nonlinear channels | 73: 20/34/19 in blocks 8/17/26 |
| Resident tensor bytes saved versus original | 469,464 bytes, approximately 0.47 MB |
| Graph setup/capture | 2.42 seconds, amortized across requests |

The timing uses batch one, eight prompts, and 32 generated tokens per prompt,
including prefill. The graph has a fixed 416-position capacity. Its large speedup
comes principally from reducing execution overhead; it is not a 3.34× speedup
attributable to basis compression alone. Generated text is not bit-identical to
ordinary HF decoding: 134/256 generated token positions match. The preserved
quality measurement is the four-family multiple-choice benchmark.

## Implementation retained

The task-Jacobian chart supplies anchors for shared polynomial graph relations.
Gröbner substitution supplies the compact polynomial coefficient block. Its outputs
are stacked with the uncovered nonlinear channels into one dense output projection.
Gate/up projections are also stacked, with a fused activation-and-packing kernel.
The deleted gate/up functions are not evaluated. The residual continues to compute
the uncovered part of the model.

The first fixed-cache prefill variant lost three additional answers. Keeping
original dynamic prefill and copying its exact BF16 KV state into the static
decode cache restored 209/256 while retaining the speedup. The preceding variant
and all original benchmark results remain preserved.

## Cross-task basis reuse

The shared library contains 2,936 distinct qualifying generators across the three
blocks, accounting for 3,615 task uses. The 679 repeated uses represent **18.78%**
cross-task reuse. Of these generators, 445 qualify on at least two tasks and 73
on all four. Qualification means that the same expression achieves development
reconstruction NRMSE ≤0.5 on the counted task; it is an approximate neural reuse
measurement, not an ideal-dimension or accuracy measurement.

## Research direction to preserve

The user accepts the tiny accuracy loss on this restricted subset and wants the
compact basis made fast, then developed further. The intended architecture is
**compressed component plus residual**. The compressed component need not replace
an entire network function independently. Early deployment thresholds describe
particular prototypes; they should not be treated as a verdict against this
emerging research direction.

Continue tracking cross-task redundancy, coverage, residual work and combined
task quality as the shared basis grows. Keep the mathematical contribution and
the implementation speedup separately attributable.

## Evidence and log

- [Full measured report](../../08%20-%20Analysis/jspace/2026-11-13_jspace-bench-tensor.md).
- [Research log](research-log.jsonl), entry `2026-11-13-jspace-packed-core-recorded`.
- [Final runtime aggregate](../../../data/experiments/jspace_bench_tensor/20261113_graph_v3/campaign_results.json), SHA-256 `6ca5d8ab006bc18f64db441811fca7f8e36d0b993b7f0019e6a6dedc3879ccff`.
- [Finalization and provenance](../../../data/experiments/jspace_bench_tensor/20261113_stacked_v1/finalization.json).
- [NAL registry record](../../../data/hypothesis_registry/jspace_bench_tensor_20261113.json), result ID `fe24f7aa5500ff8f`, verified from a fresh process.
- Implementation commit: `168aaf94608d5b45d0b632daa63cbe50985168e6`. Twenty-two focused CPU test cases passed, plus GPU numerical and generation checks.

This note logs the existing experiment; it adds no new experimental replicate.
