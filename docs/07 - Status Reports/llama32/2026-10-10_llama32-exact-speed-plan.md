# Exact speed iteration against CUDA graph replay

Date: 2026-10-10. Profile: EXPLORATORY.
Hypothesis: `llama32-3b-exact-speed-iteration-v1`.

Question: can projection packing or single-model GEMM dispatch reduce latency
by at least 10% relative to the existing CUDA graph implementation while
preserving every tested output bit and unchanged parameter bytes?

Known before this plan: the completed exact-deployment study measured graph
latencies 39.264 ms (prefill 128), 152.207 ms (prefill 512), and 697.673 ms
(128 prompt tokens plus 32 generated tokens). Those historical numbers are
context only. Every new comparison uses a freshly measured graph control.

Fixed candidates, in order: `mm` (single-model 2D GEMM instead of batched GEMM),
`packed_bmm` (concatenate Q/K/V output rows and gate/up output rows, using
batched GEMM), `packed_mm` (both changes). All use the unchanged full 28-layer
FP16 Llama 3.2 3B weights, original Flash Attention, norms and nonlinearities.
No training, approximation, quantization, or tolerance-based promotion.
Packing must reconstruct every original parameter bit and preserve total
parameter bytes. This is weight layout optimization, not compression.

Original checkpoint SHA-256:
`f01cc0653ec6c9bd7026b8676864612b91b6c79d2dc4df42599cbe4cf99eb9a9`.
Fixtures are the frozen `inputs_v2/fixtures.json` from the co-tenancy study.
Use the corrected unequal pair: first length tokens of benchmark sequence 0
and last length tokens of sequence 1. Final application tests remain sealed.
Physical GPU 0 only, RTX 3060, two CPU threads, TF32 disabled. Other services
are left unchanged. No simultaneous experiment worker or heavy CPU audit.

First run CPU contracts, then a tiny CUDA lifecycle pilot and representative
complete-model pilot. Tiny seed 779 is only a systems seed, not an independent
trained replicate. Pilot checks all candidates, changed inputs and output
ownership. A numerical mismatch is a scientific rejection, not a retryable
infrastructure error: preserve it and continue independent candidates. Runtime,
OOM, fingerprint or contention failures stop the worker. Fixes need an amended
new root. Pilot timings never support speed claims.

Full grid: each candidate at prefill 128, prefill 512, and prefill 128 plus
32 greedy tokens (no EOS early exit). Bitwise controls compare all returned
prefill logits and KV tensors; for generation, a separate untimed trace
compares every step's logits, every generated token and the final KV cache.
Control and candidate graphs must each equal their eager execution, accept
changed token IDs, and return owned outputs that survive later replays.
Save expected/actual CPU tensors in hashed artifacts for independent readback.
Rejected candidates receive no promotion; later workloads remain informative
diagnostics. Do not hide or average away failed exactness gates.

For exact cells, warm both graph arms, then collect ten alternating-order
paired wall/CUDA-event samples. Include input copies and owned output clones.
The two arms share unchanged weights where possible; packed weights have a
temporary separate allocation so both can coexist for fair paired timing.
Report that paired-test peak separately from single-candidate parameter bytes;
it is not a two-independent-model memory benchmark. Measure setup/packing,
capture and allocation costs. Keep one candidate/workload pair at a time.

A candidate is eligible only if every cell's exactness and parameter gates
pass. Workload promotion additionally requires baseline/candidate wall median
>=1.10. Report regressions; never deploy a slower shape unconditionally.
Classification is exact speedup supported, no qualifying speedup, or no exact
candidate, scoped to this checkpoint/device/input grid. Confirmation remains
false. No claim of 95% application accuracy or two-for-one model storage.

Roots: `data/experiments/llama_exact_speed/20261010_pilot_v1/` and
`20261010_v1/`. Producer snapshots and hashes include all executable inputs.
Strict JSON, failed and rejected evidence, audited aggregate, measured report,
typed conservative registry with exact/semantic readback, and durable commit
are required before closing this iteration.

Reproduce with the co-tenancy launcher `--exact-speed pilot` then
`--exact-speed full`, a new `--output`, and the original `--control` root.
