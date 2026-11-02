# Exact speed iteration: pilot memory and scheduling amendment

Date: 2026-10-10. Hypothesis remains `llama32-3b-exact-speed-iteration-v1`.
Exploratory amendment before any primary timing run.

Observed pilot v1: all three `mm` full-model workloads were bitwise exact,
with one-pair timing ratios approximately 1.000. `packed_bmm` at prefill 128
failed bitwise equality. Tiny generation also rejected both packed variants.
At packed prefill 512 the worker ran out of memory while retaining several
complete GPU output trees for the correctness audit. It stopped as specified.
Preserve that failure and all pilot artifacts; no v1 pilot timing is primary.

Change the untimed audit to move each completed output tree to CPU immediately
and compare saved bits there. Keep only the previous candidate return on GPU
for the ownership check; its immutable comparison copy is on CPU. Release
unused allocator blocks before checking outputs. Model execution, candidates,
inputs and exactness thresholds are unchanged by this memory fix.

Add two prospective candidates: `parallel2` and `parallel3`. They preserve
separate projection matrices and GEMM shapes, but schedule independent Q/K/V
and gate/up computations on respectively two or three CUDA streams, joining
before consumers use the results. This is a scheduling experiment, with the
same strict exactness gates; it is not presumed faster or exact.

The complete fixed primary grid is now five candidates times three workloads
(15 cells). Every candidate must pass all its cells to be eligible; a workload
still needs >=1.10 speed ratio to pass. No other endpoint or outcome meaning
changes. New pilot root `20261010_pilot_v2`; primary root `20261010_v2` under
`data/experiments/llama_exact_speed/`. Freeze all updated sources before each
worker starts. Audit the new pilot before launching the primary run.
