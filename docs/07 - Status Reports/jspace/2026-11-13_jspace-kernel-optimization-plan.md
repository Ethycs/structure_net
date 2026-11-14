# Faster compact Qwen structures with joint recovery

Date: 2026-11-13. Status: prospective exploratory design; no new GPU evidence.
The intervention is **a faster compact structure plus recovery training**.
Bitwise preservation of the old compressed checkpoint is not the objective.
Implementation correctness and post-recovery task quality are separate gates.
This document does not amend any frozen producer or register a new study.

Known evidence is `jspace_qwen_affine/20261113_v2`: the Gröbner final block scored
388/512 versus 393/512 native, with 10.04 MB additional storage saving versus its
matched non-Gröbner arm. Its isolated eager prefill was 4.724 ms versus native
3.405 ms initially and 3.057 ms at the end. Whole-model CUDA-graph generation was
40.87 tok/s versus native 40.51–41.09. These measurements do not identify which
operations caused the local slowdown or show a whole-model speed improvement.

## Audit findings and candidate structures

`AffineQwenBlock` runs FP32 entry/exit basis GEMMs with TF32 disabled, two reduced
RMS reductions, large gathered affine-shift epilogues, explicit attention, and
separate polynomial gather/normalization/product/concatenation operations.
For rank 1536, the two basis maps perform 6,291,456 multiply-accumulates per
token. The Gröbner gate/up epilogue spans 19,968 channels: at 384 tokens, just one
FP32 materialization is about 30.7 MB. The packed output projection has
9,984 retained channels + 325 features = **10,309 input columns**. Its odd
contraction dimension and row stride may impair the selected GEMM kernel;
the profiler must establish this rather than assuming a fallback.

Use the following small attribution matrix on development workloads before
spending a full recovery budget. Keep the same learned coefficients initially.

| Variant | Purpose |
| --- | --- |
| Native suffix and old compact suffix | Contemporaneous quality/runtime controls; same number of blocks |
| Padding only | Zero-pad 10,309 to 10,368 columns; fixed zeros in weight and feature padding |
| Fusion only | Fuse RMS/shift epilogues, staged SwiGLU, polynomial feature packing; original geometry precision |
| Padding + fusion | Measure interaction without a geometry precision change |
| Direct reduced stack, FP32 geometry | Encode once, use fixed affine bridges, reconstruct once |
| Direct stack + padding/fusion, TF32 geometry | Explicit arithmetic approximation with FP32 stored geometry |
| Direct stack + padding/fusion, BF16 geometry | Explicit approximation; BF16 runtime geometry replaces FP32 copies |

Padding to 10,368 adds 59 columns, or 181,248 BF16 weight bytes at rank 1536.
Count this overhead and write padded features directly, avoiding a second
padding/copy kernel. Do not infer a speed benefit from alignment alone.

The existing `AffineQwenStack` already folds an intermediate native
reconstruction/encoding pair into `B=U_next.T @ U_prev` plus its fixed affine
shift. For two equal-rank blocks, coordinate-map work changes from `4*d*r` to
`2*d*r+r*r` multiply-accumulates per token: 31.25% less map work at d=2048,
r=1536. This is not a 31.25% whole-block speed prediction. Removing the native
BF16 materialization changes rounding even with FP32 bridges. A bridge cannot
simply disappear into incoming projections while ignoring its effects on RMS
denominators and the carried residual. If bridge cost remains dominant, a shared
basis across adjacent blocks is a separate structural candidate requiring fresh
calibration and joint recovery, not an algebraic simplification of the old model.

BF16 geometry changes the represented subspace and its approximate orthogonality;
TF32 changes multiplication precision. Declare both explicitly and measure their
initial defects. Keep FP32 RMS accumulation and the original width-2048
denominator initially. Store original geometry in external fit artifacts for
diagnostics, not as duplicate resident buffers counted out of deployment size.

## Performance-first selection and recovery order

1. Profile the existing frozen final-block checkpoint and the native block.
   Use its eight real development prompts and a shape grid; inspect no new
   held-out labels. Check component/reference correctness before timing.
2. Measure padding-only, fusion-only and their combination. Then benchmark the
   direct two-block stack and the two declared geometry approximations on
   development inputs. Record initial development KL/CE, but do not demand exact
   old-checkpoint answers from a structural approximation.
3. Choose one faster structural candidate using development workload wall time,
   memory and cache correctness; retain the FP32 compact control. Freeze the
   choice and all runtime settings before fresh quality evaluation. Avoid a full
   cross-product of ranks, precision modes and compiler modes initially.
4. Follow staged output distillation: original teacher → recover final block;
   freeze that recovered model as the next teacher; add the preceding compressed
   block and **jointly recover both compact blocks** through the fixed coordinate
   bridge. Freeze the native prefix. Keep the matched output-KL+CE baseline;
   auxiliary local distillation remains a separately labelled comparison.
5. Use the same training identities, teacher snapshot, data order and budget for
   the matched structural controls. Start with the existing 12-epoch, 512-example
   budget and development-only checkpoint selection. Train through a
   differentiable implementation of the **deployed candidate's** geometry and
   rounding rules. An inference-only Triton kernel must not sever gradients.
   Validate a packed deployed copy during selection and after strict reload.
6. Measure frozen final candidates on held-out quality and matched whole-model
   timing. Grow the suffix beyond two blocks only as a subsequent recorded step;
   a two-block result does not establish all-layer recovery or speed.

No CPU compilation or unrelated GPU job should overlap timed measurements.
Launch only after the active distillation lease ends, using a new reservation,
new artifact root and the locked `ncm` Pixi environment. CPU preparation itself
does not acquire a GPU or alter the active experiment.

## Implementation fidelity versus approximation

Each structural candidate gets an explicit unfused reference with the same
geometry dtype, normalization, head scale, masks, caches and polynomial ordering.
Test the fused implementation against that reference, rather than rejecting a
new architecture because it differs from the old checkpoint. Check FP64 algebra
on tiny models, FP32 reduction behavior, BF16 boundaries, finite values, cache
positions 0 and 415, invalid-position rejection, and fresh-process reload.

The reference currently rounds the linear output to BF16, then adds the shift
and divides in FP32, then rounds gate/up to BF16; SiLU rounds to BF16 before the
BF16 multiplication by up. Polynomial anchors are standardized in FP32 and
quadratic features round to BF16 before the packed GEMM. A fusion that removes
those intermediate roundings or uses reciprocal-square-root multiplication
instead of sqrt/division must declare that arithmetic change. It may be a useful
recovered candidate; it is not automatically a faithful implementation of the
old arithmetic. Precision tests should report exact-element fraction, max error,
relative RMS error, and full-output-logit KL. For same-arithmetic fusion, require
full-block relative RMS error ≤0.002 and mean full-vocabulary KL ≤1e-5 on the
fixed development checks; a miss stops that implementation arm until explained.
These are numerical engineering checks, not the structural quality budget.

Keep Qwen's original scale `1/sqrt(128)` with 64-wide heads. Preserve all causal
positions and the original GQA mapping. For cached one-token decoding, use the
correct absolute-position mask; a naive square causal mask is insufficient.
The pinned PyTorch 2.4.1 SDPA has `scale` but no `enable_gqa` argument. If an
attention-kernel comparison is added, implement head grouping explicitly or
retain expansion and apply the same optimized attention backend to native and
compact controls. [Pinned SDPA source](https://raw.githubusercontent.com/pytorch/pytorch/v2.4.1/torch/nn/functional.py).

One optional compiler comparator is
`torch.compile(fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")`
on the pure block/reduced function, with fixed shape buckets. Record graph
breaks, recompiles, compile time, padding decisions and workspace bytes. Apply
manual CUDA-graph replay equally afterward; do not count a new graph wrapper as
a structural speed gain. Keep compilation outside measured steady state and
report its break-even separately. [Pinned compiler source](https://raw.githubusercontent.com/pytorch/pytorch/v2.4.1/torch/__init__.py).

## Timing, quality endpoints and decision rules

Profile separate annotated regions: entry basis; input RMS/QKV epilogue; RoPE and
cache writes; attention; attention output/residual; post-RMS/gate-up epilogue;
SwiGLU/features/packing; MLP output/residual; bridge; exit basis. Save CPU self
time, CUDA self time, kernel counts/names/shapes/dtypes, allocation/copy events
and a timeline. In particular, identify SGEMM versus Tensor Core kernels for
geometry and the 10,309/10,368-column GEMM. Profiling is a separate diagnostic
pass: profiler timings are not the reported speed benchmark. Sum of kernel time
and elapsed wall time need not differ by pure Python overhead; inspect GPU idle
gaps and stream overlap before attributing the difference.

Measure batch-one prefill at 32/64/128/256/384 tokens, plus the original eight
real development prompts. Mark resampled hidden-state shape probes as synthetic.
Measure cached single-token decoding at prefix lengths 32/128/256/384, both eager
and CUDA-graph replay. Use a prefilled static cache at a fixed absolute position
for repeated operator measurements; do not let a DynamicCache grow each repeat.
For whole-model execution use real prompts and 32 sequential greedy tokens,
including dynamic prefill, cache conversion/reset and graph launch in wall time.

Use 20 warmup calls and at least 50 prefill/200 decode calls per condition,
three fresh process runs, randomized paired condition order, and native controls
at the beginning and end. Report synchronized wall time and CUDA-event time,
median/p95 latency, prefill tok/s, decode tok/s, end-to-end tok/s, and all resident
weights/buffers/cache/workspaces. Bootstrap paired workload/process measurements,
not individual GPU kernels, for descriptive 95% speed-ratio intervals. Preserve
and repeat an invalid timing attempt if baseline drift exceeds 5% or another
GPU workload intrudes. Do not repeat a quality miss until it passes.

Before final launch, freeze a balanced fresh holdout, preferably 1,024 questions
(256/family), excluding every prior train/dev/eval identity. If fewer remain,
fix the available balanced count before measuring labels and require at least
512 for this quality screen. Otherwise label the result a reused-pool pilot.
The old 512 examples remain a continuity comparison, not a fresh test.

The engineering quality budget is **at most 1.0 percentage point below native**
and **at most 0.5 point below the matched unoptimized compact control** on the
same held-out questions. This accepts the scale of the currently observed
~1-point cost; it does not promise statistical equivalence. Report integer
gains/losses, answer agreement, per-family accuracy, paired confidence intervals
and output KL; do not infer equivalence merely from a nonsignificant test.

A useful local speed target is ≥1.10× versus both native and old compact suffixes,
with the descriptive speed interval above 1.0. Whole-model speed is a separate
endpoint: call it improved only if its matched interval is above 1.0. A local
win with indistinguishable whole-model timing is subsystem progress, not a
whole-model speed win. Require no >2% whole-model regression for advancement.
Count all precision, padding, cache and workspace storage; require a genuine
total resident-state reduction for a compression claim.

Freeze source hashes, parent checkpoint/fit hashes, model revision, split
identities, precision/TF32 settings, GPU/software versions, compiler/kernel
configuration and reservation ID. Preserve every attempted arm, its before/after
recovery quality and timing, failed controls and excluded measurements. Final
classification remains exploratory; no new registry entry is created by this
design note.
