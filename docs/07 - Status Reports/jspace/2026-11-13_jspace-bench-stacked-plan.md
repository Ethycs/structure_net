# Dense stacking of the compact polynomial core and residual

Date: 2026-11-13. Exploratory engineering extension of JSPACE-BENCH-TENSOR-01.
Root: `data/experiments/jspace_bench_tensor/20261113_stacked_v1`.

The user accepts the small accuracy loss on the restricted subset and requests
speed through a compact, stacked basis. Freeze the same 73 covered channels
(20/34/19 in blocks 8/17/26), shared coefficients, anchors and residual channels.
The preceding implementation scored 209/256 against original 210/256 and ran
about 2.5% slower. These are known engineering starting points, not a reason to
stop research. The earlier decision rules and results remain historical records.
This extension measures the implementation directly without a new accuracy cutoff.

Keep the polynomial coefficient block factored. Stack its reconstructed channels
with the retained nonlinear channels, giving one contiguous activation vector
and one dense output projection. Stack gate/up weight matrices into one GEMM,
pad retained widths to multiples of 16, and fuse SwiGLU plus polynomial packing
in a Triton kernel. There is one compact branch per affected layer; sequential
layers are not falsely treated as independent simultaneous matrix products.

The polynomial contraction uses FP32 and writes BF16 into the packed activation.
The output projection uses BF16 original columns, reordered consistently with
the stack. This changes rounding relative to separate FP32 correction GEMMs;
measure it. Deleted gate/up rows are never evaluated. Alignment adds zero rows,
not the deleted nonlinear functions. Shared generators are stored once per layer.

Controls: unchanged original; previous factorized implementation; stacked dense
original; stacked polynomial core with residual. Both stacked models apply the
same generic gate/up and activation optimization to all 36 MLP blocks. This
distinguishes generic implementation speed from the incremental compression gain.
Attention remains eager and the checkpoint/benchmark/precision are unchanged.

Before model measurements, require CPU exact-algebra/value/gradient packing tests,
then a reserved-GPU numerical shakedown for one-token and 37-token inputs, both
with and without polynomial channels. Fused/reference output RMS error must be
below 1%; nonfinite output stops execution. This is a numerical systems check,
not a task-quality criterion. Preserve failed attempts under their source hashes.

Score all 256 existing held-out questions for all four arms. Time the same eight
prompts and 32 generated tokens, including prefill and a warmup; bracket the run
with original-model timings. Also measure isolated selected MLPs at one and 128
tokens with ten warmups and 100 repetitions using CUDA events. Save per-question
predictions and the three replacement-module states. Unchanged stacked layers
are reconstructed from the pinned original checkpoint. No training, threshold
search, or held-out selection is performed.

Use root locked Pixi and an inference-ctl RTX 3060 reservation of 10,240 MiB.
Freeze this protocol and producing source before launch. Record all accuracies,
rates, tensor bytes, original timing drift and both speed comparisons; keep
`confirmed=false` because this is a single-checkpoint exploratory optimization.
