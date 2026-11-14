# Qwen adaptation of the sequential quadratic-attention recipe

Study: `JSPACE-QWEN-QUADRATIC-01`. Prospective exploratory protocol, 2026-11-14.
The user explicitly requested the supplied MNIST recipe adapted to Qwen, trained
and benchmarked. This is a separate campaign; the running all-block softmax plus
polynomial-MLP experiment remains unchanged.

## Architecture and fixed teacher

Use the pinned Qwen2.5-3B-Instruct checkpoint, all 36 blocks, native residual
width 2,048, 16 query heads and two KV heads. Reduce residual width to 1,024 and
head width from 128 to 64. Preserve the native score scale `1/sqrt(128)`, GQA,
causality, and complete retained RoPE frequency pairs. Keep every block's full
11,008-channel SwiGLU width. Do not use polynomial MLP replacements or a
hyperdeterminant regularizer.

Build an explicit block-Krylov space `[B, C B]` and Rayleigh–Ritz basis from
each native task-Jacobian Gramian, with seeded 1,024-column B. The existing
frozen native capture uses the derivative of correct-choice logit minus mean
other-choice logit, sampled at 16 positions in each of 128 calibration cases.
This is a supervised task-margin Jacobian, not a full-vocabulary Jacobian or
proof of identified semantic CAVs. Report numerical rank, retained energy,
Ritz residuals, and any completion directions unsupported by the operator.
The native capture's uncentered value moments supply the 64-dimensional V
projections; task sensitivity selects complete Q/K rotary pairs.

Use one global affine anchor per block from the frozen calibration means.
Carry reduced coordinates through direct precomputed affine bridges. Fold the
input embedding into the first basis and initialize the compact readout by
folding the native embedding/readout, native final normalization gain, and
last basis. Its trainable numerator has shape `[151936, 1024]` plus a vocabulary
bias. The fixed RMS denominator is computed from the last affine reconstruction
using its Gram matrix, anchor cross term, and anchor energy. This avoids native
residual reconstruction at either endpoint. The two reduced vocabulary tables
together have the same number of entries as the original tied full-width table;
all extra biases, geometry and buffers count toward deployed tensor bytes.

Train a positive, shared-per-KV-group, rotary-pair metric initialized to identity.
It commutes with RoPE and is folded into Q/K projection weights, offsets, shifts
and biases at export. It is a learned functional metric, without an additional
claim of semantic concept identification. The matching softmax arm uses the
same parameterization and folding.

The quadratic arm evaluates `p(s)=1+s+s*s/2` directly and normalizes each row.
Apply causal/padding exclusion to the polynomial weights, never by squaring
negative-infinity masked logits. Retain all K/V context. At inference the final
block computes only the query outputs needed for next-token prediction; apply
the same optimization to the dense and compressed-softmax controls.

## Sequential training

Reuse the verified original native BF16 boundary/logit cache, 512 training and
64 development cases from ARC-Challenge, BoolQ, HellaSwag and WinoGrande. The
teacher's weights and cached targets never change. Each arm has its own frozen
student-prefix coordinate bank. Layer i is trained on that student's actual
prefix and targets the original teacher's boundary i+1 projected into U_i.
Previous blocks are frozen and absent from the optimizer.

Fit layers 0–34 for 12 epochs with projected-coordinate feature MSE, normalized
by a fixed train-only target-energy scalar. Fit layer 35 and the compact readout
for 12 epochs with normalized projected-feature MSE plus `T*T*KL`, T=2. The final
loss uses the four positions for which full-vocabulary teacher logits were
cached; earlier layers use all unpadded positions. No CE term or auxiliary
gradient-ratio tuning is added. Use FP32 training, microbatch 4, accumulation 4,
AdamW with zero weight decay, learning rate 1.2e-4 and cosine decay, gradient
clipping at 1. Development objective chooses a checkpoint, including epoch zero;
there is no quality early stop. Fold it, verify fold parity, freeze it, and
stream the next student-prefix bank. CPU-streamed optimizer state bounds GPU
memory. All GPU work uses a separate inference-ctl lease per phase.

## Evaluation and benchmarking

Use the same fixed, previously untouched 512-case evaluation pool as the active
all-block experiment. This shared comparison was chosen before seeing its test
results. Neither arm may train or select checkpoints on these cases. Report
dense, compressed-softmax and quadratic accuracies, per-family counts, paired
wins/losses, and the quadratic-versus-softmax difference. The user's aggregate
quality allowance remains five percentage points below the dense teacher.
Report the existing 38% tensor-reduction and 1.4x whole-model-speed targets;
failure to meet them is a measured result, not a reason to hide an arm.

Primary recipe timing uses FP32, static shapes, one CPU thread, full-graph
torch.compile with freezing/constant folding, warmed execution, and compilation
time reported separately. Compare matched dense and compressed implementations
on identical prompts/shapes. Record eager and compiled controls, compilation
failures without silent fallback, output differences and model tensor bytes.
Use batch 1 for Qwen rather than transplanting MNIST's batch 256. Prefill latency
is reported as prefill latency; it must not be relabeled decoding tok/s.
Use two development-prompt prefixes at each static length 128 and 256, with
five measured repetitions after warmup. The speed gate is the smaller of the
two matched compiled speedups, preventing selection of a favorable shape.
Held-out quality is measured by FP32 layer-streamed GPU evaluation; CPU export
parity covers the timed development prefixes. Do not claim that all 512 cases
were scored by the compiled CPU backend.
GPU serving measurements, including autoregressive decoding, are a separate
hardware/precision result and require their own output-parity/accuracy checks.
Serialize final whole-model timing with the prior campaign's timing/closeout;
do not benchmark while another experiment competes for the same resources.

Before the main fit, run CPU algebra/causal/gradient/compile tests and a reserved
full-width final-layer training-memory pilot. A systems pilot is not quality
evidence. Freeze all outcome-relevant sources and input identities in a new
artifact root. Keep raw completed results append-only, record failures, use
locked offline Pixi, and register the completed exploratory result with
`confirmed=false`. Derived evaluation scratch banks may be removed after their
consumer completes; retain training checkpoints, prefix banks, final prediction
records, timing samples, configuration and source/input hashes.
