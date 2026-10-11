# Exact speed iteration: BLAS backend selection

Date: 2026-10-10. EXPLORATORY.
Hypothesis: `llama32-3b-exact-blas-iteration-v1`.

Before this plan, the projection pilot found unchanged 2D GEMM effectively
tied with the graph baseline, packed projections numerically inexact, and
two/three-stream scheduling gains around 1–3%. The primary projection run is
still executing. Its results cannot retroactively change these endpoints.

Test whether preferring the installed cuBLASLt backend for original dense
projections produces >=1.10× speedup over the original cuBLAS graph with
bitwise output equality. Two candidates: `cublaslt_bmm` and `cublaslt_mm`.
Use PyTorch 2.4.1's installed `preferred_blas_library` API, restoring the prior
backend after each projection, including on exceptions. The API is a backend
preference and may fall back; report it as such. No kernel implementation or
unverified claim of a particular selected algorithm.

All other controls match the preceding exact-speed study: complete 28-layer
FP16 Llama 3.2 3B, original checkpoint SHA-256
`f01cc0653ec6c9bd7026b8676864612b91b6c79d2dc4df42599cbe4cf99eb9a9`,
physical GPU 0, two CPU threads, TF32 disabled, unchanged Flash Attention,
norms, MLP activations and weights. No packing or duplicated weight storage.
Do not run alongside the preceding experiment or a heavy CPU audit.

Fixed grid: each candidate at 128/0, 512/0 and 128/32 prompt/generated tokens,
with EOS termination disabled. Two unequal frozen benchmark inputs, original
pre-intervention anchors, complete prefill logits/KV and generation traces
(all step logits, all tokens, final KV). Check eager and graph against anchors,
changed-input replay and output ownership. Any changed bit rejects that cell;
any rejected cell makes the whole candidate ineligible. Preserve all rejects.

Run focused CPU contracts, tiny CUDA (seed 779, lifecycle only) and full-model
memory/exactness pilot first. Audit pilot artifacts before primary launch.
Ten alternating-order paired timings after warmup for exact cells only; same
current cuBLAS graph control, copied inputs and owned outputs. Record all
wall/CUDA samples, setup, capture and allocator peaks. Numerical rejection is
not an infrastructure retry. Runtime/OOM/contention/hash failure stops the job
and requires a new amended root. No selection of favorable timing retries.

Workload promotion: candidate passes all three exactness cells and the local
baseline/candidate wall median is >=1.10. Otherwise retain the existing
configuration. Confirmation is false for this exploratory one-checkpoint
study. Report accuracy only as tested bitwise preservation, with no claim of
95% application accuracy, compression or two models in one model's budget.

Roots: `data/experiments/llama_exact_blas/20261010_pilot_v1/` and
`20261010_v1/`. Snapshot/hash all producers and this plan, strict JSON, retained
raw controls, audited aggregate/report, conservative typed registry with
exact/semantic readback and commit. Launcher option: `--exact-blas pilot/full`.
