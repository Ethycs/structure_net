# MoBA for an attention-only application

Date: 2026-09-09. Status: application contract pending; kernel characterization
may proceed independently. This protocol follows the user's clarification:
“I use an application that uses just the attention heads without the rest of
the network.” The target is the application's final accuracy and computation,
not BLiMP or a full language model.

## Scope correction

The completed TinyLLM conversion campaign remains preserved negative evidence.
The all-weight BLiMP search was stopped and cannot support the requested claim.
The subsequently drafted attention-only TinyLLM BLiMP run was never launched;
its protocol is superseded for the active application task. No further full
language-model training or full Qwen checkpoint download is authorized by this
protocol. Whole-model memory estimates are irrelevant to a heads-only workload.

The user has been asked for the application entry point, evaluation data/command,
and definition of final accuracy. Do not invent an accuracy proxy or call output
cosine similarity, routing recall, attention sparsity or BLiMP accuracy the
application's accuracy. The 95% application target cannot be tested until that
contract is available. Preserve the user's actual input shapes, mask semantics,
precision, causal/noncausal behavior and any required attention projections.

## Paper alignment

Reference: [MoBA technical report](https://arxiv.org/pdf/2502.13189), §§2–3.
The method selects blocks with query/mean-key affinity, includes the current
block, enforces causality, computes attention only over selected tokens, and
normalizes across the complete selected set. The optimized implementation
groups queries by selected block and combines FlashAttention outputs using
online softmax. The paper's 95.31% figure is sparsity, not task accuracy.

Paper timing claims concern long-context attention, with optimized full
attention as the baseline. They do not prove this application's accuracy or
speedup, and do not establish that frozen-MLP adaptation suffices. No MLP is
needed in the application workload described by the user.

Official source: `https://github.com/MoonshotAI/MoBA`, pinned commit
`b5d58363311d3ca946f1ec444182727c15e338b5`, locally under
`data/experiments/moba_paper/upstream/`. Its pinned FlashAttention is 2.6.3.
Dependencies are isolated under `data/experiments/moba_paper/environment/`;
the shared environment is unchanged. PEFT/quantization dependencies begun
before the application clarification are not needed for head-only benchmarks.

## Kernel characterization before application execution

Use CUDA on the available RTX 3060, one worker. Never call a dense score matrix
with a sparse mask a compute-saving implementation. Compare:

1. PyTorch's optimized full SDPA (and FlashAttention when available).
2. The paper's optimized MoBA kernel, including routing/reordering/combining.
3. A separately labeled experimental Triton k=2 kernel, including pooling and
   routing. Its inference-only implementation cannot support training-speed
   claims.

Measure the same shapes and precision for every comparison; report FP32 and
FP16 separately. Include 256-token TinyLLM-shaped and longer standalone heads,
then the application's actual workload. Qwen3-8B-shaped heads use query-head
count 32, KV-head count 8 and dimension 128, but synthetic head tensors are not
measurements of a loaded Qwen model or its task accuracy. Include GQA expansion
cost if a backend requires it. Do not load the rest of Qwen to time heads.

Before timing, test output parity against a dense reference implementing the
same selection/softmax rule, causal invariance, tail blocks, stable ties and
GQA. Full-budget MoBA must agree with full attention. Check output and backward
parity for any backend used for training. Preserve actual numerical tolerances,
failures and exclusions. Benchmark only valid cases.

Warm up compilation and allocations. Collect repeated paired CUDA-event
timings in alternating order and retain the samples. Report median and spread,
both ordinary calls and CUDA graphs where supported, with environment and GPU
identity. Include all routing/layout/conversion costs. Exclude unrelated GPU
workers from the measured device and keep shapes/precision fixed. Preserve
source hashes and results in a new append-only artifact root. A fast synthetic
case cannot establish a speedup for an unavailable application.

The initial unregistered exploratory k=2 pilot at (B,H,T,D)=(1,8,256,64) matched
routes and had FP32 maximum output error 7.15e-7. It measured 0.0562 ms sparse
versus 0.0712 ms full SDPA in FP32, but 0.0564 ms versus 0.0168 ms in FP16.
These single timing aggregates motivated paired verification and the official
backend comparison; neither establishes the application objective.

## Application completion gate

With the real application contract, evaluate untouched final examples and
measure its attention pipeline end-to-end. Success requires final accuracy
>=95% under its established scoring and less computation/latency against
optimized full attention under matched conditions. Report any original quality
guard that remains meaningful to the application; do not apply language-model
NLL to an application without an LM objective. Adapt only permitted attention
parameters, and verify unrelated tensors remain unchanged. Use independent
training/selection/final splits and preserve failed candidates.
