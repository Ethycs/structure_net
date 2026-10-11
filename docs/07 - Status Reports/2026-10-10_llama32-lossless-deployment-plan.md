# Exact deployment efficiency after user clarification

Date: 2026-10-10. Hypothesis `llama32-3b-lossless-deployment-efficiency-v1`.
Profile: EXPLORATORY, one frozen model/hardware configuration.

The user clarified the deployment goal: optimized tensorization of two regular
models for the same space and speed, or a faster model. Operationally, the
strong two-for-one target is two physically independent parameter sets and
requests, with combined peak resident GPU memory and paired completion latency
no greater than one ordinary dense model/request. Sharing weights between the
two copies cannot satisfy that target. The alternative is a faster exact single
model. Storage-only archive ratios do not pass either deployment gate.

Known before this plan: the complete original FP16 Llama archive passed its
bitwise round trip at 4,587,791,467 bytes versus 6,425,499,648 raw tensor bytes.
Two such archives already exceed one raw model's weight budget, before decoding
or activations. Factors plus FP32 residuals were previously larger and slower
than the dense model. The archival run subsequently stopped at a non-contiguous
factor-tensor case, after the original checkpoint completed; retain the failure
and partial evidence. No deployment claim follows from it.

First deployment intervention: CUDA graph replay of the existing complete
28-layer `TensorizedLlama`, with exactly the same original FP16 weights and
operations. This tests reduced dispatch overhead in the model-major tensor
implementation. It is not a new low-rank factorization, a trained residual
network, or a claim that tensorization reduces GEMM work. If it succeeds, the
claim is a faster execution configuration of one unchanged model.

Reference: frozen original checkpoint SHA-256
`f01cc0653ec6c9bd7026b8676864612b91b6c79d2dc4df42599cbe4cf99eb9a9`.
Physical GPU 0, RTX 3060, same attention implementation, FP16 weights and FP32
readout, two CPU threads, TF32 disabled. No other GPU job may be launched or
reconfigured. The independent CPU archival worker has terminated; no concurrent
experiment is scheduled during the deployment timing cells.

Run a tiny CUDA lifecycle pilot first, then a full-model pilot. The pilot is
systems evidence only. Fixed complete-model workloads: prefill 128; prefill 512;
prefill 128 plus 32 greedy decode steps with EOS early termination disabled.
Inputs come from the frozen `inputs_v2/fixtures.json` benchmark sequences. Use
two unequal input sequences per workload for bitwise correctness. For prefill,
compare logits and every returned KV tensor. For generation, compare every
generated token; also compare cached one-step logits/KV in the lifecycle pilot.
Graph replay must accept changed input IDs. Graph outputs are cloned so the
caller owns results and subsequent replays do not mutate previously returned
values; copying is included in timing.

For each full workload, warm each arm, then collect ten alternating-order
paired wall/CUDA-event samples with device synchronization. Report medians and
every sample/order. Eager and graph are alternative execution paths of the
same physical model, not two co-resident models. Measure eager peak allocation
before graph capture, then graph peak/capture cost separately. Retain capture
time and graph-pool memory; do not hide setup cost. Freeze/have content hashes
of every outcome-relevant source/configuration before measured execution.

Gates: any bit mismatch invalidates the exact candidate. A workload supports
the faster-single-model branch only if bitwise controls pass and eager/graph
wall median is at least 1.10, parameter bytes do not increase, and the complete
graph path fits this GPU. Report which workloads pass; do not imply improvement
for others. The two-for-one branch remains unsupported unless directly measured
with two independent resident models meeting both memory and latency bounds.
No accuracy reevaluation: preserving weights/arithmetic preserves the original
function and its earlier errors, without establishing the 95% application goal.

New roots: `data/experiments/llama_lossless_deployment/20261010_pilot_v1/` and
`20261010_v1/`. Record failures, strict JSON, full provenance, complete-model
paired results, conservative report/registry and read-back. Stop on CUDA
failure, bit mismatch, out-of-memory, source hash mismatch or GPU contention;
retain evidence and use a new amended root for implementation corrections.
