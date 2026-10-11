# Llama 3.2 3B: exact BLAS backend iteration

Date: 2026-10-10. Profile: EXPLORATORY. One checkpoint/device; confirmation remains false.
Classification: **`no_exact_candidate`**.

Aggregate `data/experiments/llama_exact_blas/20261010_v1/blas/campaign_results.json`; SHA-256 `5482034aaae033226cd9b084c8688b92578e6ec646fbed877d8d1ee54b101fa2`.
[Prospective plan](../07%20-%20Status%20Reports/2026-10-10_llama32-exact-blas-plan.md).

This follow-up prefers cuBLASLt for original dense projections using PyTorch's installed backend API, with batched GEMM or 2D GEMM. The control explicitly uses cuBLAS. This is a library preference that can fall back for unsupported operations; it does not establish which kernel algorithm the library selected. Backend state is restored after every projection, including on exceptions.

| Candidate | Prompt/generated | Bitwise controls | Baseline graph ms | Candidate graph ms | Speed ratio |
| --- | ---: | --- | ---: | ---: | ---: |
| cublaslt_bmm | 128/0 | FAIL | unmeasured | rejected | — |
| cublaslt_bmm | 512/0 | FAIL | unmeasured | rejected | — |
| cublaslt_bmm | 128/32 | FAIL | unmeasured | rejected | — |
| cublaslt_mm | 128/0 | FAIL | unmeasured | rejected | — |
| cublaslt_mm | 512/0 | FAIL | unmeasured | rejected | — |
| cublaslt_mm | 128/32 | FAIL | unmeasured | rejected | — |

Eligible candidates: `[]`. Promoted workloads: `[]`.

All six fixed cells completed: 0 exact measurements, 6 numerical rejections, no runtime failures or reused primary cells. A failed exactness cell vetoes the entire candidate. Only exact cells receive ten alternating timing pairs; a workload requires >=1.10× to pass the speed gate. The baseline is the current CUDA graph, not the slower eager implementation.

Complete original 28-layer FP16 Llama 3.2 3B, RTX 3060 physical GPU 0, two CPU threads, TF32 disabled, unchanged Flash Attention and all MLP layers. Checkpoint SHA-256 `f01cc0653ec6c9bd7026b8676864612b91b6c79d2dc4df42599cbe4cf99eb9a9`. Both paths share the same **6,425,499,648 parameter bytes**; no weight packing, compression or training.

Original eager anchors were saved before the intervention. Two unequal inputs per workload check every prefill logit/KV bit and all 32 generation-step logits, generated tokens and final KV cache. Candidate eager/graph and baseline eager/graph were compared to those anchors. Changed-input replay and output ownership were checked. All saved tensor controls, hashes, producer snapshots and timing medians were independently audited. The separate tiny/full-model pilot contains systems evidence only.

Timings include graph input copying and owned outputs, after warmup and with device synchronization. All wall/CUDA samples, capture/setup times and allocator peaks are in the aggregate. Setup is excluded from steady-state latency and requires reuse. Graphs have fixed shapes and generation lengths. GPU process/memory snapshots passed the contention check; transient external kernel activity is not measured.

Numerical differences are rejected even if generated tokens happen to agree. Bitwise checks on tested inputs do not prove equality for every possible input or library dispatch. No 95% application-accuracy result, two-for-one independent-model deployment, or universal speedup is claimed.

Reproduce with `experiments.structure_net.campaign_llama_cotenancy --exact-blas pilot` and `--exact-blas full`, separate new output roots and the original control root. Audit with `experiments.neural_architecture_lab.finalize_llama_exact_speed --backend --campaign <root>/blas --pilot-audit` for the pilot; use `--register` for the primary run.
