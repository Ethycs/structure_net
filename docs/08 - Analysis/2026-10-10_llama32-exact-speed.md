# Llama 3.2 3B: exact speed iteration

Date: 2026-10-10. Profile: EXPLORATORY.
Classification: **`no_qualifying_exact_speedup`**. One checkpoint and device; confirmation remains false.

Aggregate: `data/experiments/llama_exact_speed/20261010_v3/speed/campaign_results.json`, SHA-256 `6e8ba2182553beaf9f9eeb489b1c496a3cbc99edb643a0a7cbcdb0671b5517ff`.
[Prospective plan](../07%20-%20Status%20Reports/2026-10-10_llama32-exact-speed-plan.md).

This iteration compares five projection implementations against the already optimized CUDA graph baseline. Every arm uses the full 28-layer original FP16 model on GPU 0 (RTX 3060), with unchanged attention, MLPs and parameters. `mm` uses single-model 2D GEMM, `packed_bmm` combines Q/K/V and gate/up output rows, and `packed_mm` does both. `parallel2` and `parallel3` overlap independent, unchanged projections using two or three CUDA streams.

| Candidate | Prompt/generated | Exact controls | Baseline graph ms | Candidate graph ms | Speed ratio |
| --- | ---: | --- | ---: | ---: | ---: |
| mm | 128/0 | pass | 39.768 | 39.791 | 0.999× |
| mm | 512/0 | pass | 153.181 | 153.168 | 1.000× |
| mm | 128/32 | pass | 697.776 | 697.754 | 1.000× |
| packed_bmm | 128/0 | FAIL | unmeasured | rejected | — |
| packed_bmm | 512/0 | FAIL | unmeasured | rejected | — |
| packed_bmm | 128/32 | FAIL | unmeasured | rejected | — |
| packed_mm | 128/0 | FAIL | unmeasured | rejected | — |
| packed_mm | 512/0 | FAIL | unmeasured | rejected | — |
| packed_mm | 128/32 | FAIL | unmeasured | rejected | — |
| parallel2 | 128/0 | pass | 39.617 | 39.449 | 1.004× |
| parallel2 | 512/0 | pass | 152.724 | 148.192 | 1.031× |
| parallel2 | 128/32 | pass | 697.690 | 690.952 | 1.010× |
| parallel3 | 128/0 | pass | 39.901 | 39.585 | 1.008× |
| parallel3 | 512/0 | pass | 152.760 | 148.207 | 1.031× |
| parallel3 | 128/32 | pass | 697.775 | 688.528 | 1.013× |

Candidates passing all exactness cells: `['mm', 'parallel2', 'parallel3']`.
Promoted workload configurations (>=1.10×): `[]`.

Each exact cell uses ten alternating-order timing pairs after warmup, with device synchronization and wall/CUDA-event samples retained. Latency includes graph input copying and owned output cloning. Historical eager timings are not the baseline. A rejected candidate cannot be rescued by a faster or exact result on another shape.

Two unequal inputs per cell were checked. Prefill controls retain every output-logit and KV bit. Generation controls retain all 32 steps' logits, every token and the final KV cache, with candidate eager and graph output compared to the original reference. The separately timed token-only generation graphs were checked against those traces. Previously returned outputs were checked after a replay on changed input. Finalization independently reloads and compares the saved tensors.

The original checkpoint SHA-256 is `f01cc0653ec6c9bd7026b8676864612b91b6c79d2dc4df42599cbe4cf99eb9a9`. Every packed parameter slice was checked bitwise against its source. Parameter storage remains **6,425,499,648 bytes**. Reference projections become views into the packed allocation with their original shape and strides. Before packing, all six original reference outputs were saved; every later reference eager/graph output was checked against those immutable anchors. Both execution paths share physical weights, so paired peaks below are not the memory of two independent models.

| Candidate | Packing/setup s | Paired peak allocated GB (maximum exact cell) | Maximum candidate timing capture/warmup s |
| --- | ---: | ---: | ---: |
| mm | 0.001 | 6.646 | 7.553 |
| packed_bmm | 0.109 | unmeasured | unmeasured |
| packed_mm | 0.089 | unmeasured | unmeasured |
| parallel2 | 0.001 | 6.825 | 7.999 |
| parallel3 | 0.001 | 6.885 | 8.095 |

Capture/packing cost is excluded from steady-state timings and requires reuse to amortize. Graphs have fixed prompt and generation lengths. Each candidate/workload pair is released before the next. GPU process/memory snapshots passed the contention check; this does not measure transient external kernel activity. No concurrent experiment or heavy audit was scheduled during timing.

All fifteen primary cells reached a terminal result: 9 exact measurements and 6 numerical rejections. No retried or pooled primary cells. Pilots v1 and v2 stopped on out-of-memory failures at packed prefill 512, after observing packed-projection bit mismatches. The [first amendment](../07%20-%20Status%20Reports/2026-10-10_llama32-exact-speed-amendment.md) reduced audit buffer lifetime and added scheduling candidates. The [storage amendment](../07%20-%20Status%20Reports/2026-10-10_llama32-exact-speed-storage-amendment.md) shares packed buffers with the reference and anchors all comparisons to saved pre-packing outputs. Both preceded primary timing. Failed pilots are preserved; corrected pilot v3 contains systems evidence only. This report uses primary v3 only.

No training, tensor rank reduction, weight compression or application-accuracy improvement is claimed. Exactness is established on the tested inputs, not proved over every possible input/kernel dispatch. The prior 95% application target remains unmeasured. Two independent models within one model's storage/latency budget remain unsupported.

Reproduce using `experiments.structure_net.campaign_llama_cotenancy --exact-speed pilot` and then `--exact-speed full`, with separate new `--output` roots and the original `--control` root. Audit and register with `experiments.neural_architecture_lab.finalize_llama_exact_speed --campaign <full-root>/speed --register`.
