# Llama 3.2 3B: exact deployment efficiency

Date: 2026-10-10. Status: completed. Profile: **EXPLORATORY**. Hypothesis `llama32-3b-lossless-deployment-efficiency-v1`.
Classification: **`exact_execution_speedup_supported_no_two_for_one`**. Confirmation remains false for this one-checkpoint exploratory comparison.

Aggregate `data/experiments/llama_lossless_deployment/20261010_v2/deployment/campaign_results.json`; SHA-256 `8265a6cc920d1551f811d17da2e816614aa37816de37251b01b025f12be7939c`.
[Fixed deployment plan](../07%20-%20Status%20Reports/2026-10-10_llama32-lossless-deployment-plan.md).

CUDA graph replay speeds up some workloads of the complete unchanged 28-layer FP16 model. All measured paired outputs match bit for bit. The gain comes from reducing execution dispatch overhead; it does not establish low-rank tensor compression, a smaller residual network, fewer matrix operations, or two independent models in one model's memory budget.

## Paired latency

Ten alternating-order samples after warmup, one RTX 3060, two CPU threads. Wall medians include input copying and owned output cloning for graph replay. Eager and graph share the same one physical model as alternative execution paths.

| Workload | Eager ms | Graph ms | Speed ratio | ≥1.10× gate |
| --- | ---: | ---: | ---: | --- |
| Prefill 128 | 55.30 | 39.26 | 1.408× | pass |
| Prefill 512 | 154.00 | 152.21 | 1.012× | fail |
| Prefill 128 + 32 generated tokens | 1833.91 | 697.67 | 2.629× | pass |

All individual wall/CUDA-event samples and their alternating execution orders are retained in the aggregate. The 512-token workload does not pass the speed gate. Generation produces 32 greedy tokens with EOS early termination disabled, exactly as in the matched eager arm.

## Resident memory and setup

Parameter storage is unchanged at **6.425 GB**. Peaks below describe one workload graph at a time; they do not measure retaining every possible sequence-length graph simultaneously.

| Workload (prompt/generated) | Eager peak allocated GB | Graph peak allocated GB | Eager peak reserved GB | Graph peak reserved GB | Capture + warmup s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128/0 | 6.462 | 6.473 | 6.516 | 6.593 | 0.355 |
| 512/0 | 6.552 | 6.570 | 6.642 | 6.872 | 0.669 |
| 128/32 | 6.489 | 6.460 | 6.579 | 6.669 | 7.166 |

Eager peak allocation was measured before graph capture. Graph figures include its retained pool and the returned output copy; capture-time peaks are also retained in the raw result. These are PyTorch allocator measurements, not whole-device memory accounting. Device/process snapshots were recorded and the external-process memory contention check passed. Existing unrelated GPU services were left unchanged.

Capture/warmup is a startup cost excluded from steady-state replay latency. The graph accepts new token IDs of the captured shape; a different shape or generation length needs another capture. The steady-state speedup requires reuse. This comparison is against the repository's eager FP16 reference, not against every optimized inference engine or the deployed Q4 llama.cpp runtime.

## Exactness and integrity

The CUDA pilot checked prefill, cached decoding, generation, changed inputs, and output ownership on a tiny model; it then passed representative full-model memory and exactness checks. Pilot results are systems evidence only. The measured full run completed all 3 workloads, with 10 timing pairs each, zero failed or reused cells.

Two unequal fixed input sequences were checked for each workload. Prefill controls compare every output-logit and KV-cache bit; generation controls compare every generated token. Both expected and actual tensors are retained in hashed `.pt` control artifacts, and the finalizer independently rechecked their bitwise equality. No training or floating-point reassociation was introduced.

The exact reference checkpoint hash is `f01cc0653ec6c9bd7026b8676864612b91b6c79d2dc4df42599cbe4cf99eb9a9`. The producer includes source snapshots, plan/fixture hashes and dependency/GPU settings. That FP16 implementation was previously compared with Hugging Face using numerical tolerances; this is not a claim of bitwise agreement across different runtimes.

The first full run (`20261010_v1`) was rejected by the finalizer: its nominally different control prompts shared the same first 128 tokens. Its timing results remain diagnostics and are not pooled here. The [input-diversity amendment](../07%20-%20Status%20Reports/2026-10-10_llama32-deployment-input-amendment.md) prospectively fixed the second control to use the other sequence's tail, added a per-workload inequality assertion and regression test, and required a new pilot/full run. This report uses only corrected `20261010_v2` evidence.

## Requested deployment target

- **Faster single model:** supported for the passing workloads above, relative to this eager reference, with the same parameter storage and unchanged arithmetic.
- **Two regular independent models in one model's memory and latency:** not achieved or directly timed here. No parameter sharing is counted as independent-model compression.
- **Tensorization-specific compute/compression advantage:** not established. Graph replay can also optimize regular dense execution.

The separate [partial archive study](2026-10-10_llama32-lossless-compression.md) reduced original checkpoint storage to 4.588 GB with exact reconstruction. Two such archives already exceed one original model's 6.425 GB raw-weight budget, and decoding restores dense weights. Archive bytes are not treated as resident inference memory. The factor archive stage failed and no complete factors-plus-residuals archive total is claimed.

No new application-accuracy measurement was made. Exact execution retains the prior model's answers and errors; the previous 10/12 synthetic validation result and unmeasured 95% application target remain unchanged. One model, one GPU and fixed workloads do not establish universal speedups.

## Reproduction

```bash
data/experiments/moba_paper/environment/bin/python \
  -m experiments.structure_net.campaign_llama_cotenancy \
  --exact-deployment pilot \
  --control data/experiments/llama_tensorized_cotenancy/20261010_control_v1 \
  --output data/experiments/llama_lossless_deployment/NEW_PILOT_ROOT
```

After the pilot passes, run the same launcher with `--exact-deployment full` and a new output root. It selects physical GPU 0 before CUDA initialization. Finalize with `experiments.neural_architecture_lab.finalize_llama_exact_deployment --campaign <full-root>/deployment --register`. The reusable graph wrapper is `experiments/structure_net/llama_cuda_graph.py`.
