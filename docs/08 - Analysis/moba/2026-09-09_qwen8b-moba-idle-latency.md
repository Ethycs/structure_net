# Qwen MoBA: idle latency and correction

Date: 2026-09-09. Systems diagnostic, reused validation input.

**The approximately two-second latency was real; blaming it on another GPU job was incorrect.** Both earlier screen/final producer files show about 11,511 MiB free and 1% GPU utilization before model loading. The job that had blocked earlier pilots was already absent. The `shared_gpu` names describe the conservative memory-limited protocol, not established contention during those final measurements. The original numerical results remain unchanged.

The new controlled run starts with 11,618 MiB free and 0% utilization and records external GPU processes before/after every cell. No external contention was detected. All weights are frozen; the architecture and FP16/FP32 readout contract are unchanged. MoBA remains the official implementation with block1024/top2 across six attention layers.

## Measured latency

Five rotated/alternating CUDA-event samples after one warmup per path. Medians below exclude loading, disk I/O and component-profiling hooks. Complete-sequence MLP execution changes batching, not which layers or weights are used.

| Execution | 8K tokens | 16K tokens |
| --- | ---: | ---: |
| Full attention, MLP chunks of 2048 | 863.56 ms | 2015.38 ms |
| Full attention, complete-sequence MLP | 854.43 ms | 2005.57 ms |
| MoBA, four heads per call, MLP chunks of 2048 | 940.92 ms | 1848.80 ms |
| MoBA, all 32 heads, MLP chunks of 2048 | 886.16 ms | 1795.22 ms |
| MoBA, all 32 heads, complete-sequence MLP | 875.88 ms | 1785.16 ms |

With matched complete-sequence MLP batching, MoBA is **1.123× faster at 16K**, reducing latency by **11.0%**. At 8K it remains slower than matched full attention. Removing four-head chunking saves about 53.6 ms at 16K; changing MLP batching saves another 10.1 ms. These changes do not turn this into a low-latency prefix.

## Where time goes

Separate CUDA-event instrumentation at 16K; these are component estimates, not the primary timing samples. Ranges include CPU launch gaps within the measured component. The MLP column includes only the first five MLPs; layer six's MLP is omitted.

| Execution | Attention | Five MLPs | Q/K/V/output projections | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full attention, complete-sequence MLP | 536.8 ms | 952.6 ms | 317.1 ms | 201.8 ms | 2008.3 ms |
| MoBA, all 32 heads, complete-sequence MLP | 326.3 ms | 950.9 ms | 308.7 ms | 201.3 ms | 1787.2 ms |

Attention accounts for about 18.3% of the MoBA prefix time. The MLPs, projections and remaining work take about 1.46 seconds. Keeping those costs fixed, even eliminating the remaining attention time would yield at most 1.22× additional speedup. This bound is an arithmetic inference from the instrumented run, not a measured alternative implementation.

The benchmark recomputes a complete 16K-token prefix. It is not a measurement of one new summary or a cached incremental update, nor of an isolated final-layer attention call. MoBA alone provides a modest improvement for this prefix workload.

## Validity and limits

All 10 execution/context cells completed. Source snapshots and fingerprints, exact input window, model hashes, paired samples, memory usage and GPU process state are retained. The real-prefix graph check passed; weight hashes match before/after. Scheduling variants retain mean head/output cosine similarity above 0.9999 against their corresponding chunked execution. This checks scheduling equivalence, not application accuracy or exact numerical identity.

Only the already-used Wikipedia validation window was timed. This is not another ten-window quality confirmation. No weights were trained and no 95% application-accuracy claim follows. The earlier readout-preservation result remains separate from this latency correction.

## Provenance

- Campaign: `data/experiments/qwen_moba_prefix/20260909_idle_latency_v1/campaign_results.json`
- SHA-256: `05aa9210e3c40506418ecf5604dff7f6d22635cb7b7e0e8eef1f538fbc64525b`
- Producer fingerprint: `c81337e0042afcc376a97b62c31f0f9e9388fabe2a632ef491b033a49f270d19`
- Entry point: `experiments/structure_net/qwen_moba_idle_latency.py`
- Protocol: `docs/07 - Status Reports/2026-09-09_qwen8b-moba-idle-latency.md`
- Two focused contention/profiling tests passed; the existing prefix numerical controls are reused.

Earlier startup telemetry (preserved evidence):

```json
{
  "20260909_shared_screen_v1": "0, GPU-0370fdb0-6490-d871-15e5-c14a1fa671ff, 11511, 1\n1, GPU-8cf440df-e8f4-2210-a2cd-29a9d00aca14, 93, 0\n2, GPU-7b1db3cb-ec9c-beb3-e7db-826c2543edf9, 4239, 0\n",
  "20260909_shared_final_v1": "0, GPU-0370fdb0-6490-d871-15e5-c14a1fa671ff, 11511, 1\n1, GPU-8cf440df-e8f4-2210-a2cd-29a9d00aca14, 93, 0\n2, GPU-7b1db3cb-ec9c-beb3-e7db-826c2543edf9, 4239, 0\n"
}
```
