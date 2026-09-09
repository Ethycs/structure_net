# Qwen3-8B prefix: accelerating work outside MoBA

Date: 2026-09-09. Exploratory frozen-weight execution study.

Hypothesis: `qwen3-8b-prefix6-nonattention-acceleration-v1`. Classification: **`execution_regression_supported`**.

The baseline and candidates use the same six attention layers and first five complete MLPs of the pinned public Qwen3-8B checkpoint. Layer six's MLP is omitted in every arm. Official MoBA block1024/top2 is unchanged. No optimizer, training update or layer removal is used. FP16 variants retain the original weight values; INT8 variants use approximate copies and retain the original checkpoint for controls.

## Full-prefix validation screen

RTX 3060, one request, complete 8K/16K prefill, both corpora. Five rotated/alternating CUDA-event samples per arm; table cells are means of the two per-source medians. Loading, compilation and initialization are excluded. Fidelity is compared with original FP16 MoBA, separately from the full-attention reference.

| Execution | 8K latency | 16K latency | 16K speed vs MoBA | 16K joint passes | Max head L2 error, 16K |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original FP16, full attention | 862.9 ms | 2017.1 ms | — | — | — |
| Original FP16, MoBA | 886.3 ms | 1794.5 ms | — | — | — |
| fused_fp16 | 780.3 ms | 1589.0 ms | 1.129× | 2/2 | 0.253% |
| int8_mlp6 | 572.0 ms | 1171.1 ms | 1.536× | 0/2 | 24.470% |
| int8_mlp3 | 584.4 ms | 1195.8 ms | 1.505× | 0/2 | 23.035% |
| int8_gate_up6 | 650.7 ms | 1317.9 ms | 1.366× | 2/2 | 3.342% |

FP16 fusion packs Q/K/V, fuses pointwise normalization/rotary/residual work and the MLP activation, and computes the final output projection on segment-pooled heads. The projection commutes with pooling in exact arithmetic; FP16 rounding changes are measured. INT8 candidates additionally quantize the first five MLPs: threshold 6, threshold 3, or threshold 6 gate/up only with FP16 down projections. The compiler used pointwise fusion and vendor GEMMs; it reported insufficient GPU SMs for its GEMM max-autotuning path.

FP16 gates: head/output mean cosine ≥0.9999, both relative L2 errors ≤0.005, change Pearson ≥0.999, and strongest-change-quartile F1 ≥0.95. INT8 gates: cosine ≥0.995, L2 ≤0.05, Pearson ≥0.995, and F1 ≥0.95. Every passing cell must also be ≥1.05× faster than original MoBA with no external contention detected. Selection uses both 16K validation sources; 8K is a crossover diagnostic.

Completed 16/16 screen cells; 8 joint passes across both lengths. Selected execution: `int8_gate_up6`.

## Reused-window regression

The selected execution was fixed before checking the five existing final input windows per corpus. These are ten reused numerical/runtime regression windows, not ten newly trained models or a newly sealed application test.

| Corpus / seed | Original full | Original MoBA | Selected | Speed vs MoBA | Head L2 | Change Pearson | Change F1 | Joint pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| wikipedia / 7 | 2006.9 ms | 1785.2 ms | 1303.6 ms | 1.369× | 3.398% | 0.999928 | 1.0000 | True |
| wikipedia / 17 | 2013.1 ms | 1790.0 ms | 1306.3 ms | 1.370× | 3.402% | 0.999904 | 0.9688 | True |
| wikipedia / 29 | 2022.8 ms | 1796.5 ms | 1310.9 ms | 1.370× | 3.455% | 0.999941 | 1.0000 | True |
| wikipedia / 41 | 2023.7 ms | 1797.9 ms | 1315.5 ms | 1.367× | 3.354% | 0.999954 | 1.0000 | True |
| wikipedia / 53 | 2023.8 ms | 1801.2 ms | 1317.8 ms | 1.367× | 3.389% | 0.999905 | 0.9688 | True |
| bnc_spoken / 7 | 2024.5 ms | 1798.5 ms | 1313.7 ms | 1.369× | 3.114% | 0.999898 | 1.0000 | True |
| bnc_spoken / 17 | 2024.9 ms | 1801.4 ms | 1319.8 ms | 1.365× | 3.087% | 0.999872 | 1.0000 | True |
| bnc_spoken / 29 | 2025.2 ms | 1801.7 ms | 1317.2 ms | 1.368× | 3.007% | 0.999879 | 1.0000 | True |
| bnc_spoken / 41 | 2025.2 ms | 1799.8 ms | 1319.6 ms | 1.364× | 3.162% | 0.999946 | 0.9688 | True |
| bnc_spoken / 53 | 2025.4 ms | 1802.1 ms | 1318.8 ms | 1.366× | 3.072% | 0.999895 | 1.0000 | True |

Joint passes: **10/10**. Mean of window medians: original full **2021.6 ms**, original MoBA **1797.4 ms**, selected **1314.3 ms**. Ratio of mean latencies: **1.368×** versus original MoBA (26.9% less latency), **1.538×** versus original full attention (35.0% less latency).

Against unchanged MoBA, worst head/output relative L2: 3.455% / 2.752%. Minimum change Pearson: 0.999872; minimum change F1: 0.9688.

Against original full attention, mean head cosine ranges 0.978145–0.984051, and change Pearson 0.985526–0.997763. Those include MoBA's approximation and are not accuracy measurements.

## MLP pilot and practical limits

The preliminary microbenchmark used the actual third-layer MLP input from the 8K Wikipedia validation window. Its results did not determine full-prefix support:

| One MLP variant | Latency | Speed | Relative L2 |
| --- | ---: | ---: | ---: |
| fp16 | 93.00 ms | 1.000× | 0.000% |
| cublaslt | 93.99 ms | 0.991× | 0.043% |
| packed_fp16 | 93.60 ms | 0.994× | 0.000% |
| compiled_fp16 | 92.09 ms | 1.011× | 0.040% |
| int8_threshold6 | 51.86 ms | 1.807× | 7.805% |
| int8_threshold0 | 38.09 ms | 2.474× | 24.024% |

Numerical first forwards, including compilation when needed, ranged 0.60–13.24 seconds in the screen/regression. The earlier 8K lifecycle pilot incurred a separate 14.34-second first forward. Disk compiler caches were reused across stages; setup numbers are not universal cold-cache costs. Peak allocated CUDA memory, including retained originals and any approximate copies, was 8.22 GB.

Original checkpoint hashes match before/after in each stage. GPU process snapshots accompany every cell. This is complete-context batch-one execution on one RTX 3060; no claim is made for other GPUs, short contexts, decoding, or the gateway retrieval model. INT8 changes arithmetic precision; it does not remove the MLPs. FP16 execution reduces overhead and moves one linear projection after pooling; it does not make all five MLP matrix multiplications disappear.

**Application accuracy and the user's 95% final-accuracy target remain unmeasured.** The output checks assess preservation of attention readouts and change rankings on representative public text. No hierarchy labels, retrieval outcomes, or answer accuracy were supplied. No MLP or attention weight was retrained. Incremental caching was not benchmarked; append-only workload clarification was not available when this full-pass study ran.

## Provenance and reproduction

- Screen: `data/experiments/qwen_moba_prefix/20260909_nonattention_screen_v1/campaign_results.json`; SHA-256 `ebd4196d5989407f627c0211dc42a89c302c4f28d28b686316857b103a2a1eab`.
- Decisive campaign: `data/experiments/qwen_moba_prefix/20260909_nonattention_regression_v1/campaign_results.json`; SHA-256 `c71ca35429b9c5ad887d88d606864b188afc537de96fc5427afe8fef166e7b8c`.
- MLP pilot: `data/experiments/qwen_moba_prefix/20260909_nonattention_pilot_v1/campaign_results.json`; SHA-256 `261c2ed6832e65bf3b7ea544b2c30593b2438570d795ff85913dd639ed83388d`.
- Screen fingerprint: `27e1a460f1a4d54a4289555957c15ef303b90955bfebd42cc415b284f7cf7160`.
- Protocol: `docs/07 - Status Reports/2026-09-09_qwen8b-nonattention-acceleration.md`.
- Seven focused tests cover packed MLP order, full-prefix graph equivalence, numerical gates, two-source selection, and conservative registry status.

```bash
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH= \
 PYTHONPATH=.:src TORCHINDUCTOR_COMPILE_THREADS=2 \
 TORCHINDUCTOR_CACHE_DIR=$PWD/data/experiments/moba_paper/inductor_cache \
 TRITON_CACHE_DIR=$PWD/data/experiments/moba_paper/triton_cache \
 data/experiments/moba_paper/environment/bin/python \
 -m experiments.structure_net.run_qwen_nonattention --stage screen \
 --output data/experiments/qwen_moba_prefix/NEW_NONATTENTION_SCREEN_ROOT
```

Use `--stage regression --selection <new-screen>/campaign_results.json` and a new output root only if the fixed screen selects a candidate. The isolated environment and pinned model/upstream hashes are recorded in each producer manifest.
