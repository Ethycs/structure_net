# Recovered compact Qwen final block: measured local speedup

Status: completed exploratory measurement. Hypothesis: **JSPACE-COMPACT-PERFORMANCE-01**. Classification: **local_speedup_with_measured_quality**; `confirmed=false`.

The optimized final block is **1.297× faster for prefill and 1.275× faster for cached decode than the original native block**, with both implementations captured in CUDA graphs. The faster model answered **391/512** fresh questions correctly, compared with **393/512** for native Qwen and **390/512** for the previous recovered compact model. Whole-model graph generation remained about **41 tokens/s**; its measured improvement is small and does not establish a useful end-to-end speedup.

The machine-readable [aggregate](../../../data/experiments/jspace_fast_recovery/20261113_v2/campaign_results.json) has SHA-256 `24409198100ea5730a2835f5c3deff98b28f69c282e5c41d63895bdccc24aa48`. It joins the frozen performance screen, the paired native comparison, and fresh validation of the exact exported tensors.

## What changed

The parent is **Qwen2.5-3B-Instruct**, with its first 35 blocks frozen. Its recovered final block uses 1,536 affine residual coordinates, 64-dimensional attention heads, and benchmark-derived polynomial compilation of part of the MLP. The parent checkpoint was selected at epoch 12 using output KL plus CE. This follows the [affine experiment](2026-11-13_jspace-qwen-affine-v2.md); it does not introduce a hidden-state reconstruction objective.

The selected execution combines BF16 fixed basis geometry, polynomial-matrix padding to a multiple of 128, and fused normalization/affine-shift/SwiGLU/polynomial-feature work. Padding changes the polynomial output width from 10,309 to 10,368. BF16 geometry is an explicit numerical approximation. It is not described as lossless or as preserving every original answer.

Selection used the existing 64 development questions. The selected combination preserved their choices and changed the endpoint loss from 0.7091294102 to 0.7089672264. It therefore passed the frozen direct-validation screen, and the conditional four-epoch continuation was **not executed**. This experiment measures faster execution of an already recovered model; it does not demonstrate additional recovery training or staged joint recovery of two blocks.

## Native comparison

The [paired native comparison](../../../data/experiments/jspace_compact_performance/20261114_native_compare_v1/native_compare/result.json) used the same eight cached development inputs, lengths 60–313 tokens, on an RTX 3060. Both blocks used CUDA graphs and passed exact parity against their own eager executions. Five repetitions included both native-first and fast-first orders. Each timing used 20 calls; the aggregate is the ratio of sums of per-input median latencies.

| Isolated full final block | Native mean latency | Fast mean latency | Wall-time speed ratio | CUDA-event speed ratio |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 1.218019 ms | 0.939004 ms | **1.297140×** | 1.297499× |
| Cached decode | 0.576264 ms | 0.452049 ms | **1.274782×** | 1.275470× |

All eight prompts were faster in both modes. Prefill ratios ranged from 1.2349× to 1.4334×; decode ratios ranged from 1.2735× to 1.2769×. These are descriptive repeated timings on eight inputs, not independent model replications or confidence intervals.

The comparator is the **original native Hugging Face transformer block**. There is no separately hand-fused native baseline. The timing includes the whole final block, including its MLP, and excludes the model's final normalization/readout. The compact and native hidden outputs are allowed to differ; task quality is checked separately. The result measures the combined compressed architecture and execution work, and does not isolate a speedup caused by Gröbner relations alone.

## Whole-model throughput

The [whole-model measurement](../../../data/experiments/jspace_compact_performance/20261113_v1/full_r1536_h64_gb_combined/result.json) generated 32 greedy tokens for each of eight development prompts, batch size one, including prefill. Capture/setup costs were amortized. Native and compact controls were repeated to show temporal variation.

| Full model | CUDA graph tokens/s | Ordinary HF generation tokens/s |
| --- | ---: | ---: |
| Native, initial | 40.8077 | 12.2242 |
| Previous compact | 41.0211 | 11.8766 |
| Fast compact | **41.1687** | 12.0926 |
| Previous compact, repeat | 40.9143 | 11.4243 |
| Native, repeat | 41.0875 | 12.2375 |

The fast graph run is 0.20–0.88% above the two native measurements and 0.36–0.62% above the compact measurements. Ordinary HF generation is about 1.1–1.2% below native. These measurements support a local block improvement; they do not establish a substantial whole-model improvement. The large difference between HF generation and CUDA-graph generation is an execution-mode effect shared by the models and must not be credited to compression.

Only one of 36 blocks changed. The measured local decode saving is about 0.124 ms, compared with roughly 24 ms per whole-model generated token. That gives a useful scale for why the whole-model effect is small; it is not an extrapolation to compressing all blocks.

## Fresh task accuracy and exported models

The fresh set contains 128 questions each from ARC, BoolQ, HellaSwag, and WinoGrande. Its 512 identities are disjoint from every parent split, the earlier layer-distillation evaluation, and this run's 512 training/64 development items. No fresh answer was used to select the structure or decide whether to retrain.

| Model | ARC | BoolQ | HellaSwag | WinoGrande | Total | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 106 | 103 | 100 | 84 | 393/512 | 76.758% |
| Previous compact | 106 | 100 | 99 | 85 | 390/512 | 76.172% |
| Fast compact | 106 | 100 | 100 | 85 | **391/512** | **76.367%** |

Fast versus native has three corrected answers, five newly incorrect answers, and ten changed choices overall: −0.391 percentage points. Fast versus the previous compact model changes one answer, from incorrect to correct: +0.195 percentage points. Wrong-to-wrong changes account for the difference between changed choices and correctness reversals. Descriptive exact McNemar p-values are 0.7265625 and 1.0, respectively. Neither these p-values nor the 512 examples establish equivalence or a real accuracy improvement. The observed point estimates fit the stated one-percentage-point native and half-percentage-point compact degradation screens.

Both packed block-plus-final-norm exports were evaluated from disk and reproduced all 512 records, including recorded choice logits, exactly in a fresh process. Direct tensor identity against the parent-derived structures ties these fresh measurements to the earlier timings.

| Model | Parameters | Deployed tensor bytes, including buffers |
| --- | ---: | ---: |
| Native | 3,085,938,688 | 6,171,877,632 |
| Previous compact | 3,058,908,928 | 6,151,666,112 |
| Fast compact | 3,058,999,552 | 6,145,561,104 |

The faster version saves 6,105,008 tensor bytes relative to the previous compact version and 26,316,528 bytes relative to the full native model. The parameter count rises slightly relative to the compact control because of padding, while stored tensor bytes fall because the geometry buffers use BF16. These are deployed model tensors, not peak allocator usage, optimizer memory, or standalone checkpoint-file sizes. The exported checkpoint is a delta requiring the pinned original model.

## Execution controls and attribution

The development [performance screen](../../../data/experiments/jspace_compact_performance/20261113_v1/bench/result.json) measured the following GB variants against their own matched previous-compact control. Values below are graph timings; all choices on the 64 development cases agreed with that control.

| Variant | Prefill latency | Prefill ratio to compact | Decode latency | Decode ratio to compact |
| --- | ---: | ---: | ---: | ---: |
| Wrapper control | 1.855 ms | 1.001× | 0.541 ms | 1.002× |
| Padding | 1.767 ms | 1.054× | 0.540 ms | 1.000× |
| Fusion | 1.324 ms | 1.417× | 0.501 ms | 1.081× |
| BF16 geometry | 1.599 ms | 1.160× | 0.503 ms | 1.079× |
| Combined | **0.936 ms** | **1.999×** | **0.461 ms** | **1.176×** |

Fusion was separately checked against a Torch reference with the same BF16 geometry and padding; approximation to the old model was not used to excuse kernel implementation errors. The wrapper control shows negligible overhead. The combined structure benefits more than any single change, so independent gains should not be multiplied or attributed solely to one component.

The combined non-GB arm measured 0.965 ms prefill and 0.475 ms decode, versus 0.936/0.461 ms for GB, and the GB version uses 9,856,880 fewer tensor bytes. These two arms were not directly interleaved against each other, and the non-GB version has no evaluation on this fresh set. This is an exploratory observation supporting further matched testing, not proof of a distinct Gröbner-only speed advantage.

## Boundaries and provenance

This is a bounded prototype. Its fixed position tables have capacity **416** and default `position_policy="error"`. Accuracy inputs are at most 384 tokens; the timed prompts are 60–313 tokens; generation is capped at 32 tokens. It is not an unrestricted long-context deployment. Free-generation quality, longer outputs, larger batches, other GPUs, and compression of earlier blocks have not been established here.

The first fresh-validation attempt failed during preflight because an inference-mode copy of the final normalization was restored and then mutated outside inference mode. It produced **no held-out results**. The [preserved failure record](../../../data/experiments/jspace_fast_recovery/20261113_v1/failure.json) and frozen v1 sources remain intact. V2 runs the entire evaluation/export/reload lifecycle in inference mode and passed the regression check. This was a systems failure, not an omitted quality result.

The closeout independently verified 336 distinct artifact/input/source paths across the three campaigns. All stages used experiment reservations. Fresh evaluation and reload used lease `53d053606720`; their peak reserved allocations were 7,277,117,440 and 7,111,442,432 bytes. Sixteen CPU closeout contract tests passed. There is one original trained-model seed and zero independent confirmations.

| Evidence | Root / fingerprint |
| --- | --- |
| Performance screen and whole-model timing | `data/experiments/jspace_compact_performance/20261113_v1`; `9dc26ee292f6ace22c15e6c810f2fbcc4f0a4a5975e9291d2d3b414a58214c86` |
| Paired native timing | `data/experiments/jspace_compact_performance/20261114_native_compare_v1`; `c4777bb7e88a274a1a69ee66a2b01c068d2cb8dafad58b31a87906e8360d9fd4` |
| Fresh exported-checkpoint validation | `data/experiments/jspace_fast_recovery/20261113_v2`; `87d628fe8929976bafa0a454e9dc9948cefeb79dd92ffc20a6cc366b8690c922` |

Native comparison result SHA-256: `d2d87c6cfc3a370ac7f51da00b32b8ec7cc75fb93f726216780d3f03915f836e`. Fast export SHA-256: `47a0b77fb7bbbea0e727ef93b026fb3acbf740ad6f351380ba52dc8f1e1bddad`. Control export SHA-256: `70687585d211fdfad283be7eb2bec88ce3d59ed9a1f74c4fe12ed4baa24f519a`.

Reproduction/registration uses `experiments.neural_architecture_lab.finalize_jspace_compact_performance` in locked, offline default Pixi with `CUDA_VISIBLE_DEVICES=''`. Its `analyze` stage consumes the frozen v2 root and `--native-comparison-root`; `store`, a separate-process `readback`, and `closeout` preserve the exploratory classification. These steps do not launch training or inference.
