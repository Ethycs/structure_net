# Independent Qwen layer recovery results

Study: **JSPACE-QWEN-PARALLEL-SINGLE-LAYER-RECOVERY-02**. Classification: `isolated_recovery_with_two_quality_misses`. **Confirmed: false.**

All **34 independent trials (layers 2–35)** completed 12 epochs and fresh-process verification. **32 layers passed** the fixed isolated gate of at least 47/64; layer 12 scored 44/64 and layer 15 scored 46/64. The native control scored 50/64. These are separate one-layer substitutions on the same 64 development questions. **32/34 is not a model accuracy.**

Training improved the number correct for 20 layers; 14 retained their initial count. Initializers won selection for layers 20, 25; their final scores must not be attributed to training. All trials, including misses, completed **13,056 optimizer updates** in aggregate without a retry.

| Layer | Initial /64 | Selected /64 | Epoch | ARC /16 | BoolQ /16 | HellaSwag /16 | WinoGrande /16 | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 48 | 49 | 7 | 15 | 13 | 14 | 7 | pass |
| 3 | 49 | 49 | 12 | 15 | 13 | 13 | 8 | pass |
| 4 | 49 | 49 | 12 | 15 | 13 | 13 | 8 | pass |
| 5 | 47 | 49 | 1 | 15 | 12 | 14 | 8 | pass |
| 6 | 47 | 49 | 3 | 14 | 14 | 14 | 7 | pass |
| 7 | 47 | 48 | 12 | 15 | 14 | 13 | 6 | pass |
| 8 | 46 | 48 | 3 | 14 | 14 | 12 | 8 | pass |
| 9 | 49 | 49 | 2 | 15 | 14 | 13 | 7 | pass |
| 10 | 44 | 47 | 5 | 14 | 13 | 13 | 7 | pass |
| 11 | 45 | 47 | 1 | 14 | 12 | 13 | 8 | pass |
| 12 | 44 | 44 | 3 | 12 | 12 | 10 | 10 | miss |
| 13 | 47 | 48 | 12 | 15 | 13 | 13 | 7 | pass |
| 14 | 45 | 50 | 2 | 14 | 14 | 12 | 10 | pass |
| 15 | 40 | 46 | 12 | 14 | 13 | 12 | 7 | miss |
| 16 | 45 | 50 | 11 | 13 | 14 | 12 | 11 | pass |
| 17 | 43 | 48 | 12 | 15 | 13 | 13 | 7 | pass |
| 18 | 46 | 48 | 11 | 13 | 13 | 13 | 9 | pass |
| 19 | 47 | 48 | 12 | 13 | 13 | 13 | 9 | pass |
| 20 | 51 | 51 | 0 | 14 | 13 | 15 | 9 | pass |
| 21 | 48 | 49 | 5 | 14 | 13 | 13 | 9 | pass |
| 22 | 50 | 50 | 12 | 15 | 13 | 14 | 8 | pass |
| 23 | 47 | 47 | 10 | 15 | 13 | 13 | 6 | pass |
| 24 | 49 | 50 | 12 | 15 | 13 | 14 | 8 | pass |
| 25 | 47 | 47 | 0 | 14 | 13 | 13 | 7 | pass |
| 26 | 49 | 50 | 4 | 15 | 13 | 13 | 9 | pass |
| 27 | 48 | 49 | 12 | 14 | 13 | 14 | 8 | pass |
| 28 | 50 | 50 | 12 | 15 | 13 | 14 | 8 | pass |
| 29 | 50 | 50 | 12 | 15 | 13 | 14 | 8 | pass |
| 30 | 50 | 50 | 12 | 15 | 13 | 14 | 8 | pass |
| 31 | 49 | 50 | 12 | 15 | 13 | 14 | 8 | pass |
| 32 | 49 | 50 | 12 | 15 | 13 | 14 | 8 | pass |
| 33 | 50 | 50 | 12 | 15 | 13 | 14 | 8 | pass |
| 34 | 49 | 49 | 11 | 14 | 13 | 14 | 8 | pass |
| 35 | 50 | 50 | 3 | 15 | 13 | 14 | 8 | pass |

Each worker trained only one compact block on the same original 512 training examples and native BF16 input/output pairs. Task checks instead used authenticated native FP32 development boundaries, the selected packed block, the original downstream layers and tied vocabulary readout. Every worker's native substitution control reproduced 50/64 before training. Accuracy-first selection used relative feature error to break ties; initialization was eligible. The independent reload reproduced each selected prediction exactly, with the recorded feature/logit tolerances.

The residual stream remains 2048 wide; computed updates have rank 1024. Native 128-dimensional attention heads, softmax and the 11008-wide MLP remain. Training uses FP32 parameters, computation, gradients and AdamW. The objective is equal-case/equal-token output error divided by original teacher-update energy, with a fixed floor fitted on training data. Full epoch histories, normalized features, paired changes against initialization/native and family counts are retained in the analysis artifact.

| Tensor accounting | Bytes |
|---|---:|
| Native block, FP32 | 308,307,968 |
| Selected packed block, FP32 including basis/geometry | 162,650,116 |
| Native full model, FP32 | 12,343,754,752 |
| Projection: 34 replacements, rest native FP32 | 7,391,387,784 |
| Projection: 36 replacements, all FP32 | 7,100,072,080 |
| Original full model, BF16 | 6,171,877,376 |
| Projection: 36 FP32 replacements, remaining tensors BF16 | 6,477,738,128 |

All 34 selected checkpoint tensor shapes and dtypes match. Each packed block saves **47.24%** against the same native block in FP32. The all-36 FP32 projection saves **42.48%**, but replacing native BF16 blocks with these FP32 compact blocks increases total tensor bytes by **4.96%**. These whole-model figures are arithmetic projections, not a packaged/evaluated assembly. Tied embedding/readout is counted once; fixed geometry is included. Optimizer, activations, KV cache and serialization overhead are excluded. No speed measurement belongs to these checkpoints.

Assembly is pending in this study and has a separate [frozen protocol](../../07%20-%20Status%20Reports/jspace/2026-11-14_qwen-recovered-assembly.md). Passing isolated replacements does not establish that their errors compose. The accepted block0/block1 results are separate prior experiments; they are not included in this campaign's 34 trials. No untouched confirmation examples were accessed for this closeout.

Evidence: `data/experiments/jspace_parallel_single_layer/20261114_v1/parallel_layer_analysis.json`, SHA-256 `b6272785022d0c965a0f34b6c2a572929e90c0db3303f84d00621321f49c72e6`. All selected checkpoint bytes, stage/development records and live/archived producers rehashed at analysis. Original teacher/cache payloads retain pinned producer hashes plus current immutable file-identity verification; small consumed inputs rehashed. Later closeout stages verify hashes of small artifacts and unchanged device/inode/size/mtime/ctime of already-hashed selected checkpoints.

Registry registration requires the exact source, tests and this report to be committed, followed by exact-ID readback. The registry primary metric is **32 isolated layers passing (unit: count)**; no aggregate model-accuracy metric is stored.
