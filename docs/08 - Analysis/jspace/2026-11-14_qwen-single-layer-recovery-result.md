# Qwen single-layer recovery

Status: completed exploratory development study. Hypothesis: **JSPACE-QWEN-SINGLE-LAYER-RECOVERY-01**. Classification: `single_layer_development_gate_passed`. **Confirmed: false.**

Artifact: `data/experiments/jspace_single_block_distill/20261114_v2/single_layer_analysis.json`; SHA-256 `657a19d3dda6ee734b3bb61e07382beec9994e15931a28c6dc83c1be594dc25b`.

The selected checkpoint scored **48/64 (75.0000%)**, compared with 46/64 before training and 50/64 for the native-equivalent control. The frozen acceptance threshold was 47/64. It passed that development gate. Only the first of 36 blocks was compressed.

| Condition | Correct /64 | Accuracy | Gains / losses vs native |
|---|---:|---:|---:|
| Native-equivalent control | 50 | 78.1250% | 0 / 0 |
| Initial enriched block | 46 | 71.8750% | 0 / 4 |
| Selected recovered block | 48 | 75.0000% | 0 / 2 |
| Teacher-update projection oracle | 48 | 75.0000% | 0 / 2 |

| Family | Native | Initial | Selected |
|---|---:|---:|---:|
| arc | 15/16 | 15/16 | 15/16 |
| boolq | 13/16 | 13/16 | 13/16 |
| hellaswag | 14/16 | 12/16 | 13/16 |
| winogrande | 8/16 | 6/16 | 7/16 |

All 12 epochs and 384 optimizer updates completed on 512 training cases. Accuracy-first selection chose epoch 7; relative feature MSE broke ties. The separate systems pilot did not initialize training. A fresh process authenticated and strictly reloaded the selected checkpoint, checked unpacked and packed feature metrics, then reproduced all selected predictions and choice logits within the frozen tolerance through the 35 original layers.

The objective averages per-token squared output error divided by detached original teacher-update energy, then averages cases equally. The floor comes only from training. The full native residual is carried; the computed update has rank 1024, with native 128-dimensional attention heads, softmax, no learned metric, no polynomial replacement, and fixed geometry enriched using training data.

| Feature diagnostic | Initial | Selected |
|---|---:|---:|
| relative_mse | 0.165357834 | 0.104822495 |
| mse | 0.039725289 | 0.0233016543 |
| first_relative_mse | 0.0728070354 | 0.000152529646 |
| content_relative_mse | 0.167489226 | 0.106557471 |
| answer_relative_mse | 0.0142363181 | 0.00587178653 |
| irreducible_outside_relative_mse | 0.0945085718 | 0.0945085718 |
| retained_relative_mse | 0.0708492619 | 0.0103139235 |

The error within the retained subspace fell by 85.442%. Recovery gained 2 correct answers and lost 0 versus initialization. Selected predictions agree with the projection oracle on 64/64 cases.

Training median update energy: `0.202441216`; floor: `2.02441216e-05`. The 48/64 teacher-update projection result is a closest-feature-fit surrogate, not a task-accuracy upper bound: nonteacher features could yield better answers. The matched cached-native boundary control scored 50/64.

| Persistent tensor state | Bytes |
|---|---:|
| Native block0, matched FP32 | 308,307,968 |
| Actual selected unpacked block | 162,748,420 |
| Packed block, including U and saved geometry | 162,650,116 |
| Entire native model, matched FP32 | 12,343,754,752 |
| Entire one-block-changed model, matched FP32 | 12,198,096,900 |
| Original native checkpoint precision | 6,171,877,376 |
| Source precision plus FP32 compact block | 6,180,373,508 |

The packed block uses 52.756% of native block0's FP32 tensor bytes. Across the full model, the matched-FP32 reduction is 1.1800%; the source-precision accounting reduction is -0.1377%. Tied embedding/readout E is counted once. These totals include the fixed basis and buffers; optimizer state, activations, KV caches and transient rotary tables are excluded. The source-precision hybrid is an accounting comparison, not an evaluated mixed-precision deployment. No speed benchmark was run for this checkpoint.

The earlier v1 attempt remains preserved: its allocator peak reached 4,351,590,400 bytes against a 4 GiB reservation, causing a MemoryError after completed epochs 0–4. That was a resource-limit failure, not evidence of a quality failure. v2 restarted the unchanged scientific 12-epoch protocol under a 7 GiB lease with a 256 MiB allocator margin; it did not resume v1 optimizer state.

These 64 development questions have been inspected repeatedly. Passing this local gate does not establish held-out accuracy, equivalence, a fully compressed model, or faster generation. The remaining 35 blocks and original tied readout remain native. Further layer advances require their own explicit training and verification.

CPU reproduction after source/report commit:

```sh
CUDA_VISIBLE_DEVICES='' pixi run --locked --offline python -m experiments.neural_architecture_lab.finalize_jspace_single_layer store
CUDA_VISIBLE_DEVICES='' pixi run --locked --offline python -m experiments.neural_architecture_lab.finalize_jspace_single_layer readback
CUDA_VISIBLE_DEVICES='' pixi run --locked --offline python -m experiments.neural_architecture_lab.finalize_jspace_single_layer closeout
```
