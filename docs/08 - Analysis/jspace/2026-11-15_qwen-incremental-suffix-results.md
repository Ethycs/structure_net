# One-block expansion of the recovered Qwen suffix

Study **JSPACE-QWEN-INCREMENTAL-SUFFIX-01**. Classification: `valid_expansion_within_budget`. **Confirmed: false.**

The selected seven-block suffix scored **51/64 (79.69%)**, versus 48/64 at initialization, 51/64 for its accepted six-block parent, and 48/64 for matched native BF16 execution. The dual five-percentage-point gate requires **48/64** and passed.

Epoch 3 won by correct count, then full-vocabulary CE, with the earlier epoch retained for exact ties. Initialization remained eligible and already met the 48/64 budget; training added a net three correct development answers.

This is the historical rank-1024, head-64, polynomial-MLP suffix with fixed reduced-coordinate bridges, BF16 trainable parameters and the original geometry precision. It adds only block 29 to recovered blocks 30–35, jointly training all seven and the final norm. It differs from the later full-head, native-residual all-block repair experiment.

The frozen native prefix is supplied through authenticated BF16 boundary-29 caches. The complete trainable suffix is recomputed on current student activations; raw-token prefix execution is not rerun on every batch.

| Epoch | Correct /64 | Full-vocabulary KL | Full-vocabulary CE | Loss | Native agreement | Parent agreement |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 48 | 0.007543 | 1.652595 | 0.516900 | 93.75% | 95.31% |
| 1 | 47 | 0.053382 | 1.562893 | 0.618337 | 98.44% | 90.62% |
| 2 | 50 | 0.055520 | 1.589544 | 0.632319 | 90.62% | 95.31% |
| 3 | 51 | 0.060192 | 1.566927 | 0.638616 | 79.69% | 90.62% |
| 4 | 47 | 0.041874 | 1.592942 | 0.595130 | 98.44% | 90.62% |
| 5 | 49 | 0.032656 | 1.589378 | 0.568251 | 95.31% | 93.75% |
| 6 | 49 | 0.027404 | 1.550163 | 0.541781 | 89.06% | 96.88% |
| 7 | 50 | 0.024208 | 1.555356 | 0.534391 | 93.75% | 95.31% |
| 8 | 48 | 0.019797 | 1.552040 | 0.521043 | 90.62% | 95.31% |
| 9 | 49 | 0.019136 | 1.563301 | 0.522570 | 92.19% | 96.88% |
| 10 | 49 | 0.018661 | 1.550987 | 0.517546 | 92.19% | 96.88% |
| 11 | 49 | 0.018389 | 1.550232 | 0.516558 | 92.19% | 96.88% |
| 12 | 49 | 0.018192 | 1.553438 | 0.516968 | 92.19% | 96.88% |

| Model | ARC /16 | BoolQ /16 | HellaSwag /16 | WinoGrande /16 |
|---|---:|---:|---:|---:|
| Native BF16 | 14 | 13 | 14 | 7 |
| Accepted six-block parent | 15 | 14 | 14 | 8 |
| Initial seven-block | 14 | 13 | 13 | 8 |
| Selected seven-block | 15 | 14 | 13 | 9 |

Against the native, selected predictions gained 8 and lost 5 correct answers (net +3).

Against the parent, selected predictions gained 3 and lost 3 correct answers (net +0).

Against the initializer, selected predictions gained 6 and lost 3 correct answers (net +3).

The most frequent gold label scores 30/64 as a constant-label baseline on these reused development cases.

All 384 updates across 12 epochs completed on the original 512 training cases. The objective stayed .7T² full-vocabulary KL to the accepted parent plus .3 gold next-token CE, T=2, at four existing positions. No hidden-state auxiliary was added. The discarded pilot established finite nonzero gradients and changed FP32 optimizer masters in all seven blocks and norm while the readout and geometry remained fixed. Stored BF16 changes are a separate diagnostic; an individual update can round away at BF16 precision.

Native, packed-parent and boundary-seam controls passed. The historical Torch teacher-cache versus packed-parent logit difference remains a separate diagnostic; these paths are not required to be bitwise identical. Independent fresh selected-state verification passed strict reload, all 64 predictions and metrics, with **zero choice-logit difference**. The prospective BF16 tolerance remained 0.03125 + abs(reference)/128.

The selected suffix checkpoint contains 792,186,224 tensor bytes; the packed seven-block suffix and norm contain **627,581,808 bytes**, versus **1,079,081,984 bytes** for those seven native blocks and norm: **41.84% smaller for the treated suffix**. Including all 29 frozen native blocks, the tied embedding/readout once, fixed geometry and rotary buffers gives **5,720,377,456 persistent tensor bytes**, versus **6,171,877,632 bytes** for the original native model: **7.32% smaller overall**. Only **7 of 36 blocks** are compressed. No inference speed was measured.

This is one development-selected expansion. No untouched confirmation, speed benchmark, next-block expansion or whole-model 38%/1.4× result is established. A quality miss is retained as a completed negative experiment.

Evidence: `data/experiments/jspace_incremental_suffix/20261115_v2/incremental_suffix_analysis.json`, SHA256 `fd59d35080a82c235a77bfc6793c02d8243d418f857db89773045f61a32d1f25`.
