# Joint recovery of the recovered Qwen assembly

Study **JSPACE-QWEN-RECOVERED-JOINT-DISTILLATION-01**. Classification: `partial_recovery_below_gate`. **Confirmed: false.**

The selected full-model checkpoint scored **28/64 (43.7500%)**, compared with **27/64** for its assembly before joint adaptation and **50/64** for the original FP32 model. The fixed development gate was **47/64**; it did not pass. These are actual assembled-model task scores on the original 64 development cases.

All 3 epochs and 96 optimizer updates completed on 512 training cases. Selection chose epoch 3, using highest task accuracy then lowest conditional-choice CE. The initializer remained eligible and was not selected. The separate one-update pilot was discarded. Every scientific update reported gradients for all 36 compact blocks and final RMS gain; the pilot independently showed a parameter change in every block while fixed geometry stayed unchanged.

| Epoch | Correct /64 | Accuracy | Choice CE | Choice KL | Full KL | Answer-label mass | Teacher agreement |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 27 | 42.1875% | 9.357735 | 4.60222 | 2.892983 | 1 | 42.1875% |
| 1 | 24 | 37.5000% | 1.149108 | 1.017075 | 1.06457 | 0.9999539 | 32.8125% |
| 2 | 20 | 31.2500% | 1.191263 | 1.043833 | 1.075981 | 0.9999292 | 23.4375% |
| 3 | 28 | 43.7500% | 1.830793 | 1.318459 | 1.231313 | 0.9999313 | 35.9375% |

| Family | Native /16 | Initial /16 | Selected /16 |
|---|---:|---:|---:|
| arc | 15 | 5 | 5 |
| boolq | 13 | 7 | 9 |
| hellaswag | 14 | 7 | 5 |
| winogrande | 8 | 8 | 9 |

Against initialization, the selected checkpoint gained 10 correct answers and lost 9. Against native, it gained 8 and lost 30. Choice margin changed from -3.527148 to -0.722891. Full-vocabulary answer CE is recorded separately from the optimized conditional-choice CE.

The matched original teacher was captured with original BF16-stored weights promoted to FP32, using full FP32 forward/readout and no TF32. It reproduced native 50/64 with exact predictions and the frozen logit tolerance. Cached targets contain full vocabulary logits at each prompt's midpoint and answer. They provide supervision only: every student minibatch starts from raw tokens and recomputes all 36 blocks.

The fixed objective was `.2 T² KL_full + .5 T² KL_choice + .3 CE_choice`, with T=2. Full KL covers midpoint and answer; choice KL/CE use the answer position and its valid choices. Rank 1024 updates, native 2048-wide residual carry, 128-dimensional heads, softmax and 11008-wide MLP activations remained unchanged. Original tied E, bases, anchors and other geometry were frozen; all block parameters and final RMS gain trained in FP32. Blocks 0–23 plus tied E/norm ran on RTX 3060; blocks 24–35 ran on RTX 2060 SUPER. Gradients crossed the partition and nonreentrant activation checkpoints recomputed student blocks.

Fresh-process verification authenticated the selected checkpoint, strictly reloaded it and reproduced all 64 predictions plus metrics/logits within the frozen tolerances, with fixed state unchanged. The selected unpacked model contains 1,699,409,920 parameters and 7,103,611,024 saved FP32 tensor bytes including basis/geometry and tied E once. These counts are not a speed result or a packed deployment measurement. The earlier BF16 stored model is not used as a matched-FP32 size denominator here.

The outcome does not demonstrate successful recovery. Conditional-choice CE fell from 9.3577 to 1.8308 and full-vocabulary KL from 2.8930 to 1.2313, but the selected model gained only one net correct answer, remained below the 47/64 gate, and matched fewer teacher decisions: 23/64 versus 27/64 initially. The classification `partial_recovery_below_gate` records the small numerical gain without treating lower probability losses as adequate task recovery.

An inference from the separate [stacked projection oracle](2026-11-14_qwen-stacked-projection-oracle-results.md) is that the fixed spaces admit teacher-supplied corrections preserving native decisions on these 64 cases: that diagnostic retained 50/64 correct and all 64 native predictions. This does not show that the compact attention/MLP branches can learn those corrections from their own inputs. It helps separate the demonstrated geometric trajectory from the still-unsolved learning problem; it does not establish branch sufficiency or generalization.

A proposed next experiment, not executed here, would fit the predicted in-space correction on the **actual student input** `s_l` to `U_l.T @ (t_native,l+1 - s_l)` (column-vector notation). Start with the upstream student prefix frozen and detached, and recapture its states after accepted upstream changes. Retain endpoint distillation and actual task-accuracy checks. Substituting static oracle states for student inputs would recreate the prefix-distribution mismatch this target is intended to address.

This is one seed on repeatedly inspected development cases. It does not establish held-out quality, unrestricted generation quality or throughput. No untouched confirmation data were opened, and no packing-parity or inference-speed test is claimed. The selected checkpoint and completed failure outcomes, if any, are preserved; changing the training schedule or objective requires a separate continuation.

Evidence: `data/experiments/jspace_recovered_joint/20261114_v1/recovered_joint_analysis.json`; SHA-256 `0bfb8b10be8a4dd073228e6c4c658cb6b826582cc8276994871da274a4af629d`. Selected37 model files and teacher logit bank authenticated by fullSHA256. Stage, history and live/archived producer bytes authenticated. Prior assembly/teacher/native inputs retain original pins and immutable file-identity checks. Subsequent closeout processes recheck small hashes and unchanged identities of already-hashed large artifacts.
