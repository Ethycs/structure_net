# Qwen2.5-3B: staged ten-block recovery

Status: **10 of 36 blocks completed and freshly verified in this campaign; 26 remain**. Study `JSPACE-3B-STAGED-CAMPAIGN-01`, exploratory reused development, `confirmed=false`. Root `data/experiments/jspace_3b_staged_campaign/20261115_v1`; fingerprint `6b40611b583ca3889f4a8fb08370cf967ee95a6161cc13a99d8d70dee29765bb`. Frozen producer commit `4e3a6dc`.

The accepted nine-block parent scored 52/64. Adding the same fitted tenth block produced 47/64. Dense entrance feature fitting recovered 51/64, then cautious joint recovery recovered 52/64. A fresh process strictly reloaded and packed the selected checkpoint, reproducing every development prediction and logit exactly. This closes the five-answer net development gap for this transition; it is not completion of all 36 blocks or independent confirmation.

| Candidate | Correct /64 | Full-vocabulary CE | KL to accepted parent | KL to original native |
|---|---:|---:|---:|---:|
| Raw ten-block initializer | 47 | 1.569281 | 0.016269 | 0.586848 |
| Feature-fitted entrance | 51 | 1.535091 | 0.003610 | 0.551969 |
| Selected joint epoch 6 | 52 | 1.528007 | 0.007299 | 0.554814 |

Original native BF16 control: 48/64. Accepted nine-block parent: 52/64. Selected family counts: ARC 14/16, BoolQ 14/16, HellaSwag 13/16, WinoGrande 11/16. Versus raw, the selected model gains 7 correct answers and loses 2; versus the parent, it gains 3 and loses 3. Equal aggregate accuracy does not mean identical answers.

## Intervention and checks

The architecture, task-Jacobian calibration, fitted geometry, tied embedding/readout and compressed shapes are unchanged. The first cut reuses the authenticated original capture/fit from the same nine-block checkpoint. Original receipts and independent byte copies are retained. Parent, native and singleton-native-to-parent boundary controls all reproduce their expected logits exactly.

Only the new entrance block is fitted first, using dense valid-token supervision in the actual parent coordinates. Twelve epochs / 384 updates reduce normalized development feature error from 1.133943 to 0.640094 and answer-position error from 0.592457 to 0.162720. Parent weights, norm, readout and geometry remain unchanged during this phase; selected delta reload is exact.

Joint recovery then runs all 12 epochs / 384 updates at 2e-5. It keeps raw candidate epoch −1 and feature-fitted epoch 0 eligible. A trained candidate must not worsen CE or original-native KL relative to raw; eligible candidates are ranked by correct count, then CE. Selected epoch 6 has 52/64. Accuracy over epochs −1 through 12 is 47, 51, 51, 50, 51, 51, 52, 52, 50, 52, 52, 52, 52, 52. Later epochs do not overwrite the better selected checkpoint.

Lower learning rate and feature fitting both changed relative to the previous output-only recovery. The pre-joint 51/64 result directly isolates a benefit from feature fitting on these cases. The combined final result does not isolate the independent contribution of the learning-rate change. More transitions and untouched confirmation are needed before generalizing.

## Storage and resources

Measured persistent tensor inventory, including native prefix, fixed geometry and the tied embedding/readout once: **5,523,450,784 bytes versus 6,171,877,632 native bytes, a 10.506% reduction**. This remains a partial model with 26 native blocks. No end-to-end throughput result has been produced for this checkpoint.

Each GPU phase used a separate reservation on RTX 3060, 11264 MiB allocated with 11008 MiB allocator cap. Joint phase elapsed 1150.25 seconds including its phase bookkeeping; peak reserved tensor memory 5,003,804,672 bytes (4.660 GiB). Entrance phase elapsed 187.52 seconds. Calibration reuse/copy/authentication time is not newly performed calibration or inference throughput.

The current shapes project to 3,816,752,960 deployed bytes at 36 compressed blocks, or 38.159% reduction. This is a scheduling projection from observed packed inventories, not a measured completed model. Near-full training VRAM is likewise projected around 9.5–10.1 GiB; later pilots must verify it.

## Durable continuation and remaining gates

`jspace-3b-staged-campaign.service` proceeds from cut 25 to cut 0, running capture, fit, pilot, prealignment, joint training and fresh verification for each block. `jspace-3b-staged-completion.service` admits only a verified accepted cut 0, then runs the separately frozen final gate at `data/experiments/jspace_3b_fullmodel/20261115_v3`. Both are persistent user services; each GPU child acquires its own reservation. No incomplete phase is overwritten or automatically retried.

Still outstanding: all remaining blocks, standalone full-model export, real native/compact cache controls, untouched 512-case accuracy, actual full-model storage and warm end-to-end speed. The model benchmark is bounded by the frozen 416-position deployment capacity. No real-model native prefill-versus-incremental full-vocabulary check at the final gate's exact tolerance is on record yet; earlier graph-token parity is a different check. The final gate runs native controls before attributing any cache failure to compression.

Reused 64-case development selection is exploratory. Final criteria remain at most 5 percentage points below native held-out accuracy, at least 38% actual tensor reduction and 1.4× warm generation speed versus optimized native SDPA. These criteria have not yet been established.

Selected checkpoint SHA256: `a8c089d3538f62d470380a3b46fec42b7f9ded8826ba727487154f1f954d3506`.
Fresh verification result SHA256: `9397bcf3cfcbd3a9c0ff7f8382a8074e48c9e292ac6e942e794684cfc9c759ee`.
Fresh verification manifest SHA256: `4e2c321daaa254c95ac6fa0bc38f796721bbe32f46a5d7217c087db4d1fc9094`.
Training result SHA256: `6d30999b4f135b47c694cb45b7b489910bd6aeb94670068a288a216010ae524c`.
