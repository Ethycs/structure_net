# Qwen2.5-3B: final-feature recovery and checkpoint acceptance

The thirteen-block feature study completed **12 epochs and 384 updates**. Its unchanged scientific selection retained aligned epoch 0 at **46/64**, reproduced exactly by fresh packed verification. A separate saved best-task checkpoint reached **48/64 at epoch 8** and subsequently passed its own fresh verification: all 64 predictions, logits and metrics matched exactly, with fixed geometry unchanged. These are reused development results, `confirmed=false`.

Root: `data/experiments/jspace_3b_feature_recovery/20261115_v1`; fingerprint `26fdf78c34fc2d05b239f9cd97f6db2fb7aa90a759005dfd76fe4a4c30cadb58`.

The intervention added actual final-answer compact-coordinate matching to the existing accepted-parent output KL/CE objective. Targets came from the accepted twelve-block parent's packed execution, not original-native hidden states. The fixed feature multiplier was **0.18529506976011995**, calibrated without updates on the first eight training cases per family, 32 total, to a 0.1 gradient-norm ratio. Normalization used all **512 training-only target vectors**, with mean squared energy 159.028824. No development cases selected that coefficient. Architecture, geometry, CPU optimizer and frozen scientific selection rules were unchanged.

| Candidate | Correct / 64 | Full-vocabulary CE | Original-native KL |
|---|---:|---:|---:|
| Raw initializer | 46 | 1.572622 | 0.617583 |
| Aligned epoch 0, scientific selection | 46 | 1.556971 | 0.604104 |
| Saved best-task epoch 8 | 48 | 1.653120 | 0.645379 |
| Final epoch 12 | 47 | 1.645099 | 0.642013 |

The joint candidate curve was `47,39,44,47,46,47,47,48,47,47,47,47`. **All twelve candidates failed both frozen CE and KL eligibility conditions.** Epoch 8 exceeded raw CE by 5.119% and native KL by 4.501%. Its task count meets the original-native five-percentage-point budget, whose minimum is 45/64; the scientific eligibility conditions are stricter experimental constraints.

Training feature loss fell from 0.022195 in epoch 1 to 0.004516 in epoch 12, while mean endpoint loss fell from 0.047055 to 0.025137. These training trends do not demonstrate better development feature alignment or task generalization. Epoch 8 gained five answers and lost three against aligned epoch 0; family counts were ARC 11, BoolQ 13, HellaSwag 14 and WinoGrande 10, each out of 16.

The previous thirteen-block run, without this added feature loss, had already **observed 49/64 at epoch 5**, but its ineligible weights were not retained. Relative to those recorded predictions, the new 48/64 candidate gains three and loses four. Consequently, this run does **not** establish that feature matching improved the task trajectory. Keeping a separate best-task artifact and prospectively allowing its acceptance are additional policy changes, distinct from the feature objective. The original scientific checkpoint and results remain intact.

The separate seed-verification root is `data/experiments/jspace_3b_feature_seed/20261115_v1`, fingerprint `c68320b7246bc04dcafe20a6feffb263e1354270448903944409f011f2d3c2a8`, lease `2df5258d6fa2`. Its task-based acceptance meets the native 45/64 minimum without rewriting scientific selection. No held-out confirmation, all-36-block result or inference-speed claim follows. Artifact/source checks, full update history, training-only calibration and both fresh replays passed independent audit.

Evidence hashes:

- Scientific checkpoint: `e12dad834ef48acf76fce441fbb919ab2749cd5c707396df75018646805e0e8e`.
- Saved best-task checkpoint: `abc72e97e8f800517bc1b106a86bd740013647d90537fd68bcb7576bb0de9eec`.
- `train/result.json`: `e20f89285e90dbf0016b474ab6af5073105bca004d3941e3f043830e06112938`.
- Scientific `verify/manifest.json`: `28b04c4116e7149ffa23ceabaebe7a4833bcd77e824b702c138c687b96a568ce`.
- Separate seed `run/manifest.json`: `fdce4321e1ee849a58b48689f72fb3590879caacb23c4506f09fb29413e84112`.
