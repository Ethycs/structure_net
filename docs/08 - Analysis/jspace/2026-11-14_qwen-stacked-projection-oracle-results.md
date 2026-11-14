# Teacher-guided stacked projection oracle

Study **JSPACE-QWEN-STACKED-PROJECTION-ORACLE-01**. Classification: `teacher_guided_native_decisions_preserved`. **Confirmed: false.**

The projected trajectory preserved **64/64 native decisions (100.0% agreement)**. Both the native control and the teacher-guided oracle answered **50/64** correctly; there were **0 gains and 0 losses**. This is a teacher-guided diagnostic, not accuracy of a learned compact model.

The positive result is that the fixed subspaces admit this projected teacher trajectory on these 64 cases while preserving every native decision. The geometry therefore does not inevitably destroy these decisions when supplied with suitable updates. It remains unproven whether the compact attention/MLP branches can learn those updates from their own states, or whether the result generalizes.

At each layer the calculation receives the ORIGINAL model's next hidden state and applies:

```text
h_next = h + ((native_next - h) @ U) @ U.T
```

Here h is the accumulated oracle state, so each correction accounts for previous projection errors. The next native state is supplied at every layer; the learned attention/MLP branches do not produce it. The original readout scores the final trajectory. This is neither deployable inference nor a task-accuracy upper bound: other learned feature trajectories could make different decisions.

| Family | Native correct /16 | Teacher-guided correct /16 | Separate real student /16 |
|---|---:|---:|---:|
| arc | 15 | 15 | 5 |
| boolq | 13 | 13 | 7 |
| hellaswag | 14 | 14 | 7 |
| winogrande | 8 | 8 | 8 |

The final teacher-relative RMS discrepancy was **0.22793865** over valid tokens (equal-case aggregation) and **0.15374000** at answer positions. States and logits need not be identical for decisions to agree. The native control reproduced its reference with maximum logit difference **0**.

The valid-token metric is `sqrt(mean_cases(mean_valid_tokens(||oracle-native||² / max(||native||², 1e-20))))`; the answer metric uses each case's final valid row. All 37 boundary measurements are preserved in the analysis artifact.

For context only, the real assembled student before joint adaptation scored **27/64** in the separate artifact `data/experiments/jspace_recovered_assembly/20261114_v1/all_compact/result.json`. That student result and this oracle answer different questions. Ongoing joint-training epochs are excluded from this stabilized diagnostic.

The bases were fixed before this evaluation. The population contains the original 64 repeatedly inspected development questions, 16 from each family; no untouched confirmation examples were used. This study trains no parameters and measures no deployment speed or compression gain. Its primary registry metric is `teacher_guided_native_decision_agreement`, a fraction; 1.0 means native decision agreement, not 100% task accuracy.

Evidence: `data/experiments/jspace_stacked_projection_oracle/20261114_v1/stacked_oracle_analysis.json`; SHA-256 `4a2d0cad7ad294692a1a4738a44c3868892388073658bc5925f5fe2fb3883b58`. The frozen producer preparation fully authenticated all 36 fit files and all 37 native FP32 boundary banks. Closeout matches their SHA256 pins to the assembly/cache manifests and verifies immutable file identities using existing producer validation, then records current device/inode/size/mtime/ctime. Small result/control metadata and live/archived sources are rehashed. Large bank/model payloads are not repeatedly reread.
