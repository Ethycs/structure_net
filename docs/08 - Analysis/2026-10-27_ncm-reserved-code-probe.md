# Reserved summary codes: teacher forcing versus free generation

Classification: `reserved_teacher_prediction_also_weak`. Application semantic DEV remains0/128; this diagnostic does not measure summary accuracy.

| Source condition | Teacher code agreement | Rollout code agreement | Gap | Root agreement | Native-word agreement |
|---|---:|---:|---:|---:|---:|
| matched | 61.77% | 11.89% | 49.89% | 4.69% | 8.06% |
| shuffled | 61.65% | 11.88% | 49.77% | 1.56% | 7.46% |

All64 previously reserved component summaries were included in their original order. The public epoch2 checkpoint, original Llama features, codec and source adapters remained fixed. Teacher forcing supplies correct earlier code levels; free generation receives only source features. Shuffling cyclically replaces the source while retaining the original target for diagnostic scoring.
Teacher and rollout root predictions agreed for every condition. The frozen codec reproduced every original641-code target exactly. The first/longest-source pilot repeated exactly and matched the full run. Model hashes remained unchanged, no gradients were created, and GPU allocation stayed within4.9GB.
Source identities were checked against all1376 reviewed and162353 public TRAIN pairs. Reviewed TRAIN text overlap was rechecked; the independently audited public full-text screen was reused only after verifying its binding to the exact64 component source file. These sources have been inspected in earlier component experiments and are not a sealed test.
The descriptive exposure-error pattern requires at least90% teacher code agreement and a gap of at least20 percentage points. Teacher agreement below90% is classified as weak even with correct earlier codes. These thresholds diagnose an error pattern; they do not identify a unique cause. Different valid summaries may have different codes, and native-word agreement includes non-padding special tokens.
Inference used physicalGPU0 RTX3060, FP32, TF32 off and locked Pixi ncm while the unchanged epoch4 curriculum ran on the RTX2060. No training, semantic judge requests, sealed-final access or speed measurement occurred.
Detailed per-level counts and raw outputs: `data/experiments/ncm_paper_regime/20261027_reserved_code_probe_run_v2`.
