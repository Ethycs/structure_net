# Cached teacher loss audit: training versus development

The low training CE is present in the accepted twelve-block teacher itself. It scores **512/512 training answers**, with four-position CE **0.08633**, versus **50/64 development answers** and CE **1.55764**. This supports accumulated fitting to the reused training cases as an explanation for small student TRAIN losses. The audit found no target-row or CE-reduction mismatch. It does not isolate the full cause of the generalization gap or measure the current student's training accuracy.

This audit used all 512 training and 64 development cases, with one CPU thread and niceness increased by 10. It read existing logits only: no model forward, GPU, optimization, held-out access, or objective change. Runtime was 47.81 seconds. `confirmed=false`.

## Measurements

CE and entropy use natural logarithms and temperature 1. Four-position CE is full-vocabulary next-token/answer-label CE, not CE normalized over the available choices. Each case and each of its four stored positions has equal weight.

| Cached model / split | Cases | Four-position CE | First-three-position CE | Answer CE | Choice accuracy | Four-position entropy | Answer entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original native / TRAIN | 512 | 5.43041 | 6.33404 | 2.71952 | 403/512 (78.71%) | 2.12679 | 0.02579 |
| Original native / DEV | 64 | 5.64690 | 6.43511 | 3.28227 | 48/64 (75.00%) | 2.11042 | 0.02681 |
| Accepted parent / TRAIN | 512 | 0.08633 | 0.08531 | 0.08942 | 512/512 (100%) | 0.24608 | 0.20525 |
| Accepted parent / DEV | 64 | 1.55764 | 1.87921 | 0.59293 | 50/64 (78.13%) | 0.84748 | 0.37717 |

The accepted parent's development CE is dominated by its three sampled context positions: answer CE is 0.59293, while context CE is 1.87921. Conditional choice CE is almost identical to full-vocabulary answer CE: parent TRAIN 0.089377 versus 0.089418, DEV 0.592879 versus 0.592926. Thus confusing choice CE with full-vocabulary answer CE does not explain the large split gap here.

Original-native answer entropy is unusually small in both splits. Its high answer CE despite 75–79% accuracy is consistent with highly confident incorrect answers. The accepted parent is less sharp at the answer position, even though it fits every training answer. Full-vocabulary entropy averaged across all four positions tells a different story because it includes the context positions.

| Family | Native TRAIN correct /128 | Parent TRAIN correct /128 | Native DEV correct /16 | Parent DEV correct /16 | Parent TRAIN / DEV four-position CE |
|---|---:|---:|---:|---:|---:|
| ARC | 112 | 128 | 14 | 14 | 0.03240 / 1.74671 |
| BoolQ | 104 | 128 | 13 | 14 | 0.03724 / 1.59341 |
| HellaSwag | 102 | 128 | 14 | 13 | 0.03405 / 1.43133 |
| WinoGrande | 85 | 128 | 7 | 9 | 0.24166 / 1.45912 |

The gap is not confined to one family. These are the exact cached parent twelve-block records, whose development accuracy is 50/64; an earlier 52/64 checkpoint is a different reference.

## Which teacher is optimized

The frozen [output_objective](../../../experiments/structure_net/jspace_qwen_parallel_train.py) at lines 167–172 computes:

```text
0.7 * T² * KL(accepted_parent || student) + 0.3 * CE, T=2
```

`train_window`, lines 240–248, passes `batch['teacher']` and reports the **unscaled CE** returned by that function. Accumulation weights each microbatch by its true case count. Packed development uses the same function at lines 274–281. The original native logits are used separately at lines 282–286 to report `original_kl`; they are not an additional term in this joint objective. The feature branch adds its accepted-parent feature loss but does not change that teacher choice.

The existing name `accepted_original_kl` can obscure direction: the implementation evaluates `output_objective(parent_logits, native_logits, ...)`, so it means **KL(native || accepted_parent)**. Both directions were computed explicitly here, at temperature 2 and averaged over all four positions:

| Raw KL at T=2, without T² or 0.7 multiplier | TRAIN | DEV |
|---|---:|---:|
| KL(accepted_parent || native) | 0.77455 | 1.00027 |
| KL(native || accepted_parent) | 0.41451 | 0.58885 |

The accepted parent already departs from native logits while fitting the training labels. Continuing to match that parent can inherit its training fit and its departures from native. Whether an additional native-logit constraint improves generalization is a separate prospective intervention; this audit does not implement or recommend silently adding it.

## Method and provenance

Executable method: [jspace_3b_cached_teacher_audit.py](../../../experiments/structure_net/jspace_3b_cached_teacher_audit.py). The result is `data/experiments/jspace_3b_cached_teacher_audit/20261115_v1/result.json` (856,835 bytes), containing aggregate, per-family and all 576 per-case records. `plan.json` records input identities, existing authenticated hashes, the explicit original row map and parent checkpoint SHA. Large banks were streamed in batches of 4, without loading them fully into RAM or rehashing unrelated model/Jacobian files.

Every case's first three targets were checked against `input_ids[loss_position+1]`; the final target was checked against the correct answer-label token. Offsets, lengths, four positions and split membership were checked. All 576 original-row mappings are identity mappings. Logit values are the original BF16 bit patterns, promoted to FP32 for softmax/CE; no new quantization was applied. Gathered negative log probabilities were checked against the exact frozen `F.cross_entropy` reduction in every batch. The separate [native-coordinate audit](2026-11-15_qwen3b-native-feature-coordinate-audit.md) establishes that cached logits are post-final-RMSNorm while boundary 36 is pre-norm.

The CPU-only command is reproducible with a **new** output directory; existing outputs are never overwritten:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PIXI_CACHE_DIR=/tmp/structure-net-pixi-cache UV_CACHE_DIR=/tmp/structure-net-uv-cache \
pixi run --locked --offline -e ncm python -m experiments.structure_net.jspace_3b_cached_teacher_audit \
  --root /tmp/jspace-cached-teacher-audit-new \
  --diagnostic-root data/experiments/jspace_3b_hidden_feature/20261115_v1
```

| Evidence | SHA256 |
|---|---|
| Audit producer | `bee167c73b1c8124d75be163709f66d50907f40c21576fc50829df43b92fef9e` |
| Audit `plan.json` | `64393c934e2e30046b7bd1ec620b1b76aa99aa3ab5837c4dddc00dd85ef8e09f` |
| Audit `result.json` | `ca597304fbac655fbdfc170856892f19e1b53aa0de2795a51a977d28adb752bf` |
| Audit `manifest.json` | `b810fbae8300c5f1a92cc1b916117b271f7f48b3555ae2e53aad65380eaf0956` |
| Original `jspace_full_compact/20261113_capture_v1/capture/items.json` | `b89252aab3291f42b45ab1b9ff962c6e41c6ac6f74fbe8925240e91329c66351` |
| Original native `teacher_logits.npy`, same capture | `95c30a2f18408c29aa453a2772585c0d31eebe6ed8bed0fd5c9c8fa4626248e8` |
| Accepted `jspace_3b_staged_campaign/20261115_v1/stages/block_23/capture/packed_teacher_logits.npy` | `d31b93a5c5fa11bdf35b389919e81e5df18aaa4f0917ea752edde383cb5142ab` |
| Accepted twelve-block parent checkpoint | `db53db32597ebb4947e38f7324e31e6eb9c7686f491a9e8b9745b0c43bca8292` |
