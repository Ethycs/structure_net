# What the four-position loss measures

**Learning one fixed prompt token accounts for 93.65% of the net development CE improvement from the original native model to the accepted twelve-block parent.** Midpoint development CE worsens, while answer accuracy improves from 48/64 to 50/64. The aggregate CE improvement therefore cannot be interpreted as an equivalent improvement in task performance.

This extends the [cached teacher loss audit](2026-11-15_qwen3b-cached-teacher-loss-audit.md). It preserves that audit's producer and output directory unchanged and adds a separate position-level result. All 512 training and 64 development cases were checked using cached logits only: no model forward, GPU, optimization, held-out access, or objective change. `confirmed=false`.

## The positions and their targets

The original [capture source](../../../experiments/structure_net/jspace_full_capture.py), lines 218–219, uses `[0, floor((n−2)/2), n−2, n−1]`, where `n` is the prompt length. Lines 254–255 assign the following input token as the first three targets, and the correct answer-label token as the fourth. These are not four randomly sampled answer-bearing positions.

The first target is token **8948, `system`**, in all 576 cases: the first input token is the chat delimiter `<|im_start|>`. The penultimate target is token **198, newline**, also in all 576 cases. These are supplied prompt structure. The final target is the correct label `A`, `B`, `C`, or `D`.

All values below are full-vocabulary CE in natural logarithms at temperature 1. Each position contributes exactly one quarter of the aggregate CE.

| Split / cached model | First: `system` | Midpoint: prompt next token | Penultimate: newline | Answer label |
|---|---:|---:|---:|---:|
| TRAIN / native | 15.30284 | 3.69924 | 0.00004450 | 2.71952 |
| TRAIN / accepted parent | 0.01481 | 0.24111 | 0.00000023 | 0.08942 |
| DEV / native | 15.33368 | 3.97166 | 0.00000410 | 3.28227 |
| DEV / accepted parent | 0.01488 | 5.62276 | 0.00000020 | 0.59293 |

The first token's development CE reduction contributes **3.82970** to the aggregate reduction of **4.08926**. The negligible newline contribution brings their combined share to **93.65265%**. On training cases their combined share is 71.51877%.

Midpoint CE shows a clear split gap. Native→parent full-vocabulary midpoint top-one accuracy changes from **200→484/512** on training cases but **23→21/64** on development cases. The accepted parent's midpoint loss contributes **90.24483%** of its total development CE. Consequently, the existing aggregate CE screening criterion is now dominated by this sampled prompt-token prediction. It remains the frozen criterion; this audit changes neither selection nor training.

Answer-label CE improves on development data from 3.28227 to 0.59293, but answer accuracy changes by only two cases. The preceding audit's very low native answer entropy is consistent with confident native errors; CE can improve substantially by changing confidence without a comparably large accuracy gain. This interpretation does not establish the behavior of the current student or independent held-out cases.

## Target frequencies

| Position | TRAIN | DEV |
|---|---|---|
| First | `system`: 512/512 | `system`: 64/64 |
| Penultimate | newline: 512/512 | newline: 64/64 |
| Answer | B: 190; A: 166; C: 84; D: 72 | B: 30; A: 20; C: 8; D: 6 |
| Midpoint | 308 distinct targets | 52 distinct targets |

The most frequent training midpoint targets are `.` (26), `.\n` (22), ` the` (21), `,` (11), ` a` (11), ` of` (8), `The` (8), `B` (7), ` to` (7), and ` is` (7). Development midpoint targets are mostly unique; the largest counts are `.\n` (5), ` a` (4), ` the` (4), `,` (2), and `.` (2). Thus the midpoint is neither a fixed structural token nor a consistent semantic evaluation target.

All target frequencies, decoded tokens, per-target mean CEs, per-family aggregates and 576 per-case CE vectors are saved in the result. Among development families, midpoint CE worsens for ARC, BoolQ and HellaSwag, but improves for WinoGrande. That heterogeneous behavior is obscured by the very large improvement on the fixed `system` token.

## Reproduction and provenance

Producer: [jspace_3b_cached_position_audit.py](../../../experiments/structure_net/jspace_3b_cached_position_audit.py). Output: `data/experiments/jspace_3b_cached_teacher_audit/20261115_v2/`. Runtime: **15.28 seconds**, one CPU thread, niceness increased by 10. The producer authenticates the completed v1 result, its source, native and parent cache identities, original row mapping, and pinned local tokenizer files. It checks the exact four-position formula and every target. Each newly measured four-position CE reproduces the prior audit under `math.isclose(rel_tol=2e-6, abs_tol=2e-6)`; no new logit quantization is applied.

The prior audit records the full input hashes and accepted parent checkpoint SHA. This extension uses the identical inputs and records the tokenizer revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`. Only the local tokenizer is instantiated; no model is loaded.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PIXI_CACHE_DIR=/tmp/structure-net-pixi-cache UV_CACHE_DIR=/tmp/structure-net-uv-cache \
pixi run --locked --offline -e ncm python -m experiments.structure_net.jspace_3b_cached_position_audit \
  --root /tmp/jspace-cached-position-audit-new \
  --prior-audit data/experiments/jspace_3b_cached_teacher_audit/20261115_v1 \
  --diagnostic-root data/experiments/jspace_3b_hidden_feature/20261115_v1
```

| Evidence | SHA256 |
|---|---|
| Position-audit producer | `4daee864e060f6f3b1c64eadd7f9fc5402818c5dcccce248423b8c63ecb8e224` |
| Position-audit `plan.json` | `0d845691c5c35b1c7de67d48f01f05bde8e6d8bc07da3d0e8ecf54987f606ff0` |
| Position-audit `result.json` | `67aa60ef0d1c9dc2434a903bfc6d2828afcafa4fa7fa590b096ccd0c0129730a` |
| Position-audit `manifest.json` | `7df7592ae3b9df5373c4739d9cf51a0e33b80107f05ef315c06c5b9321033ba7` |
| Preserved v1 `result.json` | `ca597304fbac655fbdfc170856892f19e1b53aa0de2795a51a977d28adb752bf` |
