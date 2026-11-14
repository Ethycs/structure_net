# Qwen first-block residual diagnostics

Study: **JSPACE-QWEN-RESIDUAL-DIAGNOSTICS-01**. Status: completed exploratory diagnostic. **Confirmed: false.**
Classification: `residual_rank_failure_dominates_first_block_head_reduction`.

Reducing the first block's residual-update rank with the tested basis damaged development accuracy substantially more than reducing its head width. The full-rank control preserved all 64 native predictions. This localizes a failure in the tested projection before recovery training; it does not establish that other bases or recovery cannot work.

| Condition | Correct / 64 | Accuracy | Wins / losses vs native | Changed predictions |
|---|---:|---:|---:|---:|
| Native | 50 | 78.1250% | 0 / 0 | 0 |
| Rank 2048 / head 128 control | 50 | 78.1250% | 0 / 0 | 0 |
| Rank 2048 / head 64 | 45 | 70.3125% | 2 / 7 | 9 |
| Rank 1024 / head 128 | 27 | 42.1875% | 7 / 30 | 39 |
| Mean-preserving rank 1024 / head 128 | 30 | 46.8750% | 7 / 27 | 36 |

Only block 0 was changed; blocks 1–35, embedding and readout remained native. All arms use softmax, native BF16-rounded weights promoted to FP32, and the same 64 development questions: 16 each from ARC, BoolQ, HellaSwag and WinoGrande. No recovery checkpoint or learned attention metric was used.

The full-rank control's maximum absolute choice-logit difference was 0.00359344, with no changed predictions. Halving rank lost 30 previously correct answers and gained 7; halving head width lost 7 and gained 2. Preserving the training mean in one replacement basis direction raised rank-1024 accuracy from 27/64 to 30/64, but did not repair the failure.

The result concerns this task-Jacobian-derived residual-update subspace at the first block. It does not show that all low-rank subspaces fail, identify one unique cause, or invalidate compression elsewhere in the model. These repeatedly inspected development cases provide no independent held-out guarantee. Runtime from one-block-at-a-time evaluation is not a deployment speed measurement. All-block joint recovery and subsequent enriched-basis interventions are separate studies and are not pooled into this five-arm diagnostic.

Raw aggregate: `data/experiments/jspace_accuracy_recovery_diagnostics/20261114_v1/result.json`; SHA-256 `30733a8ad6a6fdc0befd195062a9f567a26c568358cd8f6c0eb7f3d9579b548f`.
Audited analysis: `data/experiments/jspace_accuracy_recovery_diagnostics/20261114_v1/campaign_results.json`; SHA-256 `e23a178b7785c3458f40a6cda17ce2906cc2300256c3cd3e0089efb90a071779`.

All 320 per-case records, matched identities/labels, paired counts, family totals, choice-logit deltas, producing plans and recorded source hashes were checked. The four ablation producers have archived source snapshots; the native probe has hashed source but no archived probe snapshot. Large native weight shards remain pinned by the original model plan and were not rehashed by this closeout. Nine focused CPU tests passed; the actual-evidence audit verified 105 hashes.

Reproduction after committing the source/report (CPU only, default Pixi):

```sh
CUDA_VISIBLE_DEVICES='' pixi run --locked --offline python -m experiments.neural_architecture_lab.finalize_jspace_residual_diagnostics store --report 'docs/08 - Analysis/jspace/2026-11-14_qwen-residual-diagnostics-registry.md'
CUDA_VISIBLE_DEVICES='' pixi run --locked --offline python -m experiments.neural_architecture_lab.finalize_jspace_residual_diagnostics readback --report 'docs/08 - Analysis/jspace/2026-11-14_qwen-residual-diagnostics-registry.md'
CUDA_VISIBLE_DEVICES='' pixi run --locked --offline python -m experiments.neural_architecture_lab.finalize_jspace_residual_diagnostics closeout --report 'docs/08 - Analysis/jspace/2026-11-14_qwen-residual-diagnostics-registry.md'
```
