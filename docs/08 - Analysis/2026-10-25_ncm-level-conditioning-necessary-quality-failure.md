# NCM training works; the new conditional summarizer still fails a necessary quality gate

The fixed final 1,024-update FP16 native655 checkpoint completed training and
independent state, optimizer, scaler and reload audits. Its source-only inference
also completed and was independently audited on all 128 unchanged development
cases. This establishes functioning training and inference, not successful
summarization or a replication of the paper's training regime.

Hypothesis: `ncm-level-conditioning-native655-necessary-quality-failure-v1`.
This is a retrospective deterministic bound using the existing requirement that
a passing complete summary must be nonempty. It introduces no positive quality
endpoint and cannot rescue the pending complete semantic study.

| Population | Empty outputs | Maximum possible passes | Required passes |
| --- | ---: | ---: | ---: |
| Short development | 6 / 64 | 58 / 64 (90.625%) | 61 / 64 |
| Long diagnostic | 4 / 64 | 60 / 64 (93.75%) | 61 / 64 |
| Combined descriptive bound | 10 / 128 | 118 / 128 (92.1875%) | Both cohort gates must pass |

Even if every nonempty output passed both judges, this candidate could not meet
the unchanged 95% gate. **92.1875% is an upper bound, not measured semantic
accuracy.** Nonempty summaries may still be incorrect. Actual semantic accuracy
remains unmeasured for this checkpoint. All first outputs and failures remain
preserved. No summaries were repaired, filtered, truncated or regenerated for
this analysis.

Checkpoint SHA-256:
`3911eab74db6ad13104f87765196e4212893a3e179ba0283e103e65e78273bec`.
All 128 outputs SHA-256:
`e64f9496bfdb142bd0307530d9cf74295fbdb080018f78363e1b9c73aea61124`.

Evidence lives under `data/experiments/ncm_paper_regime/`:

- `20261024_level_conditioning_FP16_native655_training_train_v1/`
- `20261024_level_conditioning_FP16_native655_training_audit_v1/`
- `20261024_level_conditioning_numeric_outputs_v1/`
- `20261024_level_conditioning_numeric_audit_v1/`
- `20261025_level_conditioning_necessary_quality_results_v1/`

The inference and independent CPU audit both exited successfully. Finalization
checks their actual terminal receipts and every audit-bound artifact, derives
the bound from all ordered archived predictions and checks the original empty
summary rejection rule. Registration requires exact backend readback.

The parent Llama body and MLPs were frozen and absent from this training job;
the NCM generator, word decoder and source attention were trained, while the
codec encoder and VQ geometry remained frozen. Native generation predicted all
641 codes and 128 raw word IDs. This is neither an attention-only training
claim nor an end-to-end latency measurement.

Separate semantic jobs stopped after gateway inference timeouts. Their
transport failures do not explain or erase these local empty outputs. This
analysis makes no new judge requests and does not modify their first-result
cache. The next local diagnostic should separate coarse concept prediction,
later-level error accumulation and decoding using forced oracle prefixes on
the already designated component cohort. Such oracle diagnostics are not
source-only task accuracy. The final 320-case test remains unopened and
unhashed. No speedup or 95% accuracy is established.
