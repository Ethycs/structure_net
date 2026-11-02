# Complete concept hierarchy pretrained; summary quality remains unproven

Date: 2026-10-23. Hypothesis `ncm-source-hierarchy-bootstrap-v1`.
Classification `source_hierarchy_bootstrap_completed_word_training_and_task_quality_unproven`.
Aggregate `data/experiments/ncm_paper_regime/20261023_source_hierarchy_bootstrap_results_terminal_v2/campaign_results.json`, SHA256 `b4d998c2fa6673ef2e35bed90c7e0adb5ee545477c591ab90850bccb2211eb0f`.

The fixed 16,384-update bootstrap completed on 252 vetted training summaries.
All eight coarse concept levels trained with source features, balanced active
and padded-block losses, root emphasis and uniform conditioning corruption.
The original Llama body was absent; its original weights and MLPs remain frozen.
The original codec, copied word projection/decoder and unused finest head stayed
exact. Peak allocation was 4,272,910,336 bytes; no latency
comparison was performed.

Every sampled case, full source frame, learning rate and corruption-history hash
replayed. Final model step, optimizer steps, sampler/noise RNG and checkpoint
digest matched. The maximum-frame systems pilot and full training have actual
zero-exit child receipts and exact checkpoint/source-only reload checks. Only
the fixed final checkpoint was selected; intermediate checkpoints are recovery
artifacts. The earlier terminal-unobserved pilot remains unchanged, with its
exit unknown. The repeated pilot used identical initial parameters and supplied
the required observed terminal evidence.

This stage has no complete-summary accuracy measurement. The latest graded
word-final summarizer still failed 0/64 short and 0/64 longer development cases.
All prior failures remain failures. Generated-history word training and fresh
source-only task evaluation remain necessary. Both final cohorts stay closed;
the 95% task goal is unmet and no compression or speedup is established.
