# Frozen Llama source features for fresh NCM development

Date: 2026-10-24. Hypothesis `ncm-clean-development-frozen-features-v1`.
Classification **clean128_DEV_frozen_features_prepared**. Campaign `data/experiments/ncm_paper_regime/20261024_clean_development_feature_results_v1/campaign_results.json`;
SHA256 `1bbdea1c51e879bd755915fc3ed252a839e9e83edc72193d74561d98a4842a00`.

All64 short and64 long development passages retain their complete public source
text, component IDs and spans. Canonical native prompts, GPT2 round trips,
character overlap alignment and source frames replay independently. No labels,
generated summaries or sealed final passages enter preparation.

The original28-layer Llama3.2-3B body and MLPs remain frozen. BF16 source forwards
produce detached aligned FP32 tensors for every GPT2 source token. The first
short passage and longest native prompt pass exact repeated extraction, then
their saved files are reused unchanged in the complete128-case cache after a
fresh parent load. Both workers verify original body identity and the8.6GB peak
allocation limit. Feature extraction follows the observed successful terminal
forecaster workflow, with no simultaneous body/head GPU allocation.

The CPU auditor checks every actual saved tensor's identity, complete shape,
precision, detached state, finiteness and bytes, and every observed child exit.
This establishes evaluation feature readiness. Summary quality remains
unmeasured here. The95% complete-summary goal and required semantic evaluations
remain unchanged; no speed gain is established.
