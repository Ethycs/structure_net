# Source attention in native NCM concept and word decoding

Date: 2026-10-24. Hypothesis `ncm-lm1b-dual-source-attention-BF16-lifecycle-v1`.
Classification **systems_dual_source_attention_BF16_lifecycle_passed_not_quality**.
Campaign `data/experiments/ncm_paper_regime/20261024_lm1b_grounded_cuda_pilot_results_v1/campaign_results.json`, SHA256 `5c53de4c2fd0494ed82bff43c9fe7dd95aa90a8690ad3a0ec93fe4d6f8566a50`.

New source attention now runs in both the released concept generator and all
four word-decoder blocks. Its 11,796,480 new parameters start with zero output
projections; initial FP32 concept AND word logits match the existing model exactly.
The 97M generator and measured task-domain codec remain frozen and unchanged.
The word decoder executes directly so its no_grad convenience API cannot block
gradients to new attention. No original Llama weights or MLPs were loaded.

Two real updates at microbatch four used BF16 autocast with FP32 weights, on
complete sources of 486 and 2744 aligned feature positions. Both source K/V and
both attention branches received gradients. Peak allocation was
2497147904 bytes on physical GPU0 RTX3060. A fresh load of
the frozen states plus new attention/optimizer/RNG reproduced FP32 teacher logits
and label-free concept/word generation exactly. Request memories cleared.

This is systems evidence only, not summarization quality or a speed measurement.
The source features are SHA-verified cached full-source frozen-body outputs.
The existing domain codec's 52/64 semantic result is below its qualification
gate; this experiment does not rescue it. The previous conditional student
remains 0/128. The pilot's two-update checkpoint is not a qualified summarizer
and must not silently become full-training initialization. Full 95% task
accuracy remains unproven; all final sources stayed closed.
