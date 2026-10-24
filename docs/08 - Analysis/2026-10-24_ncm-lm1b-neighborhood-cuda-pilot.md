# Native NCM clean, corrupted-history and source-only self-history lifecycle

Date: 2026-10-24. Hypothesis `ncm-lm1b-neighborhood-selfhistory-BF16-lifecycle-v1`.
Classification **systems_neighborhood_selfhistory_BF16_lifecycle_passed_not_quality**.
Campaign `data/experiments/ncm_paper_regime/20261024_lm1b_neighborhood_cuda_pilot_results_v1/campaign_results.json`, SHA256 `9c927a298b063127bcd981d7b1506b3e92e1c6b1a7136d713e8f6ce634c4c65a`.

New source attention now runs in both the released concept generator and all
four word-decoder blocks. Its 11,796,480 new parameters start with zero output
projections; initial FP32 concept AND word logits match the existing model exactly.
The 97M generator and measured task-domain codec remain frozen and unchanged.
The word decoder executes directly so its no_grad convenience API cannot block
gradients to new attention. No original Llama weights or MLPs were loaded.

Two real updates exercised all6clean/noisy/self-history losses at microbatch four,
using BF16 autocast with FP32 weights, on
complete sources of 486 and 2744 aligned feature positions. Both source K/V and
both attention branches received gradients. Root-protected uniform corruption
was independently replayed from seed96142 and all generated self histories were
retained. Concept CE weights the12levels equally. The student generated self
histories from features/mask only in FP32; teacher targets entered losses only.
Peak allocation was
2730607104 bytes on physical GPU0 RTX3060. A fresh load of
the frozen states plus new attention/optimizer/RNG reproduced FP32 teacher logits
and label-free concept/word generation exactly. Request memories cleared.

This is systems evidence only, not summarization quality or a speed measurement.
The source features are SHA-verified cached full-source frozen-body outputs.
The existing domain codec's 52/64 semantic result is below its qualification
gate; this experiment does not rescue it. The previous conditional student
remains 0/128. The pilot's two-update checkpoint is not a qualified summarizer
and must not silently become full-training initialization. Full 95% task
accuracy remains unproven; all final sources stayed closed.
