# Native NCM source-attention attachment: CUDA lifecycle

Date: 2026-10-24. Hypothesis `ncm-lm1b-source-attention-native-CUDA-lifecycle-v1`.
Classification **systems_native_NCM_source_attention_lifecycle_passed_not_quality**.
Campaign `data/experiments/ncm_paper_regime/20261024_lm1b_source_attention_cuda_pilot_results_v1/campaign_results.json`, SHA256 `c196a56ed414aea18f3d4ee2350487179d9507fc8cfe26b65efb8c0f449b9a6a`.

The real released 97M LM1B generator and 111M codec accepted the new source
attention. Zero-initialized output adapters preserved full 641-position concept
logits exactly on two TRAIN cases with 486 and 2744 full source-feature positions.
Two real gradient updates reached only the 8,257,536 new attention parameters;
released generator/codec states stayed unchanged. Native prefix, learned root,
post-upsampling convolutions and shared output heads remained intact.

A fresh released-model load plus adapter/optimizer/RNG restore exactly reproduced
teacher logits and deterministic source-only concept generation. Source memory
cleared between requests. Peak allocation was 1385227776 bytes
on physical GPU0 RTX3060, FP32/TF32 off. The original Llama body was absent and
its SHA-verified cached TRAIN features supplied the source memory.

This is systems evidence only: two cases, two updates, no semantic judging, no
new source-only task accuracy, no trained summarizer or speed claim. The failed
codec branches and previous conditional student's 0/128 result remain unchanged.
All final sources stayed closed. The full 95% task goal remains unmet.
