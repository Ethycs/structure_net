# Native NCM fine-tuning CUDA lifecycle v2

The original v1 stopped before updates on an overly strict bitwise concept-logit
guard. A separate zero-update diagnostic measured at most7.63e-5 difference,
with exact words/hard generation, unchanged weights/protected states and the
correct trainable count. The failed pilot and diagnostic are preserved. This
fresh v2 allows FP32concept-logit allclose1e-4, retaining word/hard exactness.

Hypothesis `ncm-lm1b-native-finetuning-lifecycle-v2`. Campaign `data/experiments/ncm_paper_regime/20261024_lm1b_native_finetuning_cuda_pilot_results_v2/campaign_results.json`, SHA256 `33b7a2aca4b0e4b733c2ae6ac75203d189f7f49525d7449316b75b39c590e020`.

The released native generator, domain word decoder and existing source-attention
weights completed two BF16 microbatch4 optimizer updates,127,732,736trainable
parameters in total. Both native generator and decoder actually changed; the
learned root query received nonzero gradients and changed. The CPU auditor
checked full saved gradients, FP32concept-logit allclose, bitwise word-logit and
hard-generation parity, both noise traces, actual optimizer states and native
parameter hashes. Encoder/VQ target geometry stayed frozen.

Exact fullstate/optimizer/allRNG/hard-generation reload passed. Peak CUDA allocation: 5326108160 bytes.

The original Llama body/MLPs stayed unloaded and unchanged. No weights were
randomly reinitialized. This is a systems lifecycle, not semantic task quality
or speed evidence. The two-update checkpoint is not a full-training seed.
Larger-data training and the95%source-disjoint complete-summary objective
remain pending; final sources stay unopened.
