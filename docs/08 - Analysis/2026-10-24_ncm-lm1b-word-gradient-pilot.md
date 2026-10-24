# Word-loss gradient through native NCM codebooks: systems result

Hypothesis `ncm-lm1b-word-gradient-lifecycle-v1`. Classification **systems_hard_forward_exact_word_gradient_restored**.
Campaign `data/experiments/ncm_paper_regime/20261024_lm1b_word_gradient_cuda_pilot_results_v1/campaign_results.json`, SHA256 `18bdccb38ce68e0df79886a58a455917c49bc6d5924cdac2d70a639f5e34cc24`.

On two complete TRAIN sources, the hard and surrogate paths produced
bitwise identical FP32 word logits at every position. Saved gradient tensors
show that ordinary hard-code word loss reaches only word-source attention;
the surrogate also reaches concept-source memory and adapters. The CPU
auditor replayed those tensors, complete logits, both noise batches and
the actual two-update optimizer checkpoint. No diagnostic optimizer step
preceded the forward/gradient comparison.

The combined clean/noisy/source-only-history objective completed two real
BF16 microbatch4 updates, with exact fresh attention/optimizer/allRNG/hard
generation reload. Frozen generator and codec fingerprints stayed unchanged.
Peak allocated CUDA memory was 2921136128 bytes.

This is a biased surrogate backward rule with an unchanged hard forward.
It proves a trainable gradient path, not high-quality summarization. Only
the existing 11,796,480 attention parameters updated. The original Llama
body stayed unloaded; all pretrained NCM weights stayed frozen. The
two-update checkpoint is not a full-training seed. Both readout policies
on the evaluated fixed model scored 0/128 under the qualified semantic
judge. New task accuracy is unmeasured, final sources stay closed,
speed is unmeasured and the95%objective remains unmet.
