# Attention-only scope correction

Date: 2026-09-09. This amendment supersedes the all-weight accuracy optimization
scope without changing its historical protocol or saved artifacts.

The user clarified: “we only need to modify attention” and “We shouldn't need
to retrain the fully connected layers right?” Only attention may change.
Feed-forward/MLP layers, embeddings, normalization and the language-model
output head must remain frozen. Gradient propagation through a frozen module
does not authorize updating its parameters.

The all-weight search at
`data/experiments/tinyllm_moba_accuracy95/20260909_search_v1/` was stopped with
SIGINT (exit 130). Its first candidate completed 1600 updates and missed the
two-consecutive-check selection rule. The second candidate was interrupted.
Best observed validation accuracy was 98%; neither final test was evaluated.
The exact preserved validation records and producer fingerprint are listed in
`scope_correction_stop.json`. These results cannot establish success under the
corrected attention-only requirement. No checkpoint from this search may
initialize or be selected for the attention-only follow-up.

The original completed MoBA campaign remains unchanged. It tested a dense-mask
reference, did not confirm its distillation gate, and established no speedup.

## Corrected follow-up

Initialize from the original pretrained full-attention checkpoint, not a parent
student whose MLP has already been fine-tuned. Freeze every parameter, then
enable only each attention module's Q/K/V and output projections. Optimize only
that explicit parameter list. Hash all frozen tensors before training and after
each evaluated checkpoint; fail if a frozen tensor changes or acquires a
gradient. Save the trainable names, counts and frozen hashes in direct evidence.

Retain the existing train/validation/historical/fresh grammar splits, likelihood
scoring, k=2/block32/native256 attention budget, natural-text replay, NLL guard,
98% consecutive-check validation selection and 95% final accuracy target.
The previous validation outcomes are known; this remains exploratory task
adaptation. Both final tests remain unobserved. Use a new hypothesis and artifact
root, and repeat the CPU/frozen-weight and real-CUDA pilot before training.

For Qwen3-8B, freeze the MLP/backbone and adapt attention or attention adapters.
The public configuration has 36 layers, 32 query heads, 8 KV heads and head
dimension 128. A meta-device instantiation counted 8,190,735,360 parameters:
16.38 GB of BF16 weights alone. Freezing avoids optimizer state and parameter
gradients for the backbone but does not eliminate its forward-pass storage.
Quantization/offloading or a larger GPU is still needed on the available 12 GiB
RTX 3060. The existing Qwen inference API cannot modify attention internals.

Actual acceleration requires an efficient sparse kernel and matched benchmarks
against optimized full attention. Report attention-only and end-to-end timing,
prefill and decode separately, at fixed hardware/precision/batch/context length.
The current reference and short agreement tests establish no speedup.

Sources checked on 2026-09-09:
[Qwen3-8B configuration](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json),
[official MoBA implementation](https://github.com/MoonshotAI/MoBA),
[FlashAttention](https://github.com/Dao-AILab/flash-attention).
