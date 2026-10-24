# Native NCM fine-tuning mechanism, prospective

2026-10-24. Implementation preparation only, not a trained model or quality result.
The current immutable1024 attention-only continuation and its full128 evaluation
must finish unchanged. This helper is not imported by that experiment.

Read-only CPU inventory found97,065,600 native generator parameters,18,870,656
word-decoder parameters and11,796,480 existing source-attention parameters:
127,732,736distinct candidates for fine-tuning. Generator/decoder parameters
have no registered parameter aliases into the protected encoder/VQ modules.
Estimated FP32 weights+gradients+two Adam moments total2,043,723,776bytes; this
excludes activations and frozen weights, so it is not a measured CUDA fit.

The old attention-only model's train() forces native modules into eval mode.
Its teacher_logits also builds the learned native root query under no_grad.
Simply setting requires_grad would therefore miss a native training path.
The new per-instance helper enables generator/word-decoder gradients and modes,
keeps encoder and all VQ geometry frozen, and constructs only the native learned
root query outside no_grad. Prefix embeddings and teacher-code latents remain
protected. It initializes no weights and leaves source-only hard generation intact.

Focused CPU tests must prove unchanged initial eval forward/weights, root-query
and native-block gradients, protected-codec absence of gradients and correct
train/eval transitions. Before using this in a training study, a fresh real CUDA
pilot must still prove all intended gradients, frozen encoder/VQ fingerprints,
finite full objective, peak memory, exact full native/attention/optimizer/allRNG
and hard source-only generation reload. Preserve any failure. No native fine-tuned
checkpoint exists yet. Larger training still needs its own fixed data/label,
budget and source-disjoint semantic evaluation protocol. The original Llama
body/MLPs stay unloaded during head training. No95%or speed claim here.
