# Full-native655 NCM adaptation from released weights

2026-10-24. Hypothesis `ncm-lm1b-native655-source-training-v1`.
The fixed16control actually finished256updates,100%nonpadding TRAIN-word-ID fit
and strict source-only output/checkpoint reload. Its independent CPU audit
completed. This establishes learnability, not semantic/generalization success.
The separate all16semantic review proceeds independently, with no training
selection by its outcomes. The last actual source-disjoint student remains0/128.

Train on ALL655original_approved_target_unchanged examples from the original690,
preserving order, words, native641codes and full FP32 source features. Exclude
the35known potentially noisy revised targets. No output/quality-based subset.
Labels have not all been semantically qualified; this is explicit exploratory
adaptation. Recheck original TRAIN/DEV exclusions. No new teacher labels or
selected final sources enter training. The full original Llama body/MLPs remain
frozen and unloaded; source features are from the original frozen body.

Initialize pilot and full runs from the SAME originalc1f1b7...attention checkpoint,
released97Mgenerator and domain word decoder. Initial trainable hashb4b901f5....
No new random weights. NEVER reuse the native16overfit checkpoint or a native
two-update pilot. Train127732736native generator/decoder/source-attention
parameters; keep native encoder/VQ geometry frozen. Validate nonzero root-query
gradients and actual root updates, native module changes and protected hashes.

PhysicalGPU0/nativeBF16, FP32weights, FP32hard native source-only histories, TF32
off, peak<=8GB. Full run fixed1024updates, micro4accum4(global16), with-replacement
torch.randint655 microbatch sampler121142, neighbor122142, dropout123142. AdamW
(.9,.95),eps1e-8,wd.01,clip1. Warm64to5e-5, cosine to5e-6. Same working native16
objective: clean.5C+.7W/noisy.5C+.3W, each divided4; first micro EACHupdate adds
unscaled.25selfconcept+1selfword via the tested hard-forward word surrogate.
Noise absolute5120+step,max.25percaseuniform/rootprotected. This experimental
conditional adaptation is not a matched paper replication or exposure claim.

Fresh2update pilot: one micro/update, original655first/longest source repeated
to micro4, constant5e-5. Require strict fullmodel/optimizer/allRNG/source-only
reload and memory/protected-geometry checks. Then reset original initializer
and optimizer for the full1024. Retain the same16TRAIN-only diagnostic sources
for initial and final strict source-only reload comparisons. No interim output
selection, DEV-based stopping or semantic judgments inside this workflow.

Freeze sources/plan/auditor before launch. Preserve all per-step IDs, RNG/noise,
source-only histories, losses and gradient norms. Save only final1024checkpoint.
Record actual child/whole-tool exits; independent CPU audit must replay ALL
sampler/noise/trace/state evidence. Then assess the final checkpoint on the same
128source-disjoint DEV cases under unchanged source-only semantic gates, using
the latest completed canonical cache. Only audited/registered real DEV success
can admit sealed final320. No95%or speed claim from training, TRAIN fit or DEV alone.
