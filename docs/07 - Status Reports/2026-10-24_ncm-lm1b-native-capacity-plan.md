# Fully trainable native NCM: fixed16 TRAIN learnability control

2026-10-24. Hypothesis `ncm-lm1b-native16-TRAIN-capacity-v1`.
This is an optimization/capacity diagnostic after three0/128source-only failures.
It does not replace the actual source-disjoint95%task objective. No development
or final case enters this control, and no semantic judge runs in its workflow.
Report word-ID fit as a diagnostic only, never model accuracy or generalization.
Review final source-only TRAIN summaries semantically in a separate fixed study,
inheriting the latest completed canonical cache. No concurrent cache branch.

Require native lifecycle33b7a2ac... actually audited, registered and independently
backend verified. Preserve failedv1 and zero-update numerical diagnostic. Use
the unchanged native mechanism permitting FP32concept-logit allclose1e-4 while
keeping hard generation/state exact. No fresh random trainable weights.

From the original690, use only655original_approved_target_unchanged candidates;
the35known potentially noisy revised targets are outside this TRAIN control.
Count complete sources withcl100k, then select by fixed hypothesis/ID hash:
8in192-511,2each in512-767/768-1023/1024-1535/1536-2048. No output/quality-based
source selection or substitutions. Retain all16, exact original targets and
641-code caches, full FP32 original frozen-body aligned source features.

Initialize BOTH pilot and full run from fixedc1f1b7c3... attention plus released
native generator and domain decoder. Initial full trainable hashb4b901f5....
NEVER reuse a two-update native pilot or this16-case model as a large-data
initializer. Keep127732736generator/word-decoder/source-attention parameters
trainable, native encoder/VQ frozen, original Llama body/MLPs unloaded/frozen.

Run a fresh data-path pilot: two updates, micro4, first/longest selected sources
alternating, maxLR5e-5, one micro per update. Require fullstate/optimizer/CPU+CUDA/
sampler+neighborRNG/all16hard-source-only output reload exact, frozen geometry,
both native generator and word decoder actually changed and peak<=8GB.
Then reset to the identical initial weights and optimizer before the full run.

Full: physicalGPU0 RTX3060/nativeBF16, FP32weights, TF32off, micro4accum4,
fixed256updates only. Each update randperm16 visits EVERYcase once, sampler115142.
Neighbor116142, dropout117142. AdamW(.9,.95),eps1e-8,wd.01,clip1. Warm16to5e-5,
cosine to5e-6. Clean.5concept+.7word, noisy.5concept+.3word, each divided4;
first micro EVERYupdate adds unscaled.25selfconcept+1selfword through the tested
hard-forward codebook surrogate. Source-only histories inFP32/eval; restore
native/source dropout afterward. Noise absolute5120+step (max.25), percaseuniform,
root protected. Save every permutation/ID/noise proof/history/loss/gradient norm.

No interim checkpoint selection or DEV look. Only final256 outputs are eligible
for later TRAIN semantic review. Keep all641codes/128wordIDs, full-vocabulary
argmax and originalCLS renderer. Do not pass teacher words/codes to generate.
Strict fresh fullstate/optimizer/allRNG/all16hard-output reload is mandatory.
Freeze source/plan/tests/CPUauditor before launch; record actual child/whole-tool
exits. Independent CPU audit must replay complete sampler/noise/trace/state and
word-ID diagnostics. No95%or speed claim from training or fitting16examples;
all real DEV/final gates remain unchanged and final320remain closed/unhashed.
