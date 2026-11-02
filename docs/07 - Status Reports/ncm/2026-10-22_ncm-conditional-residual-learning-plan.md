# Can the generator learn residual correction under corrupted history?

Date: 2026-10-22. EXPLORATORY `ncm-conditional-residual-learning-v1`.
The frozen codec's target-aware oracle recovered over92% of token losses after
nearby coarse errors. The full generator still scores0/64 actual development
summaries under Terra. The question here is whether the generator can learn the
oracle's conditional correction rule, before spending a larger training budget.
This is a training-mechanism gate, not95% task accuracy or a final model test.

Two sequential, matched-update arms on physicalGPU2, FP32 throughout: static
clean hierarchy targets versus residual-recomputed targets. Initialize both
from the exact final full-curriculum step5,120 checkpoint and its paired
polished codec; restore the same initial generator/adapter state and RNGs.
Freeze the entire parent body and codec. Train only NCM generator/adapter.
Use32existing training labels chosen by SHA256(`ncm-residual-learning-v1:`+id).
Do not use any development or final-test labels.

Sample the same CPU augmentation masks/replacements for both arms: per-level
corruption probability linearly.5to0; root replacements from eight nearest
non-identical code vectors (FP64 distance computation), finer replacements
uniform from their respective vocabularies. At each level, compute an expert
nearest code from target latent minus *actual* preceding corrupted code
contributions. The repair arm uses that expert as its target and uncorrupted
teacher-forced history; the static arm uses its original clean code instead.
Both arms execute the same quantizer operations, and both consume the same
saved noise draws. This changes conditional labels and their corresponding
teacher-forced context jointly. Coarse AR past labels stay clean in both arms.
No target text/latent enters free generation. This is a new extension, not the
paper's published clean-target corruption recipe.

Budget per arm:1,024updates, batch4, dropout.1, AdamW LR1e-4, betas(.9,.95),
weight decay.01, warmup32, cosine floor.1, clip1. Seed42; separate dropout52,
sampling2042, augmentation3042 and fixed evaluation perturbations4042 streams.
Save sampling/noise schedules, optimizer/RNG checkpoints at512and1,024, trace,
and every declared evaluation. One seed is underpowered.

Evaluate32training targets before/after under four conditions: fully free
generation; forced correct root; one forced neighboring wrong root; all four
forced neighboring wrong roots. The latter three know target codes and are
explicitly target-aware diagnostics. Use identical fixed perturbations across
arms, batch4, original decode policy and codec. Primary mechanism gate:
repair's final non-PAD reconstruction with all four roots wrong must exceed
static by at least5percentage points; repair's fully free reconstruction may
lose at most2points relative to its own baseline. Record all other outcomes,
including failed gates. Passing only authorizes a separately designed broader
training experiment; it cannot admit the sealed task test.

First run a separate two-update real CUDA pilot for both arms on four cases.
Require no-noise code parity with the vendor quantizer, unchanged same-level
expert targets under own-level corruption, causal coarse-input behavior,
nonzero generator-query/adapter gradients, finite objective/gradients, unchanged
codec, equal sampling/noise schedules, exact save/reload generation and intended
memory headroom while GPU0 feature extraction runs. Full training requires this
matching pilot. Stop on OOM, nonfinite values, changed dependencies or failed
invariants; no CPU fallback, silent batch reduction or semantic seed retry.

Artifacts: `20261022_residual_learning_{pilot,train}_v1/{static,repair}` under
the paper root. Original sources, checkpoints and negative results remain
unchanged. No actual-summary accuracy, generalization or speed claim here.
