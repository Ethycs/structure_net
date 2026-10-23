# Is the current NCM generator using the frozen parent body?

Date: 2026-10-22. EXPLORATORY `ncm-body-contribution-v1`.
The final full curriculum fits training tokens but its development outputs still
contain invented wording. Actual final-checkpoint grading is separate and
ongoing. A previous joint shuffle showed source dependence but did not isolate
the frozen Llama channel from the native GPT2 embedding channel.

Freeze the final `20261022_full_train_v1/step5120.pt` and its paired polished
codec. No training, model selection or new teacher requests. Measure128training
cases selected by SHA256(`ncm-body-ablation-v1:`+id), without replacement, from
the existing1,999. These are training-fit diagnostics, not held-out accuracy.

Run six matched BF16 inference conditions on physicalGPU0 after the original
parent worker exits: both channels matched; cyclically shifted body features;
zero body features; native source embeddings without the residual; cyclically
shifted native source with correct body; both channels cyclically shifted.
Shift across the full selected cohort, preserving the empirical marginal of
each shifted channel. Zero/disabled channels are explicitly out-of-distribution
stress tests. Preserve each actual native input's padding mask. Neither teacher
text nor target codes enter generation. Targets are used only after generation
to compute coarse/all-code and non-PAD token match and exact-text counts.

A four-case real CUDA pilot must reproduce the original first four training
predictions exactly, exercise all six conditions, preserve all weights and
reload exactly. Full measurement requires a matching passed pilot. The full
run repeats the anchor and records every arm's predictions and counts. Stop
on OOM, nonfinite values, changed inputs or failed parity; no CPU fallback.
Use seed42, batch4, the original code hierarchy, coarse AR and12evaluations.
Do not grade these training examples as model task accuracy, claim that a
small ablation difference proves irrelevance, or infer final accuracy from fit.
Report measured channel dependence with its sample/prototype limitations.
The source-disjoint final test remains sealed and no speed claim is allowed.

Artifacts: `20261022_body_ablation_{pilot,measure}_v1` under the NCM paper root.
Preserve original training and ongoing judge producers without source edits.
