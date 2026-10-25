# Train source-attention repair on complete engineering summaries

Use only the 732 new engineering TRAIN pairs across eight source components.
The historical4,949matched targets are extracted atoms/introduction fragments,
not complete summaries. Their binding audit is preserved, their prediction run
was stopped with actual143, and the unexecuted5,681pair proposal is archived at
`20261025_core_target_task_mismatch_v1`. No student update used those atom labels.
All512reserved components, original128DEV, sealed320final and156engineeringtest
targets stay outside gradients. Existing historical checkpoint exposure remains.

Require all engineering source features, independently replayed target codes
and actual frozen3911generator mistakes to complete first. Store all641predicted
codes, native128word outputs and full source bindings. Teacher labels remain
unchanged, unreviewed existing summaries, not accepted evaluation gold.

Freeze original Llama body/MLPs and all selected NCM generator, encoder/VQ,
decoder and source-adapter weights. Train only the existing3,778,049parameter,
two-block width256corrector. It receives actual predicted codes and complete
detached source features. Initial identity must preserve original hard codes
AND decoded words. Oracle codes/words enter losses only, never prediction.

Loss: equal-level code CE +0.05soft-native-latent MSE +0.5native word CE through
the frozen word decoder. Word forward uses actual hard repaired codes; backward
uses the existing soft-codebook straight-through gradient. No frozen parameter
or source feature may receive gradients. This aligns correction with text,
but no code/word loss establishes semantic accuracy.

First run a two-update lifecycle, micro1/accum2, LR1e-4 on first/longest source
inputs, with exact model/optimizer/all-RNG/output reload. Then a512update,
micro1/accum4TRAIN learnability control on16fixed cases. Select first/longest
from each of eight TRAIN components and fill16by source-ID hash. Its weights
never initialize the full run. Full run starts at the original identity seed,
uses4,096updates, micro1/accum8, uniform replacement over all732cases, AdamW
betas(.9,.999), eps1e-8, decay.01and norm clipping1. LR3e-4, warmup128then cosine
to3e-5; learnability warmup32. Model seed129142, sampler134142, dropout135142.
FP32, TF32off, GPU2, allocation limit4.9GB and2GiB bounded lossless host cache.
No source clipping, precision fallback or DEV-selected checkpoints. Preserve
sampled IDs, losses, gradients, intermediate checkpoints and every failure.
For repeated renderer loads, bind the actually completed baseline run and verify
the exact selected checkpoint, strict state load, native model contract and full
model tensor hash. Do not re-run unrelated historical training/archive audits on
every optimizer experiment; their already verified frozen tensor identity is
unchanged. No completed producer or its lineage guard is modified.

Fixed16TRAIN diagnostics report code changes and raw summaries as learnability
only. Save a separate target-aware TRAIN oracle-code decoding control on these
same16cases to check the frozen representation/renderer ceiling; it is never
model prediction or semantic task accuracy. Evaluate fixed final weights on original128DEV and the additional156
engineering test inputs, with unchanged baseline and source-disabled correction
controls. In the ablation the original word decoder still reads the true source.
Every empty output remains in its denominator. Bind the completed3,539semantic
firsts and uncertain allowances, require service readiness before fresh requests,
and preserve all old semantic negatives. The unchanged128token whole-summary
faithful/salient/readable criterion and original sealed qualification remain the
95%goal. Engineering test performance or TRAIN fit cannot replace that scope.

No added attention pass is a speedup claim. Measure complete source-to-summary
cost only after semantic quality qualifies.
