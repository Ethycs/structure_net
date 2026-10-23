# No-op memory correction on an intact native decoder

Date: 2026-10-23. EXPLORATORY systems-only
`ncm-intact-memory-lifecycle-v1`. This is not a summarization-quality test or an
NCM model. The intact1B baseline is readable64/64 but only50/64 summaries pass.
The failed five-block readout and deferred parallel capacity run stay unchanged.

Validate `IntactMemoryReadout` on the verified official intact16-layer1B model.
All native weights, including MLPs and tied word head, stay frozen. Add only
FP32 width256/eight-head attention over frozen3B per-token source features and
a target-independent seeded synthetic concept memory. Zero-initialize its
output projection; bound the correction L2 norm by0.1 of the native hidden norm.
This preserves native raw inputs and language blocks. Synthetic concepts are
only a shape/gradient control, never purported NCM predictions or quality data.

Use the first four previously accepted training cases, their fixed cached
source features and the original native source prompt. Verify public-source,
label-acceptance and original feature-producer fingerprints. No development or
sealed-final optimizer updates. PhysicalGPU2, FP16 native model/SDPA, FP32 new
parameters, seed42, batch2, two AdamW updates at1e-4, clipping1. Full128-native-
token supervised loss/activation ceiling, unchanged target masks. Require3GB
free, stop on OOM, nonfinite loss, identity or gradient-routing failure.

Before updates require exactly equal baseline/conditioned logits and32-token
greedy output for two training sources. Both are systems probes only. Save all
raw IDs, not a repaired summary. After two updates require nonzero new-module
gradients, no base gradients, native full hash unchanged, no residual hooks,
immutable cached features, correction bound and finite parameters. Save/reload
the new module/optimizer and require exact repeated logits. The init no-op is
not claimed to guarantee quality after training. Record peak memory and all
checks at `20261023_intact_memory_pilot_v1` with hashed producer/plan/runtime.

CPU contracts already passed3/3. Run this real GPU lifecycle before any future
use; do not scale training on the old rejected teacher data. This pilot cannot
admit a hybrid decoder as paper replication or establish a speed improvement.
