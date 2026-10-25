# Sharpen the bounded classifier before another learnability control

V2 completed all 512 updates on the fixed 16 TRAIN diagnostics and strictly
reloaded model, optimizer, RNG and hard outputs. Its word CE fell from 7.8500
(first 64 updates) to 5.1169 (last 64), a ratio of 0.65184. This fails the
prospective 0.5 threshold. All outputs changed and were nonempty, but the text
remains broken. Preserve the checkpoint and every output; do not launch v2 full
training. Its 6,308/10,256 code agreement and 229/672 active word-ID agreement
are TRAIN diagnostics, never complete-summary accuracy.

A separate target-aware TRAIN geometry control found that even queries placed
exactly at their target vectors have code CE around 3.3–4.2 on non-root levels
with scale 4. At scale 16 those values are around 0.01–0.11, and nearly every
target wins argmax. Within each native book, no vector is exactly zero or
byte-identical to another; zero FP32 norms in the earlier report came from underflow. Do not mask
or collapse IDs. The target-aware control measures classifier geometry, not
source-only prediction or an attainable semantic score.

V3 changes only the fixed bounded distance multiplier from 4 to 16. Classifier
vectors remain phi(v)=v/(1+norm(v)), learned queries stay in the unit ball,
and the positive copy preference still preserves every input code at zero
residual. Scores now lie in [-64,8], avoiding the raw root-distance explosion
while permitting sharper distributions for the soft-latent gradient surrogate.
This is a hypothesis about gradient conditioning, not a demonstrated fix.

Use new v3 source files and run roots, fresh identity initialization, the same
732 engineering TRAIN pairs, fixed 16 TRAIN diagnostics, seeds, model size,
word/code/latent objective, optimizer, accumulation and schedules as v2.
No v2 weights initialize v3. All original NCM, parent and MLP tensors stay
frozen; all native IDs and raw decoder lookups are intact. First pass CPU
identity, gradient, mask and extreme-norm checks, then an actual two-update
CUDA lifecycle with strict reload. Only after that run the same 512-update
learnability control. Before full training require last-64 mean word CE at
most half first-64, at least one changed source-only output, zero empty TRAIN
diagnostics, strict reload and an unchanged renderer. Passing is necessary,
not semantic qualification. Inspect decoded TRAIN text before scaling up.
The full-training entry point also requires a passing gate bound to the actual
capacity result hash, plus a recorded review that decoded TRAIN text is adequate
to scale. A completed lifecycle alone cannot authorize the full run.

Original 128 DEV, sealed 320 final and additional 156 engineering test scope
remain unchanged. No new semantic requests until readiness, calibration and
the completed 3,539-first judgment lineage requirements hold. No quality or
speedup claim follows from a code, word-ID, geometry or TRAIN control score.
