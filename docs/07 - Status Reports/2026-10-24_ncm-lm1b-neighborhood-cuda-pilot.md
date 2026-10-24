# Native NCM full neighborhood and self-history CUDA lifecycle

Date: 2026-10-24. Hypothesis
`ncm-lm1b-neighborhood-selfhistory-BF16-lifecycle-v1`. EXPLORATORY SYSTEMS ONLY.

The dual-source-attention clean-objective pilot passed real BF16 updates,
microbatch4, exact FP32 parity/reload and frozen pretrained state at2.50GBpeak.
The neighborhood objective's three CPU tests also pass. No new source-only
summary accuracy exists; the previous student remains0/128, and the domain
codec's52/64 failed its gate. Label revision/qualification proceeds separately.

Run the same native97M generator, measured domain codec and11,796,480 new source
attention weights from FRESH zero-output initialization. Preserve every frozen
parameter and original Llama/MLPs remain absent. Use the same complete first/
longest TRAIN systems sources and padded representative microbatch4. This does
not choose or shrink any future TRAIN/DEV/final population.

Execute exactly2updates on physicalGPU0 RTX3060 with nativeBF16, FP32weights,
BF16autocast forward/FP32loss, TF32off, AdamW1e-3 beta(.9,.95),weight_decay0,clip1.
Dropout seed94142; independent CPU neighborhood generator seed96142. Force the
corruption schedule to step1024 for BOTH systems updates so maximum rate.25 is
exercised. Sample a per-case rate uniformly from0to.25, protect the root, replace
selected non-root indices uniformly among8192codes. Preserve clean labels.

Concept loss averages CE within each of12levels and then equally across levels,
so the one-position root is not drowned by fine levels. Clean contribution is
.5concept+.7word; corrupted-history contribution is.5concept+.3word, with genuine
PAD0 ignored for word CE. Backpropagate each separately to bound activation
memory while accumulating the same gradient sum. No codebook or old-weight grad.

Force source-only self-history on BOTH pilot updates: temporarily eval the
student, generate its own full641concepts from features/mask ONLY in FP32, return
to train mode, and backpropagate.25*(conceptCE+wordCE) against the clean targets.
Store the complete predicted histories used. The later proposed full schedule
uses this auxiliary only on the first microbatch of every4th update after256;
that full campaign needs its own frozen protocol and admitted native targets.

Require finite objectives and gradients, both new attention output gradients and
both source-K/V gradients by update2, no frozen-state drift, and peak<=7.5GB.
Save attention/optimizer/globalCPU+CUDA/neighborhood RNG states. A fresh strict
pretrained initialization plus saved new state must exactly reproduce FP32
teacher logits and deterministic label-free concept/word generation, optimizer
and all RNG streams. Clear both request memories. Use fresh
`20261024_lm1b_neighborhood_cuda_pilot_v1` and preserve all source/input hashes,
full loss/history traces, checkpoint and actual terminal receipt. Any failure
requires a new wrapper/root for changed code. No task95% or speed claim follows.
