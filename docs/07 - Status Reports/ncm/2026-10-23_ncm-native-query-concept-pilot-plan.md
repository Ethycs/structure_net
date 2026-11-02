# Full native-body NCM mechanism lifecycle

Date: 2026-10-23. EXPLORATORY systems-only
`ncm-native-query-concept-lifecycle-v1`. This is a new NCM adaptation, not the
failed five-block decoder, an AR capacity repeat or an unchanged paper trainer.
The native28-block cache/query control passed two real updates with frozen
original weights and no additional decoder. The actual concept prototype has
passed its CPU prior-history, native-path and gradient tests. Its task accuracy
has not been measured. The fixed128-case teacher pipeline is still under audit.

Keep all original3B weights frozen. Train only rank8query Q/K/V/O adapters, five
256-way residual-code heads, six level biases and a zero-output rank16hint of
the current earlier-level estimate. Use existing fixed native3072-wide books,
levels4/8/16/32/64 then128parallel words. Source cache contains raw source prompt
only, excluding the last prompt-token anchor; adapters are disabled in prefill.
Each level forks the immutable source cache. Final queries use shifted earlier-
level reconstructed embeddings plus a current-position hint; no gold word prefix
or current/future-level targets enter the actual word prediction path. No parent
AR generation or verifier is used at inference.

Run first four old independently accepted training cases, seed42, BF16native /
FP32new parameters, SDPA, physicalGPU0, batch2,128native-target activation frame,
two AdamW updates, lr1e-4, weight_decay0.01, clip1. Require7.5GBfree and the terminal
passed native-cache v2 control. Load and verify the fixed concept fit and all
accepted-label evidence. Keep one source batch/cache resident at a time and CPU
validation copies; record expandable-segment allocator configuration.

Update1: root code CE against conditional residual expert at zero history for
the first two cases. Update2: word CE for the next two cases, conditioned on all
five actual predicted concept levels, with target-independent corruption rates
[0.5,0.05,1/30,1/60,0]. Separate corruption seed4042. Every coarse expert is
recomputed against the ACTUAL preceding noisy history; compare its reconstruction
error with clean-history labels at that same history and require no worsening
beyond1e-6. Never feed that expert label into the self-generated history.

Require finite losses/parameters, nonzero new gradients, zero original-weight
gradients, original frozen hash unchanged, books/cache byte-identical, adapter
contexts clean, full generated output/code repeat equality and exact new-module
save/reload. Archive initial and final free128word IDs/codes on four training
cases, including nonsense, unfinished and oversized text. These are systems
probes only; never call token fit, code agreement or this pilot95%summary accuracy.
Save all observations and provenance at
`20261023_native_query_concept_pilot_v1`. Stop on OOM or any invalidity; no silent
CPU fallback, target clipping, output repair or expensive32-case memorization.

A pass admits a separately specified training experiment using adequately
qualified data and generated/noisy histories. Actual source-disjoint whole-summary
grading under the qualified128-token profile is necessary before speed work.
Final256cases stay sealed; current short-source prototype does not establish
the2048-fragment/6300-prompt application range. Original failures remain.
