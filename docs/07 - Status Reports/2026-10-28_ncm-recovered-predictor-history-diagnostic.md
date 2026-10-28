# Recovered predictor history diagnostic

Exploratory hypothesis `ncm-recovered-predictor-history-diagnostic-v1`. The
completed step6416 predictor still gives poor summaries on the four recovered
TRAIN controls, while the fixed improved standalone decoder reconstructs their
reference codes exactly. This experiment separates correct-history learning
from accumulated rollout errors. It does not amend any semantic evaluation gate.

Use all four original preselected controls, in original order: first, shortest
source, longest source, longest target. No new example selection. Strict-load the
audited final checkpoint `edc88690abaa22b29989be5042269d86f011483a76c8f7b473e4e21b41e30799`
and verify full model hash `e2c17018d5a1d38c867854d009bc3fc895f2bee566b338e7719adb539ba84205`.
Use complete original FP32 source features and singleton-produced reference
codes, joined through bound catalog/metadata/feature indexes. Read only the four
selected feature payloads, verifying their packed and decoded byte hashes. Full
corpus validation already belongs to the completed producer and training audit.

For each source, recompute source-only generation without target-code arguments.
It must reproduce all 641 saved final codes exactly or stop with the mismatch
preserved. Measure packed next-scale prediction using correct history, the
model's own generated history, deterministic random history (CPU seed42+index),
and correct history with the next control's complete source substituted cyclically.
Report per-level cross-entropy and reference-code agreement; neither is semantic
summary accuracy. Save raw predicted codes and root logits.

Compare each same-source packed root to explicit root generation with no history.
Require allclose(rtol=1e-4,atol=1e-4) and identical argmax. Packed self-history
argmax codes must reproduce explicit rollout. Record an execution-consistency
failure if these controls fail; do not diagnose an exposure gap from an invalid
comparison. Otherwise the teacher/free-history gap and source-swap effect are
descriptive mechanism evidence. No minimum effect is called confirmatory success.

Run in locked Pixi, physicalGPU1 RTX2060, FP32, TF32off, deterministic algorithms,
batch1, no gradients or optimizer, peak allocated GPU limit4.9GB. CPU contracts
precede launch. The four-case experiment is its own bounded inference pilot,
including the longest3464-position source. No additional concurrency on GPU1.
Hash and preserve the producer, inputs, environment and outputs; do not alter
the original checkpoint or artifacts. Preserve any exception and use no automatic
retry. New root `20261028_recovered_predictor_history_diagnostic_v1` under the
NCM artifact directory. The entry point is
`experiments.structure_net.ncm_recovered_predictor_diagnostic`.

There are no held-out cases, semantic judge calls, training updates, speed claims,
or Final320 access. This diagnostic cannot establish the95percent objective.
