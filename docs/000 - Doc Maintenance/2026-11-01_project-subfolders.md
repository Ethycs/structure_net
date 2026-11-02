# Project subfolders for dated documents — 2026-11-01

> **Correction, 2026-11-01:** the compatibility symlinks described below were a mistake and have been removed at the user's direction. The old flat paths no longer exist, and no script was edited. Scripts that hardcode an old path now fail with a missing file when rerun, or write a new stray flat report instead of updating the real one. Digests remain valid because no pinned file changed. The rest of this note is the original record; statements that the symlinks exist, including "Do not delete these symlinks", no longer apply. Do not repoint a pinned script, because that breaks its lineage check. To rerun one, check out a commit from before the move.

Project-specific documents moved from the flat lanes into `<lane>/<project>/`. Each file's project comes from its filename. `figures/` and `assets/` moved with their project.

## Why the old paths still exist

The old flat paths under `07 - Status Reports` and `08 - Analysis`, plus five undated design and implementation paths, remain as relative symlinks to the new locations. They cannot be retired by updating references, because the code and receipts pin them:

- 631 Python files hardcode these paths. 523 of them are pinned by a digest in the repository or in a `data/` receipt, or hash their own source, so editing them would fail lineage checks.
- At least 2,107 receipts under `data/` record the old paths.
- 644 dated documents are pinned by digest, and preregistrations are immutable by policy.

Do not delete these symlinks. Do not repoint a historical script or receipt at a new path.

## What was and was not rewritten

Links were rewritten to the new locations only in documents that are not pinned and are not preregistrations or amendments. Pinned documents, and all Python files, keep their exact bytes. Links in those frozen documents that point within the same project and lane still work. Those that cross into another lane or project, listed below, no longer resolve. Filenames are globally unique, so `find docs -type f -name <file>` locates each target.

## Rerun behaviour

- Writing a report through a symlink updates the real file. No script was found that replaces a report path by rename, but a future one would turn its symlink into a regular file.
- About 55 scripts call `.resolve()` on a document path, so new receipts from reruns record the new location. Old receipts still verify through the symlinks.

## Links that no longer resolve in frozen documents (116)

- `01 - Design/training-weather/tinyllm-neural-gas-weather.md` → `../08 - Analysis/2026-10-05_training-weather.md`
- `07 - Status Reports/tinyllm/2026-08-07_tinyllm-calibration-degradation-causal-preregistration.md` → `../08 - Analysis/2026-08-07_tinyllm-calibration-degradation-causal.md`
- `07 - Status Reports/tinyllm/2026-08-07_tinyllm-calibration-degradation-preregistration.md` → `../08 - Analysis/2026-08-07_tinyllm-calibration-degradation.md`
- `07 - Status Reports/tinyllm/2026-08-07_tinyllm-nuisance-scalar-transformation-law-preregistration.md` → `../08 - Analysis/2026-08-07_tinyllm-nuisance-scalar-transformation-law.md`
- `07 - Status Reports/tinyllm/2026-08-07_tinyllm-scalar-action-defect-decomposition-preregistration.md` → `../08 - Analysis/2026-08-07_tinyllm-scalar-action-defect-decomposition.md`
- `07 - Status Reports/tinyllm/2026-08-07_tinyllm-scalar-groupoid-defect-preregistration.md` → `../08 - Analysis/2026-08-07_tinyllm-scalar-groupoid-defect.md`
- `07 - Status Reports/tinyllm/2026-08-11_tinyllm-c3-temporal-continuation-readout-preregistration.md` → `../08 - Analysis/2026-08-11_tinyllm-c3-temporal-quotient-d6.md`
- `07 - Status Reports/tinyllm/2026-08-11_tinyllm-c3-temporal-quotient-d6-preregistration.md` → `../08 - Analysis/2026-08-11_tinyllm-c3-temporal-quotient-stage0.md`
- `07 - Status Reports/tinyllm/2026-08-11_tinyllm-frozen-interval-readout-decomposition-preregistration.md` → `../08 - Analysis/2026-08-10_tinyllm-calibrated-architecture-replication.md`
- `07 - Status Reports/tinyllm/2026-08-11_tinyllm-frozen-interval-readout-decomposition-preregistration.md` → `../08 - Analysis/2026-08-11_tinyllm-frozen-cyclic-answer-projection.md`
- `07 - Status Reports/tinyllm/2026-08-11_tinyllm-frozen-interval-readout-decomposition-preregistration.md` → `../08 - Analysis/2026-08-11_tinyllm-frozen-scalar-domain-extension.md`
- `07 - Status Reports/tinyllm/2026-08-11_tinyllm-frozen-interval-readout-decomposition-preregistration.md` → `../08 - Analysis/2026-08-11_tinyllm-frozen-scalar-interface-decomposition.md`
- `07 - Status Reports/tinyllm/2026-08-12_tinyllm-temporal-language-identifiability-preregistration.md` → `../01 - Design/temporal-phase-language-task.md`
- `08 - Analysis/llama32/2026-10-10_llama32-exact-blas.md` → `../07 - Status Reports/2026-10-10_llama32-exact-blas-plan.md`
- `08 - Analysis/llama32/2026-10-10_llama32-exact-deployment.md` → `../07 - Status Reports/2026-10-10_llama32-deployment-input-amendment.md`
- `08 - Analysis/llama32/2026-10-10_llama32-exact-deployment.md` → `../07 - Status Reports/2026-10-10_llama32-lossless-deployment-plan.md`
- `08 - Analysis/llama32/2026-10-10_llama32-exact-speed.md` → `../07 - Status Reports/2026-10-10_llama32-exact-speed-amendment.md`
- `08 - Analysis/llama32/2026-10-10_llama32-exact-speed.md` → `../07 - Status Reports/2026-10-10_llama32-exact-speed-plan.md`
- `08 - Analysis/llama32/2026-10-10_llama32-exact-speed.md` → `../07 - Status Reports/2026-10-10_llama32-exact-speed-storage-amendment.md`
- `08 - Analysis/llama32/2026-10-10_llama32-lossless-compression.md` → `../07 - Status Reports/2026-10-10_llama32-lossless-compression-plan.md`
- `08 - Analysis/llama32/2026-10-10_llama32-lossless-compression.md` → `../07 - Status Reports/2026-10-10_llama32-lossless-deployment-plan.md`
- `08 - Analysis/llama32/2026-10-13_llama32-summary-throughput.md` → `../07 - Status Reports/2026-10-13_llama32-summary-selection.md`
- `08 - Analysis/llama32/2026-10-13_llama32-summary-throughput.md` → `../07 - Status Reports/2026-10-13_llama32-summary-throughput-plan.md`
- `08 - Analysis/llama32/2026-10-14_llama32-fp16-core-q4-tail.md` → `../07 - Status Reports/2026-10-14_llama32-fp16-core-q4-tail-plan.md`
- `08 - Analysis/llama32/2026-10-14_llama32-fp16-core-q4-tail.md` → `../07 - Status Reports/2026-10-14_llama32-q4-tail-operator-amendment.md`
- `08 - Analysis/llama32/2026-10-14_llama32-noise-tensor-continuation.md` → `../07 - Status Reports/2026-10-14_llama32-noise-tensor-continuation-plan.md`
- `08 - Analysis/llama32/2026-10-14_llama32-q4-kernel.md` → `../07 - Status Reports/2026-10-14_llama32-q4-kernel-epilogue-amendment.md`
- `08 - Analysis/llama32/2026-10-14_llama32-q4-kernel.md` → `../07 - Status Reports/2026-10-14_llama32-q4-kernel-layout-amendment.md`
- `08 - Analysis/llama32/2026-10-14_llama32-q4-kernel.md` → `../07 - Status Reports/2026-10-14_llama32-q4-kernel-plan.md`
- `08 - Analysis/llama32/2026-10-14_llama32-tensor-runtime.md` → `../07 - Status Reports/2026-10-14_llama32-tensor-runtime-plan.md`
- `08 - Analysis/llama32/2026-10-14_llama32-tensor-runtime.md` → `../07 - Status Reports/2026-10-14_llama32-tensor-runtime-warmup-amendment.md`
- `08 - Analysis/moba/2026-09-09_qwen8b-moba-feasibility.md` → `../02 - Implementation/qwen-moba-feasibility-reproduction.md`
- `08 - Analysis/moba/2026-09-09_tinyllm-moba-distillation.md` → `../07 - Status Reports/2026-09-09_tinyllm-moba-distillation-preregistration.md`
- `08 - Analysis/ncm/2026-10-21_ncm-independent-quality.md` → `../07 - Status Reports/2026-10-21_ncm-independent-quality-amendment-v2.md`
- `08 - Analysis/ncm/2026-10-21_ncm-independent-quality.md` → `../07 - Status Reports/2026-10-21_ncm-independent-quality-amendment-v3.md`
- `08 - Analysis/ncm/2026-10-21_ncm-independent-quality.md` → `../07 - Status Reports/2026-10-21_ncm-independent-quality-preregistration.md`
- `08 - Analysis/ncm/2026-10-21_ncm-independent-quality.md` → `../07 - Status Reports/2026-10-21_ncm-quality-reference-diagnostic.md`
- `08 - Analysis/ncm/2026-10-21_ncm-llama-distillation.md` → `../07 - Status Reports/2026-10-21_ncm-graph-capture-amendment.md`
- `08 - Analysis/ncm/2026-10-21_ncm-llama-distillation.md` → `../07 - Status Reports/2026-10-21_ncm-graph-runtime-plan.md`
- `08 - Analysis/ncm/2026-10-21_ncm-llama-distillation.md` → `../07 - Status Reports/2026-10-21_ncm-graph-sampler-amendment.md`
- `08 - Analysis/ncm/2026-10-21_ncm-llama-distillation.md` → `../07 - Status Reports/2026-10-21_ncm-llama-distillation-plan.md`
- `08 - Analysis/ncm/2026-10-21_ncm-serving-and-factorization.md` → `../07 - Status Reports/2026-10-21_ncm-batching-plan.md`
- `08 - Analysis/ncm/2026-10-21_ncm-serving-and-factorization.md` → `../07 - Status Reports/2026-10-21_ncm-factorization-plan.md`
- `08 - Analysis/ncm/2026-10-21_ncm-serving-and-factorization.md` → `../07 - Status Reports/2026-10-21_ncm-host-io-plan.md`
- `08 - Analysis/ncm/2026-10-21_ncm-serving-and-factorization.md` → `../07 - Status Reports/2026-10-21_ncm-replicas-plan.md`
- `08 - Analysis/ncm/2026-10-22_ncm-frozen-body.md` → `../07 - Status Reports/2026-10-22_ncm-frozen-body-plan.md`
- `08 - Analysis/ncm/2026-10-22_ncm-frozen-body.md` → `../07 - Status Reports/2026-10-22_ncm-frozen-body-seed-clarification.md`
- `08 - Analysis/ncm/2026-10-22_ncm-memory-task-accuracy.md` → `../07 - Status Reports/2026-10-22_ncm-memory-accuracy-parser-recovery.md`
- `08 - Analysis/ncm/2026-10-22_ncm-paper-regime-control-failures.md` → `../07 - Status Reports/2026-10-22_ncm-paper-recovery-v2-plan.md`
- `08 - Analysis/ncm/2026-10-22_ncm-paper-regime-control-failures.md` → `../07 - Status Reports/2026-10-22_ncm-paper-regime-plan.md`
- `08 - Analysis/ncm/2026-10-22_ncm-recipe-update-test-recovery.md` → `../07 - Status Reports/2026-10-22_ncm-recipe-candidate-recovery.md`
- `08 - Analysis/ncm/2026-10-22_ncm-recipe-update-test.md` → `../07 - Status Reports/2026-10-22_ncm-recipe-test-amendment.md`
- `08 - Analysis/ncm/2026-10-22_ncm-recipe-update.md` → `../07 - Status Reports/2026-10-22_ncm-recipe-control-windowing-fix.md`
- `08 - Analysis/ncm/2026-10-22_ncm-recipe-update.md` → `../07 - Status Reports/2026-10-22_ncm-recipe-teacher-target-amendment.md`
- `08 - Analysis/ncm/2026-10-22_ncm-recipe-update.md` → `../07 - Status Reports/2026-10-22_ncm-recipe-update-plan.md`
- `08 - Analysis/ncm/2026-10-22_ncm-training-competence.md` → `../07 - Status Reports/2026-10-22_ncm-full-curriculum-plan.md`
- `08 - Analysis/ncm/2026-10-22_ncm-training-diagnosis.md` → `../07 - Status Reports/2026-10-22_ncm-diagnosis-cache-amendment.md`
- `08 - Analysis/ncm/2026-10-22_ncm-training-diagnosis.md` → `../07 - Status Reports/2026-10-22_ncm-training-diagnosis-plan.md`
- `08 - Analysis/ncm/2026-10-23_ncm-oracle-codec-semantic-recovery.md` → `../07 - Status Reports/2026-10-23_ncm-oracle-codec-semantic-cache-recovery.md`
- `08 - Analysis/ncm/2026-10-25_ncm-broad-generator.md` → `../07 - Status Reports/2026-10-25_ncm-broad-generator-plan.md`
- `08 - Analysis/ncm/2026-10-25_ncm-broad-generator.md` → `../07 - Status Reports/2026-10-25_ncm-residual-recoding-probe-plan.md`
- `08 - Analysis/ncm/2026-10-25_ncm-broad-teacher-catalog.md` → `../07 - Status Reports/2026-10-25_ncm-broad-teacher-catalog-plan.md`
- `08 - Analysis/ncm/2026-10-25_ncm-codec-roundtrip.md` → `../07 - Status Reports/2026-10-25_ncm-codec-roundtrip-plan.md`
- `08 - Analysis/ncm/2026-10-25_ncm-generated-code-decoder.md` → `../07 - Status Reports/2026-10-25_ncm-generated-code-decoder-plan.md`
- `08 - Analysis/ncm/2026-10-25_ncm-onpolicy-residual.md` → `../07 - Status Reports/2026-10-25_ncm-onpolicy-residual-evaluation-recovery.md`
- `08 - Analysis/ncm/2026-10-25_ncm-onpolicy-residual.md` → `../07 - Status Reports/2026-10-25_ncm-onpolicy-residual-plan.md`
- `08 - Analysis/ncm/2026-10-25_ncm-residual-recoding-probe.md` → `../07 - Status Reports/2026-10-25_ncm-residual-recoding-probe-plan.md`
- `08 - Analysis/ncm/2026-10-26_ncm-example-batching-root-gradient.md` → `../07 - Status Reports/2026-10-26_ncm-public-batch4-root-gradient-repair.md`
- `08 - Analysis/ncm/2026-10-26_ncm-example-batching.md` → `../07 - Status Reports/2026-10-26_ncm-example-batching-codec-control.md`
- `08 - Analysis/ncm/2026-10-26_ncm-example-batching.md` → `../07 - Status Reports/2026-10-26_ncm-example-batching-math-control.md`
- `08 - Analysis/ncm/2026-10-26_ncm-example-batching.md` → `../07 - Status Reports/2026-10-26_ncm-example-batching-plan.md`
- `08 - Analysis/ncm/2026-10-26_ncm-public-source-metadata.md` → `../07 - Status Reports/2026-10-26_ncm-public-source-metadata-audit-plan.md`
- `08 - Analysis/ncm/2026-10-26_ncm-public-source-metadata.md` → `../07 - Status Reports/2026-10-26_ncm-public-source-metadata-plan.md`
- `08 - Analysis/ncm/2026-10-26_ncm-public-training-catalog.md` → `../07 - Status Reports/2026-10-26_ncm-public-training-catalog-plan.md`
- `08 - Analysis/ncm/2026-10-26_ncm-public-xsum-codec.md` → `../07 - Status Reports/2026-10-26_ncm-public-xsum-codec-plan.md`
- `08 - Analysis/ncm/2026-10-26_ncm-public-xsum-codec.md` → `../07 - Status Reports/2026-10-26_ncm-public-xsum-codec-recovery.md`
- `08 - Analysis/ncm/2026-10-30_ncm-pretrained-capabilities.md` → `../07 - Status Reports/2026-10-30_ncm-pretrained-capability-plan.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-charged-character-fixed-decoder.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-charged-character-fixed-decoder-corrective-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-charged-character-fixed-decoder.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-charged-character-fixed-decoder-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-connection-observation-identifiability.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-connection-observation-identifiability-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-constant-acceleration-fixed-operator.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-constant-acceleration-fixed-operator-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-full-local-gauge-closure.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-full-local-gauge-closure-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-gauge-jump-corruption-fixed-decoder.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-gauge-jump-corruption-fixed-decoder-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-gauge-jump-joint-typed-score.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-gauge-jump-evidence-backup.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-gauge-jump-joint-typed-score.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-gauge-jump-joint-typed-score-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-hidden-order-corruption-corrective.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-hidden-order-corruption-corrective-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-hidden-order-corruption-fixed-estimator.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-hidden-order-corruption-fixed-estimator-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-posterior-holonomy-interface.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-posterior-holonomy-interface-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-relational-connection-acquisition.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-relational-connection-acquisition-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-relational-connection-function-class.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-relational-connection-function-class-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-relational-connection-preflight.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-relational-connection-preflight-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-single-frame-corruption-fixed-estimator.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-single-frame-corruption-fixed-estimator-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-switching-corruption-fixed-decoder.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-switching-corruption-fixed-decoder-corrective-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-switching-corruption-fixed-decoder.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-switching-corruption-fixed-decoder-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-temporal-continuation-readout.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-temporal-continuation-readout-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-temporal-feature-trajectory-causal.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-temporal-feature-trajectory-causal-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-temporal-fixed-operator-ceiling.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-temporal-fixed-operator-ceiling-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-temporal-quotient-d6.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-temporal-quotient-d6-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-temporal-sensor-function-class.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-temporal-sensor-function-class-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-temporal-sensor-mechanism.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-temporal-sensor-mechanism-preregistration.md`
- `08 - Analysis/tinyllm/2026-08-11_tinyllm-c3-temporal-sensor-only.md` → `../07 - Status Reports/2026-08-11_tinyllm-c3-temporal-sensor-only-preregistration.md`
- `08 - Analysis/word-games/2026-10-30_word-games-phi4-firewall.md` → `../07 - Status Reports/2026-10-30_word-games-phi4-firewall-plan.md`
- `08 - Analysis/word-games/2026-10-31_word-games-bonsai-naked.md` → `../07 - Status Reports/2026-10-31_word-games-bonsai-naked-plan.md`
- `08 - Analysis/word-games/2026-10-31_word-games-cpu-expert-elimination.md` → `../07 - Status Reports/2026-10-31_word-games-cpu-expert-elimination-plan.md`
- `08 - Analysis/word-games/2026-10-31_word-games-grammar-on-demand.md` → `../07 - Status Reports/2026-10-31_word-games-grammar-on-demand-plan.md`
- `08 - Analysis/word-games/2026-10-31_word-games-grammar-on-demand.md` → `../07 - Status Reports/2026-10-31_word-games-grammar-reduction-amendment.md`
- `08 - Analysis/word-games/2026-10-31_word-games-moe-placement.md` → `../07 - Status Reports/2026-10-31_word-games-moe-experts-only-amendment.md`
- `08 - Analysis/word-games/2026-10-31_word-games-moe-placement.md` → `../07 - Status Reports/2026-10-31_word-games-moe-placement-plan.md`
- `08 - Analysis/word-games/2026-10-31_word-games-phi4-40tps.md` → `../07 - Status Reports/2026-10-31_word-games-phi4-40tps-plan.md`
- `08 - Analysis/word-games/2026-10-31_word-games-phi4-speculative.md` → `../07 - Status Reports/2026-10-31_word-games-phi4-speculative-plan.md`
- `08 - Analysis/word-games/2026-10-31_word-games-qwen-pair.md` → `../07 - Status Reports/2026-10-31_word-games-qwen-pair-plan.md`
- `08 - Analysis/word-games/2026-10-31_word-games-qwen38-moe.md` → `../07 - Status Reports/2026-10-31_word-games-qwen38-moe-plan.md`
- `08 - Analysis/word-games/2026-11-01_word-games-normal-operation.md` → `../07 - Status Reports/2026-11-01_word-games-normal-operation-followup.md`
- `08 - Analysis/word-games/2026-11-01_word-games-normal-operation.md` → `../07 - Status Reports/2026-11-01_word-games-normal-operation-plan.md`
- `08 - Analysis/word-games/2026-11-01_word-games-schema-prefill.md` → `../07 - Status Reports/2026-11-01_word-games-schema-prefill-lifecycle-amendment.md`
- `08 - Analysis/word-games/2026-11-01_word-games-schema-prefill.md` → `../07 - Status Reports/2026-11-01_word-games-schema-prefill-plan.md`
