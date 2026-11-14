# Full-model compact pass: findings and next direction

Recorded 2026-11-13T19:16:21.519768-08:00. Study `JSPACE-FULL-COMPACT-01`; completed exploratory
campaign, `confirmed=false`. Registry result `190d7ff50b506b90`.

Applied the supplied offline functional-coordinate compilation and adaptation
idea to all 36 Qwen2.5-3B-Instruct layers, alongside the packed polynomial core.
The complete residual state is retained in common coordinates. The compact
attention group's omitted dynamic coordinates are mean-anchored; retaining the
other native group does not reconstruct them.

| Variant | Correct / 256 | Resident tensor GB | Matched graph tok/s |
| --- | ---: | ---: | ---: |
| Original | 210 (82.03%) | 6.1719 | 41.02 |
| Corrected full attention + MLP, adapted | 88 (34.38%) | 5.5967 | 36.95 |
| All 36 MLPs compiled, native attention | 205 (80.08%) | 5.7501 | 41.01 |

The full configuration saves 9.32% tensor
state and runs at 0.9007x native throughput. It does not
meet the quality or speed objective. The MLP-only branch saves
6.83% state with
no demonstrated speed change. It gains 10 answers and loses 15, for a net loss
of five; this is task accuracy on a reused four-family evaluation pool.

The full-coordinate rotation gives 209/256 and the full-width split-attention
control gives 210/256. Algebraic compilation has a working control. The first
attention implementation omitted the centered value basis's affine mean term;
that implementation and its 103/256 result are preserved. The corrected version
restores the term, fuses the attention/output calls, and repeats all 3,456 updates.
Its initialization scores 93/256 and adaptation finishes at 88/256. The correction
alone does not recover the information lost by this full-stack cut.

The basis retains 92.01% of measured
Jacobian energy. The measurement uses a scalar correct-answer margin, narrower
than the supplied full-logit Jacobian. Compact attention replaces 31.5–48.9% of
calibration final-state squared deviation from its mean anchor. These facts do
not establish that the omitted finite changes are harmless. The local loss term
dominates on 63.0%
of updates. These are diagnostic leads, not isolated causal demonstrations.

Cross-task basis reuse is 104,768 uses, 51,325 unique
layer-specific elements and 53,443 repeats
(51.01%). 13,657 qualify on all four
tasks. These eligibility counts are separate from the fixed removal of 1,024
nonlinear channels per layer. The Gröbner certificate concerns the fitted
polynomial surrogate; neural relation defects remain empirical.

There are 512 training, 64 development and 256 evaluation questions. Each
trained variant uses 96 steps per layer. Both versions, component diagnostics,
projection diagnostics and unsuccessful reservation-readiness attempt remain
available in dated append-only roots. All GPU stages used inference-ctl leases
and root locked Pixi. The controller now reports no bookings or leases and a
normal state; no regular service was manually changed during closeout.

Validation: 17 CPU tests passed; real GPU backward, parameter reload and
CUDA-graph checks passed. All 580 corrected checkpoint tensors are unchanged by
tied-storage export, and a fresh GPU load reproduces all 256 logits and
predictions exactly. The export is 5,596,836,325 bytes. The registry was read in a
fresh process with matching accuracy and scope. That reader supplies a
process-local `np.NaN` compatibility alias for the locked legacy Chroma consumer;
it changes no installed dependency or scientific result.

The next useful branch is the measured MLP-only checkpoint: optimize its packed
execution and adapt its coefficients before making further attention cuts. For
attention, test richer output-Jacobian targets and preserve the compact group's
dynamic residual while measuring layer-specific tolerances. Treat a true reduced
head-width kernel as a separate systems experiment; the corrected fused call
currently pads its 96 informative head coordinates back to 128 lanes.

Report: `docs/08 - Analysis/jspace/2026-11-13_jspace-full-compact.md`.
Latest evidence: `data/experiments/jspace_full_compact/20261113_affine_v2`.
Scientific fingerprint: `014d7221345715966c6b93df3e8bd0241a7b0c7791d34e412a6ac60d47d20b29`.
Registry: `data/hypothesis_registry/jspace_full_compact_20261113.json`.
Checkpoint: `data/experiments/jspace_full_compact/20261113_affine_export_v2/compact_model_tied.pt`.
MLP checkpoint: `data/experiments/jspace_full_compact/20261113_refinement_v1/mlp_runtime/model.pt`.
