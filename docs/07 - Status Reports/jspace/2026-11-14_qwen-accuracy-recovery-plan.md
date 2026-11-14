# Qwen accuracy recovery after failed sequential compression

Study `JSPACE-QWEN-ACCURACY-RECOVERY-01`, exploratory, 2026-11-14.
The user requests an accuracy repair. The prior all-block quadratic recipe
scored 220/512 versus native396/512; its softmax control scored211/512.
Those completed checkpoints and predictions remain immutable. Its unfinished
CPU timing stage was interrupted explicitly to prioritize recovery; partial
timings are not usable benchmark evidence.

## Fixed objective and data boundary

Recover the useful task behavior of the original Qwen2.5-3B while retaining a
materially compressed generative model. The user permits at most five
percentage points of aggregate accuracy loss. A four-label classifier or
returning to the complete native model does not establish a compression repair.
The earlier 38% tensor reduction and1.4x speedup remain separate measured
targets, not assumptions about a corrected model.

Use the existing512 training/64 development examples to diagnose the failure
and select a candidate. The previously scored512-case evaluation pool is now
diagnostic-only. A fresh512-case confirmation set has been frozen, unscored, at
`data/experiments/jspace_accuracy_recovery_data/20261114_v1`, fingerprint
`827b3d4208ae95d731930c16f111e746b59b3d3166061038b0605e2fc64012aa`.
It excludes recursively collected identities from39 prior J-space task files.
No confirmation result may guide architecture, loss, training-budget, or
checkpoint selection. Additional training data, if needed, must exclude this
set and all prior evaluations and be frozen before teacher capture.

## Decision order

1. Run native-state/readout oracles and next-basis handoff diagnostics on
   development data. Separate representational loss, trained-head calibration,
   and accumulated student-prefix errors. Preserve individual predictions.
2. Test a residual-carry intervention with the native embedding/readout and
   native-width residual activations. Each compressed block computes a reduced
   update; the omitted input residual is carried unchanged. Include its actual
   energy in both RMS denominators. This does not restore native dense branch
   computation, and its remaining fixed-mean branch approximation is explicit.
   Compare native-initialized and previously fitted compact weights as needed.
3. Locate failure with compressed-prefix/native-suffix probes. Select a repair
   from the mechanism results, not from the exposed test set. Candidate training
   must supervise original-teacher behavior and measure decision accuracy during
   recovery. Any expansion or new training objective receives a frozen concrete
   configuration before its run. Do not repeat the all36-block fit unchanged.
4. Freeze one candidate, verify fresh-process serialization, and score it and
   native Qwen on the untouched confirmation set. The observed aggregate gate
   permits at most25 net lost correct answers out of512. Report per-family and
   paired counts, uncertainty, and whether all requested size/speed gates pass.
   A passing point estimate alone is not a population noninferiority proof.
5. Benchmark the confirmed candidate with matched native precision, shapes,
   output computation and runtime. Timing must be isolated from recovery jobs.
   Report generation speed separately from streamed scoring or training time.

## Execution and evidence

Every GPU invocation uses `inference-ctl lease run`; all execution uses locked
offline Pixi. New modules have `jspace_accuracy_*`, `jspace_recovery_*`,
`jspace_prefix_localize`, or `jspace_residual_carry` names, outside the frozen
recipe's producer glob. Changed interventions get new artifact directories,
with source/configuration and consumed-input hashes. Tiny causal/algebra,
gradient and reload tests precede CUDA probes. Failed and incomplete attempts
remain recorded. No numerical-null completion direction is described as proved
irrelevant, and no local reconstruction metric substitutes for task accuracy.

This is adaptive exploratory recovery. A failed fresh confirmation remains a
failure; any further tuning requires a newly held-out confirmation population.
Final records remain `confirmed=false` absent independent replication.
