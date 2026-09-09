# Qwen MoBA: exploratory shared-GPU assessment

The primary [feasibility protocol](2026-09-09_qwen8b-moba-feasibility.md) remains
unchanged. Its uncontended timing and joint confirmation are deferred because
an unrelated workload occupies the only available Ampere GPU. This auxiliary
assessment answers what is measurable under the available shared deployment.
It cannot establish uncontended speed or application accuracy.

Known before this assessment: the readout-only grid completed 96 cells through
16K, with 24 baseline capacity failures at 32K. All-layer block256/top4 passed
the readout gates on both 8K/16K corpus windows. A separately preserved synthetic
kernel diagnostic on the shared GPU measured 16K attention speedups of 1.13,
1.35 and 1.58 for block256/top4, block512/top3 and block1024/top2, respectively.
At 8K all those settings were slower. The upstream example uses block512/top3.
These observations choose the following exploratory shortlist; this is not a
claim that it was preregistered before all outcome inspection.

Test those three settings on all six attention layers and on layer six only,
at 8192 and 16384 tokens on both existing validation windows: 24 cells. Run the
unchanged Qwen prefix with 2048-token MLP chunks, four query heads per official
MoBA call, FP16 weights, FP32 pooling, all weights frozen, and a 45% allocator
limit. Both full and MoBA paths retain the same embedding, first five MLPs,
projections and readouts. The sixth MLP is omitted in both.

Save Q/K/V readout preservation and alternate five paired CUDA-event timing
samples after one warmup per path. Record GPU state before and after each cell.
Do not extrapolate these latencies to an idle GPU. Shortlist configurations
passing the original numerical readout gates and observed prefix speedup >=1.05
on both sources. Select the shortest passing length, then greatest mean speed,
then correlation, then listed order. This is an auxiliary shared-GPU selection.

If one is selected, freeze its configuration and length and evaluate the ten
previously sealed final windows (five seeds on each corpus). Record a distinct
`shared_gpu_prefix_feasibility_supported` only if all ten joint gates pass;
otherwise `shared_gpu_prefix_feasibility_not_confirmed`. No selection means
`no_shared_gpu_candidate_passed`. All are exploratory, with the main idle-GPU
claim unconfirmed. If no setting provides a speed benefit in this execution
configuration, do not train adapters to try to fix a runtime bottleneck.

Memory failures remain capacity outcomes. Preserve source snapshots, checkpoint
identity, input schedules, all cells, full gate counts and strict JSON. Use new
`20260909_shared_screen_v1` and, if eligible, `20260909_shared_final_v1` roots.
The attention-only kernel benchmark and complete-prefix runtime are separate
measurements and must be reported separately.
