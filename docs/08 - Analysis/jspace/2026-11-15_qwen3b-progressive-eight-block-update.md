# Progressive Qwen2.5-3B suffix recovery

**8/36 blocks accepted; 52/64 reused development answers correct.** Classification: `progressive_in_progress`. Confirmed: false.

Qwen2.5-3B BF16 progressive suffix recovery from the verified seven-block seed. One new preceding block is calibrated against each accepted parent; the complete suffix and final norm are jointly trained for12epochs on512 cases and selected on64 reused development cases. Native prefix boundaries are cached. No untouched confirmation, speed or general-generation evidence.

The primary metric counts verified accepted compressed blocks. Accuracy, the immediate-parent and original-native five-percentage-point gates, and storage are reported separately. A resource or execution stop is not a completed quality failure.

| Cut | Blocks | Parent | Initial | Selected | Epoch | Minimum | Gate | Whole tensor bytes |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 28 | 8 | 51/64 | 50/64 | 52/64 | 7 | 48/64 | pass | 5,654,735,232 |

Cut 28: parent comparison gains 1, loses 0 (net +1); epoch correct counts: 50, 50, 49, 46, 50, 52, 51, 52, 51, 52, 51, 51, 51.
Recovery targets: `packed_inference`. The separate differentiable calibration teacher scores 50/64; the accepted packed parent scores 51/64.

Last accepted model: cut 28, 5,654,735,232 persistent tensor bytes versus 6,171,877,632 native bytes (8.38% reduction). Includes fixed geometry and tied E once; excludes activations, KV cache and optimizer.

All36 recovery is reported only after verified passing cut0. The initial seven-block seed is51/64; the matched original BF16 teacher is48/64. Parent-relative budgets are local, while every stage also retains the original-native budget. A passing stage does not establish untouched-test accuracy or speed.

The accepted teacher uses the unchanged two-group/eight-probe full-vocabulary Jacobian estimator. Each twelve-epoch stage recomputes the entire student suffix, keeps geometry/readout fixed and adapts all suffix parameters plus final norm. Epoch0 remains eligible; accuracy selects first, full-vocabulary CE breaks ties.

When packed inference targets are present, they are stored separately from the original differentiable teacher cache and use the parent's development batch grouping. The calibration Jacobians remain unchanged; a rounding-dependent difference between the two execution paths is retained as a diagnostic.

Executed parent/native/seam control rows are retained and independently compared to their authenticated references. All selected-epoch and fresh-verification rows are retained and independently checked.

Stop/progress: `{"cut": 27, "error": null, "kind": "in_progress", "phase": "capture"}`.

Evidence: `/data/structure_net_experiments/jspace_qwen3b_progressive/20261115_v3/qwen3b_progressive_snapshot_20261115T075220806686Z.json`.
