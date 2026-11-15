# Qwen2.5-3B ten-block expansion stop

The ten-block candidate scored **47/64 (73.44%)**, and its initializer at epoch 0 remained the selected checkpoint after all 12 training epochs. The campaign stopped at the prospective immediate-parent gate; the accepted nine-block checkpoint remains **52/64**.

The two accuracy budgets have different references. Relative to the original BF16 model's 48/64, the candidate loses **1.5625 percentage points** and passes the five-point budget (minimum 45/64). Relative to the immediate nine-block parent's 52/64, it loses **7.8125 points** and misses the stricter parent minimum of 49/64. The stop therefore does not mean the original-native budget was exceeded. These are the same 64 reused development cases.

**9/36 blocks accepted; 52/64 reused development answers correct.** Classification: `progressive_quality_stop`. Confirmed: false.

Qwen2.5-3B BF16 progressive suffix recovery from the verified seven-block seed. One new preceding block is calibrated against each accepted parent; the complete suffix and final norm are jointly trained for12epochs on512 cases and selected on64 reused development cases. Native prefix boundaries are cached. No untouched confirmation, speed or general-generation evidence.

The primary metric counts verified accepted compressed blocks. Accuracy, the immediate-parent and original-native five-percentage-point gates, and storage are reported separately. A resource or execution stop is not a completed quality failure.

| Cut | Blocks | Parent | Initial | Selected | Epoch | Minimum | Gate | Whole tensor bytes |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 28 | 8 | 51/64 | 50/64 | 52/64 | 7 | 48/64 | pass | 5,654,735,232 |
| 27 | 9 | 52/64 | 43/64 | 52/64 | 7 | 49/64 | pass | 5,589,093,008 |
| 26 | 10 | 52/64 | 47/64 | 47/64 | 0 | 49/64 | fail | 5,523,450,784 |

Cut 28: parent comparison gains 1, loses 0 (net +1); epoch correct counts: 50, 50, 49, 46, 50, 52, 51, 52, 51, 52, 51, 51, 51.
Recovery targets: `packed_inference`. The separate differentiable calibration teacher scores 50/64; the accepted packed parent scores 51/64.

Cut 27: parent comparison gains 2, loses 2 (net +0); epoch correct counts: 43, 48, 48, 47, 45, 47, 51, 52, 51, 50, 51, 50, 50.
Recovery targets: `packed_inference`. The separate differentiable calibration teacher scores 51/64; the accepted packed parent scores 52/64.

Cut 26: parent comparison gains 2, loses 7 (net -5); epoch correct counts: 47, 43, 40, 44, 45, 44, 47, 45, 45, 46, 45, 45, 45.
Recovery targets: `packed_inference`. The separate differentiable calibration teacher scores 52/64; the accepted packed parent scores 52/64.

Last accepted model: cut 27, 5,589,093,008 persistent tensor bytes versus 6,171,877,632 native bytes (9.44% reduction). Includes fixed geometry and tied E once; excludes activations, KV cache and optimizer.

All36 recovery is reported only after verified passing cut0. The initial seven-block seed is51/64; the matched original BF16 teacher is48/64. Parent-relative budgets are local, while every stage also retains the original-native budget. A passing stage does not establish untouched-test accuracy or speed.

The accepted teacher uses the unchanged two-group/eight-probe full-vocabulary Jacobian estimator. Each twelve-epoch stage recomputes the entire student suffix, keeps geometry/readout fixed and adapts all suffix parameters plus final norm. Epoch0 remains eligible; accuracy selects first, full-vocabulary CE breaks ties.

When packed inference targets are present, they are stored separately from the original differentiable teacher cache and use the parent's development batch grouping. The calibration Jacobians remain unchanged; a rounding-dependent difference between the two execution paths is retained as a diagnostic.

Executed parent/native/seam control rows are retained and independently compared to their authenticated references. All selected-epoch and fresh-verification rows are retained and independently checked.

Stop/progress: `{"cut": 26, "kind": "quality_stop"}`.

Evidence: `data/experiments/jspace_qwen3b_progressive/20261115_v3/qwen3b_progressive_analysis.json`.

The ten-block initializer changes nine parent decisions: two gains and seven losses. Against the original native model it changes 15 decisions: seven gains and eight losses. Fresh verification reproduces the selected initializer's complete prediction and choice-logit records exactly (maximum absolute logit difference 0).

Online training loss fell from 0.467069 in epoch 1 to 0.045727 in epoch 12. Packed development loss rose from 0.516337 at initialization to 0.606140, and final accuracy was 45/64. Epoch 6 tied the initializer at 47/64 but had worse CE (1.608078 versus 1.569281), so accuracy-first selection correctly retained epoch 0.

| Epoch | Mean online training loss | Packed development loss | Packed development CE | Correct /64 |
|---:|---:|---:|---:|---:|
| 0 | — | 0.516337 | 1.569281 | 47 |
| 1 | 0.467069 | 0.765157 | 1.729557 | 43 |
| 2 | 0.229594 | 0.673133 | 1.699602 | 40 |
| 3 | 0.199323 | 0.731633 | 1.748454 | 44 |
| 4 | 0.269421 | 0.649363 | 1.655312 | 45 |
| 5 | 0.192638 | 0.620744 | 1.670974 | 44 |
| 6 | 0.194783 | 0.610153 | 1.608078 | 47 |
| 7 | 0.131374 | 0.660598 | 1.711257 | 45 |
| 8 | 0.086382 | 0.627643 | 1.680274 | 45 |
| 9 | 0.076899 | 0.608207 | 1.654163 | 46 |
| 10 | 0.050556 | 0.604563 | 1.653301 | 45 |
| 11 | 0.046661 | 0.605941 | 1.656640 | 45 |
| 12 | 0.045727 | 0.606140 | 1.655186 | 45 |

The training objective improved without recovery on packed development. Training metrics come from the unpacked differentiable execution on training cases, while development uses the packed inference clone on different cases. This evidence alone cannot distinguish generalization failure from a training/inference execution gap. A same-case execution comparison was not run in this closeout.

All five phases completed successfully under reservations. The pilot changed FP32 optimizer masters in all ten blocks and the final norm, with fixed geometry/readout preserved. Scientific training completed all 384 updates; selected-state and fresh packed reload checks passed. Packed teacher targets matched the accepted parent exactly, so the earlier teacher-target tie mismatch did not recur here.

| Phase | Lease ID | Elapsed seconds | Peak reserved bytes | Result SHA256 |
|---|---|---:|---:|---|
| capture | `5f4d3be96a8e` | 357.89 | 2,042,626,048 | `66e97a60d8f3477c1b99ce46111fe45a41b83e79eb1a67faa92336d6e7da324a` |
| fit | `a956d35d02f7` | 80.25 | 889,192,448 | `b84fdf41c32e330bc12d7c03db05f492d79bd8040238d2ed626bb8186123ff00` |
| pilot | `40e5c1865f51` | 44.13 | 3,674,210,304 | `017327e793380d4c190872cb42af89c95e624919abca3032612abd6c3c9f8d14` |
| train | `2bd9bf57dfde` | 1064.40 | 5,003,804,672 | `b0442e7149e25ac5930d48b997f7f887d995b642b2b627bde7f8f8f688f1a729` |
| verify | `015a72b0b7ba` | 28.26 | 2,944,401,408 | `33a94b799f6276444138f8b443c7f7340b943cf1078d2c7256958d53d6b7ef5f` |

All phase manifests bind stage fingerprint `c026718d33def9303c603ea120ff7ec8ce344a331cde03c826b280dd135af252`. The selected ten-block initializer is `d6954494f3fd74a5765c3f623b39891cae383339d1888bbef067dd6879bb4cb5`.

The ten-block candidate contains 5,523,450,784 persistent model tensor bytes, a 10.51% reduction versus native. The retained nine-block model contains 5,589,093,008 bytes, a 9.44% reduction. Neither figure measures token throughput. No held-out scoring, full-36-layer completion or speed result was produced.

Terminal analysis SHA256: `b19b8e6981f6bd0dd05a1a23d303ebf80dd3835f6aeecb52ddfcc8c59a777a4e`.
