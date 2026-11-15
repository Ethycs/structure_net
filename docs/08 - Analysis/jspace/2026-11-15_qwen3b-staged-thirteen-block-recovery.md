# Qwen2.5-3B: thirteen blocks verified; controlled trials before continuation

Status: **13 of 36 blocks compressed and freshly verified; 23 remain**. Cut 23 retains feature-aligned epoch 0 at **46/64 (71.875%)**, versus original native BF16 48/64 and accepted parent 50/64. It passes the original-native five-percentage-point budget, whose minimum is 45/64. The immediate-parent diagnostic fails its 47/64 threshold; that diagnostic is not an acceptance gate. These are reused development results, `confirmed=false`.

Study `JSPACE-3B-STAGED-CAMPAIGN-01`; root `data/experiments/jspace_3b_staged_campaign/20261115_v1`; cut 23 fingerprint `a81f36a25491608a57dba12c7c0254e7c0f67d258b1b93dfa14849732261a113`.

| Candidate | Correct / 64 | Full-vocabulary CE | KL to original native |
|---|---:|---:|---:|
| Raw initializer, epoch −1 | 46 | 1.572622 | 0.617583 |
| Feature-aligned epoch 0, selected | 46 | 1.556971 | 0.604104 |
| Joint epoch 5, ineligible | 49 | 1.587216 | 0.614561 |
| Final epoch 12, ineligible | 47 | 1.580498 | 0.614500 |

All 12 joint epochs and 384 updates completed. Every joint candidate failed the frozen CE eligibility condition. Epoch 5’s 49/64 was the best observed count, but CE exceeded raw by **0.928%**; its native KL passed. The protocol retained only eligible improvements: **epoch 5 weights were not saved; 49/64 is not a deployable checkpoint**. Epoch 0 remains selected through its lower CE than raw. The candidate curve was `46 → 46 → 44,47,46,46,49,46,48,48,48,47,47,47`.

The [authenticated curve](../../../data/experiments/jspace_3b_staged_campaign/20261115_v1/analysis/cut_23/cut_23_development_curve.png) and [summary](../../../data/experiments/jspace_3b_staged_campaign/20261115_v1/analysis/cut_23/summary.json) distinguish evaluated candidates from the retained checkpoint.

Fresh-process loading and packing reproduced all 64 selected predictions, logits and metrics exactly, with maximum logit difference 0. Fixed-state checks passed. Selected family counts are ARC 14/16, BoolQ 13/16, HellaSwag 10/16 and WinoGrande 9/16. Entrance feature improvement alone did not restore the parent’s task count.

Whole-model tensor inventory is **5,326,524,112 versus 6,171,877,632 native bytes: 13.6969% smaller**. Accounting includes the native prefix, packed suffix geometry and tied embedding/readout once; it excludes KV cache, activations and optimizer state.

Only CPU supervisor PID 3006274 received `SIGSTOP`; the independent verification worker completed normally. Receipt: `operations/hold_after_cut23_for_controlled_trials.json`. **Cut 22 has not launched and the goal remains active.** This is a controlled-trial hold, not a quality-gate failure or cancellation.

The joint objective has four-position output KL/CE but no final-hidden feature anchor. The prospective comparison distinguishes A: accepted-parent final-feature matching added to the objective, from B: original-native projected features and native KL. Neither is a proven fix or a campaign change. Reserved optimizer comparisons and feature diagnostics precede continuation. Untouched held-out confirmation and full-model throughput remain pending; no all-36-block success or speed gain is claimed.

Evidence under `stages/block_23/`:

- Selected checkpoint: `e12dad834ef48acf76fce441fbb919ab2749cd5c707396df75018646805e0e8e`.
- `train/result.json`: `f6c3b5bd26b996791aea676949d39669855bf6442fc0be537f12d7b08367fb0f`.
- `verify/result.json`: `9ef163bf729fc6c3721afdd42fa33a70c1a427b3879ef8646d2556d70a14f763`.
- `verify/manifest.json`: `329f8fae1c693116ca54506f402e9c6faf49bbbbbdb52622bf1a9b900309e0e5`.
