# Qwen2.5-3B eleven-block recovery and boundary stop

**Eleven of 36 blocks are compressed and freshly verified at 46/64 (71.875%) on the reused development set.** This passes the prospective original-native five-percentage-point budget: native BF16 is 48/64 (75%), the minimum is 45/64, and the measured loss is 3.125 points. The campaign then stopped during the next boundary control, before any twelve-block fitting or training. That stop is an execution failure, not a completed accuracy failure.

This is the new `JSPACE-QWEN3B-NATIVE-BUDGET-01` continuation. It admits the previous ten-block 47/64 checkpoint under the original-native budget; the earlier campaign’s immediate-parent quality stop remains unchanged. Immediate-parent accuracy is diagnostic here. Architecture, calibration and recovery settings remain fixed: BF16 rank1024/head64 suffix, polynomial MLP compression, 512 training cases, 12 epochs/384 updates, output loss 0.7 × T² KL + 0.3 CE at T=2, fixed geometry/readout and trainable suffix/final norm.

For cut25, the initializer scores 45/64. Accuracy-first selection chooses epoch10 at 46/64; its entire packed prediction/choice-logit record is reproduced after fresh reload with maximum absolute difference 0. Against initialization, selection gains eight answers and loses seven (16 decisions change). Against the ten-block parent47, it gains five and loses six (12 decisions change). Against native48, it gains five and loses seven (13 decisions change). Aggregate recovery therefore redistributes errors.

| Reference | Correct /64 | Selected minus reference | Gains / losses | Changed decisions |
|---|---:|---:|---:|---:|
| Initial eleven-block model | 45 | +1 | 8 / 7 | 16 |
| Ten-block parent | 47 | −1 | 5 / 6 | 12 |
| Original BF16 model | 48 | −2 | 5 / 7 | 13 |

| Epoch | Correct /64 | Mean online train KL | Packed development KL | Packed development CE |
|---:|---:|---:|---:|---:|
| 0 | 45 | — | 0.007807 | 1.610056 |
| 1 | 29 | 0.156684 | 0.143368 | 1.767699 |
| 2 | 32 | 0.106743 | 0.104897 | 1.806539 |
| 3 | 33 | 0.127260 | 0.142643 | 1.919594 |
| 4 | 40 | 0.687878 | 0.650337 | 2.852265 |
| 5 | 39 | 0.614585 | 0.441788 | 2.410947 |
| 6 | 41 | 0.801840 | 0.531537 | 2.588948 |
| 7 | 44 | 0.762079 | 0.491999 | 2.547696 |
| 8 | 42 | 0.705700 | 0.455257 | 2.447609 |
| 9 | 42 | 0.654681 | 0.442936 | 2.437428 |
| 10 | 46 | 0.625081 | 0.414640 | 2.362174 |
| 11 | 44 | 0.600129 | 0.403588 | 2.367751 |
| 12 | 44 | 0.591562 | 0.402529 | 2.363249 |

KL is the unweighted, temperature-2 full-vocabulary divergence reported by the producer. Despite the one-answer gain, selected development KL rises from 0.007807 to 0.414640 and CE from 1.610056 to 2.362174. The online training objective also deteriorates: mean loss is 0.561970 in epoch1, jumps to 2.571223 in epoch4, and finishes at 2.169163. This is not evidence of steadily improving distribution matching. Training uses the differentiable unpacked path on training cases; development uses packed inference on other cases. These logs alone cannot separate optimization instability, generalization, and execution-path effects. A same-case, same-weight execution comparison is the appropriate diagnostic; none is claimed by this report.

The next control starts from cached native boundary24, executes the original native block24, then the accepted compact25..35 parent. Directly feeding cached boundary25 into that same parent reproduces 46/64 and all logits exactly. Recomputing the boundary gives 45/64, fails exact predictions, and reaches a maximum choice-logit difference of 0.625, exceeding the unchanged tolerance `0.03125 + abs(reference) / 128`. The changed answer is WinoGrande identity `3c73e8d74ab807dcdd07e0be4d2797f64c7a12764414e61ad7d8b861e763a48e`: cached-parent logits `[34.75, 35.0]` select the correct second choice; seam logits `[34.75, 34.75]` tie and select the first. This establishes a boundary replay discrepancy; it does not establish its cause or the quality of an untrained twelve-block model.

The verified eleven-block model has **5,457,808,560 persistent tensor bytes versus 6,171,877,632 native bytes: 11.5697% reduction**. The total includes 981,628,848 bytes for packed suffix/final norm, 3,853,849,600 frozen-prefix bytes, 622,329,856 tied embedding/readout bytes counted once, and 256 rotary bytes. Geometry is included; activations, KV caches, optimizer state and disk checkpoints are excluded. This is not a throughput result.

Authentication used `jspace_qwen3b_native_budget.verify_root`, `verify_stage`, and `checked` on all five completed cut25 phases in locked default Pixi with CUDA hidden. All 74 live/archive source digests, 446 pinned input identities, phase artifact digests, epoch selection, update participation, strict reload and quality fields passed. The failed cut24 capture is preserved separately; it has no completed capture manifest.

| Completed cut25 phase | Lease | Seconds | Peak reserved bytes | Result SHA256 |
|---|---|---:|---:|---|
| capture | `12cc4e0e9340` | 354.54 | 3,376,414,720 | `415f65e31cdaf54e11396d46d49f3c6cb2d8712c07087d8d7978f57bc2dc1e80` |
| fit | `994e639671b3` | 87.03 | 889,192,448 | `4b3bb0e894e41ca78c7f291bbf26e653fe0d9139510d6a19b32999744e1ee368` |
| pilot | `818e66bb5040` | 42.41 | 3,923,771,392 | `8686edf254d16c6ae4233338dea289cc2dc7e57fa35d29faa4ae64f2a22a5922` |
| train | `433b340c0f94` | 1209.71 | 5,704,253,440 | `f2b7cbdcea2213fbd2cd989687c52d1c99c02909a191504a9bd5686a31467936` |
| verify | `006ce2519a34` | 30.25 | 3,149,922,304 | `886e656c8dfede1cdfb2a91c26471fc253cb6e30822b7d23bea150d70d3659c2` |

Campaign fingerprint: `f32d35b1727fe84b9691b781ac561ce9a70cd809ae3fc2a940b7dd5184f65a63`. Cut25 fingerprint: `263336bb1812619900a0b0b91452d4a5bad4949ca15e0542c7170f4ef44de6e0`.
Selected checkpoint: `stages/block_25/train/selected.pt`, SHA256 `faf7dbd54141a9faea0c8c4cb3aa49dc8ceab7e0d035507ee5d78cc42f71f739`.
Plan SHA256: `6b729e7ddcb0d6c655575d8e258dda6f33a6d4332adec8cc8e89eb010932fb10`; cut25 plan SHA256: `a531ec08d3715a70f9f24283a945afd62e86eaf16d360d77bb5bd21ef2a78c0d`.
Train manifest SHA256: `182f5cfb8ef36cb5eb0b6442390749d3316528dab46ca960586cfb6062862c48`; fresh-verification manifest SHA256: `dd5430140f73a31ca0980fb7c58351ad9d8765012ffd81bb086a5cbc8ade6d59`.
Cut24 capture failure SHA256: `a141e50582006293eddff860b6beaae4ddadda3ada540d210082564974787f8d`; failed lease `bc9ffed23a63`.
Completion watcher status is `campaign_execution_failure`, finished 2026-11-15T16:48:21.574702+00:00; status SHA256 `234085db0d22d0e2ec4a4cf640bb5ecfba8098742fd1fc3c104db84f48014c8e`. The reservation controller returned `no bookings or leases` during this audit.

Evidence root: `data/experiments/jspace_qwen3b_native_budget/20261115_v1`. The selected epoch, all other epoch records, raw controls, completed manifests and failure receipt remain untouched. No serializer or registry from the older dual-gate study was reused.

All accuracy figures use the same 64 development cases repeatedly used for selection. No untouched confirmation was opened, no speed benchmark ran, and full36 compression is unfinished. Confirmed: false.
