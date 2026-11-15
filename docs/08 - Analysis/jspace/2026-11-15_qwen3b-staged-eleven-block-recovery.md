# Qwen2.5-3B: eleven compressed blocks with parent choices preserved

Status: **11 of 36 blocks accepted and freshly verified; 25 remain**. Study `JSPACE-3B-STAGED-CAMPAIGN-01`, exploratory reused development, `confirmed=false`. Root `data/experiments/jspace_3b_staged_campaign/20261115_v1`; fingerprint `6b40611b583ca3889f4a8fb08370cf967ee95a6161cc13a99d8d70dee29765bb`. The reserved campaign has advanced to cut 24, the twelfth block.

The eleven-block initializer scored **51/64**. Dense entrance feature fitting restored **52/64**, reproducing every one of the accepted ten-block parent's 64 task choices. Its logits are approximate: KL to that parent is 0.002099, not zero. All 12 joint epochs and 384 updates completed, but none improved the selected checkpoint. A separate process reloaded and packed the feature-fit checkpoint and reproduced its own reference metrics and logits exactly.

| Candidate | Correct / 64 | Full-vocabulary CE | KL to original native |
|---|---:|---:|---:|
| Raw eleven-block initializer | 51 | 1.558235 | 0.565199 |
| Feature fit, selected epoch 0 | 52 | 1.535622 | 0.560210 |
| Final joint epoch 12, not selected | 48 | 1.546128 | 0.570911 |

The original native BF16 control is 48/64. Selected family counts are ARC 14/16, BoolQ 14/16, HellaSwag 13/16 and WinoGrande 11/16. Feature fitting gains two answers and loses one relative to raw, restoring the parent's exact choice vector. The earlier eleven-block branch selected 46/64; it used a different, lower-quality parent and is historical context rather than an isolated ablation.

## What the two completed transitions establish

Cut 26, adding the tenth block, improved from raw 47 to feature-fit 51 and selected joint epoch 6 at 52. Cut 25, adding the eleventh, improved from raw 51 to feature-fit 52 and selected that checkpoint unchanged. Both transitions pass fresh packed verification with zero reference-logit difference.

Dense feature fitting has therefore recovered task accuracy in two consecutive transitions while keeping the compressed shapes and fixed geometry. Joint output recovery helped the first transition and did not help the second. Keeping the feature-fit candidate eligible prevents later optimization from replacing it with a worse checkpoint.

The eleven-block feature error falls from 1.166628 to 0.672389, and answer-position error from 0.522406 to 0.124067. Only the new entrance parameters change in that phase; inherited parent parameters, norm, readout and geometry remain fixed. No matching feature-error measurement was taken after each joint epoch, so the curves do not prove that joint training destroyed the alignment.

The current joint run also differs from the old sustained instability. Through epoch 6, mean training loss falls from 0.04395 to 0.03168, and clipping falls from 11/32 updates to 0/32. The old branch clipped all 384 updates and reached a gradient norm above 3,469. Current development regression is modest despite better training fit; generalization or objective tradeoffs are plausible, but not isolated causes.

The authenticated curves distinguish each evaluated candidate from the checkpoint actually retained:

- [Tenth-block curve](../../../data/experiments/jspace_3b_staged_campaign/20261115_v1/analysis/through_cut_25/cut_26_development_curve.png)
- [Eleventh-block curve](../../../data/experiments/jspace_3b_staged_campaign/20261115_v1/analysis/through_cut_25/cut_25_development_curve.png)
- [Machine-readable summary](../../../data/experiments/jspace_3b_staged_campaign/20261115_v1/analysis/through_cut_25/summary.json)

Both PNG and SVG versions are preserved with input hashes, plotting-source archive and manifest. The default locked Pixi environment generated them; the first attempt used the training environment, which lacks Matplotlib, and failed before creating output. That CPU preparation failure is preserved separately and did not affect training.

## Storage, execution and continuation

Measured whole-model tensor inventory is **5,457,808,560 bytes versus 6,171,877,632 native bytes: 11.570% smaller**. This includes the remaining native prefix, every fixed geometry tensor, and the tied embedding/readout once. It does not count optimizer state or KV cache as model size.

All six phases used independent RTX 3060 reservations. The joint phase took 1236.54 seconds including phase bookkeeping, with 5,704,253,440 bytes of peak reserved tensor memory (5.313 GiB). These are experiment resource measurements, not inference throughput.

`jspace-3b-staged-campaign.service` continues from cut 24 through cut 0. `jspace-3b-staged-completion.service` remains armed for the final standalone export, cache checks, untouched 512-case accuracy, actual full-model storage and end-to-end generation benchmark. The new raw-text CLI has passed tiny-model CPU tests; its full-size smoke test still requires the accepted export. The calibrated deployment capacity remains 416 positions.

No all-36-block success, independent accuracy confirmation, or speed gain is claimed. The full objective remains unchanged.

Selected checkpoint SHA256: `7fb9316889ce2103e5a4d7ea6c39a4af64414f855fa4ef4ed2cc8f0049ca21b4`.
Fresh verification result SHA256: `c6fd8c5f1f24da1db0b3b0bc8d61c5177eab4f4361b9071fb423280c18b99e84`.
Fresh verification manifest SHA256: `b1880d885dfdd5cc525c3b5a3e2706f96e04a561b0c27dee2b6d56eb1eea4883`.
Curve summary SHA256: `d28dfce82c52c8b0b1340e16a8c665b0c4fca5bfe1c2e2429ae4e66170d2e147`.
Curve manifest SHA256: `836030ffcedbda42871674511c1728bb140979fb47e8bd3db2e48085b4c0728d`.
