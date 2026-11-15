# Targeted joint repair of the recovered Qwen assembly

Study **JSPACE-QWEN-LAYER-GROUP-REPAIR-01**. Classification: `partial_recovery_below_gate`. **Confirmed: false.**

After jointly training only compact blocks **[5, 10, 12]**, the selected model with **all 36 blocks compact** scored **31/64 (48.44%)**, versus 27/64 before repair and 50/64 native. The fixed 47/64 development gate failed. Epoch 6 was selected by actual task accuracy, then lower conditional-choice CE; the unchanged initializer remained eligible.

Always predicting **B** scores **30/64 (46.88%)** on these development cases: the gold labels contain A=20, B=30, C=8 and D=6. The selected result is only one correct answer above this descriptive constant-label baseline. Its predictions were A=17, B=25, C=10 and D=12. Agreement with the native teacher was **27/64 (42.19%)**, unchanged from the initializer, despite the four-answer net accuracy gain.

The preceding search found that replacing these three blocks with **native** blocks scored **32/64**, a conditional rescue of +5. That is a different model. Its accuracy is not the learned compact repair score and does not establish that the rescue is learnable. The triple was selected from the **20 triples within six candidates [5, 6, 8, 10, 12, 22]**, with all other blocks compact, using correct count, teacher agreement, lower choice CE and lexical indices. It is a conditional repair target, not a globally worst subset or an exhaustive search over all 36-layer triples.

| Epoch | Actual correct /64 | Full-vocabulary KL | Choice KL | Choice CE | Feature loss | Native agreement |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 27 | 2.892983 | 4.602220 | 9.357735 | 0.004224 | 42.19% |
| 1 | 27 | 1.198628 | 1.023234 | 1.243015 | 0.004477 | 42.19% |
| 2 | 29 | 1.386206 | 1.572472 | 2.645436 | 0.004490 | 35.94% |
| 3 | 26 | 1.385259 | 1.583542 | 2.761577 | 0.004600 | 34.38% |
| 4 | 30 | 1.472624 | 1.767708 | 3.018544 | 0.004605 | 39.06% |
| 5 | 28 | 1.550387 | 1.945174 | 3.480882 | 0.004630 | 37.50% |
| 6 | 31 | 1.368168 | 1.588748 | 2.631459 | 0.004699 | 42.19% |

| Selected layer | Initial feature loss | Selected feature loss |
|---|---:|---:|
| 5 | 0.002045 | 0.002232 |
| 10 | 0.004893 | 0.005266 |
| 12 | 0.005733 | 0.006597 |

Feature loss averages equal cases and the three selected layers: valid-token squared error in the fixed basis divided by the detached original-teacher coordinate energy. It retains gradients through the student output and earlier selected layers. At the selected epoch, this development loss increased from 0.004224 to 0.004699, with increases at all three layers. Endpoint KL and choice CE improved, but the measured feature alignment did not.

| Model | ARC /16 | BoolQ /16 | HellaSwag /16 | WinoGrande /16 |
|---|---:|---:|---:|---:|
| Native | 15 | 13 | 14 | 8 |
| Initial all compact | 5 | 7 | 7 | 8 |
| Selected learned repair | 8 | 7 | 6 | 10 |
| Search: triple native | 6 | 9 | 7 | 10 |

The selected family changes were **ARC +3, BoolQ unchanged, HellaSwag −1 and WinoGrande +2**. The selected learned model gained 5 and lost 1 correct answers relative to the 27/64 initializer. All 192 updates across 6 epochs completed on the original 512 training cases; each update reported gradients for exactly 21 tensors in the chosen three blocks. The other 33 blocks, native final norm, tied embedding/readout, bases and all buffers remained fixed. The one-update pilot was discarded.

The endpoint objective remained .2T² full-vocabulary KL + .5T² conditional-choice KL + .3 gold-choice CE at T=2, using midpoint and answer only. A unit-weight feature auxiliary was added. This longer schedule plus auxiliary is exploratory; benefit cannot be attributed solely to selecting this group rather than the earlier all-layer continuation.

The selected checkpoint stores a **three-block delta** (488,245,260 tensor bytes), requiring the 33 frozen blocks and original tied embedding/readout and norm. The complete reconstructed unpacked FP32 model contains 7,103,611,024 saved tensor bytes. The delta is not a standalone compressed model. No packing-parity or deployment-speed result is claimed by this closeout.

Original FP32 teacher targets passed native decision and selected-boundary parity checks. Strict reload and independent fresh-process verification reproduced the selected actual full-model predictions/logits and feature metrics. The same 64 development cases were reused for search and checkpoint selection; no untouched confirmation occurred.

Evidence: `data/experiments/jspace_layer_group_repair/20261114_v1/layer_group_repair_analysis.json`, SHA256 `372c390b63b4431a91893aadae214111ea12fa4cdb4b9a026787f401808d1f37`. Search selection, all stages and epoch rows, selected three-file delta, original FP32 logit cache and three new boundary target banks are authenticated. The other 33 frozen states retain inherited hashes and immutable identities. Subsequent closeout rechecks small hashes and identities of authenticated large inputs.
