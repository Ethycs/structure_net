# Qwen2.5-3B: twelve compressed blocks accepted at 50/64

Status: **12 of 36 blocks accepted and freshly verified; 24 remain**. Cut 24 selects joint epoch 7 at **50/64 (78.125%)**, versus the eleven-block parent’s 52/64 and original native BF16 control’s 48/64. It passes the fixed native quality floor of 45/64. This is exploratory selection on reused development data, `confirmed=false`.

Study `JSPACE-3B-STAGED-CAMPAIGN-01`; root `data/experiments/jspace_3b_staged_campaign/20261115_v1`; campaign fingerprint `6b40611b583ca3889f4a8fb08370cf967ee95a6161cc13a99d8d70dee29765bb`. Cut-24 fingerprint: `da687fb4b4815edde929ceba00bc0ea7cb7db545d75791cb522e46bf2fe709fc`.

| Candidate | Correct / 64 | Full-vocabulary CE | KL to original native |
|---|---:|---:|---:|
| Raw twelve-block initializer, epoch −1 | 49 | 1.593618 | 0.606601 |
| Feature-aligned initializer, epoch 0 | 50 | 1.560214 | 0.567673 |
| Selected joint epoch 7 | 50 | 1.557682 | 0.588862 |
| Final joint epoch 12, not selected | 47 | 1.561000 | 0.582989 |

All 12 joint epochs and 384 updates completed. The candidate accuracy curve was `49 → 50 → 47,45,45,49,48,48,50,48,49,48,49,47`. Epoch 7 wins the accuracy tie with epoch 0 through lower CE; its native KL remains below the protected raw initializer’s, although it exceeds the feature-aligned candidate’s. Later weaker candidates do not replace the selected checkpoint.

The [curve](../../../data/experiments/jspace_3b_staged_campaign/20261115_v1/analysis/cut_24/cut_24_development_curve.png) and [authenticated summary](../../../data/experiments/jspace_3b_staged_campaign/20261115_v1/analysis/cut_24/summary.json) distinguish candidate accuracy from the retained checkpoint.

Fresh-process loading and packing reproduced all 64 selected predictions, logits and metrics exactly: maximum reference-logit difference **0**. Fixed-state checks passed. The native cut-24 boundary control also passed exactly: an unpadded singleton native forward followed by parent batching reproduced the cached boundary and the parent’s 52/64 logits with zero difference. This resolves the previously diagnosed batching discrepancy without widening tolerances.

| Family | Parent | Feature-aligned | Selected |
|---|---:|---:|---:|
| ARC | 14 | 12 | 14 |
| BoolQ | 14 | 15 | 14 |
| HellaSwag | 13 | 13 | 13 |
| WinoGrande | 11 | 10 | 9 |

Against the parent, selection gains one answer and loses three across five changed predictions; one changed answer remains wrong. Against feature alignment, it gains three and loses three across seven changes. Both ARC regressions recover; the net parent deficit lies in WinoGrande. All three parent-to-selected regressions started with a +0.25 gold-answer margin, but one HellaSwag margin ends at −5.375, a substantial preference reversal.

The Sophia BoolQ item’s feature-aligned “Yes” was a label-scored gain. The selected checkpoint instead answers **No**, matching its passage and the parent. The pinned source itself labels this item True despite saying Sophia remains alive; all 144 used BoolQ labels map correctly from source. No labels were changed.

Whole-model tensor inventory is **5,392,166,336 versus 6,171,877,632 native bytes: 12.6333% smaller**, including fixed geometry, the remaining native prefix and tied embedding/readout once. KV cache, activations and optimizer state are excluded. Next is cut 23. Untouched held-out evaluation and full-model throughput remain pending; neither all-36-block success nor a speed gain is established.

Evidence under `stages/block_24/`:

- Selected checkpoint SHA256: `db53db32597ebb4947e38f7324e31e6eb9c7686f491a9e8b9745b0c43bca8292`.
- `train/result.json`: `de984a6e683c7956dbfb0d1875b19edb0b60ad728bdbba5d3b80dfbdad2aa613`.
- `verify/result.json`: `bd9b2538a38f8a511e7d80577fad9aaaa4a907cff4e1d61dfe96ac2281b78334`.
- `verify/manifest.json`: `2169b07eca79139222f4c6b1b8a1322bba29e1e5f570f2802c13b0d6e213cf56`.
- Curve summary SHA256: `017e62948833cbce5afa2937d662ddc3c71735359c087a9ccff3f5ea694ad9ac`.
