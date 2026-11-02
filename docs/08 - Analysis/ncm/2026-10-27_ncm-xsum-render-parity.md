# XSum rendering mismatch: PAD is not the end token

Classification: `author_render_mismatch_verified`. This is a rendering diagnostic; corrected semantic accuracy remains unmeasured.

The frozen four-arm semantic probe used the local decoder that stops at either EOS or PAD. The pinned author script stops at EOS and removes PAD tokens. The existing author-compatible local renderer matched the exact extracted author function on all512 saved word sequences. No generation or training was repeated.

| Arm | Changed texts out of128 |
|---|---:|
| native | 16 |
| initial | 18 |
| mse | 22 |
| distill | 17 |

The completed native arm had3/128 complete-summary passes under the EOS-or-PAD policy. All128 raw first judgments were independently replayed and valid. All3 passes have unchanged text. Even if every changed native text passes, retaining unchanged first judgments limits this correction alone to19/128 (14.84%). This upper bound is not a measured corrected score.

At observation, the other arms were still being graded. Their incomplete outcomes were not used here. The old experiment and its first judgments remain intact. Corrected texts and exact packets are saved for a separately governed follow-up that must inherit the completed cache.

The128 sources are the reused public XSum TRAIN probe, excluded from adapter fitting but possibly seen by the released teacher. This is not application final accuracy. No final320 source was opened; no speed was measured.

Raw evidence: `data/experiments/ncm_paper_regime/20261027_xsum_render_parity_v1/result.json`.
