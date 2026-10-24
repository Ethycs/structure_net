# Released LM1B codec boundary-format failure

Date: 2026-10-24. Hypothesis `ncm-lm1b-native-codec-compatibility-v1`.
Classification **native_codec_boundary_format_mismatch_preserved**. Campaign `data/experiments/ncm_paper_regime/20261024_lm1b_codec_results_v1/campaign_results.json`;
SHA256 `5da0429427e2231187b5af2f8bacc10b4c547483c96a5923d36d7c8195b0443f`. Conformance: EXPLORATORY target-aware control.

| Representation measurement | Result |
| --- | ---: |
| Complete TRAIN targets | 690/690 |
| Native token agreement | 62404/64323 = 97.0166% |
| Exact native token sequences | 2/690 |
| First global EOS correct | 4/690 |
| Empty reconstructed texts | 686/690 |
| Above128 storage tokens | 0/690 |
| Fit one native output frame | 670/690 |
| Overlength targets retained with complete windows | 20/690 |

These are correct-target-code reconstructions, not source-only summary accuracy.
No source-conditioned generator was trained. BERT lowercases and changes spacing;
token agreement uses its native representation, not exact cased summary matches.
The128-position frame includes native boundaries. Multi-window oracle coverage
does not show that one generated frame can express an overlength reference.

The initial control omitted a leading CLS and stopped at the first predicted CLS, which was usually the native BOS. Its empty results are preserved; they do not establish failure of a correctly framed model.

All four worker exits were observed successfully. The full fresh codec load
reproduced both retained pilot outputs; all raw counts, tokenization and texts
replayed independently. Released weights remain unchanged, FP32 with TF32 off,
on physical GPU2. Peak allocation 759482368 bytes.
Original Llama and its MLPs were absent and remain frozen. No final source was
opened. The95% source-disjoint task goal and inference speed remain unproven.
A source-grounded semantic audit is required before conditional retraining.
