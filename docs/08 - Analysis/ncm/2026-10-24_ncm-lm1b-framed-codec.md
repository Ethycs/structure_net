# Released LM1B codec with corrected CLS framing

Date: 2026-10-24. Hypothesis `ncm-lm1b-framed-codec-compatibility-v1`.
Classification **native_CLS_codec_numeric_control_completed_semantics_unmeasured**. Campaign `data/experiments/ncm_paper_regime/20261024_lm1b_framed_codec_results_v1/campaign_results.json`;
SHA256 `86ce6078e7221980f755d2b41834d30e14d603b2b4ce91ddc9bfedd59ee8c29e`. Conformance: EXPLORATORY target-aware control.

| Representation measurement | Result |
| --- | ---: |
| Complete TRAIN targets | 690/690 |
| Native token agreement | 63872/65063 = 98.1695% |
| Exact native token sequences | 271/690 |
| All window boundaries correct | 684/690 |
| Empty reconstructed texts | 0/690 |
| Above128 storage tokens | 21/690 |
| Fit one native output frame | 665/690 |
| Overlength targets retained with complete windows | 25/690 |

These are correct-target-code reconstructions, not source-only summary accuracy.
No source-conditioned generator was trained. BERT lowercases and changes spacing;
token agreement uses its native representation, not exact cased summary matches.
The128-position frame includes native boundaries. Multi-window oracle coverage
does not show that one generated frame can express an overlength reference.

Every frame has an explicit CLS boundary at each end. The parser consumes a predicted initial CLS and stops at the next predicted CLS without access to correct target length. Full raw IDs and every overlength window remain available. The separate four-frame native-data control is descriptive, not a full benchmark replication.

All four worker exits were observed successfully. The full fresh codec load
reproduced both retained pilot outputs; all raw counts, tokenization and texts
replayed independently. Released weights remain unchanged, FP32 with TF32 off,
on physical GPU2. Peak allocation 759482368 bytes.
Original Llama and its MLPs were absent and remain frozen. No final source was
opened. The95% source-disjoint task goal and inference speed remain unproven.
A source-grounded semantic audit is required before conditional retraining.
