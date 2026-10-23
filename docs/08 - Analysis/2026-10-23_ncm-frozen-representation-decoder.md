# Codec decoder transfer with the representation frozen

Date: 2026-10-23. Hypothesis `ncm-frozen-representation-decoder-transfer-v1`.
Classification **codec_reconstruction_fidelity_failed**. Aggregate `data/experiments/ncm_paper_regime/20261023_frozen_representation_decoder_results_v1/campaign_results.json`,
SHA256 `211d47945cf46e3bf9dd8cb73ac31d123321a4ffb6e36982f727b4175c364995`.

| Codec | Correct non-PAD tokens | Exact texts | Correct first EOS | Changed number sequences | Active invalid IDs |
| --- | --- | --- | --- | --- | --- |
| original | 20849/21905 (95.179%) | 23/252 | 251/252 | 53 | 0 |
| trained | 20836/21905 (95.120%) | 27/252 | 252/252 | 39 | 0 |

These measurements supply complete correct target-derived codes; they are
oracle reconstruction, not source-to-summary accuracy. All 252 final trained
code arrays equal their released-reference counterparts. The encoder, EMA
codebooks, normalization and resampling state stayed exact. Only the new codec
copy's 21,091,712 decoder parameters trained. The
parent Llama was not loaded. No conditional NCM generator was trained here.

Training used exactly 2,048 FP32 updates on GPU2, microbatch four and accumulation
eight. The 31,109 fit windows from 7,088 unpaired texts exclude every diagnostic
source group. The sampler lists and learning-rate schedules match the preceding
joint-transfer runs. Full correct-code latents were computed in eval/no-grad mode;
only decoder CE was optimized, retaining its .1 training dropout. This recipe
changes both gradient scope and the latent/dropout training distribution, so it
does not isolate every difference from earlier joint transfer.

The original reference reproduces all earlier outputs exactly. Raw IDs, codes,
complete texts and counts replayed for all 252 cases. Pilot/full save-reload
controls and frozen-state checks passed. Peak allocation was
731,230,720 bytes. Only the final checkpoint was selected.

The unchanged gate requires at least 99.5% token recovery, 99% first-EOS agreement,
zero active invalid IDs and no increase in changed-number cases. Exact-text
agreement and numeric-sequence changes are descriptive; a separate semantic
audit is required before a task-quality claim. The actual source-only native
teacher remains 56/64 (87.5%), below the goal. Both final task cohorts remain
closed; no speed or compression benefit was measured.
