# Dense latent labels for the complete NCM training population

Date: 2026-10-23. Hypothesis `ncm-dense-full-TRAIN-targets-v1`.
Classification **dense_full_TRAIN_targets_fit_passed**. Campaign `data/experiments/ncm_paper_regime/20261023_dense_training_targets_results_v1/campaign_results.json`;
SHA256 `618ce743575e8ac962ac57150546d7dc905201a62ca35bd0305027fa56edc575`.

| TRAIN population | Target token reconstruction | Exact target texts |
| --- | --- | --- |
| original_252 | 21902/21905 = 99.9863% | 249/252 |
| reviewed_438 | 41058/41060 = 99.9951% | 436/438 |
| all_690 | 62960/62965 = 99.9921% | 685/690 |

All690 approved training summaries remain in the denominator. The32 registered
256-step latent and optimizer states retain their complete traces and raw
outputs exactly. The other658 labels receive256 fixed FP32 latent-only AdamW
updates through the unchanged original decoder, with math SDPA and TF32 off.
The two-update lifecycle pilot is separate; its exact baseline and update
prefix are retained by full optimization. Every target, EOS/PAD sequence,
latent state, optimizer, raw readout and subgroup count replayed independently.

The199/200 token fidelity gate applies to both training strata and their union.
This is target-aware label preparation. It establishes no source-only student
accuracy or speed gain. Original Llama weights, codec and decoder stay frozen;
no source features or parent model are loaded. Both sealed final cohorts remain
unopened. Student training needs a separate prospective configuration and
subsequent source-disjoint semantic evaluation against the unchanged95% goal.
