# Frozen neural decoder: quantized versus continuous target latents

Date:2026-10-23. Hypothesis ncm-continuous-codec-target-fidelity-v1.
Classification **continuous_codec_target_fidelity_failed**. Campaign data/experiments/ncm_paper_regime/20261023_continuous_codec_target_results_v1/campaign_results.json;
SHA256 14c1b4227eb63c10218dffbb957403454a4f1b1873e74667e1f959c6ffa6383e.

| Same TRAIN targets | Saved quantized latents | Continuous encoder latents |
| --- | --- | --- |
| original_16 | 1509/1635 = 92.29% | 1592/1635 = 97.37% |
| additional_16 | 1480/1562 = 94.75% | 1541/1562 = 98.66% |
| all_32 | 2989/3197 = 93.49% | 3133/3197 = 98.00% |

Both arms receive complete target-derived information. These are oracle TRAIN
token-reconstruction counts including EOS and excluding PAD, not summary task
accuracy or an information-theoretic bound. The continuous arm removes VQ but
keeps the original encoder normalization and SAME frozen neural decoder.
No true normalization statistics are used to invert tokens. A qualifying
199/200 fitting result in every group only supports considering continuous
latent supervision; no new source-only model, semantic quality or training
stage is admitted automatically.

The original codec stayed byte-identical. Computation was FP32 CPU, two threads,
with four independent48-position blocks and the same0..50256 vocabulary argmax
policy in both arms. This policy excludes PAD and unused output rows and differs
from earlier full-vocabulary codec diagnostics. Complete192-position outputs
and targets stayed intact. Fresh-load lifecycle parity and full pilot retention
passed, and all raw source/target bindings, decoding/counts and inclusive group
gates replayed. No CUDA, model training or judge request was used.

The full95% source-disjoint semantic summary goal remains unmet; both sealed
final cohorts remain unopened. No speed or compression claim is made.
