# The code-repair attention module trains on native NCM codes

The new 3,778,049-parameter repair module completed two CUDA optimizer updates
on the two prospectively selected TRAIN examples. Gradients reached source
attention, the frozen native codec remained unchanged, and strict reload
reproduced model, optimizer, RNG and complete hard-code outputs. This is a
systems check; summary quality remains unmeasured.

Hypothesis: `ncm-source-attention-code-repair-lifecycle-v1`. The parent Llama
and existing NCM generator were absent. Inputs were their already archived
source-only code predictions from checkpoint `3911eab7...`, complete original
FP32 source features and frozen native codebooks. Target codes entered the
loss only. Training touched only the separate attention repair module.

The model reads all 641 predicted codes and all source positions using two
width-256, four-head attention blocks. Its output queries score the existing
frozen codebook geometry. Initialization preserved every input code exactly.
The loss combined equal-weight per-level code cross entropy with 0.05 times
full reconstructed soft-latent MSE. Both examples were from the existing
16-case TRAIN diagnostics; no held-out or sealed final data entered training.

| Check | Measured result |
| --- | --- |
| CUDA updates | 2, each accumulating both TRAIN cases |
| Trainable parameters | 3,778,049 |
| Peak allocated memory | 1,008,982,528 bytes |
| Source KV gradient L1 after clipping, update 2 | 5.73209 |
| Full frozen codec hash changed | No |
| Strict model/optimizer/RNG/hard-output reload | Passed |
| Hard code changes after two updates | 0 on both examples |

The zero initial residual head makes source-attention gradients zero at the
first update; they become nonzero at the second update. The loss decreased
slightly, but **no hard code changed in this short check**. There is no measured
summary improvement, semantic accuracy or speedup. These results establish
the mechanism's execution path only and cannot satisfy the 95% goal.

Eight focused CPU tests passed, including identity initialization, real source
influence, masked-source isolation, gradients into attention, frozen input and
codebook protection, and existing native soft-latent gradient tests. The actual
CUDA worker exited 0. The separate CPU audit verifies the strict saved state,
changed trainable weights, optimizer steps, gradient trace, selected TRAIN
identities, producer hashes and complete outputs. It does not replay CUDA
attention numerics or assign semantic judgments.

Producer commit: `5d046f0`. FP32, TF32 disabled, physical GPU2. Checkpoint SHA:
`34b10d7fedbda36a16c0f5b2c816f8a3bb951a679a29bfe3859ea3ad27ce9d27`.
Evidence is under `data/experiments/ncm_paper_regime/` in
`20261025_attention_code_repair_lifecycle_v1/`,
`20261025_attention_code_repair_lifecycle_audit_v1/` and
`20261025_attention_code_repair_lifecycle_results_v1/`.

Next steps are a fixed-budget TRAIN learnability comparison, then larger
paired-data training and source-disjoint semantic evaluation against unchanged
generator and source-ablated controls. Dataset size alone does not certify
reference quality or task fit. The user's DVC suggestion led to the likely
matching restored shared corpus: 204,967 original TRAIN atomic pairs, with
45,136 fitting the older 512-input/48-output regime. An inventory under the
current limits is a separate data-only study. Its identity as the user's
intended DVC source has not been independently confirmed. All
previous semantic failures, invalid teacher firsts and transport incidents
remain preserved. The final 320-case test stays unopened and unhashed.
