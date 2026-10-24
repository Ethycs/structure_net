# Refining target latents through the frozen NCM decoder

Date: 2026-10-23. Hypothesis `ncm-decoder-latent-refinement-v1`.
Classification **oracle_refined_latent_fit_failed**. Campaign `data/experiments/ncm_paper_regime/20261023_decoder_latent_refinement_results_v1/campaign_results.json`;
SHA256 `e3984e88a4d8844bab6fc680209855e115c9182ec8060dbdd2afa49141a6ca32`.

| Same TRAIN targets | Continuous encoder | Refined encoder |
| --- | --- | --- |
| original_16 | 1592/1635 = 97.37% | 1616/1635 = 98.84% |
| additional_16 | 1541/1562 = 98.66% | 1554/1562 = 99.49% |
| all_32 | 3133/3197 = 98.00% | 3170/3197 = 99.16% |

Each arm receives complete target-derived information. The intervention optimizes
only per-example192-by384 latent tensors for32 fixed AdamW updates through the
same frozen original decoder. The objective is nonpadding word cross entropy
plus.001 times mean squared displacement from the continuous encoder latent.
All codec parameters remain unchanged; no source-to-summary student is trained.

These are TRAIN oracle token-reconstruction measurements, including EOS and
excluding PAD. Passing199/200 in each subgroup and the union establishes usable
teacher targets on these32 known cases. It does not establish that a student
can predict these targets from a source. Any larger label construction, latent
distillation or source-only training requires a separate prospective experiment.

The two-case lifecycle pilot used only two updates and supplies no quality
estimate. All raw words, final latents, optimizer states,32-step traces, source
bindings, exact counts and group gates replayed. Same-GPU baseline pilot outputs
and fresh decoder/latent/optimizer reloads were exact. The FP32 CUDA control used
math SDPA with TF32 off on physicalGPU2, separately from the fixed dropout study
on physicalGPU0. Shared-device wall time is not an inference benchmark.

The95% source-disjoint semantic summary goal remains unmet. Both final cohorts
remain unopened; no speed or compression benefit is established here.
