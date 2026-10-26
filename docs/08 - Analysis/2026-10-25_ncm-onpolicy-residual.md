# Learning from generated histories did not improve held-out summary accuracy

Both matched arms scored **0/128 complete-summary passes (0%)**, with all 256
new Terra judgments valid. Neither arm produced a readable, faithful or salient
output under the existing component rubric. The 95% task goal remains unmet.

Study `ncm-onpolicy-residual-v1`, completed 2026-10-26, is classified
**DEV_both_semantic_quality_failed**. This is an exploratory comparison with one
matched training seed. The independently replayed aggregate is
`data/experiments/ncm_paper_regime/20261025_onpolicy_residual_results_v2/campaign_results.json`,
SHA-256 `6c788f7eed48d9658adc5237d34ed34d6f472f750ae1d8a8e31a7bd85d48bdfa`.

| Held-out cohort | Original code targets | Residual correction targets | Valid per arm |
| --- | ---: | ---: | ---: |
| Short development | 0/64 | 0/64 | 64/64 |
| Long diagnostic | 0/64 | 0/64 | 64/64 |
| Combined | **0/128** | **0/128** | **128/128** |

The endpoint evaluates the whole summary against its complete source for
faithfulness, salience and readability, with nonempty output and the 128-stored-
token cap. It accepts paraphrases; exact reference matching is not the task
metric. No output was empty or above the cap. All 128 paired decisions were
failures in both arms; there were no unresolved pairs. The descriptive Wilson
95% interval is 0–2.91% per arm. This does not prove that the methods are
equivalent outside this experiment.

Terra (`codex_sdk/gpt-5.6-terra`) was the qualified secondary judge. Neither arm
reached the required 61/64 in both cohorts, so Sol was not admitted. No candidate
was selected, the sealed final320 remained closed, and no speed test ran. The
automated judges are not independent human annotation.

The [prospective plan](../07%20-%20Status%20Reports/2026-10-25_ncm-onpolicy-residual-plan.md)
compared two continuations from the same completed
[broader-generator checkpoint](2026-10-25_ncm-broad-generator.md). Both arms
trained on their own current source-only generated histories, mixed equally
with clean-history anchors. `fixed_targets` retained the original clean code
labels. `residual_targets` recomputed each level's labels from the reference
latent minus the contributions of the actual earlier generated codes. It did
not subtract newly computed oracle targets. Reference information supervised
TRAIN labels only and never entered held-out prediction.

Each arm used all 1,376 unchanged complete-summary TRAIN pairs, 1,024 optimizer
updates and 4,096 presentations. Exactly 2,048 presentations used generated
histories; each source appeared two or three times. Initialization, source
sampling, history-mode sampling, optimizer schedule and fixed-final checkpoint
selection matched. The histories could diverge as the two policies learned.
There was no random code replacement, discriminator or contrastive loss in
these continuations.

The generator and concept source adapters trained, totaling 105,323,136
parameters. The codec, word source path and original Llama body stayed frozen.
Full FP32 source features retained all positions. TRAIN/DEV checks found no
overlap in IDs, source IDs, groups, source hashes or source components. Every
held-out output used only source inputs and the selected complete checkpoint.

| TRAIN diagnostic, all 1,376 cases | BROAD parent | Original targets | Residual targets |
| --- | ---: | ---: | ---: |
| Matching active word IDs / 96,910 | 19,091 | 38,476 | 34,585 |
| Active word agreement | 19.70% | 39.70% | 35.69% |
| Matching rollout codes / 882,016 | 353,222 | 464,720 | 462,029 |

These are exact-position TRAIN diagnostics, not summary accuracy. Residual
retargeting was worse on active-word agreement in this matched comparison.
The parent-to-continuation improvement also includes 1,024 additional updates;
it does not isolate the effect of learning from generated histories.

On the same 16 fixed controls, original-target training recovered 509/1,133
active word IDs and residual-target training recovered 413/1,133. Shuffling
sources reduced these to 75/1,133 and 82/1,133 respectively. This supports source
dependence on those TRAIN cases, not held-out competence. The retained
[matched examples](../../data/experiments/ncm_paper_regime/20261025_onpolicy_matched_fixed_examples_v1/examples.json)
show both arms still emitting repeated fragments, damaged names and incorrect
numbers. Inspection used the same first two controls and fixed longest source,
without selecting favorable examples or assigning new judgments.

The independent state audits replayed all updates, sampling records, optimizer
steps, saved outputs and diagnostic arithmetic, and found all 145 protected
tensors identical. A separate GPU audit re-encoded the TRAIN references and
verified every recorded residual target: 2,048 histories and 1,312,768 code
positions. The protected oracle outputs on the 16 controls stayed unchanged.
Strict checkpoint reload and packed-versus-explicit gradient checks passed.
Historical policy rollouts at every update and teacher-forced logits were not
independently re-executed; the audit does not claim otherwise.

Execution used locked Pixi `ncm`, physical GPU1 (RTX2060), FP32 weights, FP16
training, FP32 generation/evaluation and TF32 disabled. CPU FP32 AdamW and a
256 MiB lossless feature cache bounded memory. Both arms peaked at
2,201,964,544 allocated GPU bytes. Optimizer-loop times were 2,636.57 and
2,649.38 seconds (43.94 and 44.16 minutes). These exclude frozen-body feature
extraction and are not application latency or throughput measurements. They
do not establish that the complete Llama-plus-head system fits on this GPU.

One evaluation launch failed before model loading, inference or judge requests.
The sibling-cache guard constructed keys before installing the existing
128-token rubric. Offline replay reproduced rejection of all 128 new control
keys under the default rubric and acceptance of all 128 after installation.
The [execution amendment](../07%20-%20Status%20Reports/2026-10-25_ncm-onpolicy-residual-evaluation-recovery.md)
uses separate recovery wrappers and fresh residual DEV roots, retaining the
original guard and all completed control files. Twenty-two focused tests and
the full 4,051-first preflight passed. The failed launch, diagnosis and actual
process receipts remain preserved. No model, packet, rubric or score changed.

The final replay verified both complete output populations, checkpoint and
source bindings, pilot/full parity, raw judge responses, all denominators,
same-case decisions and first-judgment preservation. All 3,923 inherited firsts
survived, with 256 new unique firsts, giving 4,179 total. The four pilot judgments
per arm were reused in full grading; unfavorable decisions were not rerolled.

The earlier [oracle recoding probe](2026-10-25_ncm-residual-recoding-probe.md)
demonstrated that privileged reference information could improve reconstruction
after a wrong prefix. This experiment did not turn that capacity into usable
source-only summaries. It establishes failure of this fixed training recipe,
not failure of the published NCM architecture or proof that a GAN is needed.
Larger conditional pretraining and changes to source-to-code transfer remain
hypotheses requiring new prospective studies; no successful remedy is established.

Content-hashed producing sources and plans are preserved with the runs; the
recovery is commit `39ea65c`. Training entry point:
`experiments.structure_net.ncm_onpolicy_residual_training` with each arm and
`train`. The control used `ncm_onpolicy_development`; the residual evaluation
used `ncm_onpolicy_development_recovery`. Independent aggregation and registry
entry points are `finalize_ncm_onpolicy_residual_recovery` and
`register_ncm_onpolicy_residual_recovery`. Completed roots are append-only.

Checkpoint SHA-256 values:

- Original targets: `378966c7e654cd559a26b78356dd9c45c5b46aeb77d10d814a3b6f0b09a3b4e3`.
- Residual targets: `1f1981b692698ea1faecf93382f662502ef762cc475907fa33f3f05dfc87a2e1`.
