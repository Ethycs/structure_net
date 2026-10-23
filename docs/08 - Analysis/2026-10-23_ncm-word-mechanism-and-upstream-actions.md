# The word stage fails even when the coarse plan is supplied

Date: 2026-10-23. Exploratory hypothesis `ncm-native-word-mechanism-v1`.
Classification `parallel_word_recovery_failed_even_with_oracle_plan`.
The actual source-disjoint summary endpoint remains **0/64 whole-summary
passes**, with64valid Terra judgments and0readable summaries.

The final failed checkpoint was tested on the first four fixed summary
TRAIN cases and first four raw-language diagnostic cases. The table measures
valid-position token recovery, **not summary accuracy**. Oracle plans and
true previous words reveal target information and are unavailable at inference.

| Cohort | Arm | Plan | Word input | Token recovery, not task accuracy |
| --- | --- | --- | --- | --- |
| language_diagnostic | A | Source-predicted | Four parallel refinements | 37/512 (7.23%) |
| language_diagnostic | B | Target-derived oracle | Four parallel refinements | 41/512 (8.01%) |
| language_diagnostic | C | Source-predicted | True previous words | 371/512 (72.46%) |
| language_diagnostic | D | Target-derived oracle | True previous words | 370/512 (72.27%) |
| summary_train | A | Source-predicted | Four parallel refinements | 32/346 (9.25%) |
| summary_train | B | Target-derived oracle | Four parallel refinements | 37/346 (10.69%) |
| summary_train | C | Source-predicted | True previous words | 331/346 (95.66%) |
| summary_train | D | Target-derived oracle | True previous words | 332/346 (95.95%) |

Correct coarse plans barely help the four-pass decoder. Stronger recovery
under true previous words shows retained teacher-forced language ability,
especially on training summaries. It does not demonstrate free-running fluency
or generalization. The contrast locates a major failure in the noisy parallel
word interface; predicting better coarse codes alone cannot repair these cases.
It does not prove the coarse predictor is good, isolate one training defect,
or establish a failure of the NCM paper itself.

The two-case pilot passed before the fixed eight-case study. Arm A reproduced
the stored training reload probe exactly, including all raw word IDs and
coarse codes. Body, books, new weights and selected CPU caches remained
unchanged; all logits were finite. Both workers terminated successfully.
No new teacher calls, training, development targets, final test, fallback,
external verifier or speed benchmark were used.

The most relevant published training implementation is
[JacobiForcing](https://github.com/hao-ai-lab/JacobiForcing/tree/028dbd6283fdc6fa356f0b109ce04c5ca538e49f).
Its actual multiblock trainer combines clean AR CE×10 and noisy-token soft
consistency with a detached current clean branch in the same objective.
Our failed model randomly alternated clean/noisy word updates, used offline
original-teacher top32targets, and forced predicted coarse histories on every
word update. It never qualified correct-plan parallel decoding before scaling.
The released training/inference recipe is materially different; its published
coding/math results have not been reproduced locally.

| Reference | What we can copy | Boundary |
| --- | --- | --- |
| [Official NCM](https://github.com/HazyResearch/ncm/tree/c70274273f489abc73832b2b98584a8483b6b203) | Released architecture, codec/checkpoints, inference tests | Inspected release contains no trainer; local32/32decoded-text inference agreement is not training replication. |
| [JacobiForcing trainer](https://github.com/hao-ai-lab/JacobiForcing/blob/028dbd6283fdc6fa356f0b109ce04c5ca538e49f/JacobiForcing/train/soft_flexattn_cllm_trainer_multiblock.py) | Paired clean/noisy losses, detached clean targets, explicit trajectory/mask handling | Its full-model examples and iterative block decoding differ from our frozen3B/nine-pass system. |
| [SpecForge](https://github.com/sgl-project/SpecForge/tree/f586b1ee40b5e4bf3ef03954849024b57a5edb7a) | Native vocabulary/features, concrete DFlash training and Qwen3-8B configuration | Full-parent verification protects speculative outputs; it is not standalone NCM accuracy. |

Next, test paired clean/noisy word updates and a mix of correct versus predicted
coarse plans in a small, explicitly target-conditioned decoding control.
Require that control to work before another large end-to-end run. Keep original
MLP weights frozen, retain source-only generation for all task evaluation,
and judge complete unseen summaries before any acceleration claim. A passing
training control will admit more investigation, never establish95%quality.

Aggregate `data/experiments/ncm_paper_regime/20261023_native_word_mechanism_results_v1/campaign_results.json`, SHA256 `aeaaa7c5391cb595bf0f5aa59fe50d2b45c4e7f14fc905b7821fa76e34529f73`. Confirmed=false.
