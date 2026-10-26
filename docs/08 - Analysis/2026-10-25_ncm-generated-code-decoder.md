# Adapting the decoder to generated codes did not improve summary accuracy

**Result: 0/128 source-disjoint complete summaries passed (0%).** All 128 Terra
judgments were valid. The new checkpoint and its parent both failed the same
complete-summary endpoint. This fixed intervention does not approach the 95%
task-quality goal; lower training loss did not establish semantic improvement.

Hypothesis `ncm-generated-code-decoder-v1`; classification
`DEV_semantic_quality_failed`. This is one exploratory training seed and remains
underpowered for architecture-level claims. The
[prospective plan](../07%20-%20Status%20Reports/2026-10-25_ncm-generated-code-decoder-plan.md)
fixed the update budget, inputs, frozen modules and unchanged evaluation gates.

| Held-out cohort | Valid judgments | Complete-summary passes | Readable outputs |
| --- | ---: | ---: | ---: |
| Short | 64/64 | 0/64 | 0/64 |
| Long | 64/64 | 0/64 | 1/64 |
| Combined | 128/128 | 0/128 | 1/128 |

No output passed the faithfulness or salience component. None was empty or above
the 128-stored-token limit. The descriptive Wilson 95% interval for the combined
pass fraction is 0–2.91%. Sol was not admitted because Terra did not reach 61/64
in each cohort. The sealed final 320 and application-speed test remain unopened.
Every prior first judgment is retained: 3,667 inherited plus 128 new firsts,
3,795 total. The four transport-pilot judgments were reused in full grading;
unfavorable valid judgments were not retried.

The intervention trained only the existing 18,870,656-parameter text decoder on
the parent's actual source-only generated codes. Original Llama features, codec
encoder and VQ, generator, concept/source adapters and word/source adapters stayed
frozen. The actual code and word predictions for the fixed controls were
regenerated before training to qualify the saved-code inputs. No new attention
layer, repair pass, copying mechanism or output filtering was added.

| Training diagnostic | Parent | Adapted decoder |
| --- | ---: | ---: |
| Matching active native word IDs | 11,105/30,109 | 12,761/30,109 |
| Word agreement on TRAIN | 36.88% | 42.38% |
| Source-disjoint complete-summary passes | 0/128 | 0/128 |

Word agreement is exact-position agreement on training references, not semantic
accuracy. The earlier 100% reconstruction result supplied correct target codes
to the parent decoder; this study did not re-test that oracle control after
adapting the decoder. It must not be presented as the new checkpoint's measured
reconstruction accuracy.

Training used all 732 unchanged engineering examples from eight source components:
1,024 optimizer updates, accumulation four, 4,096 presentations, five or six per
example. First/last 16-update mean word cross-entropies were 11.3511 and 3.0661.
Optimizer-loop time was 1,612.31 seconds (26.87 minutes). Peak full-lifecycle GPU
allocation was 1,490,403,328 bytes on physical GPU1, the RTX 2060. These are training
measurements, not inference performance.

The locked Pixi `ncm` runtime used FP32 weights and CPU AdamW state, FP16 training,
FP32 evaluation and TF32 disabled. Source features retained their full original
FP32 precision and all positions. The lossless disk cache peaked at 268,419,072
bytes, within its 256 MiB bound. Model, optimizer, scaler, sampler and RNG states
were saved and strictly reloaded; diagnostic outputs repeated exactly. The
independent CPU audit replayed all samples, losses, update counters and frozen
tensors. Seven focused tests and the two-update CUDA lifecycle pilot had passed.

An evaluation-launch plumbing error treated the independent audit as a model run
and looked for a nonexistent `producer.json`. It exited 1 before model inference,
outputs or judge requests. The preserved failure receipt records that boundary.
The corrected wrapper checks the audit's actual exit receipt, status, auditor hash
and every audited artifact. No training or scientific criterion changed. The two
inference-pilot outputs repeated and matched their full-run counterparts. A final
independent replay verified all source bindings, decoded outputs, exact judge
packets, first judgments, component arithmetic and admission decisions.

This result narrows the diagnosis: adapting only the decoder to real generated
codes at this budget was insufficient. It does not establish that arbitrary
additional decoder training cannot help, or isolate source coverage, accumulated
code errors and source conditioning from one another. The training set covers
only eight components, and its stored targets are not independent semantic gold.
A matched comparison of source conditioning and broader task-correct supervision
is better motivated than declaring codec capacity to be the sole remaining fault.
There is no matched extra-oracle-training arm, so this before/after comparison
also cannot isolate conditioning on generated errors from additional training.

The searchable [data viewer](../../data/experiments/ncm_paper_regime/20261025_generated_code_decoder_data_review_v1/index.html)
contains all 732 TRAIN and 128 DEV sources, predictions and available judgment
reasons. Its HTML JSON payload, JSONL export, source hashes and JavaScript syntax
were independently checked; earlier viewers remain intact.

Artifacts under `data/experiments/ncm_paper_regime/`:

- `20261025_generated_code_decoder_train_v1`: checkpoint and full TRAIN outputs.
- `20261025_generated_code_decoder_train_audit_v1`: independent training replay.
- `20261025_generated_code_decoder_DEV_grade_v1`: complete semantic judgments.
- `20261025_generated_code_decoder_results_v1`: independent aggregate and registry readback.
- `20261025_generated_code_decoder_CPU_checks_v1/inference_launch_failure.json`: preserved systems failure.

Checkpoint SHA-256:
`5e808e5b204067f839fe627d99f45158d24befe5114d0712e33b00c6a2e0993b`.
Training producer fingerprint:
`7b0842050428b91a4948cc6f911eeb87e26a4f19d8fe6f98d290bdb815e8221c`.
