# Auxiliary distillation of one compressed Qwen final block

Status: **completed exploratory comparison**, `confirmed=false`.
Study: `JSPACE-QWEN-LAYERKD-01`.
Root: `data/experiments/jspace_qwen_layerkd/20261113_v1`.
Machine-readable multi-arm result: `result.json`, SHA-256
`3e5f0a5ba5c27d8a5d3c4ae078b0f444465b364063a716516886335b238f052f`.

Neither auxiliary objective establishes a reliable improvement at this budget.
Euclidean supervision scores **388/512**, versus **385/512** for output-only
distillation and **387/512** native. The Euclidean arm gains five answers and
loses two against output-only; the descriptive exact paired p-value is 0.4531.
Jacobian supervision scores **385/512**, exchanging one correct answer for one
incorrect answer relative to output-only. All compressed arms have the same
deployed architecture, tensor bytes and approximately 40 tok/s throughput.
This is a test of auxiliary supervision on **one final block**, not a test of
two-block staged compression, a previous compact teacher, or joint stack recovery.

## Measured comparison

| Model/objective | Fresh accuracy | Selected epoch | Development endpoint loss | Development KL | Graph tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native | 387/512 (75.59%) | — | 1.69407 | 6.78e-8 | 40.42 |
| Output-only KL+CE | 385/512 (75.20%) | 12 | 0.70913 | 0.12328 | 40.39 |
| Output + projected Euclidean | 388/512 (75.78%) | 7 | 0.93798 | 0.18239 | 40.40 |
| Output + full-logit Jacobian metric | 385/512 (75.20%) | 11 | 0.82014 | 0.15756 | 40.22 |
| Native timing repeat | Same native model | — | — | — | 40.29 |

All compressed arms score 47/64 on development. Checkpoints are selected by
packed development **endpoint** loss, excluding the auxiliary term; every arm
still trains all 12 epochs. The endpoint-only arm's lower development divergence
does not translate into the highest fresh answer count in this small sample.
Native endpoint loss includes ground-truth CE and is therefore not expected to
be zero even though teacher/native KL is near zero. These are four-position
training/development losses, not language-model perplexity over full sequences.

| Paired comparison | Gained correct | Lost correct | Changed answers | Net answers | Exact two-sided McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Output-only vs native | 0 | 2 | 3 | −2 | 0.5000 |
| Euclidean vs native | 4 | 3 | 9 | +1 | 1.0000 |
| Jacobian vs native | 0 | 2 | 3 | −2 | 0.5000 |
| Euclidean vs output-only | 5 | 2 | 8 | +3 | 0.4531 |
| Jacobian vs output-only | 1 | 1 | 2 | 0 | 1.0000 |

Changed-answer totals can exceed gained plus lost correctness because a model
can exchange one wrong option for another. These one-seed descriptive tests do
not establish superiority, equivalence, or noninferiority. The Euclidean result
is a small positive observation worth preserving, rather than a demonstrated
training breakthrough or evidence that its larger KL is harmless generally.

| Family | Native | Output-only | Euclidean | Jacobian |
| --- | ---: | ---: | ---: | ---: |
| ARC | 101/128 | 101/128 | 101/128 | 101/128 |
| BoolQ | 105/128 | 105/128 | 107/128 | 105/128 |
| HellaSwag | 99/128 | 99/128 | 98/128 | 98/128 |
| WinoGrande | 82/128 | 80/128 | 82/128 | 81/128 |

Deployed tensor storage is **6,151,666,112 bytes** for every compressed arm,
versus 6,171,877,632 native (0.327% less). The auxiliary metric, teacher and
gradient diagnostics are training-only. Isolated block prefill is 5.306 ms for
output-only, 5.301 ms for Euclidean and 5.427 ms for Jacobian, versus native
3.523 ms initially and 3.549 ms at the end. Timing uses the same eight
development prompts, batch one and 32 greedy tokens including prefill, with
three CUDA-graph repeats. It does not show a speed improvement from changing
the training objective. Compare against contemporaneous native timings rather
than the earlier campaign's lower isolated-block milliseconds.

## Controlled intervention

All arms use the parent's rank-1536/head-64 Gröbner architecture, fixed affine
bases/means, 24 polynomial anchors and 1,024 removed MLP channels. Only the final
block and final RMS gain train; the native prefix and tied vocabulary stay
frozen. They start from one shared unadapted initializer, not a selected adapted
parent checkpoint. Its SHA-256 is
`b440b1d69a9de2d20a1ef2ef7c8a4f426baa3e31f453ec630ea843be4b28aa7f`.
Independent CPU inspection confirmed all 26 epoch-zero tensors are identical
across all arms and that initializer. The selected endpoint-only model's 23
packed tensors exactly reproduce the parent's selected Gröbner checkpoint.
This is a useful reproducibility control; the fresh questions are different.

Each arm receives 512 training questions, 12 epochs and 1,536 optimizer updates,
with identical seed-861 example order, AdamW settings and accumulation. The
endpoint objective remains `0.7*T^2*KL + 0.3*CE`, T=2, at the same four output
positions. The Euclidean target is the teacher boundary projected into the fixed
student affine space, avoiding an unavoidable discarded-direction floor in that
loss. The discarded teacher floor is separately retained in calibration data.

The Jacobian auxiliary uses the full native-coordinate teacher difference and
the analytic differential of the **original** final RMSNorm and tied readout.
The readout Gramian `E.T @ E` is accumulated in FP32 with TF32 disabled. Together
with the RMS differential it gives the full-vocabulary first-order metric for
this positionwise suffix, without a random-probe bank. It is exact in real
arithmetic as a local differential, with ordinary floating-point approximation;
it is not an exact finite-perturbation loss or the KL/Fisher metric. This shortcut
does not extend unchanged through earlier attention layers, where cross-token
Jacobian effects matter.

The raw auxiliary coefficients are fixed from 32 balanced training calibration
cases so their aggregate initial gradient norm ratio is 0.1 relative to the
endpoint gradient over shared block parameters. They are 0.9271004 for Euclidean
and 0.2439192 for Jacobian. Fixed initial-loss normalizers preserve those raw
weighted objectives. No auxiliary fitting uses held-out data, and the final
RMS gain is excluded from coefficient calibration because the auxiliary losses
do not depend on its trainable value.

## Gradient evidence and interpretation

Initial median gradient cosine across calibration cases is 0.0783 for Euclidean
and 0.3172 for Jacobian. Later diagnostics on **one fixed training case** show
that initial balancing does not maintain a 0.1 ratio throughout optimization:

| Auxiliary | Weighted auxiliary/endpoint gradient norm over epochs | Gradient cosine over epochs |
| --- | ---: | ---: |
| Euclidean | 0.8515–1.2184 | −0.9581 to −0.6832 |
| Jacobian | 0.8817–3.2857 | −0.8160 to +0.2229 |

These are measured gradient conflicts, not merely comparisons of loss values.
They concern one diagnostic case and do not estimate the training population's
conflict frequency or prove why a particular held-out answer changed. They make
one plausible mechanism explicit: as endpoint recovery progresses, a fixed
auxiliary weight can exert a much stronger, opposing constraint than its initial
calibration suggests. Other differences between local metrics and endpoint KL
remain possible. No coefficient sweep, decay schedule or gradient-surgery test
was run, so none is established as a solution here.

The appropriate next baseline remains output-only **staged joint recovery** of
a faster multi-block structure. Keep the auxiliary comparison available without
assuming it helps every compression stage. The measured small Euclidean gain
does not establish multi-block stability, and the Jacobian tie does not invalidate
task-Jacobian calibration, Gröbner compilation, or recovery distillation.

## Integrity and limits

All six main stages completed, and the separate reload process reproduced every
held-out record, including logits, for all three checkpoints. Audit verified 33
live/archived producer sources and all 154 stage artifacts against their hashes.
The scientific fingerprint is
`037eca072c28647f42d376934bde2158b59cb4bdc8ba6a765ef816f5b4f7eff6`.
The shared initializer, fixed update budgets and identical deployed sizes were
checked separately. The maximum recorded reserved GPU memory was 8,044,675,072
bytes, under the 11,264 MiB allocation on the RTX 3060. The active campaign lease
recorded in the manifests is `b351b1730ddc`; release belongs to its owning wrapper.

The 512 new evaluation identities are unique and disjoint from all parent task
identities and this study's train/development sets. The split is balanced at 128
questions per family, limited to supported choices and prompts ≤384 tokens.
This remains a single-seed exploratory multiple-choice experiment on one pinned
Qwen checkpoint. It is not evidence of summarization quality, general generation
accuracy, long-context behavior, an optimized deployment kernel, or a staged
teacher transition. The prior one-block campaign and this auxiliary comparison
must remain distinct from the forthcoming performance-first compact-stack study.

Protocol: `docs/07 - Status Reports/jspace/2026-11-13_jspace-layerwise-distillation-plan.md`.
Producer: `experiments/structure_net/jspace_qwen_layerkd.py`, with its frozen
sources, calibration, epoch histories, selected delta checkpoints and per-case
outputs under the dated root. All GPU work used the locked `ncm` Pixi environment
and the reservation system. This analysis adds no training, source mutation,
retrospective checkpoint selection or confirmatory claim.
