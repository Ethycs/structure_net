# Qwen final-block affine coordinates and benchmark-derived polynomial compression

Status: **completed exploratory campaign**. Study: `JSPACE-QWEN-AFFINE-01`.
Classification: `partial_quality_recovery_no_speedup`; `confirmed=false`.
This is a descriptive exploratory classification, not a retrospectively imposed
confirmatory threshold. Artifact root:
`data/experiments/jspace_qwen_affine/20261113_v2`.
Aggregate SHA-256:
`e11fe670d527d835655ba8023e9ae0a4a05c2975834c20a6422b6bd11eef8659`.

Compressing and adapting Qwen2.5-3B's final block preserved most measured task
accuracy: **388/512 versus 393/512 native**, a net difference of five answers
(−0.98 percentage points). Whole-model generation remained around **41 tok/s**;
the isolated compressed block was slower. The matched Gröbner arm removed
1,024 native nonlinear channels, saved another **10.04 MB**, and produced
identical choices to its non-Gröbner control on all 512 fresh questions. Its
isolated block prefill took **17.7% longer**. This supports useful approximate
functional compression at this boundary, while the present implementation does
not establish an efficiency improvement.

## Accuracy, size and timing

| Arm | Fresh accuracy | Resident tensor GB | Saving vs native | Graph tok/s | Isolated block prefill ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native | 393/512 (76.76%) | 6.17188 | — | 40.51 | 3.405 |
| Rank 1536, head 128 | 388/512 (75.78%) | 6.16978 | 0.034% | 40.87 | 4.094 |
| Rank 1536, head 64 | 388/512 (75.78%) | 6.16171 | 0.165% | 41.04 | 4.014 |
| Rank 1024, head 64 | 388/512 (75.78%) | 6.12134 | 0.819% | 41.04 | 4.248 |
| Rank 1536, head 64, Gröbner | 388/512 (75.78%) | 6.15167 | 0.327% | 40.87 | 4.724 |
| Native repeated at end | Same weights | 6.17188 | — | 41.09 | 3.057 |

GB and MB are decimal. Tensor storage includes basis, anchor, shift and other
buffers, with the tied vocabulary counted once. Only one of 36 blocks changes;
the affine buffers also consume a substantial part of the projection-weight
saving. Parameter reduction alone would overstate deployed storage reduction.
These are model tensor bytes, not allocator peak memory or checkpoint file size.

Timing uses the same eight development prompts, batch one, 32 greedy tokens with
prefill included, and three repeats under the same CUDA-graph runtime. Capture
setup is amortized. Isolated block prefill is the mean of eight per-prompt
measurements, each averaging 20 calls after warmup. It includes the implemented
block interface, rather than timing only one smaller matrix multiplication.
Ordinary HF generation measured 9.33 tok/s initially and 10.35 on the native
repeat; compressed arms measured 10.31–10.84. Both runtime series exhibit
baseline drift. The compressed graph results lie between the two native
measurements, so their small improvement over the first native run does not
demonstrate a speedup. No kernel profile was collected to attribute the slower
block to a particular operation.

Every arm loses six native-correct answers and gains one native-wrong answer:
seven changed answers and 98.63% agreement. The descriptive exact two-sided
McNemar test gives p=0.125. This does **not** establish either a statistically
significant accuracy decrease or accuracy equivalence/noninferiority.
Rank-1536/head-128, rank-1536/head-64 and Gröbner arms have identical choices;
rank-1024 differs from them on two questions despite the same total score.

| Task | Native | Rank 1536/head 64 + Gröbner |
| --- | ---: | ---: |
| ARC | 106/128 | 106/128 |
| BoolQ | 103/128 | 102/128 |
| HellaSwag | 99/128 | 97/128 |
| WinoGrande | 85/128 | 83/128 |

On the **secondary, previously used** 256-question pool, native scores 210/256
and all four compressed arms score 211/256. That result does not replace the
fresh primary comparison. Neither pool measures summarization or free-generation
quality, and these scores are not the supplied classifier recipe's accuracy.

## What ran and what the controls establish

The pinned model is Qwen2.5-3B-Instruct, revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. The first 35 blocks and tied vocabulary
remain frozen. Local orthogonal coordinates, position-dependent training means,
exact affine RMSNorm algebra, RoPE-compatible Q/K pair selection and GQA-aware
value projection are compiled into block 35. Dynamic cache attention operates
at the smaller head width; the shared static-cache allocation remains padded,
so no static KV-cache storage reduction is claimed. The block reconstructs the
native residual width once before the original final RMSNorm and tied readout.
Both residual updates are projected into the retained affine coordinates; the
omitted dynamic directions are not restored by an exact residual correction.

The full-rank control matches all 64 native development choices and has
full-vocabulary KL 0.0000672. The native forward agrees with cached teacher
logits to KL 0.0000000678. These controls support the implementation and cache
alignment; they do not prove that a reduced subspace preserves the model.

All four arms train on the same 512 questions, with 64 development questions,
for 12 epochs (6,144 example presentations per arm). Loss is output KL+CE only:
`0.7*T^2*KL + 0.3*CE`, T=2, at three prompt positions and the final answer-letter
position. There is no hidden-state reconstruction loss. The current block's
coefficients and final RMS gain adapt jointly; the prefix and readout remain
frozen. Selection minimizes packed-copy development loss, followed by exact
selection-time/serialized-model checks.

| Arm | Selected epoch | Initial dev KL | Selected dev KL | Selected dev accuracy |
| --- | ---: | ---: | ---: | ---: |
| Rank 1536/head 128 | 11 | 2.65260 | 0.12016 | 49/64 |
| Rank 1536/head 64 | 12 | 2.69181 | 0.12302 | 47/64 |
| Rank 1024/head 64 | 12 | 3.90245 | 0.21480 | 48/64 |
| Rank 1536/head 64 + Gröbner | 12 | 2.80045 | 0.12328 | 47/64 |

Accuracy before adaptation was already 48–49/64 on development, despite much
larger full-vocabulary KL. Thus the answer counts alone would conceal substantial
distribution distortion. Training recovers much of that distributional agreement;
the remaining KL is not a test-set perplexity measurement.

## Functional basis and polynomial evidence

Calibration uses 128 training examples, two independent groups of eight
full-vocabulary random-sign VJPs, and four output positions per example. The
estimator includes the actual final RMSNorm and all vocabulary logits, with
BF16 forward/backward and FP32 Gramian accumulation. The expected Gramian is
the average full-output Jacobian Gramian; it is not a materialized exact Jacobian.

Rank 1536 retains 93.60% of the combined estimated trace energy. A basis from
one probe group retains about 82.95% on the other group; projector overlap is
0.7825 versus a random-subspace reference of 0.75. Rank 1024 retains 80.47%
combined energy, about 63.2% cross-group energy, and overlap 0.5856 versus 0.5.
The combined-fit numbers therefore exceed independent-group retention, and the
large-rank overlap should not be presented as strong stability without its
random reference. The two KV groups retain 95.99% and 94.68% of diagonal Q/K
pair-sensitivity score, but only 74.45% and 84.58% of V second-moment energy.
These are local calibration diagnostics, not guarantees for finite compression.

The native MLP derivative bank contains 8,192 rows over 11,008 coordinates.
A 64-component randomized sketch retains 90.18% of standardized derivative
energy and selects nested functional anchor sets. Degree-two relations fit
2,048 native training activation rows, using training-only normalization.
Development compares their projected-output error at the actual compact
interface, without refitting the relations to development data:

| Anchors | Polynomial features | Compact-interface relative output RMSE |
| --- | ---: | ---: |
| 16 | 153 | 2.9201% |
| 24, selected | 325 | 2.9066% |
| 32 | 561 | 3.1039% |

The selected arm physically removes 1,024 gate/up channels, retains 9,984 native
nonlinear channels, and packs the polynomial contribution into the output
projection. Its matched control has the same rank, head size, training data,
objective and budget. The additional storage saving is exactly 10,043,328 bytes;
the prefill-time ratio is 1.1769, with zero changed test answers.

Common-fit eligibility at normalized channel RMSE ≤0.5 produces 1,371 task uses
of 975 unique elements on compact-interface development data: 396 repeated uses
(28.88%), with 11 elements eligible across all four tasks. Native-interface
development gives 3,722 uses, 2,672 unique elements, 1,050 repeats (28.21%) and
76 shared across all tasks. These are common-fit eligibility and overlap counts,
not a full off-diagonal transfer matrix or proof of identified semantic CAVs.

**None of the 1,024 selected removals meets that channel threshold on all four
compact-interface task families.** Removal was chosen by training error weighted
by output-column norm, not by this diagnostic threshold. Small total projected
output error can coexist with poor individual-channel reconstruction. The fixed
removal budget therefore does not establish reliable coverage of every removed
channel. The rational graph generators have a valid Gröbner certificate for the
fitted surrogate; that certificate is not an exact identity of the neural model.
Adaptation subsequently learns the packed output coefficients.

## Integrity, boundaries and durable record

All nine main stages completed under reservation `a77a38f15c93` on the RTX 3060.
Peak reserved memory was 7,946,108,928 bytes. The frozen producer fingerprint is
`8cf88053faf728c6d66a13dbe3fd9f8d12780e511b750c1b65a4b6cda3c5181e`.
All 28 live and archived producer files and all 236 completed-stage artifact
hashes passed final verification. Fresh-process reload reproduced every saved
choice logit and record for all four arms on all 512 fresh examples. Checkpoints
are block-plus-final-norm deltas requiring the pinned native model, not complete
standalone Hugging Face models.

The v1 systems pilot failed on unsupported CUDA int64 matrix multiplication in
an overlap diagnostic. V2 uses Boolean intersection and reduction for the same
counts. The failed pilot and unexecuted v1 main root remain preserved; no failed
quality result was hidden. The v2 systems pilot passed before main execution.
No producing source changed during the main run. The controller subsequently
reported normal state and no bookings or leases; the campaign wrapper released
its own reservation. This is a historical closeout, not a claim that GPUs remain
unreserved for later work.

This is one seed, one final block, four task families, prompts ≤384 tokens and
32-token timing generations. The 512 fresh identities exclude prior train,
development and evaluation identities; the benchmark families and teacher are
shared with earlier exploratory work. There is no independent confirmation,
general language-model accuracy claim, long-context claim, or proof that stacking
will preserve these results. Staged output-only recovery with a previous compact
teacher, joint recovery of adjacent blocks, and optional local distillation are
prospective follow-ups, not methods or results of this campaign.

The next engineering question is why the smaller block costs more time; the
present measurements identify that problem without attributing it to an
unprofiled kernel. The research result is narrower and useful: the final block
can be compressed and adapted close to native benchmark accuracy, and an
additional benchmark-derived polynomial path saves state without changing these
answers. Quality recovery, broader coverage and execution efficiency remain
separate measurements.

The machine-readable aggregate is `campaign_results.json`; raw outputs remain
in `result.json` and the stage directories. The conservative searchable record
is `data/hypothesis_registry/jspace_qwen_affine_20261113_v2.json`, with independent
readback in `registry_readback.json`. Finalization and the research-log entry are
written by `experiments/neural_architecture_lab/finalize_jspace_qwen_affine.py`
using the locked default Pixi environment with CUDA hidden. The original GPU
producer uses the locked `ncm` environment and the reservation wrapper. Completed
stage roots are append-only; reruns require a fresh prepared root.
