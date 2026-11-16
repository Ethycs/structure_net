# NAL Distillation and Post-Compression Recovery Standard

**Standard ID:** NAL-STD-DISTILLATION v1  
**Date:** 2026-11-15  
**Status:** Adopted project standard for new distillation implementations and studies  
**Parent:** [NAL experiment standard](NAL-STD-EXPERIMENT-v0.md)

This standard covers recovery of a compressed language model from a frozen
teacher. It applies to structured pruning, functional-subspace compression,
tensorization, polynomial substitutions, quantization and attached heads.
It does not certify a compression method or retroactively change a running
experiment. Existing frozen studies retain their original contracts.

The intended outcome is preserved task behavior at a measured deployment
benefit. Lower training loss, teacher agreement, successful tensor replay, or
multiple-choice accuracy alone cannot establish general generation recovery.

## 1. Define what must survive

Before recovery, a study MUST declare its intended tasks, input/output lengths,
teacher and student checkpoints, tokenizer/chat-template identity, numerical
precision, compressed components and deployment hardware. The quality margin
MUST have an explicit unit: percentage points, relative loss, or another named
metric. A five-percentage-point accuracy allowance is not a five-percent
relative allowance.

MUST retain distinct references for original teacher, compressed initializer,
and each recovery arm. The student MUST start from an authenticated initializer;
pilot-mutated weights and optimizer state MUST NOT initialize the scientific
run. The original teacher stays frozen and is the recovery target unless the
study explicitly evaluates another teacher.

For J-space/Gröbner students, the first comparison SHOULD keep the fitted bases,
relations, bridges and declared residual route fixed, while training the
compiled parameters and final normalization jointly. Altering geometry,
restoring dense blocks or adding a trainable adapter is a separate intervention
whose parameter, storage and runtime costs MUST be counted.

## 2. Recovery data

The corpus MUST represent intended use. General language recovery requires
ordinary language and complete instruction/response examples. Summarization
recovery requires source/summary pairs with numerical, entity and temporal
coverage. A multiple-choice corpus supports a multiple-choice claim only unless
separate generation evidence demonstrates wider retention.

Each example MUST record a stable example ID, source URI/path and revision or
digest, source-document/conversation group ID, split, task, prompt and response
provenance. Generated responses additionally record generating model/version,
decoding settings and seed. All derivatives of a source group MUST stay within
one split. Exact content duplicates across splits MUST be rejected; the study
MUST document its near-duplicate policy and benchmark exclusions.

TRAIN fits weights and data-dependent preprocessing. DEV selects checkpoints
and training choices. TEST remains unavailable to the training worker and is
opened only after candidate selection is frozen. A previously inspected audit
is exposed evidence, not a fresh test. Benchmark validation/test items used for
DEV MUST NOT subsequently be reported as untouched benchmark results.

Tokenization MUST use the pinned template. The assistant boundary MUST be
verified against the actual token IDs. Shifted causal targets predict token
`t+1` from position `t`; masks MUST include every eligible response token and
the real end-of-response token. Prompt-only, padding and masked tokens MUST NOT
enter a response-only loss. The implementation MUST report input, response and
supervised-token counts separately. Four selected positions in a long sequence
MUST NOT be described as dense supervision.

No silent truncation is permitted. Oversize source examples are rejected or
segmented by a declared deterministic policy. Generation stopped by a token cap
MUST NOT acquire an invented EOS target. Packing MUST preserve document
boundaries through attention masks and position IDs; plain concatenation is
not equivalent. An initial implementation MAY use length-bucketed padded
batches instead of packing.

## 3. Distillation objectives and stages

### A. Dense teacher-forced recovery

Use a declared language-model mask or complete assistant-response mask. The
baseline is forward KL from frozen teacher to student at every selected token:

`L_KD = mean_t T² KL(p_teacher(. | prefix_t, T) || p_student(. | prefix_t, T))`.

Temperature, token/example weighting and any CE or feature-loss coefficient
MUST be explicit. Pure forward KL is a reference configuration; CE mixing or
intermediate-feature matching is an ablation, not an assumed improvement.
Different-width hidden-state targets MUST name their projection and coordinate
frame. Feature alignment cannot replace final behavior evaluation.

### B. Short student-rollout recovery

The student generates TRAIN continuations. The same frozen original teacher
scores those exact prefixes; supervision covers all generated response
positions, including an observed EOS. Each rollout is bound to its generating
student checkpoint, teacher, tokenizer, prompt, decoding settings and token IDs.
The worker MUST reject cached targets from different prefixes or provenance.

Start with short rollouts and extend their cap only under a prespecified rule
based on useful completion length and repetition. The first local profile may
use caps 32, 64 and 128 within its validated total context range. A repetition
detector MUST distinguish natural repetition from failure and report its
definition. If replicating ShortOPD, its detector changes future rollout
budgets; do not silently remove current repetitive suffixes from the loss.

Refreshing rollouts after student updates creates a new round and provenance
record. Reusing older student rollouts MUST be labelled offline or stale-policy
distillation, with a declared refresh/drift bound. It MUST NOT be presented as
fresh on-policy sampling. Forward KL and generalized JSD are separate configured
objectives; claims about one cannot be transferred to the other.

Stages A and B MUST be independently runnable. A full comparison SHOULD use the
same initializer, frozen geometry, prompt population, seed policy, output cap,
checkpoint rules and declared budget axis. Report both supervised tokens and
GPU time; equal examples do not imply equal supervision or compute. Existing
unmatched runs may be contextual evidence, not a clean causal control.

## 4. Efficient execution without changing the objective

Teacher and student MAY alternate on the available GPUs. Teacher targets MAY
be streamed from authenticated disk caches. Full-vocabulary loss SHOULD be
computed in bounded token/vocabulary chunks. A chunked softmax MUST retain the
normalization over the whole vocabulary, and gradients MUST match a dense
reference on a small example. Token-chunking MUST NOT change token weighting.

Top-k teacher probabilities plus a single tail-mass bucket are an optional
approximation. Such a cache MUST store IDs, probabilities/log-probabilities,
temperature, tail mass, token-position mapping and teacher provenance. Student
tail probability is the sum over the omitted vocabulary, not zero. Approximate
losses MUST be labelled and compared against an exact full-vocabulary reference
before deployment. Vocabulary sparsity does not justify sparse token coverage.

Saved compiled models SHOULD be loaded structurally without reprojecting native
weights that are immediately overwritten. Exact state, registration, dtype and
fixed-geometry checks remain required. Immutable large inputs SHOULD be hashed
once at admission and tracked by authenticated file identity afterward; do not
rehash entire teacher banks in each minibatch. Verify consumed checkpoints.

Progress MUST expose phase, initialization time, completed optimizer updates,
supervised tokens, elapsed training time and resource placement. GPU allocation
is not proof of training, and CPU initialization is not a stalled GPU kernel.
Report CPU optimizer/gradient storage and measured GPU peaks separately.

## 5. Validation before scale

The minimum implementation ladder is:

1. CPU tests for causal shifts, complete masks, EOS, split leakage, finite losses,
   target provenance, dense/chunked value and gradient agreement, and teacher
   immutability.
2. A real tiny forward/backward/update/save/reload lifecycle. Verify gradients
   and master updates in every intended compressed block, fixed geometry,
   unchanged teacher and fresh initialization after a discarded pilot.
3. Full-vocabulary inference-path parity on at least a prompt and generated
   prefixes; checking only the answer-letter logits is insufficient.
4. A representative CUDA memory/update pilot under an active reservation,
   followed by an intended-concurrency pilot when sharing resources.
5. A small exploratory recovery budget, with generation evaluated at the
   initializer and early retained checkpoints before larger budgets are spent.

Systems shakedowns are not quality evidence. The proposed local scientific
budget ladder is 100k, 1M and 10M supervised tokens; these are exploratory
engineering budgets, not sufficient or universal recovery requirements. A
study MUST freeze each rung's transition/stop rule before its outcomes are
used. Failure to improve useful generation triggers diagnosis of objective,
data, implementation and representational capacity, not an automatic epoch
increase. A failed systems check preserves its receipt and stops that launch.

## 6. Quality and deployment gates

Each study MUST record paired native/student outputs and evaluate:

| Dimension | Required evidence when applicable |
| --- | --- |
| Language modeling | Held-out NLL/perplexity and teacher divergence with mask and denominator |
| Useful generation | Task completion, instruction following, coherence and natural termination |
| Summarization | Supported facts, numerical/entity/date preservation, omissions and hallucinations |
| Structured output/code | Parser/schema checks, executable tests and exact constraints where appropriate |
| Collapse | Empty outputs, unrelated labels, destructive repetition and token-limit termination |
| General capabilities | Separate benchmark families, not only pooled accuracy |
| Efficiency | Exported bytes, actual peak memory, prefill and decode timings at intended lengths |

Semantic summaries MUST NOT be scored by exact wording. Deterministic checks
SHOULD grade numerical, schema and executable constraints. Semantic judging
MUST record judge identity/version, rubric, prompt, settings, raw verdicts,
missing/time-out handling and a reviewed sample. A teacher's agreement with its
student alone is not an independent correctness judgment.

Native controls receive an adequate, matched generation budget. Native failures
and truncations remain visible. A tiny canary set detects gross collapse but
cannot establish a five-point population accuracy bound. Use preregistered
paired intervals and sample-size assumptions for a noninferiority claim.

Checkpoint eligibility MUST include generation and task gates before aggregate
ranking. Earlier equivalent checkpoints SHOULD win ties. DEV may select; TEST
may confirm or fail but MUST NOT select a replacement. A failed family or
generation gate cannot be rescued by lower training loss or another metric.

Deployment requires a fresh selected-checkpoint reload, exported/executor
parity, and isolated measured timing. Concurrent training wall time is not
inference throughput. Any promised storage/speed gain includes residual paths,
adapters, fixed geometry, embeddings and required runtime state.

## 7. NAL and reservation integration

Use NAL `Experiment`, `ExperimentResult`, `ExperimentStatus`, scientific
fingerprints and `ExperimentResultLedger` where they fit the workload. A
study-specific worker is valid when it follows the parent standard's artifact
and lifecycle contract. Do not build a second generic scheduler or registry.

The reservation controller governs physical GPUs. `inference-ctl lease run`
sets visibility before Python starts; NAL then uses logical CUDA ordinals.
NAL slots do not grant a reservation. CPU work uses logical device `-1` with
CUDA hidden. Project execution uses locked/offline Pixi. Never stop another
authorized experiment or change its frozen source to admit a new cell.

Each replicate writes strict JSON, model/data/source fingerprints, raw quality
records, real checkpoints, optimizer/token progress and terminal failures.
Resume accepts only matching successful receipts with required artifacts.
Aggregate and typed records distinguish `prepared`, `systems_passed`,
`exploratory_quality_pass`, `exploratory_improvement`, `quality_gate_failed`, `confirmed_recovery` and
`deployment_validated`; these are not interchangeable. Confirmation requires
all declared evidence, not just a boolean supplied by the trainer. Completed
studies receive a measured report and conservative registry read-back.

## 8. Initial implementation profile and current work

The first implementation provides reusable response records, streamed targets,
dense objectives with bounded token chunks, an explicitly approximate top-k/tail
objective, short-rollout rounds, token-budgeted training, generation-aware
evidence gates and NAL lifecycle/ledger adapters. Study-specific producers still
supply generation, semantic judging and checkpoint selection. Tests
and a tiny real CPU lifecycle precede reservation-backed Qwen pilots.

The Qwen profile retains the native Qwen2.5-3B tokenizer and frozen teacher,
the compiled student and its fixed geometry, and the currently validated
416-token total positional envelope. Longer application inputs require a
separate geometry/length validation; a 128-token summary cap does not imply
support for 2,048- or 6,300-token inputs. Both existing progressive and
all-at-once studies continue with their original objectives and receipts.

The new corpus, quality thresholds, selected initializer and first scientific
budget MUST be frozen in a dated study plan after corpus inventory. Implementing
this standard or passing a toy lifecycle is not a claim that Qwen is recovered.

## 9. Evidence motivating the standard

- [Minitron, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/4822991365c962105b1b95b1107d30e5-Paper-Conference.pdf): sequence-level logit distillation and structural compression recovery.
- [Nemotron Nano 2 technical report](https://research.nvidia.com/labs/adlr/files/NVIDIA-Nemotron-Nano-2-Technical-Report.pdf): forward-KL recovery and mixed pretraining/post-training data.
- [ShortOPD v2, 2026](https://arxiv.org/abs/2607.13124v2): generation collapse after pruning, dense student-context targets and short-to-long rollout allocation. This is a research method, not validation of our Gröbner geometry.
- [Lightning OPD v3, 2026](https://arxiv.org/abs/2604.13010v3): offline teacher scoring under explicit consistency and drift assumptions.
- [Local failed recovery assessment](../08%20-%20Analysis/jspace/2026-11-15_qwen3b-broad-recovery.md): development gains did not establish independent accuracy or ordinary generation.
