# JSPACE-ACTIVE-CAV-HEADS-01: prospective pilot protocol

Written before new neural outcomes. One frozen Qwen2.5-3B-Instruct checkpoint,
BF16, previously hashed revision. This study has two distinct endpoints:
country-specific causal intervention and bounded-task head accuracy. Neither
substitutes for the other. It does not compute an exact Gröbner basis or claim
to replicate the full Jacobian-lens or VI-Net papers.

## Data and positions

Fresh named worlds, excluding the preceding relation, component and donor
localization studies. Train 512 paired worlds, development 64, evaluation 64,
plus four separate systems-shakedown worlds. Independent country×answer cells
are exactly balanced in each quality split; split name pools are disjoint.
All use the shared JSON lookup format. Anonymous mapping permutations may recur.

Retain only source/donor questions with identical token lengths, identical
pre-question tokens and a 21-token suffix. This restricts queried names to the
single-token subset and aligns positions without interpolation. Generate
max(1024,8×requested count) candidates per quality split, 128 for shakedown;
select deterministically and fail if any balanced cell is short. Seed 233;
shakedown adds 1,000,003. Source and answer-only donor worlds are also disjoint.

Two zero-based block-output sites: **24, question suffix excluding its final
token** (20×2048 scalars), and **29, final token** (2048 scalars). Keep the 20
positions separate, flattening only for linear algebra. These sites were chosen
from an earlier independent donor-localization study, not this evaluation.

Each world has a query donor (same facts, other queried person/country) and an
answer-only donor (same person/country, but swap two values in the country→colour
dictionary). The latter changes the answer while holding the country fixed.
It is not an identical-prefix donor and is never treated as such.

## Position-aware CAV and J construction

Capture source and query-donor states on train and development only. Fit their
paired differences by joint least squares against country-one-hot differences
and colour-one-hot differences. Require design rank six. Center each four-row
effect matrix. This controls the linear colour effect when estimating country
directions; it does not guarantee that a learned direction is semantically pure.

Build a new task-conditioned Jacobian dictionary on 32 balanced training
sources. Retain the prior 94-token inventory but recompute all derivatives.
For each source position pattern, average `W_U[token] @ d(h35,last)/d(H_site)`
before the final norm, preserving source positions. Batch two cotangents; use
eager attention. This is a task-specific positional dictionary, not a reused
final-token dictionary or the paper's global source/target average.

Split each country CAV into a sparse nonnegative dictionary component (budget
25, initial four country atoms, existing bounded NNOMP+NNLS implementation) and
its exact arithmetic remainder before FP32 storage. The full-cone endpoint is
diagnostic. Preserve the shared full-CAV coefficients for all component edits.

## Causal endpoint

Project the actual donor-minus-source pattern onto the learned country CAV row
space. This is **CAV-restricted donor transfer**, not a label-only synthetic swap.
Use the same coefficients for full/J/remainder pieces. Main arms at both sites:
dense donor, full country, J, remainder, norm-preserving signed/permuted full,
signed/permuted J, remainder rescaled to J norm, and colour-CAV transfer.
Random seed 251+layer; edits touch only the declared prefill positions.

For the answer-only donor, test dense, full-country and full-colour transfer.
Report original-answer retention separately from switching to the donor answer.

Natural source and query donor must each score >=90%, jointly >=80%. The main
denominator is the set correct on both clean runs, fixed before patch scoring.
Answer-only controls use source/answer-donor clean-correct pairs. Dense query
donor transfer must achieve >=80% at each site; failure stops causal claims.

Primary site is block 24. A preliminary country-CAV effect requires full-country
target success >=20%, positive advantage over its random control with paired
exact McNemar p<=0.05, and >=80% original-answer retention for the answer-only
donor. Conditional J advantage additionally requires J target success >=20%
and >=10 percentage points above both random J and the norm-matched remainder,
each with p<=0.05. Block 29 and colour controls are explanatory. These are
single-checkpoint pilot gates, not familywise confirmation or general causal
certificates. Report the complete counts even when gates fail.

## Task heads

Task: return one of four colour labels for the same lookup problem. Construct a
shared eight-dimensional encoder from the training country+colour CAV span,
then fill remaining dimensions with training PCA residual directions. Train
centering/scales only. The encoder is supervised and shared by all compact heads.

At each site fit:

- Linear head on the full pattern (additional strong baseline).
- Linear head on the common eight coordinates.
- Ordinary degree-two polynomial features, including constant, with a linear
  readout (45 monomials).
- Class-conditional, gradient-normalized low-residual polynomial features:
  generalized eigenvectors of centered class evaluation covariance versus class
  gradient covariance. Select four of the lowest eight per class using training
  other-class/own-class RMS contrast, then fit a readout on 16 absolute values.
  Coefficients remain frozen; this is not VI-Net's coefficient fine-tuning loop.
- MLP 8→64→4 with GELU. AdamW, lr0.003, 300 full-batch updates, weight decay
  {0,0.001,0.01}; shared initialization seed 263+layer per decay. Select among
  every-25-update checkpoints using development accuracy then cross-entropy.

Linear readouts use the existing ridge-family penalties 0.0001 through 100,
selected by development accuracy then MSE. Serialize FP32 heads and verify
round-trip predictions. Count every numerical tensor, including encoder and
polynomial coefficients. Compact heads share the dominant encoder cost; their
head-specific sizes/operations are not identical and must be reported. The
full-state linear baseline can be cheaper and is not disadvantaged or excluded.

Primary head contrast: block-29 polynomial-vanishing-feature accuracy minus
MLP accuracy, with paired bootstrap interval. Report all sites and heads, class
confusions and clean teacher accuracy. An accurate head does not establish
causal CAV validity, a vanishing ideal or general language-model capability.

## Optional measured compression

Before evaluation, choose one benchmark candidate by greatest development
accuracy, then earlier block, fewer numerical tensor bytes, then fixed family
order. Benchmark it only if fresh evaluation accuracy >=95% and it makes at
most one extra error versus the full model. This observed pilot gate is not a
statistical noninferiority proof.

Under a new GPU reservation, measure the full teacher completing one answer
versus an actually truncated backbone with the selected head returning one
label. Remove later blocks, final norm and vocabulary head; account for tied
embedding weights. Use the first 32 fixed evaluation prompts, batch one, four
warmups per arm, five repeats, synchronized CUDA and wall time including
tokenization/readout. Verify shortened-model predictions equal frozen-head
predictions. Record removed parameters, numerical/exported-head bytes, memory
and task latency. Report tasks/s or latency, not generic tok/s. Sequential timing
and reused benchmark inputs remain explicit limitations.

## Execution and integrity

Root locked Pixi throughout. Pure fitting of these small heads/CAVs is an
explicit CPU cell with CUDA hidden; model capture, VJPs, interventions and any
benchmark require a live 3060 reservation before CUDA queries/load. Four
shakedown worlds test real forward/backward, exact terminal logit copying,
save/reload and representative memory; they are not quality evidence. Acquire
10,240 MiB and release between GPU stages. No services are restarted.

Freeze source/protocol/dependency/model/input hashes and cases before GPU work.
Freeze all fitted CAVs, heads and the benchmark candidate before evaluation
capture. Keep strict JSON artifacts, stage manifests, failures, immutable
completed outputs, a final aggregate/report and conservative registry readback.
No evaluation retuning, silent failed-gate retries or partial-grid accuracy.
