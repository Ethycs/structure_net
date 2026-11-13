# Bounded neural relation-propagation experiment

Date: 2026-11-13 (America/Los_Angeles).  
Study: `JSPACE-RELATION-PROPAGATION-01`, exploratory neural version 1.  
Status at writing: prospective; no neural outcomes for this experiment inspected.

This implements the [relation-propagation plan](2026-11-13_jspace-relation-propagation-plan.md).
The question is whether approximate polynomial relations among named
Jacobian-derived observables have held-out consequences across actual layers of
an intact transformer. Compression is a possible later application, with its own
test. Neither reconstruction quality nor a task-accuracy threshold gates this
experiment.

## Frozen model and observables

Use the locally cached `Qwen/Qwen2.5-3B-Instruct`, revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, frozen in BF16. It has 36 decoder
blocks and residual width 2048. Record hashes of all checkpoint/tokenizer files,
the producer sources, root Pixi lock, protocol, and data before capture. Run
through the repository-root `ncm` Pixi environment; no environment in `data/`.

Observe the final prompt position at block outputs, before the final model
normalization, at zero-based blocks
`[5, 6, 11, 12, 17, 18, 23, 24, 29, 30, 34, 35]`.
The six primary edges are `5→6`, `11→12`, `17→18`, `23→24`, `29→30`, and
`34→35`. This samples the model; it does not measure every block transition.

The fixed concept order is `China, France, Germany, Russia`. Fit the corresponding
space-prefixed single-token rows of `W_U mean(J)` using the existing selected-row
Jacobian-lens implementation, derived from upstream commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`. Fit all observed blocks except 35.
Block 35 uses the unembedding row directly, because transport from the final
block output to itself is the identity. No final-normalization derivative is
folded into these rows. Normalize each row to unit length. These four linear
observables are a restricted dictionary readout, not a full sparse nonnegative
J-space decomposition or demonstrated CAV semantics.

Calibration uses the pre-existing 16 WikiText2 training paragraphs, seed 7,
whose selection file SHA-256 is
`6b5546932ef0062d87e046158b2bf0c54514d533124ba53ec2f0261b847e4dfc`.
Use at most 64 tokens, omit the first 16 and final position in the estimator,
and fit four rows per backward pass. The calibration text is independent of the
task examples. Preserve row/source hashes and upstream provenance.

Capture hidden states as FP32 CPU arrays. Project them onto FP32 unit rows using
FP64 arithmetic for CPU analysis. All scores remain continuous. Do not impose
Boolean, one-hot, country-membership, or answer-label equations on them.

## Tasks, splits, and controls

Generate self-contained two-hop lookup tasks: four named people map to the four
countries, each country maps to one of four colours, and the question asks for
one person's colour. Facts needed to answer appear in the prompt. Use seed 7
with 512 training, 128 development, 128 evaluation, and 16 separate shakedown
cases. Each split has separate names and prompt templates. Audit all named
mapping tables, question/answer labels, and counterfactual labels.

Each country–answer cell is balanced within each default prompt template.
Named worlds are unique across splits. Anonymous permutation structures recur;
this is generalization to new names and prompt families, not a claim about
unseen permutation structures. Preserve these overlap counts in the manifest.
Previously inspected pilot examples are not reused.

Use the model's chat template with one user message. Greedily decode up to eight
tokens, using the existing complete-first-word parser. Wrong, missing,
explanatory, or truncated answers stay in the denominator. Report exact colour
accuracy separately; the targets are unambiguous colours, not open-ended
summaries requiring semantic judgment.

Fit four centered PCA axes per layer from training hidden states only, with an
exact FP64 SVD and fixed sign convention. Use the same seeded random orthonormal
four-row bank at every observed layer. Apply the same relation-discovery rules
to J, PCA, and random scores. The PCA/random row numbers correspond to the fixed
country indices for interventions, but are not claimed to encode countries.

## Relation fitting and symbolic propagation

Fit coordinate means/scales and monomial normalization using training data only.
Search degrees one and two: 15 monomials within a four-variable layer and 45 in
an eight-variable source/destination pair. The 512 training rows exceed twice
these counts. Search at most eight smallest SVD directions per degree. This is
a bounded search, not a proof that no other relations exist.

Exclude low-variation coordinates/terms, constant polynomials, zero-scaled
solutions, and degree-bounded polynomial multiples of already accepted linear
relations. The residual normalization prevents arbitrary coefficient scaling
from making a polynomial look accurate. Accept at most three relations within
each layer and four genuinely mixed relations per edge. Selection uses only
development data; all candidate failures remain recorded.

For development acceptance and untouched evaluation, require normalized RMS
at most 0.10 and observed RMS at most one third of the median of 32 shuffled
controls. Shuffle each coordinate independently. Mixed relations must also
beat 32 shuffles of destination rows relative to source rows, which preserve
the marginal within-layer constraints. Numerical-zero shuffled controls are
noninformative. The degree/layer search is exploratory; no familywise
significance or independent-model replication claim follows from these gates.

For each bank and edge, form the ideal generated by up to three accepted source
relations and four accepted mixed relations. Rationalize coefficients with
denominator at most 1000, requiring development RMS change at most 0.01 and the
existing relative/absolute residual guard. Compute a lexicographic Gröbner
basis over the rationals with source variables first, then retain destination-only
polynomials. Use at most seven generators and a killable five-second subprocess
per edge (18 bounded attempts total). Record rejected rationalization, unit
ideals, empty elimination, degree/size explosion, and timeout as distinct results.

Normalize eliminated consequences on development data. Validate their
development residuals and coordinate-shuffle contrast before evaluation. A
validated propagated consequence must pass both the development and evaluation
0.10 RMS / threefold-contrast gates. Freeze every relation model, regression,
symbolic result, and development gate to disk before loading evaluation scores
for scoring. Report the actual generators, coordinate scales, basis, and
eliminated polynomials. Exact elimination describes the rational surrogate;
it is never an exact certificate about the transformer. Algebraic elimination
also need not characterize all realizable real states.

Train-only linear and quadratic source-to-destination regressions are separate
prediction baselines, evaluated on development and evaluation. Good regression
alone is not a Gröbner-propagation result.

## Causal follow-up inside the same capture

Select 64 evaluation cases before capture: within each of the 16
country–answer cells, take the first four by case ID. Do not select by the
model's answer. At blocks 11 and 23, swap the source/target pair's pseudoinverse
coefficients at the final prompt position only, with strength one. Retain the
rest of the residual, all other prompt positions, subsequent layers, and cached
continuation unchanged. Cases already specify the target country and its
counterfactual colour.

Run each selected case × site × arm (`jacobian`, `pca`, `random`), yielding 384
rows. Match PCA/random perturbation norms to the J perturbation on the same
unpatched hidden state. Record requested FP32 and realized post-BF16-cast norms,
all layer observations, original accuracy, and counterfactual accuracy. Report
originally correct subsets as secondary, with their denominators.

Evaluate frozen natural-data relations on the interventions separately from
natural held-out examples, using the corresponding measured bank. For edges
at or after the intervention site, report all cases and a separately labelled
source-compatible subset: every accepted source relation must fall within its
frozen development 95th-percentile absolute residual. If no source relation is
accepted, compatibility is undefined, not automatically passed. Preserve every
excluded case/count. An intervention outside a source relation's valid domain
does not independently falsify propagation conditional on that relation.

## Execution, artifacts, and interpretation

The intended root is
`data/experiments/jspace_relation_propagation/20261113_neural_v1/`.
All writes are exclusive or new-file writes; changed producers use a new root.
Use one GPU process and a live reservation, initially 3060=10240 MiB for 60
minutes. Verify the lease before querying CUDA or loading the model, recheck
during work, and release it through the controller wrapper when the process
exits. CPU preparation/analysis runs with CUDA hidden.

Pass focused CPU tests and actual tiny-Qwen capture tests first. Save the 16-case
shakedown under its own labelled artifact subroot; its outcomes are systems
checks only. Lens fitting and this shakedown establish representative memory
use before the full capture. With only one process, intended concurrency is one.
Nonfinite activations, missing observations, provenance mismatch, invalid lease,
out-of-memory, or malformed artifacts stop the run as a systems/validity failure.
Low task accuracy, no accepted relations, or no eliminated consequences are
scientific observations, not reasons to tune and rerun this frozen evaluation.

The result separates: held-out within-layer relations; paired transition
relations; validated eliminated consequences; J/PCA/random comparisons; causal
generalization; and task accuracy. Classify no accepted relations, relations
without validated propagation, or validated approximate propagation explicitly.
A J advantage is descriptive unless it is later tested with an independent
replication. One checkpoint, four observables, six edges, one task family, and
one data seed cannot establish a general architecture theorem. No storage,
accuracy-preserving compression, or speedup claim is part of this experiment.

Reproduction, after `prepare` has frozen the producer:

```bash
CUDA_VISIBLE_DEVICES='' pixi run --locked -e ncm python -m experiments.structure_net.jspace_relation_study prepare --output data/experiments/jspace_relation_propagation/20261113_neural_v1
/usr/bin/python3 /usr/local/bin/inference-ctl lease run --allocation 3060=10240 --minutes 60 --reason 'JSPACE-RELATION-PROPAGATION-01 intact-model capture' -- pixi run --locked -e ncm python -m experiments.structure_net.jspace_relation_study capture --output data/experiments/jspace_relation_propagation/20261113_neural_v1
CUDA_VISIBLE_DEVICES='' pixi run --locked -e ncm python -m experiments.structure_net.jspace_relation_study analyze --output data/experiments/jspace_relation_propagation/20261113_neural_v1
```
