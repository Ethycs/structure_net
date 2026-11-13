# J-derived CAV causal and sufficiency pilot

Status: prepared before model outcomes; exploratory, not a paper replication.  
Hypothesis: `JSPACE-CAV-PILOT-01`. Date: 2026-11-12 (America/Los_Angeles).  
Question: do token directions from the Jacobian lens redirect a known computation,
and can a compact span of those directions carry that computation without the
omitted residual coordinates?

## Design and boundaries

The reference is [Verbalizable Representations Form a Global Workspace in Language
Models](https://www.transformer-circuits.pub/2026/workspace/index.html), with the
[official implementation](https://github.com/anthropics/jacobian-lens) pinned to
`581d398613e5602a5af361e1c34d3a92ea82ba8e`. We compute selected rows of
`W_U @ mean(J)` by vector-Jacobian products. This preserves the reference
estimator's future-target sum, valid-source-position average, and equal prompt
average. CPU checks compare it against the full estimator and upstream code.

This experiment uses a frozen, closed vocabulary of intermediate concept names
from the published task inventory. It does not discover arbitrary concepts or
replicate the paper's nonnegative sparse probe decomposition. Its small models,
single intervention layer, finite calibration corpus, and generated-answer
endpoint also differ from the paper. A negative result does not invalidate it.

## Data, model selection, and stops

The official `data/experiments/probe-swap.json` contains 90 cases. With the cached
Qwen tokenizer, 87 cases and 71 distinct unordered concept pairs remain after
single-token-concept and direct lexical-contamination exclusions. Retain an
exclusion ledger. Select one case per pair, deterministically with seed 7:
8 donor cases, 12 development cases, and 48 evaluation cases. Pairs do not cross
splits; individual concepts may. No evaluation outcome selects a setting.

Use 16 independently selected WikiText2 training paragraphs, each at least 80
words, to estimate the lens and PCA controls. Order eligible rows by SHA256 of
`7:{row_id}` and take the first 16. Preserve source and paragraph hashes and row
IDs. Exact normalized task-prompt matches are excluded. The lens sees at most
64 tokens per paragraph, excluding the first 16 and the final position from
its source/target averages. The separate donor split does not fit the lens.

Try cached base Qwen2.5-0.5B first; use base Qwen2.5-1.5B only if its predecessor
answers fewer than 8/12 development cases correctly. If neither reaches 8/12,
stop with `parent_competence_insufficient`; do not reinterpret failed parent
knowledge as failed J-space. All model parameters remain frozen. Use FP16 on
the reserved GPU and FP32 projection and accumulation arithmetic.

The answer endpoint is the first complete lexical word or number in a greedy
completion of the original prompt, with an eight-token cap. Preserve raw
completions and tokens. An incomplete cap-bound word is incorrect. Matching
normalizes case, punctuation, and zero-through-twenty number words, without
substring matching or a judge model. Report first-token label matching as a
separate diagnostic. This endpoint can miss legitimate synonyms; inspect errors
without changing the scoring after seeing outcomes.

## Experiment 1: targeted causal redirection

Fit lenses at block outputs `floor(depth*f)-1` for `f=1/3,1/2,2/3`, and at each
successor block. Indices are zero-based, before the final decoder norm. Normalize
each selected token row to unit norm. At each candidate layer, swap the two
concept coefficients using the pair's pseudoinverse. Apply once at every prompt
position, strength 1; cached continuation steps are unmodified.

Controls use the original unembedding directions and a fixed-seed random
perturbation. Scale each control's per-position change to the J swap's norm;
thus these are norm-matched controls, not unscaled coefficient swaps. The random
seed is a fixed function of case and layer. Choose the layer maximizing
parent-correct development successes for J minus the stronger control; ties
choose the earlier layer. Lock this choice before evaluation.

Report raw and parent-correct-conditioned counterfactual answer success,
Wilson 95% intervals, and paired differences against both controls. A promising
causal result requires at least 50% conditioned J success and positive lower
paired-bootstrap bounds against both controls on parent-correct evaluation
cases. Fewer than 24/48 parent-correct evaluation cases makes the verdict
underpowered. This is targeted redirectability, not proof that the direction
encodes an intermediate independently of answer content.

## Experiment 2: retained coordinates and omitted information

At the selected layer, factor the normalized candidate bank using SVD. Retain
ranks 8, 16, 32, and 64; also test its complete numerical row span. Save the
original semantic bank, orthonormal basis, coefficient matrix, numerical rank,
and truncation residual. Truncated bases preserve their own coordinates, not
every original concept score. The full-bank reference preserves all selected
scores up to floating-point arithmetic.

Use matched-rank PCA and seeded random bases. Fit PCA and the affine origin on
valid token activations of the independent calibration paragraphs, never on
evaluation cases. Choose each case's donor from the eight donor cases by a fixed
hash of its ID. Intervene only at the final prompt position:

1. Replace the orthogonal complement with the donor's complement, preserving the
   source's retained coordinates. Strength 1 is primary; strength 0.25 is a held-out
   sensitivity diagnostic.
2. Replace the complement with the fixed calibration mean's complement. This is
   an affine retained-only reconstruction, a stronger test than small noise.

Measure actual score preservation after FP16 casting, next-block concept-score
drift using that successor's own J lens, baseline-answer retention, and original
task accuracy. Closure is tested downstream, not inferred from the projection
identity. All non-final token states remain unchanged, so this is a local,
bounded intervention test, not whole-attention closure.

A rank passes development if both primary interventions retain at least 95% of
baseline answers, lose at most 5 percentage points of task accuracy, have at
most 5% 95th-percentile relative next-block concept-score drift, and at most
0.5% maximum relative retained-coordinate error. Select the smallest passing J
rank. If none passes, evaluate rank 64 as a prespecified descriptive fallback;
always include the complete-bank reference. Evaluate all three basis families
at those same ranks. Apply the same joint thresholds on evaluation; the smaller
rank must also beat both controls in answer retention, with positive lower paired
bootstrap bounds for both primary interventions, to be called J-specific.
Even a pass supports only these perturbations and this task family.

## Execution and artifacts

Use repository-root Pixi. CPU checks do not initialize CUDA. Every GPU process
must use the deployed reservation controller, wait for `active` and `ready`,
validate ownership/PID/UUIDs before loading a model, and release its lease after
all descendants exit. No unreserved fallback and no service reactivation.

The launcher writes a lifecycle receipt. A new dated directory under
`data/experiments/jspace_cav_pilot/` stores frozen configuration, source copies
and hashes, model revision/weight hashes, data and split manifests, calibration
text, per-case generations, lens tensors, intervention diagnostics, memory and
runtime metadata, and strict-JSON results. Existing roots are never reused.
Run a small real forward/backward/save/reload check under the same reservation
before the scientific stages. This check is not quality evidence.

No training, algebra compilation, attention replacement, or speed claim is part
of this pilot. A successful result justifies testing closure with the context
and attention caches also compressed; intact context can supply omitted
information in this pilot. After that stronger test, compare an algebraic
reduced transition against ordinary low-degree fits. Any eventual speed claim
requires an actual replacement and an isolated benchmark with unchanged task
quality.
