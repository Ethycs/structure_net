# Shared task basis and Gröbner tensor compilation with a retained residual

Date: 2026-11-13. Study: `JSPACE-BENCH-TENSOR-01`.
Prospective exploratory pilot; one checkpoint, not independent training replicates.
Artifact root: `data/experiments/jspace_bench_tensor/20261113_v1`.

## Question and architecture

Run Qwen2.5-3B-Instruct on four public task families, discover one common task-Jacobian
chart per selected MLP block, derive polynomial graph relations in those coordinates,
and compile the covered channels into a polynomial coefficient tensor. Keep uncovered
MLP channels as the residual. Measure combined task accuracy, actual removed weights,
generation speed, and reuse of basis elements across tasks.

This is a partial computation rewrite, not an independent replacement of network
functions. The model remains frozen except for deterministic activation regression
and compiled replacement modules. The pilot does not establish semantic CAV causality.

## Benchmark and splits

Use official Hugging Face parquet revisions pinned before sampling: ARC-Challenge,
BoolQ, HellaSwag and WinoGrande-debiased. Per family, sample 32 calibration and 16
development questions from training, and 64 evaluation questions from labeled test
(ARC) or validation (the other families). Exclude duplicated question/options and
prompts longer than 384 tokens, without truncation. Preserve exclusions and source
row IDs. Deterministic source shuffle seed 861 plus fixed family/split offsets.

Report 256-question multiple-choice accuracy and each 64-question family separately.
Score the relevant A–D logits at the first assistant token; no answer LLM judge and
no free-generation quality claim. Calibration/development/evaluation identities must
be disjoint. Evaluation rows never select basis, coefficients, ranks or channel count.

## Common coordinates and relations

Selected zero-based MLP blocks: 8, 17, 26. Capture 16 uniformly spaced token positions
per calibration/development prompt. For the first 16 calibration questions per family,
differentiate correct-option logit minus mean incorrect-option logit with respect to
post-SwiGLU activations. Collect the final-token and sequence-mean Jacobian rows.
The concatenated bank covers all four families, not a separate model per family.

Normalize by calibration activation standard deviations (floor 1e-5), normalize
Jacobian rows, and take the leading 16 right-singular directions. Greedy rank-revealing
column selection chooses 16 physical anchor channels from that shared chart. SVD is
used to represent the J-space; the implemented weight rewrite is polynomial elimination.

For each other channel, fit degree-one and degree-two polynomials in normalized
anchor activations using ridge 0.01 (intercept unpenalized). Round coefficients to
the 2^-20 rational grid. A generator is `y_i-p_i(z)`. With eliminated variables
ordered before anchors, the distinct monic leading variables are pairwise coprime;
the product criterion certifies the complete triangular Gröbner basis. A representative
CAS normal-form audit accompanies the structural certificate. The quotient is
infinite-dimensional; the compiled degree-two feature library has 153 terms, including
constant/linear terms and 136 packed symmetric quadratic terms.

Substitute the graph relations into the original down projection. Fold constant and
linear terms into the retained down projection and bias; retain a quadratic tensor
for the remaining 136 features. Delete covered gate/up rows and down columns. The
uncovered channels continue their original nonlinear computation. Actual neural
deviations from the fitted graph are measured approximation errors; they are not
assumed zero or secretly recomputed by evaluating deleted neurons.

Rank removable channels by worst-family development reconstruction NRMSE times
calibration task-Jacobian RMS in normalized coordinates. Anchors are never removed.
Predeclare 128, 512 and 1024 removed channels per selected block.

## Cross-task redundancy

For every common generator, report the families where development reconstruction
NRMSE is <=0.5. Exclude anchor identities. Report per-family counts, their sum, the
union, repeated occurrences (sum minus union), repeat fraction, pairwise overlap,
and counts supported by at least two and by all four families. A shared occurrence
uses exactly the same common-chart polynomial, not a coefficient-similarity heuristic.

Also fit task-specific coefficients in that fixed chart. Report their eligible counts
and exact rational coefficient matches to the common generator. These are empirical
reuse counts, not the algebraic dimension, number of independent mechanisms, or model
accuracy. Approximate sharing and exact equality remain distinct.

## Controls and selection

CPU correctness tests establish exact rational normal-form decomposition into covered,
overflow and relation-defect terms; Boolean product-tensor factorization; and numeric
value/gradient equivalence between explicit neuron substitution and compiled modules.

Development evaluates the original plus each quadratic channel budget. A budget
qualifies if label agreement is >=63/64, average choice KL is <=0.02, total correct
answers decrease by at most one, and no family loses more than one correct answer.
Choose the largest qualifying count. If none qualifies, evaluate 128 descriptively
and retain the failed development gate.

Frozen held-out arms: original, selected compiled quadratic, linear relation map on
the same channels, zeroed channels, and the same quadratic surrogate without folding
its coefficients into the down projection. The latter retains an explicit predicted
channel vector and its original down-projection columns; it tests the compilation
benefit separately from the fitted polynomial model. Finite-precision differences
between compiled and uncompiled forms are measured, not assumed absent.

## Measurements and gates

The combined-quality gate requires development qualification, <=1 fewer correct
answer out of 256, >=254/256 original-label agreement, and <=1 lost correct answer in
any family. Report raw correct counts, accuracy, prediction agreement and choice KL.
This bounded pilot is not a statistical noninferiority claim or a 95% general-accuracy
claim. All studies retain `confirmed=false`.

Measure resident model tensor bytes including buffers and removed parameter counts.
Time batch-one greedy generation of exactly 32 new tokens on two development prompts
per family, after one warmup. Include prefill in wall time. Measure original before
and after intervention timings, and report both. Point-estimate efficiency requires
positive model-byte reduction and >=1.02x compressed throughput against the mean of
the two original rates. Report compiled versus uncompiled timing separately. No
free-generation accuracy is inferred from this timing test.

Classify quality plus efficiency, quality without speed gate, or combined-quality
failure. A failed gate is never repaired by relabeling another arm as primary.

## Lifecycle and provenance

Prepare/tasks and final analysis run CPU-only in root locked Pixi. Symbolic operations
are bounded CPU work; model capture, J-space decomposition, regression and evaluation
use an RTX 3060 reservation with 10,240 MiB. Verify a live reservation before CUDA.
Use root `pixi run --locked -e ncm` for GPU stages; do not run from a data directory.

Freeze source, protocol, tests, lockfiles, checkpoint and dataset identities before
GPU work. Run CPU tests, then a four-question longest-prompt CUDA shakedown with real
backward, replacement forward/cache generation and state reload. Its outputs are
systems checks only. Run capture, fit, evaluate and analyze sequentially. No concurrent
GPU workers. Nonfinite values, fingerprint mismatches and allocation violations stop
the stage; no automatic CPU fallback. OOM or code fixes require a recorded new source
revision/root before retry. Preserve outputs, register the completed scoped result,
and verify it from a fresh process.
