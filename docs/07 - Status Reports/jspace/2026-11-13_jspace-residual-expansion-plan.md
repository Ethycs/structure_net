# Recursive residual expansion with a J-derived dictionary

Date: 2026-11-13. ID: `JSPACE-RESIDUAL-EXPANSION-01`.

This is a CPU-only, post-hoc exploratory diagnostic using existing neural
captures. The original test examples and four-direction energy results have
already been inspected. This is not a fresh confirmatory test, a model-quality
evaluation, or a Gröbner-basis experiment.

Question: after subtracting a selected J-derived component, how much additional
activation energy can further dictionary vectors recover, and what remains?
The algebraic control predicts that repeating the identical orthogonal
projector adds zero. Expanding its span can decrease residual norm. Sparse
nonnegative decomposition has a different feasible set and must be reported
separately.

Use the six common blocks `[11,12,17,18,23,24]`, saved 512 training and 128
evaluation activations, and the existing 94-token J dictionary from
`jspace_cav_pilot/20261113T0728Z_pilot02`. Both captures/dictionaries use frozen
Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, BF16,
the same 16 calibration paragraphs, and the same estimator. Record the producer
difference: the older dictionary used Torch 2.5.1 and the newer four-row fit used
Torch 2.4.1. Verify identity/calibration fields and report their row discrepancy;
replace the four corresponding old rows with the newer four-row bank so every
expansion starts from the exact bank whose residual prompted this question.

Freeze source and input hashes before expansion results are inspected. The
new artifact root is
`data/experiments/jspace_residual_expansion/20261113_v1/`. Use locked root Pixi,
CUDA hidden, FP64 analysis. No GPU reservation is needed because there is no
model loading, training, derivative estimation, or CUDA access.

## Orthogonal span expansion

Start with the four named country rows and orthogonalize them. At each step,
remove the current span from every remaining dictionary row. Reject innovation
norms below 1e-6 relative to its normalized original row. Among the remainder,
select the row whose unit innovation captures most summed squared projection
of centered **training** residuals. Reorthogonalize, add it, and continue.
Freeze all choices before scoring evaluation examples.

Measure at total direction budgets `[4,8,16,25,32,64,94]`; report achieved rank
if exhaustion occurs earlier. Compare with two expansions anchored to the same
initial four rows: top principal directions of the centered training residual,
and a seeded random orthogonal complement (seed 7, separated by layer).
These are signed linear spans, not the paper's sparse nonnegative J-space.

For each layer/budget report pooled captured/residual squared-norm fractions
both on raw activations and around the fixed training mean. Do not refit the
mean on evaluation. Check nonincreasing residual norms and repeated-projector
idempotence. Save each layer's J expansion and its actual final evaluation
residual so coefficients and intermediate residuals can be reconstructed.

## Sparse nonnegative pursuit

Separately decompose each raw evaluation activation using the same normalized
94-vector dictionary. Begin by NNLS-fitting the four country rows. Select an
unused atom with the largest positive residual correlation, then jointly refit
nonnegative coefficients over all selected atoms. Continue through budgets
4, 8, 16, 25, and 32, or stop when no positive unused correlation remains.
At the final 94-atom endpoint solve NNLS over the full dictionary, which gives
the best nonnegative fit available in this bounded dictionary.

Use a thin SVD to reduce the NNLS least-squares dimension while retaining the
same objective up to its orthogonal constant. SciPy NNLS has a fixed iteration
limit of 30 times the selected atom count. Record solver failure, rather than
silently accepting a nonconverged solution. This is nonnegative orthogonal
matching pursuit with NNLS refitting, not a claim to reproduce the paper's
gradient-pursuit implementation. Per-example support selection is encoding,
not training a shared basis on evaluation examples.

Save coefficients, selected token IDs/strings, the full-NNLS residual, and
support counts. At an NNLS optimum the residual is orthogonal to the fitted
component, but it need not be orthogonal to every unused dictionary direction;
report the actual squared residual, not squared nonorthogonal coefficients.
Do not compare sparsity budgets to signed-span rank as identical constraints.

The final report should distinguish a successful geometric decomposition from
semantic/causal utility, neural quality preservation, and computational savings.
Keeping an explicit remainder makes the vector accounting complete within
floating-point precision; it does not by itself compress the model. Further
recursion can exhaust a fixed dictionary, and accumulating enough signed
directions can ultimately become a change of coordinates rather than useful
compression.
