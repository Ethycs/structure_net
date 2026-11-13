# Recursive extraction of J-derived residual components

Date: 2026-11-13. Study: `JSPACE-RESIDUAL-EXPANSION-01`.
Status: completed, CPU-only, exploratory post-hoc diagnostic; unconfirmed.

Expanding the selected J-derived span recovered additional activation energy,
and the remaining vectors were saved. Reapplying the unchanged four-vector
projector recovered only numerical roundoff. At 94 signed directions, the
expanded span captured 4.34–9.05% of raw activation energy across six layers,
leaving 90.95–95.66% in its residual. The nonnegative dictionary fit saturated
earlier, at 1–7 positive coefficients per example, leaving more than 99% of raw
energy outside that cone. This tests geometric decomposition, not task quality
or a neural Gröbner basis.

The [plan](../../07%20-%20Status%20Reports/jspace/2026-11-13_jspace-residual-expansion-plan.md)
was frozen before these expansion results were inspected. The earlier test
examples and four-direction results were already known; this is explicitly
exploratory reuse, not a new held-out confirmation.

- Artifact root: `data/experiments/jspace_residual_expansion/20261113_v1/`.
- [Direct result](../../../data/experiments/jspace_residual_expansion/20261113_v1/result.json).
- Result SHA-256: `e8d99f6b195549742115e12eda2800bed26bd7af67341dc2aaeb40da4f872b5a`.
- Scientific fingerprint: `d6e30764e08d073087ec4d43d2cc47090b5e1df6386bb09bbb3fa5714b0a4c30`.

## What “recursive” means here

For an orthonormal row basis `Q_k`, define `r_k = h - Q_k.T Q_k h`.
Project candidate dictionary vectors into its orthogonal complement, select a
new direction from those innovations, and extend the basis. All existing
coefficients remain available for reconstruction. The selection score is
captured squared energy on centered training activations. No evaluation example
selects the shared basis.

This uses a fixed previously estimated Jacobian dictionary. Residualizing its
rows implements `D_remaining = D (I - P_k)`; it does not estimate a new Jacobian
at a residual-only artificial hidden state. Across layers the candidate pool
contains 94 token-indexed vectors, including the initial four countries.

The paper defines J-space using sparse **nonnegative** combinations of dictionary
vectors, and uses gradient pursuit to estimate the component and its remainder.
A growing signed span is a different object. We therefore tested a second
representation: greedily add positively correlated atoms and jointly refit
nonnegative coefficients with NNLS. This is a matching-pursuit variant, not a
replication of their gradient-pursuit implementation. Increasing the cumulative
support relaxes the sparsity budget; it does not establish additional independent
workspaces. [Paper methods](https://www.transformer-circuits.pub/2026/workspace/index.html).

## Results

All methods in the signed comparison begin with the same four J-derived
directions. The table gives median captured **raw activation energy** across
blocks 11, 12, 17, 18, 23, and 24. Each layer's value pools squared norms over
the same 128 evaluation prompts. These percentages are not task accuracy.

| Total directions | J-derived expansion | PCA expansion | Random expansion |
| --- | ---: | ---: | ---: |
| 4 | 0.14% | 0.14% | 0.14% |
| 8 | 0.97% | 4.91% | 0.39% |
| 16 | 1.48% | 29.45% | 0.86% |
| 25 | 2.26% | 47.73% | 1.22% |
| 32 | 2.67% | 51.64% | 1.72% |
| 64 | 4.33% | 62.24% | 3.48% |
| 94 | 5.30% | 68.34% | 4.61% |

PCA was fitted to centered training residuals. Its 94-direction raw-energy
coverage ranged from 49.75% to 78.11%; random expansion ranged from 4.03% to
5.88%. PCA is a much stronger energy-reconstruction control in this experiment.
Energy alone does not determine which directions carry useful concepts.

Around the fixed training mean, 94 J-derived directions captured 4.67–7.28%
of evaluation energy (median 6.68%). The corresponding medians were 51.69%
for PCA and 4.71% for random. Means were never fitted to evaluation data.

| Block | J span: captured energy at rank 94 | J span: remaining energy | Full nonnegative dictionary: captured energy |
| --- | ---: | ---: | ---: |
| 11 | 9.052% | 90.948% | 0.936% |
| 12 | 7.252% | 92.748% | 0.363% |
| 17 | 5.244% | 94.756% | 0.079% |
| 18 | 5.174% | 94.826% | 0.010% |
| 23 | 4.340% | 95.660% | 0.116% |
| 24 | 5.358% | 94.642% | 0.244% |

The nonnegative fit reached its full-dictionary energy result, to reported
precision, by the checkpoint with 16 selected atoms. The solutions used only
1–7 positive coefficients. At the
full 94-atom endpoint, the maximum positive residual/dictionary correlation was
`2.0e-14`, consistent with the NNLS optimum: the same positive dictionary cannot
keep extracting additional components. Negative correlations can remain, which
is why its remainder differs from the signed-span remainder.

The unchanged four-vector projector's second pass captured at most
`6.2e-33` of the original energy. Every expanded signed span and every nested
NNLS support had nonincreasing residual energy. Adding the final signed
reconstruction and saved residual recovered the original activation within
`1.8e-15` maximum absolute FP64 error. This is numerical vector accounting,
not a bitwise lossless model-compression claim.

## Saved residuals and provenance

Each `layer_<block>_residuals.pt` contains:

- `orthogonal_94_residual`: actual `[128,2048]` FP64 remainder after the expanded
  signed J span;
- `orthogonal_94_coefficients`: coefficients for the saved shared basis;
- `nonnegative_94_residual`: actual remainder after full-dictionary NNLS;
- `nonnegative_coefficients`: coefficients at each recorded sparse budget;
- the normalized dictionary and its token IDs/strings.

The corresponding `layer_<block>_fit.pt` stores the training mean, nested J/PCA/
random bases, and selected dictionary indices. Intermediate signed residuals can
be reconstructed using basis prefixes and the immutable parent activations.
For example, the [block-23 residual file](../../../data/experiments/jspace_residual_expansion/20261113_v1/layer_23_residuals.pt)
contains both signed and nonnegative remainders.

The 94-vector dictionary came from the earlier frozen Qwen2.5-3B-Instruct pilot.
Both fits used the same checkpoint, calibration paragraphs, estimator and BF16,
but different Torch versions: 2.5.1 versus 2.4.1. Their largest normalized
anchor-row discrepancy was 0.0233 in L2. The four anchors were replaced by the
newer exact rows before every comparison. This mixed provenance is recorded;
the 94-word inventory is also a narrow selection, not the full vocabulary.

The diagnostic took 8.14 seconds on CPU with CUDA hidden, using root locked
Pixi. No model inference or GPU allocation occurred. Six focused tests passed,
including duplicate-direction exhaustion, projector idempotence, orthogonality,
nonnegative sign constraints, agreement with direct NNLS, and reconstruction.

## Consequence for the research question

Recursive extraction is viable as a way to accumulate components and preserve
an explicit remainder. It cannot create new information from repeated use of
an exhausted fixed dictionary. Our next candidate pool should be broader and
task-aligned; newly discovered directions should be tested for concept
discrimination and causal effect, alongside the energy controls.

The full vocabulary may span a much larger linear space, but unrestricted
reconstruction with enough coordinates can become an ordinary change of basis.
The useful question is how many **task-relevant** components are needed before
the residual can be simplified while preserving behavior. That requires a
separate model-intervention/quality test; this CPU diagnostic does not answer it.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 pixi run --locked python -m experiments.structure_net.jspace_residual_expansion --output data/experiments/jspace_residual_expansion/<new-root>
```
